"""End-to-end integration: the Identity layer composes.

The `docs/cookbook/end-to-end.md` recipe chains design → cross-engine fold →
score → cross-engine ensemble → reproducibility. The real engines are
GPU-only, so this drives the *composition* with tiny fake engines to prove
the pieces fit together (and that the recipe isn't aspirational).
"""

from __future__ import annotations

import numpy as np

from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.design import DesignLoop
from molforge.ensembles import cross_engine_fold
from molforge.generative import DesignedSequence
from molforge.reproducibility import emit_pipeline, load_pipeline, replay
from molforge.scoring import ConfidenceScorer, rank


def _helix(n: int, *, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    i = np.arange(n)
    th = np.radians(100.0) * i
    c = np.stack([2.3 * np.cos(th), 2.3 * np.sin(th), 1.5 * i], axis=1).astype(np.float64)
    if noise:
        c = c + noise * np.random.default_rng(seed).standard_normal((n, 3))
    return c


def _protein(coords: np.ndarray, *, confidence: float | None = None) -> Protein:
    n = coords.shape[0]
    p = Protein(
        AtomArray.from_dict(
            {
                "coords": coords.astype(np.float32),
                "atom_name": np.array(["CA"] * n, dtype="U4"),
                "residue_id": np.arange(1, n + 1, dtype="int32"),
                "chain_id": np.array(["A"] * n, dtype="U4"),
                "entity_type": np.array(["protein"] * n, dtype="U8"),
            }
        ),
        name="cand",
    )
    if confidence is not None:
        p.metadata[mk.MEAN_CONFIDENCE] = confidence
    return p


class _FakeDesigner:
    name = "FakeMPNN"

    def generate(self, backbone: Protein, **kwargs: object) -> list[DesignedSequence]:
        length = backbone.n_residues
        return [DesignedSequence(sequence="A" * length, score=-1.0 * (i + 1)) for i in range(4)]


class _FakeFolder:
    """A folder that returns a helix (with per-engine noise) carrying a
    real Provenance, so downstream emit/replay have something to work with."""

    def __init__(self, name: str, noise: float) -> None:
        self.name = name
        self.parallelism = "serial"
        self.noise = noise
        self._i = 0

    def predict(self, sequence: str, **kwargs: object) -> Protein:
        self._i += 1
        p = _protein(
            _helix(len(sequence), noise=self.noise, seed=self._i),
            confidence=max(0.0, 95.0 - 10.0 * self.noise),
        )
        p.metadata[mk.PROVENANCE] = Provenance.from_engine(
            self.name, operation="predict", inputs={"sequence": sequence}
        )
        return p


class TestIdentityStackComposes:
    def test_full_recipe(self, tmp_path) -> None:
        from molforge import plugins

        backbone = _protein(_helix(24))
        e1, e2, e3 = _FakeFolder("E1", 0.0), _FakeFolder("E2", 0.5), _FakeFolder("E3", 1.0)

        # 1-2. Design with a cross-engine folder, scored by self-consistency,
        #      iterated over rounds.
        loop = DesignLoop(
            designer=_FakeDesigner(),
            folder=[e1, e2],
            objective="self_consistency",
            n_designs=4,
            n_rounds=2,
            select_top=2,
        )
        table = loop.run(backbone)
        assert len(table) > 0
        best = table.best
        # Cross-engine metrics recorded on every candidate.
        assert "cross_engine_tm_mean" in best.metrics
        assert "cross_engine_rmsf_mean" in best.metrics
        # Ranked best-first.
        assert [c.score for c in table] == sorted((c.score for c in table), reverse=True)

        # 3. Score the top structures with a common, direction-aware yardstick.
        structures = [c.structure for c in table.top_n(3) if c.structure is not None]
        ranked = rank(structures, ConfidenceScorer())
        assert ranked[0][1].value >= ranked[-1][1].value

        # 4. Cross-engine ensemble on the winning sequence.
        ensemble = cross_engine_fold(best.sequence, engines=[e1, e2, e3])
        assert ensemble.disagreement().shape[0] == len(best.sequence)
        assert {"tm_mean", "rmsd_mean"} <= set(ensemble.spread())

        # 5. Reproducibility — emit a manifest, then replay it. Use the JSON
        #    form so this stays runnable without the optional ``repro`` (PyYAML)
        #    extra; the YAML serialization is covered in test_reproducibility.py.
        assert best.structure is not None
        path = tmp_path / "pipeline.json"
        emit_pipeline(best.structure, path, fmt="json")
        manifest = load_pipeline(path)
        assert len(manifest) >= 1
        assert all(s.operation == "predict" for s in manifest)

        # Replay: register a factory for each engine the manifest names.
        for step in manifest:
            plugins.register_engine(step.engine, lambda: _FakeFolder("replayed", 0.0))
        try:
            result = replay(manifest)
            assert isinstance(result, Protein)
        finally:
            plugins.clear()
