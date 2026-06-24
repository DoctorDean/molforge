"""End-to-end provenance adoption tests across all engine wrappers.

Each wrapper has detailed unit tests in its own ``test_<wrapper>.py``
file; what *this* file does is hold the wrappers to a uniform
provenance contract.

For every wrapper that produces a Protein or DockingResult, this
file asserts:

1. ``metadata[PROVENANCE]`` is set after the producing call.
2. The attached value is a :class:`Provenance`.
3. ``engine`` matches the documented name (so cross-engine code can
   filter / dispatch on it).
4. ``parameters`` contains every engine-config field we promised to
   record — the actual values may be ``None`` for unset defaults
   but the *keys* must be present, so future code can rely on
   ``prov.parameters.get("device")`` and similar.
5. ``inputs`` contains the key for the input the wrapper consumed.

The tests deliberately reach into the engines' parsing seams (e.g.
``ESMFold._pdb_to_protein``, ``Vina._parse_poses_pdbqt``) so they
don't require the heavy underlying dependency — same pattern the
per-wrapper test files already use for their post-processing tests.

When a new wrapper is added, the right place to add it is *here* and
in its own test file: the test_<wrapper>.py covers behavior, this
file checks the provenance contract.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from molforge.core import Provenance
from molforge.core import metadata_keys as mk

# ---------------------------------------------------------------------
# Shared synthetic PDB fixture (a 2-residue ALA-GLY backbone)
# ---------------------------------------------------------------------

_SYNTH_PDB = textwrap.dedent(
    """\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
    ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 80.00           C
    ATOM      3  C   ALA A   1       2.000   1.420   0.000  1.00 80.00           C
    ATOM      4  N   GLY A   2       3.314   1.420   0.000  1.00 60.00           N
    ATOM      5  CA  GLY A   2       4.000   2.840   0.000  1.00 60.00           C
    END
    """
)


def _assert_provenance(
    meta: dict[str, Any],
    *,
    engine: str,
    expected_param_keys: set[str],
    expected_input_keys: set[str],
) -> Provenance:
    """Shared assertion harness for every wrapper's provenance shape.

    Returns the Provenance object so the caller can do
    test-specific follow-up assertions (e.g. parent presence).
    """
    assert mk.PROVENANCE in meta, f"{engine}: metadata[PROVENANCE] missing"
    prov = meta[mk.PROVENANCE]
    assert isinstance(prov, Provenance), f"{engine}: provenance is not a Provenance"
    assert prov.engine == engine, f"{engine}: engine mismatch — got {prov.engine!r}"
    assert prov.molforge_version, f"{engine}: molforge_version not auto-filled"
    assert prov.timestamp, f"{engine}: timestamp not auto-filled"

    missing_params = expected_param_keys - set(prov.parameters.keys())
    assert not missing_params, (
        f"{engine}: missing parameter keys {missing_params}; got {set(prov.parameters.keys())}"
    )

    missing_inputs = expected_input_keys - set(prov.inputs.keys())
    assert not missing_inputs, (
        f"{engine}: missing input keys {missing_inputs}; got {set(prov.inputs.keys())}"
    )
    return prov


# ---------------------------------------------------------------------
# Folding wrappers
# ---------------------------------------------------------------------


class TestFoldingProvenance:
    def test_esmfold(self) -> None:
        from molforge.wrappers.folding.esmfold import ESMFold

        engine = ESMFold()
        protein = engine._pdb_to_protein(_SYNTH_PDB, sequence="AG")

        _assert_provenance(
            protein.metadata,
            engine="ESMFold",
            expected_param_keys={"model_name", "device", "chunk_size", "dtype"},
            expected_input_keys={"sequence"},
        )
        # Sequence carried verbatim.
        assert protein.metadata[mk.PROVENANCE].inputs["sequence"] == "AG"

    def test_alphafold(self) -> None:
        from molforge.wrappers.folding.alphafold import AlphaFold

        engine = AlphaFold()
        protein = engine._pdb_to_protein(_SYNTH_PDB, sequence="AG")

        _assert_provenance(
            protein.metadata,
            engine="AlphaFold",
            expected_param_keys={
                "mode",
                "model_type",
                "msa_mode",
                "num_models",
                "num_recycles",
                "device",
            },
            expected_input_keys={"sequence"},
        )

    def test_rosettafold(self) -> None:
        from molforge.wrappers.folding.rosettafold import RoseTTAFold

        engine = RoseTTAFold()
        protein = engine._parse_outputs(pdb_text=_SYNTH_PDB, confidence={}, sequence="AG")

        _assert_provenance(
            protein.metadata,
            engine="RoseTTAFold",
            expected_param_keys={
                "repo_dir",
                "python_executable",
                "max_cycle",
                "job_name",
                "extra_overrides",
            },
            expected_input_keys={"sequence"},
        )

    def test_boltz(self) -> None:
        """Boltz writes both a CIF structure and a confidence JSON;
        the seam is ``_outputs_to_protein`` which takes both."""
        from molforge.wrappers.folding.boltz import Boltz

        # Boltz needs a CIF — easiest is to write a tiny dipeptide one.
        cif = textwrap.dedent(
            """\
            data_test
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.pdbx_formal_charge
            _atom_site.auth_seq_id
            _atom_site.auth_comp_id
            _atom_site.auth_asym_id
            _atom_site.auth_atom_id
            _atom_site.pdbx_PDB_model_num
            ATOM 1 N N . ALA A 1 ? 0.000 0.000 0.000 1.00 80.00 ? 1 ALA A N 1
            ATOM 2 C CA . ALA A 1 ? 1.458 0.000 0.000 1.00 80.00 ? 1 ALA A CA 1
            ATOM 3 N N . GLY A 2 ? 3.314 1.420 0.000 1.00 60.00 ? 2 GLY A N 1
            #
            """
        )
        confidence_json = {"ptm": 0.83, "iptm": 0.0, "confidence_score": 0.78}
        engine = Boltz()
        protein = engine._parse_outputs(
            cif_text=cif,
            confidence_json=confidence_json,
            sequence="AG",
        )

        _assert_provenance(
            protein.metadata,
            engine="Boltz",
            expected_param_keys={
                "model_version",
                "use_msa_server",
                "recycling_steps",
                "diffusion_samples",
                "sampling_steps",
                "device",
            },
            expected_input_keys={"sequence"},
        )


# ---------------------------------------------------------------------
# load_alphafold (io helper, not a wrapper class)
# ---------------------------------------------------------------------


class TestLoadAlphaFoldProvenance:
    def test_load_alphafold_attaches_provenance(self, tmp_path: Path) -> None:
        from molforge.io import load_alphafold

        pdb_path = tmp_path / "af.pdb"
        pdb_path.write_text(_SYNTH_PDB)

        protein = load_alphafold(pdb_path)

        prov = _assert_provenance(
            protein.metadata,
            # load_alphafold is a loader, not an engine — the engine
            # name reflects that. The original AlphaFold pipeline ran
            # at some earlier time; molforge.io.load_alphafold is
            # what produced *this in-memory object*.
            engine="load_alphafold",
            expected_param_keys=set(),
            expected_input_keys={"path"},
        )
        assert str(pdb_path) in prov.inputs["path"]


# ---------------------------------------------------------------------
# Docking wrappers
# ---------------------------------------------------------------------


class TestDockingProvenance:
    """Vina and DiffDock produce DockingResult objects whose
    metadata[PROVENANCE] describes the whole run. Per-pose metadata
    keeps engine-specific extras (confidence, source_file) but no
    per-pose provenance — poses aren't independently produced."""

    def test_vina_parse_poses(self) -> None:
        """Drive the parser seam directly with a synthetic 1-pose
        PDBQT output. Avoids needing the real vina binary."""
        from molforge.wrappers.docking.vina import Vina

        # Minimal Vina-style PDBQT output: one MODEL with a REMARK
        # VINA RESULT and a couple of ATOM lines.
        text = textwrap.dedent(
            """\
            MODEL 1
            REMARK VINA RESULT:     -8.4    0.000    0.000
            ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
            ATOM      2  O   LIG A   1       1.230   0.000   0.000  1.00  0.00     0.000 O
            ENDMDL
            """
        )

        result = Vina()._parse_poses_pdbqt(
            text,
            provenance_parameters={
                "center": [0.0, 0.0, 0.0],
                "box_size": [20.0, 20.0, 20.0],
                "exhaustiveness": 8,
                "n_poses": 9,
            },
            provenance_inputs={
                "receptor": "/tmp/receptor.pdbqt",
                "ligand": "/tmp/ligand.sdf",
            },
        )

        prov = _assert_provenance(
            result.metadata,
            engine="Vina",
            expected_param_keys={"center", "box_size", "exhaustiveness", "n_poses"},
            expected_input_keys={"receptor", "ligand"},
        )
        # Verify the prov has no parent (no upstream provenance was
        # passed); subsequent end-to-end tests cover parent chaining.
        assert prov.parent is None

    def test_diffdock_parse_outputs(self, tmp_path: Path) -> None:
        """Drive DiffDock's _parse_outputs seam with a synthetic
        rank1 SDF — avoids needing torch / the real DiffDock repo."""
        from molforge.wrappers.docking.diffdock import DiffDock

        out_dir = tmp_path / "out"
        complex_dir = out_dir / "complex_0"
        complex_dir.mkdir(parents=True)
        # Minimal valid V2000 SDF: title / blank / blank / counts /
        # atom block / M END / $$$$.
        sdf = (
            "test_ligand\n"
            "\n"
            "\n"
            "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "M  END\n"
            "$$$$\n"
        )
        (complex_dir / "rank1_confidence0.50.sdf").write_text(sdf)

        engine = DiffDock(samples_per_complex=5)
        result = engine._parse_outputs(
            out_dir,
            receptor=None,
            receptor_ref="/tmp/rec.pdb",
            ligand_ref="/tmp/lig.sdf",
        )

        _assert_provenance(
            result.metadata,
            engine="DiffDock",
            expected_param_keys={"samples_per_complex", "inference_steps", "batch_size"},
            expected_input_keys={"receptor", "ligand"},
        )

    def test_diffdock_without_refs_skips_provenance(self, tmp_path: Path) -> None:
        """When neither receptor_ref nor ligand_ref is provided, no
        Provenance is attached — the legacy parse path stays usable
        for tests that don't care about provenance."""
        from molforge.wrappers.docking.diffdock import DiffDock

        out_dir = tmp_path / "out"
        complex_dir = out_dir / "complex_0"
        complex_dir.mkdir(parents=True)
        sdf = (
            "test_ligand\n"
            "\n"
            "\n"
            "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "M  END\n"
            "$$$$\n"
        )
        (complex_dir / "rank1_confidence0.50.sdf").write_text(sdf)

        result = DiffDock()._parse_outputs(out_dir, receptor=None)
        assert mk.PROVENANCE not in result.metadata


# ---------------------------------------------------------------------
# Generative wrappers
# ---------------------------------------------------------------------


class TestGenerativeProvenance:
    def test_rfdiffusion_parse_outputs(self, tmp_path: Path) -> None:
        """Drive RFdiffusion's _parse_outputs seam with synthetic
        design PDBs — no need for the real RFdiffusion install."""
        from molforge.wrappers.generative.rfdiffusion import RFdiffusion

        (tmp_path / "design_0.pdb").write_text(_SYNTH_PDB)
        (tmp_path / "design_1.pdb").write_text(_SYNTH_PDB)

        engine = RFdiffusion()
        designs = engine._parse_outputs(
            tmp_path,
            source_args={
                "length": 50,
                "target_pdb": None,
                "contigs": None,
                "hotspot_residues": None,
                "symmetry": None,
                "diffusion_steps": 50,
                "num_designs": 2,
            },
        )

        assert len(designs) == 2
        for i, design in enumerate(designs):
            prov = _assert_provenance(
                design.metadata,
                engine="RFdiffusion",
                expected_param_keys={
                    "length",
                    "diffusion_steps",
                    "num_designs",
                    "rfdiffusion_dir",
                    "python_executable",
                    "device",
                },
                expected_input_keys=set(),
            )
            # design_index stays separate from provenance.parameters.
            assert design.metadata["design_index"] == i
            assert "design_index" not in prov.parameters

    def test_proteinmpnn_parse_fasta_provenance_via_design(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ProteinMPNN attaches Provenance inside design(), not
        _parse_fasta. We mock _resolve_proteinmpnn_dir and _run_cli
        to drive the attachment path without needing the binary."""
        from molforge.generative import DesignedSequence
        from molforge.wrappers.generative.proteinmpnn import ProteinMPNN

        synthetic_designs = [
            DesignedSequence(sequence="MKLA", score=-1.2),
            DesignedSequence(sequence="MKLG", score=-1.5),
        ]

        engine = ProteinMPNN()

        monkeypatch.setattr(engine, "_resolve_proteinmpnn_dir", lambda: Path("/fake"))
        monkeypatch.setattr(
            engine,
            "_run_cli",
            lambda **kwargs: synthetic_designs,
        )

        designs = engine.generate(backbone="/tmp/scaffold.pdb")

        assert len(designs) == 2
        # All designs share the SAME Provenance object (immutable, so
        # this is by-reference sharing).
        provs = [d.metadata[mk.PROVENANCE] for d in designs]
        assert provs[0] is provs[1]

        _assert_provenance(
            designs[0].metadata,
            engine="ProteinMPNN",
            expected_param_keys={
                "model_name",
                "use_soluble_model",
                "ca_only",
                "num_seqs",
                "sampling_temp",
                "omit_aas",
                "seed",
                "chains_to_design",
                "fixed_positions",
                "proteinmpnn_dir",
                "python_executable",
            },
            expected_input_keys={"backbone"},
        )


# ---------------------------------------------------------------------
# Parent-chain integration
# ---------------------------------------------------------------------


class TestProvenanceChaining:
    """When a wrapper receives a Protein with existing Provenance, the
    output's Provenance.parent must point to it — that's what makes
    the fold-dock-md chain traceable."""

    def test_vina_chains_to_receptor_provenance(self) -> None:
        """A receptor that's a Protein with provenance: Vina's output
        DockingResult.metadata[PROVENANCE].parent points back to the
        receptor's provenance."""
        from molforge.core import AtomArray, Protein
        from molforge.wrappers.docking.vina import Vina

        # Build a receptor Protein with its own provenance (as if it
        # came from a folding engine).
        receptor = Protein(AtomArray(0), name="rec")
        receptor_prov = Provenance.from_engine(
            engine="ESMFold",
            parameters={"model_name": "esmfold_v1"},
            inputs={"sequence": "ACDEF"},
        )
        receptor.metadata[mk.PROVENANCE] = receptor_prov

        # Drive parse with parent=receptor_prov.
        text = textwrap.dedent(
            """\
            MODEL 1
            REMARK VINA RESULT:     -8.4    0.000    0.000
            ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
            ENDMDL
            """
        )
        result = Vina()._parse_poses_pdbqt(
            text,
            provenance_parameters={"exhaustiveness": 8},
            provenance_inputs={"receptor": "rec", "ligand": "/tmp/l.sdf"},
            provenance_parent=receptor_prov,
        )

        result_prov = result.metadata[mk.PROVENANCE]
        assert result_prov.parent is receptor_prov
        # chain() now lists the producers oldest-first.
        engines = [step.engine for step in result_prov.chain()]
        assert engines == ["ESMFold", "Vina"]


# =====================================================================
# Pass 2: MD wrappers and prep functions
# =====================================================================
#
# Pass 1 used parent chaining only ACROSS wrappers (ESMFold -> Vina).
# Pass 2 exercises chaining WITHIN a single wrapper's multi-step
# pipeline (OpenMM/GROMACS prepare -> minimize -> run) and across the
# four prep functions chained by prepare_for_md.
#
# OpenMM and PDBFixer are reasonable to require in CI; GROMACS needs a
# real `gmx` binary which most CI environments lack, so the GROMACS
# provenance assertions stay at the unit-test level (drive the call
# chain without actually invoking gmx).


def _openmm_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("openmm") is not None


def _pdbfixer_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("pdbfixer") is not None


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures"


@pytest.mark.skipif(not _openmm_available(), reason="openmm not installed")
class TestOpenMMProvenanceChain:
    """A full OpenMM prepare -> minimize -> run pipeline must produce
    a Trajectory whose Provenance.chain() reads as the three steps
    oldest-first. When the input Protein already has a Provenance
    (e.g. it came from ESMFold), the chain extends back through it."""

    def test_three_step_chain(self) -> None:
        """prepare -> minimize -> run, no upstream provenance."""
        from molforge.io import load
        from molforge.wrappers.md.openmm import OpenMM

        tripeptide = load(_fixtures_dir() / "pdb" / "ala_tripeptide_heavy.pdb")

        engine = OpenMM(platform="CPU")
        sim = engine.prepare(tripeptide, force_field="amber14-all")
        sim = engine.minimize(sim, max_iterations=5)
        traj = engine.run(sim, n_steps=10, save_every=10)

        prov = traj.metadata[mk.PROVENANCE]
        assert isinstance(prov, Provenance)
        engines = [step.engine for step in prov.chain()]
        assert engines == ["OpenMM.prepare", "OpenMM.minimize", "OpenMM.run"]

    def test_chain_extends_through_upstream(self) -> None:
        """If the input Protein has a Provenance (simulating an
        ESMFold prediction), the OpenMM chain extends back through
        it so the final Trajectory traces all the way to the
        sequence."""
        from molforge.io import load
        from molforge.wrappers.md.openmm import OpenMM

        tripeptide = load(_fixtures_dir() / "pdb" / "ala_tripeptide_heavy.pdb")
        # Pretend this Protein came from a folding engine.
        tripeptide.metadata[mk.PROVENANCE] = Provenance.from_engine(
            engine="ESMFold",
            parameters={"model_name": "esmfold_v1"},
            inputs={"sequence": "AAA"},
        )

        engine = OpenMM(platform="CPU")
        sim = engine.prepare(tripeptide, force_field="amber14-all")
        traj = engine.run(sim, n_steps=10, save_every=10)

        prov = traj.metadata[mk.PROVENANCE]
        engines = [step.engine for step in prov.chain()]
        # ESMFold sat upstream; the MD pipeline adds prepare -> run.
        assert engines == ["ESMFold", "OpenMM.prepare", "OpenMM.run"]


class TestGROMACSProvenanceWiring:
    """GROMACS needs the `gmx` binary which CI usually lacks, so we
    can't run the pipeline end-to-end here. We can still verify the
    Provenance attachment by checking that GROMACS's metadata-build
    paths reference the right Provenance fields — that's a tighter
    unit-test than nothing.

    We do this by inspecting the source: every Simulation/Trajectory
    return statement should include a PROVENANCE key. Reading the
    AST would be more precise; a simple substring check is good
    enough as a regression-net for the adoption pattern."""

    def test_prepare_attaches_provenance(self) -> None:
        from molforge.wrappers.md import gromacs

        src = Path(gromacs.__file__).read_text()
        # Each of the three pipeline steps emits a Provenance with the
        # expected engine string.
        assert 'engine="GROMACS.prepare"' in src
        assert 'engine="GROMACS.minimize"' in src
        assert 'engine="GROMACS.run"' in src

    def test_parent_provenance_helper_used(self) -> None:
        from molforge.wrappers.md import gromacs

        src = Path(gromacs.__file__).read_text()
        # Every chained step uses _parent_provenance(...) to extract
        # the parent — not the raw .get() which would skip the type
        # narrowing.
        assert "_parent_provenance(" in src


# ---------------------------------------------------------------------
# Prep functions
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not (_openmm_available() and _pdbfixer_available()),
    reason="openmm + pdbfixer required for prep",
)
class TestPrepProvenanceChain:
    """The prep subpackage's five functions chain Provenance through
    metadata. Calling prepare_for_md (which composes four of them in
    sequence) leaves the result with a 4-deep chain that reads as
    the pipeline oldest-first."""

    def test_remove_heterogens_attaches_provenance(self) -> None:
        from molforge.io import load
        from molforge.prep import remove_heterogens

        protein = load(_fixtures_dir() / "pdb" / "ala_tripeptide_heavy.pdb")
        cleaned = remove_heterogens(protein)

        _assert_provenance(
            cleaned.metadata,
            engine="molforge.prep.remove_heterogens",
            expected_param_keys={"keep_water", "keep_ions", "keep_ligands", "keep"},
            expected_input_keys=set(),
        )

    def test_chain_through_two_prep_calls(self) -> None:
        from molforge.io import load
        from molforge.prep import fix_missing_atoms, remove_heterogens

        protein = load(_fixtures_dir() / "pdb" / "ala_tripeptide_heavy.pdb")
        cleaned = remove_heterogens(protein)
        fixed = fix_missing_atoms(cleaned)

        prov = fixed.metadata[mk.PROVENANCE]
        engines = [step.engine for step in prov.chain()]
        assert engines == [
            "molforge.prep.remove_heterogens",
            "molforge.prep.fix_missing_atoms",
        ]

    def test_prepare_for_md_full_chain(self) -> None:
        """The composite: remove_heterogens -> fix_missing_atoms ->
        add_caps -> add_hydrogens leaves the result with a 4-deep
        Provenance chain."""
        from molforge.io import load
        from molforge.prep import prepare_for_md

        protein = load(_fixtures_dir() / "pdb" / "ala_tripeptide_heavy.pdb")
        prepared = prepare_for_md(protein)

        prov = prepared.metadata[mk.PROVENANCE]
        engines = [step.engine for step in prov.chain()]
        assert engines == [
            "molforge.prep.remove_heterogens",
            "molforge.prep.fix_missing_atoms",
            "molforge.prep.add_caps",
            "molforge.prep.add_hydrogens",
        ]

    def test_prepare_for_md_chain_extends_through_upstream(self) -> None:
        """When the input Protein already has Provenance, the prep
        chain extends back through it. This is the headline
        traceability scenario: sequence -> ESMFold -> 4 prep steps."""
        from molforge.io import load
        from molforge.prep import prepare_for_md

        protein = load(_fixtures_dir() / "pdb" / "ala_tripeptide_heavy.pdb")
        protein.metadata[mk.PROVENANCE] = Provenance.from_engine(
            engine="ESMFold",
            parameters={"model_name": "esmfold_v1"},
            inputs={"sequence": "AAA"},
        )

        prepared = prepare_for_md(protein)
        engines = [step.engine for step in prepared.metadata[mk.PROVENANCE].chain()]
        assert engines == [
            "ESMFold",
            "molforge.prep.remove_heterogens",
            "molforge.prep.fix_missing_atoms",
            "molforge.prep.add_caps",
            "molforge.prep.add_hydrogens",
        ]
