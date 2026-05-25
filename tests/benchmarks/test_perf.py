"""Performance benchmarks for molforge's structural-analysis hot paths.

These establish baseline timings for the five functions most likely
to sit in an inner loop of a real pipeline: RMSD, DSSP, lDDT, distance
maps, and sequence alignment. They exist to catch performance
*regressions* — a future change that makes one of these noticeably
slower shows up as a slower benchmark.

Running them:

    pytest -m benchmark                 # run the benchmarks
    pytest -m benchmark --benchmark-only
    pytest -m benchmark --benchmark-save=baseline   # save a baseline
    pytest -m benchmark --benchmark-compare         # compare to it

They require ``pytest-benchmark`` (in the ``[dev]`` extra) and are
marked ``benchmark`` so a normal ``pytest`` run skips them — they're
measurement tools, not correctness tests, and timing assertions in a
plain test run would be flaky on shared CI hardware.

Each benchmark still asserts that the function returns something
sensible, so a benchmark that silently starts measuring a no-op
(e.g. because the function began returning early) is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from molforge.metrics import lddt
from molforge.sequence import align
from molforge.structure import contact_map, distance_map, dssp, rmsd

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


class TestRmsdBenchmark:
    def test_rmsd_200_residues(self, benchmark, helix_200, helix_200_perturbed) -> None:
        """RMSD between two 200-residue structures (with Kabsch alignment)."""
        result = benchmark(rmsd, helix_200, helix_200_perturbed)
        assert result >= 0.0

    def test_rmsd_no_align_200_residues(
        self, benchmark, helix_200, helix_200_perturbed
    ) -> None:
        """RMSD without superposition — isolates the distance math from Kabsch."""
        result = benchmark(rmsd, helix_200, helix_200_perturbed, align=False)
        assert result >= 0.0


class TestDsspBenchmark:
    def test_dssp_200_residues(self, benchmark, helix_200) -> None:
        """DSSP secondary-structure assignment over a 200-residue protein."""
        result = benchmark(dssp, helix_200)
        assert "codes_8" in result
        assert len(result["codes_8"]) == helix_200.n_residues  # type: ignore[arg-type]


class TestLddtBenchmark:
    def test_lddt_200_residues(self, benchmark, helix_200, helix_200_perturbed) -> None:
        """lDDT between two 200-residue structures."""
        result = benchmark(lddt, helix_200, helix_200_perturbed)
        assert 0.0 <= result <= 1.0


class TestDistanceMapBenchmark:
    def test_distance_map_200_residues(self, benchmark, helix_200) -> None:
        """CA-CA distance map for a 200-residue protein (200x200 matrix)."""
        result = benchmark(distance_map, helix_200)
        assert result.shape == (helix_200.n_residues, helix_200.n_residues)

    def test_contact_map_200_residues(self, benchmark, helix_200) -> None:
        """CB-CB contact map for a 200-residue protein."""
        result = benchmark(contact_map, helix_200)
        assert result.shape == (helix_200.n_residues, helix_200.n_residues)
        assert result.dtype == np.bool_


class TestAlignmentBenchmark:
    # A 200-residue sequence pair. The second is the first with a
    # handful of point mutations, so the aligner does real work
    # (a perfect match is a trivially easy DP fill).
    _SEQ_A = "ACDEFGHIKLMNPQRSTVWY" * 10
    _SEQ_B = (
        "ACDEFGHIKLMNPQRSTVWY" * 4
        + "ACDEFGHIKLMNPQRSTVWA"   # final Y->A
        + "ACDEFGHIKLMNPQRSTVWY" * 5
    )

    def test_global_align_200_residues(self, benchmark) -> None:
        """Global (Needleman-Wunsch) alignment of two ~200-residue sequences."""
        result = benchmark(align, self._SEQ_A, self._SEQ_B, mode="global")
        assert result.score is not None

    def test_local_align_200_residues(self, benchmark) -> None:
        """Local (Smith-Waterman) alignment of two ~200-residue sequences."""
        result = benchmark(align, self._SEQ_A, self._SEQ_B, mode="local")
        assert result.score is not None
