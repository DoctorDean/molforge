"""Tests for the P2Rank pocket-detection wrapper.

P2Rank is a Java application that can't run in CI, so these tests target
the non-trivial logic — the predictions-CSV parser and the not-installed
guard — the same way the fpocket wrapper is tested. The subprocess
invocation itself is a thin shell.
"""

from __future__ import annotations

import numpy as np
import pytest

from molforge.core import Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.docking import Pocket
from molforge.wrappers.pockets import P2RankNotInstalledError, detect_pockets_p2rank
from molforge.wrappers.pockets.p2rank import (
    _parse_p2rank_predictions,
    _parse_residue_ids,
    _read_csv_rows,
)

# A realistic P2Rank <name>_predictions.csv: space-padded, comma-separated,
# with space-separated residue_ids / surf_atom_ids list columns.
_REAL_CSV = (
    "rank,   name,   score, probability, sas_points, surf_atoms,"
    "   center_x,   center_y,   center_z, residue_ids, surf_atom_ids\n"
    "   1, pocket1,  28.14,       0.812,        142,         56,"
    "     12.340,      5.670,     -3.210, A_45 A_46 A_50 B_12,  1 2 3\n"
    "   2, pocket2,   9.03,       0.240,         40,         18,"
    "      0.000,     -1.230,      8.900, A_88 A_90,  7 8\n"
)


class TestReadCsvRows:
    def test_parses_two_rows(self) -> None:
        rows = _read_csv_rows(_REAL_CSV)
        assert len(rows) == 2
        assert rows[0]["name"] == "pocket1"
        assert rows[0]["score"] == "28.14"
        assert rows[0]["residue_ids"] == "A_45 A_46 A_50 B_12"

    def test_header_only_returns_empty(self) -> None:
        assert _read_csv_rows("rank, name, score\n") == []

    def test_empty_returns_empty(self) -> None:
        assert _read_csv_rows("") == []

    def test_malformed_row_skipped(self) -> None:
        csv = "rank,name,score\n1,pocket1\n2,pocket2,3.0\n"
        rows = _read_csv_rows(csv)
        assert len(rows) == 1  # the short row is dropped
        assert rows[0]["name"] == "pocket2"


class TestParseResidueIds:
    def test_basic(self) -> None:
        assert _parse_residue_ids("A_45 A_46 B_12") == [("A", 45, ""), ("A", 46, ""), ("B", 12, "")]

    def test_insertion_code(self) -> None:
        assert _parse_residue_ids("A_45_B") == [("A", 45, "B")]

    def test_skips_unparseable_tokens(self) -> None:
        # "99X" isn't an int; "garbage" has no underscore; both dropped.
        assert _parse_residue_ids("A_45 X_99X garbage B_7") == [("A", 45, ""), ("B", 7, "")]

    def test_empty(self) -> None:
        assert _parse_residue_ids("") == []


class TestParsePredictions:
    def _pockets(self) -> list[Pocket]:
        return _parse_p2rank_predictions(
            _REAL_CSV, protein=Protein(name="test"), prank_executable="prank"
        )

    def test_two_pockets_best_first(self) -> None:
        pockets = self._pockets()
        assert len(pockets) == 2
        assert [p.rank for p in pockets] == [0, 1]
        assert all(isinstance(p, Pocket) for p in pockets)

    def test_field_mapping(self) -> None:
        p = self._pockets()[0]
        assert p.score == pytest.approx(28.14)
        assert p.druggability == pytest.approx(0.812)  # <- probability
        assert p.volume is None  # P2Rank reports no volume
        np.testing.assert_allclose(p.center, [12.34, 5.67, -3.21], atol=1e-4)
        assert p.residues[0] == ("A", 45, "")
        assert len(p.residues) == 4

    def test_zero_coordinate_preserved(self) -> None:
        # A legitimate 0.0 centre must not collapse to nan.
        assert self._pockets()[1].center[0] == pytest.approx(0.0)

    def test_metadata_and_provenance(self) -> None:
        p = self._pockets()[0]
        assert p.metadata["engine"] == "p2rank"
        assert isinstance(p.metadata[mk.PROVENANCE], Provenance)
        assert p.metadata[mk.PROVENANCE].engine == "p2rank"
        assert p.metadata["descriptors"]["probability"] == "0.812"

    def test_provenance_parents_input_chain(self) -> None:
        parent = Provenance.from_engine(engine="ESMFold")
        prot = Protein(name="folded")
        prot.metadata[mk.PROVENANCE] = parent
        pockets = _parse_p2rank_predictions(_REAL_CSV, protein=prot, prank_executable="prank")
        assert pockets[0].metadata[mk.PROVENANCE].parent is parent

    def test_no_pockets_returns_empty(self) -> None:
        assert (
            _parse_p2rank_predictions(
                "rank, name, score\n", protein=Protein(name="x"), prank_executable="prank"
            )
            == []
        )


class TestNotInstalled:
    def test_missing_binary_raises(self) -> None:
        with pytest.raises(P2RankNotInstalledError, match="not found"):
            detect_pockets_p2rank(Protein(name="x"), prank_executable="molforge_no_such_prank")
