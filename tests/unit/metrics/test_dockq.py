"""Tests for DockQ complex-quality metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import Protein
from molforge.core.atom_array import AtomArray
from molforge.io import read_pdb
from molforge.metrics import dockq, fnat, irms, lrms

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestFnat:
    def test_native_vs_self_is_one(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        assert fnat(p, p) == pytest.approx(1.0)

    def test_good_model_recovers_most_contacts(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        model = read_pdb(FIXTURES / "mini_complex_good.pdb")
        # Small noise (0.3 Å) means most native contacts survive
        f = fnat(model, native)
        assert 0.5 < f <= 1.0

    def test_bad_model_loses_all_contacts(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        bad = read_pdb(FIXTURES / "mini_complex_bad.pdb")
        # Chain B is 30 Å away -- no interface contacts survive
        assert fnat(bad, native) == 0.0

    def test_default_chains_auto_picked(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        # Should automatically pick A and B
        result = fnat(native, native)
        assert result == pytest.approx(1.0)

    def test_explicit_chains(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        assert fnat(native, native, chain_a="A", chain_b="B") == pytest.approx(1.0)

    def test_fnat_robust_to_atom_indexing(self) -> None:
        """Fnat counts residue-residue contacts (CAPRI definition), so a
        model differing from the reference only in per-atom indexing must
        still recover every native contact. Here the model is the native
        with one extra heavy atom prepended to chain A (far from the
        interface, adding no contact) — which shifts every chain-A atom
        index. An atom-index-based intersection scored this ~0.67; the
        residue-level definition correctly gives 1.0.
        """
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        arr = native.atom_array
        ai = next(
            i
            for i in range(arr.n_atoms)
            if str(arr.chain_id[i]) == "A" and str(arr.entity_type[i]) == "protein"
        )
        far = (arr.coords.mean(axis=0) + np.array([100.0, 100.0, 100.0], np.float32)).reshape(1, 3)
        extra = AtomArray.from_dict(
            {
                "coords": far.astype(np.float32),
                "element": np.array(["C"], dtype="U2"),
                "atom_name": np.array(["CB"], dtype="U4"),
                "chain_id": np.array([str(arr.chain_id[ai])], dtype="U4"),
                "residue_id": np.array([int(arr.residue_id[ai])], dtype="int32"),
                "insertion_code": np.array(
                    [str(arr.insertion_code[ai])], dtype=arr.insertion_code.dtype
                ),
                "model_id": np.array([int(arr.model_id[ai])], dtype="int32"),
                "entity_type": np.array(["protein"], dtype="U8"),
            }
        )
        model = Protein(extra.append(arr))
        assert model.atom_array.n_atoms == native.atom_array.n_atoms + 1
        assert fnat(model, native) == pytest.approx(1.0)


class TestIrms:
    def test_native_vs_self_is_zero(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        # iRMS on self is zero (or near-zero for floating point)
        assert irms(p, p) == pytest.approx(0.0, abs=1e-4)

    def test_good_model_has_low_irms(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        model = read_pdb(FIXTURES / "mini_complex_good.pdb")
        # Backbone perturbed by 0.3 Å noise -> iRMS should be < 1 Å
        score = irms(model, native)
        assert 0.1 < score < 1.0

    def test_irms_robust_to_missing_backbone_atom(self) -> None:
        """iRMS selects backbone atoms by residue position, so a residue
        missing one must not shift the indexing. The old flat p*4 slicing
        crashed (or silently paired the wrong atoms). Dropping a
        non-interface O from the model leaves the interface geometry
        identical, so iRMS stays ~0.
        """
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        arr = native.atom_array
        first = next(
            sl
            for sl in arr.iter_residue_slices()
            if str(arr.chain_id[sl.start]) == "A" and str(arr.entity_type[sl.start]) == "protein"
        )
        o_global = first.start + int(np.where(arr.atom_name[first] == "O")[0][0])
        mask = np.ones(arr.n_atoms, dtype=bool)
        mask[o_global] = False
        model = Protein(arr.select(mask))
        assert model.atom_array.n_atoms == native.atom_array.n_atoms - 1
        assert irms(model, native) == pytest.approx(0.0, abs=1e-4)


class TestLrms:
    def test_native_vs_self_is_zero(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        assert lrms(p, p) == pytest.approx(0.0, abs=1e-4)

    def test_bad_model_has_huge_lrms(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        bad = read_pdb(FIXTURES / "mini_complex_bad.pdb")
        # Ligand chain shifted 30 Å away after superposing the receptor
        assert lrms(bad, native) > 10.0


class TestDockQ:
    def test_native_vs_self_is_one(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        result = dockq(p, p)
        assert result["DockQ"] == pytest.approx(1.0, abs=1e-3)
        assert result["fnat"] == pytest.approx(1.0)
        assert result["iRMS"] == pytest.approx(0.0, abs=1e-4)
        assert result["LRMS"] == pytest.approx(0.0, abs=1e-4)

    def test_good_model_high_score(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        model = read_pdb(FIXTURES / "mini_complex_good.pdb")
        result = dockq(model, native)
        # Most metrics should be in the "high quality" band
        assert result["DockQ"] > 0.7

    def test_bad_model_low_score(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        bad = read_pdb(FIXTURES / "mini_complex_bad.pdb")
        result = dockq(bad, native)
        # No native contacts + huge LRMS = very low DockQ
        assert result["DockQ"] < 0.3
        assert result["fnat"] == 0.0
        assert result["LRMS"] > 10.0

    def test_keys_present(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        result = dockq(p, p)
        assert set(result.keys()) == {"DockQ", "fnat", "iRMS", "LRMS"}
