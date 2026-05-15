"""Integration tests that exercise the full library against realistic fixtures.

These hit the IO -> data model -> structural analysis pipeline end-to-end,
which catches integration bugs that pure unit tests miss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb, write_pdb
from molforge.structure import (
    centroid,
    contact_map,
    dssp_3state,
    phi_psi_omega,
    radius_of_gyration,
    rmsd,
    sasa_per_residue,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pdb"


class TestMixedStructure:
    """The mini_mixed fixture has helix + loop + strand topology."""

    @pytest.fixture
    def protein(self):  # type: ignore[no-untyped-def]
        return read_pdb(FIXTURES / "mini_mixed.pdb")

    def test_loads_15_residues(self, protein) -> None:  # type: ignore[no-untyped-def]
        assert protein.n_residues == 15
        assert protein.n_chains == 1

    def test_dssp_finds_helix_and_strand(self, protein) -> None:  # type: ignore[no-untyped-def]
        ss = dssp_3state(protein)
        # We expect both H and E in this mixed structure
        assert "H" in ss
        assert "E" in ss

    def test_radius_of_gyration_reasonable(self, protein) -> None:  # type: ignore[no-untyped-def]
        rg = radius_of_gyration(protein)
        # For a ~50 A end-to-end peptide, Rg should be in the ballpark of 5-15 A.
        assert 3.0 < rg < 25.0

    def test_phi_psi_recoverable(self, protein) -> None:  # type: ignore[no-untyped-def]
        phi, psi, _ = phi_psi_omega(protein)
        # Chain termini are NaN, but the rest of the chain should have
        # defined phi/psi spanning a wide range (helix ~ 60, strand ~ 120).
        valid_phi = phi[~np.isnan(phi)]
        valid_psi = psi[~np.isnan(psi)]
        # Helix and strand regions have notably different phi values
        assert valid_phi.max() - valid_phi.min() > 30.0
        assert valid_psi.max() - valid_psi.min() > 30.0

    def test_sasa_residues_nonnegative(self, protein) -> None:  # type: ignore[no-untyped-def]
        # Use a cheap sphere-point count for integration tests
        sasa = sasa_per_residue(protein, n_sphere_points=32)
        assert sasa.shape == (15,)
        assert (sasa >= 0).all()


class TestNMREnsemble:
    """The mini_ensemble fixture has 3 NMR-style models."""

    def test_three_models_loaded(self) -> None:
        p = read_pdb(FIXTURES / "mini_ensemble.pdb")
        models = {int(m) for m in p.atom_array.model_id}
        assert models == {1, 2, 3}

    def test_load_specific_model(self) -> None:
        p = read_pdb(FIXTURES / "mini_ensemble.pdb", model=2)
        models = {int(m) for m in p.atom_array.model_id}
        assert models == {2}

    def test_models_differ_slightly(self) -> None:
        """The ensemble was built with random noise — models should not be identical."""
        m1 = read_pdb(FIXTURES / "mini_ensemble.pdb", model=1)
        m2 = read_pdb(FIXTURES / "mini_ensemble.pdb", model=2)
        assert m1.n_atoms == m2.n_atoms
        # They should differ but not by a huge amount (noise sigma was 0.15 A)
        diff = np.abs(m1.atom_array.coords - m2.atom_array.coords)
        assert diff.max() > 0.01  # not identical
        assert diff.max() < 2.0  # not crazy different


class TestLigandStructure:
    """The mini_with_ligand fixture has protein + ligand + water."""

    @pytest.fixture
    def protein(self):  # type: ignore[no-untyped-def]
        return read_pdb(FIXTURES / "mini_with_ligand.pdb")

    def test_entity_classification(self, protein) -> None:  # type: ignore[no-untyped-def]
        et = protein.atom_array.entity_type
        entity_counts = {}
        for e in np.unique(et):
            entity_counts[str(e)] = int((et == e).sum())
        assert entity_counts.get("protein", 0) == 20
        assert entity_counts.get("ligand", 0) == 5
        assert entity_counts.get("water", 0) == 2

    def test_protein_only_drops_hetero(self, protein) -> None:  # type: ignore[no-untyped-def]
        prot_only = protein.protein_only()
        assert prot_only.n_atoms == 20  # 5 residues x 4 backbone atoms
        # No ligand chain
        assert "L" not in [c.chain_id for c in prot_only.chains]

    def test_remove_water(self, protein) -> None:  # type: ignore[no-untyped-def]
        dry = protein.remove_water()
        # Drops 2 water atoms
        assert dry.n_atoms == protein.n_atoms - 2

    def test_sequence_skips_hetero(self, protein) -> None:  # type: ignore[no-untyped-def]
        seq = protein.sequence
        # Only the 5 protein residues count; ligand and water are skipped.
        assert len(seq.replace("/", "")) == 5


class TestRoundTripOnFixtures:
    """Write and re-read each fixture; structure must survive intact."""

    @pytest.mark.parametrize(
        "fixture",
        ["mini_beta_sheet.pdb", "mini_mixed.pdb", "mini_with_ligand.pdb"],
    )
    def test_round_trip_preserves_atoms_and_coords(self, fixture: str, tmp_path: Path) -> None:
        original = read_pdb(FIXTURES / fixture)
        out = tmp_path / f"rt_{fixture}"
        write_pdb(original, out)
        reloaded = read_pdb(out)
        assert reloaded.n_atoms == original.n_atoms
        assert reloaded.n_residues == original.n_residues
        np.testing.assert_allclose(
            reloaded.atom_array.coords,
            original.atom_array.coords,
            atol=1e-3,
        )

    def test_ensemble_round_trip_preserves_models(self, tmp_path: Path) -> None:
        original = read_pdb(FIXTURES / "mini_ensemble.pdb")
        out = tmp_path / "rt_ensemble.pdb"
        write_pdb(original, out)
        reloaded = read_pdb(out)
        assert {int(m) for m in reloaded.atom_array.model_id} == {1, 2, 3}


class TestAnalysisPipeline:
    """End-to-end analysis chain on a single fixture."""

    def test_full_pipeline(self) -> None:
        """Load -> structural analysis -> mutation -> compare."""
        from molforge.sequence import mutate_protein

        p = read_pdb(FIXTURES / "mini_mixed.pdb")

        # Initial analysis
        rg_wt = radius_of_gyration(p)
        ss_wt = dssp_3state(p)
        c_wt = centroid(p)

        # Mutate one residue
        mut = mutate_protein(p, "A5G", chain_id="A")

        # Sequence-only mutation: geometry should be identical
        assert radius_of_gyration(mut) == pytest.approx(rg_wt, abs=0.001)
        assert centroid(mut)[0] == pytest.approx(c_wt[0], abs=0.001)

        # DSSP unchanged (sequence-only mutation doesn't move atoms)
        ss_mut = dssp_3state(mut)
        assert ss_mut == ss_wt

        # RMSD = 0 because we didn't move anything
        assert rmsd(p, mut, subset="ca") == pytest.approx(0.0, abs=1e-5)

    def test_contact_map_density(self) -> None:
        """Compact structures should have a denser contact map than extended ones."""
        # The helix fixture is compact; beta_sheet was built with strands
        # only loosely adjacent. We just check the shape and that the map
        # is non-empty.
        p = read_pdb(FIXTURES / "mini_mixed.pdb")
        cmap = contact_map(p, cutoff=8.0, atom_choice="ca")
        assert cmap.shape == (15, 15)
        # Some contacts exist for a 15-residue structure
        assert cmap.sum() > 0
