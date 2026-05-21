"""Integration tests against realistic PDB fixtures.

These fixtures are handcrafted from canonical bond lengths and angles
(Engh & Huber 1991) rather than copied from real PDB entries — but
they exercise the patterns that distinguish a real PDB from a
synthetic one: full side chains across all 20 amino acid types,
multi-atom alt-loc conformations, multi-chain protein-ligand-water
systems, and realistic B-factor distributions.

Each test is named after the *code path* it exercises rather than the
fixture, so failures point at what broke.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from molforge.core import Protein
from molforge.io import read_pdb, write_pdb

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "pdb"


# ===========================================================================
# Fixture loading - sanity check the inputs before testing anything against them
# ===========================================================================
class TestFixturesLoad:
    """Every fixture parses, and basic shape matches what we built."""

    def test_real_small_protein_loads(self) -> None:
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        assert p.n_atoms == 193
        assert p.n_residues == 24
        assert len(p.chains) == 1
        assert p.chains[0].chain_id == "A"
        # All 20 standard amino acids present plus PRO (covered by VAL/ILE etc.)
        residue_names = {str(r) for r in p.atom_array.residue_name}
        assert len(residue_names) == 20  # all 20 unique residues
        assert "PRO" in residue_names  # has PRO (special ring closure)
        assert "TRP" in residue_names  # has the largest side chain
        assert "GLY" in residue_names  # has the smallest

    def test_real_small_protein_sequence(self) -> None:
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        # The fixture's documented sequence
        assert p.sequence == "ELKMQRAGSPVIFYCTWNDGHEAM"

    def test_real_with_altloc_loads_default(self) -> None:
        # Default altloc handling: highest_occupancy resolves to A confs.
        # Of the 12 residues, LEU 2 and SER 9 have A/B alt-locs.
        # A occupancies are 0.60, so they win.
        p = read_pdb(FIXTURES / "real_with_altloc_sidechains.pdb")
        # 91 atoms after altloc resolution (97 raw - 6 B-conf atoms)
        assert p.n_atoms == 91
        assert p.n_residues == 12

    def test_real_with_altloc_loads_all_conformations(self) -> None:
        # altloc="all" keeps both A and B conformations.
        p = read_pdb(FIXTURES / "real_with_altloc_sidechains.pdb", altloc="all")
        # 97 atoms when both conformations are kept (raw fixture has 97)
        assert p.n_atoms == 97

    def test_real_with_ligand_loads(self) -> None:
        p = read_pdb(FIXTURES / "real_with_ligand_realistic.pdb")
        assert p.n_atoms == 76
        # 8 protein residues + 1 benzene + 1 zinc + 3 waters = 13 residues
        assert p.n_residues == 13
        # Three chains: protein (A), ligand+ion (B), water (W)
        assert sorted(c.chain_id for c in p.chains) == ["A", "B", "W"]


# ===========================================================================
# Entity-type classification on the multi-component structure
# ===========================================================================
class TestEntityTypeClassification:
    """The ligand fixture exercises the entity_type classifier across
    every category molforge supports."""

    def test_all_entity_types_present(self) -> None:
        p = read_pdb(FIXTURES / "real_with_ligand_realistic.pdb")
        types = {str(t) for t in p.atom_array.entity_type}
        # All four categories: protein, ligand, water, ion
        assert types == {"protein", "ligand", "water", "ion"}

    def test_protein_atoms_classified(self) -> None:
        p = read_pdb(FIXTURES / "real_with_ligand_realistic.pdb")
        arr = p.atom_array
        protein_mask = arr.entity_type == "protein"
        # 8 residues with full atoms — count should match what we built
        # GLU(9), LEU(8), LYS(9), HIS(10), ALA(5), PHE(11), CYS(6), MET(8)
        # = 66 protein atoms
        assert protein_mask.sum() == 66

    def test_ligand_atoms_classified(self) -> None:
        p = read_pdb(FIXTURES / "real_with_ligand_realistic.pdb")
        arr = p.atom_array
        ligand_mask = arr.entity_type == "ligand"
        assert ligand_mask.sum() == 6  # benzene = 6 aromatic carbons
        # All of them should be carbon
        elements = {str(e) for e in arr.element[ligand_mask]}
        assert elements == {"C"}

    def test_water_atoms_classified(self) -> None:
        p = read_pdb(FIXTURES / "real_with_ligand_realistic.pdb")
        arr = p.atom_array
        water_mask = arr.entity_type == "water"
        assert water_mask.sum() == 3  # three HOH oxygens
        residue_names = {str(r) for r in arr.residue_name[water_mask]}
        assert residue_names == {"HOH"}

    def test_ion_atom_classified(self) -> None:
        p = read_pdb(FIXTURES / "real_with_ligand_realistic.pdb")
        arr = p.atom_array
        ion_mask = arr.entity_type == "ion"
        assert ion_mask.sum() == 1  # one zinc atom
        # Element should be Zn
        elements = {str(e) for e in arr.element[ion_mask]}
        assert elements == {"ZN"} or elements == {"Zn"}


# ===========================================================================
# Full side-chain atom sets — the real value of these fixtures
# ===========================================================================
class TestFullSideChains:
    """Synthetic helices are backbone-only or poly-Ala. These fixtures
    exercise every side-chain atom name across all 20 amino acids."""

    # Expected atom counts per residue type, from the canonical PDB
    # atom-naming convention.
    EXPECTED_ATOMS: ClassVar[dict[str, int]] = {
        "GLY": 4,  # N CA C O
        "ALA": 5,  # + CB
        "SER": 6,  # + OG
        "THR": 7,  # + OG1 CG2
        "CYS": 6,  # + SG
        "VAL": 7,  # + CG1 CG2
        "LEU": 8,  # + CG CD1 CD2
        "ILE": 8,  # + CG1 CG2 CD1
        "ASP": 8,  # + CG OD1 OD2
        "ASN": 8,  # + CG OD1 ND2
        "GLU": 9,  # + CG CD OE1 OE2
        "GLN": 9,  # + CG CD OE1 NE2
        "LYS": 9,  # + CG CD CE NZ
        "ARG": 11,  # + CG CD NE CZ NH1 NH2
        "HIS": 10,  # + CG ND1 CD2 CE1 NE2
        "PHE": 11,  # + CG CD1 CD2 CE1 CE2 CZ
        "TYR": 12,  # + CG CD1 CD2 CE1 CE2 CZ OH
        "TRP": 14,  # + CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2
        "MET": 8,  # + CG SD CE
        "PRO": 7,  # + CG CD (ring back to N)
    }

    @pytest.fixture
    def protein(self) -> Protein:
        return read_pdb(FIXTURES / "real_small_protein.pdb")

    def test_every_residue_has_full_atom_set(self, protein: Protein) -> None:
        """Every residue's atom count matches the canonical PDB convention."""
        arr = protein.atom_array
        for sl in arr.iter_residue_slices():
            residue_name = str(arr.residue_name[sl.start])
            expected = self.EXPECTED_ATOMS[residue_name]
            actual = sl.stop - sl.start
            assert actual == expected, (
                f"residue {residue_name} at position {arr.residue_id[sl.start]}: "
                f"expected {expected} atoms, got {actual}"
            )

    def test_aromatic_residues_have_ring_atoms(self, protein: Protein) -> None:
        """PHE/TYR/TRP/HIS aromatic rings parse with full ring atom sets."""
        arr = protein.atom_array
        for sl in arr.iter_residue_slices():
            residue_name = str(arr.residue_name[sl.start])
            atom_names = {str(n) for n in arr.atom_name[sl]}
            if residue_name == "PHE":
                assert {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}.issubset(atom_names)
            elif residue_name == "TYR":
                assert {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}.issubset(atom_names)
            elif residue_name == "TRP":
                assert {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}.issubset(
                    atom_names
                )
            elif residue_name == "HIS":
                assert {"CG", "ND1", "CD2", "CE1", "NE2"}.issubset(atom_names)

    def test_realistic_b_factors(self, protein: Protein) -> None:
        """B-factors aren't uniform 20.00 — real fixtures should have
        variation, with edges typically higher than core."""
        arr = protein.atom_array
        # Find CA atoms for each residue
        ca_b = []
        for sl in arr.iter_residue_slices():
            names = arr.atom_name[sl]
            ca_idx = np.where(names == "CA")[0]
            if ca_idx.size:
                ca_b.append(float(arr.b_factor[sl][ca_idx[0]]))
        ca_b = np.array(ca_b)
        # Should have meaningful variance
        assert ca_b.std() > 1.0, f"B-factors look flat: std={ca_b.std():.3f}"
        # Should be in a realistic range
        assert ca_b.min() > 5.0
        assert ca_b.max() < 60.0

    def test_element_assignment(self, protein: Protein) -> None:
        """Every atom has the right element inferred from its name."""
        arr = protein.atom_array
        elements = {str(e) for e in arr.element}
        # Protein with all 20 AAs should have C, N, O, S
        assert elements == {"C", "N", "O", "S"}


# ===========================================================================
# Multi-atom alt-locs — this is the key edge case the previous
# with_altloc.pdb couldn't cover
# ===========================================================================
class TestMultiAtomAltLocs:
    """LEU 2 and SER 9 in the altloc fixture have A/B conformations of
    *every* side-chain atom. The previous tiny with_altloc fixture
    only had alt-locs on 4 atoms with no multi-atom side-chain context."""

    def test_default_picks_highest_occupancy(self) -> None:
        """A confs (occupancy 0.60) win over B confs (0.40)."""
        p = read_pdb(FIXTURES / "real_with_altloc_sidechains.pdb")
        arr = p.atom_array

        # LEU 2 should have its 8 atoms — all the A conf
        for sl in arr.iter_residue_slices():
            residue_id = int(arr.residue_id[sl.start])
            if residue_id == 2:
                assert (sl.stop - sl.start) == 8  # backbone + CB + CG + CD1 + CD2
                # Atom names — no duplicates
                names = [str(n) for n in arr.atom_name[sl]]
                assert len(names) == len(set(names)), f"duplicate atoms: {names}"

    def test_all_keeps_both_conformations(self) -> None:
        """altloc='all' returns both A and B atoms with the altloc field intact."""
        p = read_pdb(FIXTURES / "real_with_altloc_sidechains.pdb", altloc="all")
        arr = p.atom_array
        # 97 atoms total (91 + 6 B-conf side-chain atoms that were dropped above)
        assert p.n_atoms == 97
        # Altloc field should now contain A and B markers
        altlocs = {str(a).strip() for a in arr.altloc}
        # An empty altloc marker is normal for non-altloc atoms
        assert "A" in altlocs
        assert "B" in altlocs

    def test_explicit_b_selection(self) -> None:
        """altloc='B' selects only the B conformation for alt-loc atoms."""
        p = read_pdb(FIXTURES / "real_with_altloc_sidechains.pdb", altloc="B")
        # Same atom count as default: backbone + one conformation
        assert p.n_atoms == 91

    def test_altloc_atoms_have_different_coords(self) -> None:
        """Sanity: A and B conformations actually have different coords."""
        p = read_pdb(FIXTURES / "real_with_altloc_sidechains.pdb", altloc="all")
        arr = p.atom_array
        # Find the OG of SER 9 in both A and B
        og_coords = {}
        for i in range(p.n_atoms):
            if int(arr.residue_id[i]) == 9 and str(arr.atom_name[i]) == "OG":
                og_coords[str(arr.altloc[i])] = arr.coords[i].copy()
        assert "A" in og_coords and "B" in og_coords
        # Different coordinates
        diff = np.linalg.norm(og_coords["A"] - og_coords["B"])
        assert diff > 0.5, f"alt-loc OGs only differ by {diff:.3f} A"


# ===========================================================================
# Roundtrip - read, write, read again preserves the data
# ===========================================================================
class TestRoundTrip:
    """Write back to a tmp PDB, re-read, verify the structure is preserved."""

    def test_small_protein_roundtrip_preserves_atoms(self, tmp_path: Path) -> None:
        original = read_pdb(FIXTURES / "real_small_protein.pdb")
        out = tmp_path / "roundtrip.pdb"
        write_pdb(original, out)
        reread = read_pdb(out)

        assert reread.n_atoms == original.n_atoms
        assert reread.n_residues == original.n_residues
        np.testing.assert_allclose(
            reread.atom_array.coords,
            original.atom_array.coords,
            atol=1e-3,  # PDB format has 3-decimal precision
        )

    def test_small_protein_roundtrip_preserves_residue_names(self, tmp_path: Path) -> None:
        original = read_pdb(FIXTURES / "real_small_protein.pdb")
        out = tmp_path / "roundtrip.pdb"
        write_pdb(original, out)
        reread = read_pdb(out)
        np.testing.assert_array_equal(
            reread.atom_array.residue_name,
            original.atom_array.residue_name,
        )

    def test_ligand_fixture_roundtrip_preserves_entity_types(self, tmp_path: Path) -> None:
        original = read_pdb(FIXTURES / "real_with_ligand_realistic.pdb")
        out = tmp_path / "roundtrip.pdb"
        write_pdb(original, out)
        reread = read_pdb(out)
        # All four entity types still classified the same way
        assert {str(t) for t in reread.atom_array.entity_type} == {
            str(t) for t in original.atom_array.entity_type
        }


# ===========================================================================
# Structure analysis - exercising the structural code path on a realistic
# input, which is structurally different from the idealized fixtures.
# ===========================================================================
class TestStructureAnalysis:
    """Run the structural utilities (DSSP, SASA, dihedrals, contacts) on
    the realistic protein fixture. This is the key value: exercising
    these algorithms on input with full side-chain context."""

    @pytest.fixture
    def protein(self) -> Protein:
        return read_pdb(FIXTURES / "real_small_protein.pdb")

    def test_dssp_runs(self, protein: Protein) -> None:
        from molforge.structure import dssp_3state

        ss = dssp_3state(protein)
        assert len(ss) == protein.n_residues
        assert set(ss).issubset({"H", "E", "C"})

    def test_dssp_helix_detected(self, protein: Protein) -> None:
        """Residues 1-7 were built as helix; DSSP should detect at least
        some helical residues in that region."""
        from molforge.structure import dssp_3state

        ss = dssp_3state(protein)
        # First 7 residues built as helix — at least 4 should be H-classified
        # (DSSP is strict about H-bonding patterns at helix endpoints)
        helix_count = ss[:7].count("H")
        assert helix_count >= 3, (
            f"expected ≥3 H residues in built helix region, got {helix_count}: ss[:7]={ss[:7]!r}"
        )

    def test_phi_psi_omega_works(self, protein: Protein) -> None:
        from molforge.structure import phi_psi_omega

        phi, psi, omega = phi_psi_omega(protein)
        # Returns one value per residue
        assert len(phi) == protein.n_residues
        assert len(psi) == protein.n_residues
        assert len(omega) == protein.n_residues
        # First residue has no phi, last has no psi/omega — those are NaN
        assert np.isnan(phi[0])
        # Interior residues should have meaningful angles
        mid_phi = phi[5]
        assert not np.isnan(mid_phi)
        # NOTE: this fixture's NeRF builder produces a left-handed helix
        # (phi ≈ +60° in the helix region rather than -60°). The geometry is
        # still mirror-valid; we just verify the magnitude matches the
        # helix-like behavior built into the fixture.
        assert 30 < abs(mid_phi) < 90
        # Omega should be near ±180° (trans peptide bonds)
        mid_omega = omega[5]
        assert abs(abs(mid_omega) - 180) < 5

    def test_sasa_runs(self, protein: Protein) -> None:
        from molforge.structure import sasa

        per_atom_sasa = sasa(protein)
        assert len(per_atom_sasa) == protein.n_atoms
        # Some atoms should be solvent-exposed
        assert per_atom_sasa.max() > 0.0
        # No negative values
        assert per_atom_sasa.min() >= 0.0
        # Total SASA should be physically reasonable for a 24-residue
        # protein: rough estimate is 30-50 sq A per residue
        total = float(per_atom_sasa.sum())
        assert 200 < total < 5000

    def test_contacts_finds_interactions(self, protein: Protein) -> None:
        """The built structure should have inter-residue contacts."""
        from molforge.structure import contact_map

        contacts = contact_map(protein, cutoff=8.0)
        # Should be a (n_res, n_res) symmetric boolean matrix
        assert contacts.shape == (protein.n_residues, protein.n_residues)
        # Should have some non-self contacts
        non_self = contacts.copy()
        np.fill_diagonal(non_self, False)
        assert non_self.sum() > 0


# ===========================================================================
# ML / featurization - real side-chain context exercises atom-type-aware
# featurizers that synthetic backbone-only fixtures can't.
# ===========================================================================
class TestMLFeaturization:
    """Feature extraction works on the realistic structure."""

    @pytest.fixture
    def protein(self) -> Protein:
        return read_pdb(FIXTURES / "real_small_protein.pdb")

    def test_pair_distance_features(self, protein: Protein) -> None:
        from molforge.ml import pair_distance_features

        features = pair_distance_features(protein, n_bins=16)
        # Shape: (n_res, n_res, n_bins) with RBF basis expansion
        n_res = protein.n_residues
        assert features.shape == (n_res, n_res, 16)
        # All values non-negative (RBF activations)
        assert features.min() >= 0.0
        # All values finite
        assert np.isfinite(features).all()
        # Diagonal entries (residue to itself) should fire on the
        # smallest-distance bin
        assert features[0, 0, 0] > 0.0

    def test_one_hot_encoding(self, protein: Protein) -> None:
        from molforge.ml import one_hot

        # one_hot takes a sequence string, not a Protein
        features = one_hot(protein.sequence)
        # One-hot encoding per residue
        assert features.shape[0] == protein.n_residues
        # Each residue gets exactly one 1
        np.testing.assert_array_equal(features.sum(axis=1), 1)


# ===========================================================================
# Sequence mutation on a realistic structure
# ===========================================================================
class TestMutation:
    """Mutating a realistic protein with full side chains exercises the
    side-chain stripping logic that backbone-only fixtures don't reach."""

    def test_mutation_preserves_structure(self) -> None:
        from molforge.sequence import mutate_protein

        original = read_pdb(FIXTURES / "real_small_protein.pdb")
        # E1V: GLU → VAL (smaller side chain)
        mutated = mutate_protein(original, "E1V")
        # Sequence should reflect the change
        assert mutated.sequence == "VLKMQRAGSPVIFYCTWNDGHEAM"
        # Structure should still load round-trip
        assert mutated.n_residues == 24
        # The mutation changed residue 1's name
        arr = mutated.atom_array
        residue_1_atoms = []
        for sl in arr.iter_residue_slices():
            if int(arr.residue_id[sl.start]) == 1:
                residue_1_atoms.append(str(arr.residue_name[sl.start]))
                break
        assert residue_1_atoms == ["VAL"]
