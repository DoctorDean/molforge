"""Tests for molforge.core.metadata_keys.

These verify that the documented metadata vocabulary is internally
consistent and that molforge's own producers (parsers, folding
wrappers) only write keys that are in the documented set — i.e. the
contract and the implementation don't drift apart.
"""

from __future__ import annotations

from molforge.core import ProteinMetadata
from molforge.core import metadata_keys as mk


class TestVocabularyConsistency:
    def test_documented_keys_is_frozenset(self) -> None:
        assert isinstance(mk.DOCUMENTED_KEYS, frozenset)

    def test_documented_keys_nonempty(self) -> None:
        assert len(mk.DOCUMENTED_KEYS) > 0

    def test_every_string_constant_is_in_documented_keys(self) -> None:
        """Each public string constant must appear in DOCUMENTED_KEYS."""
        for name in mk.__all__:
            if name in ("ProteinMetadata", "DOCUMENTED_KEYS"):
                continue
            value = getattr(mk, name)
            assert isinstance(value, str), f"{name} should be a str constant"
            assert value in mk.DOCUMENTED_KEYS, (
                f"constant {name}={value!r} missing from DOCUMENTED_KEYS"
            )

    def test_documented_keys_count_matches_constants(self) -> None:
        """DOCUMENTED_KEYS should have exactly one entry per string constant."""
        n_constants = sum(
            1 for name in mk.__all__
            if name not in ("ProteinMetadata", "DOCUMENTED_KEYS")
        )
        assert len(mk.DOCUMENTED_KEYS) == n_constants

    def test_typeddict_keys_match_documented_keys(self) -> None:
        """ProteinMetadata's annotated keys must equal DOCUMENTED_KEYS."""
        typeddict_keys = set(ProteinMetadata.__annotations__.keys())
        assert typeddict_keys == set(mk.DOCUMENTED_KEYS)

    def test_constant_values_are_their_own_names_lowercased(self) -> None:
        """Sanity: e.g. MEAN_CONFIDENCE constant holds 'mean_confidence'."""
        assert mk.MEAN_CONFIDENCE == "mean_confidence"
        assert mk.PDB_ID == "pdb_id"
        assert mk.ENGINE == "engine"
        assert mk.PAE_INTER == "pae_inter"


class TestProteinMetadataTypedDict:
    def test_is_total_false(self) -> None:
        """Every key must be optional — a Protein may carry any subset."""
        # TypedDict with total=False has empty __required_keys__.
        assert ProteinMetadata.__total__ is False

    def test_accepts_partial_dict_at_runtime(self) -> None:
        """At runtime ProteinMetadata is just a dict; partial is fine."""
        # This is a typing aid; runtime behavior is plain dict.
        meta: ProteinMetadata = {"engine": "ESMFold"}
        assert meta["engine"] == "ESMFold"

    def test_accepts_empty_dict(self) -> None:
        meta: ProteinMetadata = {}
        assert meta == {}


class TestParsersUseDocumentedKeys:
    """Parsers should only write keys from the documented vocabulary."""

    def test_pdb_parser_keys_are_documented(self) -> None:
        from molforge.io import read_pdb_string

        pdb = (
            "HEADER    HYDROLASE                               01-JAN-00   1ABC\n"
            "TITLE     A TEST STRUCTURE\n"
            "EXPDTA    X-RAY DIFFRACTION\n"
            "REMARK   2 RESOLUTION.    1.50 ANGSTROMS.\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
            "END\n"
        )
        protein = read_pdb_string(pdb)
        for key in protein.metadata:
            assert key in mk.DOCUMENTED_KEYS, (
                f"PDB parser wrote undocumented metadata key {key!r}"
            )

    def test_pdb_parser_populates_expected_keys(self) -> None:
        from molforge.io import read_pdb_string

        pdb = (
            "HEADER    HYDROLASE                               01-JAN-00   1ABC\n"
            "TITLE     A TEST STRUCTURE\n"
            "EXPDTA    X-RAY DIFFRACTION\n"
            "REMARK   2 RESOLUTION.    1.50 ANGSTROMS.\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
            "END\n"
        )
        protein = read_pdb_string(pdb)
        assert protein.metadata[mk.PDB_ID] == "1ABC"
        assert protein.metadata[mk.CLASSIFICATION] == "HYDROLASE"
        assert protein.metadata[mk.TITLE] == "A TEST STRUCTURE"
        assert protein.metadata[mk.EXPERIMENTAL_METHOD] == "X-RAY DIFFRACTION"
        assert protein.metadata[mk.RESOLUTION] == 1.50

    def test_cif_parser_keys_are_documented(self) -> None:
        from molforge.io import read_cif_string

        cif = (
            "data_test\n"
            "_entry.id  1ABC\n"
            "_struct.title  'A test structure'\n"
            "_exptl.method  'X-RAY DIFFRACTION'\n"
            "#\n"
            "loop_\n"
            "_atom_site.group_PDB\n"
            "_atom_site.id\n"
            "_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n"
            "_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n"
            "_atom_site.label_seq_id\n"
            "_atom_site.Cartn_x\n"
            "_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n"
            "_atom_site.occupancy\n"
            "_atom_site.B_iso_or_equiv\n"
            "ATOM 1 C CA ALA A 1 0.000 0.000 0.000 1.00 20.00\n"
            "#\n"
        )
        protein = read_cif_string(cif)
        for key in protein.metadata:
            assert key in mk.DOCUMENTED_KEYS, (
                f"CIF parser wrote undocumented metadata key {key!r}"
            )


class TestUniformConfidenceKeysAcrossEngines:
    """Every folding engine + load_alphafold must agree on the uniform keys."""

    def test_folding_wrappers_emit_uniform_keys(self) -> None:
        """ESMFold, AlphaFold, Boltz, RoseTTAFold post-processing all set
        the four uniform keys."""
        from molforge.wrappers.folding import AlphaFold, Boltz, ESMFold, RoseTTAFold

        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N\n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C\n"
            "END\n"
        )
        cif = (
            "data_query\n#\nloop_\n"
            "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
            "_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n"
            "ATOM 1 N N ALA A 1 0.000 0.000 0.000 1.00 80.00\n"
            "ATOM 2 C CA ALA A 1 1.000 0.000 0.000 1.00 80.00\n#\n"
        )

        uniform = {
            mk.ENGINE,
            mk.SOURCE_SEQUENCE,
            mk.CONFIDENCE_PER_RESIDUE,
            mk.CONFIDENCE_PER_ATOM,
            mk.MEAN_CONFIDENCE,
        }

        esm_meta = ESMFold()._pdb_to_protein(pdb, sequence="A").metadata
        af_meta = AlphaFold()._pdb_to_protein(pdb, sequence="A").metadata
        boltz_meta = Boltz()._parse_outputs(
            cif_text=cif, confidence_json={}, sequence="A"
        ).metadata
        rf_meta = RoseTTAFold()._parse_outputs(
            pdb_text=pdb, confidence={}, sequence="A"
        ).metadata

        for engine_name, meta in [
            ("ESMFold", esm_meta),
            ("AlphaFold", af_meta),
            ("Boltz", boltz_meta),
            ("RoseTTAFold", rf_meta),
        ]:
            missing = uniform - set(meta)
            assert not missing, f"{engine_name} missing uniform keys: {missing}"

    def test_load_alphafold_emits_uniform_keys(self, tmp_path) -> None:
        """load_alphafold must populate the uniform keys, not just the
        legacy plddt* keys — this was the key-mismatch bug the audit
        flagged."""
        from molforge.io import load_alphafold

        pdb_path = tmp_path / "af.pdb"
        pdb_path.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 92.50           N\n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 92.50           C\n"
            "END\n"
        )
        protein = load_alphafold(pdb_path)

        # Uniform keys present.
        assert protein.metadata[mk.ENGINE] == "AlphaFold"
        assert mk.CONFIDENCE_PER_ATOM in protein.metadata
        assert mk.CONFIDENCE_PER_RESIDUE in protein.metadata
        assert protein.metadata[mk.MEAN_CONFIDENCE] == 92.5

        # Legacy keys still present for backward compatibility.
        assert mk.PLDDT in protein.metadata
        assert mk.PLDDT_PER_RESIDUE in protein.metadata
        assert protein.metadata[mk.MEAN_PLDDT] == 92.5
        assert protein.metadata[mk.SOURCE] == "alphafold"

    def test_load_alphafold_uniform_and_legacy_agree(self, tmp_path) -> None:
        """The uniform and legacy keys must carry identical values."""
        import numpy as np

        from molforge.io import load_alphafold

        pdb_path = tmp_path / "af.pdb"
        pdb_path.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 88.00           N\n"
            "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 70.00           C\n"
            "END\n"
        )
        protein = load_alphafold(pdb_path)
        m = protein.metadata

        np.testing.assert_array_equal(m[mk.PLDDT], m[mk.CONFIDENCE_PER_ATOM])
        np.testing.assert_array_equal(
            m[mk.PLDDT_PER_RESIDUE], m[mk.CONFIDENCE_PER_RESIDUE]
        )
        assert m[mk.MEAN_PLDDT] == m[mk.MEAN_CONFIDENCE]
