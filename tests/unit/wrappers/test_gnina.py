"""Tests for the gnina docking wrapper.

The gnina binary is rarely in CI (no pip package; users install via
system package or download a release), so the strategy mirrors
test_vina.py / test_amber.py / test_rosettafold.py: end-to-end
tests skip when the binary is missing, and the SDF parser and the
command-line builder are tested in isolation against realistic
synthetic outputs.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.docking import DockingEngineNotInstalledError, DockingResult
from molforge.wrappers.docking import Gnina

# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


# A realistic 2-pose gnina SDF. Constructed from the format
# documented in gnina/issues/294 and gnina's own example outputs.
# Per gnina spec: each molecule is followed by SDF tag fields named
# <minimizedAffinity> / <CNNscore> / <CNNaffinity> (and optionally
# <CNNvariance>), terminated by $$$$.
_REAL_GNINA_SDF = textwrap.dedent(
    """\
    pose_1
         RDKit          3D

      1  0  0  0  0  0  0  0  0  0999 V2000
        0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    M  END
    > <minimizedAffinity>
    -8.4

    > <CNNscore>
    0.83

    > <CNNaffinity>
    6.5

    > <CNNvariance>
    0.12

    $$$$
    pose_2
         RDKit          3D

      1  0  0  0  0  0  0  0  0  0999 V2000
        1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    M  END
    > <minimizedAffinity>
    -7.1

    > <CNNscore>
    0.65

    > <CNNaffinity>
    5.8

    > <CNNvariance>
    0.18

    $$$$
    """
)


# A 1-pose SDF in cnn_scoring="none" mode: only minimizedAffinity,
# no CNN keys. Tests that the parser gracefully tolerates missing
# CNN fields rather than crashing.
_VINA_ONLY_SDF = textwrap.dedent(
    """\
    pose_1
         RDKit          3D

      1  0  0  0  0  0  0  0  0  0999 V2000
        0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    M  END
    > <minimizedAffinity>
    -8.4

    $$$$
    """
)


def _tiny_protein() -> Protein:
    """Minimal valid Protein for tests that don't need real coords."""
    return Protein(AtomArray(0), name="test_protein")


# ---------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        engine = Gnina()
        assert engine.gnina_executable == "gnina"
        assert engine.cnn_scoring == "rescore"  # gnina default
        assert engine.cnn is None
        assert engine.sort_order == "CNNscore"  # gnina default
        assert engine.scoring == "vina"
        assert engine.seed is None
        assert engine.cpu == 0
        assert engine.timeout == 600.0
        assert engine.verbose is False

    def test_custom_path_and_options(self) -> None:
        engine = Gnina(
            gnina_executable="/opt/gnina/bin/gnina",
            cnn_scoring="refinement",
            cnn="dense_default2018",
            sort_order="CNNaffinity",
            scoring="vinardo",
            seed=42,
            cpu=4,
        )
        assert engine.gnina_executable == "/opt/gnina/bin/gnina"
        assert engine.cnn_scoring == "refinement"
        assert engine.cnn == "dense_default2018"
        assert engine.sort_order == "CNNaffinity"
        assert engine.scoring == "vinardo"
        assert engine.seed == 42
        assert engine.cpu == 4

    def test_invalid_cnn_scoring_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown cnn_scoring"):
            Gnina(cnn_scoring="bogus")

    def test_invalid_sort_order_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown sort_order"):
            Gnina(sort_order="nonsense")

    def test_invalid_scoring_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scoring"):
            Gnina(scoring="autodock")

    def test_negative_cpu_raises(self) -> None:
        with pytest.raises(ValueError, match="cpu must be >= 0"):
            Gnina(cpu=-1)

    def test_non_positive_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be > 0"):
            Gnina(timeout=0.0)
        with pytest.raises(ValueError, match="timeout must be > 0"):
            Gnina(timeout=-10.0)

    def test_construction_does_not_resolve_binary(self) -> None:
        """Construction is lazy: an instance with a nonexistent
        binary path is creatable. The resolution happens inside
        dock()."""
        engine = Gnina(gnina_executable="/nonexistent/gnina")
        assert engine.gnina_executable == "/nonexistent/gnina"


# ---------------------------------------------------------------------
# Missing-binary error path
# ---------------------------------------------------------------------


class TestMissingBinaryError:
    def test_friendly_error_when_gnina_missing(self) -> None:
        with pytest.raises(DockingEngineNotInstalledError) as exc:
            Gnina(gnina_executable="/nonexistent/gnina").dock(
                _tiny_protein(),
                _tiny_protein(),
                center=(0.0, 0.0, 0.0),
            )
        msg = str(exc.value)
        # Tells the user what couldn't be found and how to fix it.
        assert "gnina executable" in msg
        assert any(s in msg for s in ("brew", "github", "releases"))
        # Mentions Vina as the no-CNN fallback so a user can pivot
        # without reinstalling.
        assert "Vina" in msg


# ---------------------------------------------------------------------
# Command-line builder
# ---------------------------------------------------------------------


class TestCommandBuilder:
    def test_required_flags_present(self) -> None:
        engine = Gnina()
        cmd = engine._build_command(
            gnina="/usr/bin/gnina",
            receptor_path=Path("/tmp/r.pdb"),
            ligand_path=Path("/tmp/l.sdf"),
            out_sdf=Path("/tmp/out.sdf"),
            center=(10.0, 5.0, -2.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=8,
            n_poses=9,
            min_rmsd=1.0,
        )
        # The flags gnina insists on.
        assert "--receptor" in cmd
        assert "/tmp/r.pdb" in cmd
        assert "--ligand" in cmd
        assert "/tmp/l.sdf" in cmd
        assert "--out" in cmd
        assert "/tmp/out.sdf" in cmd
        # Center, box, exhaustiveness.
        assert "--center_x" in cmd
        assert "10.0" in cmd
        assert "--size_y" in cmd
        assert "--exhaustiveness" in cmd
        assert "8" in cmd
        assert "--num_modes" in cmd
        assert "9" in cmd
        # CNN config from defaults.
        assert "--cnn_scoring" in cmd
        assert "rescore" in cmd
        assert "--pose_sort_order" in cmd
        assert "CNNscore" in cmd
        # The first element is the binary path.
        assert cmd[0] == "/usr/bin/gnina"

    def test_cnn_model_flag_included_when_set(self) -> None:
        engine = Gnina(cnn="dense_default2018")
        cmd = engine._build_command(
            gnina="/usr/bin/gnina",
            receptor_path=Path("/tmp/r.pdb"),
            ligand_path=Path("/tmp/l.sdf"),
            out_sdf=Path("/tmp/o.sdf"),
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=8,
            n_poses=9,
            min_rmsd=1.0,
        )
        idx = cmd.index("--cnn")
        assert cmd[idx + 1] == "dense_default2018"

    def test_cnn_flag_omitted_when_unset(self) -> None:
        """Default cnn=None: the wrapper should not pass --cnn at
        all, letting gnina pick its own default model."""
        engine = Gnina()  # cnn=None
        cmd = engine._build_command(
            gnina="/usr/bin/gnina",
            receptor_path=Path("/tmp/r.pdb"),
            ligand_path=Path("/tmp/l.sdf"),
            out_sdf=Path("/tmp/o.sdf"),
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=8,
            n_poses=9,
            min_rmsd=1.0,
        )
        assert "--cnn" not in cmd

    def test_seed_flag_only_when_provided(self) -> None:
        engine = Gnina(seed=42)
        cmd = engine._build_command(
            gnina="/usr/bin/gnina",
            receptor_path=Path("/tmp/r.pdb"),
            ligand_path=Path("/tmp/l.sdf"),
            out_sdf=Path("/tmp/o.sdf"),
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=8,
            n_poses=9,
            min_rmsd=1.0,
        )
        assert "--seed" in cmd
        assert "42" in cmd

        engine2 = Gnina()  # seed=None
        cmd2 = engine2._build_command(
            gnina="/usr/bin/gnina",
            receptor_path=Path("/tmp/r.pdb"),
            ligand_path=Path("/tmp/l.sdf"),
            out_sdf=Path("/tmp/o.sdf"),
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=8,
            n_poses=9,
            min_rmsd=1.0,
        )
        assert "--seed" not in cmd2

    def test_cpu_flag_only_when_positive(self) -> None:
        """cpu=0 means "let gnina decide" — don't pass --cpu."""
        engine = Gnina(cpu=0)
        cmd = engine._build_command(
            gnina="/usr/bin/gnina",
            receptor_path=Path("/tmp/r.pdb"),
            ligand_path=Path("/tmp/l.sdf"),
            out_sdf=Path("/tmp/o.sdf"),
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=8,
            n_poses=9,
            min_rmsd=1.0,
        )
        assert "--cpu" not in cmd

        engine = Gnina(cpu=4)
        cmd = engine._build_command(
            gnina="/usr/bin/gnina",
            receptor_path=Path("/tmp/r.pdb"),
            ligand_path=Path("/tmp/l.sdf"),
            out_sdf=Path("/tmp/o.sdf"),
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=8,
            n_poses=9,
            min_rmsd=1.0,
        )
        idx = cmd.index("--cpu")
        assert cmd[idx + 1] == "4"


# ---------------------------------------------------------------------
# SDF REMARK extractor
# ---------------------------------------------------------------------


class TestExtractRemarks:
    def test_two_pose_extraction(self) -> None:
        remarks = Gnina._extract_remarks(_REAL_GNINA_SDF)
        assert len(remarks) == 2
        assert remarks[0] == {
            "minimizedAffinity": -8.4,
            "CNNscore": 0.83,
            "CNNaffinity": 6.5,
            "CNNvariance": 0.12,
        }
        assert remarks[1] == {
            "minimizedAffinity": -7.1,
            "CNNscore": 0.65,
            "CNNaffinity": 5.8,
            "CNNvariance": 0.18,
        }

    def test_vina_only_mode_has_no_cnn_keys(self) -> None:
        """cnn_scoring='none' produces SDF with only minimizedAffinity.
        The extractor must not invent missing CNN values."""
        remarks = Gnina._extract_remarks(_VINA_ONLY_SDF)
        assert len(remarks) == 1
        assert remarks[0] == {"minimizedAffinity": -8.4}
        assert "CNNscore" not in remarks[0]
        assert "CNNaffinity" not in remarks[0]

    def test_empty_sdf_returns_empty_list(self) -> None:
        assert Gnina._extract_remarks("") == []

    def test_malformed_value_is_skipped(self) -> None:
        """A garbled numeric value shouldn't crash the parser —
        gnina is well-behaved, but truncated output during a
        crash is a real scenario."""
        garbled = textwrap.dedent(
            """\
            pose_1
                 RDKit          3D

              1  0  0  0  0  0  0  0  0  0999 V2000
                0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
            M  END
            > <minimizedAffinity>
            not-a-float

            > <CNNscore>
            0.83

            $$$$
            """
        )
        remarks = Gnina._extract_remarks(garbled)
        assert remarks[0] == {"CNNscore": 0.83}
        # minimizedAffinity wasn't extracted — graceful skip.
        assert "minimizedAffinity" not in remarks[0]


# ---------------------------------------------------------------------
# Full output parsing
# ---------------------------------------------------------------------


class TestParseSdfOutput:
    """Drive _parse_sdf_output with synthetic input and verify the
    DockingResult shape, score selection, and Provenance attachment."""

    def test_two_poses_with_default_sort_order(self) -> None:
        engine = Gnina()  # sort_order = "CNNscore"
        result = engine._parse_sdf_output(
            _REAL_GNINA_SDF,
            receptor=None,
            provenance_parameters={"cnn_scoring": "rescore"},
            provenance_inputs={"receptor": "rec", "ligand": "lig"},
            provenance_parent=None,
        )
        assert isinstance(result, DockingResult)
        assert len(result.poses) == 2
        # poses[0].score is CNNscore (0.83), since sort_order="CNNscore".
        assert result.poses[0].score == 0.83
        assert result.poses[1].score == 0.65
        # Ranks set correctly (gnina has already sorted).
        assert result.poses[0].rank == 0
        assert result.poses[1].rank == 1

    def test_sort_order_energy_uses_minimized_affinity(self) -> None:
        """sort_order='Energy' selects minimizedAffinity as the
        per-pose .score — bridges the gnina-internal naming
        ('Energy' as sort order, 'minimizedAffinity' as SDF tag)."""
        engine = Gnina(sort_order="Energy")
        result = engine._parse_sdf_output(
            _REAL_GNINA_SDF,
            receptor=None,
            provenance_parameters={},
            provenance_inputs={},
            provenance_parent=None,
        )
        assert result.poses[0].score == -8.4
        assert result.poses[1].score == -7.1

    def test_sort_order_cnn_affinity(self) -> None:
        engine = Gnina(sort_order="CNNaffinity")
        result = engine._parse_sdf_output(
            _REAL_GNINA_SDF,
            receptor=None,
            provenance_parameters={},
            provenance_inputs={},
            provenance_parent=None,
        )
        assert result.poses[0].score == 6.5
        assert result.poses[1].score == 5.8

    def test_per_pose_metadata_carries_all_three_scores(self) -> None:
        """Every pose's metadata exposes all of vina_affinity,
        cnn_score, and cnn_affinity — not just the one used for
        ranking. Lets users post-filter on a different metric."""
        result = Gnina()._parse_sdf_output(
            _REAL_GNINA_SDF,
            receptor=None,
            provenance_parameters={},
            provenance_inputs={},
            provenance_parent=None,
        )
        top = result.poses[0]
        assert top.metadata["vina_affinity"] == -8.4
        assert top.metadata["cnn_score"] == 0.83
        assert top.metadata["cnn_affinity"] == 6.5
        assert top.metadata["cnn_variance"] == 0.12

    def test_provenance_attached_at_result_level(self) -> None:
        result = Gnina()._parse_sdf_output(
            _REAL_GNINA_SDF,
            receptor=None,
            provenance_parameters={
                "cnn_scoring": "rescore",
                "center": [0.0, 0.0, 0.0],
            },
            provenance_inputs={"receptor": "rec", "ligand": "lig"},
            provenance_parent=None,
        )
        prov = result.metadata[mk.PROVENANCE]
        assert isinstance(prov, Provenance)
        assert prov.engine == "Gnina"
        assert prov.parameters["cnn_scoring"] == "rescore"
        assert "receptor" in prov.inputs
        assert "ligand" in prov.inputs

    def test_provenance_chains_through_upstream(self) -> None:
        """A receptor with its own Provenance (e.g. from ESMFold)
        becomes the parent — the headline cross-engine chain works
        end-to-end."""
        upstream = Provenance.from_engine(
            engine="ESMFold",
            parameters={"model_name": "esmfold_v1"},
            inputs={"sequence": "AAA"},
        )
        result = Gnina()._parse_sdf_output(
            _REAL_GNINA_SDF,
            receptor=None,
            provenance_parameters={},
            provenance_inputs={"receptor": "rec", "ligand": "lig"},
            provenance_parent=upstream,
        )
        chain = result.metadata[mk.PROVENANCE].chain()
        engines = [s.engine for s in chain]
        assert engines == ["ESMFold", "Gnina"]

    def test_vina_only_mode_yields_zero_score_for_cnn_sort(self) -> None:
        """If a user runs cnn_scoring='none' with the default
        sort_order='CNNscore', the SDF has no CNNscore tag, so the
        score is None internally — coerced to 0.0 rather than
        crashing. The vina_affinity field is still populated."""
        engine = Gnina(cnn_scoring="none")  # sort_order still default
        result = engine._parse_sdf_output(
            _VINA_ONLY_SDF,
            receptor=None,
            provenance_parameters={},
            provenance_inputs={},
            provenance_parent=None,
        )
        # score = 0.0 (CNNscore missing).
        assert result.poses[0].score == 0.0
        # The Vina-style score is still recoverable.
        assert result.poses[0].metadata["vina_affinity"] == -8.4

    def test_empty_sdf_raises(self) -> None:
        with pytest.raises(RuntimeError, match="no parseable molecules"):
            Gnina()._parse_sdf_output(
                "",
                receptor=None,
                provenance_parameters={},
                provenance_inputs={},
                provenance_parent=None,
            )


# ---------------------------------------------------------------------
# Subprocess seam (mocked)
# ---------------------------------------------------------------------


class TestSubprocessSeam:
    """Drive the dock() pipeline with subprocess.run mocked. The
    mock writes a synthetic SDF to the expected out_sdf path so
    the post-condition check and SDF parser both succeed."""

    @patch("molforge.wrappers.docking.gnina.shutil.which")
    @patch("molforge.wrappers.docking.gnina.subprocess.run")
    def test_dock_invokes_gnina(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/bin/gnina"

        # Capture the cwd-equivalent (temp dir) where gnina expects
        # to write out.sdf. We side-effect-write the synthetic SDF
        # at the path that appears in the command's --out flag.
        def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            out_idx = cmd.index("--out")
            out_path = Path(cmd[out_idx + 1])
            out_path.write_text(_REAL_GNINA_SDF, encoding="utf-8")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        mock_run.side_effect = fake_run

        result = Gnina().dock(
            _tiny_protein(),
            _tiny_protein(),
            center=(10.0, 5.0, -2.0),
            box_size=(20.0, 20.0, 20.0),
        )

        # gnina was called once with the right binary.
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/gnina"

        # Result has both poses with the right ranking.
        assert len(result.poses) == 2
        assert result.poses[0].score == 0.83  # default sort: CNNscore

    @patch("molforge.wrappers.docking.gnina.shutil.which")
    @patch("molforge.wrappers.docking.gnina.subprocess.run")
    def test_dock_raises_on_nonzero_exit(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/bin/gnina"
        # gnina exits non-zero, doesn't write the output file.
        result_proc = MagicMock()
        result_proc.returncode = 1
        result_proc.stdout = ""
        result_proc.stderr = "fatal: receptor preparation failed"
        mock_run.return_value = result_proc

        with pytest.raises(RuntimeError, match="exited with code 1"):
            Gnina().dock(
                _tiny_protein(),
                _tiny_protein(),
                center=(0.0, 0.0, 0.0),
            )

    @patch("molforge.wrappers.docking.gnina.shutil.which")
    @patch("molforge.wrappers.docking.gnina.subprocess.run")
    def test_dock_raises_on_missing_output_sdf(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """gnina returned 0 but produced no SDF — points at a
        binary-side bug that's worth surfacing clearly."""
        mock_which.return_value = "/usr/bin/gnina"
        result_proc = MagicMock()
        result_proc.returncode = 0
        result_proc.stdout = "(silently produced nothing)"
        result_proc.stderr = ""
        mock_run.return_value = result_proc

        with pytest.raises(RuntimeError, match="did not produce"):
            Gnina().dock(
                _tiny_protein(),
                _tiny_protein(),
                center=(0.0, 0.0, 0.0),
            )


# ---------------------------------------------------------------------
# End-to-end (real gnina binary)
# ---------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("gnina") is None, reason="gnina binary not installed")
class TestRealGnina:
    """Run the real gnina binary against a small system. Skipped
    in CI by default since gnina isn't pip-installable. Run locally
    with gnina on $PATH to exercise the full pipeline."""

    def test_docks_aspirin_against_1ake(self, tmp_path: Path) -> None:
        from molforge.io import fetch

        receptor = fetch("1AKE")
        # Aspirin SMILES needs preparing to SDF first.
        from molforge.wrappers.docking import prepare_ligand

        ligand_sdf = prepare_ligand("CC(=O)OC1=CC=CC=C1C(=O)O", out_path=tmp_path / "lig.sdf")

        result = Gnina(seed=42).dock(
            receptor=receptor,
            ligand=ligand_sdf,
            # Centroid as a placeholder — for a real test you'd use
            # a known site.
            center=(receptor.atom_array.coords.mean(axis=0)),
            box_size=(25.0, 25.0, 25.0),
            exhaustiveness=4,  # fast
            n_poses=3,
        )
        assert len(result.poses) >= 1
        assert mk.PROVENANCE in result.metadata
        # All three score types populated for cnn_scoring='rescore'.
        top = result.poses[0]
        assert top.metadata["vina_affinity"] is not None
        assert top.metadata["cnn_score"] is not None
        assert top.metadata["cnn_affinity"] is not None
