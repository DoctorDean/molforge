"""Tests for io.fetch_ccd / fetch_ccd_many.

Same two boundaries as test_chembl.py: urllib.request.urlopen (the RCSB
download) and, for the success paths, the ``molforge.core._rdkit`` shim (so a
Molecule can be built without RDKit). The genuine RDKit-absent path is checked
with only the network mocked.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from molforge.core import Molecule, RDKitNotInstalledError, _rdkit
from molforge.io import fetch_ccd, fetch_ccd_many


class _FakeMol:
    def __init__(self, molblock: str) -> None:
        self.molblock = molblock

    def GetNumAtoms(self) -> int:
        return len(self.molblock)


# A CCD SDF's title line is the code itself; the counts line and the rest are
# irrelevant to the wrapper, which hands the whole block to RDKit.
STI_SDF = """STI
  CCTOOLS-0919241030

 68 75  0  0  1  0  0  0  0  0999 V2000
$$$$
"""


def _rdkit_available() -> bool:
    return importlib.util.find_spec("rdkit") is not None


def _response(body: str) -> object:
    resp = MagicMock()
    resp.read.return_value = body.encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


@pytest.fixture
def fake_molblock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rdkit, "mol_from_molblock", lambda b, **k: _FakeMol(b))


class TestFetchCcd:
    def test_builds_molecule(self, fake_molblock: None) -> None:
        with patch("urllib.request.urlopen", return_value=_response(STI_SDF)) as mock:
            mol = fetch_ccd("STI")
        assert isinstance(mol, Molecule)
        assert mol.name == "STI"
        assert mol.metadata["source"] == "rcsb-ccd"
        assert mol.metadata["ccd_code"] == "STI"
        assert mock.call_args[0][0] == "https://files.rcsb.org/ligands/download/STI_ideal.sdf"

    def test_code_is_uppercased(self, fake_molblock: None) -> None:
        """RCSB serves uppercase; a lowercase code shouldn't 404."""
        with patch("urllib.request.urlopen", return_value=_response(STI_SDF)) as mock:
            mol = fetch_ccd("sti")
        assert mock.call_args[0][0].endswith("/STI_ideal.sdf")
        assert mol.name == "STI"

    def test_surrounding_whitespace_stripped(self, fake_molblock: None) -> None:
        with patch("urllib.request.urlopen", return_value=_response(STI_SDF)) as mock:
            fetch_ccd("  STI\n")
        assert mock.call_args[0][0].endswith("/STI_ideal.sdf")

    def test_timeout_forwarded(self, fake_molblock: None) -> None:
        with patch("urllib.request.urlopen", return_value=_response(STI_SDF)) as mock:
            fetch_ccd("STI", timeout=5.0)
        assert mock.call_args.kwargs["timeout"] == 5.0

    def test_sanitize_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``sanitize=False`` is the escape hatch for metal-coordinated components."""
        seen: dict[str, object] = {}

        def _capture(block: str, **kwargs: object) -> _FakeMol:
            seen.update(kwargs)
            return _FakeMol(block)

        monkeypatch.setattr(_rdkit, "mol_from_molblock", _capture)
        with patch("urllib.request.urlopen", return_value=_response(STI_SDF)):
            fetch_ccd("HEM", sanitize=False)
        assert seen["sanitize"] is False

    def test_empty_code_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fetch_ccd("  ")

    @pytest.mark.parametrize("code", ["../etc/passwd", "ST I", "STI/", "ST-I"])
    def test_non_alphanumeric_code_raises(self, code: str) -> None:
        """A code can't smuggle a path segment into the URL."""
        with pytest.raises(ValueError, match="alphanumeric"):
            fetch_ccd(code)

    def test_http_error_becomes_oserror(self) -> None:
        import urllib.error

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(OSError, match="CCD fetch failed: RCSB returned HTTP 404"),
        ):
            fetch_ccd("ZZZZ9")

    def test_url_error_becomes_oserror(self) -> None:
        import urllib.error

        with (
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")),
            pytest.raises(OSError, match="could not reach RCSB"),
        ):
            fetch_ccd("STI")

    def test_unparseable_sdf_names_the_escape_hatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valence failure should point at ``sanitize=False``, not just fail."""

        def _boom(block: str, **kwargs: object) -> _FakeMol:
            raise ValueError("Explicit valence for atom # 13 O, 2, is greater than permitted")

        monkeypatch.setattr(_rdkit, "mol_from_molblock", _boom)
        with (
            patch("urllib.request.urlopen", return_value=_response(STI_SDF)),
            pytest.raises(ValueError) as excinfo,
        ):
            fetch_ccd("HEM")
        message = str(excinfo.value)
        assert "'HEM'" in message
        assert "sanitize=False" in message

    def test_no_escape_hatch_hint_when_already_unsanitized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(block: str, **kwargs: object) -> _FakeMol:
            raise ValueError("nope")

        monkeypatch.setattr(_rdkit, "mol_from_molblock", _boom)
        with (
            patch("urllib.request.urlopen", return_value=_response(STI_SDF)),
            pytest.raises(ValueError) as excinfo,
        ):
            fetch_ccd("HEM", sanitize=False)
        assert "sanitize=False" not in str(excinfo.value)

    @pytest.mark.skipif(_rdkit_available(), reason="RDKit installed; nothing to assert")
    def test_rdkit_absent_raises(self) -> None:
        # Network mocked, RDKit genuinely absent -> the shim raises.
        with (
            patch("urllib.request.urlopen", return_value=_response(STI_SDF)),
            pytest.raises(RDKitNotInstalledError),
        ):
            fetch_ccd("STI")


class TestFetchCcdMany:
    def test_fetches_all_in_order(self, fake_molblock: None) -> None:
        with patch("urllib.request.urlopen", return_value=_response(STI_SDF)):
            mols = fetch_ccd_many(["STI", "NAD"])
        assert [m.name for m in mols] == ["STI", "NAD"]

    def test_empty_input_returns_empty(self, fake_molblock: None) -> None:
        assert fetch_ccd_many([]) == []

    def test_bad_on_error_raises(self) -> None:
        with pytest.raises(ValueError, match="on_error must be"):
            fetch_ccd_many(["STI"], on_error="nope")

    def test_raise_stops_at_first_failure(self, fake_molblock: None) -> None:
        import urllib.error

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with (
            patch("urllib.request.urlopen", side_effect=[err, _response(STI_SDF)]),
            pytest.raises(OSError, match="CCD fetch failed"),
        ):
            fetch_ccd_many(["BAD", "STI"])

    def test_skip_drops_download_failures(self, fake_molblock: None) -> None:
        import urllib.error

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=[err, _response(STI_SDF)]):
            mols = fetch_ccd_many(["ZZZZ9", "STI"], on_error="skip")
        assert [m.name for m in mols] == ["STI"]

    def test_skip_drops_unparseable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One metal-coordinated component shouldn't lose the rest of the set."""
        calls = {"n": 0}

        def _sometimes(block: str, **kwargs: object) -> _FakeMol:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("valence")
            return _FakeMol(block)

        monkeypatch.setattr(_rdkit, "mol_from_molblock", _sometimes)
        with patch("urllib.request.urlopen", return_value=_response(STI_SDF)):
            mols = fetch_ccd_many(["HEM", "STI"], on_error="skip")
        assert [m.name for m in mols] == ["STI"]

    def test_skip_can_drop_everything(self, fake_molblock: None) -> None:
        import urllib.error

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err):
            assert fetch_ccd_many(["A1", "B2"], on_error="skip") == []


@pytest.mark.slow
class TestAgainstRcsb:
    """Hits the real RCSB endpoint. Deselected by default with ``-m 'not slow'``."""

    def test_fetches_imatinib(self) -> None:
        pytest.importorskip("rdkit")
        mol = fetch_ccd("STI")
        assert mol.name == "STI"
        assert 490 < mol.molecular_weight < 500
        assert mol.metadata["ccd_code"] == "STI"

    def test_ideal_sdf_carries_a_conformer(self) -> None:
        """The point of the SDF over a SMILES lookup: 3D coordinates."""
        pytest.importorskip("rdkit")
        mol = fetch_ccd("ATP")
        assert mol._mol.GetNumConformers() == 1

    def test_unknown_code_404s(self) -> None:
        with pytest.raises(OSError, match="HTTP 404"):
            fetch_ccd("ZZZZ9")
