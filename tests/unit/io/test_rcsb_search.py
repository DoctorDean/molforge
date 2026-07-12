"""Tests for io.search_rcsb.

urllib.request.urlopen is mocked, so no real request hits RCSB; the tests
decode the URL the search would have sent to assert the query was built
correctly, and parse mocked JSON responses.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from molforge.io import search_rcsb


def _response(body: str) -> object:
    resp = MagicMock()
    resp.read.return_value = body.encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _sent_payload(mock: MagicMock) -> dict:
    url = mock.call_args[0][0]
    return json.loads(parse_qs(urlsplit(url).query)["json"][0])


class TestSearchRcsb:
    def test_returns_identifiers_in_order(self) -> None:
        body = json.dumps(
            {"result_set": [{"identifier": "1ABC"}, {"identifier": "2XYZ"}], "total_count": 2}
        )
        with patch("urllib.request.urlopen", return_value=_response(body)) as mock:
            ids = search_rcsb("hemoglobin")
        assert ids == ["1ABC", "2XYZ"]
        payload = _sent_payload(mock)
        assert payload["query"]["service"] == "full_text"
        assert payload["query"]["parameters"]["value"] == "hemoglobin"
        assert payload["return_type"] == "entry"

    def test_limit_maps_to_rows(self) -> None:
        with patch("urllib.request.urlopen", return_value=_response('{"result_set": []}')) as mock:
            search_rcsb("kinase", limit=5)
        assert _sent_payload(mock)["request_options"]["paginate"]["rows"] == 5

    def test_no_matches_returns_empty(self) -> None:
        # RCSB answers "nothing found" with HTTP 204 and an empty body.
        with patch("urllib.request.urlopen", return_value=_response("")):
            assert search_rcsb("zzzznotarealthing") == []

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            search_rcsb("   ")

    def test_bad_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="limit must be"):
            search_rcsb("kinase", limit=0)

    def test_http_error_becomes_oserror(self) -> None:
        import urllib.error

        err = urllib.error.HTTPError(url="u", code=500, msg="err", hdrs=None, fp=None)  # type: ignore[arg-type]
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(OSError, match="RCSB search failed"),
        ):
            search_rcsb("kinase")

    def test_network_error_becomes_oserror(self) -> None:
        import urllib.error

        err = urllib.error.URLError("unreachable")
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(OSError, match="could not reach"),
        ):
            search_rcsb("kinase")
