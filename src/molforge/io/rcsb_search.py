"""Full-text search against the RCSB PDB.

:func:`search_rcsb` turns a free-text query into a ranked list of PDB IDs
using the RCSB Search API, ready to hand straight to
:func:`molforge.io.fetch_many`. Like :func:`molforge.io.fetch`, it uses only
the standard library (:mod:`urllib`), so it adds no dependency.
"""

from __future__ import annotations

_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

__all__ = ["search_rcsb"]


def search_rcsb(query: str, *, limit: int = 25, timeout: float = 30.0) -> list[str]:
    """Full-text search the RCSB PDB, returning matching entry IDs.

    Runs a full-text query against the RCSB Search API and returns the PDB
    IDs of the best matches, most relevant first — feed them to
    :func:`molforge.io.fetch_many` to download the structures.

    Args:
        query: Free-text query, e.g. ``"hemoglobin"`` or ``"CRISPR Cas9"``.
        limit: Maximum number of IDs to return (the top ``limit`` hits).
        timeout: Network timeout in seconds.

    Returns:
        Up to ``limit`` PDB IDs ranked by relevance; empty if nothing matches.

    Raises:
        ValueError: If ``query`` is empty or ``limit`` is less than 1.
        OSError: If the search request fails — network error, timeout, or a
            non-2xx response from RCSB.

    Example:
        >>> from molforge.io import search_rcsb, fetch_many
        >>> ids = search_rcsb("hemoglobin", limit=5)
        >>> structures = fetch_many(ids)
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query.strip()},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit}},
    }
    url = f"{_SEARCH_URL}?" + urllib.parse.urlencode({"json": json.dumps(payload)})

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise OSError(f"RCSB search failed: returned HTTP {e.code} for query {query!r}.") from e
    except urllib.error.URLError as e:
        raise OSError(
            f"RCSB search failed: could not reach RCSB ({e.reason}). Check your network connection."
        ) from e

    # RCSB answers "no matches" with HTTP 204 and an empty body.
    if not body.strip():
        return []
    data = json.loads(body)
    return [str(hit["identifier"]) for hit in data.get("result_set", [])]
