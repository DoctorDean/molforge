"""Best-effort engine version detection and drift warnings.

The engine wrappers parse each tool's output by conventions — filename
globs, PDB/CIF conversion utilities — that a new engine release can change
without notice. Two lightweight guards help:

- :func:`engine_version` records the installed version (into
  :class:`~molforge.core.provenance.Provenance`), so a parse failure is
  diagnosable and reproducibility manifests carry a real version rather
  than a blank.
- :func:`check_engine_version` warns — *non-fatally* — when the installed
  version drifts outside the range the parser was written against.

Both only see **pip-installed** distributions via ``importlib.metadata``;
engines run from a cloned repo through a subprocess (ProteinMPNN,
DiffDock, RFdiffusion) have no discoverable version, so these return ``""``
and warn nothing. Those seams are instead guarded by their result parsers,
which raise a clear error when the expected output files are absent.
"""

from __future__ import annotations

import warnings
from importlib import metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.version import Version


def engine_version(distribution: str) -> str:
    """Return the installed version of ``distribution``, or ``""``.

    Args:
        distribution: The PyPI distribution name (e.g. ``"transformers"``,
            ``"boltz"``, ``"chai_lab"``).

    Returns:
        The version string, or ``""`` when the distribution isn't installed
        or its version can't be read (e.g. a repo-run engine).
    """
    try:
        return metadata.version(distribution)
    except Exception:
        # PackageNotFoundError, or any metadata quirk — best-effort only.
        return ""


def check_engine_version(
    distribution: str,
    *,
    engine: str,
    minimum: str | None = None,
    tested_max: str | None = None,
) -> str:
    """Warn if the installed ``distribution`` is outside the tested range.

    The warning is advisory — a version we haven't validated may still work,
    and the user may know better — so a mismatch emits a
    :class:`UserWarning`, never an error.

    Args:
        distribution: PyPI distribution name to inspect.
        engine: Human-readable engine name for the message (e.g. ``"ESMFold"``).
        minimum: Lowest version whose output the parser was written for.
            Below this, the engine may not produce what the wrapper expects.
        tested_max: Highest version the parser was validated against. Above
            this, a new release may have changed the output layout the
            wrapper globs for. ``None`` = no upper bound is asserted.

    Returns:
        The detected version (``""`` if undetectable — no warning is issued
        in that case, since there's nothing to compare).
    """
    version = engine_version(distribution)
    if not version:
        return ""
    parsed = _parse(version)
    if parsed is None:
        # Unparseable (a dev build, VCS install, ...) — don't second-guess it.
        return version

    min_parsed = _parse(minimum) if minimum is not None else None
    max_parsed = _parse(tested_max) if tested_max is not None else None

    if min_parsed is not None and parsed < min_parsed:
        warnings.warn(
            f"{engine}: installed {distribution} {version} is older than the "
            f"minimum {minimum} molforge's parser was written for; its output "
            "may not be parsed correctly. Consider upgrading.",
            UserWarning,
            stacklevel=2,
        )
    elif max_parsed is not None and parsed > max_parsed:
        warnings.warn(
            f"{engine}: installed {distribution} {version} is newer than the "
            f"last version molforge's {engine} parser was validated against "
            f"({tested_max}); if results look wrong, the engine's output format "
            "may have changed — please file an issue.",
            UserWarning,
            stacklevel=2,
        )
    return version


def _parse(version: str) -> Version | None:
    """Parse a version string with ``packaging`` (lazy), or ``None``.

    ``packaging`` isn't a molforge dependency; when it's absent (or the
    string is non-PEP 440) we simply skip the range comparison.
    """
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return None
    try:
        return Version(version)
    except InvalidVersion:
        return None
