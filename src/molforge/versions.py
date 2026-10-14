"""One place to ask *what is installed here*.

:class:`~molforge.core.provenance.Provenance` answers "what produced this
output", but it can only record a version the wrapper managed to read at
the time. Wrappers that shell out to a native binary — fpocket, P2Rank,
GROMACS, ``sander``, gnina — record no version at all, so a manifest
covering them has blanks exactly where a reader most wants a number. And
there is no way to ask the question *before* a run, which is what you need
when one output is going to be attributed to thousands of predictions.

This module is that question, answered once::

    >>> from molforge import engine_versions
    >>> backends = engine_versions()                        # doctest: +SKIP
    >>> backends["OpenMM"].version                          # doctest: +SKIP
    '8.1.1'
    >>> [b.name for b in backends.values() if b.available]  # doctest: +SKIP
    ['OpenMM', 'RDKit', 'NumPy', 'molforge']

Three kinds of backend, because they fail to have a version for three
different reasons and a manifest should not conflate them:

``python``
    A pip/conda distribution. Read from :mod:`importlib.metadata` — free,
    exact, no subprocess.

``executable``
    A native binary found on ``$PATH``. Presence comes from
    :func:`shutil.which`; the version, where the tool exposes one, comes
    from running it with a known flag. Some of these tools ship no version
    flag at all (fpocket, ``tleap``), so they report ``available=True`` with
    an empty version and a :attr:`~BackendVersion.detail` saying why. That
    is a different fact from "not installed", and it is recorded as one.

``repo``
    Run from a checkout the *caller* points at (RoseTTAFold, DiffDock,
    ProteinMPNN, RFdiffusion take a ``repo_dir``). There is nothing global
    to inspect, so molforge cannot detect these at all and says so rather
    than reporting a confident ``False``.

Nothing here raises. A backend that cannot be inspected reports
``available=False`` and an empty version — the registry's whole job is to
be safe to call on any machine, including one with none of this installed.

See also:
    :func:`molforge.wrappers._versions.engine_version`, the per-wrapper
    lookup this builds on, and
    :func:`molforge.wrappers._versions.check_engine_version`, which warns
    when a wrapper's parser is outside its tested range.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from molforge.wrappers._versions import engine_version

__all__ = ["BackendVersion", "engine_versions"]


#: Default seconds to wait for a ``--version`` probe. Generous enough for a
#: cold binary on a loaded filesystem, short enough that a hung tool doesn't
#: stall a manifest.
DEFAULT_PROBE_TIMEOUT = 5.0

#: A version-looking token: two or more dot-separated numbers, optionally
#: with a trailing qualifier (``2023.3``, ``1.2.5``, ``8.1.1-beta``).
_VERSION_TOKEN = re.compile(r"\b(\d+\.\d+(?:\.\d+)*(?:[-+.][\w.]+)?)")


@dataclass(frozen=True)
class BackendVersion:
    """What molforge can determine about one engine or backend.

    Attributes:
        name: molforge's name for the backend, matching the ``engine``
            field a wrapper writes into :class:`Provenance` (``"Boltz"``,
            ``"fpocket"``, ``"GROMACS"``).
        category: Which part of molforge drives it — ``"folding"``,
            ``"docking"``, ``"pockets"``, ``"md"``, ``"freeenergy"``,
            ``"generative"``, or ``"runtime"`` for the shared numeric and
            chemistry libraries whose version changes results everywhere.
        kind: ``"python"``, ``"executable"``, or ``"repo"``. See the module
            docstring — these are three different reasons a version can be
            missing.
        requirement: What was looked for: the distribution name that
            resolved, or the executable name searched for on ``$PATH``.
        available: Whether molforge can find this backend right now.
            Always ``False`` for ``repo`` backends, which molforge cannot
            detect without the ``repo_dir`` the caller supplies.
        version: The detected version, or ``""``. Empty is never an error
            — read :attr:`detail` for which of the several reasons applies.
        location: Resolved path for an ``executable`` backend; ``""``
            otherwise.
        detail: Plain-language reason :attr:`version` is empty, or ``""``
            when a version was found.
    """

    name: str
    category: str
    kind: str
    requirement: str
    available: bool
    version: str = ""
    location: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable plain dict.

        Keys with nothing to say (``location``, ``detail``) are omitted so
        an embedded manifest block stays readable.
        """
        out: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "kind": self.kind,
            "requirement": self.requirement,
            "available": self.available,
            "version": self.version,
        }
        if self.location:
            out["location"] = self.location
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class _Backend:
    """How to detect one backend. Internal: the registry's row type."""

    name: str
    category: str
    kind: str
    #: Candidates tried in order; the first that resolves wins. Python
    #: backends need this because distributions get renamed (``rdkit`` was
    #: ``rdkit-pypi``) and Amber's dynamics binary may be ``pmemd`` or
    #: ``sander`` depending on the licence the site holds.
    requirements: tuple[str, ...] = ()
    #: Arguments that make an executable print its version. ``()`` means
    #: the tool exposes no such flag — then ``no_version_reason`` says so.
    version_argv: tuple[str, ...] = ()
    #: Tried before the generic token scan, for tools whose banner is full
    #: of other numbers (GROMACS prints a build date and a compiler).
    version_pattern: str = ""
    no_version_reason: str = ""


# Ordered by category so the registry reads like molforge's own table of
# contents. Distribution names are the ones the wrappers actually import
# or `pip install`; executable names are the wrappers' defaults, which the
# user can override per-call (``fpocket_executable=``, ``gmx_executable=``)
# — a non-default binary is recorded in that call's Provenance, not here.
_REGISTRY: tuple[_Backend, ...] = (
    # ---- folding -----------------------------------------------------
    _Backend("Boltz", "folding", "python", ("boltz",)),
    _Backend("Chai1", "folding", "python", ("chai_lab",)),
    _Backend("ESMFold", "folding", "python", ("transformers",)),
    _Backend("AlphaFold", "folding", "python", ("colabfold",)),
    _Backend(
        "RoseTTAFold",
        "folding",
        "repo",
        no_version_reason="run from a checkout passed as repo_dir; no global install to inspect",
    ),
    # ---- docking -----------------------------------------------------
    _Backend("Vina", "docking", "python", ("vina",)),
    _Backend("Gnina", "docking", "executable", ("gnina",), version_argv=("--version",)),
    _Backend(
        "DiffDock",
        "docking",
        "repo",
        no_version_reason="run from a checkout passed as repo_dir; no global install to inspect",
    ),
    _Backend("Meeko", "docking", "python", ("meeko",)),
    # ---- pockets -----------------------------------------------------
    _Backend(
        "fpocket",
        "pockets",
        "executable",
        ("fpocket",),
        no_version_reason="fpocket exposes no version flag",
    ),
    _Backend(
        "p2rank",
        "pockets",
        "executable",
        ("prank",),
        no_version_reason="the prank launcher exposes no version flag",
    ),
    # ---- molecular dynamics ------------------------------------------
    _Backend("OpenMM", "md", "python", ("openmm",)),
    _Backend(
        "GROMACS",
        "md",
        "executable",
        ("gmx",),
        version_argv=("--version",),
        # `gmx --version` prints a whole build report; anchor on the line
        # that actually names the version.
        version_pattern=r"GROMACS version:?\s*(\S+)",
    ),
    _Backend(
        "AMBER",
        "md",
        "executable",
        # pmemd is the faster, licensed binary; sander ships with the free
        # AmberTools. The wrapper prefers pmemd when present, so look for
        # it first and fall back the same way the wrapper does.
        ("pmemd", "sander"),
        version_argv=("--version",),
    ),
    _Backend(
        "tleap",
        "md",
        "executable",
        ("tleap",),
        no_version_reason="tleap exposes no version flag",
    ),
    # ---- free energy --------------------------------------------------
    _Backend(
        "AmberMMGBSA",
        "freeenergy",
        "executable",
        ("MMPBSA.py",),
        version_argv=("--version",),
    ),
    _Backend(
        "GromacsMMGBSA",
        "freeenergy",
        "executable",
        ("gmx_MMPBSA",),
        version_argv=("--version",),
    ),
    _Backend("alchemlyb", "freeenergy", "python", ("alchemlyb",)),
    _Backend("cinnabar", "freeenergy", "python", ("cinnabar",)),
    # ---- generative ---------------------------------------------------
    _Backend("ESM-IF1", "generative", "python", ("fair-esm",)),
    _Backend(
        "ProteinMPNN",
        "generative",
        "repo",
        no_version_reason="run from a checkout passed as repo_dir; no global install to inspect",
    ),
    _Backend(
        "RFdiffusion",
        "generative",
        "repo",
        no_version_reason="run from a checkout passed as repo_dir; no global install to inspect",
    ),
    # ---- shared runtime -----------------------------------------------
    # Not engines, but a change in any of them moves numbers everywhere,
    # so a manifest that omits them under-describes the run.
    _Backend("molforge", "runtime", "python", ("molforge",)),
    _Backend("NumPy", "runtime", "python", ("numpy",)),
    _Backend("RDKit", "runtime", "python", ("rdkit", "rdkit-pypi")),
    _Backend("PyTorch", "runtime", "python", ("torch",)),
    _Backend("Biopython", "runtime", "python", ("biopython",)),
    _Backend("MDTraj", "runtime", "python", ("mdtraj",)),
)


# Probing spawns subprocesses, so memoize: a manifest emitted per output
# would otherwise re-run every installed tool's --version. Keyed by the
# arguments that change the answer; `refresh=True` drops the entry.
_probe_cache: dict[tuple[bool, float], dict[str, BackendVersion]] = {}


def engine_versions(
    *,
    category: str | None = None,
    probe_executables: bool = True,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    refresh: bool = False,
) -> dict[str, BackendVersion]:
    """Report every engine and backend molforge knows how to drive.

    Args:
        category: Return only this category (``"folding"``, ``"docking"``,
            ``"pockets"``, ``"md"``, ``"freeenergy"``, ``"generative"``,
            ``"runtime"``). ``None`` returns all of them.
        probe_executables: Run each installed native binary's version flag.
            ``False`` skips every subprocess: executables are still
            reported as available or not via ``$PATH``, but with an empty
            version. Use it when you want the sweep to cost nothing, or in
            a sandbox where spawning processes is unwelcome.
        timeout: Seconds to wait for one probe. A tool that exceeds it is
            reported as available with no version, never as an error.
        refresh: Re-probe instead of reusing the memoized sweep. The
            result is cached because the answer only changes when
            something is installed, and a caller emitting one manifest per
            output would otherwise re-run every binary on the machine.

    Returns:
        :class:`BackendVersion` per backend, keyed by
        :attr:`BackendVersion.name` and ordered by category. Every known
        backend appears, installed or not — an absent engine is a fact
        about the environment worth recording.

    Raises:
        ValueError: If ``category`` isn't one molforge uses. A typo would
            otherwise silently return nothing, which reads as "nothing is
            installed".

    Example:
        >>> from molforge import engine_versions
        >>> md = engine_versions(category="md", probe_executables=False)
        >>> sorted(md)
        ['AMBER', 'GROMACS', 'OpenMM', 'tleap']
    """
    known = {b.category for b in _REGISTRY}
    if category is not None and category not in known:
        raise ValueError(f"unknown category {category!r}; expected one of {sorted(known)}")

    cache_key = (probe_executables, timeout)
    if refresh or cache_key not in _probe_cache:
        _probe_cache[cache_key] = {
            backend.name: _inspect(backend, probe_executables=probe_executables, timeout=timeout)
            for backend in _REGISTRY
        }
    swept = _probe_cache[cache_key]

    if category is None:
        return dict(swept)
    return {name: bv for name, bv in swept.items() if bv.category == category}


def _inspect(backend: _Backend, *, probe_executables: bool, timeout: float) -> BackendVersion:
    """Resolve one registry row against this machine."""
    if backend.kind == "python":
        return _inspect_python(backend)
    if backend.kind == "executable":
        return _inspect_executable(backend, probe_executables=probe_executables, timeout=timeout)
    return BackendVersion(
        name=backend.name,
        category=backend.category,
        kind=backend.kind,
        requirement="",
        available=False,
        detail=backend.no_version_reason,
    )


def _inspect_python(backend: _Backend) -> BackendVersion:
    """Look a distribution up in importlib.metadata, trying each alias."""
    for requirement in backend.requirements:
        version = engine_version(requirement)
        if version:
            return BackendVersion(
                name=backend.name,
                category=backend.category,
                kind="python",
                requirement=requirement,
                available=True,
                version=version,
            )
    canonical = backend.requirements[0] if backend.requirements else ""
    return BackendVersion(
        name=backend.name,
        category=backend.category,
        kind="python",
        requirement=canonical,
        available=False,
        detail=f"no installed distribution named {canonical!r}",
    )


def _inspect_executable(
    backend: _Backend, *, probe_executables: bool, timeout: float
) -> BackendVersion:
    """Find a binary on ``$PATH`` and, when it will say, ask its version."""
    for requirement in backend.requirements:
        resolved = shutil.which(requirement)
        if resolved is None:
            continue

        if backend.no_version_reason:
            detail = backend.no_version_reason
        elif not probe_executables:
            detail = "probe_executables=False"
        else:
            detail = ""

        version = ""
        if not detail:
            version = _probe_version(resolved, backend, timeout=timeout)
            if not version:
                detail = "the version probe produced nothing recognisable"

        return BackendVersion(
            name=backend.name,
            category=backend.category,
            kind="executable",
            requirement=requirement,
            available=True,
            version=version,
            location=resolved,
            detail=detail,
        )

    canonical = backend.requirements[0] if backend.requirements else ""
    return BackendVersion(
        name=backend.name,
        category=backend.category,
        kind="executable",
        requirement=canonical,
        available=False,
        detail=f"{canonical!r} not found on $PATH",
    )


def _probe_version(executable: str, backend: _Backend, *, timeout: float) -> str:
    """Run the version flag and pull a version out of what comes back.

    Best-effort by construction: a wrong guess here would put a wrong
    number in a manifest, so anything unrecognised yields ``""`` rather
    than a hopeful substring.
    """
    try:
        proc = subprocess.run(
            [executable, *backend.version_argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # Not executable, killed, timed out, undecodable output — all the
        # same outcome: we don't know the version.
        return ""

    # Version banners go to stdout or stderr depending on the tool, and
    # tools that reject the flag often still print their banner first.
    for stream in (proc.stdout, proc.stderr):
        if not stream:
            continue
        if backend.version_pattern:
            match = re.search(backend.version_pattern, stream)
            if match:
                return match.group(1)
        # Only the first non-empty line: a banner puts the version there,
        # while the rest of a build report is full of other numbers.
        first_line = next((ln for ln in stream.splitlines() if ln.strip()), "")
        token = _VERSION_TOKEN.search(first_line)
        if token:
            return token.group(1)
    return ""
