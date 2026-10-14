"""Documented key vocabulary for :attr:`molforge.core.Protein.metadata`.

``Protein.metadata`` is a free-form ``dict[str, Any]`` by design — it
carries whatever a parser or engine wants to attach, including
open-ended things like PDB ``REMARK`` records. Keeping it a plain dict
means no breaking change for code that writes arbitrary keys.

But "free-form" shouldn't mean "undocumented". The keys below are the
ones molforge's own parsers and engine wrappers produce, and they form
the *contract*: downstream code can rely on these names and value
types being stable across the 1.x series. Keys outside this list are
still permitted but carry no stability guarantee.

Two things to use here:

- **String constants** (``PDB_ID``, ``MEAN_CONFIDENCE``, ...). Prefer
  these over bare string literals when reading or writing metadata, so
  a typo is a ``NameError`` at import time rather than a silently
  missing key at runtime.
- **:class:`ProteinMetadata`** — a ``TypedDict`` (``total=False``,
  every key optional) that documents the value type of each key. It's
  a typing aid only: ``Protein.metadata`` is still a plain ``dict`` at
  runtime, but annotating a local as ``ProteinMetadata`` gives editors
  and ``mypy`` the key/type information.

Key groups:

- **Structural-IO header keys** — set by :func:`molforge.io.read_pdb`
  and :func:`molforge.io.read_cif` from file header records.
- **Uniform folding-engine keys** — set by *every* folding-engine
  wrapper (ESMFold, AlphaFold, Boltz, RoseTTAFold) so downstream code
  can read prediction confidence without knowing which engine ran.
  :func:`molforge.io.load_alphafold` also populates these.
- **Engine-specific folding keys** — set by some folding wrappers but
  not all; presence depends on the engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ----------------------------------------------------------------------
# Structural-IO header keys (PDB / mmCIF parsers)
# ----------------------------------------------------------------------

PDB_ID = "pdb_id"
"""4-character PDB accession code, e.g. ``"1ABC"`` (str)."""

TITLE = "title"
"""Free-text structure title from the PDB ``TITLE`` / mmCIF ``_struct.title`` (str)."""

CLASSIFICATION = "classification"
"""PDB ``HEADER`` classification field, e.g. ``"HYDROLASE"`` (str)."""

DEPOSITION_DATE = "deposition_date"
"""Deposition date string as it appears in the PDB ``HEADER`` (str)."""

EXPERIMENTAL_METHOD = "experimental_method"
"""Experimental method, e.g. ``"X-RAY DIFFRACTION"`` (str)."""

RESOLUTION = "resolution"
"""Resolution in Angstrom (float). Absent for non-diffraction structures."""

# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

PROVENANCE = "provenance"
"""First-class provenance record (:class:`molforge.core.Provenance`).

The canonical key for "what produced this output." Carries engine
name, version, parameters, inputs, and a recursive parent pointer to
the upstream step's provenance. See :mod:`molforge.core.provenance`
for construction helpers and the data shape.

This is the documented replacement for the older ad-hoc
``metadata["engine"]`` / ``metadata["source_args"]`` keys. Both
continue to work for backwards compatibility; new code should write a
:class:`Provenance` to this key instead."""

# ----------------------------------------------------------------------
# Uniform folding-engine keys (every folding wrapper sets these)
# ----------------------------------------------------------------------

ENGINE = "engine"
"""Name of the folding engine that produced the structure (str), e.g.
``"ESMFold"``, ``"AlphaFold"``, ``"Boltz"``, ``"RoseTTAFold"``."""

SOURCE_SEQUENCE = "source_sequence"
"""The one-letter input sequence the engine folded (str)."""

CONFIDENCE_PER_RESIDUE = "confidence_per_residue"
"""``(L,)`` float32 array of per-residue pLDDT-style confidence (0-100)."""

CONFIDENCE_PER_ATOM = "confidence_per_atom"
"""``(N_atoms,)`` float32 array of per-atom confidence (0-100)."""

MEAN_CONFIDENCE = "mean_confidence"
"""Scalar mean per-residue confidence (float, 0-100)."""

# ----------------------------------------------------------------------
# Engine-specific folding keys (presence depends on the engine)
# ----------------------------------------------------------------------

SOURCE = "source"
"""Provenance tag (str). Set to ``"alphafold"`` by
:func:`molforge.io.load_alphafold`."""

MODEL_NAME = "model_name"
"""Engine-internal model identifier (str). Set by ESMFold."""

MODEL_TYPE = "model_type"
"""Model-type identifier (str). Set by AlphaFold, e.g. ``"monomer"``."""

MODEL_VERSION = "model_version"
"""Model-version identifier (str). Set by Boltz, e.g. ``"boltz2"``."""

JOB_NAME = "job_name"
"""Job name used for engine output files (str). Set by RoseTTAFold."""

USE_MSA_SERVER = "use_msa_server"
"""Whether an MSA server was used (bool). Set by Boltz."""

PTM = "ptm"
"""Predicted TM-score for the whole structure (float). Set by Boltz."""

IPTM = "iptm"
"""Interface predicted TM-score (float). Set by Boltz; meaningful for complexes."""

CONFIDENCE_SCORE = "confidence_score"
"""Composite confidence score (float). Set by Boltz."""

AFFINITY_VALUE = "affinity_value"
"""Predicted binding affinity (float). Set by Boltz-2's affinity prediction —
its ``affinity_pred_value``, a log-scale IC50-like value where *lower* means
stronger predicted binding."""

AFFINITY_PROBABILITY = "affinity_probability"
"""Probability that the ligand is a binder (float, 0-1). Set by Boltz-2 from
``affinity_probability_binary``."""

PAE = "pae"
"""``(L, L)`` predicted aligned error matrix (float array). Set by RoseTTAFold."""

PDE = "pde"
"""``(L, L)`` predicted distance error matrix (float array). Set by RoseTTAFold."""

PAE_INTER = "pae_inter"
"""Scalar mean inter-chain PAE (float). RoseTTAFold's headline interface
metric; values below ~10 indicate a high-quality interface."""

PAE_PROT = "pae_prot"
"""Scalar mean PAE over protein residues only (float). Set by RoseTTAFold."""

MEAN_PAE = "mean_pae"
"""Scalar mean of the full PAE matrix (float). Set by RoseTTAFold."""

MEAN_PLDDT = "mean_plddt"
"""Scalar mean pLDDT (float). Set by RoseTTAFold and
:func:`molforge.io.load_alphafold`. Equivalent to :data:`MEAN_CONFIDENCE`;
the latter is the cross-engine-uniform name and should be preferred."""

PLDDT = "plddt"
"""``(N_atoms,)`` float32 per-atom pLDDT. Legacy key set by
:func:`molforge.io.load_alphafold`; :data:`CONFIDENCE_PER_ATOM` is the
cross-engine-uniform name and should be preferred."""

PLDDT_PER_RESIDUE = "plddt_per_residue"
"""``(L,)`` float32 per-residue pLDDT. Legacy key set by
:func:`molforge.io.load_alphafold`; :data:`CONFIDENCE_PER_RESIDUE` is the
cross-engine-uniform name and should be preferred."""


class ProteinMetadata(TypedDict, total=False):
    """Typed view of the documented :attr:`Protein.metadata` keys.

    Every key is optional (``total=False``). This is a typing aid only —
    ``Protein.metadata`` remains a plain ``dict[str, Any]`` at runtime,
    and keys outside this set are still permitted (without stability
    guarantees). Annotate a local variable as ``ProteinMetadata`` to get
    editor / ``mypy`` support for the documented vocabulary.
    """

    # Structural-IO header keys.
    pdb_id: str
    title: str
    classification: str
    deposition_date: str
    experimental_method: str
    resolution: float

    # Provenance — first-class record of what produced this output.
    # The value type is molforge.core.Provenance; declared here as
    # ``Any`` to avoid a runtime circular import (the dataclass lives
    # in molforge.core.provenance which imports nothing from this
    # module).
    provenance: Any

    # Uniform folding-engine keys.
    engine: str
    source_sequence: str
    confidence_per_residue: NDArray[Any]
    confidence_per_atom: NDArray[Any]
    mean_confidence: float

    # Engine-specific folding keys.
    source: str
    model_name: str
    model_type: str
    model_version: str
    job_name: str
    use_msa_server: bool
    ptm: float
    iptm: float
    confidence_score: float
    pae: NDArray[Any]
    pde: NDArray[Any]
    pae_inter: float
    pae_prot: float
    mean_pae: float
    mean_plddt: float
    plddt: NDArray[Any]
    plddt_per_residue: NDArray[Any]


#: Every documented key, as a frozenset — useful for validation or tests
#: that want to assert "this metadata dict uses only documented keys".
DOCUMENTED_KEYS: frozenset[str] = frozenset(
    {
        PDB_ID,
        TITLE,
        CLASSIFICATION,
        DEPOSITION_DATE,
        EXPERIMENTAL_METHOD,
        RESOLUTION,
        PROVENANCE,
        ENGINE,
        SOURCE_SEQUENCE,
        CONFIDENCE_PER_RESIDUE,
        CONFIDENCE_PER_ATOM,
        MEAN_CONFIDENCE,
        SOURCE,
        MODEL_NAME,
        MODEL_TYPE,
        MODEL_VERSION,
        JOB_NAME,
        USE_MSA_SERVER,
        PTM,
        IPTM,
        CONFIDENCE_SCORE,
        PAE,
        PDE,
        PAE_INTER,
        PAE_PROT,
        MEAN_PAE,
        MEAN_PLDDT,
        PLDDT,
        PLDDT_PER_RESIDUE,
    }
)


__all__ = [
    # TypedDict + key set
    "ProteinMetadata",
    "DOCUMENTED_KEYS",
    # Structural-IO header keys
    "PDB_ID",
    "TITLE",
    "CLASSIFICATION",
    "DEPOSITION_DATE",
    "EXPERIMENTAL_METHOD",
    "RESOLUTION",
    # Provenance
    "PROVENANCE",
    # Uniform folding-engine keys
    "ENGINE",
    "SOURCE_SEQUENCE",
    "CONFIDENCE_PER_RESIDUE",
    "CONFIDENCE_PER_ATOM",
    "MEAN_CONFIDENCE",
    # Engine-specific folding keys
    "SOURCE",
    "MODEL_NAME",
    "MODEL_TYPE",
    "MODEL_VERSION",
    "JOB_NAME",
    "USE_MSA_SERVER",
    "PTM",
    "IPTM",
    "CONFIDENCE_SCORE",
    "PAE",
    "PDE",
    "PAE_INTER",
    "PAE_PROT",
    "MEAN_PAE",
    "MEAN_PLDDT",
    "PLDDT",
    "PLDDT_PER_RESIDUE",
]
