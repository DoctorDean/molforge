"""Receptor and ligand preparation for AutoDock Vina via meeko / RDKit.

What "preparation" means for Vina:

- **Receptors**: protein structure with polar hydrogens added, atom
  types assigned per the AutoDock 4 force field, Gasteiger partial
  charges computed, and the whole thing serialized as PDBQT (PDB
  format extended with two extra columns for charge and atom type).
- **Ligands**: small-molecule structure with explicit Hs, atom types
  and charges as above, plus identification of **rotatable bonds**
  so Vina knows what to twist during the search.

This module wraps [meeko](https://github.com/forlilab/Meeko) and
[RDKit](https://www.rdkit.org/) to do that prep automatically. The
heavy imports are lazy — installing meeko/RDKit is only required if
you actually call these functions.

All functions write to a temp file and return its path, since
``vina.Vina`` consumes file paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from molforge.docking import DockingEngineNotInstalledError

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


def _require_meeko() -> Any:
    """Import meeko or raise a clean DockingEngineNotInstalledError."""
    try:
        import meeko
    except ImportError as e:
        raise DockingEngineNotInstalledError(
            "Receptor/ligand preparation requires `meeko`. Install with:\n"
            "    pip install meeko\n"
            "Meeko also requires RDKit; install via `pip install 'molforge[docking]'`.\n"
            f"Underlying error: {e}"
        ) from e
    return meeko


def _require_rdkit() -> Any:
    """Import rdkit or raise a clean DockingEngineNotInstalledError."""
    try:
        from rdkit import Chem
    except ImportError as e:
        raise DockingEngineNotInstalledError(
            "Ligand preparation from SMILES / SDF requires RDKit. Install with:\n"
            "    pip install 'molforge[docking]'\n"
            f"Underlying error: {e}"
        ) from e
    return Chem


# ----------------------------------------------------------------------
# Receptor prep
# ----------------------------------------------------------------------
def prepare_receptor(
    receptor: Protein | str | PathLike[str],
    out_path: str | PathLike[str],
    *,
    rigid_only: bool = True,
) -> Path:
    """Prepare a receptor for AutoDock Vina, writing a PDBQT to ``out_path``.

    Args:
        receptor: A :class:`molforge.core.Protein`, a path to a PDB / mmCIF
            file, or a path to an already-prepared PDBQT (in which case
            we copy it to ``out_path`` and return).
        out_path: Destination path for the PDBQT file.
        rigid_only: If True (default), treat the entire receptor as rigid
            (the usual case). Set False if you've added side-chain
            flexibility annotations upstream.

    Returns:
        The path the PDBQT was written to.

    Raises:
        DockingEngineNotInstalledError: If ``meeko`` isn't installed.
    """
    out = Path(out_path)
    # Fast path: caller already has a PDBQT
    if not hasattr(receptor, "atom_array"):
        src = Path(receptor)
        if src.suffix.lower() == ".pdbqt":
            out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return out

    meeko = _require_meeko()
    # meeko's receptor prep takes a PDB file path. If we have a Protein,
    # write it to a temp PDB first; if we have a path that's already
    # PDB/mmCIF, just hand it over.
    if hasattr(receptor, "atom_array"):
        # Protein input — serialize via molforge's own PDB writer.
        from molforge.io import write_pdb

        tmp_pdb = out.with_suffix(".pdb")
        write_pdb(receptor, tmp_pdb)  # type: ignore[arg-type]
        src_pdb = tmp_pdb
    else:
        src_pdb = Path(receptor)

    # Newer meeko (>=0.5) exposes MoleculePreparation for receptors via the
    # ProteinPrepper / PDBQTReceptor classes; older versions use a CLI tool.
    # We use the Python API.
    prep_cls = getattr(meeko, "PDBQTReceptor", None) or getattr(meeko, "MoleculePreparation", None)
    if prep_cls is None:
        raise DockingEngineNotInstalledError(
            "meeko is installed but no usable receptor preparer found. "
            "Upgrade with `pip install -U meeko`."
        )

    # The interface varies slightly between meeko versions. We try the most
    # common idioms; if all fail we surface the error with the meeko hint.
    try:
        # Modern meeko: PDBQTReceptor(pdb_path).write_pdbqt(out_path)
        if hasattr(meeko, "PDBQTReceptor"):
            receptor_prep = meeko.PDBQTReceptor(str(src_pdb))
            receptor_prep.write_pdbqt(str(out))
            return out
    except (AttributeError, TypeError):
        pass

    # Fallback: try the CLI-equivalent function if available.
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem

        # This path works for ligand-ish receptors only; for real
        # protein receptors users should upgrade meeko.
        mol = Chem.MolFromPDBFile(str(src_pdb), removeHs=False)
        if mol is None:
            raise RuntimeError(f"RDKit could not load receptor PDB at {src_pdb}")
        prep = MoleculePreparation(rigid_macrocycles=rigid_only)
        prep.prepare(mol)
        pdbqt_text, _, _ = PDBQTWriterLegacy.write_string(prep.setup)
        out.write_text(pdbqt_text, encoding="utf-8")
        return out
    except Exception as e:
        raise DockingEngineNotInstalledError(
            f"Could not prepare receptor: {e}. Try upgrading meeko, "
            "or fall back to mk_prepare_receptor.py from the meeko CLI."
        ) from e


# ----------------------------------------------------------------------
# Ligand prep
# ----------------------------------------------------------------------
def prepare_ligand(
    ligand: str | PathLike[str],
    out_path: str | PathLike[str],
    *,
    from_smiles: bool = False,
    add_hydrogens: bool = True,
    generate_3d: bool = True,
) -> Path:
    """Prepare a ligand for AutoDock Vina, writing a PDBQT to ``out_path``.

    Args:
        ligand: One of:

            - A path to an already-prepared ``.pdbqt`` file (copied as-is).
            - A path to an ``.sdf`` / ``.mol`` / ``.mol2`` / ``.pdb`` file.
            - A SMILES string (if ``from_smiles=True``).
        out_path: Destination path for the PDBQT file.
        from_smiles: Treat ``ligand`` as a SMILES string rather than a path.
        add_hydrogens: Add explicit hydrogens (needed by Vina).
        generate_3d: Generate a 3-D conformer via RDKit's ETKDG.

    Returns:
        The path the PDBQT was written to.

    Raises:
        DockingEngineNotInstalledError: If ``meeko`` or ``rdkit`` is missing.
    """
    out = Path(out_path)

    # Fast path: caller already has a PDBQT file
    if not from_smiles:
        src = Path(ligand)
        if src.exists() and src.suffix.lower() == ".pdbqt":
            out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return out

    chem = _require_rdkit()
    _require_meeko()
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit.Chem import AllChem

    # Load the molecule
    if from_smiles:
        mol = chem.MolFromSmiles(str(ligand))
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {ligand!r}")
    else:
        src = Path(ligand)
        suffix = src.suffix.lower()
        if suffix == ".sdf":
            supplier = chem.SDMolSupplier(str(src), removeHs=False)
            mol = next((m for m in supplier if m is not None), None)
        elif suffix == ".mol":
            mol = chem.MolFromMolFile(str(src), removeHs=False)
        elif suffix == ".mol2":
            mol = chem.MolFromMol2File(str(src), removeHs=False)
        elif suffix == ".pdb":
            mol = chem.MolFromPDBFile(str(src), removeHs=False)
        else:
            raise ValueError(
                f"unsupported ligand file extension {suffix!r}; "
                "expected .pdbqt, .sdf, .mol, .mol2, .pdb, or pass from_smiles=True"
            )
        if mol is None:
            raise ValueError(f"could not load ligand from {src}")

    # Add Hs and 3D coords if requested
    if add_hydrogens:
        mol = chem.AddHs(mol)
    if generate_3d and mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

    # Prep with meeko
    prep = MoleculePreparation()
    prep.prepare(mol)
    pdbqt_text, _, _ = PDBQTWriterLegacy.write_string(prep.setup)
    out.write_text(pdbqt_text, encoding="utf-8")
    return out


# ----------------------------------------------------------------------
# Convenience: best-effort dispatcher
# ----------------------------------------------------------------------
def is_pdbqt_path(value: object) -> bool:
    """True iff ``value`` is a path string pointing to a .pdbqt file."""
    if hasattr(value, "atom_array"):
        return False
    try:
        return Path(value).suffix.lower() == ".pdbqt"  # type: ignore[arg-type]
    except TypeError:
        return False
