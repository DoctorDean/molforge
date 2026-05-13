# Acknowledgements

`molforge` exists in conversation with a community of open-source projects
that came before it. This document records intellectual debts, technical
debts, and the work we've learned from.

## Direct inspiration

### Protkit

[Protkit](https://github.com/silicogenesis/protkit) (SilicoGenesis,
GPL-3.0) was the proximate inspiration for `molforge`. Protkit articulated
clearly the case for a **unified, hierarchical representation for
protein structures in Python** and the value of treating folding,
docking, scoring, and engineering tools as composable parts of a
shared pipeline rather than independent silos.

`molforge` adopts that thesis and extends it in a different direction:

- **Different internal architecture.** Protkit's data model is
  hierarchically canonical (Protein → Chain → Residue → Atom).
  `molforge` is linearly canonical (a NumPy-backed `AtomArray` is the
  source of truth; hierarchical access is via lightweight view
  classes). This trades some traversal locality for vectorized
  performance and zero-copy ML interop.
- **Different scope emphasis.** Protkit emphasizes the data
  representation itself. `molforge` emphasizes cross-tool workflows —
  the friction of moving structures between docking engines, MD
  packages, folding models, and design networks.
- **Different licensing.** Protkit is GPL-3.0; `molforge` is MIT, which
  reflects a preference for permissive licensing in a library meant to
  be embedded across an ecosystem of academic and industrial tools.

`molforge` contains no Protkit source code. The data model was designed
independently and shares no implementation. Where conceptual debts
exist — particularly around the "unified hierarchical representation"
framing — we credit Protkit by name in the README.

## Broader intellectual debts

The data-model design draws on patterns established by:

- **[Biotite](https://www.biotite-python.org/)** — for the
  NumPy-backed `AtomArray` pattern with parallel arrays per atomic
  property. Biotite demonstrated that this approach scales and
  interoperates cleanly with the scientific Python stack.
- **[Biopython](https://biopython.org/)** — for the canonical
  `Structure → Model → Chain → Residue → Atom` hierarchy that has shaped
  how Python programmers think about protein structure for two decades.
- **[BioPandas](https://biopandas.github.io/biopandas/)** — for
  popularizing tabular views of PDB data.
- **[MDAnalysis](https://www.mdanalysis.org/)** and
  **[MDTraj](https://mdtraj.org/)** — for trajectory abstractions and
  selection-language patterns.
- **[OpenMM](https://openmm.org/)** — for the modern Python interface
  to MD that makes wrapping possible.
- **[RDKit](https://www.rdkit.org/)** — for everything small-molecule.

## File-format specifications

`molforge`'s parsers implement the following public specifications,
without using their reference code:

- **PDB**: [wwPDB v3.30](https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html)
- **mmCIF / PDBx**: [PDBx/mmCIF dictionary](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Index/)
- **FASTA**: [NCBI specification](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp)
- **PDBQT**: [AutoDock Vina format documentation](https://autodock-vina.readthedocs.io/)
- **PQR**: [PDB2PQR documentation](https://pdb2pqr.readthedocs.io/)
- **SDF / MOL2**: standard chemistry exchange formats

## Funding and affiliation

`molforge` is an independent project maintained by Dean Sherry. It is
not affiliated with, endorsed by, or supported by any of the projects
listed above.

## How to suggest an addition

If you maintain a project that has materially influenced `molforge`'s
design or implementation, please open an issue or PR — we'd like to
credit you.
