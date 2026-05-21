# molforge.wrappers

Thin wrappers around external engines. Each subcategory has its own
abstract base class and one or more concrete engines; see the
[wrappers guide](../guide/wrappers.md) for the design rationale.

| Subpackage                                                  | Engines                                                  |
| ----------------------------------------------------------- | -------------------------------------------------------- |
| [`molforge.wrappers.folding`](wrappers/folding.md)          | `ESMFold`, `AlphaFold`, `Boltz` (stub), `Rosetta` (stub) |
| [`molforge.wrappers.docking`](wrappers/docking.md)          | `Vina`, `DiffDock` (stub) + receptor/ligand prep         |
| [`molforge.wrappers.md`](wrappers/md.md)                    | `OpenMM`, `GROMACS` (stub)                               |
| [`molforge.wrappers.generative`](wrappers/generative.md)    | `RFdiffusion`, `ProteinMPNN`                             |
