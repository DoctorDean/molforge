# Examples

These notebooks show molforge as it's actually used — multi-step
pipelines tying several subpackages together. Each one is
self-contained and runnable end-to-end (pre-baked outputs are
included so you can read them without re-running).

| Notebook                                                  | What it shows                                                                                              |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [Cross-engine validation](cross_engine_validation.ipynb)  | Fold the same sequence with ESMFold and AlphaFold, then validate with consensus rules from `molforge.validation`. |
| [De novo design](de_novo_design.ipynb)                    | RFdiffusion → ProteinMPNN → ESMFold pipeline: scaffold, sequence-design, fold-check, filter.               |
| [End-to-end design](end_to_end_design.ipynb)              | Long-form example: from a binding-site spec to a ranked set of designs ready for wet-lab follow-up.        |

Looking for a guided tour of one subpackage at a time?
See the [walkthroughs](../walkthroughs/01_sequences.ipynb) instead —
six shorter notebooks, one per subpackage.

The raw notebook files live in
[notebooks/examples/](https://github.com/DoctorDean/molforge/tree/master/notebooks/examples)
on GitHub.
