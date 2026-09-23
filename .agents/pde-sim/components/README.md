# Component library

Reusable, **MMS-verified** building blocks promoted from completed studies. Part
of the Level-1 self-improvement loop (see `docs/DECISIONS.md` D18): the
`code-generation` agent starts from the closest component instead of writing from
scratch, and the orchestrator finds candidates via `components/case-index.json`.

## Rules for this library
- A component enters here **only after it passed its MMS/convergence checks** in a
  study. Unverified code stays in `artifacts/<study-id>/`.
- Each component carries a `component.json` (what it is, problem class, what it
  provides, how it was verified, provenance) and a short `README.md`.
- Each component also carries a `makefile` exposing a `run` target that rebuilds,
  runs the refinement sequence, and writes a `convergence.csv` with columns
  `h,L2,Linf` (one row per level, coarsest to finest). This is the contract
  `tests/run_regression.sh` checks: it runs `make run` in every
  `components/*/` directory and requires the last two levels to show ~2nd-order
  (or better) convergence. A component **without** a `makefile` is silently
  skipped by the suite, so a promoted component must include one to be guarded.
- Changes are guarded by `tests/run_regression.sh` — a component edit is trusted
  only if convergence still holds. Every change is a revertible git commit.

## Contents
No components have been promoted yet. The first verified study to pass its
MMS/convergence checks seeds this library and gets an entry in
`case-index.json`.
