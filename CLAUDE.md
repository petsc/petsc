# PETSc

PETSc is a C library for parallel numerical computation using MPI. The codebase has ~1M lines of C with Python bindings (petsc4py).

## Project Layout

- `src/<package>/` — source organized by package: `vec`, `mat`, `ksp`, `snes`, `ts`, `dm`, `tao`, `sys`
- `src/<package>/impls/` — implementations (subclasses) of each package's abstract type
- `src/<package>/interface/` — public API for each package
- `src/<package>/tests/` and `src/<package>/tutorials/` — tests and examples
- `include/` — public headers (`petsc*.h`)
- `include/petsc/private/` — private headers (`*impl.h`)
- `src/binding/petsc4py/` — Python bindings (Cython)

## Key References

- **Online docs (development):** https://petsc.org/main/overview/
- **Online docs (release):** https://petsc.org/release/overview/
- **Style and conventions:** `doc/developers/style.md` — read this before modifying code or reviewing MRs
- **Testing system:** `doc/developers/testing.md` — test block syntax (`/*TEST ... TEST*/`)
- **MR process:** `doc/developers/contributing/`
- **Formatting config:** `.clang-format` in repo root

## Common Commands

- `make clangformat` — auto-format code
- `make checkclangformat` — check formatting (CI runs this)
- `make checkbadSource` — check style rules (CI runs this)
- `make test search='<pattern>'` — run specific tests
- `make alltests TIMEOUT=600` — run full test suite
- `make branch-review` — run AI-assisted code review on the current branch

## GitLab CI

- Config: `.gitlab-ci.yml`
- Stages: `.pre` (checks) → `stage-1` (quick builds) → `stage-2` (full tests) → `stage-3` (extensive) → `stage-4` (coverage analysis) → `.post` (coverage review, pipeline analysis)
- MRs trigger pipelines automatically; `docs-only` label skips build/test jobs
