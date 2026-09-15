---
name: petsc-lint
description: >-
  Format PETSc C/C++ and handwritten Fortran, and run checks for C docstrings, Python configure
  code, and petsc4py. Also explain CI checks for manual-page metadata, shell scripts, and
  petsclinter. Use for source formatting, C docstring changes, or check failures. Documentation
  builds are covered by petsc-docs.
---

# Lint and format PETSc and petsc4py

Run from the repository root unless stated otherwise; `arch-name` is the user-supplied
architecture. Apply only checks relevant to the changed files.

The repository-root `make lint` runs PETSc's C/C++ source and docstring linter. The `make lint`
target in `src/binding/petsc4py/` runs the bindings' Cython and Ruff checks. Select the directory
and target according to the source being checked.

## C and C++

For files in PETSc's C/C++ formatting set, use `.clang-format` and restrict the make target to
changed tracked files with `GITCFSRC` pathspecs, for example:

```console
$ make clangformat GITCFSRC='src/ksp/ksp/interface/itfunc.c'
```

Inspect only the `checkclangformatversion`, `GITCFSRC`, `GITSRCEXCL`, and `GITCFSRCEXCL`
definitions in `lib/petsc/conf/rules_util.mk`. Reuse these details while the definitions are
unchanged and remain in context. Select the required clang-format major version with
`PETSCCLANGFORMAT` if needed:

```console
$ make PETSCCLANGFORMAT=/path/to/clang-format clangformat GITCFSRC='src/ksp/ksp/interface/itfunc.c'
```

`GITCFSRC` replaces the whole pathspec, including the exclusions in `GITSRCEXCL` and
`GITCFSRCEXCL` (for example `src/binding/`, `khash`, `yaml`, `finclude`); never name files
from those locations. The target uses `git ls-files`, so format new untracked files directly with
the same executable. Without `GITCFSRC`, it processes the entire formatting set. Preserve existing edits and inspect
both the diff and output: the recipe ignores formatter errors. If the required version is
unavailable, report it and the formatting gap.

## Handwritten Fortran

Run `make fprettify` for changes to handwritten `.h90` or `.F90` files. It requires `fprettify`
and reformats all tracked files with those suffixes. Inspect the diff and preserve unrelated
edits.

## C docstrings

The repository-root `make lint` invokes `petsclinter`. For C docstring changes, follow the
[docstring conventions](../../conventions/c-docstrings.md) and run a local check of the
affected directory when dependencies are available. The `linux-analyzer` CI job checks `./src`
with `--werror 1`, so unresolved warnings there fail CI. Local lint is not required for every
source change. For a scoped check, use:

```console
$ make PETSC_ARCH=arch-name lint DIRECTORY=src/ksp/ksp/interface LINTER_OPTIONS="--werror 1"
```

It requires the Python `clang` package and a compatible `libclang`. Inspect `help-lint` and the
requirements under `lib/petsc/bin/maint/petsclinter/` when dependency or invocation details are
needed. If the local check is skipped, report that verification gap and inspect the changed
docstrings against the conventions above; identify unavailable dependencies when relevant.

For Markdown and Sphinx documentation work, follow [petsc-docs](../petsc-docs/SKILL.md).

## Python configure code

For Python under `config/`, run `make PETSC_ARCH=arch-name vermin`. PETSc does not impose other
general formatter or linter rules there. If `vermin` is unavailable, report that instead of
substituting an unrelated policy.

## CI checks for reference

Run these checks, or equivalent commands, only when specifically requested. They describe CI
behavior, not mandatory local verification.

- `make checkfprettifyformat`: The `checksource` job requires a clean tracked working tree,
  runs `make fprettify`, and fails if formatting changes it.
- `make checkbadSource`: The `checksource` job checks source conventions defined in
  `lib/petsc/conf/rules_util.mk` and reports violations.
- `make checkbadManualPages`: The `checksource` job invokes `lib/petsc/bin/getAPI.py` to check
  manual-page metadata, including `.seealso:` formatting. It does not build documentation.
- `make checkshellcheck`: The `checksource` job applies ShellCheck's suggested patches to the
  scripts selected by the root makefile, then fails if the tracked working tree differs from
  `HEAD`. This target modifies files.
- `make test-lint`: The `linux-analyzer` job runs Vermin, MyPy, package consistency checks, and
  petsclinter's regression tests.

## petsc4py

Before concluding a petsc4py code change, run from `src/binding/petsc4py/`:

```console
$ make PETSC_ARCH=arch-name lint
```

This target recreates `petsc4py-lint-env`, installs the lint requirements, and runs Cython and
Ruff checks. Inspect failures before applying fixes; report unavailable tools or dependencies.
All new petsc4py code must be documented. A documentation build is a separate task covered by
[petsc-docs](../petsc-docs/SKILL.md).
