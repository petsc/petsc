---
name: petsc-build
description: >-
  Build PETSc libraries or petsc4py in a user-supplied PETSC_ARCH, regenerate Fortran bindings
  after public API changes, and diagnose compilation or linking failures. Use for development
  builds; documentation builds are covered by petsc-docs.
---

# Build PETSc and petsc4py

Run PETSc commands from the repository root; `arch-name` is the user-supplied architecture.
Use its existing configuration. For configuration changes, follow
[petsc-configure](../petsc-configure/SKILL.md).

## PETSc libraries and complete builds

During iteration, build the libraries:

```console
$ make -f makefile PETSC_ARCH=arch-name libs
```

Keep `-f makefile` to use the configured make, parallelism, and load limits through
`lib/petsc/conf/rules_doc.mk`; newer GNU Make versions can send `make libs` directly to
`gmakefile`. The library target suffices for library-only work and targeted-test prerequisites.

A request to build the existing configuration includes its configured post-build
packages (`self.builtafterpetsc = 1` under `config/BuildSystem/config/packages/`). After editing,
generation, and formatting, finish that request with:

```console
$ make PETSC_ARCH=arch-name all
```

This needs no separate approval. Do not repeat `all` during iteration unless changing a
post-build package. For petsc4py-only builds, use the petsc4py section below.

## Fortran bindings

If a public C header or API change affects a generated Fortran interface and the configuration
enables those bindings, regenerate them before building:

```console
$ make PETSC_ARCH=arch-name fortranbindings
```

## petsc4py

Before the first bindings build, run the library command above unless the libraries have already
been rebuilt from the current checkout in this task or the user confirms they are current.
Rebuild after further PETSc source changes affecting the bindings.

From `src/binding/petsc4py/`, using the same architecture:

```console
$ PETSC_ARCH=arch-name CFLAGS='-O0' python setup.py build
```

Disable C optimization for development unless the user requests it: Cython generates a large
source file that is slow to optimize. Use the same Python interpreter and environment for
building and testing.

## Diagnose failures

Inspect the first relevant compiler or linker error and the build log under
`arch-name/lib/petsc/conf/`; for petsc4py, inspect `setup.py build` output too. Distinguish source
errors from missing configuration capabilities. Report the failing command and error; do not
switch architectures to obtain a passing build.
