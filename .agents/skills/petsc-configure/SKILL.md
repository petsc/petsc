---
name: petsc-configure
description: >-
  Configure or reconfigure PETSc, including the PETSc configuration used by petsc4py.
  Use when selecting configure options, changing a user-supplied PETSC_ARCH, or diagnosing
  configure failures. Also use for configure command proposals that should not be executed.
---

# Configure PETSc

Run from the repository root; `arch-name` is the user-supplied architecture.

## Choose and run configure

Consult `./configure --help` when option details are needed. For a command proposal or plan,
state assumptions and provide the command without executing it or adding build commands;
an architecture placeholder suffices for a proposal.

Use the requested or approved flags and packages. Ask about unresolved choices only when they
materially affect the configuration, such as the compiler, MPI implementation, or accelerator.
For new configurations, otherwise use configure's defaults and the development settings below.
Preserve existing settings when reconfiguring, and do not add unrequested packages.

Unless the user requests otherwise, enable debugging and strict `PetscErrorCode` checking in a
new development configuration:

```console
$ ./configure PETSC_ARCH=arch-name --with-debugging=1 --with-strict-petscerrorcode=1 [user-requested options]
```

Fortran bindings default to enabled when a suitable compiler is found.

Configure normally sanitizes exported compiler and tool variables. Pass supported values as
arguments, such as `CC`, `CXX`, `FC`, `CFLAGS`, `CXXFLAGS`, `FFLAGS`, `LDFLAGS`, or `LIBS`.
Select MPI with `--with-mpi-dir` and build parallelism with `--with-make-np`; not every sanitized
environment variable is a configure option.

```console
$ ./configure PETSC_ARCH=arch-name CC=mpicc CFLAGS='-g -O0' [other options]
```

On failure, inspect `arch-name/lib/petsc/conf/configure.log`. If it was not created or updated,
use this run's output and repository-root `configure.log`, if present. Report the relevant error
and command; do not retry with speculative options.

## Reconfigure

Use `./configure` with the complete updated options instead of `make reconfigure`. Recover the
previous options from `arch-name/lib/petsc/conf/reconfigure-*.py` when available, preserving
settings outside the requested change. Ask before removing an existing architecture directory
to resolve library conflicts.

`config/examples/*.py` primarily defines CI configurations. Consult these files for package
knowledge; execute one only to reproduce its configuration or when requested.

Report capabilities missing from the supplied architecture. Adding packages or changing scalar
type, precision, index size, language support, MPI implementation, or accelerator backend changes
the configuration.
