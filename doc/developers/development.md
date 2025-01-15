(development)=

# PETSc Development Environment

In the course of developing PETSc, you may find the following useful in setting up your
build and development environment.

## Influential `configure` flags

- `--with-strict-petscerrorcode`:

  This makes `PetscErrorCode` non-discardable (see `PETSC_NODISCARD`) in order to
  catch instances of missing `PetscCall()` and friends. For this reason it is *highly
  encouraged* that you `configure` with this option. CI will already have it enabled,
  doing so locally will save you the pain of re-running it.

  For the vast majority of cases (this includes C++ constructors/destructors!), you must
  fix discarded `PetscErrorCode` warnings by wrapping your call in the appropriate
  `PetscCall()` variant. If you are choosing to intentionally silence the warnings by
  ignoring the return code you may do so in the following way:

  ```
  PetscErrorCode ierr;

  ierr = SomePetscFunction(); // OK, capturing result
  (void)ierr; // to silence set-but-not-used warnings
  ```

## Editor Integrations

### Emacs

TODO

### Vim

TODO
