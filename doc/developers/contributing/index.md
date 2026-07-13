(ch_contributing)=

# Contributing to PETSc

As you gain experience in building, using, and debugging with PETSc, you
will be able to contribute!

Before contributing code to PETSc, please read {any}`style`. You may also
be interested to read about {any}`design`.

PETSc uses [Git](https://git-scm.com/), [GitLab](https://gitlab.com/petsc/petsc),
and its testing system, for its source code management.
All new code in PETSc is accepted via merge requests (MRs).

By submitting code, the contributor gives irretrievable consent to the
redistribution and/or modification of the contributed source code as
described in the [PETSc open-source license](https://gitlab.com/petsc/petsc/-/blob/main/CONTRIBUTING).

## How-Tos

Some of the source code is documented to provide direct examples/templates for common
contributions, adding new implementations for solver components:

- [Add a new PC type](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/pc/impls/jacobi/jacobi.c)
- [Add a new KSP type](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/impls/cg/cg.c)
- [Add a new subclass of a matrix type (implementation inheritance)](https://gitlab.com/petsc/petsc/-/blob/main/src/mat/impls/aij/seq/superlu/superlu.c)

## Contributing functionality from an external package

In addition to contributing code directly to the PETSc repository, developers may choose to layer (portions of)
their software that lives in its own repository on top of the PETSc API, for example providing their own `PCType`.
An example of this is [PFLARE](https://github.com/PFLAREProject/PFLARE), which is obtained with `--download-pflare`
and registers the `PCAIR` and `PCPFLAREINV` preconditioners.

Such a package can also contribute its own PETSc manual pages, generated from the docstrings in its own sources.
A package opts in by setting `self.providesDocs = 1` and `self.docsDirs` (the subdirectories that contain the
documented sources, for example `['src', 'include']`) in its `config/BuildSystem/config/packages/<package>.py`.
When the documentation is built, the sources of each such package are cloned at their pinned `self.gitcommit` and
scanned for the usual PETSc docstring blocks (`/*@ @*/`, `/*MC M*/`, `/*E E*/`), producing manual pages alongside
the PETSc ones. See `config/BuildSystem/config/packages/PFLARE.py` for a prototypical example.

(sec_git)=

(sec_setup_git)=

## Setting up Git

We provide some information on common operations here; for more details, see `git help`, `man git`, or [the Git book](https://git-scm.com/book/en/).

- [Install Git](https://git-scm.com/downloads) if it is not already installed on your machine, then see below to obtain PETSc.
- [Set up your Git environment](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) to establish your identity.
- To stay oriented when working with branches, we encourage configuring
  [git-prompt](https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh).
- To get tab-completion for Git commands, one can download and then source
  [git-completion.bash](https://raw.github.com/git/git/master/contrib/completion/git-completion.bash).

## Obtaining the development version of PETSc

- If you have write access to the PETSc [GitLab repository](https://gitlab.com/petsc/petsc), use `git clone git@gitlab.com/petsc/petsc`
  (or use a clone you already have).

- Otherwise, [Create a fork](https://gitlab.com/petsc/petsc/-/forks/new) (your own copy of the PETSc repository).

  - You will be asked to "Select a namespace to fork the project"; click the green "Select" button.

  - If you already have a clone on your machine of the PETSc repository you would like to reuse

    ```console
    $ git remote set-url origin git@gitlab.com:YOURGITLABUSERNAME/petsc.git
    ```

  - otherwise

    ```console
    $ git clone git@gitlab.com:YOURGITLABUSERNAME/petsc.git
    ```

PETSc can now be configured as specified on the
[Installation page](https://petsc.org/release/install/)

To update your copy of PETSc

```console
$ git pull
```

Once updated, you will usually want to rebuild it completely

```console
$ make reconfigure all
```

This is equivalent to

```console
$ $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py && make all
```

```{toctree}
:maxdepth: 1

developingmr
submittingmr
pipelines
```
