(doc_download)=

# Download

## Recommended: Obtain Release Version With Git

Use `release` branch from PETSc git repository - it provides the latest release with additional crucial bug fixes.

```console
$ git clone -b release https://gitlab.com/petsc/petsc.git petsc
$ git pull # obtain new release fixes (since a prior clone or pull)
```

To anchor to a release version (without intermediate fixes), use:

```console
$ git checkout vMAJOR.MINOR.PATCH
```

We recommend users join the official PETSc {ref}`mailing lists <doc_mail>` to submit
any questions they may have directly to the development team, to be notified of new
releases, or to simply keep up to date with the current state of the
library.

## Alternative: Obtain Release Version with Tarball

Tarball which contains only the source. Documentation available [online](https://petsc.org/release).

- [petsc-3.23.2.tar.gz](https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.23.2.tar.gz)

Tarball which includes all documentation, recommended for offline use.

- [petsc-with-docs-3.23.2.tar.gz](https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-with-docs-3.23.2.tar.gz)

Tarball to enable a separate installation of petsc4py.

- [petsc4py-3.23.2.tar.gz](https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc4py-3.23.2.tar.gz)

To extract the sources use:

```console
$ tar xf petsc-<version number>.tar.gz
```

Current and older release tarballs are available at:

- [Primary server](https://web.cels.anl.gov/projects/petsc/download/release-snapshots/)

:::{Note}
Older release tarballs of PETSc should only be used for
applications that have not been updated to the latest release. We urge you, whenever
possible, to upgrade to the latest version of PETSc.
:::

## Advanced: Obtain PETSc Development Version With Git

Improvements and new features get added to `main` branch of PETSc Git repository. To obtain development sources, use:

```console
$ git clone https://gitlab.com/petsc/petsc.git petsc
```

or if you already have a local clone of PETSc Git repository

```console
$ git checkout main
$ git pull
```

More details on contributing to PETSc development are at {any}`ch_contributing`. The development version of
the documentation, which is largely the same as the release documentation is [available](https://petsc.org/main).

(doc_releaseschedule)=

## Release Schedule

We provide new releases every 6 months, and patch updates to the current release every month.

Releases (for example: 3.20.0, 3.21.0, etc. with corresponding Git tags v3.20.0, v3.21.0, etc):

- March (end of the month)
- September (end of the month)

Patch updates (for example: 3.21.1, 2.21.2, etc. with corresponding Git tags v3.21.1, v3.21.2, etc)
contain the latest release plus crucial bug fixes since that release:

- Last week of every month (or first week on next month - if delayed)

The monthly updates do not contain new features or any development work since the release, they merely contain crucial
bug fixes.

The ordering of PETSc branches and tags, as of May 2024 is given by (each level also contains the commits below it):

- May (features added since v3.21.0) main branch
- May (bug fixes since v3.21.1) release branch
- April end (bug fixes since v3.21.0) v3.21.1 tag and tarball
- March end (features added after v3.20.0) v3.21.0 tag and tarball
- March end (bug fixes since v3.20.5) v3.20.6 tag and tarball
- etc
- October end (bug fixes since v3.20.0) v3.20.1 tag and tarball
- September end (features added after v3.19.0) v3.20.0 tag and tarball

After a new release of PETSc, the old version no longer gets patch updates. I.e., when 3.22.0 is released, bug fixes
will go to 3.22.x - and petsc-3.21, petsc-3.20, etc., will not get any additional patch updates.

PETSc does not follow **Semantic Versioning**, {cite}`semver-webpage`, rather it follows:

- MAJOR version, a major reorganization. Unlikely to change in foreseeable future.
- MINOR version, with new functionality and likely small API changes; most changes are backward compatible with deprecation. On a 6 month cycle.
- PATCH version, with bug fixes - and minor functionality updates preserving the current API. On a monthly cycle.

PETSc provides tools to allow you to stipulate what versions of PETSc it works with at configure time, compile time, or runtime of your package, see
{any}`ch_versionchecking`.

```{rubric} References
```

```{bibliography} /petsc.bib
:filter: docname in docnames
```
