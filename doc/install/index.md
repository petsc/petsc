(doc_install)=

# Install

:::{note}
PETSc is available from multiple package managers.
Depending on your exact needs (for example limited use of {any}`external packages <doc_externalsoftware>`) they are possibly the easiest way for
you to install PETSc.
Always verify that the package manager is providing a recent enough release of PETSc with support for the external packages you need.
Some package managers provide separate packages for the complex number installation of PETSc.

- Archlinux <https://aur.archlinux.org/packages/petsc>
- Conda: <https://anaconda.org/conda-forge/petsc>
  : `conda install -c conda-forge petsc`
- Debian: <https://packages.debian.org/petsc-dev>
  : `sudo apt install petsc-dev`
- Fedora: <https://packages.fedoraproject.org/pkgs/petsc/petsc>
  : `sudo yum install petsc-mpich-devel`
- Homebrew: <https://formulae.brew.sh/formula/petsc>
  : `brew install petsc`
- MacPorts: <https://ports.macports.org/port/petsc>
  : `sudo port install petsc`
- MSYS2 (Windows) <https://packages.msys2.org/package/mingw-w64-x86_64-petsc>
- openSUSE <https://software.opensuse.org/package/petsc>
- Python: <https://pypi.org/project/petsc>
  : `python -m pip install petsc petsc4py`
- Slackware: <https://slackbuilds.org/repository/15.0/academic/petsc/?search=petsc>
- Spack: <https://spack.io>
  : - debug install - `spack install petsc +debug`
    - optimized install -`spack install petsc cflags='-g -O3 -march=native -mtune=native' fflags='-g -O3 -march=native -mtune=native'  cxxflags='-g -O3 -march=native -mtune=native'`
    - install with some external packages - `spack install petsc +superlu-dist +metis +hypre +hdf5`
    - list available variants (configurations) - `spack info petsc`
- Ubuntu: <https://packages.ubuntu.com/petsc-dev>
  : `sudo apt install petsc-dev`
:::

Information and tutorials on setting up a PETSc installation.

```{toctree}
:maxdepth: 2

download
install_tutorial
install
windows
multibuild
external_software
license
```
