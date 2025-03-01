(doc_windows)=

# Installing PETSc On Microsoft Windows

:::{admonition} Are You Sure?
:class: yellow

Are you sure you want to use Microsoft Windows?

Developing HPC software is more difficult on Microsoft Windows than Linux and macOS systems.
We recommend using a Microsoft Windows system for PETSc only when necessary.
:::

There are many ways to install PETSc on Microsoft Windows's systems.

- {any}`sec_linux_on_windows`
- {any}`sec_cygwin_gnu_on_windows`
- {any}`sec_native_compilers_on_windows`
- {any}`sec_msys2_mingw_compilers_on_windows`

______________________________________________________________________

(sec_linux_on_windows)=

## Linux on Microsoft Windows

- Microsoft Windows Subsystem for Linux 2 ([WLS2](https://docs.microsoft.com/en-us/windows/wsl/install-win10)). Be sure to use WSL2 for best performance.
- [Docker](https://docs.docker.com/docker-for-windows/) for Microsoft
  Windows.
- Linux virtual machine via [VirtualBox](https://www.virtualbox.org/) or similar. One sample
  tutorial is at <https://www.psychocats.net/ubuntu/virtualbox>. Google can provide more
  tutorials.

______________________________________________________________________

(sec_cygwin_gnu_on_windows)=

## Cygwin/GNU Compilers on Microsoft Windows

Cygwin/GNU compilers allow building standalone PETSc libraries and binaries
that work on Microsoft Windows, with Cygwin pre-built libraries for BLAS, LAPACK, and Open MPI.

1. Install Cygwin:

   Download and install Cygwin from <http://www.cygwin.com> and make sure the
   following Cygwin components are installed:

   - python3
   - make
   - gcc-core gcc-g++ gcc-fortran
   - liblapack-devel
   - openmpi libopenmpi-devel libhwloc-devel libevent-devel zlib-devel

2. To build with Cygwin installed BLAS, LAPACK, and Open MPI (from default locations), do (from `Cygwin64 Terminal`):

   ```console
   $ ./configure
   ```

3. Follow the Unix instructions for any additional configuration or build options.

:::{note}
- Libraries built with Cygwin/GNU compilers are **not** compatible and cannot be linked with Microsoft or Intel compilers.
- Native libraries like MS-MPI, Intel MPI cannot be used from Cygwin/GNU compilers.
- Most {ref}`external packages <doc_externalsoftware>` are likely to work,
  however the `configure` option `--download-mpich` does not work.
:::

______________________________________________________________________

(sec_native_compilers_on_windows)=

## Native Microsoft/Intel Windows Compilers

Microsoft Windows does not provide a Unix shell environment. Also
the native Microsoft/Intel compilers behave differently than other (Unix) compilers. So to
install PETSc on Microsoft Windows with Microsoft or Intel compilers one has to install Cygwin (for the Unix environment)
and use `win32fe` [^win32] compiler wrapper, to use the Microsoft/Intel compilers.

1. Install Cygwin:

   Download and install Cygwin from <http://www.cygwin.com> and make sure the
   following Cygwin components are installed:

   - python3
   - make

   Additional Cygwin components like git and CMake can be useful for installing
   {ref}`external packages <doc_externalsoftware>`.

2. Remove Cygwin link.exe:

   Cygwin link.exe can conflict with Intel ifort compiler. If you are using ifort -
   do (from `Cygwin64 Terminal`):

   ```console
   $ mv /usr/bin/link.exe /usr/bin/link-cygwin.exe
   ```

3. Setup `Cygwin64 Terminal` with working compilers:

   We require the compilers to be setup properly in a Cygwin bash command shell, so that
   `cl foo.c` or `ifort foo.F` works from this shell. For example - if using `Visual Studio 2022 C`
   and `Intel oneAPI 2022 Fortran`, one can do:

   1. `Start` -> `Programs` -> `Intel oneAPI 2022` -> `Intel oneAPI command prompt for Intel 64 for Visual Studio 2022`.
      This should start a `DOS Command shell` with working compilers.

   2. Within this `DOS Command shell` - run `Cygwin64 Terminal`, i.e., `mintty.exe` as:

      ```powershell
      C:\cygwin64\bin\mintty.exe -
      ```

   3. Verify if the compilers are usable (by running `cl` and `ifort` in this `Cygwin64 Terminal`).

   4. Now run `configure` with compiler wrapper script `win32fe_cl` that uses `win32fe` [^win32] and then build the libraries with
      `make` (as per the usual instructions).

### Example Configure Usage With Microsoft Windows Compilers

Use `configure` with `Visual Studio 2022 C` and `Intel oneAPI 2022 Fortran` (without MPI):

```console
$ ./configure --with-cc='win32fe_cl' --with-fc='win32fe_ifort' --with-cxx='win32fe_cl' --with-mpi=0 --download-fblaslapack
```

If Fortran or C++ usage is not required, use:

```console
$ ./configure --with-cc='win32fe_cl' --with-fc=0 --with-cxx=0 --download-f2cblaslapack
```

:::{note}
- One can use Intel oneAPI C/C++ compiler `icl` or `icx` instead of Microsoft `cl`, for ex: `--with-cc=win32fe_icl --with-cxx=win32fe_icl`.
- A shorter form `--with-cc=cl` that translates to `--with-cc=win32fe_cl` is also supported. Similarly for `icl`, `icx`, `ifort`, `ifx`.
- Intel oneAPI `ifx` currently works with `--with-shared-libraries=0` only, `ifort` is recommended for default shared library build.
- The `--download-package` option may work with some {ref}`external packages <doc_externalsoftware>` and fail with most packages.
:::

### Using MPI, MKL

We support both MS-MPI (64-bit) and Intel MPI on Microsoft Windows. We also support using Intel MKL as BLAS and LAPACK implementations.
For example usages, check `$PETSC_DIR/config/examples/arch-mswin*.py`

:::{warning}
**Avoid spaces in \$PATH**

It is better to avoid spaces or similar special chars when specifying `configure` options. On
Microsoft Windows - this usually affects specifying MPI or MKL. Microsoft Windows
supports DOS short form for directory names - so it is better to use this notation. Cygwin
tool `cygpath` can be used to get paths in this notation. For example:

```console
$ cygpath -u `cygpath -ms '/cygdrive/c/Program Files (x86)/Microsoft SDKs/MPI'`
/cygdrive/c/PROGRA~2/MICROS~2/MPI
$ cygpath -u `cygpath -ms '/cygdrive/c/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64'`
/cygdrive/c/PROGRA~2/INTELS~1/COMPIL~2/windows/mkl/lib/intel64
```

Then use in `configure` as follows:

```console
$ ./configure --with-cc='win32fe_cl' --with-fc='win32fe_ifort' --with-cxx='win32fe_cl' \
--with-shared-libraries=0 \
--with-mpi-include='[/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include,/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/x64]' \
--with-mpi-lib='-L/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64 msmpifec.lib msmpi.lib' \
--with-mpiexec=/cygdrive/c/PROGRA~1/MICROS~2/Bin/mpiexec \
--with-blaslapack-lib='-L/cygdrive/c/PROGRA~2/INTELS~1/COMPIL~2/windows/mkl/lib/intel64 mkl_intel_lp64_dll.lib mkl_sequential_dll.lib mkl_core_dll.lib'
```
:::

### Project Files

We cannot provide Microsoft Visual Studio project files for users as they are specific to
the `configure` options, location of {ref}`external packages <doc_externalsoftware>`,
compiler versions etc. used for any given build of PETSc, so they are potentially
different for each build of PETSc. So if you need a project file for use with PETSc -
do the following.

1. Create an empty project file with one of the examples, say
   `$PETSC_DIR/src/ksp/ksp/tutorials/ex2.c`

2. Try compiling the example from Cygwin bash shell - using `make` - i.e.:

   ```console
   $ cd $PETSC_DIR/src/ksp/ksp/tutorials
   $ make ex2
   ```

3. If the above works - then make sure all the compiler/linker options used by `make`
   are also present in the project file in the correct notation.

4. If errors - redo the above step. If all the options are correctly specified, the
   example should compile from Microsoft Visual Studio.

______________________________________________________________________

(sec_msys2_mingw_compilers_on_windows)=

## MSYS2/MinGW (GNU) Compilers on Microsoft Windows

These allow building standalone Microsoft Windows libraries and
applications that are compatible with the Microsoft and Intel compilers.

1. Install MSYS2 and MS-MPI:

   Download and install MSYS2 from <https://www.msys2.org>.
   If you want to use MPI, we recommend you use MS-MPI from <https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi>.

2. Update MSYS2 and install base packages:

   First, launch a MSYS2 MinGW x64 shell. Double-check this is the proper type of shell by typing

   ```console
   $  echo $MINGW_PREFIX
   /mingw64
   ```

   If you see something else, e.g., `/clang64`, this is not the correct type
   of shell, it may still work, but this is less tested. Then, update your
   installation using `pacman` (you may be asked to quit and re-open your shell).

   ```console
   $  pacman -Syu
   ```

   Install the following packages that are needed
   by some PETSc dependencies.

   ```console
   $  pacman -S autoconf automake-wrapper bison bsdcpio make git \
   mingw-w64-x86_64-toolchain patch python flex \
   pkg-config pkgfile tar unzip mingw-w64-x86_64-cmake \
   mingw-w64-x86_64-msmpi mingw-w64-x86_64-openblas mingw-w64-x86_64-jq
   ```

3. Configuring:

   The two difficulties here are: 1) make sure PETSc configure picks up the proper Python installation, as there are more than one available in a MSYS2 MinGW shell and 2) tell PETSc where MS-MPI `mpiexec` is. We recommend not using shared libraries as it is easier to create standalone binaries that way.

   ```console
   $  /usr/bin/python ./configure --with-mpiexec='/C/Program\ Files/Microsoft\ MPI/Bin/mpiexec' \
   --with-shared-libraries=0
   ```

:::{note}
`MinGW` (GNU) compilers can also be installed/used via `Cygwin` (not just MSYS2).
:::

### Debugging on Microsoft Windows

Running PETSc programs with `-start_in_debugger` is not supported on Microsoft Windows. Debuggers need to be initiated manually.
Make sure your environment is properly configured to use the appropriate debugger for your compiler.
The debuggers can be initiated using Microsoft Visual Studio:

```console
$ devenv ex1.exe
```

Intel Enhanced Debugger:

```console
$ edb ex1.exe
```

or GNU Debugger

```console
$ gdb ex1.exe
```

```{rubric} Footnotes
```

[^win32]: [PETSc Win32 Development Tool Front End](https://bitbucket.org/petsc/win32fe) (`win32fe`): This tool is used as a wrapper to Microsoft
    and Intel compilers and associated tools - to enable building PETSc libraries using
    Cygwin `make` and other Unix tools. For additional info, run
    `${PETSC_DIR}/lib/petsc/bin/win32fe/win32fe --help`
