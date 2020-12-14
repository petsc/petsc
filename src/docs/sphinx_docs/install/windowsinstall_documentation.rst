
.. include:: <isonum.txt>

.. _doc_windows:

*************************************
Installing PETSc On Microsoft Windows
*************************************

.. admonition:: Are You Sure?
   :class: yellow

   Are you sure you want to use Microsoft Windows? We recommend using Linux if possible
   (and minimize troubleshooting Microsoft Windows related issues).

Recommended Installation Methods
================================

The following configurations are much like regular Unix-like systems. Our regular
(Unix-like) instructions should work with them. Most :ref:`external packages
<doc_externalsoftware>` will also work. The ``configure`` option ``--download-mpich``
should work for these systems. Note however that these **do not** support Microsoft/Intel
Windows compilers; nor can you use MS-MPI, Intel-MPI or MPICH2).

- `Cygwin <https://www.cygwin.com/>`__ Unix emulator for Microsoft Windows. See the
  instructions below for installing Cygwin for PETSc.

  .. note::

     Be sure to install the GNU compilers, and commons components, **do not** use the
     ``win32fe`` [#win32]_ script:

     - python3
     - make
     - gcc-core gcc-g++ gcc-fortran
     - liblapack-devel
     - openmpi libopenmpi-devel

- Microsoft Windows Subsystem for Linux 2 (`WLS2
  <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__). Largely untested, let
  us know your experience. Be sure to use WSL2 vs WSL1 for best performance.

- `Docker <https://docs.docker.com/docker-for-windows/>`__ for Microsoft
  Windows. Untested, let us know your experience.

- `Git bash <https://www.atlassian.com/git/tutorials/git-bash>`__. You must also install
  the GNU compilers via `MinGW
  <https://yichaoou.github.io/tutorials/software/2016/06/28/git-bash-install-gcc>`__. Untested,
  let us know your experience.

- Linux virtual machine via `VirtualBox <https://www.virtualbox.org/>`__. One sample
  tutorial is at https://www.psychocats.net/ubuntu/virtualbox. Google can provide more
  tutorials. Untested, let us know your experience.

Installation With Microsoft/Intel Windows Compilers
===================================================

Microsoft Windows does not provide the same unix shell enviornment as the other OSes. Also
the default Microsoft/Intel compilers behave differently than other unix compilers. So to
install PETSc on Microsoft Windows - one has to install Cygwin (for the unix enviornment)
and use ``win32fe`` [#win32]_ (located at ``$PETSC_DIR/lib/petsc/bin/win32fe``, to
interface to Microsoft/Intel compilers).

#. Install Cygwin:

   Please download and install Cygwin package from http://www.cygwin.com and make sure the
   following components are installed:

   .. note::

      Make sure the following Cygwin components are installed:

      - python3
      - make
      - (default selection should already have diff and other tools)

      Additional cygwin components like git cmake can be useful for installing
      :ref:`external packages <doc_externalsoftware>`.

#. Remove Cygwin link.exe:

   Cygwin link.exe can conflict with Intel ifort compiler. If you are using ifort - please
   do (from Cygwin terminal/bash-shell):

   .. code-block:: console

      > mv /usr/bin/link.exe /usr/bin/link-cygwin.exe

#. Setup Cygwin terminal/bash-shell with Working Compilers:

   We require the compilers to be setup properly in a Cygwin bash command shell, so that
   ``cl foo.c or ``ifort foo.f`` works from this shell. For example - if using VS2005 C
   and Intel 10 Fortran one can do:

   #. Start |rarr| Programs |rarr| Intel Software Development Tools |rarr| Intel Fortran
      Compiler 10 |rarr| Visual Fortran Build Enviornment (32bit or 64bit depending on
      your usage). This should start a "dos cmd" shell.

   #. Within this shell - run Cygwin terminal/bash-shell ``mintty.exe`` as:

      .. code-block:: console

         > C:\cygwin\bin\mintty.exe

   #. Verify if the compilers are useable (by running cl, ifort in this Cygwin
      terminal/bash-shell).

   #. Now run ``configure`` with ``win32fe`` [#win32]_ and then build the libraries with
      make (as per the usual instructions)

Example Configure Usage With Microsoft Windows Compilers
--------------------------------------------------------

Use ``configure`` with VC2005 C and Intel Fortran 10 (without MPI):

.. code-block:: console

   > ./configure --with-cc='win32fe cl' --with-fc='win32fe ifort' --with-cxx='win32fe cl' --with-mpi=0 --download-fblaslapack

If fortran, c++ usage is not required, use:

.. code-block:: console

   > ./configure --with-cc='win32fe cl' --with-fc=0 --with-cxx=0 --download-f2cblaslapack

Using MPI
^^^^^^^^^

We support both MS-MPI [64-bit] and Intel MPI on Microsoft Windows (MPICH2 does not work,
do not use it). For example usages, check ``$PETSC_DIR/config/examples/arch-mswin*.py``

.. warning::

   **Avoid spaces in $PATH**

   Its best to avoid spaces or similar special chars when specifying ``configure`` options. On
   Microsoft Windows - this usually affects specifying MPI or MKL. Microsoft Windows
   supports dos short form for dir names - so its best to use this notation. Cygwin
   tool ``cygpath`` can be used to get paths in this notation. For example:

   .. code-block:: console

      > cygpath -u `cygpath -ms '/cygdrive/c/Program Files (x86)/Microsoft SDKs/MPI'`
      /cygdrive/c/PROGRA~2/MICROS~2/MPI
      > cygpath -u `cygpath -ms '/cygdrive/c/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64'`
      /cygdrive/c/PROGRA~2/INTELS~1/COMPIL~2/windows/mkl/lib/intel64

   The use in ``configure`` as follows:

   .. code-block:: console

      > ./configure --with-cc='win32fe cl' --with-fc='win32fe ifort' --with-cxx='win32fe cl' \
      --with-shared-libraries=0 \
      --with-mpi-include='[/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include,/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/x64]' \
      --with-mpi-lib='-L/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64 msmpifec.lib msmpi.lib' \
      --with-mpiexec=/cygdrive/c/PROGRA~1/MICROS~2/Bin/mpiexec \
      --with-blaslapack-lib='-L/cygdrive/c/PROGRA~2/INTELS~1/COMPIL~2/windows/mkl/lib/intel64 mkl_intel_lp64_dll.lib mkl_sequential_dll.lib mkl_core_dll.lib'

External Packages
^^^^^^^^^^^^^^^^^

The ``--download-package`` option does not work with many :ref:`external packages
<doc_externalsoftware>` on Microsoft Windows.

Project Files
^^^^^^^^^^^^^

We cannot provide Microsoft Visual Studio project files for users as they are specific to
the ``configure`` options, location of :ref:`external packages <doc_externalsoftware>`,
compiler versions etc. used for any given build of PETSc, so they are potentially
different for each build of PETSc. So if you need a project file for use with PETSc -
please do the following.

#. Create an empty project file with one of the examples say
   ``$PETSC_DIR/src/ksp/ksp/tutorials/ex2.c``

#. Try compiling the example from Cygwin bash shell - using makefile - i.e.:

   .. code-block:: console

      > cd $PETSC_DIR/src/ksp/ksp/tutorials
      > make ex2

#. If the above works - then make sure all the compiler/linker options used by ``make``
   are also present in the project file in the correct notation.

#. If errors - redo the above step. If all the options are correctly specified, the
   example should compile from MSDev.

Debugger
^^^^^^^^

Running PETSc probrams with ``-start_in_debugger`` is not supported on this platform, so debuggers will need to be initiated manually. Make sure your environment is properly configured to use the appropriate debugger for your compiler. The debuggers can be initiated using Microsoft Visual Studio 6:

.. code-block:: console

   > msdev ex1.exe

Microsoft Visual Studio .NET:

.. code-block:: console

   > devenv ex1.exe

Intel Enhanced Debugger:

.. code-block:: console

   > edb ex1.exe

or GNU Debugger

.. code-block:: console

   > gdb ex1.exe

Using MinGW With Microsoft/Intel Windows Compilers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users report that it is possible to build to build PETSc using MinGW and link against them
using the Microsoft/Intel Windows Compilers. We have no experience with this, nor
knowledge on how it can be accomplished. If you have experience using MinGW and windows
and/or have successfully built PETSc using this configuration let us know your experience
by providing feedback at petsc-maint@mcs.anl.gov.

Notes On Using Other Systems Besides Cygwin To Compile With Microsoft/Intel Compilers
-------------------------------------------------------------------------------------

For any alternate system, we would have to redo ``win32fe`` [#win32]_ functionality for
that system. This includes:

- Marshal unix type compiler options to Cl (Microsoft compiler).
- Convert paths in some of these options from this system (for example Cygwin paths) to
  Microsoft Windows paths.
- Have python that works with system path notation.
- Have the ability equivalent to Microsoft Windows process spawning; Cygwin process
  spawning produces Microsoft Windows processes. WSL1 lacked this.

.. rubric:: Footnotes

.. [#win32] PETSc win32 front end ("``win32fe``"): This tool is used as a wrapper to Microsoft
       and Intel compilers and associated tools - to enable building PETSc libraries using
       Cygwin make and other UNIX tools. For additional info, run
       ``${PETSC_DIR}/lib/petsc/bin/win32/win32fe`` without any options.
