.. _tut_install:

====================
Quick Start Tutorial
====================

QQTW (Quickest Quick-start in The West)
=======================================

On systems where MPI and `BLAS/LAPACK <https://www.netlib.org/lapack/lug/node11.html>`__
are installed, :ref:`download <doc_download>` PETSc and build with:

.. code-block:: console

   $ ./configure
   $ make all check

Or to specify compilers and have PETSc download and install `MPICH
<https://www.mpich.org/>`__ and `BLAS/LAPACK
<https://www.netlib.org/lapack/lug/node11.html>`_ [#blas]_ (when they are not already on
your machine):

.. code-block:: console

   $ ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-fblaslapack
   $ make all check

Don't need Fortran? Use ``--with-fortran-bindings=0`` to reduce the build times. If you
are not using :ref:`external packages <doc_externalsoftware>` that use Fortran (for
example, `MUMPS <http://mumps.enseeiht.fr/>`__ requires Fortran) you can use
``--with-fc=0`` for even faster build times.

.. admonition:: Encounter problems?

   #. Read the error message from ``configure``!
   #. Read help ``./configure --help``.
   #. Refer to :ref:`configuration faq <doc_config_faq>` (e.g. build PETSc without a
      Fortran compiler).
   #. ``make`` problems? Just copy/paste ``make`` command printed by the ``configure``
       footer including any ``$PETSC_DIR`` and ``$PETSC_ARCH`` options. It may look
       similar to:

      ::

         xxx=========================================================================xxx
         Configure stage complete. Now build PETSc libraries with:
         make PETSC_DIR=/Users/jacobfaibussowitsch/NoSync/petsc PETSC_ARCH=arch-darwin-c-debug all
         xxx=========================================================================xxx

   #. Check the :ref:`bug-reporting <doc_creepycrawly>` section.

------------------

.. _tut_install_prereq:

Prerequisites
=============
.. important::

   This tutorial assumes basic knowledge on the part of the user on how to
   navigate your system using the Command-Line Interface (CLI), a.k.a. "from the
   terminal". Being a programmable solver suite, PETSc does not have a
   front-end Graphical User Interface, so any and all tutorial examples here will
   almost exclusively use the terminal.

   While this tutorial will provide all commands necessary, it will not explain the usage
   or syntax of commands not directly implemented by PETSc. If you are unfamiliar with the
   command line, or would like to refresh your understanding, consider reviewing tutorials
   on basic `UNIX <https://www.tutorialspoint.com/unix/index.htm>`__ and `shell
   <https://www.tutorialspoint.com/unix/shell_scripting.htm>`__ usage.


Before beginning, make sure you have the following pre-requisites installed and up to
date:

- `make <https://www.gnu.org/software/make/>`__

- `python3 <https://www.python.org/>`__ [#]_

- C Compiler (e.g. `gcc <https://gcc.gnu.org/>`__ or `clang <https://clang.llvm.org/>`__)

- [OPTIONAL] Fortran Compiler (e.g. `gfortran <https://gcc.gnu.org/wiki/GFortran>`__)

- [OPTIONAL] `git <https://git-scm.com/>`__

It is important to make sure that your compilers are correctly installed [#]_ (i.e. functional
and in your ``$PATH``). To test the compilers, run the following commands:

.. code-block:: console

   $ printf '#include<stdio.h>\nint main(){printf("cc OK!\\n");}' > t.c && cc t.c && ./a.out && rm -f t.c a.out

.. note::

   While it is recommended that you have functional C++ and Fortran compilers installed,
   they are not directly required to run PETSc in its default state. If they are
   functioning, PETSc will automatically find them during the configure stage, however it
   is always useful to test them on your own.

   .. code-block:: console

      $ printf '#include<iostream>\nint main(){std::cout<<"c++ OK!"<<std::endl;}' > t.cpp && c++ t.cpp && ./a.out && rm -f t.cpp a.out
      $ printf 'program t\nprint"(a)","gfortran OK!"\nend program' > t.f90 && gfortran t.f90 && ./a.out && rm -f t.f90 a.out

If compilers are working, each command should print out ``<compiler_name> OK!`` on the command
line.

.. _tut_install_download:

Downloading Source
==================

See the :ref:`download documentation <doc_download>` for additional details.

With all dependencies installed, navigate to a suitable directory on your machine and pull
the latest version of the PETSc library to your machine with git. The following commands
will create a directory "petsc" inside the current directory and retrieve the latest
release branch of the repository.

.. code-block:: console

   $ mkdir ~/projects
   $ cd ~/projects
   $ git clone -b release https://gitlab.com/petsc/petsc
   $ cd petsc

.. note::

  If git is not available - or if pre-generated Fortran stubs are required (i.e avoid download and
  install of sowing package - that also requires a C++ compiler) one can download a release tarball.
  See :ref:`download documentation <doc_download>` for additional details.

.. Warning::

   It is **IMPERATIVE** to install PETSc in a directory whose path does not contain any of
   the following special characters:

   ~ ! @ # $ % ^ & * ( ) ` ; < > ? , [ ] { } ' " | (including spaces!)

   While PETSc is equipped to handle these errors, other installed dependencies may not be
   so well protected.

The download process may take a few minutes to complete. Successfully running this command
should yield a similar output:

.. code-block:: console

   $ git clone -b release https://gitlab.com/petsc/petsc.git petsc
   Cloning into 'petsc'...
   remote: Enumerating objects: 862597, done.
   remote: Counting objects: 100% (862597/862597), done.
   remote: Compressing objects: 100% (197622/197622), done.
   remote: Total 862597 (delta 660708), reused 862285 (delta 660444)
   Receiving objects: 100% (862597/862597), 205.11 MiB | 3.17 MiB/s, done.
   Resolving deltas: 100% (660708/660708), done.
   Updating files: 100% (7748/7748), done.
   $ cd petsc
   $ git pull # Not strictly necessary, but nice to check
   Already up to date.

.. _tut_install_config:

Configuration
=============

See :ref:`install documentation <doc_config_faq>` for more details.

Next, PETSc needs to be configured using ``configure`` for your system with your
specific options. This is the stage where users can specify the exact parameters to
customize their PETSc installation. Common configuration options are:

- :ref:`Specifying different compilers. <doc_config_compilers>`

- :ref:`Specifying different MPI implementations. <doc_config_mpi>`

- Enabling `CUDA <https://developer.nvidia.com/cuda-toolkit>`__/`OpenCL
  <https://www.khronos.org/opencl/>`__/`ViennaCL <http://viennacl.sourceforge.net/>`__
  :ref:`support. <doc_config_accel>`

- :ref:`Specifying options <doc_config_blaslapack>` for `BLAS/LAPACK
  <https://www.netlib.org/lapack/lug/node11.html>`__.

- :ref:`Specifying external packages <doc_config_externalpack>` to use or download
  automatically. PETSc can automatically download and install a wide range of other
  supporting software.

- Setting various known machine quantities for PETSc to use such as known integral sizes,
  memory alignment, or additional compiler flags.

.. important::

   You MUST specify all of your configuration options at this stage. In order to enable
   additional options or packages in the future, you will have to reconfigure your PETSc
   installation in a similar manner with these options enabled.

   For a full list of available options call

   .. code-block:: console

      $ ./configure --help

All PETSc options and flags follow the standard CLI formats ``--option-string=<value>`` or
``--option-string``, where ``<value>`` is typically either ``1`` (for true) or ``0`` (for
false) or a directory path. Directory paths must be absolute (i.e. full path from the root
directory of your machine), but do accept environment variables as input.

From ``$PETSC_DIR`` call the following ``configure`` command to configure PETSc as well
as download and install `MPICH <https://www.mpich.org/>`__ and a `BLAS/LAPACK
<https://www.netlib.org/lapack/lug/node11.html>`__ [#blas]_ `reference implementation
<https://bitbucket.org/petsc/pkg-fblaslapack/src/master/>`__ on your system.

.. code-block:: console

   $ ./configure --download-mpich --download-fblaslapack

PETSc will begin configuring and printing its progress. A successful ``configure`` will
have the following general structure as its output:

.. code-block:: text

   ===============================================================================
             Configuring PETSc to compile on your system
   ===============================================================================
   TESTING: configureSomething from PETSc.something(config/PETSc/configurescript.py:lineNUM)
   ===============================================================================
             Trying to download MPICH_DOWNLOAD_URL for MPICH
   ===============================================================================
   ===============================================================================
             Running configure on MPICH; this may take several minutes
   ===============================================================================
   ===============================================================================
	     Running make on MPICH; this may take several minutes
   ===============================================================================
   ===============================================================================
             Running make install on MPICH; this may take several minutes
   ===============================================================================
   ===============================================================================
             Trying to download FBLASLAPACK_URL for FBLASLAPACK
   ===============================================================================
   ===============================================================================
             Compiling FBLASLAPACK; this may take several minutes
   ===============================================================================
   ===============================================================================
             Trying to download SOWING_DOWNLOAD_URL for SOWING
   ===============================================================================
   ===============================================================================
             Running configure on SOWING; this may take several minutes
   ===============================================================================
   ===============================================================================
             Running make on SOWING; this may take several minutes
   ===============================================================================
   ===============================================================================
             Running make install on SOWING; this may take several minutes
   ===============================================================================
   Compilers:
     C Compiler:   Location information and flags
     C++ Compiler: Location information and flags
   .
   .
   .
   MPI:
        Includes: Include path
   Other Installed Packages:
   .
   .
   .
   PETSc:
        PETSC_ARCH: {YOUR_PETSC_ARCH}
        PETSC_DIR:  {YOUR_PETSC_DIR}
   .
   .
   .
   .

   xxx=========================================================================xxx
   Configure stage complete. Now build PETSc libraries with (gnumake build):
   make PETSC_DIR=/your/petsc/dir PETSC_ARCH=your-petsc-arch all
   xxx=========================================================================xxx

.. _tut_install_compile:

Compilation
===========

After successfully configuring, build the binaries from source using the ``make``
command. This stage may take a few minutes, and will consume a great deal of system
resources as the PETSc is compiled in parallel.

.. code-block:: console

   $ make all check

A successful ``make`` will provide an output of the following structure:

.. code-block:: text

   -----------------------------------------
   PETSC_VERSION_RELEASE
   .
   .
   .
   -----------------------------------------
   #define SOME_PETSC_VARIABLE
   .
   .
   .
   -----------------------------------------
   Installed Compiler, Package, and Library Information
   .
   .
   .
   =========================================
          FC arch-darwin-c-debug/obj/sys/f90-mod/petscsysmod.o
          FC arch-darwin-c-debug/obj/sys/fsrc/somefort.o
          FC arch-darwin-c-debug/obj/sys/f90-src/fsrc/f90_fwrap.o
          CC arch-darwin-c-debug/obj/sys/info/verboseinfo.o
          CC arch-darwin-c-debug/obj/sys/info/ftn-auto/verboseinfof.o
          CC arch-darwin-c-debug/obj/sys/info/ftn-custom/zverboseinfof.o
	  .
	  .
	  .
	  FC arch-darwin-c-debug/obj/snes/f90-mod/petscsnesmod.o
          FC arch-darwin-c-debug/obj/ts/f90-mod/petsctsmod.o
          FC arch-darwin-c-debug/obj/tao/f90-mod/petsctaomod.o
     CLINKER arch-darwin-c-debug/lib/libpetsc.PETSC_MAJOR.PETSC_MINOR.PETSC_PATCH.dylib
    DSYMUTIL arch-darwin-c-debug/lib/libpetsc.PETSC_MAJOR.PETSC_MINOR.PETSC_PATCH.dylib
   gmake[2]: Leaving directory '/your/petsc/dir'
   gmake[1]: Leaving directory '/your/petsc/dir'
   =========================================
   Running test examples to verify correct installation
   Using PETSC_DIR=/your/petsc/dir and PETSC_ARCH=your-petsc-arch
   C/C++ example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process
   C/C++ example src/snes/examples/tutorials/ex19 run successfully with 2 MPI processes
   Fortran example src/snes/examples/tutorials/ex5f run successfully with 1 MPI process
   Completed test examples

.. _tut_install_fin:

Congratulations!
================

You now have a working PETSc installation and are ready to start using the library!

.. rubric:: Footnotes

.. [#] It is possible to configure PETSc using python2, however support for python2 will be
   discontinued in the future and so we recommend that users do not configure their PETSc
   installations using it.

.. [#] Should you be missing any of these dependencies or would like to update them, either
   download and install the latest versions from their respective websites, or use your
   preferred package manager to update them. For example on macOS using the package manager
   `homebrew <https://brew.sh/>`__ to install `python3 <https://www.python.org/>`__

.. code-block:: console

   $ brew update
   $ brew list            # Show all packages installed through brew
   $ brew upgrade         # Update packages already installed through brew
   $ brew install python3

.. [#blas] The `BLAS/LAPACK <https://www.netlib.org/lapack/lug/node11.html>`__ package
   installed as part of this tutorial is a `reference implementation
   <https://bitbucket.org/petsc/pkg-fblaslapack/src/master/>`__ and a suitable starting
   point to get PETSc running, but is generally not as performant as more optimized
   libraries. See the :ref:`libaray guide <ch_blas-lapack_avail-libs>` for further
   details.
