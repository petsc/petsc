.. _doc_config_faq:

#################
Configuring PETSc
#################

.. important::

   Obtain PETSc via the repository or download the latest tarball: :ref:`download documentation <doc_download>`.

   See :ref:`quick-start tutorial <tut_install>` for a step-by-step walk-through of the installation process.

.. contents:: Table of Contents
   :local:
   :backlinks: entry
   :depth: 1

Common Example Usages
=====================

.. attention::

   There are many example ``configure`` scripts at ``config/examples/*.py``. These cover a
   wide variety of systems, and we use some of these scripts locally for testing. One can
   modify these files and run them in lieu of writing one yourself. For example:

   .. code-block:: console

      $ ./config/examples/arch-ci-osx-dbg.py

   If there is a system for which we do not yet have such a ``configure`` script and/or
   the script in the examples directory is outdated we welcome your feedback by submitting
   your recommendations to petsc-maint@mcs.anl.gov. See bug report :ref:`documentation
   <doc_creepycrawly>` for more information.

* If you do not have a Fortran compiler or `MPICH <https://www.mpich.org/>`__ installed
  locally (and want to use PETSc from C only).

  .. code-block:: console

     $ ./configure --with-cc=gcc --with-cxx=0 --with-fc=0 --download-f2cblaslapack --download-mpich

* Same as above - but install in a user specified (prefix) location.

  .. code-block:: console

     $ ./configure --prefix=/home/user/soft/petsc-install --with-cc=gcc --with-cxx=0 --with-fc=0 --download-f2cblaslapack --download-mpich

* If `BLAS/LAPACK`_, MPI sources (in "-devel" packages in most Linux distributions) are already
  installed in default system/compiler locations and ``mpicc``, ``mpif90``, mpiexec are available
  via ``$PATH`` - configure does not require any additional options.

  .. code-block:: console

     $ ./configure

* If `BLAS/LAPACK`_, MPI are already installed in known user location use:

  .. code-block:: console

     $ ./configure --with-blaslapack-dir=/usr/local/blaslapack --with-mpi-dir=/usr/local/mpich

  or

  .. code-block:: console

     $ ./configure --with-blaslapack-dir=/usr/local/blaslapack --with-cc=/usr/local/mpich/bin/mpicc --with-mpi-f90=/usr/local/mpich/bin/mpif90 --with-mpiexec=/usr/local/mpich/bin/mpiexec

.. admonition:: Note
   :class: yellow

   Do not specify ``--with-cc``, ``--with-fc`` etc for the above when using
   ``--with-mpi-dir`` - so that ``mpicc``/ ``mpif90`` will be picked up from mpi-dir!

* Build Complex version of PETSc (using c++ compiler):

  .. code-block:: console

     $ ./configure --with-cc=gcc --with-fc=gfortran --with-cxx=g++ --with-clanguage=cxx --download-fblaslapack --download-mpich --with-scalar-type=complex

* Install 2 variants of PETSc, one with gnu, the other with Intel compilers. Specify
  different ``$PETSC_ARCH`` for each build. See multiple PETSc install :ref:`documentation
  <doc_multi>` for further recommendations:

  .. code-block:: console

     $ ./configure PETSC_ARCH=linux-gnu --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich
     $ make PETSC_ARCH=linux-gnu all test
     $ ./configure PETSC_ARCH=linux-gnu-intel --with-cc=icc --with-cxx=icpc --with-fc=ifort --download-mpich --with-blaslapack-dir=/usr/local/mkl
     $ make PETSC_ARCH=linux-gnu-intel all test

.. _doc_config_compilers:

Compilers
=========

.. important::

   If no compilers are specified - configure will automatically look for available MPI or
   regular compilers in the user's ``$PATH`` in the following order:

   #. ``mpicc``/``mpicxx``/``mpif90``
   #. ``gcc``/``g++``/``gfortran``
   #. ``cc``/``CC`` etc..

* Specify compilers using the options ``--with-cc``/``--with-cxx``/``--with-fc`` for c,
  c++, and fortran compilers respectively:

  .. code-block:: console

     $ ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran

.. important::

   It's best to use MPI compilers as this will avoid the situation where MPI is compiled
   with one set of compilers (like ``gcc``/``gfortran``) and the user specified incompatible
   compilers to PETSc (perhaps ``icc``/``ifort``). This can be done by either specifying
   ``--with-cc=mpicc`` or ``--with-mpi-dir`` (and not ``--with-cc=gcc``)

   .. code-block:: console

      $ ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90

   or the following (but **without** ``--with-cc=gcc``)

   .. code-block:: console

      $ ./configure --with-mpi-dir=/opt/mpich2-1.1

* If a fortran compiler is not available or not needed - disable using:

  .. code-block:: console

     $ ./configure --with-fc=0

* If a c++ compiler is not available or not needed - disable using:

  .. code-block:: console

     $ ./configure --with-cxx=0

``configure`` defaults to building PETSc in debug mode. One can switch to optimized
mode with the ``configure`` option ``--with-debugging=0`` (we suggest using a different
``$PETSC_ARCH`` for debug and optimized builds, for example arch-debug and arch-opt, this
way you can switch between debugging your code and running for performance by simply
changing the value of ``$PETSC_ARCH``). See multiple install :ref:`documentation
<doc_multi>` for further details.

Additionally one can specify more suitable optimization flags with the options
``COPTFLAGS``, ``FOPTFLAGS``, ``CXXOPTFLAGS``. For example when using gnu compilers with
corresponding optimization flags:

.. code-block:: console

   $ ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --with-debugging=0 COPTFLAGS='-O3 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' FOPTFLAGS='-O3 -march=native -mtune=native' --download-mpich

.. warning::

   ``configure`` cannot detect compiler libraries for certain set of compilers. In this
   case one can specify additional system/compiler libraries using the ``LIBS`` option:

   .. code-block:: console

      $ ./configure --LIBS='-ldl /usr/lib/libm.a'

.. _doc_config_externalpack:

External Packages
=================

.. admonition:: Note
   :class: yellow

   `BLAS/LAPACK`_ is the only **required** :ref:`external package <doc_externalsoftware>`
   (other than of course build tools such as compilers and ``make``). PETSc may be built
   and run without MPI support if processing only in serial.

   For any :ref:`external packages <doc_externalsoftware>` used with PETSc we highly
   recommend you have PETSc download and install the packages, rather than you installing
   them separately first. This insures that:

   - The packages are installed with the same compilers and compiler options as PETSc
     so that they can work together.
   - A **compatible** version of the package is installed. A generic install of this
     package might not be compatible with PETSc (perhaps due to version differences - or
     perhaps due to the requirement of additional patches for it to work with PETSc).
   - Some packages have bug fixes, portability patches, and upgrades for dependent
     packages that have not yet been included in an upstream release, and hence may not
     play nice with PETSc.

PETSc provides interfaces to various :ref:`external packages <doc_externalsoftware>`.  One
can optionally use external solvers like `HYPRE`_, `MUMPS`_, and others from within PETSc
applications.

PETSc ``configure`` has the ability to download and install these :ref:`external packages
<doc_externalsoftware>`. Alternatively if these packages are already installed, then
``configure`` can detect and use them.

If you are behind a firewall and cannot use a proxy for the downloads or have a very slow
network, use the additional option ``--with-packages-download-dir=/path/to/dir``. This
will trigger ``configure`` to print the URLs of all the packages you must download. You
may then download the packages to some directory (do not uncompress or untar the files)
and then point ``configure`` to these copies of the packages instead of trying to download
them directly from the internet.

The following modes can be used to download/install :ref:`external packages
<doc_externalsoftware>` with ``configure``.

- ``--download-PACKAGENAME``: Download specified package and install it, enabling PETSc to
  use this package. **This is the recommended method to couple any external packages with PETSc**:

  .. code-block:: console

     $ ./configure --download-fblaslapack --download-mpich

- ``--download-PACKAGENAME=/path/to/PACKAGENAME.tar.gz``: If ``configure`` cannot
  automatically download the package (due to network/firewall issues), one can download
  the package by alternative means (perhaps wget, curl, or scp via some other
  machine). Once the tarfile is downloaded, the path to this file can be specified to
  configure with this option. ``configure`` will proceed to install this package and then
  configure PETSc with it:

  .. code-block:: console

     $ ./configure --download-mpich=/home/petsc/mpich2-1.0.4p1.tar.gz

- ``--with-PACKAGENAME-dir=/path/to/dir``: If the external package is already installed -
  specify its location to ``configure`` (it will attempt to detect and include relevant
  library files from this location). Normally this corresponds to the top-level
  installation directory for the package:

  .. code-block:: console

     $ ./configure --with-mpi-dir=/home/petsc/software/mpich2-1.0.4p1


- ``--with-PACKAGENAME-include=/path/to/include/dir`` and
  ``--with-PACKAGENAME-lib=LIBRARYLIST``: Usually a package is defined completely by its
  include file location and library list. If the package is already installed one can use
  these two options to specify the package to ``configure``. For example:

  .. code-block:: console

     $ ./configure --with-superlu-include=/home/petsc/software/superlu/include --with-superlu-lib=/home/petsc/software/superlu/lib/libsuperlu.a

  or

  .. code-block:: console

     $ ./configure --with-parmetis-include=/sandbox/balay/parmetis/include --with-parmetis-lib="-L/sandbox/balay/parmetis/lib -lparmetis -lmetis"

  or

  .. code-block:: console

     $ ./configure --with-parmetis-include=/sandbox/balay/parmetis/include --with-parmetis-lib=[/sandbox/balay/parmetis/lib/libparmetis.a,libmetis.a]

.. note::

   - Run ``./configure --help`` to get the list of :ref:`external packages
     <doc_externalsoftware>` and corresponding additional options (for example
     ``--with-mpiexec`` for `MPICH`_).
   - Generally one would use either one of the above installation modes for any given
     package - and not mix these. (i.e combining ``--with-mpi-dir`` and
     ``--with-mpi-include`` etc. should be avoided).
   - Some packages might not support certain options like ``--download-PACKAGENAME`` or
     ``--with-PACKAGENAME-dir``. Architectures like Microsoft Windows might have issues
     with these options. In these cases, ``--with-PACKAGENAME-include`` and
     ``--with-PACKAGENAME-lib`` options should be preferred.

- ``--with-packages-build-dir=PATH``: By default, external packages will be unpacked and
  the build process is run in ``$PETSC_DIR/$PETSC_ARCH/externalpackages``. However one
  can choose a different location where these packages are unpacked and the build process
  is run.

.. _doc_config_blaslapack:

BLAS/LAPACK
===========

These packages provide some basic numeric kernels used by PETSc. ``configure`` will
automatically look for `BLAS/LAPACK`_ in certain standard locations, on most systems you
should not need to provide any information about `BLAS/LAPACK`_ in the ``configure``
command.

One can use the following options to let ``configure`` download/install `BLAS/LAPACK`_
automatically:

- When fortran compiler is present:

  .. code-block:: console

     $ ./configure --download-fblaslapack

- Or when configuring without a Fortran compiler - i.e ``--with-fc=0``:

  .. code-block:: console

     $ ./configure --download-f2cblaslapack

Alternatively one can use other options like one of the following:

.. code-block:: console

   $ ./configure --with-blaslapack-lib=libsunperf.a
   $ ./configure --with-blas-lib=libblas.a --with-lapack-lib=liblapack.a
   $ ./configure --with-blaslapack-dir=/soft/com/packages/intel/13/079/mkl

Intel MKL
^^^^^^^^^

Intel provides `BLAS/LAPACK`_ via the `MKL`_ library. One can specify it
to PETSc ``configure`` with ``--with-blaslapack-dir=$MKLROOT`` or
``--with-blaslapack-dir=/soft/com/packages/intel/13/079/mkl``.  If the above option does
not work - one could determine the correct library list for your compilers using Intel
`MKL Link Line Advisor`_ and specify with the ``configure`` option
``--with-blaslapack-lib``

IBM ESSL
^^^^^^^^

Sadly, IBM's `ESSL`_ does not have all the routines of `BLAS/LAPACK`_ that some
packages, such as `SuperLU`_ expect; in particular slamch, dlamch and xerbla. In this
case instead of using `ESSL`_ we suggest ``--download-fblaslapack``. If you really want
to use `ESSL`_, see https://www.pdc.kth.se/hpc-services.

.. _doc_config_mpi:

MPI Problems/I Don't Want MPI
=============================

The Message Passing Interface (MPI) provides the parallel functionality for PETSc.

``configure`` will automatically look for MPI compilers ``mpicc``/``mpif90`` etc and use
them if found in your PATH. One can use the following options to let ``configure``
download/install MPI automatically:

- For `MPICH`_:

  .. code-block:: console

     $ ./configure --download-mpich

- For `OpenMPI`_:

  .. code-block:: console

     $ ./configure --download-openmpi

Using MPI Compilers
^^^^^^^^^^^^^^^^^^^

It's best to install PETSc with MPI compiler wrappers (often called ``mpicc``,
``mpicxx``, ``mpif90``) - this way, the SAME compilers used to build MPI are used to
build PETSc. See the section on :ref:`compilers <doc_config_compilers>` above for more
details.

- Vendor provided MPI might already be installed. IBM, Intel, NVIDIA, and Cray provide their own:

  .. code-block:: console

     $ ./configure --with-cc=vendor_mpicc --with-fc=vendor_mpif90

- If using `MPICH`_ which is already installed (perhaps using myrinet/gm) then use
  (without specifying ``--with-cc=gcc`` etc. so that ``configure`` picks up ``mpicc``
  from mpi-dir):

  .. code-block:: console

     $  ./configure --with-mpi-dir=/absolute/path/to/mpich/install

Installing Without MPI
^^^^^^^^^^^^^^^^^^^^^^

You can build (sequential) PETSc without MPI. This is useful for quickly installing PETSc:

.. code-block:: console

   $ ./configure --with-mpi=0

However - if there is any MPI code in user application, then its best to install a full
MPI implementation - even if the usage is currently limited to uniprocessor mode:


Installing With Open MPI With Shared MPI Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`OpenMPI`_ defaults to building shared libraries for MPI. However, the binaries generated
by MPI wrappers ``mpicc``/``mpif90`` etc. require ``$LD_LIBRARY_PATH`` to be set to the
location of these libraries.

Due to this `OpenMPI`_ restriction one has to set ``$LD_LIBRARY_PATH`` correctly (per `OpenMPI`_ `installation instructions`_), before running PETSc ``configure``. If you do not set this environmental variables you will get messages when running ``configure`` such as:

.. code-block:: text

   UNABLE to EXECUTE BINARIES for config/configure.py
   -------------------------------------------------------------------------------
   Cannot run executables created with C. If this machine uses a batch system
   to submit jobs you will need to configure using/configure.py with the additional option --with-batch.
   Otherwise there is problem with the compilers. Can you compile and run code with your C/C++ (and maybe Fortran) compilers?

or when running a code compiled with `OpenMPI`_:

.. code-block:: text

   error while loading shared libraries: libmpi.so.0: cannot open shared object file: No such file or directory

.. _doc_macos_install:

Installing On macOS
===================

For development on macOS we recommend installing **both** the Apple Xcode GUI development system (install from the Apple macOS store) and the Xcode Command Line tools [#]_ install with

.. code-block:: console

   $ xcode-select --install

The Apple compilers are ``clang`` and ``clang++`` [#]_. Apple also provides ``/usr/bin/gcc``, which is, confusingly, a wrapper to the ``clang`` compiler, not the GNU compiler.

We also recommend installing the package manager `homebrew <https://brew.sh/>`__.  To install ``gfortran`` one can use

.. code-block:: console

   $ brew update
   $ brew list            # Show all packages installed through brew
   $ brew upgrade         # Update packages already installed through brew
   $ brew install gcc

This installs gfortran, gcc, and g++  with the compiler names
``gfortran-version`` (also available as ``gfortran``), ``gcc-version`` and ``g++-version``, for example ``gfortran-12``, ``gcc-12``, and ``g++-12``.

After upgrading macOS, you generally need to update the Xcode GUI development system (using the standard Apple software update system),
and the Xcode Command Line tools (run ``xcode-select --install`` again).

Its best to update ``brew`` after all macOS or Xcode upgrades (use ``brew upgrade``). Sometimes gfortran will not work correctly after an upgrade. If this happens
it is best to reinstall all ``brew`` packages using, for example,

.. code-block:: console

   $ brew leaves > leaves.lst      # save packages list to re-install
   $ emacs leaves.lst 	           # edit leaves.lst to remove any un-needed pkgs
   $ brew uninstall `brew list`	   # delete all installed packages
   $ brew cleanup
   $ brew update
   $ brew install `cat leaves.lst` # install needed packages

.. _doc_config_install:

Installation Location: In-place or Out-of-place
===============================================

By default, PETSc does an in-place installation, meaning the libraries are kept in the
same directories used to compile PETSc. This is particularly useful for those application
developers who follow the PETSc git repository main or release branches since rebuilds
for updates are very quick and painless.

.. note::

   The libraries and include files are located in ``$PETSC_DIR/$PETSC_ARCH/lib`` and
   ``$PETSC_DIR/$PETSC_ARCH/include``

Out-of-place Installation With ``--prefix``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the libraries and include files in another location use the ``--prefix`` option

.. code-block:: console

   $ ./configure --prefix=/home/userid/my-petsc-install --some-other-options

The libraries and include files will be located in ``/home/userid/my-petsc-install/lib``
and ``/home/userid/my-petsc-install/include``.

Installation in Root Location, **Not Recommended** (Uncommon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   One should never run ``configure`` or make on any package using root access. **Do so at
   your own risk**.

If one wants to install PETSc in a common system location like ``/usr/local`` or ``/opt``
that requires root access we suggest creating a directory for PETSc with user privileges,
and then do the PETSc install as a **regular/non-root** user:

.. code-block:: console

   $ sudo mkdir /opt/petsc
   $ sudo chown user:group /opt/petsc
   $ cd /home/userid/petsc
   $ ./configure --prefix=/opt/petsc/my-root-petsc-install --some-other-options
   $ make
   $ make install

Installs For Package Managers: Using ``DESTDIR`` (Very uncommon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ ./configure --prefix=/opt/petsc/my-root-petsc-install
   $ make
   $ make install DESTDIR=/tmp/petsc-pkg

Package up ``/tmp/petsc-pkg``. The package should then be installed at
``/opt/petsc/my-root-petsc-install``

Multiple Installs Using ``--prefix`` (See ``DESTDIR``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify a different ``--prefix`` location for each configure of different options - at
configure time. For example:

.. code-block:: console

   $ ./configure --prefix=/opt/petsc/petsc-3.19.0-mpich --with-mpi-dir=/opt/mpich
   $ make
   $ make install [DESTDIR=/tmp/petsc-pkg]
   $ ./configure --prefix=/opt/petsc/petsc-3.19.0-openmpi --with-mpi-dir=/opt/openmpi
   $ make
   $ make install [DESTDIR=/tmp/petsc-pkg]

In-place Installation
^^^^^^^^^^^^^^^^^^^^^

The PETSc libraries and generated included files are placed in the sub-directory off the
current directory ``$PETSC_ARCH`` which is either provided by the user with, for example:

.. code-block:: console

   $ export PETSC_ARCH=arch-debug
   $ ./configure
   $ make
   $ export PETSC_ARCH=arch-opt
   $ ./configure --some-optimization-options
   $ make

or

.. code-block:: console

   $ ./configure PETSC_ARCH=arch-debug
   $ make
   $ ./configure --some-optimization-options PETSC_ARCH=arch-opt
   $ make

If not provided ``configure`` will generate a unique value automatically (for in-place non
``--prefix`` configurations only).

.. code-block:: console

   $ ./configure
   $ make
   $ ./configure --with-debugging=0
   $ make

Produces the directories (on an Apple MacOS machine) ``$PETSC_DIR/arch-darwin-c-debug`` and
``$PETSC_DIR/arch-darwin-c-opt``.

Installing On Machine Requiring Cross Compiler Or A Job Scheduler
=================================================================

On systems where you need to use a job scheduler or batch submission to run jobs use the
``configure`` option ``--with-batch``. **On such systems the make check option will not
work**.

- You must first ensure you have loaded appropriate modules for the compilers etc that you
  wish to use. Often the compilers are provided automatically for you and you do not need
  to provide ``--with-cc=XXX`` etc. Consult with the documentation and local support for
  such systems for information on these topics.

- On such systems you generally should not use ``--with-blaslapack-dir`` or
  ``--download-fblaslapack`` since the systems provide those automatically (sometimes
  appropriate modules must be loaded first).

- Some package's ``--download-package`` options do not work on these systems, for example
  `HDF5`_. Thus you must use modules to load those packages and ``--with-package`` to
  configure with the package.

- Since building :ref:`external packages <doc_externalsoftware>` on these systems is often
  troublesome and slow we recommend only installing PETSc with those configuration
  packages that you need for your work, not extras.

.. _doc_config_tau:

Installing With TAU Instrumentation Package
===========================================

`TAU`_ package and the prerequisite `PDT`_ packages need to be installed separately (perhaps with MPI). Now use tau_cc.sh as compiler to PETSc configure:

.. code-block:: console

   $ export TAU_MAKEFILE=/home/balay/soft/linux64/tau-2.20.3/x86_64/lib/Makefile.tau-mpi-pdt
   $ ./configure CC=/home/balay/soft/linux64/tau-2.20.3/x86_64/bin/tau_cc.sh --with-fc=0 PETSC_ARCH=arch-tau

.. _doc_config_accel:

Installing PETSc To Use GPUs And Accelerators
=============================================

PETSc is able to take adavantage of GPU's and certain accelerator libraries, however some require additional ``configure`` options.

.. _doc_config_accel_cuda:

`CUDA`_
^^^^^^^

.. important::

   An NVIDIA GPU is **required** to use `CUDA`_-accelerated code. Check that your machine
   has a `CUDA`_ enabled GPU by consulting https://developer.nvidia.com/cuda-gpus.

On Linux - make sure you have compatible `NVIDIA driver
<https://developer.nvidia.com/cuda-downloads>`__ installed.

On Microsoft Windows - Use either `Cygwin`_ or `WSL`_ the latter of which is entirely untested right
now. If you have experience with `WSL`_ and/or have successfully built PETSc on Microsoft Windows
for use with `CUDA`_ we welcome your input at petsc-maint@mcs.anl.gov. See the
bug-reporting :ref:`documentation <doc_creepycrawly>` for more details.

In most cases you need only pass the configure option ``--with-cuda``; check
``config/examples/arch-ci-linux-cuda-double.py`` for example usage.

CUDA build of PETSc currently works on Mac OS X, Linux, Microsoft Windows with `Cygwin`_.

Examples that use CUDA have the suffix .cu; see ``$PETSC_DIR/src/snes/tutorials/ex47.cu``

.. _doc_config_accel_kokkos:

`Kokkos`_
^^^^^^^^^

In most cases you need only pass the configure option ``--download-kokkos`` ``--download-kokkos-kernels``
and one of ``--with-cuda``, ``--with-openmp``, or ``--with-pthread`` (or nothing to use sequential
`Kokkos`_). See the :ref:`CUDA installation documenation <doc_config_accel_cuda>`,
:ref:`OpenMPI installation documentation <doc_config_mpi>` for further reference on their
respective requirements.

Examples that use `Kokkos`_ at user-level have the suffix .kokkos.cxx; see
``src/snes/tutorials/ex3k.kokkos.cxx``. More examples use `Kokkos`_ through options database;
search them with ``grep -r -l "requires:.*kokkos_kernels" src/``.

.. _doc_config_accel_opencl:

`OpenCL`_/`ViennaCL`_
^^^^^^^^^^^^^^^^^^^^^

Requires the `OpenCL`_ shared library, which is shipped in the vendor graphics driver and
the `OpenCL`_ headers; if needed you can download them from the Khronos Group
directly. Package managers on Linux provide these headers through a package named
'opencl-headers' or similar. On Apple systems the `OpenCL`_ drivers and headers are always
available and do not need to be downloaded.

Always make sure you have the latest GPU driver installed. There are several known issues
with older driver versions.

Run ``configure`` with ``--download-viennacl``; check
``config/examples/arch-ci-linux-viennacl.py`` for example usage.

`OpenCL`_/`ViennaCL`_ builds of PETSc currently work on Mac OS X, Linux, and Microsoft Windows.

.. _doc_emcc:

Installing To Run in Browser with Emscripten
============================================

PETSc can be used to run applications in the browser using https://emscripten.org, see https://emscripten.org/docs/getting_started/downloads.html,
for instructions on installing Emscripten. Run

.. code-block:: console

   $  ./configure --with-cc=emcc --with-cxx=0 --with-fc=0 --with-ranlib=emranlib --with-ar=emar --with-shared-libraries=0 --download-f2cblaslapack=1 --with-mpi=0 --with-batch

Applications may be compiled with, for example,

.. code-block:: console

   $  make ex19.html

The rule for linking may be found in `lib/petsc/conf/rules <PETSC_DOC_OUT_ROOT_PLACEHOLDER/lib/petsc/conf/rules>`__

.. _doc_config_hpc:

Installing On Large Scale DOE Systems
=====================================

There are some notes on our `GitLab Wiki <https://gitlab.com/petsc/petsc/-/wikis/Installing-and-Running-on-Large-Scale-Systems>`__
which may be helpful in installing and running PETSc on large scale
systems.  Also note the configuration examples in ``config/examples``.

.. _MPICH: https://www.mpich.org/
.. _BLAS/LAPACK: https://www.netlib.org/lapack/lug/node11.html
.. _MUMPS: https://mumps-solver.org/
.. _HYPRE: https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods
.. _SuperLU_DIST: https://github.com/xiaoyeli/superlu_dist
.. _SuperLU: https://portal.nersc.gov/project/sparse/superlu/
.. _METIS: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
.. _ParMETIS: http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview
.. _MKL: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html
.. _MKL Link Line Advisor: https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html
.. _ESSL: https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html
.. _OpenMPI: https://www.open-mpi.org/
.. _installation instructions: https://www.open-mpi.org/faq/?category=building
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _TAU: https://www.cs.uoregon.edu/research/tau/home.php
.. _PDT: https://www.cs.uoregon.edu/research/pdt/home.php
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _Cygwin: https://www.cygwin.com/
.. _WSL: https://docs.microsoft.com/en-us/windows/wsl/install-win10
.. _Kokkos: https://github.com/kokkos/kokkos
.. _OpenCL: https://www.khronos.org/opencl/
.. _ViennaCL: http://viennacl.sourceforge.net/
.. _Modules: https://www.alcf.anl.gov/support-center/theta/compiling-and-linking-overview-theta-thetagpu

.. rubric:: Footnotes

.. [#] The two packages provide slightly different (though largely overlapping) functionality which can only be fully used if both packages are installed.
.. [#] Apple provides customized ``clang`` and ``clang++`` for its system. To use the unmodified LLVM project ``clang`` and ``clang++``
       install them with brew.
