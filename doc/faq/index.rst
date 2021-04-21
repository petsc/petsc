.. include:: <isonum.txt>

.. _doc_faq:

==================================
 Frequently Asked Questions (FAQ)
==================================

This page provides help with the most common questions about PETSc and its design,
execution, and general organization.

.. contents:: Table Of Contents
   :local:
   :backlinks: top

--------------------------------------------------

General
=======

How Can I Subscribe To The PETSc Mailing Lists?
-----------------------------------------------

See mailing list :ref:`documenation <doc_mail>`

Any Useful Books On Numerical Computing?
----------------------------------------

`Bueler, PETSc for Partial Differential Equations: Numerical Solutions in C and Python
<https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137>`__

`Writing Scientific Software: A Guide to Good Style
<https://www.mcs.anl.gov/core/books/writing-scientific-software/23206704175AF868E43FE3FB399C2F53>`__

.. _doc_faq_general_parallel:

What Kind Of Parallel Computers Or Clusters Are Needed To Use PETSc? Or Why Do I Get Little Speedup?
----------------------------------------------------------------------------------------------------

.. important::

   PETSc can be used with any kind of parallel system that supports MPI BUT for any decent
   performance one needs:

   - Fast, **low-latency** interconnect; any ethernet (even 10 GigE) simply cannot provide
     the needed performance.

   - High per-CPU **memory** performance. Each CPU (core in multi-core systems) needs to
     have its **own** memory bandwith of at least 2 or more gigabytes/second. Most modern
     computers are no longer bottlenecked by how fast they can perform a single
     calculation, rather, they are usually restricted by how quickly they can get their
     data.

To obtain good performance it is important that you know your machine! I.e. how many
compute nodes (generally, how many motherboards), how many memory sockets per node and how
many cores per memory socket and how much memory bandwidth for each.

If you do not know this and can run MPI programs with mpiexec (that is, you don't have
batch system), run the following from ``$PETSC_DIR``:

.. code-block:: console

   > make streams [NPMAX=maximum_number_of_mpi_processes_you_plan_to_use]


This will provide a summary of the bandwidth received with different number of MPI
processes and potential speedups. If you have a batch system:

.. code-block:: console

   > cd $PETSC_DIR/src/benchmarks/streams
   > make MPIVersion
   submit MPIVersion to the batch system a number of times with 1, 2, 3, etc MPI processes
   collecting all of the output from the runs into the single file scaling.log. Copy
   scaling.log into the src/benchmarks/streams directory.
   > ./process.py createfile ; process.py

Even if you have enough memory bandwidth if the OS switches processes between cores
performance can degrade. Smart process to core/socket binding (this just means locking a
process to a particular core or memory socket) may help you. For example, consider using
fewer processes than cores and binding processes to separate sockets so that each process
uses a different memory bus:

- `MPICH2 binding with the Hydra process manager
  <https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager#Process-core_Binding>`__

  .. code-block:: console

     > mpiexec.hydra -n 4 --binding cpu:sockets

- `Open MPI binding <http://www.open-mpi.org/doc/v1.5/man1/mpiexec.1.php#sect8>`__

  .. code-block:: console

     > mpiexec -n 4 --bysocket --bind-to-socket --report-bindings

- ``taskset``, part of the `util-linux <https://github.com/karelzak/util-linux>`__ package

  Check ``man taskset`` for details. Make sure to set affinity for **your** program,
  **not** for the ``mpiexec`` program.

- ``numactl``

  In addition to task affinity, this tool also allows changing the default memory affinity
  policy. On Linux, the default policy is to attempt to find memory on the same memory bus
  that serves the core that a thread is running on at whatever time the memory is faulted
  (not when ``malloc()`` is called). If local memory is not available, it is found
  elsewhere, possibly leading to serious memory imbalances.

  The option ``--localalloc`` allocates memory on the local NUMA node, similar to the
  ``numa_alloc_local()`` function in the ``libnuma`` library. The option
  ``--cpunodebind=nodes`` binds the process to a given NUMA node (note that this can be
  larger or smaller than a CPU (socket); a NUMA node usually has multiple cores).

  The option ``--physcpubind=cpus`` binds the process to a given processor core (numbered
  according to ``/proc/cpuinfo``, therefore including logical cores if Hyper-threading is
  enabled).

  With Open MPI, you can use knowledge of the NUMA hierarchy and core numbering on your
  machine to calculate the correct NUMA node or processor number given the environment
  variable ``OMPI_COMM_WORLD_LOCAL_RANK``. In most cases, it is easier to make mpiexec or
  a resource manager set affinities.

The software `Open-MX <http://open-mx.gforge.inria.fr>`__ provides faster speed for
ethernet systems, we have not tried it but it claims it can dramatically reduce latency
and increase bandwidth on Linux system. You must first install this software and then
install MPICH or Open MPI to use it.

What Kind Of License Is PETSc Released Under?
---------------------------------------------

See licensing :ref:`documentation <doc_license>`

Why Is PETSc Written In C, Instead Of Fortran Or C++?
-----------------------------------------------------

C enables us to build data structures for storing sparse matrices, solver information,
etc. in ways that Fortran simply does not allow. ANSI C is a complete standard that all
modern C compilers support. The language is identical on all machines. C++ is still
evolving and compilers on different machines are not identical. Using C function pointers
to provide data encapsulation and polymorphism allows us to get many of the advantages of
C++ without using such a large and more complicated language. It would be natural and
reasonable to have coded PETSc in C++; we opted to use C instead.

Does All The PETSc Error Checking And Logging Reduce PETSc's Efficiency?
------------------------------------------------------------------------

No

.. _doc_faq_maintenance_strats:

How Do Such A Small Group Of People Manage To Write And Maintain Such A Large And Marvelous Package As PETSc?
-------------------------------------------------------------------------------------------------------------

#. **We work very efficiently**.

   - We use Emacs for all editing (and strongly *discourage* our developers from using
     other editors). Having a uniform editing environment speeds development up immensely.

   - Our manual pages are generated automatically from formatted comments in the code,
     thus alleviating the need for creating and maintaining manual pages.

   - We employ automatic nightly tests of the entire PETSc library on several different
     machine architectures. This process **significantly** protects (no bug-catching
     process is perfect) against inadvertently introducing bugs with new additions. Every
     new feature **must** pass our suite of hundreds of tests as well as formal code
     review before it may be included.

#. **We are very careful in our design (and are constantly revising our design)**

   - PETSc as a package should be easy to use, write, and maintain. Our mantra is to write
     code like everyone is using it.

#. **We are willing to do the grunt work**

   - PETSc is regularly checked to make sure that all code conforms to our interface
     design. We will never keep in a bad design decision simply because changing it will
     require a lot of editing; we do a lot of editing.

#. **We constantly seek out and experiment with new design ideas**

   - We retain the useful ones and discard the rest. All of these decisions are based not
     just on performance, but also on **practicality**.

#. **Function and variable names must adhere to strict guidelines**

   - Even the rules about capitalization are designed to make it easy to figure out the
     name of a particular object or routine. Our memories are terrible, so careful
     consistent naming puts less stress on our limited human RAM.

#. **The PETSc directory tree is carefully designed to make it easy to move throughout the
   entire package**

#. **We have a rich, robust, and fast bug reporting system**

   - petsc-maint@mcs.anl.gov is always checked, and we pride ourselves on responding
     quickly and accurately. Email is very lightweight, and so bug reports system retains
     an archive of all reported problems and fixes, so it is easy to re-find fixes to
     previously discovered problems.

#. **We contain the complexity of PETSc by using powerful object-oriented programming
   techniques**

   - Data encapsulation serves to abstract complex data formats or movement to
     human-readable format. This is why your program cannot, for example, look directly
     at what is inside the object ``Mat``.

   - Polymorphism makes changing program behavior as easy as possible, and further
     abstracts the *intent* of your program from what is *written* in code. You call
     ``MatMult()`` regardless of whether your matrix is dense, sparse, parallel or
     sequential; you don't call a different routine for each format.

#. **We try to provide the functionality requested by our users**

#. **There is evil here that does not sleep, and the Great Eye is ever watchful**

For Complex Numbers Will I Get Better Performance With C++?
-----------------------------------------------------------

To use PETSc with complex numbers you mayuse the following ``configure`` options;
``--with-scalar-type=complex`` and either ``--with-clanguage=c++`` or (the default)
``--with-clanguage=c``. In our experience they will deliver very similar performance
(speed), but if one is concerned they should just try both and see if one is faster.

How Come When I Run The Same Program On The Same Number Of Processes I Get A "Different" Answer?
------------------------------------------------------------------------------------------------

Inner products and norms in PETSc are computed using the ``MPI_Allreduce()`` command. For
different runs the order at which values arrive at a given process (via MPI) can be in a
different order, thus the order in which some floating point arithmetic operations are
performed will be different. Since floating point arithmetic arithmetic is not
associative, the computed quantity may be slightly different.

Over a run the many slight differences in the inner products and norms will effect all the
computed results. It is important to realize that none of the computed answers are any
less right or wrong (in fact the sequential computation is no more right then the parallel
ones). All answers are equal, but some are more equal than others.

The discussion above assumes that the exact same algorithm is being used on the different
number of processes. When the algorithm is different for the different number of processes
(almost all preconditioner algorithms except Jacobi are different for different number of
processes) then one expects to see (and does) a greater difference in results for
different numbers of processes. In some cases (for example block Jacobi preconditioner) it
may be that the algorithm works for some number of processes and does not work for others.

How Come When I Run The Same Linear Solver On A Different Number Of Processes It Takes A Different Number Of Iterations?
------------------------------------------------------------------------------------------------------------------------

The convergence of many of the preconditioners in PETSc including the default parallel
preconditioner block Jacobi depends on the number of processes. The more processes the
(slightly) slower convergence it has. This is the nature of iterative solvers, the more
parallelism means the more "older" information is used in the solution process hence
slower convergence.

.. _doc_faq_gpuhowto:

Can PETSc Use GPU's To Speedup Computations?
--------------------------------------------

.. seealso::

   See GPU development :ref:`roadmap <doc_gpu_roadmap>` for the latest information
   regarding the state of PETSc GPU integration.

   See GPU install :ref:`documentation <doc_config_accel>` for up-to-date information on
   installing PETSc to use GPU's.

Quick summary of usage with CUDA:

- The ``VecType`` ``VECSEQCUDA``, ``VECMPICUDA``, or ``VECCUDA`` may be used with
  ``VecSetType()`` or ``-vec_type seqcuda``, ``mpicuda``, or ``cuda`` when
  ``VecSetFromOptions()`` is used.

- The ``MatType`` ``MATSEQAIJCUSPARSE``, ``MATMPIAIJCUSPARSE``, or ``MATAIJCUSPARSE``
  maybe used with ``MatSetType()`` or ``-mat_type seqaijcusparse``, ``mpiaijcusparse``, or
  ``aijcusparse`` when ``MatSetOptions()`` is used.

- If you are creating the vectors and matrices with a ``DM``, you can use ``-dm_vec_type
  cuda`` and ``-dm_mat_type aijcusparse``.

Quick summary of usage with OpenCL (provided by the ViennaCL library):

- The ``VecType`` ``VECSEQVIENNACL``, ``VECMPIVIENNACL``, or ``VECVIENNACL`` may be used
  with ``VecSetType()`` or ``-vec_type seqviennacl``, ``mpiviennacl``, or ``viennacl``
  when ``VecSetFromOptions()`` is used.

- The ``MatType`` ``MATSEQAIJVIENNACL``, ``MATMPIAIJVIENNACL``, or ``MATAIJVIENNACL``
  maybe used with ``MatSetType()`` or ``-mat_type seqaijviennacl``, ``mpiaijviennacl``, or
  ``aijviennacl`` when ``MatSetOptions()`` is used.

- If you are creating the vectors and matrices with a ``DM``, you can use ``-dm_vec_type
  viennacl`` and ``-dm_mat_type aijviennacl``.

General hints:

- It is useful to develop your code with the default vectors and then run production runs
  with the command line options to use the GPU since debugging on GPUs is difficult.

- All of the Krylov methods except ``KSPIBCGS`` run on the GPU.

- Parts of most preconditioners run directly on the GPU. After setup, ``PCGAMG`` runs
  fully on GPUs, without any memory copies between the CPU and GPU.

Some GPU systems (for example many laptops) only run with single precision; thus, PETSc
must be built with the ``configure`` option ``--with-precision=single``.

.. _doc_faq_extendedprecision:

Can I Run PETSc With Extended Precision?
----------------------------------------

Yes, with gcc 4.6 and later (and gfortran 4.6 and later). ``configure`` PETSc using the
options ``--with-precision=__float128`` and `` --download-f2cblaslapack``.

.. admonition:: Warning
   :class: yellow

   External packages are not guaranteed to work in this mode!

Why Doesn't PETSc Use Qd To Implement Support For Exended Precision?
--------------------------------------------------------------------

We tried really hard but could not. The problem is that the QD c++ classes, though they
try to, implement the built-in data types of ``double`` are not native types and cannot
"just be used" in a general piece of numerical source code. Ratherm the code has to
rewritten to live within the limitations of QD classes. However PETSc can be built to use
quad precision, as detailed :ref:`here <doc_faq_extendedprecision>`.

How do I Cite PETSc?
--------------------
Use :any:`these citations <doc_index_citing_petsc>`.

--------------------------------------------------

Installation
============

How Do I Begin Using PETSc If The Software Has Already Been Completely Built And Installed By Someone Else?
-----------------------------------------------------------------------------------------------------------

Assuming that the PETSc libraries have been successfully built for a particular
architecture and level of optimization, a new user must merely:

#. Set ``$PETSC_DIR`` to the full path of the PETSc home
   directory. This will be the location of the ``configure`` script, and usually called
   "petsc" or some vairation of that (for example, /home/username/petsc).

#. Set ``$PETSC_ARCH``, which indicates the configuration on which PETSc will be
   used. Note that the ``$PETSC_ARCH`` is simply a name the installer used when installing
   the libraries. There will exist a directory within ``$PETSC_DIR`` that is named after
   its corresponding ``$PETSC_ARCH``. There many be several on a single system, for
   example "linux-c-debug" for the debug versions compiled by a c compiler or
   "linux-c-opt" for the optimized version.

.. admonition:: Still Stuck?

   See the :ref:`quick-start tutorial <tut_install>` for a step-by-step guide on
   installing PETSc, in case you have missed a step.

   See the users manual section on :ref:`getting started <sec-getting-started>`.

The PETSc Distribution Is SO Large. How Can I Reduce My Disk Space Usage?
-------------------------------------------------------------------------

#. The directory ``$PETSC_DIR/docs`` contains a set of HTML manual pages in for use with a
   browser. You can delete these pages to save some disk space.

#. The PETSc users manual is provided in PDF format at ``$PETSC_DIR/docs/manual.pdf``. You
   can delete this.

#. The PETSc test suite contains sample output for many of the examples. These are
   contained in the PETSc directories ``$PETSC_DIR/src/*/tutorials/output`` and
   ``$PETSC_DIR/src/*/tests/output``. Once you have run the test examples, you may remove
   all of these directories to save some disk space.

#. The debugging versions of the libraries are larger than the optimized versions. In a
   pinch you can work with the optimized version, although we bid you good luck in
   finnding bugs as it is much easier with the debug version.

I Want To Use PETSc Only For Uniprocessor Programs. Must I Still Install And Use A Version Of MPI?
--------------------------------------------------------------------------------------------------

No, run ``configure`` with the option ``--with-mpi=0``

Can I Install PETSc To Not Use X Windows (Either Under Unix Or Windows With GCC)?
---------------------------------------------------------------------------------

Yes. Run ``configure`` with the additional flag ``--with-x=0``

Why Do You Use MPI?
-------------------

MPI is the message-passing standard. Because it is a standard, it will not change over
time; thus, we do not have to change PETSc every time the provider of the message-passing
system decides to make an interface change. MPI was carefully designed by experts from
industry, academia, and government labs to provide the highest quality performance and
capability.

For example, the careful design of communicators in MPI allows the easy nesting of
different libraries; no other message-passing system provides this support. All of the
major parallel computer vendors were involved in the design of MPI and have committed to
providing quality implementations.

In addition, since MPI is a standard, several different groups have already provided
complete free implementations. Thus, one does not have to rely on the technical skills of
one particular group to provide the message-passing libraries. Today, MPI is the only
practical, portable approach to writing efficient parallel numerical software.

What Do I Do If My MPI Compiler Wrappers Are Invalid?
-----------------------------------------------------

Most MPI implementations provide compiler wrappers (such as ``mpicc``) which give the
include and link options necessary to use that verson of MPI to the underlying compilers
. These wrappers are either absent or broken in the MPI pointed to by
``--with-mpi-dir``. You can rerun the configure with the additional option
``--with-mpi-compilers=0``, which will try to auto-detect working compilers; however,
these compilers may be incompatible with the particular MPI build. If this fix does not
work, run with ``--with-cc=[your_c_compiller]`` where you know ``your_c_compiler`` works
with this particular MPI, and likewise for C++ and Fortran.

When Should/Can I Use The ``configure`` Option ``--with-64-bit-indices``?
-------------------------------------------------------------------------

By default the type that PETSc uses to index into arrays and keep sizes of arrays is a
``PetscInt`` defined to be a 32 bit ``int``. If your problem:

- Involves more than 2^31 - 1 unknowns (around 2 billion).

- Your matrix might contain more than 2^31 - 1 nonzeros on a single process.

Then you need to use this option. Otherwise you will get strange crashes.

This option can be used when you are using either 32 bit or 64 bit pointers. You do not
need to use this option if you are using 64 bit pointers unless the two conditions above
hold.

What If I Get An Internal Compiler Error?
-----------------------------------------

You can rebuild the offending file individually with a lower optimization level. **Then
make sure to complain to the compiler vendor and file a bug report**. For example, if the
compiler chokes on ``src/mat/impls/baij/seq/baijsolvtrannat.c`` you can run the following
from ``$PETSC_DIR``:

.. code-block:: console

   > make -f gmakefile PCC_FLAGS="-O1" $PETSC_ARCH/obj/src/mat/impls/baij/seq/baijsolvtrannat.o
   > make all

How Do I Install petsc4py With The Development PETSc?
-----------------------------------------------------

#. Install `Cython <https://cython.org/>`__.

#. ``configure`` PETSc with the ``--download-petsc4py`` option.

What Fortran Compiler Do You Recommend On macOS?
------------------------------------------------

We recommend using `homebrew <https://brew.sh/>`__ to install `gfortran
<https://gcc.gnu.org/wiki/GFortran>`__

Please contact Apple at https://www.apple.com/feedback/ and urge them to bundle gfortran
with future versions of Xcode.

How Can I Find The URL Locations Of The Packages You Install Using ``--download-PACKAGE``?
------------------------------------------------------------------------------------------

.. code-block:: console

   > grep "self.download " $PETSC_DIR/config/BuildSystem/config/packages/*.py

How To Fix The Problem: PETSc Was Configured With One MPICH (Or Open MPI) ``mpi.h`` Version But Now Appears To Be Compiling Using A Different MPICH (Or Open MPI) ``mpi.h`` Version
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This happens for generally one of two reasons:

- You previously ran ``configure`` with the option ``--download-mpich`` (or ``--download-openmpi``)
  but later ran ``configure`` to use a version of MPI already installed on the
  machine. Solution:

  .. code-block:: console

     > rm -rf $PETSC_DIR/$PETSC_ARCH
     > ./configure --your-args

What Does It Mean When ``make check`` Errors On PetscOptionsInsertFile()?
-------------------------------------------------------------------------

For example:

.. code-block:: none

   Possible error running C/C++ src/snes/tutorials/ex19 with 2 MPI processes
   See https://www.mcs.anl.gov/petsc/documentation/faq.html
   [0]PETSC ERROR: #1 PetscOptionsInsertFile() line 563 in /Users/barrysmith/Src/PETSc/src/sys/objects/options.c
   [0]PETSC ERROR: #2 PetscOptionsInsert() line 720 in /Users/barrysmith/Src/PETSc/src/sys/objects/options.c
   [0]PETSC ERROR: #3 PetscInitialize() line 828 in /Users/barrysmith/Src/PETSc/src/sys/objects/pinit.c

- You may be using the wrong ``mpiexec`` for the MPI you have linked PETSc with.

- You have VPN enabled on your machine whose network settings may not play well with MPI.

The machine has a funky network configuration and for some reason MPICH is unable to
communicate between processes with the socket connections it has established. This can
happen even if you are running MPICH on just one machine. Often you will find that ``ping
hostname`` fails with this network configuration; that is, processes on the machine cannot
even connect to the same machine. You can try completely disconnecting your machine from
the network and see if ``make check`` then works or speaking with your system
administrator. You can also try the ``configure`` options ``--download-mpich`` or
``--download-mpich-device=ch3:nemesis``.

--------------------------------------------------

Usage
=====

How Can I Redirect PETSc's ``stdout`` And ``stderr`` When Programming With A Gui Interface In Windows Developer Studio Or To C++ Streams?
-----------------------------------------------------------------------------------------------------------------------------------------

To overload just the error messages write your own ``MyPrintError()`` function that does
whatever you want (including pop up windows etc) and use it like below.

.. code-block:: c

   extern "C" {
     int PASCAL WinMain(HINSTANCE,HINSTANCE,LPSTR,int);
   };

   #include <petscsys.h>
   #include <mpi.h>

   const char help[] = "Set up from main";

   int MyPrintError(const char error[], ...)
   {
     printf("%s", error);
     return 0;
   }

   int main(int ac, char *av[])
   {
     char           buf[256];
     HINSTANCE      inst;
     PetscErrorCode ierr;

     inst = (HINSTANCE)GetModuleHandle(NULL);
     PetscErrorPrintf = MyPrintError;

     buf[0] = 0;
     for (int i = 1; i < ac; ++i) {
       strcat(buf, av[i]);
       strcat(buf, " ");
     }

     ierr = PetscInitialize(&ac, &av, NULL, help);if (ierr) return ierr;

     return WinMain(inst, NULL, buf, SW_SHOWNORMAL);
   }

Place this file in the project and compile with this preprocessor definitions:

::

   WIN32
   _DEBUG
   _CONSOLE
   _MBCS
   USE_PETSC_LOG
   USE_PETSC_BOPT_g
   USE_PETSC_STACK
   _AFXDLL

And these link options:

::

   /nologo
   /subsystem:console
   /incremental:yes
   /debug
   /machine:I386
   /nodefaultlib:"libcmtd.lib"
   /nodefaultlib:"libcd.lib"
   /nodefaultlib:"mvcrt.lib"
   /pdbtype:sept

.. note::

   The above is compiled and linked as if it was a console program. The linker will search
   for a main, and then from it the ``WinMain`` will start. This works with MFC templates and
   derived classes too.

   When writing a Window's console application you do not need to do anything, the ``stdout``
   and ``stderr`` is automatically output to the console window.

To change where all PETSc ``stdout`` and ``stderr`` go, (you can also reassign
``PetscVFPrintf()`` to handle ``stdout`` and ``stderr`` any way you like) write the
following function:

::

   PetscErrorCode mypetscvfprintf(FILE *fd, const char format[], va_list Argp)
   {
     PetscErrorCode ierr;

     PetscFunctionBegin;
     if (fd != stdout && fd != stderr) { /* handle regular files */
       ierr = PetscVFPrintfDefault(fd, format, Argp);CHKERRQ(ierr);
     } else {
       char buff[1024]; /* Make sure to assign a large enough buffer */
       int  length;

       ierr = PetscVSNPrintf(buff, 1024, format, &length, Argp);CHKERRQ(ierr);

       /* now send buff to whatever stream or whatever you want */
     }
     PetscFunctionReturn(0);
   }

Then assign ``PetscVFPrintf = mypetscprintf`` before ``PetscInitialize()`` in your main
program.

I Want To Use Hypre boomerAMG Without GMRES But When I Run ``-pc_type hypre -pc_hypre_type boomeramg -ksp_type preonly`` I Don't Get A Very Accurate Answer!
------------------------------------------------------------------------------------------------------------------------------------------------------------

You should run with ``-ksp_type richardson`` to have PETSc run several V or W
cycles. ``-ksp_type preonly`` causes boomerAMG to use only one V/W cycle. You can control
how many cycles are used in a single application of the boomerAMG preconditioner with
``-pc_hypre_boomeramg_max_iter <it>`` (the default is 1). You can also control the
tolerance boomerAMG uses to decide if to stop before ``max_iter`` with
``-pc_hypre_boomeramg_tol <tol>`` (the default is 1.e-7). Run with ``-ksp_view`` to see
all the hypre options used and ``-help | grep boomeramg`` to see all the command line
options.

How Do I Use PETSc For Domain Decomposition?
--------------------------------------------

PETSc includes Additive Schwarz methods in the suite of preconditioners under the umbrella
of ``PCASM``. These may be activated with the runtime option ``-pc_type asm``. Various
other options may be set, including the degree of overlap ``-pc_asm_overlap <number>`` the
type of restriction/extension ``-pc_asm_type [basic,restrict,interpolate,none]`` sets ASM
type and several others. You may see the available ASM options by using ``-pc_type asm
-help``. See the procedural interfaces in the manual pages, for example ``PCASMType()``
and check the index of the users manual for ``PCASMCreateSubdomains()``.

PETSc also contains a domain decomposition inspired wirebasket or face based two level
method where the coarse mesh to fine mesh interpolation is defined by solving specific
local subdomain problems. It currently only works for 3D scalar problems on structured
grids created with PETSc ``DMDA``. See the manual page for ``PCEXOTIC`` and
``src/ksp/ksp/tutorials/ex45.c`` for an example.

PETSc also contains a balancing Neumann-Neumann type preconditioner, see the manual page
for ``PCBDDC``. This requires matrices be constructed with ``MatCreateIS()`` via the finite
element method. See ``src/ksp/ksp/tests/ex59.c`` for an example.

You Have AIJ And BAIJ Matrix Formats, And SBAIJ For Symmetric Storage, How Come No SAIJ?
----------------------------------------------------------------------------------------

Just for historical reasons; the SBAIJ format with blocksize one is just as efficient as
an SAIJ would be.

Can I Create BAIJ Matrices With Different Size Blocks For Different Block Rows?
-------------------------------------------------------------------------------

No. The ``MATBAIJ`` format only supports a single fixed block size on the entire
matrix. But the ``MATAIJ`` format automatically searches for matching rows and thus still
takes advantage of the natural blocks in your matrix to obtain good performance.

.. note::

   If you use ``MATAIJ`` you cannot use the ``MatSetValuesBlocked()``.

How Do I Access The Values Of A Remote Parallel PETSc Vec?
----------------------------------------------------------

#. On each process create a local ``Vec`` large enough to hold all the values it wishes to
   access.

#. Create a ``VecScatter`` that scatters from the parallel ``Vec`` into the local ``Vec``.

#. Use ``VecGetArray()`` to access the values in the local ``Vec``.


For example, assuming we have distributed a vector ``vecGlobal`` of size :math:`N` to
:math:`R` ranks and each remote rank holds :math:`N/R = m` values (similarly assume that
:math:`N` is cleanly divisible by :math:`R`). We want each rank :math:`r` to gather the
first :math:`n` (also assume :math:`n \leq m`) values from it's immediately superior neighbor
:math:`r+1` (final rank will retrieve from rank 0).

::

   Vec            vecLocal;
   IS             isLocal, isGlobal;
   VecScatter     ctx;
   PetscScalar    *arr;
   PetscInt       N, firstGlobalIndex;
   MPI_Comm       comm;
   PetscMPIInt    r, R;
   PetscErrorCode ierr;

   /* Create sequential local vector, big enough to hold local portion */
   ierr = VecCreateSeq(PETSC_COMM_SELF, n, &vecLocal);CHKERRQ(ierr);

   /* Create IS to describe where we want to scatter to */
   ierr = ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &isLocal);CHKERRQ(ierr);

   /* Compute the global indices */
   ierr = VecGetSize(vecGlobal, &N);CHKERRQ(ierr);
   ierr = PetscObjectGetComm((PetscObject) vecGlobal, &comm);CHKERRQ(ierr);
   ierr = MPI_Comm_rank(comm, &r);CHKERRMPI(ierr);
   ierr = MPI_Comm_size(comm, &R);CHKERRMPI(ierr);
   firstGlobalIndex = r == R-1 ? 0 : (N/R)*(r+1);

   /* Create IS that describes where we want to scatter from */
   ierr = ISCreateStride(comm, n, firstGlobalIndex, 1, &isGlobal);CHKERRQ(ierr);

   /* Create the VecScatter context */
   ierr = VecScatterCreate(vecGlobal, isGlobal, vecLocal, isLocal, &ctx);CHKERRQ(ierr);

   /* Gather the values */
   ierr = VecScatterBegin(ctx, vecGlobal, vecLocal, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
   ierr = VecScatterEnd(ctx, vecGlobal, vecLocal, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

   /* Retrieve and do work */
   ierr = VecGetArray(vecLocal, &arr);CHKERRQ(ierr);
   /* Work */
   ierr = VecRestoreArray(vecLocal, &arr);CHKERRQ(ierr);

   /* Don't forget to clean up */
   ierr = ISDestroy(&isLocal);CHKERRQ(ierr);
   ierr = ISDestroy(&isGlobal);CHKERRQ(ierr);
   ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
   ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);

.. _doc_faq_usage_alltoone:

How Do I Collect To A Single Processor All The Values From A Parallel PETSc Vec?
--------------------------------------------------------------------------------

#. Create the ``VecScatter`` context that will do the communication:

   ::

      Vec            in_par, out_seq;
      VecScatter     ctx;
      PetscErrorCode ierr;

      ierr = VecScatterCreateToAll(in_par, &ctx, &out_seq);CHKERRQ(ierr);

#. Initiate the communication (this may be repeated if you wish):

   ::

      ierr = VecScatterBegin(ctx, in_par, out_seq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ctx, in_par, out_seq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
      /* May destroy context now if no additional scatters are needed, otherwise reuse it */
      ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

Note that this simply concatenates in the parallel ordering of the vector (computed by the
``MPI_Comm_rank`` of the owning process). If you are using a ``Vec`` from
``DMCreateGlobalVector()`` you likely want to first call ``DMDAGlobalToNaturalBegin()``
followed by ``DMDAGlobalToNaturalEnd()`` to scatter the original ``Vec`` into the natural
ordering in a new global ``Vec`` before calling ``VecScatterBegin()``/``VecScatterEnd()``
to scatter the natural ``Vec`` onto all processes.

How Do I Collect To The Zero'th Processor All The Values From A Parallel PETSc Vec?
-----------------------------------------------------------------------------------

See FAQ entry on collecting to :ref:`an arbitrary processor <doc_faq_usage_alltoone>`, but
replace

::

   ierr = VecScatterCreateToAll(in_par, &ctx, &out_seq);CHKERRQ(ierr);

with

::

   ierr = VecScatterCreateToZero(in_par, &ctx, &out_seq);CHKERRQ(ierr);

.. note::

   The same ordering considerations as discussed in the aforementioned entry also apply
   here.

How Can I Read In Or Write Out A Sparse Matrix In Matrix Market, Harwell-Boeing, Slapc Or Other ASCII Format?
-------------------------------------------------------------------------------------------------------------

If you can read or write your matrix using Python or MATLAB/Octave, ``PetscBinaryIO``
modules are provided at ``$PETSC_DIR/lib/petsc/bin`` for each language that can assist
with reading and writing. If you just want to convert ``MatrixMarket``, you can use:

.. code-block:: console

   > python -m $PETSC_DIR/lib/petsc/bin/PetscBinaryIO convert matrix.mtx

To produce ``matrix.petsc``.

You can also call the script directly or import it from your Python code. There is also a
``PETScBinaryIO.jl`` Julia package.

For other formats, either adapt one of the above libraries or see the examples in
``$PETSC_DIR/src/mat/tests``, specifically ``ex72.c`` or ``ex78.c``. You will likely need
to modify the code slightly to match your required ASCII format.

.. note::

   Never read or write in parallel an ASCII matrix file.

   Instead read in sequentially with a standalone code based on ``ex72.c`` or ``ex78.c``
   then save the matrix with the binary viewer ``PetscViewerBinaryOpen()`` and load the
   matrix in parallel in your "real" PETSc program with ``MatLoad()``.

   For writing save with the binary viewer and then load with the sequential code to store
   it as ASCII.


Does TSSetFromOptions(), SNESSetFromOptions() or KSPSetFromOptions() reset all the parameters I previously set or how come do they not seem to work?
----------------------------------------------------------------------------------------------------------------------------------------------------

If ``XXSetFromOptions()`` is used (with ``-xxx_type aaaa``) to change the type of the
object then all parameters associated with the previous type are removed. Otherwise it
does not reset parameters.

``TS/SNES/KSPSetXXX()`` commands that set properties for a particular type of object (such
as ``KSPGMRESSetRestart()``) ONLY work if the object is ALREADY of that type. For example,
with

::

   KSP            ksp;
   PetscErrorCode ierr;

   ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
   ierr = KSPGMRESSetRestart(ksp, 10);CHKERRQ(ierr);

the restart will be ignored since the type has not yet been set to ``KSPGMRES``. To have
those values take effect you should do one of the following:

- Allow setting the type from the command line, if it is not on the command line then the
  default type is automatically set.

::

   /* Create generic object */
   XXXCreate(..,&obj);
   /* Must set all settings here, or default */
   XXXSetFromOptions(obj);

- Hardwire the type in the code, but allow the user to override it via a subsequent
  ``XXXSetFromOptions()`` call. This essentially allows the user to customize what the
  "default" type to of the object.

::

   /* Create generic object */
   XXXCreate(..,&obj);
   /* Set type directly */
   XXXSetYYYYY(obj,...);
   /* Can always change to different type */
   XXXSetFromOptions(obj);

How Do I Compile And Link My Own PETSc Application Codes And Can I Use My Own ``makefile`` Or Rules For Compiling Code, Rather Than PETSc's?
--------------------------------------------------------------------------------------------------------------------------------------------

See the :ref:`section <sec_writing_application_codes>` of the users manual on writing
application codes with PETSc. This provides a simple makefile that can be used to compiler
user code. You are free to modify this makefile or completely replace it with your own
makefile.

Can I Use Cmake To Build My Own Project That Depends On PETSc?
--------------------------------------------------------------

Use `FindPkgConfig.cmake
<https://cmake.org/cmake/help/latest/module/FindPkgConfig.html>`__, which is installed by
default with CMake. PETSc installs ``$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/petsc.pc``,
which can be be read by ``FindPkgConfig.cmake``. If you must use a very old version of
CMake and/or PETSc, you can use the ``FindPETSc.cmake`` module from `this repository
<https://github.com/jedbrown/cmake-modules/>`__.

How Can I Put Carriage Returns In PetscPrintf() Statements From Fortran?
------------------------------------------------------------------------

You can use the same notation as in C, just put a ``\n`` in the string. Note that no other C
format instruction is supported.

Or you can use the Fortran concatination ``//`` and ``char(10)``; for example ``'some
string'//char(10)//'another string`` on the next line.

How Can I Implement Callbacks Using C++ Class Methods?
------------------------------------------------------

Declare the class method static. Static methods do not have a ``this`` pointer, but the
``void*`` context parameter will usually be cast to a pointer to the class where it can
serve the same function.

.. admonition:: Remember

   All PETSc callbacks return ``PetscErrorCode``.

Everyone Knows That When You Code Newton's Method You Should Compute The Function And Its Jacobian At The Same Time. How Can One Do This In PETSc?
--------------------------------------------------------------------------------------------------------------------------------------------------

The update in Newton's method is computed as

.. math::

   u^{n+1} = u^n - \lambda * \left[J(u^n)] * F(u^n) \right]^{\dagger}


The reason PETSc doesn't default to computing both the function and Jacobian at the same
time is:

- In order to do the line search :math:`F \left(u^n - \lambda * \text{step} \right)` may
  need to be computed for several :math:`\lambda`. The Jacobian is not needed for each of
  those and one does not know in advance which will be the final :math:`\lambda` until
  after the function value is computed, so many extra Jacobians may be computed.

- In the final step if :math:`|| F(u^p)||` satisfies the convergence criteria then a
  Jacobian need not be computed.

You are free to have your ``FormFunction()`` compute as much of the Jacobian at that point
as you like, keep the information in the user context (the final argument to
``FormFunction()`` and ``FormJacobian()``) and then retreive the information in your
``FormJacobian()`` function.

Computing The Jacobian Or Preconditioner Is Time Consuming, Is There Any Way To Compute It Less Often?
------------------------------------------------------------------------------------------------------

PETSc has a variety of ways of lagging the computation of the Jacobian or the
preconditioner. They are documented in the manual page for ``SNESComputeJacobian()``
and in the :ref:`users manual <chapter_snes>`:

-snes_lag_jacobian  (``SNESSetLagJacobian()``) How often Jacobian is rebuilt (use -1 to
                    never rebuild, use -2 to rebuild the next time requested and then
                    never again).

-snes_lag_jacobian_persists  (``SNESSetLagJacobianPersists()``) Forces lagging of Jacobian
                             through multiple ``SNES`` solves , same as passing -2 to
                             ``-snes_lag_jacobian``. By default, each new ``SNES`` solve
                             normally triggers a recomputation of the Jacobian.


-snes_lag_preconditioner  (``SNESSetLagPreconditioner()``) how often the preconditioner is
                          rebuilt. Note: if you are lagging the Jacobian the system will
                          know the the matrix has not changed and will not recompute the
                          (same) preconditioner.

-snes_lag_preconditioner_persists  (``SNESSetLagPreconditionerPersists()``) Preconditioner
                                   lags through multiple ``SNES`` solves


.. note::

   These are often (but does not need to be) used in combination with
   ``-snes_mf_operator`` which applies the fresh Jacobian matrix free for every
   matrix-vector product. Otherwise the out-of-date matrix vector product, computed with
   the lagged Jacobian will be used.

By using ``KSPMonitorSet()`` and/or ``SNESMonitorSet()`` one can provide code that monitors the
convergence rate and automatically triggers an update of the Jacobian or preconditioner
based on decreasing convergence of the iterative method. For example if the number of ``SNES``
iterations doubles one might trigger a new computation of the Jacobian. Experimentation is
the only general purpose way to determine which approach is best for your problem.

.. important::

   It is also vital to experiment on your true problem at the scale you will be solving
   the problem since the performance benifits depend on the exact problem and and problem
   size!

How Can I Use Newton's Method Jacobian Free? Can I Difference A Different Function Than Provided With SNESSetFunction()?
------------------------------------------------------------------------------------------------------------------------

The simplest way is with the option ``-snes_mf``, this will use finite differencing of the
function provided to ``SNESComputeFunction()`` to approximate the action of Jacobian.

.. important::

   Since no matrix-representation of the Jacobian is provided the ``-pc_type`` used with
   this option must be ``-pc_type none``. You may provide a custom preconditioner with
   ``SNESGetKSP()`` |rarr| ``KSPGetPC()`` |rarr| ``PCSetType()`` and use ``PCSHELL``.

The option ``-snes_mf_operator`` will use Jacobian free to apply the Jacobian (in the
Krylov solvers) but will use whatever matrix you provided with ``SNESSetJacobian()``
(assuming you set one) to compute the preconditioner.

To write the code (rather than use the options above) use ``MatCreateSNESMF()`` and pass
the resulting matrix object to ``SNESSetJacobian()``.

For purely matrix-free (like ``-snes_mf``) pass the matrix object for both matrix
arguments and pass the function ``MatMFFDComputeJacobian()``.

To provide your own approximate Jacobian matrix to compute the preconditioner (like
``-snes_mf_operator``), pass this other matrix as the second matrix argument to
``SNESSetJacobian()``. Make sure your provided ``computejacobian()`` function calls
``MatAssemblyBegin()`` |rarr| ``MatAssemblyEnd()`` separately on **BOTH** matrix arguments
to this function. See ``src/snes/tests/ex7.c``.

To difference a different function than that passed to ``SNESSetJacobian()`` to compute the
matrix-free Jacobian multiply call ``MatMFFDSetFunction()`` to set that other function. See
``src/snes/tests/ex7.c.h``.

.. _doc_faq_usage_condnum:

How Can I Determine The Condition Number Of A Matrix?
-----------------------------------------------------

For small matrices, the condition number can be reliably computed using

.. code-block:: text

   -pc_type svd -pc_svd_monitor

For larger matrices, you can run with

.. code-block:: text

   -pc_type none -ksp_type gmres -ksp_monitor_singular_value -ksp_gmres_restart 1000

to get approximations to the condition number of the operator. This will generally be
accurate for the largest singular values, but may overestimate the smallest singular value
unless the method has converged. Make sure to avoid restarts. To estimate the condition
number of the preconditioned operator, use ``-pc_type somepc`` in the last command.

How Can I Compute The Inverse Of A Matrix In PETSc?
---------------------------------------------------

.. admonition:: Are you sure?
   :class: yellow

   It is very expensive to compute the inverse of a matrix and very rarely needed in
   practice. We highly recommend avoiding algorithms that need it.

The inverse of a matrix (dense or sparse) is essentially always dense, so begin by
creating a dense matrix B and fill it with the identity matrix (ones along the diagonal),
also create a dense matrix X of the same size that will hold the solution. Then factor the
matrix you wish to invert with ``MatLUFactor()`` or ``MatCholeskyFactor()``, call the
result A. Then call ``MatMatSolve(A,B,X)`` to compute the inverse into X. See also section
on :any:`Schur's complement <how_can_i_compute_the_schur_complement>`.

.. _how_can_i_compute_the_schur_complement:

How Can I Compute The Schur Complement In PETSc?
------------------------------------------------

.. admonition:: Are you sure?
   :class: yellow

   It is very expensive to compute the Schur complement of a matrix and very rarely needed
   in practice. We highly recommend avoiding algorithms that need it.

The Schur complement of the matrix :math:`M \in \mathbb{R}^{\left(p+q \right) \times
\left(p + q \right)}`

.. math::

   M = \begin{bmatrix}
   A & B \\
   C & D
   \end{bmatrix}

where

.. math::

   A \in \mathbb{R}^{p \times p}, \quad B \in \mathbb{R}^{p \times q}, \quad C \in \mathbb{R}^{q \times p}, \quad D \in \mathbb{R}^{q \times q}

is given by

.. math::

   S_D := A - BD^{-1}C \\
   S_A := D - CA^{-1}B

Like the inverse, the Schur complement of a matrix (dense or sparse) is essentially always
dense, so assuming you wish to calculate :math:`S_A = D - C \underbrace{
\overbrace{(A^{-1})}^{U} B}_{V}` begin by:

#. Forming a dense matrix :math:`B`

#. Also create another dense matrix :math:`V` of the same size.

#. Then either factor the matrix :math:`A` directly with ``MatLUFactor()`` or
   ``MatCholeskyFactor()``, or use ``MatGetFactor()`` followed by
   ``MatLUFactorSymbolic()`` followed by ``MatLUFactorNumeric()`` if you wish to use and
   external solver package like SuperLU_Dist. Call the result :math:`U`.

#. Then call ``MatMatSolve(U,B,V)``.

#. Then call ``MatMatMult(C,V,MAT_INITIAL_MATRIX,1.0,&S)``.

#. Now call ``MatAXPY(S,-1.0,D,MAT_SUBSET_NONZERO)``.

#. Followed by ``MatScale(S,-1.0)``.

For computing Schur complements like this it does not make sense to use the ``KSP``
iterative solvers since for solving many moderate size problems using a direct
factorization is much faster than iterative solvers. As you can see, this requires a great
deal of work space and computation so is best avoided.

However, it is not necessary to assemble the Schur complement :math:`S` in order to solve
systems with it. Use ``MatCreateSchurComplement(A,A_pre,B,C,D,&S)`` to create a
matrix that applies the action of :math:`S` (using ``A_pre`` to solve with ``A``), but
does not assemble.

Alternatively, if you already have a block matrix ``M = [A, B; C, D]`` (in some
ordering), then you can create index sets (``IS``) ``isa`` and ``isb`` to address each
block, then use ``MatGetSchurComplement()`` to create the Schur complement and/or an
approximation suitable for preconditioning.

Since :math:`S` is generally dense, standard preconditioning methods cannot typically be
applied directly to Schur complements. There are many approaches to preconditioning Schur
complements including using the ``SIMPLE`` approximation

.. math::

   D - C \text{diag}(A)^{-1} B

to create a sparse matrix that approximates the Schur complement (this is returned by
default for the optional "preconditioning" matrix in ``MatGetSchurComplement()``).

An alternative is to interpret the matrices as differential operators and apply
approximate commutator arguments to find a spectrally equivalent operation that can be
applied efficiently (see the "PCD" preconditioners :cite:`elman_silvester_wathen_2014`). A
variant of this is the least squares commutator, which is closely related to the
Moore-Penrose pseudoinverse, and is available in ``PCLSC`` which operates on matrices of
type ``MATSCHURCOMPLEMENT``.

Do You Have Examples Of Doing Unstructured Grid Finite Element Method (FEM) With PETSc?
---------------------------------------------------------------------------------------

There are at least two ways to write a finite element code using PETSc:

#. Use ``DMPLEX``, which is a high level approach to manage your mesh and
   discretization. See the :ref:`tutorials sections <tut_stokes>` for further information,
   or see ``src/snes/tutorial/ex62.c``.

#. Manage the grid data structure yourself and use PETSc ``IS`` and ``VecScatter`` to
   communicate the required ghost point communication. See
   ``src/snes/tutorials/ex10d/ex10.c``.

DMDA Decomposes The Domain Differently Than The Mpi_Cart_create() Command. How Can One Use Them Together?
---------------------------------------------------------------------------------------------------------

The ``MPI_Cart_create()`` first divides the mesh along the z direction, then the y, then
the x. ``DMDA`` divides along the x, then y, then z. Thus, for example, rank 1 of the
processes will be in a different part of the mesh for the two schemes. To resolve this you
can create a new MPI communicator that you pass to ``DMDACreate()`` that renumbers the
process ranks so that each physical process shares the same part of the mesh with both the
``DMDA`` and the ``MPI_Cart_create()``. The code to determine the new numbering was
provided by Rolf Kuiper:

::

   // the numbers of processors per direction are (int) x_procs, y_procs, z_procs respectively
   // (no parallelization in direction 'dir' means dir_procs = 1)

   MPI_Comm       NewComm;
   int            x, y, z;
   PetscMPIInt    MPI_Rank, NewRank;
   PetscErrorCode ierr;

   // get rank from MPI ordering:
   ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MPI_Rank);CHKERRMPI(ierr);

   // calculate coordinates of cpus in MPI ordering:
   x = MPI_rank / (z_procs*y_procs);
   y = (MPI_rank % (z_procs*y_procs)) / z_procs;
   z = (MPI_rank % (z_procs*y_procs)) % z_procs;

   // set new rank according to PETSc ordering:
   NewRank = z*y_procs*x_procs + y*x_procs + x;

   // create communicator with new ranks according to PETSc ordering
   ierr = MPI_Comm_split(PETSC_COMM_WORLD, 1, NewRank, &NewComm);CHKERRMPI(ierr);

   // override the default communicator (was MPI_COMM_WORLD as default)
   PETSC_COMM_WORLD = NewComm;

When Solving A System With Dirichlet Boundary Conditions I Can Use MatZeroRows() To Eliminate The Dirichlet Rows But This Results In A Non-Symmetric System. How Can I Apply Dirichlet Boundary Conditions But Keep The Matrix Symmetric?
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- For nonsymmetric systems put the appropriate boundary solutions in the x vector and use
  ``MatZeroRows()`` followed by ``KSPSetOperators()``.

- For symmetric problems use ``MatZeroRowsColumns()``.

- If you have many Dirichlet locations you can use ``MatZeroRows()`` (**not**
  ``MatZeroRowsColumns()``) and ``-ksp_type preonly -pc_type redistribute`` (see
  ``PCREDISTRIBUTE``) and PETSc will repartition the parallel matrix for load
  balancing. In this case the new matrix solved remains symmetric even though
  ``MatZeroRows()`` is used.

An alternative approach is, when assemblying the matrix (generating values and passing
them to the matrix), never to include locations for the Dirichlet grid points in the
vector and matrix, instead taking them into account as you put the other values into the
load.

How Can I Get PETSc Vectors And Matrices To MATLAB Or Vice Versa?
-----------------------------------------------------------------

There are numerous  ways to work with PETSc and MATLAB:

#. Using the `MATLAB Engine
   <https://www.mathworks.com/help/matlab/calling-matlab-engine-from-c-programs-1.html>`__,
   allowing PETSc to automatically call MATLAB to perform some specific computations. This
   does not allow MATLAB to be used interactively by the user. See the
   ``PetscMatlabEngine``.

#. To save PETSc ``Mat`` and ``Vec`` to files that can be read from MATLAB use
   ``PetscViewerBinaryOpen()`` viewer and ``VecView()`` or ``MatView()`` to save objects
   for MATLAB and ``VecLoad()`` and ``MatLoad()`` to get the objects that MATLAB has
   saved. See ``share/petsc/matlab/PetscBinaryRead.m`` and
   ``share/petsc/matlab/PetscBinaryWrite.m`` for loading and saving the objects in MATLAB.

#. You can open a socket connection between MATLAB and PETSc to allow sending objects back
   and forth between an interactive MATLAB session and a running PETSc program. See
   ``PetscViewerSocketOpen()`` for access from the PETSc side and
   ``share/petsc/matlab/PetscReadBinary.m`` for access from the MATLAB side.

#. You can save PETSc ``Vec`` (**not** ``Mat``) with the ``PetscViewerMatlabOpen()``
   viewer that saves ``.mat`` files can then be loaded into MATLAB.

How Do I Get Started With Cython So That I Can Extend petsc4py?
---------------------------------------------------------------

#. Learn how to `build a Cython module
   <http://docs.cython.org/src/quickstart/build.html>`__.

#. Go through the simple `example
   <https://stackoverflow.com/questions/3046305/simple-wrapping-of-c-code-with-cython>`__. Note
   also the next comment that shows how to create numpy arrays in the Cython and pass them
   back.

#. Check out `this page <http://docs.cython.org/src/tutorial/numpy.html>`__ which tells
   you how to get fast indexing.

#. Have a look at the petsc4py `array source
   <http://code.google.com/p/petsc4py/source/browse/src/PETSc/arraynpy.pxi>`__.

I Would Like To Compute A Custom Norm For KSP To Use As A Convergence Test Or For Monitoring?
---------------------------------------------------------------------------------------------

You need to call ``KSPBuildResidual()`` on your ``KSP`` object and then compute the
appropriate norm on the resulting residual. Note that depending on the
``KSPSetNormType()`` of the method you may not return the same norm as provided by the
method. See also ``KSPSetPCSide()``.

If I Have A Sequential Program Can I Use A Parallel Direct Solver?
------------------------------------------------------------------


.. important::

   Do not expect to get great speedups! Much of the speedup gained by using parallel
   solvers comes from building the underlying matrices and vectors in parallel to begin
   with. You should see some reduction in the time for the linear solvers.

Yes, you must set up PETSc with MPI (even though you will not use MPI) with at least the
following options:

.. code-block:: console

   > ./configure --download-superlu_dist --download-parmetis --download-metis --with-openmp

Your compiler must support OpenMP. To have the linear solver run in parallel run your
program with

.. code-block:: console

   > OMP_NUM_THREADS=n ./myprog -pc_type lu -pc_factor_mat_solver superlu_dist

where ``n`` is the number of threads and should be less than or equal to the number of cores
available.

.. note::

   If your code is MPI parallel you can also use these same options to have SuperLU_dist
   utilize multiple threads per MPI process for the direct solver. Make sure that the
   ``$OMP_NUM_THREADS`` you use per MPI process is less than or equal to the number of
   cores available for each MPI process. For example if your compute nodes have 6 cores
   and you use 2 MPI processes per node then set ``$OMP_NUM_THREADS`` to 2 or 3.


TS Or SNES Produces Infeasible (Out Of Domain) Solutions Or States, How Can I Prevent This?
-------------------------------------------------------------------------------------------

For ``TS`` call ``TSSetFunctionDomainError()``. For both ``TS`` and ``SNES`` call
``SNESSetFunctionDomainError()`` when the solver passes an infeasible (out of domain)
solution or state to your routines.

If it occurs for DAEs, it is important to insure the algebraic constraints are well
satisfied, which can prevent "breakdown" later. Thus, one can try using a tight tolerance
for ``SNES``, using a direct solver when possible, and reducing the timestep (or
tightening ``TS`` tolerances for adaptive time stepping).

Can PETSc Work With Hermitian Matrices?
---------------------------------------

PETSc's support of Hermitian matrices is very limited. Many operations and solvers work
for symmetric (``MATSBAIJ``) matrices and operations on transpose matrices but there is
little direct support for Hermitian matrices and Hermitian transpose (complex conjugate
transpose) operations. There is ``KSPSolveTranspose()`` for solving the transpose of a
linear system but no ``KSPSolveHermitian()``.

For creating known Hermitian matrices:

- ``MatCreateNormalHermitian()``

- ``MatCreateHermitianTranspose()``

For determining or setting Hermitian status on existing matrices:

- ``MatIsHermitian()``

- ``MatIsHermitianKnown()``

- ``MatIsStructurallySymmetric()``

- ``MatIsSymmetricKnown()``

- ``MatIsSymmetric()``

- ``MatSetOption()`` (use with ``MAT_SYMMETRIC`` or ``MAT_HERMITIAN`` to assert to PETSc
  that either is the case).

For performing matrix operations on known Hermitian matrices (note that regular ``Mat``
functions such as ``MatMult()`` will of course also work on Hermitian matrices):

- ``MatMultHermitianTranspose()``

- ``MatMultHermitianTransposeAdd()`` (very limited support)

How Can I Assemble A Bunch Of Similar Matrices?
-----------------------------------------------

You can first add the values common to all the matrices, then use ``MatStoreValues()`` to
stash the common values. Each iteration you call ``MatRetrieveValues()``, then set the
unique values in a matrix and assemble.

Can One Resize Or Change The Size Of Petsc Matrices Or Vectors?
---------------------------------------------------------------

No, once the vector or matrices sizes have been set and the matrices or vectors are fully
usuable one cannot change the size of the matrices or vectors or number of processors they
live on. One may create new vectors and copy, for example using ``VecScatterCreate()``,
the old values from the previous vector.

How Can One Compute The Nullspace Of A Sparse Matrix With MUMPS?
----------------------------------------------------------------

Assuming you have an existing matrix :math:`A` whose nullspace :math:`V` you want to find:

::

   Mat            F, work, V;
   PetscInt       N, rows;
   PetscErrorCode ierr;

   /* Determine factorability */
   ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &F);CHKERRQ(ierr);
   ierr = MatGetLocalSize(A, &rows, NULL);CHKERRQ(ierr);

   /* Set MUMPS options, see MUMPS documentation for more information */
   ierr = MatMumpsSetIcntl(F, 24, 1);CHKERRQ(ierr);
   ierr = MatMumpsSetIcntl(F, 25, 1);CHKERRQ(ierr);

   /* Perform factorization */
   ierr = MatLUFactorSymbolic(F, A, NULL, NULL, NULL);CHKERRQ(ierr);
   ierr = MatLUFactorNumeric(F, A, NULL);CHKERRQ(ierr);

   /* This is the dimension of the null space */
   ierr = MatMumpsGetInfog(F, 28, &N);CHKERRQ(ierr);

   /* This will contain the null space in the columns */
   ierr = MatCreateDense(comm, rows, N, PETSC_DETERMINE, PETSC_DETERMINE, NULL, &V);CHKERRQ(ierr);

   ierr = MatDuplicate(V, MAT_DO_NOT_COPY_VALUES, &work);CHKERRQ(ierr);
   ierr = MatMatSolve(F, work, V);CHKERRQ(ierr);

--------------------------------------------------

Execution
=========

PETSc Executables Are SO Big And Take SO Long To Link
-----------------------------------------------------

.. note::

   See :ref:`shared libraries section <doc_faq_sharedlibs>` for more information.

We find this annoying as well. On most machines PETSc can use shared libraries, so
executables should be much smaller, run ``configure`` with the additional option
``--with-shared-libraries`` (this is the default). Also, if you have room, compiling and
linking PETSc on your machine's ``/tmp`` disk or similar local disk, rather than over the
network will be much faster.

How Does PETSc's ``-help`` Option Work? Why Is It Different For Different Programs?
-----------------------------------------------------------------------------------

There are 2 ways in which one interacts with the options database:

- ``PetscOptionsGetXXX()`` where ``XXX`` is some type or data structure (for example
  ``PetscOptionsGetBool()`` or ``PetscOptionsGetScalarArray()``). This is a classic
  "getter" function, which queries the command line options for a matching option name,
  and returns the specificied value.

- ``PetscOptionsXXX()`` where ``XXX`` is some type or data structure (for example
  ``PetscOptionsBool()`` or ``PetscOptionsScalarArray()``). This is a so-called "provider"
  function. It first records the option name in an internal list of previously encountered
  options before calling ``PetscOptionsGetXXX()`` to query the status of said option.

While users generally use the first option, developers will *always* use the second
(provider) variant of functions. Thus, as the program runs, it will build up a list of
encountered option names which are then printed **in the order of their appearance on the
root rank**. Different programs may take different paths through PETSc source code, so
they will encounter different providers, and therefore have different ``-help`` output.

PETSc Has So Many Options For My Program That It Is Hard To Keep Them Straight
------------------------------------------------------------------------------

Running the PETSc program with the option ``-help`` will print out many of the options. To
print the options that have been specified within a program, employ ``-options_left`` to
print any options that the user specified but were not actually used by the program and
all options used; this is helpful for detecting typo errors.

PETSc Automatically Handles Many Of The Details In Parallel PDE Solvers. How Can I Understand What Is Really Happening Within My Program?
-----------------------------------------------------------------------------------------------------------------------------------------

You can use the option ``-info`` to get more details about the solution process. The
option ``-log_view`` provides details about the distribution of time spent in the various
phases of the solution process. You can run with ``-ts_view`` or ``-snes_view`` or
``-ksp_view`` to see what solver options are being used. Run with ``-ts_monitor``,
``-snes_monitor``, or ``-ksp_monitor`` to watch convergence of the
methods. ``-snes_converged_reason`` and ``-ksp_converged_reason`` will indicate why and if
the solvers have converged.

Assembling Large Sparse Matrices Takes A Long Time. What Can I Do Make This Process Faster? Or MatSetValues() Is So Slow, What Can I Do To Speed It Up?
-------------------------------------------------------------------------------------------------------------------------------------------------------

See the :ref:`performance chapter <ch_performance>` of the users manual.

How Can I Generate Performance Summaries With PETSc?
----------------------------------------------------

Use these options at runtime:

-log_view  Outputs a comprehensive timing, memory consumption, and comunications digest
           for your program. See the :ref:`profiling chapter <ch_profiling>` of the users
           manual for information on interpreting the summary data.

-snes_view  Generates performance and operational summaries for nonlinear solvers.

-ksp_view   Generates performance and operational summaries for nonlinear solvers.

.. note::

   Only the highest level PETSc object used needs to specify the view option.

How Do I Know The Amount Of Time Spent On Each Level Of The Multigrid Solver/Preconditioner?
--------------------------------------------------------------------------------------------

Run with ``-log_view`` and ``-pc_mg_log``

Where Do I Get The Input Matrices For The Examples?
---------------------------------------------------

Some examples use ``$DATAFILESPATH/matrices/medium`` and other files. These test matrices
in PETSc binary format can be found in the `datafiles repository
<https://gitlab.com/petsc/datafiles>`__.

When I Dump Some Matrices And Vectors To Binary, I Seem To Be Generating Some Empty Files With ``.info`` Extensions. What's The Deal With These?
------------------------------------------------------------------------------------------------------------------------------------------------

PETSc binary viewers put some additional information into ``.info`` files like matrix
block size. It is harmless but if you *really* don't like it you can use
``-viewer_binary_skip_info`` or ``PetscViewerBinarySkipInfo()``.

.. note::

   You need to call ``PetscViewerBinarySkipInfo()`` before
   ``PetscViewerFileSetName()``. In other words you **cannot** use
   ``PetscViewerBinaryOpen()`` directly.

Why Is My Parallel Solver Slower Than My Sequential Solver, Or I Have Poor Speed-Up?
------------------------------------------------------------------------------------

This can happen for many reasons:

#. Make sure it is truly the time in ``KSPSolve()`` that is slower (by running the code
   with ``-log_view``). Often the slower time is in generating the matrix or some other
   operation.

#. There must be enough work for each process to overweigh the communication time. We
   recommend an absolute minimum of about 10,000 unknowns per process, better is 20,000 or
   more.

#. Make sure the :ref:`communication speed of the parallel computer
   <doc_faq_general_parallel>` is good enough for parallel solvers.

#. Check the number of solver iterates with the parallel solver against the sequential
   solver. Most preconditioners require more iterations when used on more processes, this
   is particularly true for block Jacobi (the default parallel preconditioner). You can
   try ``-pc_type asm`` (``PCASM``) its iterations scale a bit better for more
   processes. You may also consider multigrid preconditioners like ``PCMG`` or BoomerAMG
   in ``PCHYPRE``.

What Steps Are Necessary To Make The Pipelined Solvers Execute Efficiently?
---------------------------------------------------------------------------

Pipelined solvers like ``KSPPGMRES``, ``KSPPIPECG``, ``KSPPIPECR``, and ``KSPGROPPCG`` may
require special MPI configuration to effectively overlap reductions with computation. In
general, this requires an MPI-3 implementation, an implementation that supports multiple
threads, and use of a "progress thread". Consult the documentation from your vendor or
computing facility for more.

.. glossary::
   :sorted:

   Cray MPI
      Cray MPT-5.6 supports MPI-3, but setting ``$MPICH_MAX_THREAD_SAFETY`` to "multiple"
      for threads, plus either ``$MPICH_ASYNC_PROGRESS`` or
      ``$MPICH_NEMESIS_ASYNC_PROGRESS``. E.g.

      .. code-block:: console

         > export MPICH_MAX_THREAD_SAFETY=multiple
         > export MPICH_ASYNC_PROGRESS=1
         > export MPICH_NEMESIS_ASYNC_PROGRESS=1

   MPICH
    MPICH version 3.0 and later implements the MPI-3 standard and the default
    configuration supports use of threads. Use of a progress thread is configured by
    setting the environment variable ``$MPICH_ASYNC_PROGRESS``. E.g.

    .. code-block:: console

       > export MPICH_ASYNC_PROGRESS=1

When Using PETSc In Single Precision Mode (``--with-precision=single`` When Running ``configure``) Are The Operations Done In Single Or Double Precision?
---------------------------------------------------------------------------------------------------------------------------------------------------------

PETSc does **NOT** do any explicit conversion of single precision to double before
performing computations; it depends on the hardware and compiler for what happens. For
example, the compiler could choose to put the single precision numbers into the usual
double precision registers and then use the usual double precision floating point unit. Or
it could use SSE2 instructions that work directly on the single precision numbers. It is a
bit of a mystery what decisions get made sometimes. There may be compiler flags in some
circumstances that can affect this.

Why Is Newton'S Method (SNES) Not Converging, Or Converges Slowly?
------------------------------------------------------------------

Newton's method may not converge for many reasons, here are some of the most common:

- The Jacobian is wrong (or correct in sequential but not in parallel).

- The linear system is :ref:`not solved <doc_faq_execution_kspconv>` or is not solved
  accurately enough.

- The Jacobian system has a singularity that the linear solver is not handling.

- There is a bug in the function evaluation routine.

- The function is not continuous or does not have continuous first derivatives (e.g. phase
  change or TVD limiters).

- The equations may not have a solution (e.g. limit cycle instead of a steady state) or
  there may be a "hill" between the initial guess and the steady state (e.g. reactants
  must ignite and burn before reaching a steady state, but the steady-state residual will
  be larger during combustion).

Here are some of the ways to help debug lack of convergence of Newton:

- Run on one processor to see if the problem is only in parallel.

- Run with ``-info`` to get more detailed information on the solution process.

- Run with the options

  .. code-block:: text

     -snes_monitor -ksp_monitor_true_residual -snes_converged_reason -ksp_converged_reason

  - If the linear solve does not converge, check if the Jacobian is correct, then see
    :ref:`this question <doc_faq_execution_kspconv>`.

  - If the preconditioned residual converges, but the true residual does not, the
    preconditioner may be singular.

  - If the linear solve converges well, but the line search fails, the Jacobian may be
    incorrect.

- Run with ``-pc_type lu`` or ``-pc_type svd`` to see if the problem is a poor linear
  solver.

- Run with ``-mat_view`` or ``-mat_view draw`` to see if the Jacobian looks reasonable.

- Run with ``-snes_test_jacobian -snes_test_jacobian_view`` to see if the Jacobian you are
  using is wrong. Compare the output when you add ``-mat_fd_type ds`` to see if the result
  is sensitive to the choice of differencing parameter.

- Run with ``-snes_mf_operator -pc_type lu`` to see if the Jacobian you are using is
  wrong. If the problem is too large for a direct solve, try

  .. code-block:: text

     -snes_mf_operator -pc_type ksp -ksp_ksp_rtol 1e-12.

  Compare the output when you add ``-mat_mffd_type ds`` to see if the result is sensitive
  to choice of differencing parameter.

- Run with ``-snes_linesearch_monitor`` to see if the line search is failing (this is
  usually a sign of a bad Jacobian). Use ``-info`` in PETSc 3.1 and older versions,
  ``-snes_ls_monitor`` in PETSc 3.2 and ``-snes_linesearch_monitor`` in PETSc 3.3 and
  later.

Here are some ways to help the Newton process if everything above checks out:

- Run with grid sequencing (``-snes_grid_sequence`` if working with a ``DM`` is all you
  need) to generate better initial guess on your finer mesh.

- Run with quad precision, i.e.

  .. code-block:: console

     > ./configure --with-precision=__float128 --download-f2cblaslapack

  .. note::

     quad precision requires PETSc 3.2 and later and recent versions of the GNU compilers.

- Change the units (nondimensionalization), boundary condition scaling, or formulation so
  that the Jacobian is better conditioned. See `Buckingham pi theorem
  <https://en.wikipedia.org/wiki/Buckingham_%CF%80_theorem>`__ and `Dimensional and
  Scaling Analysis <https://epubs.siam.org/doi/pdf/10.1137/16M1107127>`__.

- Mollify features in the function that do not have continuous first derivatives (often
  occurs when there are "if" statements in the residual evaluation, e.g. phase change or
  TVD limiters). Use a variational inequality solver (``SNESVINEWTONRSLS``) if the
  discontinuities are of fundamental importance.

- Try a trust region method (``-ts_type tr``, may have to adjust parameters).

- Run with some continuation parameter from a point where you know the solution, see
  ``TSPSEUDO`` for steady-states.

- There are homotopy solver packages like PHCpack that can get you all possible solutions
  (and tell you that it has found them all) but those are not scalable and cannot solve
  anything but small problems.

.. _doc_faq_execution_kspconv:

Why Is The Linear Solver (KSP) Not Converging, Or Converges Slowly?
-------------------------------------------------------------------

.. tip::

   Always run with ``-ksp_converged_reason -ksp_monitor_true_residual`` when trying to
   learn why a method is not converging!

Common reasons for KSP not converging are:

- A symmetric method is being used for a non-symmetric problem.

- The equations are singular by accident (e.g. forgot to impose boundary
  conditions). Check this for a small problem using ``-pc_type svd -pc_svd_monitor``.

- The equations are intentionally singular (e.g. constant null space), but the Krylov
  method was not informed, see ``MatSetNullSpace()``. Always inform your local Krylov
  subspace solver of any change of singularity. Failure to do so will result in the
  immediate revocation of your computing and keyboard operator licenses, as well as
  a stern talking-to by the nearest Krylov Subspace Method representative.

- The equations are intentionally singular and ``MatSetNullSpace()`` was used, but the
  right hand side is not consistent. You may have to call ``MatNullSpaceRemove()`` on the
  right hand side before calling ``KSPSolve()``. See ``MatSetTransposeNullSpace()``.

- The equations are indefinite so that standard preconditioners don't work. Usually you
  will know this from the physics, but you can check with

  .. code-block:: text

     -ksp_compute_eigenvalues -ksp_gmres_restart 1000 -pc_type none

  For simple saddle point problems, try

  .. code-block:: text

     -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_detect_saddle_point

  For more difficult problems, read the literature to find robust methods and ask
  petsc-users@mcs.anl.gov or petsc-maint@mcs.anl.gov if you want advice about how to
  implement them.

- If the method converges in preconditioned residual, but not in true residual, the
  preconditioner is likely singular or nearly so. This is common for saddle point problems
  (e.g. incompressible flow) or strongly nonsymmetric operators (e.g. low-Mach hyperbolic
  problems with large time steps).

- The preconditioner is too weak or is unstable. See if ``-pc_type asm -sub_pc_type lu``
  improves the convergence rate. If GMRES is losing too much progress in the restart, see
  if longer restarts help ``-ksp_gmres_restart 300``. If a transpose is available, try
  ``-ksp_type bcgs`` or other methods that do not require a restart.

  .. note::

     Unfortunately convergence with these methods is frequently erratic.

- The preconditioner is nonlinear (e.g. a nested iterative solve), try ``-ksp_type
  fgmres`` or ``-ksp_type gcr``.

- You are using geometric multigrid, but some equations (often boundary conditions) are
  not scaled compatibly between levels. Try ``-pc_mg_galerkin`` both to algebraically
  construct a correctly scaled coarse operator or make sure that all the equations are
  scaled in the same way if you want to use rediscretized coarse levels.

- The matrix is very ill-conditioned. Check the :ref:`condition number <doc_faq_usage_condnum>`.

  - Try to improve it by choosing the relative scaling of components/boundary conditions.

  - Try ``-ksp_diagonal_scale -ksp_diagonal_scale_fix``.

  - Perhaps change the formulation of the problem to produce more friendly algebraic
    equations.

- Change the units (nondimensionalization), boundary condition scaling, or formulation so
  that the Jacobian is better conditioned. See `Buckingham pi theorem
  <https://en.wikipedia.org/wiki/Buckingham_%CF%80_theorem>`__ and `Dimensional and
  Scaling Analysis <https://epubs.siam.org/doi/pdf/10.1137/16M1107127>`__.

- Classical Gram-Schmidt is becoming unstable, try ``-ksp_gmres_modifiedgramschmidt`` or
  use a method that orthogonalizes differently, e.g. ``-ksp_type gcr``.

I Get The Error Message: Actual argument at (1) to assumed-type dummy is of derived type with type-bound or FINAL procedures
----------------------------------------------------------------------------------------------------------------------------

Use the following code-snippet:

.. code-block:: fortran

   module context_module
   #include petsc/finclude/petsc.h
   use petsc
   implicit none
   private
   type, public ::  context_type
     private
     PetscInt :: foo
   contains
     procedure, public :: init => context_init
   end type context_type
   contains
   subroutine context_init(self, foo)
     class(context_type), intent(in out) :: self
     PetscInt, intent(in) :: foo
     self%foo = foo
   end subroutine context_init
   end module context_module

   !------------------------------------------------------------------------

   program test_snes
   use iso_c_binding
   use petsc
   use context_module
   implicit none

   SNES :: snes
   type(context_type),target :: context
   type(c_ptr) :: contextOut
   PetscErrorCode :: ierr

   call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
   call SNESCreate(PETSC_COMM_WORLD, snes, ierr)
   call context%init(1)

   contextOut = c_loc(context) ! contextOut is a C pointer on context

   call SNESSetConvergenceTest(snes, convergence, contextOut, PETSC_NULL_FUNCTION, ierr)

   call SNESDestroy(snes, ierr)
   call PetscFinalize(ierr)

   contains

   subroutine convergence(snes, num_iterations, xnorm, pnorm,fnorm, reason, contextIn, ierr)
   SNES, intent(in) :: snes

   PetscInt, intent(in) :: num_iterations
   PetscReal, intent(in) :: xnorm, pnorm, fnorm
   SNESConvergedReason, intent(out) :: reason
   type(c_ptr), intent(in out) :: contextIn
   type(context_type), pointer :: context
   PetscErrorCode, intent(out) :: ierr

   call c_f_pointer(contextIn,context)  ! convert the C pointer to a Fortran pointer to use context as in the main program
   reason = 0
   ierr = 0
   end subroutine convergence
   end program test_snes

In C++ I Get A Crash On VecDestroy() (Or Some Other PETSc Object) At The End Of The Program
-------------------------------------------------------------------------------------------

This can happen when the destructor for a C++ class is automatically called at the end of
the program after ``PetscFinalize()``. Use the following code-snippet:

::

   main()
   {
     PetscErrorCode ierr;

     ierr = PetscInitialize();if (ierr) {return ierr;}
     {
       your variables
       your code

       ...   /* all your destructors are called here automatically by C++ so they work correctly */
     }
     ierr = PetscFinalize();if (ierr) {return ierr;}
     return 0
   }

--------------------------------------------------

Debugging
=========

How Do I Turn Off PETSc Signal Handling So I Can Use The ``-C`` Option On ``xlf``?
----------------------------------------------------------------------------------

Immediately after calling ``PetscInitialize()`` call ``PetscPopSignalHandler()``.

Some Fortran compilers including the IBM xlf, xlF etc compilers have a compile option
(``-C`` for IBM's) that causes all array access in Fortran to be checked that they are
in-bounds. This is a great feature but does require that the array dimensions be set
explicitly, not with a \*.

How Do I Debug If ``-start_in_debugger`` Does Not Work On My Machine?
---------------------------------------------------------------------

The script https://github.com/Azrael3000/tmpi can be used with OpenMPI to run multiple MPI
ranks in the debugger using tmux.

On newer macOS machines - one has to be in admin group to be able to use debugger.

On newer UBUNTU linux machines - one has to disable ``ptrace_scope`` with

.. code-block:: console

   > sudo echo 0 > /proc/sys/kernel/yama/ptrace_scope

to get start in debugger working.

If ``-start_in_debugger`` does not really work on your OS, for a uniprocessor job, just
try the debugger directly, for example: ``gdb ex1``. You can also use `TotalView
<https://totalview.io/products/totalview>`__ which is a good graphical parallel debugger.

How Do I See Where My Code Is Hanging?
--------------------------------------

You can use the ``-start_in_debugger`` option to start all processes in the debugger (each
will come up in its own xterm) or run in `TotalView
<https://totalview.io/products/totalview>`__. Then use ``cont`` (for continue) in each
xterm. Once you are sure that the program is hanging, hit control-c in each xterm and then
use 'where' to print a stack trace for each process.

How Can I Inspect PETSc Vector and Matrix Values When In The Debugger?
----------------------------------------------------------------------

I will illustrate this with ``gdb``, but it should be similar on other debuggers. You can
look at local ``Vec`` values directly by obtaining the array. For a ``Vec`` v, we can
print all local values using:

.. code-block:: console

   (gdb) p ((Vec_Seq*) v->data)->array[0]@v->map.n

However, this becomes much more complicated for a matrix. Therefore, it is advisable to use the default viewer to look at the object. For a ``Vec`` v and a ``Mat`` m, this would be:

.. code-block:: console

   (gdb) call VecView(v, 0)
   (gdb) call MatView(m, 0)

or with a communicator other than ``MPI_COMM_WORLD``:

.. code-block:: console

   (gdb) call MatView(m, PETSC_VIEWER_STDOUT_(m->comm))

Totalview 8.8.0+ has a new feature that allows libraries to provide their own code to
display objects in the debugger. Thus in theory each PETSc object, ``Vec``, ``Mat`` etc
could have custom code to print values in the object. We have only done this for the most
elementary display of ``Vec`` and ``Mat``. See the routine ``TV_display_type()`` in
``src/vec/vec/interface/vector.c`` for an example of how these may be written. Contact us
if you would like to add more.

How Can I Find The Cause Of Floating Point Exceptions Like Not-A-Number (NaN) Or Infinity?
------------------------------------------------------------------------------------------

The best way to locate floating point exceptions is to use a debugger. On supported
architectures (including Linux and glibc-based systems), just run in a debugger and pass
``-fp_trap`` to the PETSc application. This will activate signaling exceptions and the
debugger will break on the line that first divides by zero or otherwise generates an
exceptions.

Without a debugger, running with ``-fp_trap`` in debug mode will only identify the
function in which the error occurred, but not the line or the type of exception. If
``-fp_trap`` is not supported on your architecture, consult the documentation for your
debugger since there is likely a way to have it catch exceptions.

Error while loading shared libraries: libimf.so: cannot open shared object file: No such file or directory
----------------------------------------------------------------------------------------------------------

The Intel compilers use shared libraries (like libimf) that cannot by default at run
time. When using the Intel compilers (and running the resulting code) you must make sure
that the proper Intel initialization scripts are run. This is usually done by putting some
code into your ``.cshrc``, ``.bashrc``, ``.profile`` etc file. Sometimes on batch file
systems that do now access your initialization files (like .cshrc) you must include the
initialization calls in your batch file submission.

For example, on my Mac using ``csh`` I have the following in my ``.cshrc`` file:

.. code-block:: csh

   source /opt/intel/cc/10.1.012/bin/iccvars.csh
   source /opt/intel/fc/10.1.012/bin/ifortvars.csh
   source /opt/intel/idb/10.1.012/bin/idbvars.csh

And in my ``.profile`` I have

.. code-block:: csh

   source /opt/intel/cc/10.1.012/bin/iccvars.sh
   source /opt/intel/fc/10.1.012/bin/ifortvars.sh
   source /opt/intel/idb/10.1.012/bin/idbvars.sh

What Does Object Type Not Set: Argument # N Mean?
-------------------------------------------------

Many operations on PETSc objects require that the specific type of the object be set before the operations is performed. You must call ``XXXSetType()`` or ``XXXSetFromOptions()`` before you make the offending call. For example

::

   Mat            A;
   PetscErrorCode ierr;

   ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
   ierr = MatSetValues(A,....);CHKERRQ(ierr);

will not work. You must add ``MatSetType()`` or ``MatSetFromOptions()`` before the call to ``MatSetValues()``. I.e.

::

   Mat            A;
   PetscErrorCode ierr;

   ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);

   ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
   /* Will override MatSetType() */
   ierr = MatSetFromOptions();CHKERRQ(ierr);

   ierr = MatSetValues(A,....);CHKERRQ(ierr);

What Does Error Detected In PetscSplitOwnership() About "Sum Of Local Lengths ...": Mean?
-----------------------------------------------------------------------------------------

In a previous call to ``VecSetSizes()``, ``MatSetSizes()``, ``VecCreateXXX()`` or
``MatCreateXXX()`` you passed in local and global sizes that do not make sense for the
correct number of processors. For example if you pass in a local size of 2 and a global
size of 100 and run on two processors, this cannot work since the sum of the local sizes
is 4, not 100.

What Does Corrupt Argument Or Caught Signal Or SEGV Or Segmentation Violation Or Bus Error Mean? Can I Use Valgrind To Debug Memory Corruption Issues?
------------------------------------------------------------------------------------------------------------------------------------------------------

Sometimes it can mean an argument to a function is invalid. In Fortran this may be caused
by forgetting to list an argument in the call, especially the final ``PetscErrorCode``.

Otherwise it is usually caused by memory corruption; that is somewhere the code is writing
out of array bounds. To track this down rerun the debug version of the code with the
option ``-malloc_debug``. Occasionally the code may crash only with the optimized version,
in that case run the optimized version with ``-malloc_debug``. If you determine the
problem is from memory corruption you can put the macro CHKMEMQ in the code near the crash
to determine exactly what line is causing the problem.

If ``-malloc_debug`` does not help: on GNU/Linux and (supported) macOS machines - you can
use `valgrind <http://valgrind.org>`__. Follow the below instructions:

#. ``configure`` PETSc with ``--download-mpich --with-debugging``.

#. On macOS you need to:

   #. use valgrind from https://github.com/LouisBrunner/valgrind-macos. Follow the Usage
      instructions in the README.md on that page (no need to clone the repository).

   #. use the additional ``configure`` options ``--download-fblaslapack`` or
      ``--download-f2cblaslapack``

   #. use the additional valgrind option ``--dsymutil=yes``

#. Compile your application code with this build of PETSc.

#. Run with valgrind.

   .. code-block:: console

      > $PETSC_DIR/lib/petsc/bin/petscmpiexec -valgrind -n NPROC PETSCPROGRAMNAME PROGRAMOPTIONS

   or

   .. code-block:: console

      > mpiexec -n NPROC valgrind --tool=memcheck -q --num-callers=20 \
      --suppressions=$PETSC_DIR/lib/petsc/bin/maint/petsc-val.supp \
      --log-file=valgrind.log.%p PETSCPROGRAMNAME -malloc off PROGRAMOPTIONS

.. note::


   - option ``--with-debugging`` enables valgrind to give stack trace with additional
     source-file\:line-number info.

   - option ``--download-mpich`` is valgrind clean, other MPI builds are not valgrind clean.

   - when ``--download-mpich`` is used - mpiexec will be in ``$PETSC_ARCH/bin``

   - ``--log-file=valgrind.log.%p`` option tells valgrind to store the output from each
     process in a different file [as %p i.e PID, is different for each MPI process.

   - ``memcheck`` will not find certain array access that violate static array
     declarations so if memcheck runs clean you can try the ``--tool=exp-ptrcheck``
     instead.

You might also consider using http://drmemory.org which has support for GNU/Linux, Apple
Mac OS and Microsoft Windows machines. (Note we haven't tried this ourselves).

What Does Detected Zero Pivot In LU Factorization Mean?
-------------------------------------------------------

A zero pivot in LU, ILU, Cholesky, or ICC sparse factorization does not always mean that
the matrix is singular. You can use

.. code-block:: text

   -pc_factor_shift_type nonzero -pc_factor_shift_amount [amount]

or

.. code-block:: text

   -pc_factor_shift_type positive_definite -[level]_pc_factor_shift_type nonzero
    -pc_factor_shift_amount [amount]

or

.. code-block:: text

   -[level]_pc_factor_shift_type positive_definite

to prevent the zero pivot. [level] is "sub" when lu, ilu, cholesky, or icc are employed in
each individual block of the bjacobi or ASM preconditioner. [level] is "mg_levels" or
"mg_coarse" when lu, ilu, cholesky, or icc are used inside multigrid smoothers or to the
coarse grid solver. See ``PCFactorSetShiftType()``, ``PCFactorSetShiftAmount()``.

This error can also happen if your matrix is singular, see ``MatSetNullSpace()`` for how
to handle this. If this error occurs in the zeroth row of the matrix, it is likely you
have an error in the code that generates the matrix.

You Create Draw Windows Or ``ViewerDraw`` Windows Or Use Options ``-ksp_monitor draw::draw_lg`` Or ``-snes_monitor draw::draw_lg`` And The Program Seems To Run Ok But Windows Never Open
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The libraries were compiled without support for X windows. Make sure that ``configure``
was run with the option ``--with-x``.

The Program Seems To Use More And More Memory As It Runs, Even Though You Don'T Think You Are Allocating More Memory
--------------------------------------------------------------------------------------------------------------------

Some of the following may be occuring:

- You are creating new PETSc objects but never freeing them.

- There is a memory leak in PETSc or your code.

- Something much more subtle: (if you are using Fortran). When you declare a large array
  in Fortran, the operating system does not allocate all the memory pages for that array
  until you start using the different locations in the array. Thus, in a code, if at each
  step you start using later values in the array your virtual memory usage will "continue"
  to increase as measured by ``ps`` or ``top``.

- You are running with the ``-log``, ``-log_mpe``, or ``-log_all`` option. With these
  options, a great deal of logging information is stored in memory until the conclusion of
  the run.

- You are linking with the MPI profiling libraries; these cause logging of all MPI
  activities. Another symptom is at the conclusion of the run it may print some message
  about writing log files.

The following may help:

- Run with the ``-malloc_debug`` option and ``-malloc_view``. Or use ``PetscMallocDump()``
  and ``PetscMallocView()`` sprinkled about your code to track memory that is allocated
  and not later freed. Use the commands ``PetscMallocGetCurrentUsage()`` and
  ``PetscMemoryGetCurrentUsage()`` to monitor memory allocated and
  ``PetscMallocGetMaximumUsage()`` and ``PetscMemoryGetMaximumUsage()`` for total memory
  used as the code progresses.

- This is just the way Unix works and is harmless.

- Do not use the ``-log``, ``-log_mpe``, or ``-log_all`` option, or use
  ``PetscLogEventDeactivate()`` or ``PetscLogEventDeactivateClass()`` to turn off logging of
  specific events.

- Make sure you do not link with the MPI profiling libraries.

When Calling MatPartitioningApply() You Get A Message Error! Key 16615 Not Found
--------------------------------------------------------------------------------

The graph of the matrix you are using is not symmetric. You must use symmetric matrices
for partitioning.

With GMRES At Restart The Second Residual Norm Printed Does Not Match The First
-------------------------------------------------------------------------------

I.e.

.. code-block:: text

   26 KSP Residual norm 3.421544615851e-04
   27 KSP Residual norm 2.973675659493e-04
   28 KSP Residual norm 2.588642948270e-04
   29 KSP Residual norm 2.268190747349e-04
   30 KSP Residual norm 1.977245964368e-04
   30 KSP Residual norm 1.994426291979e-04 <----- At restart the residual norm is printed a second time

This is actually not surprising! GMRES computes the norm of the residual at each iteration
via a recurrence relation between the norms of the residuals at the previous iterations
and quantities computed at the current iteration. It does not compute it via directly
:math:`|| b - A x^{n} ||`.

Sometimes, especially with an ill-conditioned matrix, or computation of the matrix-vector
product via differencing, the residual norms computed by GMRES start to "drift" from the
correct values. At the restart, we compute the residual norm directly, hence the "strange
stuff," the difference printed. The drifting, if it remains small, is harmless (doesn't
affect the accuracy of the solution that GMRES computes).

If you use a more powerful preconditioner the drift will often be smaller and less
noticeable. Of if you are running matrix-free you may need to tune the matrix-free
parameters.

Why Do Some Krylov Methods Seem To Print Two Residual Norms Per Iteration?
--------------------------------------------------------------------------

I.e.

.. code-block:: text

   1198 KSP Residual norm 1.366052062216e-04
   1198 KSP Residual norm 1.931875025549e-04
   1199 KSP Residual norm 1.366026406067e-04
   1199 KSP Residual norm 1.931819426344e-04

Some Krylov methods, for example ``KSPTFQMR``, actually have a "sub-iteration" of size 2
inside the loop. Each of the two substeps has its own matrix vector product and
application of the preconditioner and updates the residual approximations. This is why you
get this "funny" output where it looks like there are two residual norms per
iteration. You can also think of it as twice as many iterations.

Unable To Locate PETSc Dynamic Library ``libpetsc``
---------------------------------------------------

When using DYNAMIC libraries - the libraries cannot be moved after they are
installed. This could also happen on clusters - where the paths are different on the (run)
nodes - than on the (compile) front-end. **Do not use dynamic libraries & shared
libraries**. Run ``configure`` with
``--with-shared-libraries=0 --with-dynamic-loading=0``.

.. important::

   This option has been removed in petsc v3.5

How Do I Determine What Update To PETSc Broke My Code?
------------------------------------------------------

If at some point (in PETSc code history) you had a working code - but the latest PETSc
code broke it, its possible to determine the PETSc code change that might have caused this
behavior. This is achieved by:

- Using Git to access PETSc sources

- Knowing the Git commit for the known working version of PETSc

- Knowing the Git commit for the known broken version of PETSc

- Using the `bisect
  <https://mirrors.edge.kernel.org/pub/software/scm/git/docs/git-bisect.html>`__
  functionality of Git

Git bisect can be done as follows:

#. Get PETSc development (main branch in git) sources

   .. code-block:: console

      > git clone https://gitlab.com/petsc/petsc.git

#. Find the good and bad markers to start the bisection process. This can be done either
   by checking ``git log`` or ``gitk`` or https://gitlab.com/petsc/petsc or the web
   history of petsc-release clones. Lets say the known bad commit is 21af4baa815c and
   known good commit is 5ae5ab319844.

#. Start the bisection process with these known revisions. Build PETSc, and test your code
   to confirm known good/bad behavior:

   .. code-block:: console

      > git bisect start 21af4baa815c 5ae5ab319844

   build/test, perhaps discover that this new state is bad

   .. code-block:: console

      > git bisect bad

   build/test, perhaps discover that this state is good

   .. code-block:: console

      > git bisect good

   Now until done - keep bisecting, building PETSc, and testing your code with it and
   determine if the code is working or not. After something like 5-15 iterations, ``git
   bisect`` will pin-point the exact code change that resulted in the difference in
   application behavior.

.. tip::

   See `git-bisect(1)
   <https://mirrors.edge.kernel.org/pub/software/scm/git/docs/git-bisect.html>`__ and the
   `debugging section of the Git Book
   <https://git-scm.com/book/en/Git-Tools-Debugging-with-Git>`__ for more debugging tips.

How To Fix The Error PMIX Error: error in file gds_ds12_lock_pthread.c?
-----------------------------------------------------------------------

This seems to be an error when using OpenMPI and OpenBLAS with threads (or perhaps other
packages that use threads).

.. code-block:: console

   > export PMIX_MCA_gds=hash

Should resolve the problem.

--------------------------------------------------

.. _doc_faq_sharedlibs:

Shared Libraries
================

Can I Install PETSc Libraries As Shared Libraries?
--------------------------------------------------

Yes. Use

.. code-block:: console

   > ./configure --with-shared-libraries

Why Should I Use Shared Libraries?
----------------------------------

When you link to shared libraries, the function symbols from the shared libraries are not
copied in the executable. This way the size of the executable is considerably smaller than
when using regular libraries. This helps in a couple of ways:

- Saves disk space when more than one executable is created

- Improves the compile time immensly, because the compiler has to write a much smaller
  file (executable) to the disk.

How Do I Link To The PETSc Shared Libraries?
--------------------------------------------

By default, the compiler should pick up the shared libraries instead of the regular
ones. Nothing special should be done for this.

What If I Want To Link To The Regular .a Library Files?
-------------------------------------------------------

You must run ``configure`` with the option ``--with-shared-libraries=0`` (you can use a
different ``$PETSC_ARCH`` for this build so you can easily switch between the two).

What Do I Do If I Want To Move My Executable To A Different Machine?
--------------------------------------------------------------------

You would also need to have access to the shared libraries on this new machine. The other
alternative is to build the exeutable without shared libraries by first deleting the
shared libraries, and then creating the executable.

.. bibliography:: /../src/docs/tex/petsc.bib
   :filter: docname in docnames

.. bibliography:: /../src/docs/tex/petscapp.bib
   :filter: docname in docnames
