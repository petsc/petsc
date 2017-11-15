

/**

  \page manual-user-page-getting-started Getting Started


The Portable, Extensible Toolkit for Scientific Computation (%PETSc)
has successfully demonstrated that the use of modern programming
paradigms can ease the development of large-scale scientific
application codes in Fortran, C, C++, and Python.  Begun several years ago,
the software has evolved into a powerful set of tools for the
numerical solution of partial differential equations and related problems
on high-performance computers. %PETSc is also callable directly from MATLAB (sequential) to
allow "trying out" %PETSc's solvers from prototype MATLAB code.

PETSc consists of a variety of libraries (similar to classes in C++),
which are discussed in detail in Parts II and III of the users manual.
Each library manipulates a particular family of objects (for instance,
vectors) and the operations one would like to perform on the objects.
The objects and operations in %PETSc are derived from our long
experiences with scientific computation. Some of the %PETSc modules deal with
  - index sets (\ref is-class "IS"), including permutations, for indexing into vectors, renumbering, etc;
  - vectors (\ref vec-class "Vec");
  - matrices (\ref mat-class "Mat") (generally sparse);
  - managing interactions between mesh data structures and vectors and matrices (\ref dm-class "DM");
  - over fifteen Krylov subspace methods (\ref ksp-class "KSP");
  - dozens of preconditioners, including multigrid, block solvers, and sparse direct solvers (\ref pc-class "PC");
  - nonlinear solvers (\ref snes-class "SNES"); and
  - timesteppers for solving time-dependent (nonlinear) PDEs, including support for differential algebraic equations (\ref ts-class "TS").

Each consists of an abstract interface
(simply a set of calling sequences) and one or more implementations
using particular data structures. Thus, %PETSc provides clean and
effective codes for the various phases of solving PDEs, with a uniform
approach for each class of problems.  This design
enables easy comparison and use of different algorithms (for example,
to experiment with different Krylov subspace methods, preconditioners,
or truncated Newton methods).
Hence, %PETSc provides a rich environment for modeling scientific
applications as well as for rapid algorithm design and prototyping.

The libraries enable easy customization and extension of both algorithms
and implementations.  This approach promotes code reuse and
flexibility, and separates the issues of parallelism from the choice
of algorithms.  The %PETSc infrastructure creates a
foundation for building large-scale applications.

It is useful to consider the interrelationships among different
pieces of %PETSc.  Here is a diagram of some of these pieces:
\anchor manual-user-fig-1
\image html petscwww.png
<center><b>Organization of the %PETSc Libraries</b></center>

 
Several of the individual parts in more detail are as follows:

\anchor manual-user-fig-2
\image html zoom.png
<center><b>Numerical Libraries of PETSc</b></center>

These figures illustrate the library's hierarchical organization,
which enables users to employ the level of abstraction that is most
appropriate for a particular problem.


\section manual-user-introduction-suggested-reading Suggested Reading

The manual is divided into three parts:
  - Part I - \ref manual-user-page-introduction "Introduction to PETSc"
  - Part II - \ref manual-user-page-programming-with-petsc "Programming with PETSc"
  - Part III - \ref manual-user-page-additional-information "Additional Information"

Part I describes
the basic procedure for using the %PETSc library and presents two
simple examples of solving linear systems with %PETSc.  This section
conveys the typical style used throughout the library and enables the
application programmer to begin using the software immediately.
Part I is also distributed separately for individuals interested in an
overview of the %PETSc software, excluding the details of library usage.
Readers of this separate distribution of Part I should note that all
references within the text to particular chapters and sections
indicate locations in the complete users manual.

Part II explains in detail the use of the various %PETSc libraries,
such as vectors, matrices, index sets, linear and nonlinear
solvers, and graphics.  Part III describes a variety of useful
information, including profiling, the options database, viewers, error
handling, makefiles, and some details of
%PETSc design.

%PETSc has evolved to become quite a comprehensive package, and therefore the
*PETSc Users Manual* can be rather intimidating for new users. We
recommend that one initially reads the entire document before proceeding with
serious use of PETSc, but bear in mind that %PETSc can be used efficiently
before one understands all of the material presented here. Furthermore, the
definitive reference for any %PETSc function is always the online manualpage.


Within the %PETSc distribution, the directory `${PETSC_DIR}/docs`
contains all documentation.
Manual pages for all %PETSc functions can be
accessed at http://www.mcs.anl.gov/petsc/documentation.
The manual pages
provide hyperlinked indices (organized by
both concepts and routine names) to the tutorial examples and enable
easy movement among related topics.

Emacs and Vi/Vim users may find the
`etags`/`ctags`  option to be extremely useful for exploring the PETSc
source code.  Details of this feature are provided in
Section~\ref{sec_emacs}.

The file `manual.pdf` contains
the complete *PETSc Users Manual* in the portable document format (PDF),
while `intro.pdf`
includes only the introductory segment, Part I.
The complete %PETSc distribution, users
manual, manual pages, and additional information are also available via
the %PETSc home page at http://www.mcs.anl.gov/petsc .
The %PETSc home page also
contains details regarding installation, new features and changes in recent
versions of PETSc, machines that we currently support, and a FAQ list for frequently asked questions.


**Note to Fortran Programmers**: In most of the
manual, the examples and calling sequences are given for the C/C++
family of programming languages.  We follow this convention because we
recommend that %PETSc applications be coded in C or C++.
However, pure Fortran programmers can use most of the
functionality of %PETSc from Fortran, with only minor differences in
the user interface.  Chapter \ref{ch_fortran} provides a discussion of the
differences between using %PETSc from Fortran and C, as well as several
complete Fortran examples.  This chapter also introduces some
routines that support direct use of Fortran90 pointers.

**Note to Python Programmers**: To program with %PETSc in Python you need to install the PETSc4py package developed by
Lisandro Dalcin. This can be done by configuring %PETSc with the option `--download-petsc4py`. See the %PETSc installation guide
for more details http://www.mcs.anl.gov/petsc/documentation/installation.html.

**Note to MATLAB Programmers**: To program with %PETSc in MATLAB read the information in `src/matlab/classes/PetscInitialize.m`. Numerious examples
are given in `src/matlab/classes/examples/tutorials`. Run the program demo in that directory.




\section manual-user-sec-running-petsc Running PETSc Programs

Before using PETSc, the user must first set the environmental variable
`PETSC_DIR`, indicating the full path of the %PETSc home
directory.  For example, under the UNIX bash shell a command of the form
\code
   export PETSC_DIR=\$HOME/petsc
\endcode
 can be placed in the user's `.bashrc` or other startup file.  In addition, the user must set the environmental
variable `PETSC_ARCH` to specify the architecture. Note that
`PETSC_ARCH` is just a name selected by the installer to refer to
the libraries compiled for a particular set of compiler options and
machine type. Using different `PETSC_ARCH` allows one to manage
several different sets of libraries easily.

All %PETSc programs use the MPI (Message Passing Interface) standard
for message-passing communication \cite MPI-final.  Thus, to execute
PETSc programs, users must know the procedure for beginning MPI jobs
on their selected computer system(s).  For instance, when using the
MPICH implementation of MPI \cite mpich-web-page and many others, the following
command initiates a program that uses eight processors:
\code
   mpiexec -np 8 ./petsc_program_name petsc_options
\endcode

%PETSc also comes with a script
\code
   ${PETSC_DIR}/bin/petscmpiexec -np 8 ./petsc_program_name petsc_options
\endcode
that uses the information set in `${PETSC_DIR}/${PETSC_ARCH}/conf/petscvariables` to
automatically use the correct `mpiexec` for your configuration.

All PETSc-compliant programs support the use of the `-h` or `-help` option as well as the `-v`
or `-version` option.


Certain options are supported by all %PETSc programs.  We list a few
particularly useful ones below; a complete list can be obtained by
running any %PETSc program with the option `-help`.
  - `-log_summary` - summarize the program's performance
  - `-fp_trap` - stop on floating-point exceptions; \findex{-fp_trap}
      for example divide by zero
  - `-malloc_dump` - enable memory tracing; dump list of unfreed memory
      at conclusion of the run
  - `-malloc_debug` - enable memory tracing (by default this is activated for debugging versions)
  - `-start_in_debugger` `[noxterm,gdb,dbx,xxgdb]` `[-display name]` - start all processes in debugger
  - `-on_error_attach_debugger`  `[noxterm,gdb,dbx,xxgdb]` `[-display name]` - start debugger only on encountering an error
  - `-info` - print a great deal of information about what the programming is doing as it runs
  - `-options_file` `filename` - read options from a file

See Section \ref manual-user-sec-debugging for more information on debugging %PETSc programs.

\section manual-user-sec-writing Writing PETSc Programs

Most %PETSc programs begin with a call to
\code
  PetscInitialize(int *argc,char ***argv,char *file,char *help);
\endcode
which initializes %PETSc and MPI.  The arguments `argc` and
`argv` are the command line arguments delivered in all C and C++
programs. The argument `file`
optionally indicates an alternative name for the %PETSc options file,
`.petscrc`, which resides by default in the user's home directory.
Section `sec_options` provides details regarding
this file and the %PETSc options database, which can be used for runtime
customization. The final argument, `help`, is an optional
character string that will be printed if the program is run with the
`-help` option.  In Fortran the initialization command has the form
\code
   call PetscInitialize(character(*) file,integer ierr)
\endcode
PetscInitialize() automatically calls `MPI_Init()` if MPI
has not been not previously initialized. In certain
circumstances in which MPI needs to be initialized directly (or is
initialized by some other library), the user can first call
`MPI_Init()` (or have the other library do it), and then call
PetscInitialize().
By default, PetscInitialize() sets the %PETSc "world"
communicator, given by `PETSC_COMM_WORLD`, to `MPI_COMM_WORLD`.

For those not familar with MPI, a *communicator* is a way of
indicating a collection of processes that will be involved together
in a calculation or communication. Communicators have the variable type
`MPI_Comm`. In most cases users can employ the communicator
`PETSC_COMM_WORLD` to indicate all processes in a given run and
`PETSC_COMM_SELF` to indicate a single process.

MPI provides routines
for generating new communicators consisting of subsets of processors,
though most users rarely need to use these. The book *Using MPI*,
by Lusk, Gropp, and Skjellum \cite using-mpi provides an excellent
introduction to the concepts in MPI, see also the MPI homepage http://www.mcs.anl.gov/mpi/ .
Note that %PETSc users need not program much message passing directly
with MPI, but they must be familar with the basic concepts of message
passing and distributed memory computing.

All %PETSc routines return an integer indicating whether an error has
occurred during the call.  The error code is set to be nonzero if an
error has been detected; otherwise, it is zero.  For the C/C++
interface, the error variable is the routine's return value, while for
the Fortran version, each %PETSc routine has as its final argument an
integer error variable.  Error tracebacks are discussed in the following
section.

All %PETSc programs should call PetscFinalize()
as their final (or nearly final) statement, as given below in the C/C++
and Fortran formats, respectively:
\code
  PetscFinalize();\\
  call PetscFinalize(ierr)
\endcode
This routine handles options to be called at the conclusion of
the program, and calls `MPI_Finalize()`
if PetscInitialize()
began MPI. If MPI was initiated externally from %PETSc (by either
the user or another software package), the user is
responsible for calling `MPI_Finalize()`.

\section manual-user-sec-simple Simple PETSc Examples

To help the user start using %PETSc immediately, we begin with a simple
\ref manual-user-fig-example1 " uniprocessor example" that solves the
one-dimensional Laplacian problem with finite differences.  This
sequential code, which can be found in
`${PETSC_DIR}/src/ksp/ksp/examples/tutorials/ex1.c`,
illustrates the solution of a linear system with KSP, the
interface to the preconditioners, Krylov subspace methods, and direct
linear solvers of PETSc.  Following the code we highlight a few of the most important
parts of this example.

\anchor manual-user-fig-example1
\includelineno ksp/ksp/examples/tutorials/ex1.c
<center><b>Example of Uniprocessor %PETSc Code</b></center>

\subsection manual-user-subsec-ksp-ex1-include-files Include Files

The C/C++ include files for %PETSc should be used via statements such as
\code
   #include "petscksp.h"
\endcode
where `petscksp.h` is the include file for the linear solver library.
Each %PETSc program must specify an
include file that corresponds to the highest level %PETSc objects
needed within the program; all of the required lower level include
files are automatically included within the higher level files.  For
example, `petscksp.h` includes `petscmat.h` (matrices),
`petscvec.h` (vectors), and `petscsys.h` (base %PETSc file).
The %PETSc include files are located in the directory
`${PETSC_DIR}/include`.  See Section \ref manual-user-sec-fortran-includes
for a discussion of %PETSc include files in Fortran programs.

\subsection manual-user-subsec-ksp-ex1-options The Options Database

As shown in \ref manual-user-fig-example1 " in the snippet above", the user can input control data
at run time using the options database. In this example the command
`PetscOptionsGetInt(NULL,"-n",\&n,\&flg);` checks whether the user has
provided a command line option to set the value of `n`, the
problem dimension.  If so, the variable `n` is set accordingly;
otherwise, `n` remains unchanged. A complete description of the
options database may be found in Section \ref manual-user-sec-options.

\subsection manual-user-subsec-ksp-ex1-vectors Vectors

One creates a new parallel or
sequential vector, `x`, of global dimension `M` with the
commands
\code
  VecCreate(MPI_Comm comm,Vec *x);\\
  VecSetSizes(Vec x, int m, int M);
\endcode
where `comm` denotes the MPI communicator and `m` is the optional local size
which may be PETSC_DECIDE. The type of storage
for the vector may be set with either calls to
VecSetType() or VecSetFromOptions().
Additional vectors of the same type can be formed with
\code
  VecDuplicate(Vec old,Vec *new);
\endcode
The commands
\code
  VecSet(Vec x,PetscScalar value);
  VecSetValues(Vec x,int n,int *indices,PetscScalar *values,INSERT_VALUES);
\endcode
respectively set all the components of a vector to a particular scalar
value and assign a different value to each component.  More
detailed information about %PETSc vectors, including their basic
operations, scattering/gathering, index sets, and distributed arrays, is
discussed \ref manual-user-chapter-vectors " in this Chapter".

Note the use of the %PETSc variable type PetscScalar in this example.
The PetscScalar is simply defined to be `double` in C/C++
(or correspondingly `double` `precision` in Fortran) for versions of
PETSc that have *not* been compiled for use with complex numbers.
The PetscScalar data type enables
identical code to be used when the %PETSc libraries have been compiled
for use with complex numbers.  Section \ref manual-user-sec-complex discusses the
use of complex numbers in %PETSc programs.

\subsection manual-user-subsec-ksp-ex1-matrices Matrices

Usage of %PETSc matrices and vectors is similar.
The user can create a new parallel or sequential matrix, `A`, which
has `M` global rows and `N` global columns, with the routines
\code
  MatCreate(MPI_Comm comm,Mat *A);
  MatSetSizes(Mat A,int m,int n,int M,int N);
\endcode
where the matrix format can be specified at runtime.  The user could
alternatively specify each processes' number of local rows and columns
using `m` and `n`.
Generally one then sets the "type" of the matrix, with, for example,
\code
  MatSetType(Mat A,MATAIJ);
\endcode
This causes the matrix to used the compressed sparse row storage format to store the
matrix entries. See MatType for a list of all matrix types.
Values can then be set with the command
\code
  MatSetValues(Mat A,int m,int *im,int n,int *in,PetscScalar *values,INSERT_VALUES);
\endcode
After  all elements have been inserted into the
matrix, it must be processed with the pair of commands
\code
  MatAssemblyBegin(Mat A,MAT_FINAL_ASSEMBLY);\\
  MatAssemblyEnd(Mat A,MAT_FINAL_ASSEMBLY);
\endcode
\ref manual-user-chapter-matrices "This Chapter" discusses various matrix formats as
well as the details of some basic matrix manipulation routines.

\subsection manual-user-subsec-ksp-ex1-linear-solvers Linear Solvers

After creating the matrix and vectors that define a linear system,
`Ax = b`, the user can then use KSP to solve the system
with the following sequence of commands:
\code
  KSPCreate(MPI_Comm comm,KSP *ksp);
  KSPSetOperators(KSP ksp,Mat A,Mat PrecA,MatStructure flag);
  KSPSetFromOptions(KSP ksp);
  KSPSolve(KSP ksp,Vec b,Vec x);
  KSPDestroy(KSP ksp);
\endcode
The user first creates the KSP context and sets the operators
associated with the system (linear system matrix and optionally different
preconditioning matrix).  The user then sets various options for
customized solution, solves the linear system, and finally destroys
the KSP context.  We emphasize the command KSPSetFromOptions(),
which enables the user to customize the linear solution
method at runtime by using the options database, which is discussed in
Section~\ref manual-user-sec-options. Through this database, the user not only
can select an iterative method and preconditioner, but also can prescribe
the convergence tolerance, set various monitoring routines, etc.
(see, e.g., the \ref manual-user-fig-exprof " subsequent profiling example").

\ref manual-user-ch-ksp "The KSP Chapter " describes in detail the KSP package, including
the PC and KSP packages for preconditioners and Krylov subspace methods.

\subsection manual-user-subsec-ksp-ex1-nonlinear-solvers Nonlinear Solvers
Most PDE problems of interest are inherently nonlinear. %PETSc provides
an interface to tackle the nonlinear problems directly called SNES. Chapter
\ref manual-user-chapter-snes describes the nonlinear solvers in detail. We recommend
most %PETSc users work directly with SNES, rather than using PETSc
for the linear problem within a nonlinear solver.

\subsection manual-user-subsec-ksp-ex1-error-checking Error Checking

All %PETSc routines return an integer indicating whether an error
has occurred during the call.  The %PETSc macro `CHKERRQ(ierr)`
checks the value of `ierr` and calls the %PETSc error handler
upon error detection.  `CHKERRQ(ierr)` should be used in all
subroutines to enable a complete error traceback.
Below we \ref manual-user-fig-traceback indicate a
" traceback generated by error detection " within a sample PETSc
program. The error occurred on line 1673 of the file 
`${PETSC_DIR}/src/mat/impls/aij/seq/aij.c` and was caused by trying to allocate too
large an array in memory. The routine was called in the program
`ex3.c` on line 71.  See Section `sec_fortran_errors` for
details regarding error checking when using the %PETSc Fortran interface.

\anchor manual-user-fig-traceback
\code
   eagle:mpiexec -n 1 ./ex3 -m 10000\\
   %PETSc ERROR: MatCreateSeqAIJ() line 1673 in src/mat/impls/aij/seq/aij.c\\
   %PETSc ERROR:   Out of memory. This could be due to allocating\\
   %PETSc ERROR:   too large an object or bleeding by not properly\\
   %PETSc ERROR:   destroying unneeded objects.\\
   %PETSc ERROR:   Try running with -trdump for more information.\\
   %PETSc ERROR: MatSetType() line 99 in src/mat/utils/gcreate.c  \\
   %PETSc ERROR: main() line 71 in src/ksp/ksp/examples/tutorials/ex3.c\\
   MPI Abort by user Aborting program !\\
   Aborting program! \\
   p0\_28969:  p4\_error: : 1
\endcode
<center><b>Example of Error Traceback</b></center>

When running the debug version of the %PETSc libraries, it
does a great deal of checking for memory corruption (writing outside of
array bounds etc). The macros `CHKMEMQ` can be called
anywhere in the code to check the current status of the memory for corruption.
By putting several (or many) of these macros into your code you can usually
easily track down in what small segment of your code the corruption has occured.

\subsection manual-user-subsec-ksp-ex1-parallel-programming Parallel Programming

Since %PETSc uses the message-passing model for
parallel programming and employs MPI for all interprocessor
communication, the user is free to employ MPI routines as needed
throughout an application code.  However, by default the user is
shielded from many of the details of message passing within PETSc,
since these are hidden within parallel objects, such as vectors,
matrices, and solvers.  In addition, %PETSc provides tools such as
generalized vector scatters/gathers and distributed arrays to assist
in the management of parallel data.

Recall that the user must specify a communicator upon creation of any
PETSc object (such as a vector, matrix, or solver) to indicate the
processors over which the object is to be distributed.  For example,
as mentioned above, some commands for matrix, vector, and linear solver
creation are:
\code
  MatCreate(MPI_Comm comm,Mat *A);
  VecCreate(MPI_Comm comm,Vec *x);
  KSPCreate(MPI_Comm comm,KSP *ksp);
\endcode
The creation routines are collective over all processors in the
communicator; thus, all processors in the communicator *must*
call the creation routine.  In addition, if a sequence of
collective routines is being used, they *must* be called
in the same order on each processor.

The \ref manual-user-fig-example2 " next example", illustrates the
solution of a linear system in parallel.  This code, corresponding to
`${PETSC_DIR}/src/ksp/ksp/examples/tutorials/ex2.c`, handles the
two-dimensional Laplacian discretized with finite differences, where
the linear system is again solved with KSP.  The code performs the
same tasks as the sequential version \ref manual-user-fig-example1 " in the first example".
Note that the user interface for initiating the program, creating
vectors and matrices, and solving the linear system is *exactly*
the same for the uniprocessor and multiprocessor examples.  The
primary difference between the \ref manual-user-fig-example1 " first "
and the \ref manual-user-fig-example2 " second " example is that each processor forms only its local
part of the matrix and vectors in the parallel case.

\anchor manual-user-fig-example2
\includelineno ksp/ksp/examples/tutorials/ex2.c
<center><b>Example of Multiprocessor %PETSc Code</b></center>

\subsection manual-user-subsec-ksp-ex1-compiling-running Compiling and Running Programs

The \ref manual-user-fig-exrun " snippet below" illustrates compiling and running a %PETSc program
using MPICH.  Note that different sites may have slightly different
library and compiler names.  See Chapter \ref manual-user-ch-makefiles
for a discussion about compiling %PETSc programs.
Users who are experiencing difficulties linking %PETSc programs should
refer to the FAQ  via the %PETSc WWW home page
http://www.mcs.anl.gov/petsc or
given in the file `${PETSC\_DIR}/docs/faq.html`.

\anchor manual-user-fig-exrun
\code
   eagle: make ex2\\
   gcc  -pipe -c  -I../../../  -I../../..//include   \\
       -I/usr/local/mpi/include  -I../../..//src -g \\
       -DPETSC\_USE\_DEBUG -DPETSC\_MALLOC -DPETSC\_USE\_LOG ex1.c\\
   gcc -g -DPETSC\_USE\_DEBUG -DPETSC\_MALLOC -DPETSC\_USE\_LOG -o ex1 ex1.o \\
      /home/bsmith/petsc/lib/libg/sun4/libpetscksp.a \\
      -L/home/bsmith/petsc/lib/libg/sun4 -lpetscstencil -lpetscgrid  -lpetscksp \\
      -lpetscmat  -lpetscvec -lpetscsys -lpetscdraw  \\
      /usr/local/lapack/lib/lapack.a /usr/local/lapack/lib/blas.a \\
      /usr/lang/SC1.0.1/libF77.a -lm /usr/lang/SC1.0.1/libm.a -lX11 \\
      /usr/local/mpi/lib/sun4/ch\_p4/libmpi.a\\
      /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o  \\
      /usr/lang/SC1.0.1/libF77.a -lm /usr/lang/SC1.0.1/libm.a -lm\\
   rm -f ex1.o\\
   eagle: mpiexec -np 1 ./ex2\\
   Norm of error 3.6618e-05 iterations 7\\
   eagle:\\
   eagle: mpiexec -np 2 ./ex2\\
   Norm of error 5.34462e-05 iterations 9
\endcode
<center><b>Running a %PETSc Program</b></center>


As shown in the \ref manual-user-fig-exprof " profiler output", the option 
`-log_summary` activates printing of a performance summary, including
times, floating point operation (flop) rates, and message-passing
activity.  Chapter \ref manual-user-ch-profiling
provides details about profiling, including interpretation of the
\ref manual-user-fig-exprof " output data below". This particular example involves the solution of a linear
system on one processor using GMRES and ILU.  The low floating point
operation (flop) rates in this example are due to the fact that the
code solved a tiny system.  We include this example merely to
demonstrate the ease of extracting performance information.

\anchor manual-user-fig-exprof
\verbatim
eagle> mpiexec -n 1 ./ex1 -n 1000 -pc_type ilu -ksp_type gmres -ksp_rtol 1.e-7 -log_summary
-------------------------------- %PETSc Performance Summary: -------------------------------

ex1 on a sun4 named merlin.mcs.anl.gov with 1 processor, by curfman Wed Aug 7 17:24 1996

                         Max         Min        Avg        Total
Time (sec):           1.150e-01      1.0   1.150e-01
Objects:              1.900e+01      1.0   1.900e+01
Flops:                3.998e+04      1.0   3.998e+04  3.998e+04
Flops/sec:            3.475e+05      1.0              3.475e+05
MPI Messages:         0.000e+00      0.0   0.000e+00  0.000e+00
MPI Messages:         0.000e+00      0.0   0.000e+00  0.000e+00 (lengths)
MPI Reductions:       0.000e+00      0.0

--------------------------------------------------------------------------------------
Phase              Count      Time (sec)       Flops/sec     Messages    -- Global --
                             Max     Ratio    Max    Ratio Avg len  Redc %T %F %M %L %R
--------------------------------------------------------------------------------------
Mat Mult               2  2.553e-03    1.0  3.9e+06    1.0  0.0 0.0 0.0  2 25  0  0  0
Mat AssemblyBegin      1  2.193e-05    1.0  0.0e+00    0.0  0.0 0.0 0.0  0  0  0  0  0
Mat AssemblyEnd        1  5.004e-03    1.0  0.0e+00    0.0  0.0 0.0 0.0  4  0  0  0  0
Mat GetOrdering        1  3.004e-03    1.0  0.0e+00    0.0  0.0 0.0 0.0  3  0  0  0  0
Mat ILUFctrSymbol      1  5.719e-03    1.0  0.0e+00    0.0  0.0 0.0 0.0  5  0  0  0  0
Mat LUFactorNumer      1  1.092e-02    1.0  2.7e+05    1.0  0.0 0.0 0.0  9  7  0  0  0
Mat Solve              2  4.193e-03    1.0  2.4e+06    1.0  0.0 0.0 0.0  4 25  0  0  0
Mat SetValues       1000  2.461e-02    1.0  0.0e+00    0.0  0.0 0.0 0.0 21  0  0  0  0
Vec Dot                1     60e-04    1.0  9.7e+06    1.0  0.0 0.0 0.0  0  5  0  0  0
Vec Norm               3  5.870e-04    1.0  1.0e+07    1.0  0.0 0.0 0.0  1 15  0  0  0
Vec Scale              1  1.640e-04    1.0  6.1e+06    1.0  0.0 0.0 0.0  0  3  0  0  0
Vec Copy               1  3.101e-04    1.0  0.0e+00    0.0  0.0 0.0 0.0  0  0  0  0  0
Vec Set                3  5.029e-04    1.0  0.0e+00    0.0  0.0 0.0 0.0  0  0  0  0  0
Vec AXPY               3  8.690e-04    1.0  6.9e+06    1.0  0.0 0.0 0.0  1 15  0  0  0
Vec MAXPY              1  2.550e-04    1.0  7.8e+06    1.0  0.0 0.0 0.0  0  5  0  0  0
KSP Solve              1  1.288e-02    1.0  2.2e+06    1.0  0.0 0.0 0.0 11 70  0  0  0
KSP SetUp              1  2.669e-02    1.0  1.1e+05    1.0  0.0 0.0 0.0 23  7  0  0  0
KSP GMRESOrthog        1  1.151e-03    1.0  3.5e+06    1.0  0.0 0.0 0.0  1 10  0  0  0
PC SetUp               1  2.4e-02      1.0  1.5e+05    1.0  0.0 0.0 0.0 18  7  0  0  0
PC Apply               2  4.474e-03    1.0  2.2e+06    1.0  0.0 0.0 0.0  4 25  0  0  0
--------------------------------------------------------------------------------------

Memory usage is given in bytes:

Object Type      Creations   Destructions   Memory  Descendants' Mem.
Index set             3              3      12420     0
Vector                8              8      65728     0
Matrix                2              2     184924     4140
Krylov Solver         1              1      16892     41080
Preconditioner        1              1          0     64872

\endverbatim
<center><b>Running a %PETSc Program with Profiling</b></center>


\subsection manual-user-subsec-ksp-ex1-writing-app-codes Writing Application Codes with PETSc

The examples throughout the library demonstrate the software usage
and can serve as templates for developing
custom applications.  We suggest that new PETSc
users examine programs in the directories
\code
  ${PETSC_DIR}/src/<library>/examples/tutorials,
\endcode
where `<library>`
denotes any of the %PETSc libraries (listed in the following
section), such as \trl{snes} or \trl{ksp}.
The manual pages located at ${PETSC_DIR}/docs/index.html or http://www.mcs.anl.gov/petsc/documentation
provide indices (organized by both routine names and concepts) to the tutorial examples.

To write a new application program using PETSc, we suggest the
following procedure:
  - Install and test %PETSc according to the instructions at the %PETSc web site.
  - Copy one of the many %PETSc examples in the directory
      that corresponds to the class of problem of interest (e.g.,
      for linear solvers, see `${PETSC_DIR}/src/ksp/ksp/examples/tutorials`).
  - Copy the corresponding makefile within the example directory;
      compile and run the example program.
  - Use the example program as a starting point for developing a custom code.


\section manual-user-sec-citing-petsc Citing PETSc

When citing %PETSc in a publication please cite the following:
\code

@Misc{petsc-web-page,
   Author = "Satish Balay and Jed Brown and Kris Buschelman and Victor Eijkhout and William~D. Gropp and Dinesh Kaushik and Matthew~G. Knepley and Lois Curfman McInnes and Barry~F. Smith and Hong Zhang",
   Title  = "{PETS}c {W}eb page",
   Note   = "http://www.mcs.anl.gov/petsc",
   Year   = "2013" }

@TechReport{petsc-user-ref,
   Author      = "Satish Balay and Jed Brown and Kris Buschelman and Victor Eijkhout and William~D. Gropp and Dinesh Kaushik and Matthew~G. Knepley and Lois Curfman McInnes and Barry~F. Smith and Hong Zhang",
   Title       = "PETSc Users Manual",
   Number      = "ANL-95/11 - Revision 3.4",
   Institution = "Argonne National Laboratory",
   Year        = "2013" }

@InProceedings{petsc-efficient,
   Author    = "Satish Balay and William D. Gropp and Lois C. McInnes and Barry F. Smith",
   Title     = "Efficienct Management of Parallelism in Object Oriented Numerical Software Libraries",
   Booktitle = "Modern Software Tools in Scientific Computing",
   Editor    = "E. Arge and A. M. Bruaset and H. P. Langtangen",
   Pages     = "163--202",
   Publisher = "Birkhauser Press",
   Year      = "1997" }
\endcode


\section manual-user-sec-directory-structure Directory Structure

We conclude this introduction with an overview of the
organization of the %PETSc software.
The root directory of %PETSc contains the following directories:

  - `docs` - All documentation for PETSc. The files \trl{manual.pdf}
                   contains the hyperlinked users manual, suitable for printing
                   or on-screen viewering. Includes the subdirectory
     - `manualpages` (on-line manual pages).
  - `bin` - Utilities and short scripts for use with PETSc, including
     - `petscmpiexec` (utility for setting running MPI jobs),

  - conf - Base %PETSc makefile that defines the standard make variables and rules used by PETSc
  - `include` - All include files for %PETSc that are visible to the user.
  - `include/finclude`    - %PETSc include files for Fortran programmers using the .F suffix (recommended).
  - `include/private`    - Private %PETSc include files that should *not* be used by application programmers.
  - `share` - Some small test matrices in data files
  - `src` - The source code for all %PETSc libraries, which currently includes
      - `vec` - vectors,
        - `is` - index sets,

      - `mat` - matrices,
      - `dm` - data management between meshes and vectors and matrices,
      - `ksp` - complete linear equations solvers,
        - `ksp` - Krylov subspace accelerators,
        - `pc` - preconditioners,

      - `snes` - nonlinear solvers
      - `ts` - ODE solvers and timestepping,
      - `sys` - general system-related routines,
        - `plog` - %PETSc logging and profiling routines,
        - `draw` - simple graphics,
      - `contrib` - contributed modules that use %PETSc but are not
    part of the official %PETSc package.  We encourage users who have
    developed such code that they wish to share with others to let us
    know by writing to petsc-maint@mcs.anl.gov.


Each %PETSc source code library directory has the following subdirectories:
  - `examples` - Example programs for the component, including
    - `tutorials` - Programs designed to teach users about PETSc.  These
          codes can serve as templates for the design of custom applications.
    - `tests` - Programs designed for thorough testing of PETSc.  As such,
          these codes are not intended for examination by users.
  - `interface` - The calling sequences for the abstract interface
        to the component.
        Code here does not know about particular implementations.
  - `impls` - Source code for one or more implementations.
  - `utils` - Utility routines.  Source here may know about the
          implementations, but ideally will not know about implementations
          for other components.

*/


