(sec_getting_started)=

# Getting Started

PETSc consists of a collection of classes,
which are discussed in detail in later parts of the manual ({doc}`programming` and {doc}`additional`).
The important PETSc classes include

- index sets (`IS`), for indexing into
  vectors, renumbering, permuting, etc;
- {any}`ch_vectors` (`Vec`);
- (generally sparse) {any}`ch_matrices` (`Mat`)
- {any}`ch_ksp` (`KSP`);
- preconditioners, including multigrid, block solvers, patch solvers, and
  sparse direct solvers (`PC`);
- {any}`ch_snes` (`SNES`);
- {any}`ch_ts` for solving time-dependent (nonlinear) PDEs, including
  support for differential-algebraic-equations, and the computation of
  adjoints (sensitivities/gradients of the solutions) (`TS`);
- scalable {any}`ch_tao` including a rich set of gradient-based optimizers,
  Newton-based optimizers and optimization with constraints (`Tao`).
- {any}ch_regressor (`PetscRegressor)`
- {any}`ch_dmbase` code for managing interactions between mesh data structures and vectors,
  matrices, and solvers (`DM`);

Each class consists of an abstract interface (simply a set of calling
sequences corresponding to an abstract base class in C++) and an implementation for each algorithm and data structure.
This design enables easy comparison and use of different
algorithms (for example, experimenting with different Krylov subspace
methods, preconditioners, or truncated Newton methods). Hence, PETSc
provides a rich environment for modeling scientific applications as well
as for rapid algorithm design and prototyping.

The classes enable easy customization and extension of both algorithms
and implementations. This approach promotes code reuse and flexibility.
The PETSc infrastructure creates a foundation for building large-scale
applications.

It is useful to consider the interrelationships among different pieces
of PETSc. {any}`fig_library` is a diagram of some
of these pieces. The figure illustrates the library’s hierarchical
organization, enabling users to employ the most appropriate solvers for a particular problem.

:::{figure} /images/manual/library_structure.svg
:alt: PETSc numerical libraries
:name: fig_library

Numerical Libraries in PETSc
:::

## Suggested Reading

The manual is divided into four parts:

- {doc}`introduction`
- {doc}`programming`
- {doc}`dm`
- {doc}`additional`

{doc}`introduction` describes the basic procedure for using the PETSc library and
presents simple examples of solving linear systems with PETSc. This
section conveys the typical style used throughout the library and
enables the application programmer to begin using the software
immediately.

{doc}`programming` explains in detail the use of the various PETSc algebraic objects, such
as vectors, matrices, index sets, and PETSc solvers, including linear and nonlinear solvers, time integrators,
and optimization support.

{doc}`dm` details how a user's models and discretizations can easily be interfaced with the
solvers by using the `DM` construct.

{doc}`additional` describes a variety of useful information, including
profiling, the options database, viewers, error handling, and some
details of PETSc design.

[Visual Studio Code](https://code.visualstudio.com/), Eclipse, Emacs, and Vim users may find their development environment's options for
searching in the source code are
useful for exploring the PETSc source code. Details of this feature are provided in {any}`sec_developer_environments`.

**Note to Fortran Programmers**: In most of the manual, the examples and calling sequences are given
for the C/C++ family of programming languages. However, Fortran
programmers can use all of the functionality of PETSc from Fortran,
with only minor differences in the user interface.
{any}`ch_fortran` provides a discussion of the differences between
using PETSc from Fortran and C, as well as several complete Fortran
examples.

**Note to Python Programmers**: To program with PETSc in Python, you need to enable Python bindings
(i.e. petsc4py) with the configure option `--with-petsc4py=1`. See the
{doc}`PETSc installation guide </install/index>`
for more details.

(sec_running)=

## Running PETSc Programs

Before using PETSc, the user must first set the environmental variable
`PETSC_DIR` to indicate the full path of the PETSc home directory. For
example, under the Unix bash shell, a command of the form

```console
$ export PETSC_DIR=$HOME/petsc
```

can be placed in the user’s `.bashrc` or other startup file. In
addition, the user may need to set the environment variable
`$PETSC_ARCH` to specify a particular configuration of the PETSc
libraries. Note that `$PETSC_ARCH` is just a name selected by the
installer to refer to the libraries compiled for a particular set of
compiler options and machine type. Using different values of
`$PETSC_ARCH` allows one to switch between several different sets (say
debug and optimized versions) of libraries easily. To determine if you need to
set `$PETSC_ARCH`, look in the directory indicated by `$PETSC_DIR`, if
there are subdirectories beginning with `arch` then those
subdirectories give the possible values for `$PETSC_ARCH`.

See {any}`handson` to immediately jump in and run PETSc code.

All PETSc programs use the MPI (Message Passing Interface) standard for
message-passing communication {cite}`mpi-final`. Thus, to
execute PETSc programs, users must know the procedure for beginning MPI
jobs on their selected computer system(s). For instance, when using the
[MPICH](https://www.mpich.org/) implementation of MPI and many
others, the following command initiates a program that uses eight
processors:

```console
$ mpiexec -n 8 ./petsc_program_name petsc_options
```

PETSc also provides a script that automatically uses the correct
`mpiexec` for your configuration.

```console
$ $PETSC_DIR/lib/petsc/bin/petscmpiexec -n 8 ./petsc_program_name petsc_options
```

Certain options are supported by all PETSc programs. We list a few
particularly useful ones below; a complete list can be obtained by
running any PETSc program with the option `-help`.

- `-log_view` - summarize the program’s performance (see {any}`ch_profiling`)
- `-fp_trap` - stop on floating-point exceptions; for example divide
  by zero
- `-malloc_dump` - enable memory tracing; dump list of unfreed memory
  at conclusion of the run, see
  {any}`detecting_memory_problems`,
- `-malloc_debug` - enable memory debugging (by default, this is
  activated for the debugging version of PETSc), see
  {any}`detecting_memory_problems`,
- `-start_in_debugger` `[noxterm,gdb,lldb]`
  `[-display name]` - start all (or a subset of the) processes in a debugger. See
  {any}`sec_debugging`, for more information on
  debugging PETSc programs.
- `-on_error_attach_debugger` `[noxterm,gdb,lldb]`
  `[-display name]` - start debugger only on encountering an error
- `-info` - print a great deal of information about what the program
  is doing as it runs
- `-version` - display the version of PETSc being used

(sec_writing)=

## Writing PETSc Programs

Most PETSc programs begin with a call to

```
PetscInitialize(int *argc,char ***argv,char *file,char *help);
```

which initializes PETSc and MPI. The arguments `argc` and `argv` are
the usual command line arguments in C and C++ programs. The
argument `file` optionally indicates an alternative name for the PETSc
options file, `.petscrc`, which resides by default in the user’s home
directory. {any}`sec_options` provides details
regarding this file and the PETSc options database, which can be used
for runtime customization. The final argument, `help`, is an optional
character string that will be printed if the program is run with the
`-help` option. In Fortran, the initialization command has the form

```fortran
call PetscInitialize(character(*) file,integer ierr)
```

where the file argument is optional.

`PetscInitialize()` automatically calls `MPI_Init()` if MPI has not
been not previously initialized. In certain circumstances in which MPI
needs to be initialized directly (or is initialized by some other
library), the user can first call `MPI_Init()` (or have the other
library do it), and then call `PetscInitialize()`. By default,
`PetscInitialize()` sets the PETSc “world” communicator
`PETSC_COMM_WORLD` to `MPI_COMM_WORLD`.

For those unfamiliar with MPI, a *communicator* indicates
a collection of processes that will be involved in a
calculation or communication. Communicators have the variable type
`MPI_Comm`. In most cases, users can employ the communicator
`PETSC_COMM_WORLD` to indicate all processes in a given run and
`PETSC_COMM_SELF` to indicate a single process.

MPI provides routines for generating new communicators consisting of
subsets of processors, though most users rarely need to use these. The
book *Using MPI*, by Lusk, Gropp, and Skjellum
{cite}`using-mpi` provides an excellent introduction to the
concepts in MPI. See also the [MPI homepage](https://www.mcs.anl.gov/research/projects/mpi/).
Note that PETSc users
need not program much message passing directly with MPI, but they must
be familiar with the basic concepts of message passing and distributed
memory computing.

All PETSc programs should call `PetscFinalize()` as their final (or
nearly final) statement. This routine handles options to be called at the conclusion of the
program and calls `MPI_Finalize()` if `PetscInitialize()` began
MPI. If MPI was initiated externally from PETSc (by either the user or
another software package), the user is responsible for calling
`MPI_Finalize()`.

### Error Checking

Most PETSc functions return a `PetscErrorCode`, an integer
indicating whether an error occurred during the call. The error code
is set to be nonzero if an error has been detected; otherwise, it is
zero. For the C/C++ interface, the error variable is the routine’s
return value, while for the Fortran version, each PETSc routine has an integer error variable as
its final argument.

One should always check these routine values as given below in the C/C++
formats, respectively:

```c
PetscCall(PetscFunction(Args));
```

or for Fortran

```fortran
! within the main program
PetscCallA(PetscFunction(Args,ierr))
```

```fortran
! within any subroutine
PetscCall(PetscFunction(Args,ierr))
```

These macros check the returned error code, and if it is nonzero, they call the PETSc error
handler and then return from the function with the error code. The macros above should be used on all PETSc calls to enable
a complete error traceback. See {any}`sec_error2` for more details on PETSc error handling.

(sec_simple)=

## Simple PETSc Examples

To help the user use PETSc immediately, we begin with a simple
uniprocessor example that
solves the one-dimensional Laplacian problem with finite differences.
This sequential code illustrates the solution of
a linear system with `KSP`, the interface to the preconditioners,
Krylov subspace methods and direct linear solvers of PETSc. Following
the code, we highlight a few of the most important parts of this example.

:::{admonition} Listing: <a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex1.c.html">KSP Tutorial src/ksp/ksp/tutorials/ex1.c</a>
:name: ksp-ex1

```{literalinclude} /../src/ksp/ksp/tutorials/ex1.c
:end-before: /*TEST
```
:::

### Include Files

The C/C++ include files for PETSc should be used via statements such as

```
#include <petscksp.h>
```

where <a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petscksp.h.html">petscksp.h</a>
is the include file for the linear solver library.
Each PETSc program must specify an include file corresponding to the
highest level PETSc objects needed within the program; all of the
required lower level include files are automatically included within the
higher level files. For example, <a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petscksp.h.html">petscksp.h</a> includes
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petscmat.h.html">petscmat.h</a>
(matrices),
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petscvec.h.html">petscvec.h</a>
(vectors), and
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petscsys.h.html">petscsys.h</a>
(base PETSc
file). The PETSc include files are located in the directory
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/index.html">\$PETSC_DIR/include</a>.
See {any}`sec_fortran_includes`
for a discussion of PETSc include files in Fortran programs.

(the_options_database)=

### The Options Database

As shown in {any}`sec_simple`, the user can
input control data at run time using the options database. In this
example the command `PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);`
checks whether the user has provided a command line option to set the
value of `n`, the problem dimension. If so, the variable `n` is set
accordingly; otherwise, `n` remains unchanged. A complete description
of the options database may be found in {any}`sec_options`.

(sec_vecintro)=

### Vectors

One creates a new parallel or sequential vector, `x`, of global
dimension `M` with the commands

```
VecCreate(MPI_Comm comm,Vec *x);
VecSetSizes(Vec x, PetscInt m, PetscInt M);
```

where `comm` denotes the MPI communicator and `m` is the optional
local size which may be `PETSC_DECIDE`. The type of storage for the
vector may be set with either calls to `VecSetType()` or
`VecSetFromOptions()`. Additional vectors of the same type can be
formed with

```
VecDuplicate(Vec old,Vec *new);
```

The commands

```
VecSet(Vec x,PetscScalar value);
VecSetValues(Vec x,PetscInt n,PetscInt *indices,PetscScalar *values,INSERT_VALUES);
```

respectively set all the components of a vector to a particular scalar
value and assign a different value to each component. More detailed
information about PETSc vectors, including their basic operations,
scattering/gathering, index sets, and distributed arrays is available
in Chapter {any}`ch_vectors`.

Note the use of the PETSc variable type `PetscScalar` in this example.
`PetscScalar` is defined to be `double` in C/C++ (or
correspondingly `double precision` in Fortran) for versions of PETSc
that have *not* been compiled for use with complex numbers. The
`PetscScalar` data type enables identical code to be used when the
PETSc libraries have been compiled for use with complex numbers.
{any}`sec_complex` discusses the use of complex
numbers in PETSc programs.

(sec_matintro)=

### Matrices

The usage of PETSc matrices and vectors is similar. The user can create a
new parallel or sequential matrix, `A`, which has `M` global rows
and `N` global columns, with the routines

```
MatCreate(MPI_Comm comm,Mat *A);
MatSetSizes(Mat A,PETSC_DECIDE,PETSC_DECIDE,PetscInt M,PetscInt N);
```

where the matrix format can be specified at runtime via the options
database. The user could alternatively specify each processes’ number of
local rows and columns using `m` and `n`.

```
MatSetSizes(Mat A,PetscInt m,PetscInt n,PETSC_DETERMINE,PETSC_DETERMINE);
```

Generally, one then sets the “type” of the matrix, with, for example,

```
MatSetType(A,MATAIJ);
```

This causes the matrix `A` to use the compressed sparse row storage
format to store the matrix entries. See `MatType` for a list of all
matrix types. Values can then be set with the command

```
MatSetValues(Mat A,PetscInt m,PetscInt *im,PetscInt n,PetscInt *in,PetscScalar *values,INSERT_VALUES);
```

After all elements have been inserted into the matrix, it must be
processed with the pair of commands

```
MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
```

{any}`ch_matrices` discusses various matrix formats as
well as the details of some basic matrix manipulation routines.

### Linear Solvers

After creating the matrix and vectors that define a linear system,
`Ax` $=$ `b`, the user can then use `KSP` to solve the
system with the following sequence of commands:

```
KSPCreate(MPI_Comm comm,KSP *ksp);
KSPSetOperators(KSP ksp,Mat Amat,Mat Pmat);
KSPSetFromOptions(KSP ksp);
KSPSolve(KSP ksp,Vec b,Vec x);
KSPDestroy(KSP ksp);
```

The user first creates the `KSP` context and sets the operators
associated with the system (matrix that defines the linear system,
`Amat` and matrix from which the preconditioner is constructed,
`Pmat` ). The user then sets various options for customized solutions,
solves the linear system, and finally destroys the `KSP` context. The command `KSPSetFromOptions()` enables the user to
customize the linear solution method at runtime using the options
database, which is discussed in {any}`sec_options`. Through this database, the
user not only can select an iterative method and preconditioner, but
can also prescribe the convergence tolerance, set various monitoring
routines, etc. (see, e.g., {any}`sec_profiling_programs`).

{any}`ch_ksp` describes in detail the `KSP` package,
including the `PC` and `KSP` packages for preconditioners and Krylov
subspace methods.

### Nonlinear Solvers

PETSc provides
an interface to tackle nonlinear problems called `SNES`.
{any}`ch_snes` describes the nonlinear
solvers in detail. We highly recommend most PETSc users work directly with
`SNES`, rather than using PETSc for the linear problem and writing their own
nonlinear solver. Similarly, users should use `TS` rather than rolling their own time integrators.

(sec_error2)=

### Error Checking

As noted above, PETSc functions return a `PetscErrorCode`, which is an integer
indicating whether an error has occurred during the call. Below, we indicate a traceback
generated by error detection within a sample PETSc program. The error
occurred on line 3618 of the file
`$PETSC_DIR/src/mat/impls/aij/seq/aij.c` and was caused by trying to
allocate too large an array in memory. The routine was called in the
program `ex3.c` on line 66. See
{any}`sec_fortran_errors` for details regarding error checking
when using the PETSc Fortran interface.

```none
$ cd $PETSC_DIR/src/ksp/ksp/tutorials
$ make ex3
$ mpiexec -n 1 ./ex3 -m 100000
[0]PETSC ERROR: --------------------- Error Message --------------------------------
[0]PETSC ERROR: Out of memory. This could be due to allocating
[0]PETSC ERROR: too large an object or bleeding by not properly
[0]PETSC ERROR: destroying unneeded objects.
[0]PETSC ERROR: Memory allocated 11282182704 Memory used by process 7075897344
[0]PETSC ERROR: Try running with -malloc_dump or -malloc_view for info.
[0]PETSC ERROR: Memory requested 18446744072169447424
[0]PETSC ERROR: PETSc Development Git Revision: v3.7.1-224-g9c9a9c5 Git Date: 2016-05-18 22:43:00 -0500
[0]PETSC ERROR: ./ex3 on a arch-darwin-double-debug named Patricks-MacBook-Pro-2.local by patrick Mon Jun 27 18:04:03 2016
[0]PETSC ERROR: Configure options PETSC_DIR=/Users/patrick/petsc PETSC_ARCH=arch-darwin-double-debug --download-mpich --download-f2cblaslapack --with-cc=clang --with-cxx=clang++ --with-fc=gfortran --with-debugging=1 --with-precision=double --with-scalar-type=real --with-viennacl=0 --download-c2html -download-sowing
[0]PETSC ERROR: #1 MatSeqAIJSetPreallocation_SeqAIJ() line 3618 in /Users/patrick/petsc/src/mat/impls/aij/seq/aij.c
[0]PETSC ERROR: #2 PetscTrMallocDefault() line 188 in /Users/patrick/petsc/src/sys/memory/mtr.c
[0]PETSC ERROR: #3 MatSeqAIJSetPreallocation_SeqAIJ() line 3618 in /Users/patrick/petsc/src/mat/impls/aij/seq/aij.c
[0]PETSC ERROR: #4 MatSeqAIJSetPreallocation() line 3562 in /Users/patrick/petsc/src/mat/impls/aij/seq/aij.c
[0]PETSC ERROR: #5 main() line 66 in /Users/patrick/petsc/src/ksp/ksp/tutorials/ex3.c
[0]PETSC ERROR: PETSc Option Table entries:
[0]PETSC ERROR: -m 100000
[0]PETSC ERROR: ----------------End of Error Message ------- send entire error message to petsc-maint@mcs.anl.gov----------
```

When running the debug version [^debug-footnote] of the PETSc libraries, it checks for memory corruption (writing outside of array bounds
, etc.). The macro `CHKMEMQ` can be called anywhere in the code to check
the current status of the memory for corruption. By putting several (or
many) of these macros into your code, you can usually easily track down
in what small segment of your code the corruption has occurred. One can
also use Valgrind to track down memory errors; see the [FAQ](https://petsc.org/release/faq/).

For complete error handling, calls to MPI functions should be made with `PetscCallMPI(MPI_Function(Args))`.
In Fortran subroutines use `PetscCallMPI(MPI_Function(Args, ierr))` and in Fortran main use
`PetscCallMPIA(MPI_Function(Args, ierr))`.

PETSc has a small number of C/C++-only macros that do not explicitly return error codes. These are used in the style

```c
XXXBegin(Args);
other code
XXXEnd();
```

and include `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscObjectOptionsBegin()`,
`PetscOptionsHeadBegin()`, `PetscOptionsHeadEnd()`, `PetscDrawCollectiveBegin()`, `PetscDrawCollectiveEnd()`,
`MatPreallocateEnd()`, and `MatPreallocateBegin()`. These should not be checked for error codes.
Another class of functions with the `Begin()` and `End()` paradigm
including `MatAssemblyBegin()`, and `MatAssemblyEnd()` do return error codes that should be checked.

PETSc also has a set of C/C++-only macros that return an object, or `NULL` if an error has been detected. These include
`PETSC_VIEWER_STDOUT_WORLD`, `PETSC_VIEWER_DRAW_WORLD`, `PETSC_VIEWER_STDOUT_(MPI_Comm)`, and `PETSC_VIEWER_DRAW_(MPI_Comm)`.

Finally `PetscObjectComm((PetscObject)x)` returns the communicator associated with the object `x` or `MPI_COMM_NULL` if an
error was detected.

(sec_parallel)=

# Parallel and GPU Programming

Numerical computing today has multiple levels of parallelism (concurrency).

- Low-level, single instruction multiple data (SIMD) parallelism or, somewhat similar, on-GPU parallelism,
- medium-level, multiple instruction multiple data shared memory parallelism (thread parallelism), and
- high-level, distributed memory parallelism.

Traditional CPUs support the lower two levels via, for example, Intel AVX-like instructions ({any}`sec_cpu_simd`) and Unix threads, often managed by using OpenMP pragmas ({any}`sec_cpu_openmp`),
(or multiple processes). GPUs also support the lower two levels via kernel functions ({any}`sec_gpu_kernels`) and streams ({any}`sec_gpu_streams`).
Distributed memory parallelism is created by combining multiple
CPUs and/or GPUs and using MPI for communication ({any}`sec_mpi`).

In addition, there is also concurrency between computations (floating point operations) and data movement (from memory to caches and registers
and via MPI between distinct memory nodes).

PETSc supports all these parallelism levels, but its strongest support is for MPI-based distributed memory parallelism.

(sec_mpi)=

## MPI Parallelism

Since PETSc uses the message-passing model for parallel programming and
employs MPI for all interprocessor communication, the user can
employ MPI routines as needed throughout an application code. However,
by default, the user is shielded from many of the details of message
passing within PETSc since these are hidden within parallel objects,
such as vectors, matrices, and solvers. In addition, PETSc provides
tools such as vector scatter and gather to assist in the
management of parallel data.

Recall that the user must specify a communicator upon creation of any
PETSc object (such as a vector, matrix, or solver) to indicate the
processors over which the object is to be distributed. For example, as
mentioned above, some commands for matrix, vector, and linear solver
creation are:

```
MatCreate(MPI_Comm comm,Mat *A);
VecCreate(MPI_Comm comm,Vec *x);
KSPCreate(MPI_Comm comm,KSP *ksp);
```

The creation routines are collective on all processes in the
communicator; thus, all processors in the communicator *must* call the
creation routine. In addition, if a sequence of collective routines is
being used, they *must* be called in the same order on each process.

The next example, given below,
illustrates the solution of a linear system in parallel. This code,
corresponding to
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex2.c.html">KSP Tutorial ex2</a>,
handles the two-dimensional Laplacian discretized with finite
differences, where the linear system is again solved with KSP. The code
performs the same tasks as the sequential version within
{any}`sec_simple`. Note that the user interface
for initiating the program, creating vectors and matrices, and solving
the linear system is *exactly* the same for the uniprocessor and
multiprocessor examples. The primary difference between the examples in
{any}`sec_simple` and
here is each processor forms only its
local part of the matrix and vectors in the parallel case.

:::{admonition} Listing: <a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex2.c.html">KSP Tutorial src/ksp/ksp/tutorials/ex2.c</a>
:name: ksp-ex2

```{literalinclude} /../src/ksp/ksp/tutorials/ex2.c
:end-before: /*TEST
```
:::

(sec_cpu_simd)=

## CPU SIMD parallelism

SIMD parallelism occurs most commonly in the Intel advanced vector extensions (AVX) families of instructions (see [Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)).
It may be automatically used by the optimizing compiler or in low-level libraries that PETSc uses, such as BLAS
(see [BLIS](https://github.com/flame/blis)), or rarely,
directly in PETSc C/C++ code, as in [MatMult_SeqSELL](https://petsc.org/main/src/mat/impls/sell/seq/sell.c.html#MatMult_SeqSELL).

(sec_cpu_openmp)=

## CPU OpenMP parallelism

OpenMP parallelism is thread parallelism. Multiple threads (independent streams of instructions) process data and perform computations on different
parts of memory that is
shared (accessible) to all of the threads. The OpenMP model is based on inserting pragmas into code, indicating that a series of instructions
(often within a loop) can be run in parallel. This is also called a fork-join model of parallelism since much of the code remains sequential and only the
computationally expensive parts in the 'parallel region' are parallel. Thus, OpenMP makes it relatively easy to add some
parallelism to a conventional sequential code in a shared memory environment.

POSIX threads (pthreads) is a library that may be called from C/C++. The library contains routines to create, join, and remove threads, plus manage communications and
synchronizations between threads. Pthreads is rarely used directly in numerical libraries and applications. Sometimes OpenMP is implemented on top of pthreads.

If one adds
OpenMP parallelism to an MPI code, one must not over-subscribe the hardware resources. For example, if MPI already has one MPI process (rank)
per hardware core, then
using four OpenMP threads per MPI process will slow the code down since now one core must switch back and forth between four OpenMP threads.

For application codes that use certain external packages, including BLAS/LAPACK, SuperLU_DIST, MUMPS, MKL, and SuiteSparse, one can build PETSc and these
packages to take advantage of OpenMP by using the configure option `--with-openmp`. The number of OpenMP threads used in the application can be controlled with
the PETSc command line option `-omp_num_threads <num>` or the environmental variable `OMP_NUM_THREADS`. Running a PETSc program with `-omp_view` will display the
number of threads used. The default number is often absurdly high for the given hardware, so we recommend always setting it appropriately.

Users can also put OpenMP pragmas into their own code. However, since standard PETSc is not thread-safe, they should not, in general,
call PETSc routines from inside the parallel regions.

There is an OpenMP thread-safe subset of PETSc that may be configured for using `--with-threadsafety` (often used along with `--with-openmp` or
`--download-concurrencykit`). <a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex61f.F90.html">KSP Tutorial ex61f</a> demonstrates
how this may be used with OpenMP. In this mode, one may have individual OpenMP threads that each manage their own
(sequential) PETSc objects (each thread can interact only with its own objects). This
is useful when one has many small systems (or sets of ODEs) that must be integrated in an
"embarrassingly parallel" fashion on multicore systems.

The ./configure option `--with-openmp-kernels` causes some PETSc numerical kernels to be compiled using OpenMP pragmas to take advantage of multiple cores.
One must be careful to ensure the number of threads used by each MPI process **times** the number of MPI processes is less than the number of
cores on the system; otherwise the code will slow down dramatically.

PETSc's MPI-based linear solvers may be accessed from a sequential or non-MPI OpenMP program, see {any}`sec_pcmpi`.

:::{seealso}
Edward A. Lee, [The Problem with Threads](https://digitalassets.lib.berkeley.edu/techreports/ucb/text/EECS-2006-1.pdf), Technical Report No. UCB/EECS-2006-1 January [[DOI]](https://doi.org/10.1109/MC.2006.180)
10, 2006
:::

(sec_gpu_kernels)=

## GPU kernel parallelism

GPUs offer at least two levels of clearly defined parallelism. Kernel-level parallelism is much like SIMD parallelism applied to loops;
many "iterations" of the loop index run on different hardware in "lock-step".
PETSc utilizes this parallelism with three similar but slightly different models:

- CUDA, which is provided by NVIDIA and runs on NVIDIA GPUs
- HIP, provided by AMD, which can, in theory, run on both AMD and NVIDIA GPUs
- and Kokkos, an open-source package that provides a slightly higher-level programming model to utilize GPU kernels.

To utilize this one configures PETSc with either `--with-cuda` or `--with-hip` and, if they plan to use Kokkos, also `--download-kokkos --download-kokkos-kernels`.

In the GPU programming model that PETSc uses, the GPU memory is distinct from the CPU memory. This means that data that resides on the CPU
memory must be copied to the GPU (often, this copy is done automatically by the libraries, and the user does not need to manage it)
if one wishes to use the GPU computational power on it. This memory copy is slow compared to the GPU speed; hence, it is crucial to minimize these copies. This often
translates to trying to do almost all the computation on the GPU and not constantly switching between computations on the CPU and the GPU on the same data.

PETSc utilizes GPUs by providing vector and matrix classes (Vec and Mat) specifically written to run on the GPU. However, since it is difficult to
write an entire PETSc code that runs only on the GPU, one can also access and work with (for example, put entries into) the vectors and matrices
on the CPU. The vector classes
are `VECCUDA`, `MATAIJCUSPARSE`, `VECKOKKOS`, `MATAIJKOKKOS`, and `VECHIP` (matrices are not yet supported by PETSc with HIP).

More details on using GPUs from PETSc will follow in this document.

(sec_gpu_streams)=

## GPU stream parallelism

Please contribute to this document.

```{raw} latex
\newpage
```

# Compiling and Running Programs

The output below illustrates compiling and running a
PETSc program using MPICH on a macOS laptop. Note that different
machines will have compilation commands as determined by the
configuration process. See {any}`sec_writing_application_codes` for
a discussion about how to compile your PETSc programs. Users who are
experiencing difficulties linking PETSc programs should refer to the [FAQ](https://petsc.org/release/faq/).

```none
$ cd $PETSC_DIR/src/ksp/ksp/tutorials
$ make ex2
/Users/patrick/petsc/arch-debug/bin/mpicc -o ex2.o -c -g3   -I/Users/patrick/petsc/include -I/Users/patrick/petsc/arch-debug/include `pwd`/ex2.c
/Users/patrick/petsc/arch-debug/bin/mpicc -g3  -o ex2 ex2.o  -Wl,-rpath,/Users/patrick/petsc/arch-debug/lib -L/Users/patrick/petsc/arch-debug/lib  -lpetsc -lf2clapack -lf2cblas -lmpifort -lgfortran -lgcc_ext.10.5 -lquadmath -lm -lclang_rt.osx -lmpicxx -lc++ -ldl -lmpi -lpmpi -lSystem
/bin/rm -f ex2.o
$ $PETSC_DIR/lib/petsc/bin/petscmpiexec -n 1 ./ex2
Norm of error 0.000156044 iterations 6
$ $PETSC_DIR/lib/petsc/bin/petscmpiexec -n 2 ./ex2
Norm of error 0.000411674 iterations 7
```

(sec_profiling_programs)=

# Profiling Programs

The option
`-log_view` activates printing of a performance summary, including
times, floating point operation (flop) rates, and message-passing
activity. {any}`ch_profiling` provides details about
profiling, including the interpretation of the output data below.
This particular example involves
the solution of a linear system on one processor using GMRES and ILU.
The low floating point operation (flop) rates in this example are because the code solved a tiny system. We include this example
merely to demonstrate the ease of extracting performance information.

(listing_exprof)=

```none
$ $PETSC_DIR/lib/petsc/bin/petscmpiexec -n 1 ./ex1 -n 1000 -pc_type ilu -ksp_type gmres -ksp_rtol 1.e-7 -log_view
...
------------------------------------------------------------------------------------------------------------------------
Event                Count      Time (sec)     Flops                             --- Global ---  --- Stage ----  Total
                   Max Ratio  Max     Ratio   Max  Ratio  Mess   AvgLen  Reduct  %T %F %M %L %R  %T %F %M %L %R Mflop/s
------------------------------------------------------------------------------------------------------------------------

VecMDot                1 1.0 3.2830e-06 1.0 2.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  0  5  0  0  0   0  5  0  0  0   609
VecNorm                3 1.0 4.4550e-06 1.0 6.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  0 14  0  0  0   0 14  0  0  0  1346
VecScale               2 1.0 4.0110e-06 1.0 2.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  0  5  0  0  0   0  5  0  0  0   499
VecCopy                1 1.0 3.2280e-06 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0
VecSet                11 1.0 2.5537e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  2  0  0  0  0   2  0  0  0  0     0
VecAXPY                2 1.0 2.0930e-06 1.0 4.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  0 10  0  0  0   0 10  0  0  0  1911
VecMAXPY               2 1.0 1.1280e-06 1.0 4.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  0 10  0  0  0   0 10  0  0  0  3546
VecNormalize           2 1.0 9.3970e-06 1.0 6.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  1 14  0  0  0   1 14  0  0  0   638
MatMult                2 1.0 1.1177e-05 1.0 9.99e+03 1.0 0.0e+00 0.0e+00 0.0e+00  1 24  0  0  0   1 24  0  0  0   894
MatSolve               2 1.0 1.9933e-05 1.0 9.99e+03 1.0 0.0e+00 0.0e+00 0.0e+00  1 24  0  0  0   1 24  0  0  0   501
MatLUFactorNum         1 1.0 3.5081e-05 1.0 4.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  2 10  0  0  0   2 10  0  0  0   114
MatILUFactorSym        1 1.0 4.4259e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  3  0  0  0  0   3  0  0  0  0     0
MatAssemblyBegin       1 1.0 8.2015e-08 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0
MatAssemblyEnd         1 1.0 3.3536e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  2  0  0  0  0   2  0  0  0  0     0
MatGetRowIJ            1 1.0 1.5960e-06 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0
MatGetOrdering         1 1.0 3.9791e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  3  0  0  0  0   3  0  0  0  0     0
MatView                2 1.0 6.7909e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  5  0  0  0  0   5  0  0  0  0     0
KSPGMRESOrthog         1 1.0 7.5970e-06 1.0 4.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00  1 10  0  0  0   1 10  0  0  0   526
KSPSetUp               1 1.0 3.4424e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  2  0  0  0  0   2  0  0  0  0     0
KSPSolve               1 1.0 2.7264e-04 1.0 3.30e+04 1.0 0.0e+00 0.0e+00 0.0e+00 19 79  0  0  0  19 79  0  0  0   121
PCSetUp                1 1.0 1.5234e-04 1.0 4.00e+03 1.0 0.0e+00 0.0e+00 0.0e+00 11 10  0  0  0  11 10  0  0  0    26
PCApply                2 1.0 2.1022e-05 1.0 9.99e+03 1.0 0.0e+00 0.0e+00 0.0e+00  1 24  0  0  0   1 24  0  0  0   475
------------------------------------------------------------------------------------------------------------------------

Memory usage is given in bytes:

Object Type          Creations   Destructions     Memory  Descendants' Mem.
Reports information only for process 0.

--- Event Stage 0: Main Stage

              Vector     8              8        76224     0.
              Matrix     2              2       134212     0.
       Krylov Solver     1              1        18400     0.
      Preconditioner     1              1         1032     0.
           Index Set     3              3        10328     0.
              Viewer     1              0            0     0.
========================================================================================================================
...
```

(sec_writing_application_codes)=

# Writing C/C++ or Fortran Applications

The examples throughout the library demonstrate the software usage and
can serve as templates for developing custom applications. We suggest
that new PETSc users examine programs in the directories
`$PETSC_DIR/src/<library>/tutorials` where `<library>` denotes any
of the PETSc libraries (listed in the following section), such as
`SNES` or `KSP`, `TS`, or `TAO`. The manual pages at
<https://petsc.org/release/documentation/> provide links (organized by
routine names and concepts) to the tutorial examples.

To develop an application program that uses PETSc, we suggest the following:

- {ref}`Download <doc_download>` and {ref}`install <doc_install>` PETSc.

- For completely new applications

  > 1. Make a directory for your source code: for example, `mkdir $HOME/application`
  >
  > 2. Change to that directory, for
  >    example, `cd $HOME/application`
  >
  > 3. Copy an example in the directory that corresponds to the
  >    problems of interest into your directory, for
  >    example, `cp $PETSC_DIR/src/snes/tutorials/ex19.c app.c`
  >
  > 4. Select an application build process. The `PETSC_DIR` (and `PETSC_ARCH` if the `--prefix=directoryname`
  >    option was not used when configuring PETSc) environmental variable(s) must be
  >    set for any of these approaches.
  >
  >    - make (recommended). It uses the [pkg-config](https://en.wikipedia.org/wiki/Pkg-config) tool
  >      and is the recommended approach. Copy \$PETSC_DIR/share/petsc/Makefile.user or \$PETSC_DIR/share/petsc/Makefile.basic.user
  >      to your directory, for example, `cp $PETSC_DIR/share/petsc/Makefile.user makefile`
  >
  >      Examine the comments in this makefile.
  >
  >      Makefile.user uses the [pkg-config](https://en.wikipedia.org/wiki/Pkg-config) tool and is the recommended approach.
  >
  >      Use `make app` to compile your program.
  >
  >    - CMake. Copy \$PETSC_DIR/share/petsc/CMakeLists.txt to your directory, for example, `cp $PETSC_DIR/share/petsc/CMakeLists.txt CMakeLists.txt`
  >
  >      Edit CMakeLists.txt, read the comments on usage, and change the name of the application from ex1 to your application executable name.
  >
  > 5. Run the program, for example,
  >    `./app`
  >
  > 6. Start to modify the program to develop your application.

- For adding PETSc to an existing application

  > 1. Start with a working version of your code that you build and run to confirm that it works.
  >
  > 2. Upgrade your build process. The `PETSC_DIR` (and `PETSC_ARCH` if the `--prefix=directoryname`
  >    option was not used when configuring PETSc) environmental variable(s) must be
  >    set for any of these approaches.
  >
  >    - Using make. Update the application makefile to add the appropriate PETSc include
  >      directories and libraries.
  >
  >      - Recommended approach. Examine the comments in \$PETSC_DIR/share/petsc/Makefile.user and transfer selected portions of
  >        that file to your makefile.
  >
  >      - Minimalist. Add the line
  >
  >        ```console
  >        include ${PETSC_DIR}/lib/petsc/conf/variables
  >        ```
  >
  >        to the bottom of your makefile. This will provide a set of PETSc-specific make variables you may use in your makefile. See
  >        the comments in the file \$PETSC_DIR/share/petsc/Makefile.basic.user for details on the usage.
  >
  >      - Simple, but hands the build process over to PETSc's control. Add the lines
  >
  >        ```console
  >        include ${PETSC_DIR}/lib/petsc/conf/variables
  >        include ${PETSC_DIR}/lib/petsc/conf/rules
  >        ```
  >
  >        to the bottom of your makefile. See the comments in the file \$PETSC_DIR/share/petsc/Makefile.basic.user for details on the usage.
  >        Since PETSc's rules now control the build process, you will likely need to simplify and remove much of the material that is in
  >        your makefile.
  >
  >      - Not recommended since you must change your makefile for each new configuration/computing system. This approach does not require
  >        the environmental variable `PETSC_DIR` to be set when building your application since the information will be hardwired in your
  >        makefile. Run the following command in the PETSc root directory to get the information needed by your makefile:
  >
  >        ```console
  >        $ make getlinklibs getincludedirs getcflags getcxxflags getfortranflags getccompiler getfortrancompiler getcxxcompiler
  >        ```
  >
  >        All the libraries listed need to be linked into your executable, and the
  >        include directories and flags need to be passed to the compiler(s). Usually,
  >        this is done by setting `LDFLAGS=<list of library flags and libraries>` and
  >        `CFLAGS=<list of -I and other flags>` and `FFLAGS=<list of -I and other flags>` etc in your makefile.
  >
  >    - Using CMake. Update the application CMakeLists.txt by examining the code and comments in
  >      \$PETSC_DIR/share/petsc/CMakeLists.txt
  >
  > 3. Rebuild your application and ensure it still runs correctly.
  >
  > 4. Add a `PetscInitialize()` near the beginning of your code and `PetscFinalize()` near the end with appropriate include commands
  >    (and use statements in Fortran).
  >
  > 5. Rebuild your application and ensure it still runs correctly.
  >
  > 6. Slowly start utilizing PETSc functionality in your code, and ensure that your code continues to build and run correctly.

(sec_oo)=

# PETSc's Object-Oriented Design

Though PETSc has a large API, conceptually, it's rather simple.
There are three abstract basic data objects (classes): index sets, `IS`, vectors, `Vec`, and matrices, `Mat`.
Plus, a larger number of abstract algorithm objects (classes) starting with: preconditioners, `PC`, Krylov solvers, `KSP`, and so forth.

Let `Object`
represent any of these objects. Objects are created with

```
Object obj;
ObjectCreate(MPI_Comm, &obj);
```

The object is initially empty, and little can be done with it. A particular implementation of the class is associated with the object by setting the object's "type", where type
is merely a string name of an implementation class using

```
ObjectSetType(obj,"ImplementationName");
```

Some objects support subclasses, which are specializations of the type. These are set with

```
ObjectNameSetType(obj,"ImplementationSubName");
```

For example, within `TS` one may do

```
TS ts;
TSCreate(PETSC_COMM_WORLD,&ts);
TSSetType(ts,TSARKIMEX);
TSARKIMEXSetType(ts,TSARKIMEX3);
```

The abstract class `TS` can embody any ODE/DAE integrator scheme.
This example creates an additive Runge-Kutta ODE/DAE IMEX integrator, whose type name is `TSARKIMEX`, using a 3rd-order scheme with an L-stable implicit part,
whose subtype name is `TSARKIMEX3`.

To allow PETSc objects to be runtime configurable, PETSc objects provide a universal way of selecting types (classes) and subtypes at runtime from
what is referred to as the PETSc "options database". The code above can be replaced with

```
TS obj;
TSCreate(PETSC_COMM_WORLD,&obj);
TSSetFromOptions(obj);
```

now, both the type and subtype can be conveniently set from the command line

```console
$ ./app -ts_type arkimex -ts_arkimex_type 3
```

The object's type (implementation class) or subclass can also be changed at any time simply by calling `TSSetType()` again (though to override command line options, the call to `TSSetType()` must be made \_after\_ `TSSetFromOptions()`). For example:

```
// (if set) command line options "override" TSSetType()
TSSetType(ts, TSGLLE);
TSSetFromOptions(ts);

// TSSetType() overrides command line options
TSSetFromOptions(ts);
TSSetType(ts, TSGLLE);
```

Since the later call always overrides the earlier call, the second form shown is rarely -- if ever -- used, as it is less flexible than configuring command line settings.

The standard methods on an object are of the general form.

```
ObjectSetXXX(obj,...);
ObjectGetXXX(obj,...);
ObjectYYY(obj,...);
```

For example

```
TSSetRHSFunction(obj,...)
```

Particular types and subtypes of objects may have their own methods, which are given in the form

```
ObjectNameSetXXX(obj,...);
ObjectNameGetXXX(obj,...);
ObjectNameYYY(obj,...);
```

and

```
ObjectNameSubNameSetXXX(obj,...);
ObjectNameSubNameGetXXX(obj,...);
ObjectNameSubNameYYY(obj,...);
```

where Name and SubName are the type and subtype names (for example, as above `TSARKIMEX` and `3`. Most "set" operations have options database versions with the same
names in lower case, separated by underscores, and with the word "set" removed. For example,

```
KSPGMRESSetRestart(obj,30);
```

can be set at the command line with

```console
$ ./app -ksp_gmres_restart 30
```

A special subset of type-specific methods is ignored if the type does not match the function name. These are usually setter functions that control some aspect specific to the subtype.
Note that we leveraged this functionality in the MPI example above ({any}`sec_mpi`) by calling `Mat*SetPreallocation()` for a number of different matrix types. As another example,

```
KSPGMRESSetRestart(obj,30);   // ignored if the type is not KSPGMRES
```

These allow cleaner application code since it does not have many if statements to avoid inactive methods. That is, one does not need to write code like

```
if (type == KSPGMRES) {     // unneeded clutter
  KSPGMRESSetRestart(obj,30);
}
```

Many "get" routines give one temporary access to an object's internal data. They are used in the style

```
XXX xxx;
ObjectGetXXX(obj,&xxx);
// use xxx
ObjectRestoreXXX(obj,&xxx);
```

Objects obtained with a "get" routine should be returned with a "restore" routine, generally within the same function. Objects obtained with a "create" routine should be freed
with a "destroy" routine.

There may be variants of the "get" routines that give more limited access to the obtained object. For example,

```
const PetscScalar *x;

// specialized variant of VecGetArray()
VecGetArrayRead(vec, &x);
// one can read but not write with x[]
PetscReal y = 2*x[0];
// don't forget to restore x after you are done with it
VecRestoreArrayRead(vec, &x);
```

Objects can be displayed (in a large number of ways) with

```
ObjectView(obj,PetscViewer viewer);
ObjectViewFromOptions(obj,...);
```

Where `PetscViewer` is an abstract object that can represent standard output, an ASCII or binary file, a graphical window, etc. The second
variant allows the user to delay until runtime the decision of what viewer and format to use to view the object or if to view the object at all.

Objects are destroyed with

```
ObjectDestroy(&obj)
```

:::{figure} /images/manual/objectlife.svg
:name: fig_objectlife

Sample lifetime of a PETSc object
:::

## User Callbacks

The user may wish to override or provide custom functionality in many situations. This is handled via callbacks, which the library will call at the appropriate time. The most general way to apply a callback has this form:

```
ObjectCallbackSetter(obj, callbackfunction(), void *ctx, contextdestroy(void *ctx));
```

where `ObjectCallbackSetter()` is a callback setter such as `SNESSetFunction()`. `callbackfunction()` is what will be called
by the library, `ctx` is an optional data structure (array, struct, PETSc object) that is used by `callbackfunction()`
and `contextdestroy(void *ctx)` is an optional function that will be called when `obj` is destroyed to free
anything in `ctx`. The use of the `contextdestroy()` allows users to "set and forget"
data structures that will not be needed elsewhere but still need to be deleted when no longer needed. Here is an example of the use of a full-fledged callback

```
TS              ts;
TSMonitorLGCtx *ctx;

TSMonitorLGCtxCreate(..., &ctx)
TSMonitorSet(ts, TSMonitorLGTimeStep, ctx, (PetscCtxDestroyFn *)TSMonitorLGCtxDestroy);
TSSolve(ts);
```

Occasionally, routines to set callback functions take additional data objects that will be used by the object but are not context data for the function. For example,

```
SNES obj;
Vec  r;
void *ctx;

SNESSetFunction(snes, r, UserApplyFunction(SNES,Vec,Vec,void *ctx), ctx);
```

The `r` vector is an optional argument provided by the user, which will be used as work-space by `SNES`. Note that this callback does not provide a way for the user
to have the `ctx` destroyed when the `SNES` object is destroyed; the users must ensure that they free it at an appropriate time. There is no logic to the various ways
PETSc accepts callback functions in different places in the code.

See {any}`fig_taocallbacks` for a cartoon on callbacks in `Tao`.

(sec_directory)=

# Directory Structure

We conclude this introduction with an overview of the organization of
the PETSc software. The root directory of PETSc contains the following
directories:

- `doc` The source code and Python scripts for building the website and documentation

- `lib/petsc/conf` - Base PETSc configuration files that define the standard
  make variables and rules used by PETSc

- `include` - All include files for PETSc that are visible to the
  user.

- `include/petsc/finclude` - PETSc Fortran include files.

- `include/petsc/private` - Private PETSc include files that should
  *not* need to be used by application programmers.

- `share` - Some small test matrices and other data files

- `src` - The source code for all PETSc libraries, which currently
  includes

  - `vec` - vectors,

    - `is` - index sets,

  - `mat` - matrices,

  - `ksp` - complete linear equations solvers,

    - `ksp` - Krylov subspace accelerators,
    - `pc` - preconditioners,

  - `snes` - nonlinear solvers

  - `ts` - ODE/DAE solvers and timestepping,

  - `tao` - optimizers,

  - `ml` - Machine Learning

    - `regressor` - Regression solvers

  - `dm` - data management between meshes and solvers, vectors, and
    matrices,

  - `sys` - general system-related routines,

    - `logging` - PETSc logging and profiling routines,

    - `classes` - low-level classes

      - `draw` - simple graphics,
      - `viewer` - a mechanism for printing and visualizing PETSc
        objects,
      - `bag` - mechanism for saving and loading from disk user
        data stored in C structs.
      - `random` - random number generators.

Each PETSc source code library directory has the following subdirectories:

- `tutorials` - Programs designed to teach users about PETSc.
  These codes can serve as templates for applications.
- `tests` - Programs designed for thorough testing of PETSc. As
  such, these codes are not intended for examination by users.
- `interface` - Provides the abstract base classes for the objects.
  The code here does not know about particular implementations and does not perform
  operations on the underlying numerical data.
- `impls` - Source code for one or more implementations of the class for particular
  data structures or algorithms.
- `utils` - Utility routines. The source here may know about the
  implementations, but ideally, will not know about implementations for
  other components.

```{rubric} Footnotes
```

[^debug-footnote]: Configure PETSc with `--with-debugging`.

```{eval-rst}
.. bibliography:: /petsc.bib
   :filter: docname in docnames
```
