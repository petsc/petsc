

/**

   \page manual-user-page-vectors   Vectors and Distributing Parallel Data


The vector (denoted by Vec) is one of the simplest PETSc
objects.  Vectors are used to store discrete PDE solutions, right-hand
sides for linear systems, etc. This chapter is organized as follows:

  - (Vec) \ref manual-user-sec-veccreate "Creation " and \ref manual-user-sec-vecbasic " basic usage of vectors"
  - \ref manual-user-sec-indexingandordering "Management of the various numberings " of
               degrees of freedom, vertices, cells, etc.
    - (AO) Mapping between different global numberings
    - (ISLocalToGlobalMapping) Mapping between local and global numberings
  - (DM) \ref manual-user-sec-da "Management of grids"
  - (IS, VecScatter) \ref manual-user-sec-unstruct "Management of vectors related to unstructured grids"


\section manual-user-sec-veccreate Creating and Assembling Vectors

%PETSc currently provides two basic vector types: sequential and parallel
(MPI based). To create a sequential vector with `m` components,
one can
use the command
\code
  VecCreateSeq(PETSC_COMM_SELF,int m,Vec *x);
\endcode
To create a parallel vector one can either specify the number of
components that will be stored on each process or let %PETSc decide.
The command
\code
  VecCreateMPI(MPI_Comm comm,int m,int M,Vec *x);
\endcode
creates a vector that is distributed over all processes in the communicator,
comm, where `m` indicates the number
of components to store on the local process, and `M` is the
total number of vector components.  Either the local or global
dimension, but not both, can be set to `PETSC_DECIDE` to
indicate that %PETSc should determine it.
More generally, one can use the routines
\code
  VecCreate(MPI_Comm comm,Vec *v);\\
  VecSetSizes(Vec v, int m, int M);\\
  VecSetFromOptions(Vec v);
\endcode
which automatically generates the appropriate vector type
(sequential or parallel) over all processes in `comm`.
The option `-vec_type` `mpi` can be used in conjunction with
VecCreate() and VecSetFromOptions() to specify the use of MPI
vectors even for the uniprocess case.

We emphasize that all processes in `comm` **must** call the
vector creation routines, since these routines are collective over all
processes in the communicator. If you are not familar with MPI communicators,
see \ref manual-user-sec-writing " this introductory section".
In addition, if a sequence of 
VecCreateXXX() routines is used, they must be called in the same
order on each process in the communicator.

One can assign a single value to all components of a vector with the
command
\code
  VecSet(Vec x,PetscScalar value);
\endcode
Assigning values to individual components of the vector is more
complicated, in order to make it possible to write efficient parallel
code. Assigning a set of components is a two-step process: one
first calls
\code
  VecSetValues(Vec x,int n,int *indices,PetscScalar *values,INSERT_VALUES);
\endcode
any number of times on any or all of the processes. The argument
`n` gives the number of components being set in this
insertion. The integer array `indices` contains the *global component
indices*, and `values` is the array of values to be inserted.
Any process can set any components of the vector; %PETSc insures that
they are automatically stored in the correct location.
Once all of the values have been inserted with VecSetValues(),
one must call
\code
  VecAssemblyBegin(Vec x);
\endcode
followed by
\code
  VecAssemblyEnd(Vec x);
\endcode
to perform any needed message passing of nonlocal components.
In order to allow the overlap of communication and calculation,
the user's code can perform any series of other actions between these
two calls while the messages are in transition.

Example usage of VecSetValues() may be found in
`${PETSC_DIR}/src/vec/vec/examples/tutorials/ex2.c` or `ex2f.F`.

Often, rather than inserting elements in a vector, one may wish to
add values. This process
is also done with the command
\code
  VecSetValues(Vec x,int n,int *indices, PetscScalar *values,ADD_VALUES);
\endcode
Again one must call the assembly routines
VecAssemblyBegin() and VecAssemblyEnd() after all of the values
have been added.  Note that addition and insertion calls to
VecSetValues() **cannot** be mixed.  Instead, one must add and insert
vector elements in phases, with intervening calls to the assembly
routines. This phased assembly procedure overcomes the nondeterministic
behavior that
would occur if two different processes generated values
for the same location, with one process adding while the other is inserting
its value.  (In this case the addition and insertion actions could be performed
in either order,
thus resulting in different values at the particular location. Since
%PETSc does not allow the simultaneous use of INSERT_VALUES and
ADD_VALUES this nondeterministic behavior will not occur in PETSc.)

You can called VecGetValues() to pull local values from a vector (but
not off-process values),
an alternative method for extracting some components of a vector are
the vector scatter routines. \ref manual-user-sec-scatter " Read here" for details; see also
below for VecGetArray().

One can examine a vector with the command
\code
  VecView(Vec x,PetscViewer v);
\endcode
To print the vector to the screen, one can use the viewer
`PETSC_VIEWER_STDOUT_WORLD`,
which ensures that parallel vectors are printed correctly to
`stdout`. To display the vector in an X-window, one can use the
default X-windows viewer PETSC_VIEWER_DRAW_WORLD,
or one can create a viewer with the
routine PetscViewerDrawOpenX().  A variety of viewers are discussed
further \ref manual-user-sec-viewers " here".

To create a new vector of the same format as an existing vector, one uses
the command
\code
  VecDuplicate(Vec old,Vec *new);
\endcode
To create several new vectors of the same format as an existing vector,
one uses the command
\code
  VecDuplicateVecs(Vec old,int n,Vec **new);
\endcode
This routine creates an array of pointers to vectors. The two routines
are very useful because they allow one to write library code that does
not depend on the particular format of the vectors being used. Instead,
the subroutines can automatically correctly create work vectors
based on the specified existing vector.  As discussed \ref manual-user-sec-fortvecd " here", the Fortran interface for VecDuplicateVecs()
differs slightly.

When a vector is no longer needed, it should be destroyed with the
command
\code
  VecDestroy(Vec *x);
\endcode
To destroy an array of vectors, use the command
\code
  VecDestroyVecs(PetscInt n,Vec **vecs);
\endcode
Note that the Fortran interface for VecDestroyVecs() differs slightly,
as described \ref manual-user-sec-fortvecd " here".

It is also possible to create vectors that use an array provided by the user,
rather than having %PETSc internally allocate the array space.
Such vectors can be created with the routines
\code
  VecCreateSeqWithArray(PETSC_COMM_SELF,int bs,int n,PetscScalar *array,Vec *V);
\endcode
and
\code
  VecCreateMPIWithArray(MPI_Comm comm,int bs,int n,int N,PetscScalar *array,Vec *vv);
\endcode
Note that here one must provide the value `n`, it cannot be PETSC_DECIDE and
the user is responsible for providing enough space in the array; `n*sizeof(PetscScalar)`.


\section manual-user-sec-vecbasic Basic Vector Operations

\anchor manual-user-fig-vectorops

<center>
<TABLE>
<TR><TH>Function Name</TH><TH>Operation</TH></TR>

<TR><TD>VecAXPY(Vec y,PetscScalar a,Vec x);</TD><TD> \f$ y = y + a*x\f$ </TD></TR>
<TR><TD>VecAYPX(Vec y,PetscScalar a,Vec x);</TD><TD> \f$ y = x + a*y\f$ </TD></TR>
<TR><TD>VecWAXPY(Vec w,PetscScalar a,Vec x,Vec y);</TD><TD> \f$ w = a*x + y\f$ </TD></TR>
<TR><TD>VecAXPBY(Vec y,PetscScalar a,PetscScalar b,Vec x);</TD><TD> \f$ y = a*x + b*y\f$ </TD></TR>
<TR><TD>VecScale(Vec x, PetscScalar a);</TD><TD> \f$ x = a*x \f$ </TD></TR>
<TR><TD>VecDot(Vec x, Vec y, PetscScalar *r);</TD><TD> \f$ r = \bar{x}'*y\f$ </TD></TR>
<TR><TD>VecTDot(Vec x, Vec y, PetscScalar *r);</TD><TD> \f$ r = x'*y\f$ </TD></TR>
<TR><TD>VecNorm(Vec x,NormType type,  PetscReal *r);</TD><TD> \f$ r = ||x||_{type}\f$ </TD></TR>
<TR><TD>VecSum(Vec x,   PetscScalar *r);</TD><TD> \f$ r = \sum x_{i}\f$ </TD></TR>
<TR><TD>VecCopy(Vec x, Vec y);</TD><TD> \f$ y = x \f$  </TD></TR>
<TR><TD>VecSwap(Vec x, Vec y);</TD><TD> \f$ y = x \f$ while \f$ x = y\f$ </TD></TR>
<TR><TD>VecPointwiseMult(Vec w,Vec x,Vec y);</TD><TD> \f$ w_{i} = x_{i}*y_{i} \f$ </TD></TR>
<TR><TD>VecPointwiseDivide(Vec w,Vec x,Vec y);</TD><TD> \f$ w_{i} = x_{i}/y_{i} \f$ </TD></TR>
<TR><TD>\link VecMDot VecMDot\endlink(Vec x,int n,Vec y[],PetscScalar *r);</TD><TD> \f$ r[i] = \bar{x}'*y[i]\f$ </TD></TR>
<TR><TD>\link VecMTDot VecMTDot\endlink(Vec x,int n,Vec y[],PetscScalar *r);</TD><TD> \f$ r[i] = x'*y[i]\f$ </TD></TR>
<TR><TD>\link VecMAXPY VecMAXPY\endlink(Vec y,int n, PetscScalar *a, Vec x[]); </TD><TD> \f$ y = y + \sum_i a_{i}*x[i] \f$  </TD></TR>
<TR><TD>\link VecMax VecMax\endlink(Vec x,  int *idx, PetscReal *r);</TD><TD> \f$ r = \max x_{i}\f$ </TD></TR>
<TR><TD>\link VecMin VecMin\endlink(Vec x,  int *idx, PetscReal *r);</TD><TD> \f$ r = \min x_{i}\f$ </TD></TR>
<TR><TD>VecAbs(Vec x);</TD><TD> \f$ x_i = |x_{i}|\f$  </TD></TR>
<TR><TD>VecReciprocal(Vec x);</TD><TD> \f$ x_i = 1/x_{i}\f$ </TD></TR>
<TR><TD>VecShift(Vec x,PetscScalar s);</TD><TD> \f$ x_i = s + x_{i}\f$ </TD></TR>
<TR><TD>VecSet(Vec x,PetscScalar alpha);</TD><TD> \f$ x_i = \alpha\f$ </TD></TR>
</TABLE>
<b>PETSc Vector Operations</b></center>

We have chosen certain basic vector operations to support within the %PETSc vector library.
These operations were selected because they often arise in application
codes. The NormType argument to VecNorm() is one of
NORM_1, NORM_2, or NORM_INFINITY.
The 1-norm is
\f$ \sum_i |x_{i}|\f$, the 2-norm is \f$( \sum_{i} x_{i}^{2})^{1/2} \f$ and the
infinity norm is \f$ \max_{i} |x_{i}|\f$.


For parallel vectors that are distributed across the processes by ranges,
it is possible to determine
a process's local range with the routine
\code
  VecGetOwnershipRange(Vec vec,int *low,int *high);
\endcode
The argument `low` indicates the first component owned by the local
process, while `high` specifies *one more than* the
last owned by the local process.
This command is useful, for instance, in assembling parallel vectors.

On occasion, the user needs to access the actual elements of the vector.
The routine VecGetArray()
returns a pointer to the elements local to the process:
\code
  VecGetArray(Vec v,PetscScalar **array);
\endcode
When access to the array is no longer
needed, the user should call
\code
  VecRestoreArray(Vec v, PetscScalar **array);
\endcode
Minor differences exist in the Fortran interface for VecGetArray() and
VecRestoreArray(), as discussed in Section \ref{sec_fortranarrays}.
It is important to note that VecGetArray() and VecRestoreArray()
do {\em not} copy the vector elements; they merely give users direct
access to the vector elements. Thus, these routines require essentially
no time to call and can be used efficiently.

The number of elements stored locally can be accessed with
\code
  VecGetLocalSize(Vec v,int *size);
\endcode
The global vector length can be determined by
\code
  VecGetSize(Vec v,int *size);
\endcode


In addition to VecDot() and VecMDot() and VecNorm(), %PETSc provides
split phase versions of these that allow several independent inner products and/or norms
to share the same communication (thus improving parallel efficiency). For example,
one may have code such as
\code
 VecDot(Vec x,Vec y,PetscScalar *dot);\\
 VecMDot(Vec x,PetscInt nv, Vec y[],PetscScalar *dot);\\
 VecNorm(Vec x,NormType NORM_2,PetscReal *norm2);\\
 VecNorm(Vec x,NormType NORM_1,PetscReal *norm1);
\endcode
This code works fine, the problem is that it performs three separate parallel communication
operations. Instead one can write
\code
 VecDotBegin(Vec x,Vec y,PetscScalar *dot);\\
 VecMDotBegin(Vec x, PetscInt nv,Vec y[],PetscScalar *dot);\\
 VecNormBegin(Vec x,NormType NORM_2,PetscReal *norm2);\\
 VecNormBegin(Vec x,NormType NORM_1,PetscReal *norm1);\\
 VecDotEnd(Vec x,Vec y,PetscScalar *dot);\\
 VecMDotEnd(Vec x, PetscInt nv,Vec y[],PetscScalar *dot);\\
 VecNormEnd(Vec x,NormType NORM_2,PetscReal *norm2);\\
 VecNormEnd(Vec x,NormType NORM_1,PetscReal *norm1);
\endcode
With this code,
the communication is delayed until the first call to
`VecxxxEnd()` at which
a single MPI reduction is used to communicate all the required values. It is required that the
calls to the VecxxxEnd() are performed in the same order as the calls to the
VecxxxBegin(); however if you mistakenly make the calls in the wrong order PETSc
will generate an error,
informing you of this. There are additional routines VecTDotBegin() and
VecTDotEnd(), VecMTDotBegin(), VecMTDotEnd().

Note: these routines use only MPI 1 functionality; so they do not allow you to overlap
computation and communication (assuming no threads are spawned within a MPI process).
Once MPI 2 implementations are more common we'll improve these
routines to allow overlap of inner product and norm calculations with other calculations.
Also currently these routines only work for the %PETSc built in vector types.

\section manual-user-sec-indexingandordering Indexing and Ordering

  When writing parallel PDE codes there is extra complexity caused by
having multiple ways of indexing (numbering) and ordering objects such
as vertices and degrees of freedom. For example, a grid generator
or partitioner may renumber the nodes, requiring adjustment of the
other data structures that refer to these objects; see 
\ref manual-user-fig-daao " this figure".  In addition, local numbering (on a single process)
of objects may be different than the global (cross-process)
numbering. %PETSc provides a variety of tools that help to manage the
mapping among the various numbering systems. The two most basic are
the AO (application ordering), which enables mapping between
different global (cross-process) numbering schemes and the ISLocalToGlobalMapping, which allows mapping between local
(on-process) and global (cross-process) numbering.

\subsection manual-user-sec-ao Application Orderings

In many applications it is desirable to work with one or more
"orderings" (or numberings) of degrees of freedom, cells, nodes,
etc.  Doing so in a parallel environment is
complicated by the fact that each process cannot keep complete lists
of the mappings between different orderings. In addition, the
orderings used in the %PETSc linear algebra routines (often contiguous
ranges) may not correspond to the "natural" orderings for the application.

PETSc provides certain utility routines that allow one to deal cleanly
and efficiently with the various orderings. To define a new application ordering
(called an AO in PETSc), one can call the routine
\code
  AOCreateBasic(MPI_Comm comm,int n,const int apordering[],const int petscordering[],AO *ao);
\endcode
The arrays `apordering` and `petscordering`, respectively, contain a list of integers
in the application ordering and their corresponding mapped values in the %PETSc
ordering. Each process can provide whatever subset of the ordering it
chooses, but multiple processes should never contribute duplicate values.
The argument `n` indicates the number of local contributed values.

For example, consider a vector of length five, where node 0 in the application ordering
corresponds to node 3 in the %PETSc ordering.  In addition, nodes 1, 2, 3, and 4 of
the application ordering correspond, respectively, to nodes 2, 1, 4, and 0 of
the %PETSc ordering.
We can write this correspondence as
\f[
 { 0, 1, 2, 3, 4 }  \to  { 3, 2, 1, 4, 0 }.
\f]
The user can create the PETSc-AO mappings in a number of ways.  For example,
if using two processes, one could call
\code
  AOCreateBasic(PETSC_COMM_WORLD,2,{0,3},{3,4},&ao);
\endcode
on the first process and
\code
 AOCreateBasic(PETSC_COMM_WORLD,3,{1,2,4},{2,1,0},&ao);
\endcode
on the other process.

Once the application ordering has been created, it can be used
with either of the commands
\code
  AOPetscToApplication(AO ao,int n,int *indices);
  AOApplicationToPetsc(AO ao,int n,int *indices);
\endcode
Upon input, the `n`-dimensional array `indices` specifies
the indices to be mapped, while upon output, `indices` contains
the mapped values.
Since we, in general, employ a parallel database for the
AO mappings, it is crucial that all processes that
called AOCreateBasic() also call these routines; these
routines **cannot** be called by just a subset of processes
in the MPI communicator that was used in the call to AOCreateBasic().

An alternative routine to create the application ordering, AO, is
\code
  AOCreateBasicIS(IS apordering,IS petscordering,AO *ao);
\endcode
where index sets (see \ref manual-user-sec-indexset " here") are used instead of integer arrays.

The mapping routines
\code
  AOPetscToApplicationIS(AO ao,IS indices);
  AOApplicationToPetscIS(AO ao,IS indices);
\endcode
will map index sets (IS objects) between orderings. Both the `AOXxxToYyy()` and
`AOXxxToYyyIS()` routines can be used regardless of whether the AO was
created with a AOCreateBasic() or AOCreateBasicIS().

The AO context should be destroyed with AODestroy(AO *ao)
and viewed with AOView(AO ao,PetscViewer viewer).

Although we refer to the two orderings as "PETSc" and
"application" orderings, the user is free to use them both for
application orderings and to maintain relationships among a variety of
orderings by employing several AO contexts.

The `AOxxToxx()` routines allow negative entries in the input
integer array. These entries are not mapped; they simply remain
unchanged.  This functionality enables, for example, mapping neighbor
lists that use negative numbers to indicate nonexistent neighbors due
to boundary conditions, etc.

\subsection manual-user-sec-islocaltoglobalmapping Local to Global Mappings

In many applications one works with a global representation of a vector
(usually on a vector obtained with VecCreateMPI())
and a local representation of the same vector that includes ghost points
required for local computation.
%PETSc provides routines to help map indices from a local numbering scheme to
the %PETSc global numbering scheme. This is done via the following routines
\code
  ISLocalToGlobalMappingCreate(MPI_Comm comm,int N,int* globalnum,PetscCopyMode mode,ISLocalToGlobalMapping* ctx);\\
  ISLocalToGlobalMappingApply(ISLocalToGlobalMapping ctx,int n,int *in,int *out);\\
  ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping ctx,IS isin,IS* isout);\\
  ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *ctx);
\endcode
Here `N` denotes the number of local indices, `globalnum` contains the
global number of each local number, and ISLocalToGlobalMapping is the
resulting %PETSc object that contains the information needed to apply the mapping with
either ISLocalToGlobalMappingApply() or
ISLocalToGlobalMappingApplyIS().

Note that the ISLocalToGlobalMapping routines serve a different purpose
than the AO routines. In the former case they provide a mapping
from  a local numbering scheme (including ghost points) to a global numbering scheme,
while in the latter they provide a mapping between two global numbering schemes.
In fact, many applications may use both AO and ISLocalToGlobalMapping routines.
The AO routines are first used to map from an application global ordering
(that has no relationship to parallel processing etc.) to the %PETSc ordering scheme
(where each process has a contiguous set of indices in the numbering). Then in order
to perform function or Jacobian evaluations locally on each process, one works
with a local numbering scheme that includes ghost points.  The mapping from this local
numbering scheme back to the global %PETSc numbering can be handled with the
ISLocalToGlobalMapping routines.

If one is given a list of indices in a global numbering, the routine
\code
  ISGlobalToLocalMappingApply(ISLocalToGlobalMapping ctx,
                              ISGlobalToLocalMappingType type,int nin,int idxin[],int *nout,int idxout[]);
\endcode
will provide
a new list of indices in the local numbering. Again, negative values in
`idxin` are left unmapped.  But, in addition, if `type` is set to
`IS_GTOLM_MASK`,
then `nout` is set to `nin` and all global values
in `idxin` that are not represented in the local to global mapping
are replaced by -1. When `type` is set to `IS_GTOLM_DROP`,
the values in `idxin` that are not
represented locally in the mapping are not included in `idxout`, so that
potentially `nout` is smaller than `nin`.  One must
pass in an array long enough to hold all the indices. One can call
ISGlobalToLocalMappingApply() with `idxout` equal to
NULL to determine the required length (returned in
`nout`) and then allocate the required space and call
ISGlobalToLocalMappingApply() a second time to set the values.

Often it is convenient to set elements into a vector using the local node
numbering rather than the global node numbering (e.g.,  each process may
maintain its own sublist of vertices and elements and number them locally).
To set values into a vector with the local numbering, one must first call
\code
  VecSetLocalToGlobalMapping(Vec v,ISLocalToGlobalMapping ctx);
\endcode
and then call
\code
  VecSetValuesLocal(Vec x,int n,const int indices[],const PetscScalar values[],INSERT_VALUES);
\endcode
Now the `indices` use the local numbering, rather than the global, meaning
the entries lie in \f$[0,n)\f$ where \f$n\f$ is the local size of the vector.

\section manual-user-sec-da Structured Grids Using Distributed Arrays

  Distributed arrays (DMDAs), which are used in
conjunction with %PETSc vectors, are intended for use with *logically regular rectangular grids* when communication of nonlocal data is
needed before certain local computations can occur.  %PETSc distributed
arrays are designed only for the case in which data can be thought of
as being stored in a standard multidimensional array; thus, DMDAs
are **not** intended for parallelizing unstructured grid problems, etc.
DAs are intended for communicating vector (field) information; they
are not intended for storing matrices.

For example, a typical situation one encounters in solving
PDEs in parallel is that, to evaluate a local function, \f$f(x)\f$, each process
requires its local portion of the vector `x` as well as its ghost
points (the bordering portions of the vector
that are owned by neighboring processes).  \ref manual-user-fig-ghosts " Here "
is an illustration of the ghost points for the seventh process of a
two-dimensional, regular parallel grid.  Each box represents a
process; the ghost points for the seventh process's local part of
a parallel array are shown in gray.

\anchor manual-user-fig-ghosts
\image html ghost.png
<center><b>Ghost Points for Two Stencil Types on the Seventh Process</b></center>

\subsection manual-user-subsec-vec-creating-da Creating Distributed Arrays

The %PETSc DMDA object manages the parallel communication required
while working with data stored in regular arrays. The actual data
is stored in approriately sized vector objects; the DMDA object
only contains the parallel data layout information and communication
information, however it may be used to create vectors and matrices with the
proper layout.

One creates a distributed array communication data structure
in two dimensions with the command
\code
  DMDACreate2d(MPI_Comm comm,DMDABoundaryType xperiod,DMDABoundaryType yperiod,DMDAStencilType st,int M,
               int N,int m,int n,int dof,int s,int *lx,int *ly,DM *da);
\endcode
The  arguments `M` and `N` indicate the global
numbers of grid points in each direction, while `m` and `n`
denote the process partition in each direction; `m*n` must equal
the number of processes in the MPI communicator, `comm`.
Instead of specifying the process layout, one may use
PETSC_DECIDE for `m` and `n`
so that %PETSc will determine the partition using MPI. The type of
periodicity of the array is specified by `xperiod` and `yperiod`, which can be
DMDA_BOUNDARY_NONE (no periodicity),
DMDA_BOUNDARY_PERIODIC (periodic in that direction),
DMDA_BOUNDARY_GHOSTED,
or DMDA_BOUNDARY_MIRROR. The argument `dof`
indicates the number of degrees of freedom at each array point,
and `s` is the stencil width (i.e., the width of the ghost point region).
The optional arrays `lx` and `ly` may contain the number of nodes
along the x and y axis for each cell, i.e. the dimension of `lx` is
`m` and the dimension of `ly` is `n`; or NULL
may be passed in.

Two types of distributed array communication data structures
can be created, as specified by `st`.
Star-type stencils that radiate outward only in the coordinate
directions are indicated by DMDA_STENCIL_STAR,
while box-type stencils are specified by
DA_STENCIL_BOX. For example, for the
two-dimensional case,
DA_STENCIL_STAR with width 1 corresponds to the standard 5-point
stencil, while DMDA_STENCIL_BOX with width 1 denotes the
standard 9-point stencil.  In both instances the ghost points are
identical, the only difference being that with star-type stencils
certain ghost points are ignored, decreasing substantially
the number of messages sent.  Note that the DMDA_STENCIL_STAR
stencils can save interprocess communication in two and three
dimensions.

These DMDA stencils have nothing directly to do with any finite
difference stencils one might chose to use for a discretization; they
only ensure that the correct values are in place for application of a
user-defined finite difference stencil (or any other
discretization technique).

The commands for creating distributed array communication data structures
in one and three dimensions are analogous:
\code
  DMDACreate1d(MPI_Comm comm,DMDABoundaryType xperiod,int M,int w,int s,int *lc,DM *inra);
  DMDACreate3d(MPI_Comm comm,DMDABoundaryType xperiod,DMDABoundaryType yperiod, 
               DMDABoundaryType zperiod, DMDAStencilType stencil_type,
               int M,int N,int P,int m,int n,int p,int w,int s,int *lx,int *ly,int *lz,DM *inra);
\endcode
The routines to create distributed arrays are collective, so that all
processes in the communicator `comm` must call `DACreateXXX()`.

\subsection manual-user-subsec-vec-locglob Local/Global Vectors and Scatters

Each DMDA object defines the layout of two vectors: a distributed
global vector and a local vector that includes room for the
appropriate ghost points.  The DMDA object provides information
about the size and layout of these vectors, but does not internally
allocate any associated storage space for field values.  Instead, the
user can create vector objects that use the DMDA layout
information with the routines
\code
  DMCreateGlobalVector(DM da,Vec *g);
  DMCreateLocalVector(DM da,Vec *l);
\endcode
These vectors will generally serve as the building blocks for local
and global PDE solutions, etc.  If additional vectors with such
layout information are needed in a code, they can be obtained by
duplicating `l` or `g` via
VecDuplicate() or VecDuplicateVecs().

We emphasize that a distributed array provides the information needed
to communicate the ghost value information between processes.  In most
cases, several different vectors can share the same communication
information (or, in other words, can share a given DMDA).  The
design of the DMDA object makes this easy, as each DMDA
operation may operate on vectors of the appropriate size, as obtained
via DMCreateLocalVector() and DMCreateGlobalVector() or as
produced by VecDuplicate().  As such, the DMDA
scatter/gather operations (e.g., DMGlobalToLocalBegin()) require
vector input/output arguments, as discussed below.

PETSc currently provides no container for multiple arrays sharing the
same distributed array communication; note, however, that the `dof`
parameter handles many cases of interest.

At certain stages of many applications, there is a need to work
on a local portion of the vector, including the ghost points.
This may be done by scattering a global vector into its
local parts by using the two-stage commands
\code
  DMGlobalToLocalBegin(DM da,Vec g,InsertMode iora,Vec l);
  DMGlobalToLocalEnd(DM da,Vec g,InsertMode iora,Vec l);
\endcode
which allow the overlap of communication and computation.
Since the global and local vectors, given by `g` and `l`, respectively,
must be compatible with the distributed array, `da`, they should be
generated by DMCreateGlobalVector()
and DMCreateLocalVector()
(or be duplicates of such a vector obtained via VecDuplicate()).
The InsertMode can be either ADD_VALUES or INSERT_VALUES.

One can scatter the local patches into the distributed vector
with the command
\code
  DMLocalToGlobalBegin(DM da,Vec l,InsertMode mode,Vec g);
  DMLocalToGlobalEnd(DM da,Vec l,InsertMode mode,Vec g);
\endcode
Note that this function is not
subdivided into beginning and ending phases, since it is purely local.

A third type of distributed array scatter is from a local
vector (including ghost points that contain irrelevant values) to
a local vector with correct ghost point values.
This scatter may be done by
commands
\code
  DMDALocalToLocalBegin(DM da,Vec l1,InsertMode iora,Vec l2);
  DMDALocalToLocalEnd(DM da,Vec l1,InsertMode iora,Vec l2);
\endcode
Since both local vectors, `l1` and `l2`,
must be compatible with the distributed array, `da`, they should be
generated by DMCreateLocalVector()
(or be duplicates of such vectors obtained via VecDuplicate()).
The InsertMode can be either `ADD_VALUES` or `INSERT_VALUES`.

It is possible to directly access the vector scatter contexts (see below)
used in the local-to-global (`ltog`), global-to-local
(`gtol`), and local-to-local (`ltol`)
scatters with the command
\code
  DMDAGetScatter(DM da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol);
\endcode
Most users should not need to use these contexts.

\subsection manual-user-subsec-vec-ghosted Local (Ghosted) Work Vectors
In most applications the local ghosted vectors are only needed during user
"function evaluations". %PETSc provides an easy light-weight (requiring
essentially no CPU time) way to obtain these work vectors and return them when
they are no longer needed. This is done with the routines
\code
  DMGetLocalVector(DM da,Vec *l);
   .... use the local vector l
  DMRestoreLocalVector(DM da,Vec *l);
\endcode

\subsection manual-user-subsec-vec-accessing-entries-dmda Accessing the Vector Entries for DMDA Vectors
PETSc provides an easy way to set values into the DMDA Vectors and access them using
the natural grid indexing. This is done with the routines
\code
  DMDAVecGetArray(DM da,Vec l,void *array);
   ... use the array indexing it with 1 or 2 or 3 dimensions 
   ... depending on the dimension of the DMDA
  DMDAVecRestoreArray(DM da,Vec l,void *array);
\endcode
where `array` is a multidimensional C array with the same dimension as
`da`. The vector `l` can be either a global vector or a local vector.
The `array` is accessed using the usual **global** indexing
on the entire grid, but the user may **only** refer to the local and ghost
entries of this array as all other entries are undefined. For example for a
scalar problem in two dimensions one could do
\code
   PetscScalar **f,**u;
   ...
  DMDAVecGetArray(DM da,Vec local,&u);
  DMDAVecGetArray(DM da,Vec global,&f);
   ...
      f[i][j] = u[i][j] - ...
   ...
  DMDAVecRestoreArray(DM da,Vec local,&u);
  DMDAVecRestoreArray(DM da,Vec global,&f);
\endcode
The recommended approach for multi-component PDEs is to declare a struct representing the fields defined at each node of the grid, e.g.
\code
  typedef struct { 
    PetscScalar u,v,omega,temperature; 
  } Node;
\endcode
and write residual evaluation using
\code
  Node **f,**u;
  DMDAVecGetArray(DM da,Vec local,&u);
  DMDAVecGetArray(DM da,Vec global,&f);
   ...
      f[i][j].omega = ...
   ...
  DMDAVecRestoreArray(DM da,Vec local,&u);
  DMDAVecRestoreArray(DM da,Vec global,&f);
\endcode
See `${PETSC_DIR}/src/snes/examples/tutorials/ex5.c` for a
complete example and see `${PETSC_DIR}/src/snes/examples/tutorials/ex19.c` for an
example for a multi-component PDE.

%---------------------------------------------------------------------------
\subsection{Grid Information}

The global indices of the lower left corner of the local portion of the array
as well as the local array size can be obtained with the commands
\code
  DMDAGetCorners(DM da,int *x,int *y,int *z,int *m,int *n,int *p);
  DMDAGetGhostCorners(DM da,int *x,int *y,int *z,int *m,int *n,int *p);
\endcode
The first version excludes any ghost points, while the second version
includes them.
The routine DMDAGetGhostCorners()
deals with the fact that subarrays along boundaries of the problem
domain have ghost points only on their interior edges, but not on
their boundary edges.

When either type of stencil is used, DMDA_STENCIL_STAR or
DA_STENCIL_BOX, the local vectors (with the ghost points)
represent rectangular arrays, including the extra corner elements in
the DMDA_STENCIL_STAR case. This configuration provides simple
access to the elements by employing two- (or three-) dimensional indexing.
The only difference between the
two cases is that when DMDA_STENCIL_STAR is used, the extra
corner components are **not** scattered between the processes and thus
contain undefined values that should **not** be used.

To assemble global stiffness matrices, one needs either
  - the global node number of each local node
including the ghost nodes. This number may be determined by using the
command
\code
  DMDAGetGlobalIndices(DM da,int *n,int *idx[]);
\endcode
The output argument `n` contains the number of
local nodes, including ghost nodes, while `idx` contains a list of length
`n` containing the global indices that correspond to the local nodes. Either
parameter may be omitted by passing NULL. Note that the Fortran
interface differs slightly; see \ref manual-user-sec-fortranarrays " here " for details.
  - or to set up the vectors and matrices so that their entries may be
added using the local numbering. This is done by first calling
\code
  DMDAGetISLocalToGlobalMapping(DM da,ISLocalToGlobalMapping *map);
\endcode
followed by
\code
  VecSetLocalToGlobalMapping(Vec v,ISLocalToGlobalMapping map);
  MatSetLocalToGlobalMapping(Mat A,ISLocalToGlobalMapping map);
\endcode
Now entries may be added to the vector and matrix using the local numbering
and VecSetValuesLocal() and MatSetValuesLocal().


Since the global ordering that %PETSc uses to manage its parallel vectors
(and matrices) does not usually correspond to the ``natural'' ordering
of a two- or three-dimensional array, the DMDA structure provides
an application ordering AO (see \ref manual-user-sec-ao " here") that maps
between the natural ordering on a rectangular grid and the ordering PETSc
uses to parallize. This ordering context can be obtained with the command
\code
  DMDAGetAO(DM da,AO *ao);
\endcode
In \ref manual-user-fig-daao " this figure " we indicate the orderings for a two-dimensional distributed
array, divided among four processes.

\anchor manual-user-fig-daao
\image html danumbering.png
<center><b>Natural Ordering and %PETSc Ordering for a 2D Distributed Array (Four Processes)</b></center>

The example
`${PETSC_DIR}/src/snes/examples/tutorials/ex5.c`,
illustrates the use of a distributed array in the solution of
a nonlinear problem.  The analogous Fortran program is
`${PETSC_DIR}/src/snes/examples/tutorials/ex5f.F`;
see the \ref manual-user-page-snes " SNES Chapter " for a discussion of the nonlinear
solvers.

\section manual-user-sec-unstruct Software for Managing Vectors Related to Unstructured Grids


\subsection manual-user-sec-indexset Index Sets

To facilitate general vector scatters and gathers used, for example, in updating
ghost points for problems defined on unstructured grids, %PETSc employs the
concept of an index set.  An index set, which is a generalization of a
set of integer indices, is used to define scatters, gathers, and similar
operations on vectors and matrices.

The following command creates a index set based on a list
of integers:
\code
  ISCreateGeneral(MPI_Comm comm,int n,int *indices,PetscCopyMode mode, IS *is);
\endcode
When mode is PETSC_COPY_VALUES this routine copies the `n` indices passed
to it by the integer array `indices`.
Thus, the user should be sure to free the integer array `indices`
when it is no longer needed, perhaps directly after the call to
ISCreateGeneral(). The communicator, `comm`, should consist of all
processes that will be using the IS.

Another standard index set is defined by a starting point (`first`) and a
stride (`step`), and can be created with the command
\code
  ISCreateStride(MPI_Comm comm,int n,int first,int step,IS *is);
\endcode

Index sets can be destroyed with the command
\code
  ISDestroy(IS &is);
\endcode

On rare occasions the user may need to access information directly
from an index set.
Several commands assist in this process:
\code
  ISGetSize(IS is,int *size);
  ISStrideGetInfo(IS is,int *first,int *stride);
  ISGetIndices(IS is,int **indices);
\endcode
The function ISGetIndices() returns a pointer to a list of the
indices in the index set.
For certain index sets, this may be a
temporary array of indices created specifically for a given routine.
Thus, once the user finishes using the array of indices,
the routine
\code
  ISRestoreIndices(IS is, int **indices);
\endcode
should be called to ensure that the system can free the space it
may have used to generate the list of indices.

A blocked version of the index sets can be created with the command
\code
  ISCreateBlock(MPI_Comm comm,int bs,int n,int *indices,PetscCopyMode mode, IS *is);
\endcode
This version is used for defining operations in which each element of the index
set refers to a block of `bs` vector entries.  Related routines analogous
to those described above exist as well, including
ISBlockGetIndices(), ISBlockGetSize(), ISBlockGetLocalSize(), ISGetBlockSize().
See the man pages for details.


\subsection manual-user-sec-scatter Scatters and Gathers

PETSc vectors have full support for general scatters and
gathers. One can select any subset of the components of a vector to
insert or add to any subset of the components of another vector.
We refer to these operations as generalized scatters, though they are
actually a combination of scatters and gathers.

To copy selected components from one vector
to another, one uses the following set of commands:
\code
  VecScatterCreate(Vec x,IS ix,Vec y,IS iy,VecScatter *ctx);
  VecScatterBegin(VecScatter ctx,Vec x,Vec y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(VecScatter ctx,Vec x,Vec y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterDestroy(VecScatter *ctx);
\endcode
Here `ix` denotes the index set of the first vector, while `iy` indicates the index set of the destination vector.  The vectors
can be parallel or sequential. The only requirements are that the
number of entries in the index set of the first vector, `ix`,
equal the number in the destination index set, `iy`, and that the
vectors be long enough to contain all the indices referred to in the
index sets.  The argument INSERT_VALUES specifies that the
vector elements will be inserted into the specified locations of the
destination vector, overwriting any existing values.  To add the
components, rather than insert them, the user should select the option
ADD_VALUES instead of INSERT_VALUES.

To perform a conventional gather operation, the user simply makes
 the destination index set, `iy`, be a stride index set with a stride of one.  Similarly, a
conventional scatter can be done with an initial (sending) index set
consisting of a stride.  The scatter routines are collective operations
(i.e. all processes that own
a parallel vector **must** call the scatter routines). When scattering from a
parallel vector to sequential vectors, each process has its own sequential
vector that receives values from locations as indicated in its own
index set. Similarly, in scattering
from sequential vectors to a parallel vector, each process has its
own sequential vector that makes contributions to the parallel vector.

**Caution**: When INSERT_VALUES is used, if two different
processes contribute different values to the same component in a
parallel vector, either value may end up being inserted. When
ADD_VALUES is used, the correct sum is added to the correct
location.

In some cases one may wish to "undo" a scatter, that is perform the
scatter backwards switching the roles of the sender and receiver. This is
done by using
\code
  VecScatterBegin(VecScatter ctx,Vec y,Vec x,INSERT_VALUES,SCATTER_REVERSE);
  VecScatterEnd(VecScatter ctx,Vec y,Vec x,INSERT_VALUES,SCATTER_REVERSE);
\endcode
Note that the roles of the first
two arguments to these routines must be swapped whenever the SCATTER_REVERSE
option is used.

Once a VecScatter object has been created it may be used with any vectors
that have the appropriate parallel data layout. That is, one can call
VecScatterBegin() and VecScatterEnd() with different vectors than
used in the call to VecScatterCreate() so long as they have the same
parallel layout (number of elements on each process are the same). Usually,
these "different" vectors would have been obtained via calls to
VecDuplicate() from the original vectors used in the call to
VecScatterCreate().

There is a %PETSc routine that is nearly the opposite of VecSetValues(), that is, VecGetValues(), but it can only get
local values from the vector.
To get off process values, the user should create a new vector where
the components are to be stored and perform the appropriate vector
scatter. For example, if one desires to obtain the values of the
100th and 200th entries of a parallel vector, `p`, one could use
a code such as that \ref manual-user-fig-vecscatter " in the snippet below".
In this example, the values of the 100th and 200th components are
placed in the array
values. In this example each process now has the 100th and
200th component, but obviously each process could gather any
elements it needed, or none by creating an index set with no entries.

\anchor manual-user-fig-vecscatter
\code
   Vec         p, x;         /* initial vector, destination vector */
   VecScatter  scatter;      /* scatter context */
   IS          from, to;     /* index sets that define the scatter */
   PetscScalar *values;
   int         idx_from[] = {100,200}, idx_to[] = {0,1};

   VecCreateSeq(PETSC_COMM_SELF,2,&x);
   ISCreateGeneral(PETSC_COMM_SELF,2,idx_from,PETSC_COPY_VALUES,&from);
   ISCreateGeneral(PETSC_COMM_SELF,2,idx_to,PETSC_COPY_VALUES,&to);
   VecScatterCreate(p,from,x,to,&scatter);
   VecScatterBegin(scatter,p,x,INSERT_VALUES,SCATTER_FORWARD);
   VecScatterEnd(scatter,p,x,INSERT_VALUES,SCATTER_FORWARD);
   VecGetArray(x,&values);
   ISDestroy(&from);
   ISDestroy(&to); 
   VecScatterDestroy(&scatter);
\endcode
<center><b>Example Code for Vector Scatters</b></center>

The scatter comprises two stages, in order to allow overlap of
communication and computation. The introduction of the
VecScatter context allows the communication patterns for the scatter
to be computed once and then reused repeatedly. Generally, even
setting up the communication for a scatter requires communication;
hence, it is best to reuse such information when possible.

\subsection manual-user-subsec-vec-scatter-ghost Scattering Ghost Values

The scatters provide a very general method for managing the communication of
required ghost values for unstructured grid computations. One scatters
the global vector into a local "ghosted" work vector, performs the computation
on the local work vectors, and then scatters back into the global solution
vector. In the simplest case this may be written as
\code
   Function: (Input Vec globalin, Output Vec globalout)

  VecScatterBegin(VecScatter scatter,Vec globalin,Vec localin,InsertMode INSERT_VALUES,
                  ScatterMode SCATTER_FORWARD);
  VecScatterEnd(VecScatter scatter,Vec globalin,Vec localin,InsertMode INSERT_VALUES,
                ScatterMode SCATTER_FORWARD);
  /* For example, do local calculations from localin to localout */
  VecScatterBegin(VecScatter scatter,Vec localout,Vec globalout,InsertMode ADD_VALUES,
                          ScatterMode SCATTER_REVERSE);
  VecScatterEnd(VecScatter scatter,Vec localout,Vec globalout,InsertMode ADD_VALUES,
                        ScatterMode SCATTER_REVERSE);
\endcode



\subsection manual-user-subsec-vectors-locations-for-ghost-values Vectors with Locations for Ghost Values


There are two minor drawbacks to the basic approach described above:
  - the extra memory requirement for the local work vector, `localin`, which
      duplicates the memory in `globalin`, and
  - the extra time required to copy the local values from `localin` to `globalin`.

An alternative approach is to allocate global vectors with space preallocated for
the ghost values; this may be done with either
\code
  VecCreateGhost(MPI_Comm comm,int n,int N,int nghost,int *ghosts,Vec *vv)
\endcode
or
\code
  VecCreateGhostWithArray(MPI_Comm comm,int n,int N,int nghost,int *ghosts, PetscScalar *array,Vec *vv)
\endcode
Here, `n` is the
number of local vector entries, `N` is the number of
global entries (or NULL), and `nghost` is the number of
ghost entries. The array `ghosts` is of size `nghost` and contains the
global vector location for each local ghost location. Using VecDuplicate()
or VecDuplicateVecs() on a ghosted vector will generate additional ghosted vectors.

In many ways a ghosted vector behaves just like any other `MPI` vector created
by VecCreateMPI(), the difference is that the ghosted vector has an additional
"local" representation that allows one to access the ghost locations. This is done
through the call to
\code
 VecGhostGetLocalForm(Vec g,Vec *l);
\endcode
The vector `l` is a
sequential representation of the parallel vector `g`
that shares the same array space (and hence numerical values); but allows one to
access the "ghost" values past "the end of the" array. Note that one access the
entries in `l` using the local numbering of elements and ghosts, while they
are accessed in `g` using the global numbering.

A common usage of a ghosted vector is given by
\code
  VecGhostUpdateBegin(Vec globalin,InsertMode INSERT_VALUES,ScatterMode SCATTER_FORWARD);
  VecGhostUpdateEnd(Vec globalin,InsertMode INSERT_VALUES,ScatterMode SCATTER_FORWARD);
  VecGhostGetLocalForm(Vec globalin,Vec *localin);
  VecGhostGetLocalForm(Vec globalout,Vec *localout);
   /*
      Do local calculations from localin to localout
   */
  VecGhostRestoreLocalForm(Vec globalin,Vec *localin);
  VecGhostRestoreLocalForm(Vec globalout,Vec *localout);
  VecGhostUpdateBegin(Vec globalout,InsertMode ADD_VALUES,ScatterMode SCATTER_REVERSE);
  VecGhostUpdateEnd(Vec globalout,InsertMode ADD_VALUES,ScatterMode SCATTER_REVERSE);
\endcode

The routines VecGhostUpdateBegin() and VecGhostUpdateEnd() are equivalent to the routines VecScatterBegin() and VecScatterEnd()
above except that since they are scattering into the ghost locations, they do not need
to copy the local vector values, which are already in place. In addition, the user does not
have to allocate the local work vector, since the ghosted vector already has allocated
slots to contain the ghost values.

The input arguments INSERT_VALUES and SCATTER_FORWARD
cause the ghost values to be correctly updated from the appropriate
process. The arguments ADD_VALUES and SCATTER_REVERSE
update the "local" portions of the vector from all the other
processes' ghost values.  This would be appropriate, for example,
when performing a finite element assembly of a load vector.

\ref manual-user-sec-partitioning "Read here about " the important topic of partitioning an unstructured grid.





*/
