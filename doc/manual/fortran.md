(ch_fortran)=

# PETSc for Fortran Users

Make sure the suffix of your Fortran files is .F90, not .f or .f90.

## Fortran and MPI

By default PETSc uses the MPI Fortran module `mpi`. To use `mpi_f08` run `./configure` with `--with-mpi-ftn-module=mpi_f08`.

We do not recommend it but it is possible to write Fortran code that works with both `use mpi` or `use mpi_f08`.
You must declare MPI objects using `MPIU_XXX` (for example `MPIU_Comm`, which the PETSc include files map to either `integer4` or `type(MPI_XXX)`.
In addition, you must handle `MPIU_Status` declarations and access to entries using the Fortran preprocessor. For example,

```fortran
#if defined(PETSC_USE_MPI_F08)
  MPIU_Status status
#else
  MPIU_Status status(MPI_STATUS_SIZE)
#endif
```

and then

```fortran
#if defined(PETSC_USE_MPI_F08)
      tag = status%MPI_TAG
#else
      tag = status(MPI_TAG)
#endif
```

## Numerical Constants

Since Fortran compilers do not automatically change the length of numerical constant arguments (integer and real) in subroutine calls to the expected length PETSc provides parameters that indicate the constants' kind.

```fortran
   interface
     subroutine SampleSubroutine(real, complex, integer, MPIinteger)
       PetscReal real         ! real(PETSC_REAL_KIND)
       PetscComplex complex   ! real(PETSC_REAL_KIND) or complex(PETSC_REAL_KIND)
       PetscInt integer       ! integer(PETSC_INT_KIND)
       PetscMPIInt MPIinteger ! integer(PETSC_MPIINT_KIND)
     end subroutine
   end interface
   ...
   ! Fortran () automatically sets the complex KIND to correspond to the KIND of constant arguments
   call SampleSubroutine(real = 1.0_PETSC_REAL_KIND, complex = (0.1_PETSC_REAL_KIND, 0.0_PETSC_REAL_KIND), integer = 1_PETSC_INT_KIND, MPIinteger = 1_PETSC_MPIINT_KIND)
   ! For variable arguments one must set the complex kind explicitly or it defaults to single precision
   PetscReal a = 0.1, b = 0.0
   call SampleSubroutine(real = 1.0_PETSC_REAL_KIND, complex = cmplx(a, b, PETSC_REAL_KIND), integer = 1_PETSC_INT_KIND, MPIinteger = 1_PETSC_MPIINT_KIND)
```

## Basic Fortran API Differences

(sec_fortran_includes)=

### Modules and Include Files

You must use both PETSc include files and modules.
At the beginning of every function and module definition you need something like

```fortran
#include "petsc/finclude/petscXXX.h"
   use petscXXX
```

The Fortran include files for PETSc are located in the directory
`$PETSC_DIR/$PETSC_ARCH/include/petsc/finclude` and the module files are located in `$PETSC_DIR/$PETSC_ARCH/include`

The include files are nested, that is, for example, `petsc/finclude/petscmat.h` automatically includes
`petsc/finclude/petscvec.h` and so on. The modules are also nested. One can use

```c
use petsc
```

to include all of them.

### Declaring PETSc Object Variables

You can declare PETSc object variables using either of the following:

```fortran
XXX variablename
```

```fortran
type(tXXX) variablename
```

For example,

```fortran
#include "petsc/finclude/petscvec.h"
  use petscvec

  Vec b
  type(tVec) x
```

PETSc types like `PetscInt` and `PetscReal` are simply aliases for basic Fortran types and cannot be written as `type(tPetscInt)`

PETSc objects are always automatically initialized when declared so you do not need to (and should not) do

```fortran
type(tXXX) x = PETSC_NULL_XXX
XXX x = PETSC_NULL_XXX
```

To make a variable no longer point to its previously assigned PETSc object use, for example,

```fortran
   Vec x, y
   PetscInt one = 1
   PetscCallA(VecCreateMPI(PETSC_COMM_WORLD, one, PETSC_DETERMINE, x, ierr))
   y = x
   PetscCallA(VecDestroy(x, ierr))
   PetscObjectNullify(y)
```

Otherwise `y` will be a dangling pointer whose access will cause a crash.


### Calling Sequences

The calling sequences for the Fortran version are in most cases
identical to the C version, except for the error checking variable
discussed in {any}`sec_fortran_errors`.

The key differences in handling arguments when calling PETSc functions from Fortran are

- One cannot pass a scalar variable to a function expecting an array, {any}`sec_passarray`.
- One must use type specific `PETSC_NULL` arguments, such as `PETSC_NULL_INTEGER`, {any}`sec_nullptr`.
- One must pass pointers to arrays for arguments that output an array, for example `PetscScalar, pointer \:\: a(\:)`,
  {any}`sec_fortranarrays`.
- `PETSC_DECIDE` and friends need to match the argument type, for example `PETSC_DECIDE_INTEGER`.

When passing floating point numbers into PETSc Fortran subroutines, always
make sure you have them marked as double precision (e.g., pass in `10.d0`
instead of `10.0` or declare them as PETSc variables, e.g.
`PetscScalar one = 1.0`). Otherwise, the compiler interprets the input as a single
precision number, which can cause crashes or other mysterious problems.
We **highly** recommend using the `implicit none`
option at the beginning of each Fortran subroutine and declaring all variables.

(sec_fortran_errors)=

### Error Checking

In the Fortran version, each PETSc routine has as its final argument an
integer error variable. The error code is
nonzero if an error has been detected; otherwise, it is zero. For
example, the Fortran and C variants of `KSPSolve()` are given,
respectively, below, where `ierr` denotes the `PetscErrorCode` error variable:

```fortran
call KSPSolve(ksp, b, x, ierr) ! Fortran
ierr = KSPSolve(ksp, b, x);    // C
```

For proper error handling one should not use the above syntax instead one should use

```fortran
PetscCall(KSPSolve(ksp, b, x, ierr))   ! Fortran subroutines
PetscCallA(KSPSolve(ksp, b, x, ierr))  ! Fortran main program
PetscCall(KSPSolve(ksp, b, x))         // C
```

(sec_passarray)=

### Passing Arrays To PETSc Functions

Many PETSc functions take arrays as arguments; in Fortran they must be passed as arrays even if the "array"
is of length one (unlike Fortran 77 where one can pass scalars to functions expecting arrays). When passing
a single value one can use the Fortran [] notation to pass the scalar as an array, for example

```fortran
PetscCall(VecSetValues(v, one, [i], [val], ierr))
```

This trick can only be used for arrays used to pass data into a PETSc routine, it cannot be used
for arrays used to receive data from a PETSc routine. For example,

```fortran
PetscCall(VecGetValues(v, one, idx, [val], ierr))
```

is invalid and will not set `val` with the correct value.

(sec_nullptr)=

### Passing null pointers to PETSc functions

Many PETSc C functions have the option of passing a `NULL`
argument (for example, the fifth argument of `MatCreateSeqAIJ()`).
From Fortran, users *must* pass `PETSC_NULL_XXX` to indicate a null
argument (where `XXX` is `INTEGER`, `DOUBLE`, `CHARACTER`,
`SCALAR`, `VEC`, `MAT`, etc depending on the argument type). For example, when no options prefix is desired
in the routine `PetscOptionsGetInt()`, one must use the following
command in Fortran:

```fortran
PetscCall(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, PETSC_NULL_CHARACTER, '-name', N, flg, ierr))
```

Where the code expects an array, then use `PETSC_NULL_XXX_ARRAY`. For example:

```fortran
PetscCall(MatCreateSeqDense(comm, m, n, PETSC_NULL_SCALAR_ARRAY, A))
```

When a PETSc function returns multiple arrays, such as `DMDAGetOwnershipRanges()` and the user does not need
certain arrays they must pass `PETSC_NULL_XXX_POINTER` as the argument. For example,

```fortran
PetscInt, pointer :: lx(:), ly(:)
PetscCallA(DMDAGetOwnershipRanges(dm, lx, ly, PETSC_NULL_INTEGER_POINTER, ierr))
PetscCallA(DMDARestoreOwnershipRanges(dm, lx, ly, PETSC_NULL_INTEGER_POINTER, ierr))
```

Arguments that are fully defined Fortran derived types (C structs), such as `MatFactorInfo` or `PetscSFNode`,
cannot be passed as null from Fortran. A properly defined variable must be passed in for those arguments.

Finally when a subroutine returns a `PetscObject` through an argument, to check if it is `NULL` you must use:

```fortran
if (PetscObjectIsNull(dm)) then
if (.not. PetscObjectIsNull(dm)) then
```

you cannot use

```fortran
if (dm .eq. PETSC_NULL_DM) then
```

Note that

```fortran
if (PetscObjectIsNull(PETSC_NULL_VEC)) then
```

will always return true, for any PETSc object.

These specializations with `NULL` types are required because of Fortran's strict type checking system and lack of a concept of `NULL`,
the Fortran compiler will often warn you if the wrong `NULL` type is passed.

(sec_fortranarrays)=

### Output Arrays from PETSc functions

For PETSc routine arguments that return an array of `PetscInt`, `PetscScalar`, `PetscReal` or of PETSc objects,
one passes in a pointer to an array and the PETSc routine returns an array containing the values. For example,

```c
PetscScalar *a;
Vec         v;
VecGetArray(v, &a);
```

is in Fortran,

```fortran
PetscScalar, pointer :: a(:)
Vec,         v
VecGetArray(v, a, ierr)
```

For PETSc routine arguments that return a character string (array), e.g. `const char *str[]` pass a string long enough to hold the
result. For example,

```fortran
character*(80)  str
PetscCall(KSPGetType(ksp,str,ierr))
```

The result is copied into `str`.

Similarly, for PETSc routines where the user provides a character array (to be filled) followed by the array's length, e.g. `char name[], size_t nlen`.
In Fortran pass a string long enough to hold the result, but not the separate length argument. For example,

```fortran
character*(80)  str
PetscCall(PetscGetHostName(name,ierr))
```

### Matrix, Vector and IS Indices

All matrices, vectors and `IS` in PETSc use zero-based indexing in the PETSc API
regardless of whether C or Fortran is being used. For example,
`MatSetValues()` and `VecSetValues()` always use
zero indexing. See {any}`sec_matoptions` for further
details.

Indexing into Fortran arrays, for example obtained with `VecGetArray()`, uses the Fortran
convention and generally begin with 1 except for special routines such as `DMDAVecGetArray()` which uses the ranges
provided by `DMDAGetCorners()`.

### Setting Routines and Contexts

Some PETSc functions take as arguments user-functions and contexts for the function. For example

```fortran
external func
SNESSetFunction(snes, r, func, ctx, ierr)
SNES snes
Vec r
PetscErrorCode ierr
```

where `func` has the calling sequence

```fortran
subroutine func(snes, x, f, ctx, ierr)
SNES snes
Vec x,f
PetscErrorCode ierr
```

and `ctx` can be almost anything (represented as `void *` in C).

It can be a Fortran derived type as in

```fortran
subroutine func(snes, x, f, ctx, ierr)
SNES snes
Vec x,f
type (userctx)   ctx
PetscErrorCode ierr
...

external func
SNESSetFunction(snes, r, func, ctx, ierr)
SNES snes
Vec r
PetscErrorCode ierr
type (userctx)   ctx
```

or a PETSc object

```fortran
subroutine func(snes, x, f, ctx, ierr)
SNES snes
Vec x,f
Vec ctx
PetscErrorCode ierr
...

external func
SNESSetFunction(snes, r, func, ctx, ierr)
SNES snes
Vec r
PetscErrorCode ierr
Vec ctx
```

or nothing

```fortran
subroutine func(snes, x, f, dummy, ierr)
SNES snes
Vec x,f
integer dummy(*)
PetscErrorCode ierr
...

external func
SNESSetFunction(snes, r, func, 0, ierr)
SNES snes
Vec r
PetscErrorCode ierr
```

When a function pointer (declared as external in Fortran) is passed as an argument to a PETSc function,
it is assumed that this
function references a routine written in the same language as the PETSc
interface function that was called. For instance, if
`SNESSetFunction()` is called from C, the function must be a C function. Likewise, if it is called from Fortran, the
function must be (a subroutine) written in Fortran.

If you are using Fortran classes that have bound functions (methods) as in
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tests/ex18f90.F90.html">src/snes/tests/ex18f90.F90</a>, the context cannot be passed
to function pointer setting routines, such as `SNESSetFunction()`. Instead, one must use `SNESSetFunctionNoInterface()`,
and define the interface directly in the user code, see
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tests/ex18f90.F90.html">ex18f90.F90</a>
for a full demonstration.

(sec_fortcompile)=

### Compiling and Linking Fortran Programs

See {any}`sec_writing_application_codes`.


(sec_fortran_examples)=

## Sample Fortran Programs

Sample programs that illustrate the PETSc interface for Fortran are
given below, corresponding to
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/vec/vec/tests/ex19f.F90.html">Vec Test ex19f</a>,
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/vec/vec/tutorials/ex4f.F90.html">Vec Tutorial ex4f</a>,
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/sys/classes/draw/tests/ex5f.F90.html">Draw Test ex5f</a>,
and
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex1f.F90.html">SNES Tutorial ex1f</a>,
respectively. We also refer Fortran programmers to the C examples listed
throughout the manual, since PETSc usage within the two languages
differs only slightly.

:::{admonition} Listing: `src/vec/vec/tests/ex19f.F90`
:name: vec-test-ex19f

```{literalinclude} /../src/vec/vec/tests/ex19f.F90
:end-at: end
:language: fortran
```
:::

(listing_vec_ex4f)=

:::{admonition} Listing: `src/vec/vec/tutorials/ex4f.F90`
:name: vec-ex4f

```{literalinclude} /../src/vec/vec/tutorials/ex4f.F90
:end-before: '!/*TEST'
:language: fortran
```
:::

:::{admonition} Listing: `src/sys/classes/draw/tests/ex5f.F90`
:name: draw-test-ex5f

```{literalinclude} /../src/sys/classes/draw/tests/ex5f.F90
:end-at: end
:language: fortran
```
:::

:::{admonition} Listing: `src/snes/tutorials/ex1f.F90`
:name: snes-ex1f

```{literalinclude} /../src/snes/tutorials/ex1f.F90
:end-before: '!/*TEST'
:language: fortran
```
:::

### Calling Fortran Routines from C (and C Routines from Fortran)

The information here applies only if you plan to call your **own**
C functions from Fortran or Fortran functions from C.
Different compilers have different methods of naming Fortran routines
called from C (or C routines called from Fortran). Most Fortran
compilers change the capital letters in Fortran routines to
all lowercase. With some compilers, the Fortran compiler appends an underscore
to the end of each Fortran routine name; for example, the Fortran
routine `Dabsc()` would be called from C with `dabsc_()`. Other
compilers change all the letters in Fortran routine names to capitals.

PETSc provides two macros (defined in C/C++) to help write portable code
that mixes C/C++ and Fortran. They are `PETSC_HAVE_FORTRAN_UNDERSCORE`
and `PETSC_HAVE_FORTRAN_CAPS` , which will be defined in the file
`$PETSC_DIR/$PETSC_ARCH/include/petscconf.h` based on the compilers
conventions. The macros are used,
for example, as follows:

```fortran
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dabsc_ DABSC
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dabsc_ dabsc
#endif
.....
dabsc_( &n,x,y); /* call the Fortran function */
```
