/*
   This is the main PETSc include file (for C and C++).  It is included by all
   other PETSc include files, so it almost never has to be specifically included.
   Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#pragma once

/*MC
   PeOP - indicates an argument to a PETSc function is optional and one can pass `NULL` instead. This is used by the Fortran API generator

   Level: developer

   Example:
.vb
   PetscErrorCode XXXX(Vec v, PeOp PetscObject obj, PeOp PetscInt *idx, PeOp PetscInt *array[])
.ve

   Notes:
   This is not part of the PETSc public API and should only be used in PETSc source code.

   Put this in the function declaration in front of each variable that is optional

   Developer Note:
   Shortened form of PETSc optional

.seealso: `PeNS`, `PeNSS`, `PeCtx`, `PetscInitialize()`
M*/
#define PeOp

/*MC
   PeNS - indicates a function that does not use the PETSc standard arguments which make it easy to generate automatic language stubs for other languages

   Level: developer

   Notes:
   This is not part of the PETSc public API and should only be used in PETSc source code.

   Put this at the end of the function declaration closing parenthesis

   Developer Note:
   Shortened form of PETSc non-standard

.seealso: `PeOp`, `PeNSS`, `PeCtx`, `PetscInitialize()`
M*/
#define PeNS

/*MC
   PeNSS - indicates a function that needs a special treatment in the C-side stub when generating the binding for other languages

   Level: developer

   Notes:
   This is not part of the PETSc public API and should only be used in PETSc source code.

   Put this at the end of the function declaration closing parenthesis

   It is similar to PeNS; in Fortran it will generate the Fortran interface definition automatically but not the C stub, which should be added manually under the appropriate `ftn-custom` directory

   Developer Note:
   Shortened form of PETSc non-standard stub

.seealso: `PeOp`, `PeNS`, `PeCtx`, `PetscInitialize()`
M*/
#define PeNSS

/* ========================================================================== */
/*
   petscconf.h is contained in ${PETSC_ARCH}/include/petscconf.h it is
   found automatically by the compiler due to the -I${PETSC_DIR}/${PETSC_ARCH}/include that
   PETSc's makefiles add to the compiler rules.
   For --prefix installs the directory ${PETSC_ARCH} does not exist and petscconf.h is in the same
   directory as the other PETSc include files.
*/
#include <petscconf.h>
#include <petscpkg_version.h>
#include <petscconf_poison.h>
#include <petscfix.h>
#include <petscmacros.h>

/* SUBMANSEC = Sys */

#if defined(PETSC_DESIRE_FEATURE_TEST_MACROS)
  /*
   Feature test macros must be included before headers defined by IEEE Std 1003.1-2001
   We only turn these in PETSc source files that require them by setting PETSC_DESIRE_FEATURE_TEST_MACROS
*/
  #if defined(PETSC__POSIX_C_SOURCE_200112L) && !defined(_POSIX_C_SOURCE)
    #define _POSIX_C_SOURCE 200112L
  #endif
  #if defined(PETSC__BSD_SOURCE) && !defined(_BSD_SOURCE)
    #define _BSD_SOURCE
  #endif
  #if defined(PETSC__DEFAULT_SOURCE) && !defined(_DEFAULT_SOURCE)
    #define _DEFAULT_SOURCE
  #endif
  #if defined(PETSC__GNU_SOURCE) && !defined(_GNU_SOURCE)
    #define _GNU_SOURCE
  #endif
#endif

#include <petscsystypes.h>

/* ========================================================================== */

/*
    Defines the interface to MPI allowing the use of all MPI functions.

    PETSc does not use the C++ binding of MPI at ALL. The following flag
    makes sure the C++ bindings are not included. The C++ bindings REQUIRE
    putting mpi.h before ANY C++ include files, we cannot control this
    with all PETSc users. Users who want to use the MPI C++ bindings can include
    mpicxx.h directly in their code
*/
#if !defined(MPICH_SKIP_MPICXX)
  #define MPICH_SKIP_MPICXX 1
#endif
#if !defined(OMPI_SKIP_MPICXX)
  #define OMPI_SKIP_MPICXX 1
#endif
#if defined(PETSC_HAVE_MPIUNI)
  #include <petsc/mpiuni/mpi.h>
#else
  #include <mpi.h>
#endif

/*
   Perform various sanity checks that the correct mpi.h is being included at compile time.
   This usually happens because
      * either an unexpected mpi.h is in the default compiler path (i.e. in /usr/include) or
      * an extra include path -I/something (which contains the unexpected mpi.h) is being passed to the compiler
   Note: with MPICH and OpenMPI, accept versions [x.y.z, x+1.0.0) as compatible
*/
#if defined(PETSC_HAVE_MPIUNI)
  #ifndef MPIUNI_H
    #error "PETSc was configured with --with-mpi=0 but now appears to be compiling using a different mpi.h"
  #endif
#elif defined(PETSC_HAVE_I_MPI)
  #if !defined(I_MPI_NUMVERSION)
    #error "PETSc was configured with I_MPI but now appears to be compiling using a non-I_MPI mpi.h"
  #elif I_MPI_NUMVERSION != PETSC_PKG_I_MPI_NUMVERSION
    #error "PETSc was configured with one I_MPI mpi.h version but now appears to be compiling using a different I_MPI mpi.h version"
  #endif
#elif defined(PETSC_HAVE_MVAPICH2)
  #if !defined(MVAPICH2_NUMVERSION)
    #error "PETSc was configured with MVAPICH2 but now appears to be compiling using a non-MVAPICH2 mpi.h"
  #elif MVAPICH2_NUMVERSION != PETSC_PKG_MVAPICH2_NUMVERSION
    #error "PETSc was configured with one MVAPICH2 mpi.h version but now appears to be compiling using a different MVAPICH2 mpi.h version"
  #endif
#elif defined(PETSC_HAVE_MPICH)
  #if !defined(MPICH_NUMVERSION) || defined(MVAPICH2_NUMVERSION) || defined(I_MPI_NUMVERSION)
    #error "PETSc was configured with MPICH but now appears to be compiling using a non-MPICH mpi.h"
  #elif PETSC_PKG_MPICH_VERSION_GT(MPICH_NUMVERSION / 10000000, MPICH_NUMVERSION / 100000 % 100, MPICH_NUMVERSION / 1000 % 100)
    #error "PETSc was configured with one MPICH mpi.h version but now appears to be compiling using an older MPICH mpi.h version"
  #elif PETSC_PKG_MPICH_VERSION_LT(MPICH_NUMVERSION / 10000000, 0, 0)
    #error "PETSc was configured with one MPICH mpi.h version but now appears to be compiling using a newer major MPICH mpi.h version"
  #endif
#elif defined(PETSC_HAVE_OPENMPI)
  #if !defined(OMPI_MAJOR_VERSION)
    #error "PETSc was configured with Open MPI but now appears to be compiling using a non-Open MPI mpi.h"
  #elif PETSC_PKG_OPENMPI_VERSION_GT(OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION)
    #error "PETSc was configured with one Open MPI mpi.h version but now appears to be compiling using an older Open MPI mpi.h version"
  #elif PETSC_PKG_OPENMPI_VERSION_LT(OMPI_MAJOR_VERSION, 0, 0)
    #error "PETSc was configured with one Open MPI mpi.h version but now appears to be compiling using a newer major Open MPI mpi.h version"
  #endif
#elif defined(PETSC_HAVE_MSMPI_VERSION)
  #if !defined(MSMPI_VER)
    #error "PETSc was configured with MSMPI but now appears to be compiling using a non-MSMPI mpi.h"
  #elif (MSMPI_VER != PETSC_HAVE_MSMPI_VERSION)
    #error "PETSc was configured with one MSMPI mpi.h version but now appears to be compiling using a different MSMPI mpi.h version"
  #endif
#elif defined(OMPI_MAJOR_VERSION) || defined(MPICH_NUMVERSION) || defined(MSMPI_VER)
  #error "PETSc was configured with undetermined MPI - but now appears to be compiling using any of Open MPI, MS-MPI or a MPICH variant"
#endif

/*
    Need to put stdio.h AFTER mpi.h for MPICH2 with C++ compiler
    see the top of mpicxx.h in the MPICH2 distribution.
*/
#include <stdio.h>

/* MSMPI on 32-bit Microsoft Windows requires this yukky hack - that breaks MPI standard compliance */
#if !defined(MPIAPI)
  #define MPIAPI
#endif

PETSC_EXTERN MPI_Datatype MPIU_ENUM PETSC_ATTRIBUTE_MPI_TYPE_TAG(PetscEnum);
#define MPIU_BOOL MPI_C_BOOL PETSC_DEPRECATED_MACRO(3, 24, 0, "MPI_C_BOOL", )

/*MC
   MPIU_INT - Portable MPI datatype corresponding to `PetscInt` independent of the precision of `PetscInt`

   Level: beginner

   Note:
   In MPI calls that require an MPI datatype that matches a `PetscInt` or array of `PetscInt` values, pass this value.

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_COUNT`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`
M*/

PETSC_EXTERN MPI_Datatype MPIU_FORTRANADDR;

#if defined(PETSC_USE_64BIT_INDICES)
  #define MPIU_INT MPIU_INT64
#else
  #define MPIU_INT MPI_INT
#endif

/*MC
   MPIU_COUNT - Portable MPI datatype corresponding to `PetscCount` independent of the precision of `PetscCount`

   Level: beginner

   Note:
   In MPI calls that require an MPI datatype that matches a `PetscCount` or array of `PetscCount` values, pass this value.

  Developer Note:
  It seems `MPI_AINT` is unsigned so this may be the wrong choice here since `PetscCount` is signed

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_INT`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`
M*/
#define MPIU_COUNT MPI_AINT

/*
    For the rare cases when one needs to send a size_t object with MPI
*/
PETSC_EXTERN MPI_Datatype MPIU_SIZE_T PETSC_ATTRIBUTE_MPI_TYPE_TAG(size_t);

/*
      You can use PETSC_STDOUT as a replacement of stdout. You can also change
    the value of PETSC_STDOUT to redirect all standard output elsewhere
*/
PETSC_EXTERN FILE *PETSC_STDOUT;

/*
      You can use PETSC_STDERR as a replacement of stderr. You can also change
    the value of PETSC_STDERR to redirect all standard error elsewhere
*/
PETSC_EXTERN FILE *PETSC_STDERR;

/*
  Handle inclusion when using clang compiler with CUDA support
  __float128 is not available for the device
*/
#if defined(__clang__) && (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  #define PETSC_SKIP_REAL___FLOAT128
#endif

/*
    Declare extern C stuff after including external header files
*/

PETSC_EXTERN PetscBool PETSC_RUNNING_ON_VALGRIND;
/*
    Defines elementary mathematics functions and constants.
*/
#include <petscmath.h>

/*MC
   PETSC_IGNORE - same as `NULL`, means PETSc will ignore this argument

   Level: beginner

   Note:
   Accepted by many PETSc functions to not set a parameter and instead use a default value

   Fortran Note:
   Use `PETSC_NULL_INTEGER`, `PETSC_NULL_SCALAR` etc

.seealso: `PETSC_DECIDE`, `PETSC_DEFAULT`, `PETSC_DETERMINE`
M*/
#define PETSC_IGNORE PETSC_NULLPTR
#define PETSC_NULL   PETSC_DEPRECATED_MACRO(3, 19, 0, "PETSC_NULLPTR", ) PETSC_NULLPTR

/*MC
   PETSC_UNLIMITED - standard way of passing an integer or floating point parameter to indicate PETSc there is no bound on the value allowed

   Level: beginner

   Example Usage:
.vb
   KSPSetTolerances(ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_UNLIMITED, PETSC_UNLIMITED);
.ve
  indicates that the solver is allowed to take any number of iterations and will not stop early no matter how the residual gets.

   Fortran Note:
   Use `PETSC_UNLIMITED_INTEGER` or `PETSC_UNLIMITED_REAL`.

.seealso: `PETSC_DEFAULT`, `PETSC_IGNORE`, `PETSC_DETERMINE`, `PETSC_DECIDE`
M*/

/*MC
   PETSC_DECIDE - standard way of passing an integer or floating point parameter to indicate PETSc should determine an appropriate value

   Level: beginner

   Example Usage:
.vb
   VecSetSizes(ksp, PETSC_DECIDE, 10);
.ve
  indicates that the global size of the vector is 10 and the local size will be automatically determined so that the sum of the
  local sizes is the global size, see `PetscSplitOwnership()`.

   Fortran Note:
   Use `PETSC_DECIDE_INTEGER` or `PETSC_DECIDE_REAL`.

.seealso: `PETSC_DEFAULT`, `PETSC_IGNORE`, `PETSC_DETERMINE`, `PETSC_UNLIMITED`
M*/

/*MC
   PETSC_DETERMINE - standard way of passing an integer or floating point parameter to indicate PETSc should determine an appropriate value

   Level: beginner

    Example Usage:
.vb
   VecSetSizes(ksp, 10, PETSC_DETERMINE);
.ve
  indicates that the local size of the vector is 10 and the global size will be automatically summing up all the local sizes.

   Note:
   Same as `PETSC_DECIDE`

   Fortran Note:
   Use `PETSC_DETERMINE_INTEGER` or `PETSC_DETERMINE_REAL`.

   Developer Note:
   I would like to use const `PetscInt` `PETSC_DETERMINE` = `PETSC_DECIDE`; but for
   some reason this is not allowed by the standard even though `PETSC_DECIDE` is a constant value.

.seealso: `PETSC_DECIDE`, `PETSC_DEFAULT`, `PETSC_IGNORE`, `VecSetSizes()`, `PETSC_UNLIMITED`
M*/

/*MC
   PETSC_CURRENT - standard way of indicating to an object not to change the current value of the parameter in the object

   Level: beginner

   Note:
   Use `PETSC_DECIDE` to use the value that was set by PETSc when the object's type was set

   Fortran Note:
   Use `PETSC_CURRENT_INTEGER` or `PETSC_CURRENT_REAL`.

.seealso: `PETSC_DECIDE`, `PETSC_IGNORE`, `PETSC_DETERMINE`, `PETSC_DEFAULT`, `PETSC_UNLIMITED`
M*/

/*MC
   PETSC_DEFAULT - deprecated, see `PETSC_CURRENT` and `PETSC_DETERMINE`

   Level: beginner

   Note:
   The name is confusing since it tells the object to continue to use the value it is using, not the default value when the object's type was set.

   Developer Note:
   Unfortunately this was used for two different purposes in the past, to actually trigger the use of a default value or to continue the
   use of currently set value (in, for example, `KSPSetTolerances()`.

.seealso: `PETSC_DECIDE`, `PETSC_IGNORE`, `PETSC_DETERMINE`, `PETSC_CURRENT`, `PETSC_UNLIMITED`
M*/

/* These MUST be preprocessor defines! see https://gitlab.com/petsc/petsc/-/issues/1370 */
#define PETSC_DECIDE    (-1)
#define PETSC_DETERMINE PETSC_DECIDE
#define PETSC_CURRENT   (-2)
#define PETSC_UNLIMITED (-3)
/*  PETSC_DEFAULT is deprecated in favor of PETSC_CURRENT for use in KSPSetTolerances() and similar functions */
#define PETSC_DEFAULT PETSC_CURRENT

/*MC
   PETSC_COMM_WORLD - the equivalent of the `MPI_COMM_WORLD` communicator which represents all the processes that PETSc knows about.

   Level: beginner

   Notes:
   By default `PETSC_COMM_WORLD` and `MPI_COMM_WORLD` are identical unless you wish to
   run PETSc on ONLY a subset of `MPI_COMM_WORLD`. In that case create your new (smaller)
   communicator, call it, say comm, and set `PETSC_COMM_WORLD` = comm BEFORE calling
   `PetscInitialize()`, but after `MPI_Init()` has been called.

   The value of `PETSC_COMM_WORLD` should never be used or accessed before `PetscInitialize()`
   is called because it may not have a valid value yet.

.seealso: `PETSC_COMM_SELF`
M*/
PETSC_EXTERN MPI_Comm PETSC_COMM_WORLD;

/*MC
   PETSC_COMM_SELF - This is always `MPI_COMM_SELF`

   Level: beginner

   Note:
   Do not USE/access or set this variable before `PetscInitialize()` has been called.

.seealso: `PETSC_COMM_WORLD`
M*/
#define PETSC_COMM_SELF MPI_COMM_SELF

/*MC
   PETSC_MPI_THREAD_REQUIRED - the required threading support used if PETSc initializes MPI with `MPI_Init_thread()`.

   No Fortran Support

   Level: beginner

   Note:
   By default `PETSC_MPI_THREAD_REQUIRED` equals `MPI_THREAD_FUNNELED` when the MPI implementation provides `MPI_Init_thread()`, otherwise it equals `MPI_THREAD_SINGLE`

.seealso: `PetscInitialize()`
M*/
PETSC_EXTERN PetscMPIInt PETSC_MPI_THREAD_REQUIRED;

/*MC
   PetscBeganMPI - indicates if PETSc initialized MPI using `MPI_Init()` during `PetscInitialize()` or if MPI was already initialized with `MPI_Init()`

   Synopsis:
   #include <petscsys.h>
   PetscBool PetscBeganMPI;

   No Fortran Support

   Level: developer

   Note:
   `MPI_Init()` can never be called after `PetscInitialize()`

.seealso: `PetscInitialize()`, `PetscInitializeCalled`
M*/
PETSC_EXTERN PetscBool PetscBeganMPI;

PETSC_EXTERN PetscBool PetscErrorHandlingInitialized;
PETSC_EXTERN PetscBool PetscInitializeCalled;
PETSC_EXTERN PetscBool PetscFinalizeCalled;
PETSC_EXTERN PetscBool PetscViennaCLSynchronize;

PETSC_EXTERN PetscErrorCode PetscSetHelpVersionFunctions(PetscErrorCode (*)(MPI_Comm), PetscErrorCode (*)(MPI_Comm));
PETSC_EXTERN PetscErrorCode PetscCommDuplicate(MPI_Comm, MPI_Comm *, int *);
PETSC_EXTERN PetscErrorCode PetscCommDestroy(MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscCommGetComm(MPI_Comm, MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscCommRestoreComm(MPI_Comm, MPI_Comm *);

#if defined(PETSC_HAVE_KOKKOS)
PETSC_EXTERN PetscErrorCode PetscKokkosInitializeCheck(void); /* Initialize Kokkos if not yet. */
#endif

#if defined(PETSC_HAVE_NVSHMEM)
PETSC_EXTERN PetscBool      PetscBeganNvshmem;
PETSC_EXTERN PetscBool      PetscNvshmemInitialized;
PETSC_EXTERN PetscErrorCode PetscNvshmemFinalize(void);
#endif

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_EXTERN PetscErrorCode PetscElementalInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscElementalInitialized(PetscBool *);
PETSC_EXTERN PetscErrorCode PetscElementalFinalizePackage(void);
#endif

/*MC
   PetscMalloc - Allocates memory for use with PETSc. One should use `PetscNew()`, `PetscMalloc1()` or `PetscCalloc1()` usually instead of `PetscMalloc()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc(size_t m,void **result)

   Not Collective

   Input Parameter:
.  m - number of bytes to allocate

   Output Parameter:
.  result - memory allocated

   Level: beginner

   Notes:
   Memory is always allocated at least double aligned

   It is safe to allocate with an m of 0 and pass the resulting pointer to `PetscFree()`.
   However, the pointer should never be dereferenced or the program will crash.

   Developer Note:
   All the `PetscMallocN()` routines actually call `PetscMalloc()` behind the scenes.

   Except for data structures that store information about the PETSc options database all memory allocated by PETSc is
   obtained with `PetscMalloc()` or `PetscCalloc()`

.seealso: `PetscFree()`, `PetscNew()`, `PetscCalloc()`
M*/
#define PetscMalloc(a, b) ((*PetscTrMalloc)((a), PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void **)(b)))

/*MC
   PetscRealloc - Reallocates memory

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscRealloc(size_t m,void **result)

   Not Collective

   Input Parameters:
+  m      - number of bytes to allocate
-  result - previous memory

   Output Parameter:
.  result - new memory allocated

   Level: developer

   Notes:
   `results` must have already been obtained with `PetscMalloc()`

   Memory is always allocated at least double aligned

.seealso: `PetscMalloc()`, `PetscFree()`, `PetscNew()`
M*/
#define PetscRealloc(a, b) ((*PetscTrRealloc)((a), __LINE__, PETSC_FUNCTION_NAME, __FILE__, (void **)(b)))

/*MC
   PetscAddrAlign - Rounds up an address to `PETSC_MEMALIGN` alignment

   Synopsis:
    #include <petscsys.h>
   void *PetscAddrAlign(void *addr)

   Not Collective

   Input Parameter:
.  addr - address to align (any pointer type)

   Level: developer

.seealso: `PetscMallocAlign()`
M*/
#define PetscAddrAlign(a) ((void *)((((PETSC_UINTPTR_T)(a)) + (PETSC_MEMALIGN - 1)) & ~(PETSC_MEMALIGN - 1)))

/*MC
   PetscCalloc - Allocates a cleared (zeroed) memory region aligned to `PETSC_MEMALIGN`, similar to `PetscMalloc()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc(size_t m,void **result)

   Not Collective

   Input Parameter:
.  m - number of bytes to allocate

   Output Parameter:
.  result - memory allocated

   Level: beginner

   Notes:
   Memory is always allocated at least double aligned. This macro is useful in allocating memory pointed by void pointers

   It is safe to allocate with an m of 0 and pass the resulting pointer to `PetscFree()`.

   However, the pointer should never be dereferenced or the program will crash.

   Developer Note:
   All `PetscCallocN()` routines call `PetscCalloc()` behind the scenes.

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`
M*/
#define PetscCalloc(m, result) PetscMallocA(1, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)m), (result))

/*MC
   PetscMalloc1 - Allocates an array of memory aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc1(size_t m1,type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate  (may be zero)

   Output Parameter:
.  r1 - memory allocated

   Level: beginner

   Note:
   This uses `sizeof()` of the memory type requested to determine the total memory to be allocated; therefore, you should not
   multiply the number of elements requested by the `sizeof()` the type. For example, use
.vb
  PetscInt *id;
  PetscMalloc1(10,&id);
.ve
       not
.vb
  PetscInt *id;
  PetscMalloc1(10*sizeof(PetscInt),&id);
.ve

  Does not zero the memory allocated, use `PetscCalloc1()` to obtain memory that has been zeroed.

  The `PetscMalloc[N]()` and `PetscCalloc[N]()` take an argument of type `size_t`! However, most codes use `value`, computed via `int` or `PetscInt` variables. This can overflow in
  32bit `int` computation - while computation in 64bit `size_t` would not overflow!
  It's best if any arithmetic that is done for size computations is done with `size_t` type - avoiding arithmetic overflow!

  `PetscMalloc[N]()` and `PetscCalloc[N]()` attempt to work-around this by casting the first variable to `size_t`.
  This works for most expressions, but not all, such as
.vb
  PetscInt *id, a, b;
  PetscMalloc1(use_a_squared ? a * a * b : a * b, &id); // use_a_squared is cast to size_t, but a and b are still PetscInt
  PetscMalloc1(a + b * b, &id); // a is cast to size_t, but b * b is performed at PetscInt precision first due to order-of-operations
.ve

  These expressions should either be avoided, or appropriately cast variables to `size_t`:
.vb
  PetscInt *id, a, b;
  PetscMalloc1(use_a_squared ? (size_t)a * a * b : (size_t)a * b, &id); // Cast a to size_t before multiplication
  PetscMalloc1(b * b + a, &id); // b is automatically cast to size_t and order-of-operations ensures size_t precision is maintained
.ve

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscCalloc1()`, `PetscMalloc2()`
M*/
#define PetscMalloc1(m1, r1) PetscMallocA(1, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1))

/*MC
   PetscCalloc1 - Allocates a cleared (zeroed) array of memory aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc1(size_t m1,type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate in 1st chunk  (may be zero)

   Output Parameter:
.  r1 - memory allocated

   Level: beginner

   Note:
   See `PetscMalloc1()` for more details on usage.

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc2()`
M*/
#define PetscCalloc1(m1, r1) PetscMallocA(1, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1))

/*MC
   PetscMalloc2 - Allocates 2 arrays of memory both aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc2(size_t m1,type **r1,size_t m2,type **r2)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
-  m2 - number of elements to allocate in 2nd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
-  r2 - memory allocated in second chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc2()`
M*/
#define PetscMalloc2(m1, r1, m2, r2) PetscMallocA(2, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2))

/*MC
   PetscCalloc2 - Allocates 2 cleared (zeroed) arrays of memory both aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc2(size_t m1,type **r1,size_t m2,type **r2)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
-  m2 - number of elements to allocate in 2nd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
-  r2 - memory allocated in second chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscCalloc1()`, `PetscMalloc2()`
M*/
#define PetscCalloc2(m1, r1, m2, r2) PetscMallocA(2, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2))

/*MC
   PetscMalloc3 - Allocates 3 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc3(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
-  m3 - number of elements to allocate in 3rd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
-  r3 - memory allocated in third chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc3()`, `PetscFree3()`
M*/
#define PetscMalloc3(m1, r1, m2, r2, m3, r3) \
  PetscMallocA(3, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3))

/*MC
   PetscCalloc3 - Allocates 3 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc3(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
-  m3 - number of elements to allocate in 3rd chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
-  r3 - memory allocated in third chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscCalloc2()`, `PetscMalloc3()`, `PetscFree3()`
M*/
#define PetscCalloc3(m1, r1, m2, r2, m3, r3) \
  PetscMallocA(3, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3))

/*MC
   PetscMalloc4 - Allocates 4 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc4(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
-  m4 - number of elements to allocate in 4th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
-  r4 - memory allocated in fourth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc4()`, `PetscFree4()`
M*/
#define PetscMalloc4(m1, r1, m2, r2, m3, r3, m4, r4) \
  PetscMallocA(4, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4))

/*MC
   PetscCalloc4 - Allocates 4 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc4(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
-  m4 - number of elements to allocate in 4th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
-  r4 - memory allocated in fourth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc4()`, `PetscFree4()`
M*/
#define PetscCalloc4(m1, r1, m2, r2, m3, r3, m4, r4) \
  PetscMallocA(4, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4))

/*MC
   PetscMalloc5 - Allocates 5 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc5(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
-  m5 - number of elements to allocate in 5th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
-  r5 - memory allocated in fifth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc5()`, `PetscFree5()`
M*/
#define PetscMalloc5(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5) \
  PetscMallocA(5, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5))

/*MC
   PetscCalloc5 - Allocates 5 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc5(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
-  m5 - number of elements to allocate in 5th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
-  r5 - memory allocated in fifth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc5()`, `PetscFree5()`
M*/
#define PetscCalloc5(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5) \
  PetscMallocA(5, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5))

/*MC
   PetscMalloc6 - Allocates 6 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc6(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
-  m6 - number of elements to allocate in 6th chunk  (may be zero)

   Output Parameteasr:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
-  r6 - memory allocated in sixth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc6()`, `PetscFree3()`, `PetscFree4()`, `PetscFree5()`, `PetscFree6()`
M*/
#define PetscMalloc6(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6) \
  PetscMallocA(6, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6))

/*MC
   PetscCalloc6 - Allocates 6 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc6(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
-  m6 - number of elements to allocate in 6th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
-  r6 - memory allocated in sixth chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscMalloc6()`, `PetscFree6()`
M*/
#define PetscCalloc6(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6) \
  PetscMallocA(6, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6))

/*MC
   PetscMalloc7 - Allocates 7 arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc7(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6,size_t m7,type **r7)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
-  m7 - number of elements to allocate in 7th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
.  r6 - memory allocated in sixth chunk
-  r7 - memory allocated in seventh chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscCalloc7()`, `PetscFree7()`
M*/
#define PetscMalloc7(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6, m7, r7) \
  PetscMallocA(7, PETSC_FALSE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6), ((size_t)((size_t)m7) * sizeof(**(r7))), (r7))

/*MC
   PetscCalloc7 - Allocates 7 cleared (zeroed) arrays of memory, all aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc7(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6,size_t m7,type **r7)

   Not Collective

   Input Parameters:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
-  m7 - number of elements to allocate in 7th chunk  (may be zero)

   Output Parameters:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
.  r6 - memory allocated in sixth chunk
-  r7 - memory allocated in seventh chunk

   Level: developer

.seealso: `PetscFree()`, `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscMalloc7()`, `PetscFree7()`
M*/
#define PetscCalloc7(m1, r1, m2, r2, m3, r3, m4, r4, m5, r5, m6, r6, m7, r7) \
  PetscMallocA(7, PETSC_TRUE, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ((size_t)((size_t)m1) * sizeof(**(r1))), (r1), ((size_t)((size_t)m2) * sizeof(**(r2))), (r2), ((size_t)((size_t)m3) * sizeof(**(r3))), (r3), ((size_t)((size_t)m4) * sizeof(**(r4))), (r4), ((size_t)((size_t)m5) * sizeof(**(r5))), (r5), ((size_t)((size_t)m6) * sizeof(**(r6))), (r6), ((size_t)((size_t)m7) * sizeof(**(r7))), (r7))

/*MC
   PetscNew - Allocates memory of a particular type, zeros the memory! Aligned to `PETSC_MEMALIGN`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscNew(type **result)

   Not Collective

   Output Parameter:
.  result - memory allocated, sized to match pointer `type`

   Level: beginner

   Developer Note:
   Calls `PetscCalloc()` with the appropriate memory size obtained from `type`

.seealso: `PetscFree()`, `PetscMalloc()`, `PetscCall()`, `PetscCalloc1()`, `PetscMalloc1()`
M*/
#define PetscNew(b) PetscCalloc1(1, (b))

#define PetscNewLog(o, b) PETSC_DEPRECATED_MACRO(3, 18, 0, "PetscNew()", ) PetscNew(b)

/*MC
   PetscFree - Frees memory

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree(void *memory)

   Not Collective

   Input Parameter:
.   memory - memory to free (the pointer is ALWAYS set to `NULL` upon success)

   Level: beginner

   Notes:
   Do not free memory obtained with `PetscMalloc2()`, `PetscCalloc2()` etc, they must be freed with `PetscFree2()` etc.

   It is safe to call `PetscFree()` on a `NULL` pointer.

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc1()`, `PetscCalloc1()`
M*/
#define PetscFree(a) ((PetscErrorCode)((*PetscTrFree)((void *)(a), __LINE__, PETSC_FUNCTION_NAME, __FILE__) || ((a) = PETSC_NULLPTR, PETSC_SUCCESS)))

/*MC
   PetscFree2 - Frees 2 chunks of memory obtained with `PetscMalloc2()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree2(void *memory1,void *memory2)

   Not Collective

   Input Parameters:
+   memory1 - memory to free
-   memory2 - 2nd memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc2()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc2()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`
M*/
#define PetscFree2(m1, m2) PetscFreeA(2, __LINE__, PETSC_FUNCTION_NAME, __FILE__, &(m1), &(m2))

/*MC
   PetscFree3 - Frees 3 chunks of memory obtained with `PetscMalloc3()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree3(void *memory1,void *memory2,void *memory3)

   Not Collective

   Input Parameters:
+   memory1 - memory to free
.   memory2 - 2nd memory to free
-   memory3 - 3rd memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc3()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc3()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`
M*/
#define PetscFree3(m1, m2, m3) PetscFreeA(3, __LINE__, PETSC_FUNCTION_NAME, __FILE__, &(m1), &(m2), &(m3))

/*MC
   PetscFree4 - Frees 4 chunks of memory obtained with `PetscMalloc4()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree4(void *m1,void *m2,void *m3,void *m4)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
-   m4 - 4th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc4()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc4()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`
M*/
#define PetscFree4(m1, m2, m3, m4) PetscFreeA(4, __LINE__, PETSC_FUNCTION_NAME, __FILE__, &(m1), &(m2), &(m3), &(m4))

/*MC
   PetscFree5 - Frees 5 chunks of memory obtained with `PetscMalloc5()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree5(void *m1,void *m2,void *m3,void *m4,void *m5)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
-   m5 - 5th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc5()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc5()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`, `PetscMalloc5()`
M*/
#define PetscFree5(m1, m2, m3, m4, m5) PetscFreeA(5, __LINE__, PETSC_FUNCTION_NAME, __FILE__, &(m1), &(m2), &(m3), &(m4), &(m5))

/*MC
   PetscFree6 - Frees 6 chunks of memory obtained with `PetscMalloc6()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree6(void *m1,void *m2,void *m3,void *m4,void *m5,void *m6)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
.   m5 - 5th memory to free
-   m6 - 6th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc6()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc6()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`, `PetscMalloc5()`, `PetscMalloc6()`
M*/
#define PetscFree6(m1, m2, m3, m4, m5, m6) PetscFreeA(6, __LINE__, PETSC_FUNCTION_NAME, __FILE__, &(m1), &(m2), &(m3), &(m4), &(m5), &(m6))

/*MC
   PetscFree7 - Frees 7 chunks of memory obtained with `PetscMalloc7()`

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree7(void *m1,void *m2,void *m3,void *m4,void *m5,void *m6,void *m7)

   Not Collective

   Input Parameters:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
.   m5 - 5th memory to free
.   m6 - 6th memory to free
-   m7 - 7th memory to free

   Level: developer

   Notes:
    Memory must have been obtained with `PetscMalloc7()`

    The arguments need to be in the same order as they were in the call to `PetscMalloc7()`

.seealso: `PetscNew()`, `PetscMalloc()`, `PetscMalloc2()`, `PetscFree()`, `PetscMalloc3()`, `PetscMalloc4()`, `PetscMalloc5()`, `PetscMalloc6()`,
          `PetscMalloc7()`
M*/
#define PetscFree7(m1, m2, m3, m4, m5, m6, m7) PetscFreeA(7, __LINE__, PETSC_FUNCTION_NAME, __FILE__, &(m1), &(m2), &(m3), &(m4), &(m5), &(m6), &(m7))

PETSC_EXTERN PetscErrorCode PetscMallocA(int, PetscBool, int, const char *, const char *, size_t, void *, ...);
PETSC_EXTERN PetscErrorCode PetscFreeA(int, int, const char *, const char *, void *, ...);
PETSC_EXTERN PetscErrorCode (*PetscTrMalloc)(size_t, PetscBool, int, const char[], const char[], void **);
PETSC_EXTERN PetscErrorCode (*PetscTrFree)(void *, int, const char[], const char[]);
PETSC_EXTERN PetscErrorCode (*PetscTrRealloc)(size_t, int, const char[], const char[], void **);
PETSC_EXTERN PetscErrorCode PetscMallocSetCoalesce(PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocSet(PetscErrorCode (*)(size_t, PetscBool, int, const char[], const char[], void **), PetscErrorCode (*)(void *, int, const char[], const char[]), PetscErrorCode (*)(size_t, int, const char[], const char[], void **));
PETSC_EXTERN PetscErrorCode PetscMallocClear(void);

/*
  Unlike PetscMallocSet and PetscMallocClear which overwrite the existing settings, these two functions save the previous choice of allocator, and should be used in pair.
*/
PETSC_EXTERN PetscErrorCode PetscMallocSetDRAM(void);
PETSC_EXTERN PetscErrorCode PetscMallocResetDRAM(void);
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscMallocSetCUDAHost(void);
PETSC_EXTERN PetscErrorCode PetscMallocResetCUDAHost(void);
#endif
#if defined(PETSC_HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscMallocSetHIPHost(void);
PETSC_EXTERN PetscErrorCode PetscMallocResetHIPHost(void);
#endif

#define MPIU_PETSCLOGDOUBLE  MPI_DOUBLE
#define MPIU_2PETSCLOGDOUBLE MPI_2DOUBLE_PRECISION

/*
   Routines for tracing memory corruption/bleeding with default PETSc memory allocation
*/
PETSC_EXTERN PetscErrorCode PetscMallocDump(FILE *);
PETSC_EXTERN PetscErrorCode PetscMallocView(FILE *);
PETSC_EXTERN PetscErrorCode PetscMallocGetCurrentUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocGetMaximumUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocPushMaximumUsage(int);
PETSC_EXTERN PetscErrorCode PetscMallocPopMaximumUsage(int, PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocSetDebug(PetscBool, PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocGetDebug(PetscBool *, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscMallocValidate(int, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscMallocViewSet(PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscMallocViewGet(PetscBool *);
PETSC_EXTERN PetscErrorCode PetscMallocLogRequestedSizeSet(PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocLogRequestedSizeGet(PetscBool *);

PETSC_EXTERN PetscErrorCode PetscDataTypeToMPIDataType(PetscDataType, MPI_Datatype *);
PETSC_EXTERN PetscErrorCode PetscMPIDataTypeToPetscDataType(MPI_Datatype, PetscDataType *);
PETSC_EXTERN PetscErrorCode PetscDataTypeGetSize(PetscDataType, size_t *);
PETSC_EXTERN PetscErrorCode PetscDataTypeFromString(const char *, PetscDataType *, PetscBool *);

/*
   These are MPI operations for MPI_Allreduce() etc
*/
PETSC_EXTERN MPI_Op MPIU_MAXSUM_OP;
#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
PETSC_EXTERN MPI_Op MPIU_SUM;
PETSC_EXTERN MPI_Op MPIU_MAX;
PETSC_EXTERN MPI_Op MPIU_MIN;
#else
  #define MPIU_SUM MPI_SUM
  #define MPIU_MAX MPI_MAX
  #define MPIU_MIN MPI_MIN
#endif
PETSC_EXTERN MPI_Op         Petsc_Garbage_SetIntersectOp;
PETSC_EXTERN PetscErrorCode PetscMaxSum(MPI_Comm, const PetscInt[], PetscInt *, PetscInt *);

#if (defined(PETSC_HAVE_REAL___FLOAT128) && !defined(PETSC_SKIP_REAL___FLOAT128)) || (defined(PETSC_HAVE_REAL___FP16) && !defined(PETSC_SKIP_REAL___FP16))
/*MC
   MPIU_SUM___FP16___FLOAT128 - MPI_Op that acts as a replacement for `MPI_SUM` with
   custom `MPI_Datatype` `MPIU___FLOAT128`, `MPIU___COMPLEX128`, and `MPIU___FP16`.

   Level: advanced

   Developer Note:
   This should be unified with `MPIU_SUM`

.seealso: `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`
M*/
PETSC_EXTERN MPI_Op MPIU_SUM___FP16___FLOAT128;
#endif

/*
     These are so that in extern C code we can cast function pointers to non-extern C
   function pointers. Since the regular C++ code expects its function pointers to be C++
*/

/*S
  PetscVoidFn - A prototype of a `void fn(void)` function

  Level: advanced

  Notes:
  `PetscVoidFn *` plays the role of `void *` for function pointers in the PETSc API that do not return an error code.
  It is used where a function pointer is needed but it is not possible to use the full prototype of the function.

  `PetscErrorCodeFn` is similar to `PetscVoidFn` but should be used when the function returns a `PetscErrorCode`

  The deprecated `PetscVoidFunction` works as a replacement for `PetscVoidFn` *.

  The deprecated `PetscVoidStarFunction` works as a replacement for `PetscVoidFn` **.

.seealso: `PetscErrorCodeFn`, `PetscObject`, `PetscObjectDestroy()`
S*/
PETSC_EXTERN_TYPEDEF typedef void PetscVoidFn(void);

PETSC_EXTERN_TYPEDEF typedef PetscVoidFn  *PetscVoidFunction;
PETSC_EXTERN_TYPEDEF typedef PetscVoidFn **PetscVoidStarFunction;

/*S
  PetscErrorCodeFn - a function typedef that represents abstractly a function that returns a PETSc error code
  and takes any number of arguments. Since C/C++ has no way to express this concept, it is implemented as `void (fn)(void)`.

  Level: advanced

  Notes:
  `PetscErrorCodeFn *` plays the role of `void *` for function pointers in the PETSc API that return an error code.
  It is used where a function pointer is needed but it is not possible to use the full prototype of the function,
  for example `VecSetOperation()`.

  `PetscVoidFn` is similar to `PetscErrorCodeFn` but should be used when the function does not return a `PetscErrorCode`.

  The deprecated `PetscErrorCodeFunction` works as a replacement for `PetscErrorCodeFn` *.

  Developer Notes:
  This function type is equivalent to `PetscVoidFn`*.

  At the C/C++ syntax level this construct adds nothing of value to the PETSc source code. It provides a way, at the abstract
  PETSc API level, to indicate specifically functions that return PETSc error codes as opposed to any C/C++ function.

.seealso: `PetscVoidFn`, `PetscObject`, `PetscObjectDestroy()`, `VecSetOperation()`
S*/
PETSC_EXTERN_TYPEDEF typedef void PetscErrorCodeFn(void);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCodeFn *PetscErrorCodeFunction;

/*S
  PetscCtxDestroyFn - A prototype of a `PetscErrorCode (*)(void **)` function that is used to free user contexts

  Level: intermediate

  Note:
  Used in the prototype of functions such as `DMSetApplicationContextDestroy()`

.seealso: `PetscObject`, `PetscCtxDestroyDefault()`, `PetscObjectDestroy()`, `DMSetApplicationContextDestroy()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscCtxDestroyFn(void **);

PETSC_EXTERN PetscCtxDestroyFn PetscCtxDestroyDefault;

/*
    Defines PETSc error handling.
*/
#include <petscerror.h> // IWYU pragma: export

PETSC_EXTERN PetscBool   PetscCIEnabled;                    /* code is running in the PETSc test harness CI */
PETSC_EXTERN PetscBool   PetscCIEnabledPortableErrorOutput; /* error output is stripped to ensure portability of error messages across systems */
PETSC_EXTERN const char *PetscCIFilename(const char *);
PETSC_EXTERN int         PetscCILinenumber(int);

#define PETSC_SMALLEST_CLASSID 1211211
PETSC_EXTERN PetscClassId   PETSC_LARGEST_CLASSID;
PETSC_EXTERN PetscClassId   PETSC_OBJECT_CLASSID;
PETSC_EXTERN PetscErrorCode PetscClassIdRegister(const char[], PetscClassId *);
PETSC_EXTERN PetscErrorCode PetscObjectGetId(PetscObject, PetscObjectId *);
PETSC_EXTERN PetscErrorCode PetscObjectCompareId(PetscObject, PetscObjectId, PetscBool *);

/*
   Routines that get memory usage information from the OS
*/
PETSC_EXTERN PetscErrorCode PetscMemoryGetCurrentUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMemoryGetMaximumUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMemorySetGetMaximumUsage(void);
PETSC_EXTERN PetscErrorCode PetscMemoryTrace(const char[]);

PETSC_EXTERN PetscErrorCode PetscSleep(PetscReal);

/*
   Initialization of PETSc
*/
PETSC_EXTERN PetscErrorCode PetscInitialize(int *, char ***, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscInitializeNoPointers(int, char *[], const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscInitializeNoArguments(void);
PETSC_EXTERN PetscErrorCode PetscInitialized(PetscBool *);
PETSC_EXTERN PetscErrorCode PetscFinalized(PetscBool *);
PETSC_EXTERN PetscErrorCode PetscFinalize(void);
PETSC_EXTERN PetscErrorCode PetscInitializeFortran(void);
PETSC_EXTERN PetscErrorCode PetscGetArgs(int *, char ***);
PETSC_EXTERN PetscErrorCode PetscGetArguments(char ***);
PETSC_EXTERN PetscErrorCode PetscFreeArguments(char **);

PETSC_EXTERN PetscErrorCode PetscEnd(void);
PETSC_EXTERN PetscErrorCode PetscSysInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscSysFinalizePackage(void);

PETSC_EXTERN PetscErrorCode PetscPythonInitialize(const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscPythonFinalize(void);
PETSC_EXTERN PetscErrorCode PetscPythonPrintError(void);
PETSC_EXTERN PetscErrorCode PetscPythonMonitorSet(PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode PetscMonitorCompare(PetscErrorCode (*)(void), void *, PetscCtxDestroyFn *, PetscErrorCode (*)(void), void *, PetscCtxDestroyFn *, PetscBool *);

/*
    Functions that can act on any PETSc object.
*/
PETSC_EXTERN PetscErrorCode PetscObjectDestroy(PetscObject *);
PETSC_EXTERN PetscErrorCode PetscObjectGetComm(PetscObject, MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscObjectGetClassId(PetscObject, PetscClassId *);
PETSC_EXTERN PetscErrorCode PetscObjectGetClassName(PetscObject, const char *[]);
PETSC_EXTERN PetscErrorCode PetscObjectGetType(PetscObject, const char *[]);
PETSC_EXTERN PetscErrorCode PetscObjectSetName(PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectGetName(PetscObject, const char *[]);
PETSC_EXTERN PetscErrorCode PetscObjectSetTabLevel(PetscObject, PetscInt);
PETSC_EXTERN PetscErrorCode PetscObjectGetTabLevel(PetscObject, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscObjectIncrementTabLevel(PetscObject, PetscObject, PetscInt);
PETSC_EXTERN PetscErrorCode PetscObjectReference(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectGetReference(PetscObject, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscObjectDereference(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectGetNewTag(PetscObject, PetscMPIInt *);
PETSC_EXTERN PetscErrorCode PetscObjectCompose(PetscObject, const char[], PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectRemoveReference(PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectQuery(PetscObject, const char[], PetscObject *);
PETSC_EXTERN PetscErrorCode PetscObjectComposeFunction_Private(PetscObject, const char[], PetscErrorCodeFn *);
#define PetscObjectComposeFunction(a, b, ...) PetscObjectComposeFunction_Private((a), (b), (PetscErrorCodeFn *)(__VA_ARGS__))
PETSC_EXTERN PetscErrorCode PetscObjectSetFromOptions(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSetUp(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSetPrintedOptions(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectInheritPrintedOptions(PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscCommGetNewTag(MPI_Comm, PetscMPIInt *);

/*MC
   PetscObjectParameterSetDefault - sets a parameter default value in a `PetscObject` to a new default value.
   If the current value matches the old default value, then the current value is also set to the new value.

   No Fortran Support

   Synopsis:
   #include <petscsys.h>
   PetscBool PetscObjectParameterSetDefault(PetscObject obj, char* NAME, PetscReal value);

   Input Parameters:
+  obj - the `PetscObject`
.  NAME - the name of the parameter, unquoted
-  value - the new value

   Level: developer

   Notes:
   The defaults for an object are the values set when the object's type is set.

   This should only be used in object constructors, such as, `SNESCreate_NGS()`.

   This only works for parameters that are declared in the struct with `PetscObjectParameterDeclare()`

.seealso: `PetscObjectParameterDeclare()`, `PetscInitialize()`, `PetscFinalize()`, `PetscObject`, `SNESParametersInitialize()`
M*/
#define PetscObjectParameterSetDefault(obj, NAME, value) \
  do { \
    if (obj->NAME == obj->default_##NAME) obj->NAME = value; \
    obj->default_##NAME = value; \
  } while (0)

/*MC
   PetscObjectParameterDeclare - declares a parameter in a `PetscObject` and a location to store its default

   No Fortran Support

   Synopsis:
   #include <petscsys.h>
   PetscBool PetscObjectParameterDeclare(type, char* NAME)

   Input Parameters:
+  type - the type of the parameter, for example `PetscInt`
-  NAME - the name of the parameter, unquoted

   Level: developer.

.seealso: `PetscObjectParameterSetDefault()`, `PetscInitialize()`, `PetscFinalize()`, `PetscObject`, `SNESParametersInitialize()`
M*/
#define PetscObjectParameterDeclare(type, NAME)    type NAME, default_##NAME
#define PetscObjectParameterDeclarePtr(type, NAME) type *NAME, *default_##NAME

#include <petscviewertypes.h>
#include <petscoptions.h>

PETSC_EXTERN PetscErrorCode PetscMallocTraceSet(PetscViewer, PetscBool, PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscMallocTraceGet(PetscBool *);

PETSC_EXTERN PetscErrorCode PetscObjectsListGetGlobalNumbering(MPI_Comm, PetscInt, PetscObject[], PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode PetscMemoryView(PetscViewer, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectPrintClassNamePrefixType(PetscObject, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscObjectView(PetscObject, PetscViewer);
#define PetscObjectQueryFunction(obj, name, fptr) PetscObjectQueryFunction_Private((obj), (name), (PetscErrorCodeFn **)(fptr))
PETSC_EXTERN PetscErrorCode PetscObjectHasFunction(PetscObject, const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscObjectQueryFunction_Private(PetscObject, const char[], PetscErrorCodeFn **);
PETSC_EXTERN PetscErrorCode PetscObjectSetOptionsPrefix(PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectAppendOptionsPrefix(PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectPrependOptionsPrefix(PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectGetOptionsPrefix(PetscObject, const char *[]);
PETSC_EXTERN PetscErrorCode PetscObjectChangeTypeName(PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectRegisterDestroy(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectRegisterDestroyAll(void);
PETSC_EXTERN PetscErrorCode PetscObjectViewFromOptions(PetscObject, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectName(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectTypeCompare(PetscObject, const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscObjectObjectTypeCompare(PetscObject, PetscObject, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscObjectBaseTypeCompare(PetscObject, const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscObjectTypeCompareAny(PetscObject, PetscBool *, const char[], ...);
PETSC_EXTERN PetscErrorCode PetscObjectBaseTypeCompareAny(PetscObject, PetscBool *, const char[], ...);
PETSC_EXTERN PetscErrorCode PetscRegisterFinalize(PetscErrorCode (*)(void));
PETSC_EXTERN PetscErrorCode PetscRegisterFinalizeAll(void);

#if defined(PETSC_HAVE_SAWS)
PETSC_EXTERN PetscErrorCode PetscSAWsBlock(void);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsViewOff(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsSetBlock(PetscObject, PetscBool);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsBlock(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsGrantAccess(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsTakeAccess(PetscObject);
PETSC_EXTERN void           PetscStackSAWsGrantAccess(void);
PETSC_EXTERN void           PetscStackSAWsTakeAccess(void);
PETSC_EXTERN PetscErrorCode PetscStackViewSAWs(void);
PETSC_EXTERN PetscErrorCode PetscStackSAWsViewOff(void);

#else
  #define PetscSAWsBlock()                  PETSC_SUCCESS
  #define PetscObjectSAWsViewOff(obj)       PETSC_SUCCESS
  #define PetscObjectSAWsSetBlock(obj, flg) PETSC_SUCCESS
  #define PetscObjectSAWsBlock(obj)         PETSC_SUCCESS
  #define PetscObjectSAWsGrantAccess(obj)   PETSC_SUCCESS
  #define PetscObjectSAWsTakeAccess(obj)    PETSC_SUCCESS
  #define PetscStackViewSAWs()              PETSC_SUCCESS
  #define PetscStackSAWsViewOff()           PETSC_SUCCESS
  #define PetscStackSAWsTakeAccess()
  #define PetscStackSAWsGrantAccess()

#endif

PETSC_EXTERN PetscErrorCode PetscDLOpen(const char[], PetscDLMode, PetscDLHandle *);
PETSC_EXTERN PetscErrorCode PetscDLClose(PetscDLHandle *);
PETSC_EXTERN PetscErrorCode PetscDLSym(PetscDLHandle, const char[], void **);
PETSC_EXTERN PetscErrorCode PetscDLAddr(PetscVoidFn *, char *[]);
#ifdef PETSC_HAVE_CXX
PETSC_EXTERN PetscErrorCode PetscDemangleSymbol(const char *, char *[]);
#endif

PETSC_EXTERN PetscErrorCode PetscMallocGetStack(void *, PetscStack **);

PETSC_EXTERN PetscErrorCode PetscObjectsDump(FILE *, PetscBool);
PETSC_EXTERN PetscErrorCode PetscObjectsView(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscObjectsGetObject(const char *, PetscObject *, const char *[]);
PETSC_EXTERN PetscErrorCode PetscObjectListDestroy(PetscObjectList *);
PETSC_EXTERN PetscErrorCode PetscObjectListFind(PetscObjectList, const char[], PetscObject *);
PETSC_EXTERN PetscErrorCode PetscObjectListReverseFind(PetscObjectList, PetscObject, const char *[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscObjectListAdd(PetscObjectList *, const char[], PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectListRemoveReference(PetscObjectList *, const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectListDuplicate(PetscObjectList, PetscObjectList *);

/*
    Dynamic library lists. Lists of names of routines in objects or in dynamic
  link libraries that will be loaded as needed.
*/

#define PetscFunctionListAdd(list, name, fptr) PetscFunctionListAdd_Private((list), (name), (PetscErrorCodeFn *)(fptr))
PETSC_EXTERN PetscErrorCode PetscFunctionListAdd_Private(PetscFunctionList *, const char[], PetscErrorCodeFn *);
PETSC_EXTERN PetscErrorCode PetscFunctionListDestroy(PetscFunctionList *);
PETSC_EXTERN PetscErrorCode PetscFunctionListClear(PetscFunctionList);
#define PetscFunctionListFind(list, name, fptr) PetscFunctionListFind_Private((list), (name), (PetscErrorCodeFn **)(fptr))
PETSC_EXTERN PetscErrorCode PetscFunctionListFind_Private(PetscFunctionList, const char[], PetscErrorCodeFn **);
PETSC_EXTERN PetscErrorCode PetscFunctionListPrintTypes(MPI_Comm, FILE *, const char[], const char[], const char[], const char[], PetscFunctionList, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscFunctionListDuplicate(PetscFunctionList, PetscFunctionList *);
PETSC_EXTERN PetscErrorCode PetscFunctionListView(PetscFunctionList, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscFunctionListGet(PetscFunctionList, const char ***, int *);
PETSC_EXTERN PetscErrorCode PetscFunctionListPrintNonEmpty(PetscFunctionList);
PETSC_EXTERN PetscErrorCode PetscFunctionListPrintAll(void);

PETSC_EXTERN PetscDLLibrary PetscDLLibrariesLoaded;
PETSC_EXTERN PetscErrorCode PetscDLLibraryAppend(MPI_Comm, PetscDLLibrary *, const char[]);
PETSC_EXTERN PetscErrorCode PetscDLLibraryPrepend(MPI_Comm, PetscDLLibrary *, const char[]);
PETSC_EXTERN PetscErrorCode PetscDLLibrarySym(MPI_Comm, PetscDLLibrary *, const char[], const char[], void **);
PETSC_EXTERN PetscErrorCode PetscDLLibraryPrintPath(PetscDLLibrary);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRetrieve(MPI_Comm, const char[], char *, size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDLLibraryOpen(MPI_Comm, const char[], PetscDLLibrary *);
PETSC_EXTERN PetscErrorCode PetscDLLibraryClose(PetscDLLibrary);

/*
     Useful utility routines
*/
PETSC_EXTERN PetscErrorCode PetscSplitOwnership(MPI_Comm, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSplitOwnershipBlock(MPI_Comm, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSplitOwnershipEqual(MPI_Comm, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSequentialPhaseBegin(MPI_Comm, PetscMPIInt);
PETSC_EXTERN PetscErrorCode PetscSequentialPhaseEnd(MPI_Comm, PetscMPIInt);
PETSC_EXTERN PetscErrorCode PetscBarrier(PetscObject);
PETSC_EXTERN PetscErrorCode PetscMPIDump(FILE *);
PETSC_EXTERN PetscErrorCode PetscGlobalMinMaxInt(MPI_Comm, const PetscInt[2], PetscInt[2]);
PETSC_EXTERN PetscErrorCode PetscGlobalMinMaxReal(MPI_Comm, const PetscReal[2], PetscReal[2]);

/*MC
    PetscNot - negates a logical type value and returns result as a `PetscBool`

    Level: beginner

    Note:
    This is useful in cases like
.vb
     int        *a;
     PetscBool  flag = PetscNot(a)
.ve
     where !a would not return a `PetscBool` because we cannot provide a cast from int to `PetscBool` in C.

.seealso: `PetscBool`, `PETSC_TRUE`, `PETSC_FALSE`
M*/
#define PetscNot(a) ((a) ? PETSC_FALSE : PETSC_TRUE)

/*MC
   PetscHelpPrintf - Prints help messages.

   Synopsis:
    #include <petscsys.h>
     PetscErrorCode (*PetscHelpPrintf)(MPI_Comm comm, const char format[],args);

   Not Collective, only applies on MPI rank 0; No Fortran Support

   Input Parameters:
+  comm - the MPI communicator over which the help message is printed
.  format - the usual printf() format string
-  args - arguments to be printed

   Level: developer

   Notes:
   You can change how help messages are printed by replacing the function pointer with a function that does not simply write to stdout.

   To use, write your own function, for example,
.vb
   PetscErrorCode mypetschelpprintf(MPI_Comm comm,const char format[],....)
   {
     PetscFunctionReturn(PETSC_SUCCESS);
   }
.ve
then do the assignment
.vb
  PetscHelpPrintf = mypetschelpprintf;
.ve

  You can do the assignment before `PetscInitialize()`.

  The default routine used is called `PetscHelpPrintfDefault()`.

.seealso: `PetscFPrintf()`, `PetscSynchronizedPrintf()`, `PetscErrorPrintf()`, `PetscHelpPrintfDefault()`
M*/
PETSC_EXTERN PetscErrorCode (*PetscHelpPrintf)(MPI_Comm, const char[], ...) PETSC_ATTRIBUTE_FORMAT(2, 3);

/*
     Defines PETSc profiling.
*/
#include <petsclog.h>

/*
      Simple PETSc parallel IO for ASCII printing
*/
PETSC_EXTERN PetscErrorCode PetscFixFilename(const char[], char[]);
PETSC_EXTERN PetscErrorCode PetscFOpen(MPI_Comm, const char[], const char[], FILE **);
PETSC_EXTERN PetscErrorCode PetscFClose(MPI_Comm, FILE *);
PETSC_EXTERN PetscErrorCode PetscFPrintf(MPI_Comm, FILE *, const char[], ...) PETSC_ATTRIBUTE_FORMAT(3, 4);
PETSC_EXTERN PetscErrorCode PetscFFlush(FILE *);
PETSC_EXTERN PetscErrorCode PetscPrintf(MPI_Comm, const char[], ...) PETSC_ATTRIBUTE_FORMAT(2, 3);
PETSC_EXTERN PetscErrorCode PetscSNPrintf(char *, size_t, const char[], ...) PETSC_ATTRIBUTE_FORMAT(3, 4);
PETSC_EXTERN PetscErrorCode PetscSNPrintfCount(char *, size_t, const char[], size_t *, ...) PETSC_ATTRIBUTE_FORMAT(3, 5);
PETSC_EXTERN PetscErrorCode PetscFormatRealArray(char[], size_t, const char *, PetscInt, const PetscReal[]);

PETSC_EXTERN PetscErrorCode PetscErrorPrintfDefault(const char[], ...) PETSC_ATTRIBUTE_FORMAT(1, 2);
PETSC_EXTERN PetscErrorCode PetscErrorPrintfNone(const char[], ...) PETSC_ATTRIBUTE_FORMAT(1, 2);
PETSC_EXTERN PetscErrorCode PetscHelpPrintfDefault(MPI_Comm, const char[], ...) PETSC_ATTRIBUTE_FORMAT(2, 3);

PETSC_EXTERN PetscErrorCode PetscFormatConvertGetSize(const char *, size_t *);
PETSC_EXTERN PetscErrorCode PetscFormatConvert(const char *, char *);

PETSC_EXTERN PetscErrorCode PetscPOpen(MPI_Comm, const char[], const char[], const char[], FILE **);
PETSC_EXTERN PetscErrorCode PetscPClose(MPI_Comm, FILE *);
PETSC_EXTERN PetscErrorCode PetscPOpenSetMachine(const char[]);

PETSC_EXTERN PetscErrorCode PetscSynchronizedPrintf(MPI_Comm, const char[], ...) PETSC_ATTRIBUTE_FORMAT(2, 3);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFPrintf(MPI_Comm, FILE *, const char[], ...) PETSC_ATTRIBUTE_FORMAT(3, 4);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFlush(MPI_Comm, FILE *);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFGets(MPI_Comm, FILE *, size_t, char[]);
PETSC_EXTERN PetscErrorCode PetscStartMatlab(MPI_Comm, const char[], const char[], FILE **);
PETSC_EXTERN PetscErrorCode PetscGetPetscDir(const char *[]);

/*MC
   PeCtx - indicates an argument that returns a pointer to a user defined C struct (or Fortran derived type)

   Level: developer

   Notes:
   This is not part of the PETSc public API and should only be used in PETSc source code.

   This should not be used for functions that return PETSc objects, or pointers to arrays of unknown type. Thus it is used for, for example,
   `KSPGetApplicationContext()` but not used for `DMNetworkGetComponent()`

   For pointers to arrays of unknown type and for functions that return PETSc internal objects that are opaque to users, such
   as `KSPMonitorDynamicToleranceCreate()` a `void **` should be used.

   Fortran Note:
   Should only be used with user defined Fortran datatypes
.vb
   type(tUserType), pointer :: ctx
.ve

   Developer Note:
   Put this in function declaration for the argument type instead of `void *`, or `void **`.

   C compilers generate a warning or error if one passes a pointer to a pointer to a specific type (instead of `void`), for example,
.vb
   extern calledfunction(void **);
   SomeCtx *ctx;
   calledfunction(&ctx);   << warning that it is passing a pointer to a pointer to a SomeCtx instead of a void **
.ve
   By using the common practice of prototyping the function as
.vb
   extern calledfunction(void *);
.ve
   the warning message is averted. `PeCtx` is used in PETSc source code so that the getAPI() code processor knows the argument is
   actually handled internally as `void **` so it can generate correct bindings for other languages.

.seealso: `PeOp`, `PeNS`, `PetscInitialize()`
M*/
typedef void *PeCtx;

PETSC_EXTERN PetscClassId   PETSC_CONTAINER_CLASSID;
PETSC_EXTERN PetscErrorCode PetscContainerGetPointer(PetscContainer, void *);
PETSC_EXTERN PetscErrorCode PetscContainerSetPointer(PetscContainer, void *);
PETSC_EXTERN PetscErrorCode PetscContainerDestroy(PetscContainer *);
PETSC_EXTERN PetscErrorCode PetscContainerCreate(MPI_Comm, PetscContainer *);
PETSC_EXTERN PetscErrorCode PetscContainerSetCtxDestroy(PetscContainer, PetscCtxDestroyFn *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 23, 0, "PetscContainerSetCtxDestroy()", ) PetscErrorCode PetscContainerSetUserDestroy(PetscContainer, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode PetscObjectContainerCompose(PetscObject, const char *name, void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode PetscObjectContainerQuery(PetscObject, const char *, PeCtx);

PETSC_DEPRECATED_FUNCTION(3, 23, 0, "PetscCtxDestroyDefault()", ) static inline PetscErrorCode PetscContainerCtxDestroyDefault(void **a)
{
  return PetscCtxDestroyDefault(a);
}

/*
   For use in debuggers
*/
PETSC_EXTERN PetscMPIInt    PetscGlobalRank;
PETSC_EXTERN PetscMPIInt    PetscGlobalSize;
PETSC_EXTERN PetscErrorCode PetscIntViewNumColumns(PetscInt, PetscInt, const PetscInt[], PetscViewer);
PETSC_EXTERN PetscErrorCode PetscRealViewNumColumns(PetscInt, PetscInt, const PetscReal[], PetscViewer);
PETSC_EXTERN PetscErrorCode PetscScalarViewNumColumns(PetscInt, PetscInt, const PetscScalar[], PetscViewer);
PETSC_EXTERN PetscErrorCode PetscIntView(PetscInt, const PetscInt[], PetscViewer);
PETSC_EXTERN PetscErrorCode PetscRealView(PetscInt, const PetscReal[], PetscViewer);
PETSC_EXTERN PetscErrorCode PetscScalarView(PetscInt, const PetscScalar[], PetscViewer);

/*
    Basic memory and string operations. These are usually simple wrappers
   around the basic Unix system calls, but a few of them have additional
   functionality and/or error checking.
*/
#include <petscstring.h>

#include <stddef.h>
#include <stdlib.h>

#if defined(PETSC_CLANG_STATIC_ANALYZER)
  #define PetscPrefetchBlock(a, b, c, d)
#else
  /*MC
   PetscPrefetchBlock - Prefetches a block of memory

   Synopsis:
    #include <petscsys.h>
    void PetscPrefetchBlock(const anytype *a,size_t n,int rw,int t)

   Not Collective

   Input Parameters:
+  a  - pointer to first element to fetch (any type but usually `PetscInt` or `PetscScalar`)
.  n  - number of elements to fetch
.  rw - 1 if the memory will be written to, otherwise 0 (ignored by many processors)
-  t  - temporal locality (PETSC_PREFETCH_HINT_{NTA,T0,T1,T2}), see note

   Level: developer

   Notes:
   The last two arguments (`rw` and `t`) must be compile-time constants.

   Adopting Intel's x86/x86-64 conventions, there are four levels of temporal locality.  Not all architectures offer
   equivalent locality hints, but the following macros are always defined to their closest analogue.
+  `PETSC_PREFETCH_HINT_NTA` - Non-temporal.  Prefetches directly to L1, evicts to memory (skips higher level cache unless it was already there when prefetched).
.  `PETSC_PREFETCH_HINT_T0`  - Fetch to all levels of cache and evict to the closest level.  Use this when the memory will be reused regularly despite necessary eviction from L1.
.  `PETSC_PREFETCH_HINT_T1`  - Fetch to level 2 and higher (not L1).
-  `PETSC_PREFETCH_HINT_T2`  - Fetch to high-level cache only.  (On many systems, T0 and T1 are equivalent.)

   This function does nothing on architectures that do not support prefetch and never errors (even if passed an invalid
   address).

M*/
  #define PetscPrefetchBlock(a, n, rw, t) \
    do { \
      const char *_p = (const char *)(a), *_end = (const char *)((a) + (n)); \
      for (; _p < _end; _p += PETSC_LEVEL1_DCACHE_LINESIZE) PETSC_Prefetch(_p, (rw), (t)); \
    } while (0)
#endif
/*
      Determine if some of the kernel computation routines use
   Fortran (rather than C) for the numerical calculations. On some machines
   and compilers (like complex numbers) the Fortran version of the routines
   is faster than the C/C++ versions. The flag --with-fortran-kernels
   should be used with ./configure to turn these on.
*/
#if defined(PETSC_USE_FORTRAN_KERNELS)

  #if !defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
    #define PETSC_USE_FORTRAN_KERNEL_MULTCRL
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
    #define PETSC_USE_FORTRAN_KERNEL_MULTAIJ
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
    #define PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)
    #define PETSC_USE_FORTRAN_KERNEL_MAXPY
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
    #define PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ)
    #define PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
    #define PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
    #define PETSC_USE_FORTRAN_KERNEL_MDOT
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    #define PETSC_USE_FORTRAN_KERNEL_XTIMESY
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
    #define PETSC_USE_FORTRAN_KERNEL_AYPX
  #endif

  #if !defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    #define PETSC_USE_FORTRAN_KERNEL_WAXPY
  #endif

#endif

/*
    Macros for indicating code that should be compiled with a C interface,
   rather than a C++ interface. Any routines that are dynamically loaded
   (such as the PCCreate_XXX() routines) must be wrapped so that the name
   mangler does not change the functions symbol name. This just hides the
   ugly extern "C" {} wrappers.
*/
#if defined(__cplusplus)
  #define EXTERN_C_BEGIN extern "C" {
  #define EXTERN_C_END   }
#else
  #define EXTERN_C_BEGIN
  #define EXTERN_C_END
#endif

/*MC
   MPI_Comm - the basic object used by MPI to determine which processes are involved in a
   communication

   Level: beginner

   Note:
   This manual page is a place-holder because MPICH does not have a manual page for `MPI_Comm`

.seealso: `PETSC_COMM_WORLD`, `PETSC_COMM_SELF`
M*/

#if defined(PETSC_HAVE_MPIIO)
PETSC_EXTERN PetscErrorCode MPIU_File_write_all(MPI_File, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(2, 4);
PETSC_EXTERN PetscErrorCode MPIU_File_read_all(MPI_File, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(2, 4);
PETSC_EXTERN PetscErrorCode MPIU_File_write_at(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
PETSC_EXTERN PetscErrorCode MPIU_File_read_at(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
PETSC_EXTERN PetscErrorCode MPIU_File_write_at_all(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
PETSC_EXTERN PetscErrorCode MPIU_File_read_at_all(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
#endif

#if defined(PETSC_HAVE_MPI_COUNT)
typedef MPI_Count MPIU_Count;
#else
typedef PetscInt64 MPIU_Count;
#endif

/*@C
   PetscIntCast - casts a `MPI_Count`, `PetscInt64`, `PetscCount`, or `size_t` to a `PetscInt` (which may be 32-bits in size), generates an
   error if the `PetscInt` is not large enough to hold the number.

   Not Collective; No Fortran Support

   Input Parameter:
.  a - the `PetscInt64` value

   Output Parameter:
.  b - the resulting `PetscInt` value, or `NULL` if the result is not needed

   Level: advanced

   Note:
   If integers needed for the applications are too large to fit in 32-bit ints you can ./configure using `--with-64-bit-indices` to make `PetscInt` use 64-bit integers

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscMPIIntCast()`, `PetscBLASIntCast()`, `PetscCIntCast()`, `PetscIntMultError()`, `PetscIntSumError()`
@*/
static inline PetscErrorCode PetscIntCast(MPIU_Count a, PetscInt *b)
{
  PetscFunctionBegin;
  if (b) *b = 0; /* to prevent compilers erroneously suggesting uninitialized variable */
  PetscCheck(sizeof(MPIU_Count) <= sizeof(PetscInt) || (a <= (MPIU_Count)PETSC_INT_MAX && a >= (MPIU_Count)PETSC_INT_MIN), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt64_FMT " is too big for PetscInt, you may need to ./configure using --with-64-bit-indices", (PetscInt64)a);
  if (b) *b = (PetscInt)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscBLASIntCast - casts a `MPI_Count`, `PetscInt`, `PetscCount` or `PetscInt64` to a `PetscBLASInt` (which may be 32-bits in size), generates an
   error if the `PetscBLASInt` is not large enough to hold the number.

   Not Collective; No Fortran Support

   Input Parameter:
.  a - the `PetscInt` value

   Output Parameter:
.  b - the resulting `PetscBLASInt` value, or `NULL` if the result is not needed

   Level: advanced

   Note:
   Errors if the integer is negative since PETSc calls to BLAS/LAPACK never need to cast negative integer inputs

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscMPIIntCast()`, `PetscCIntCast()`, `PetscIntCast()`
@*/
static inline PetscErrorCode PetscBLASIntCast(MPIU_Count a, PetscBLASInt *b)
{
  PetscFunctionBegin;
  if (b) *b = 0; /* to prevent compilers erroneously suggesting uninitialized variable */
  PetscCheck(sizeof(MPIU_Count) <= sizeof(PetscBLASInt) || a <= (MPIU_Count)PETSC_BLAS_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt64_FMT " is too big for BLAS/LAPACK, which is restricted to 32-bit integers. Either you have an invalidly large integer error in your code or you must ./configure PETSc with --with-64-bit-blas-indices for the case you are running", (PetscInt64)a);
  PetscCheck(a >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Passing negative integer to BLAS/LAPACK routine");
  if (b) *b = (PetscBLASInt)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscCuBLASIntCast - like `PetscBLASIntCast()`, but for `PetscCuBLASInt`.

   Not Collective; No Fortran Support

   Input Parameter:
.  a - the `PetscInt` value

   Output Parameter:
.  b - the resulting `PetscCuBLASInt` value, or `NULL` if the result is not needed

   Level: advanced

   Note:
   Errors if the integer is negative since PETSc calls to cuBLAS and friends never need to cast negative integer inputs

.seealso: `PetscCuBLASInt`, `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscMPIIntCast()`, `PetscCIntCast()`, `PetscIntCast()`
@*/
static inline PetscErrorCode PetscCuBLASIntCast(MPIU_Count a, PetscCuBLASInt *b)
{
  PetscFunctionBegin;
  if (b) *b = 0; /* to prevent compilers erroneously suggesting uninitialized variable */
  PetscCheck(sizeof(MPIU_Count) <= sizeof(PetscCuBLASInt) || a <= (MPIU_Count)PETSC_CUBLAS_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt64_FMT " is too big for cuBLAS, which is restricted to 32-bit integers.", (PetscInt64)a);
  PetscCheck(a >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Passing negative integer %" PetscInt64_FMT "to cuBLAS routine", (PetscInt64)a);
  if (b) *b = (PetscCuBLASInt)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscHipBLASIntCast - like `PetscBLASIntCast()`, but for `PetscHipBLASInt`.

   Not Collective; No Fortran Support

   Input Parameter:
.  a - the `PetscInt` value

   Output Parameter:
.  b - the resulting `PetscHipBLASInt` value, or `NULL` if the result is not needed

   Level: advanced

   Note:
   Errors if the integer is negative since PETSc calls to hipBLAS and friends never need to cast negative integer inputs

.seealso: `PetscHipBLASInt`, `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscMPIIntCast()`, `PetscCIntCast()`, `PetscIntCast()`
@*/
static inline PetscErrorCode PetscHipBLASIntCast(MPIU_Count a, PetscHipBLASInt *b)
{
  PetscFunctionBegin;
  if (b) *b = 0; /* to prevent compilers erroneously suggesting uninitialized variable */
  PetscCheck(sizeof(MPIU_Count) <= sizeof(PetscHipBLASInt) || a <= (MPIU_Count)PETSC_HIPBLAS_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt64_FMT " is too big for hipBLAS, which is restricted to 32-bit integers.", (PetscInt64)a);
  PetscCheck(a >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Passing negative integer %" PetscInt64_FMT "to hipBLAS routine", (PetscInt64)a);
  if (b) *b = (PetscHipBLASInt)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscMPIIntCast - casts a `MPI_Count`, `PetscInt`, `PetscCount`, or `PetscInt64` to a `PetscMPIInt` (which is always 32-bits in size), generates an
   error if the `PetscMPIInt` is not large enough to hold the number.

   Not Collective; No Fortran Support

   Input Parameter:
.  a - the `PetscInt` value

   Output Parameter:
.  b - the resulting `PetscMPIInt` value, or `NULL` if the result is not needed

   Level: advanced

.seealso: [](stylePetscCount), `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscIntCast()`
@*/
static inline PetscErrorCode PetscMPIIntCast(MPIU_Count a, PetscMPIInt *b)
{
  PetscFunctionBegin;
  if (b) *b = 0; /* to prevent compilers erroneously suggesting uninitialized variable */
  PetscCheck(a <= (MPIU_Count)PETSC_MPI_INT_MAX && a >= (MPIU_Count)PETSC_MPI_INT_MIN, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt64_FMT " is too big for MPI buffer length. Maximum supported value is %d", (PetscInt64)a, PETSC_MPI_INT_MAX);
  if (b) *b = (PetscMPIInt)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscCIntCast - casts a `MPI_Count`, `PetscInt`, `PetscCount`, or `PetscInt64` to a `int`, generates an error if the `int` is not large enough to hold the number.

   Not Collective; No Fortran Support

   Input Parameter:
.  a - the `PetscInt` value

   Output Parameter:
.  b - the resulting `int` value, or `NULL` if the result is not needed

   Level: advanced

.seealso: [](stylePetscCount), `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscMPIIntCast()`, `PetscBLASIntCast()`, `PetscIntCast()`
@*/
static inline PetscErrorCode PetscCIntCast(MPIU_Count a, int *b)
{
  PetscFunctionBegin;
  if (b) *b = 0; /* to prevent compilers erroneously suggesting uninitialized variable */
  PetscCheck(a <= INT_MAX && a >= INT_MIN, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt64_FMT " is too big to be casted to an int. Maximum supported value is %d", (PetscInt64)a, INT_MAX);
  if (b) *b = (int)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PetscInt64Mult - Computes the product of two variables after casting them to `PetscInt64`.

   Not Collective; No Fortran Support

   Input Parameters:
+  a - the first variable
-  b - the second variable

   Level: advanced

.seealso: [](stylePetscCount), `PetscIntMultError()`, `PetscIntMultTruncate()`
M*/
#if defined(PETSC_USE_64BIT_INDICES)
  #define PetscInt64Mult(a, b) ((a) * (b))
#else
  #define PetscInt64Mult(a, b) (((PetscInt64)(a)) * ((PetscInt64)(b)))
#endif

/*@C
  PetscRealIntMultTruncate - Computes the product of a positive `PetscReal` and a positive
  `PetscInt` and truncates the value to slightly less than the maximal possible value.

  Not Collective; No Fortran Support

  Input Parameters:
+ a - The `PetscReal` value
- b - The `PetscInt` value

  Level: advanced

  Notes:
  Returns the result as a `PetscInt` value.

  Use `PetscInt64Mult()` to compute the product of two `PetscInt` as a `PetscInt64`.

  Use `PetscIntMultTruncate()` to compute the product of two positive `PetscInt` and truncate
  to fit a `PetscInt`.

  Use `PetscIntMultError()` to compute the product of two `PetscInt` if you wish to generate an
  error if the result will not fit in a `PetscInt`.

  Developer Notes:
  We currently assume that `PetscInt` addition can never overflow, this is obviously wrong but
  requires many more checks.

  This is used where we compute approximate sizes for workspace and need to insure the
  workspace is index-able.

.seealso: `PetscReal`, `PetscInt`, `PetscInt64Mult()`, `PetscIntMultError()`, `PetscIntSumError()`
@*/
static inline PetscInt PetscRealIntMultTruncate(PetscReal a, PetscInt b)
{
  PetscInt64 r = (PetscInt64)(a * (PetscReal)b);
  if (r > PETSC_INT_MAX - 100) r = PETSC_INT_MAX - 100;
#if defined(PETSC_USE_64BIT_INDICES)
  return r;
#else
  return (PetscInt)r;
#endif
}

/*@C
   PetscIntMultTruncate - Computes the product of two positive `PetscInt` and truncates the value to slightly less than the maximal possible value

   Not Collective; No Fortran Support

   Input Parameters:
+  a - the `PetscInt` value
-  b - the second value

   Returns:
   The result as a `PetscInt` value

   Level: advanced

   Notes:
   Use `PetscInt64Mult()` to compute the product of two `PetscInt` as a `PetscInt64`

   Use `PetscRealIntMultTruncate()` to compute the product of a `PetscReal` and a `PetscInt` and truncate to fit a `PetscInt`

   Use `PetscIntMultError()` to compute the product of two `PetscInt` if you wish to generate an error if the result will not fit in a `PetscInt`

   Developer Notes:
   We currently assume that `PetscInt` addition can never overflow, this is obviously wrong but requires many more checks.

   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscInt64Mult()`, `PetscIntMultError()`, `PetscIntSumError()`,
          `PetscIntSumTruncate()`
@*/
static inline PetscInt PetscIntMultTruncate(PetscInt a, PetscInt b)
{
  PetscInt64 r = PetscInt64Mult(a, b);
  if (r > PETSC_INT_MAX - 100) r = PETSC_INT_MAX - 100;
#if defined(PETSC_USE_64BIT_INDICES)
  return r;
#else
  return (PetscInt)r;
#endif
}

/*@C
   PetscIntSumTruncate - Computes the sum of two positive `PetscInt` and truncates the value to slightly less than the maximal possible value

   Not Collective; No Fortran Support

   Input Parameters:
+  a - the `PetscInt` value
-  b - the second value

   Returns:
   The result as a `PetscInt` value

   Level: advanced

   Notes:
   Use `PetscInt64Mult()` to compute the product of two `PetscInt` as a `PetscInt64`

   Use `PetscRealIntMultTruncate()` to compute the product of a `PetscReal` and a `PetscInt` and truncate to fit a `PetscInt`

   Use `PetscIntMultError()` to compute the product of two `PetscInt` if you wish to generate an error if the result will not fit in a `PetscInt`

   Developer Note:
   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscInt64Mult()`, `PetscIntMultError()`
@*/
static inline PetscInt PetscIntSumTruncate(PetscInt a, PetscInt b)
{
  PetscInt64 r = a;

  r += b;
  if (r > PETSC_INT_MAX - 100) r = PETSC_INT_MAX - 100;
#if defined(PETSC_USE_64BIT_INDICES)
  return r;
#else
  return (PetscInt)r;
#endif
}

/*@C
   PetscIntMultError - Computes the product of two positive `PetscInt` and generates an error with overflow.

   Not Collective; No Fortran Support

   Input Parameters:
+  a - the `PetscInt` value
-  b - the second value

   Output Parameter:
.  result - the result as a `PetscInt` value, or `NULL` if you do not want the result, you just want to check if it overflows

   Level: advanced

   Notes:
   Use `PetscInt64Mult()` to compute the product of two `PetscInt` and store in a `PetscInt64`

   Use `PetscIntMultTruncate()` to compute the product of two `PetscInt` and truncate it to fit in a `PetscInt`

   Developer Note:
   In most places in the source code we currently assume that `PetscInt` addition does not overflow, this is obviously wrong but requires many more checks.
   `PetscIntSumError()` can be used to check for this situation.

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscInt64Mult()`, `PetscIntSumError()`
@*/
static inline PetscErrorCode PetscIntMultError(PetscInt a, PetscInt b, PetscInt *result)
{
  PetscInt64 r = PetscInt64Mult(a, b);

  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  if (result) *result = r;
#else
  if (result) *result = (PetscInt)r;
#endif
  if (!PetscDefined(USE_64BIT_INDICES)) {
    PetscCheck(r <= PETSC_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "Product of two integers %" PetscInt_FMT " %" PetscInt_FMT " overflow, either you have an invalidly large integer error in your code or you must ./configure PETSc with --with-64-bit-indices for the case you are running", a, b);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C

   PetscIntSumError - Computes the sum of two positive `PetscInt` and generates an error with overflow.

   Not Collective; No Fortran Support

   Input Parameters:
+  a - the `PetscInt` value
-  b - the second value

   Output Parameter:
.  c - the result as a `PetscInt` value,  or `NULL` if you do not want the result, you just want to check if it overflows

   Level: advanced

   Notes:
   Use `PetscInt64Mult()` to compute the product of two 32-bit `PetscInt` and store in a `PetscInt64`

   Use `PetscIntMultTruncate()` to compute the product of two `PetscInt` and truncate it to fit in a `PetscInt`

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`, `PetscInt64Mult()`, `PetscIntMultError()`
@*/
static inline PetscErrorCode PetscIntSumError(PetscInt a, PetscInt b, PetscInt *result)
{
  PetscInt64 r = a;

  PetscFunctionBegin;
  r += b;
#if defined(PETSC_USE_64BIT_INDICES)
  if (result) *result = r;
#else
  if (result) *result = (PetscInt)r;
#endif
  if (!PetscDefined(USE_64BIT_INDICES)) {
    PetscCheck(r <= PETSC_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "Sum of two integers %" PetscInt_FMT " %" PetscInt_FMT " overflow, either you have an invalidly large integer error in your code or you must ./configure PETSc with --with-64-bit-indices for the case you are running", a, b);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     The IBM include files define hz, here we hide it so that it may be used as a regular user variable.
*/
#if defined(hz)
  #undef hz
#endif

#if defined(PETSC_HAVE_SYS_TYPES_H)
  #include <sys/types.h>
#endif

/*MC

    PETSC_VERSION - This manual page provides information about how PETSc documents and uses its version information. This information is available to both C/C++
                    and Fortran compilers when `petscsys.h` is included.

    The current PETSc version and the API for accessing it are defined in <A HREF="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petscversion.h.html">include/petscversion.html</A>

    The complete version number is given as the triple  PETSC_VERSION_MAJOR.PETSC_VERSION_MINOR.PETSC_VERSION_SUBMINOR (in short hand x.y.z)

    A change in the minor version number (y) indicates possible/likely changes in the PETSc API. Note this is different than with the semantic versioning convention
    where only a change in the major version number (x) indicates a change in the API.

    A subminor greater than zero indicates a patch release. Version x.y.z maintains source and binary compatibility with version x.y.w for all z and w

    Use the macros PETSC_VERSION_EQ(x,y,z), PETSC_VERSION_LT(x,y,z), PETSC_VERSION_LE(x,y,z), PETSC_VERSION_GT(x,y,z),
    PETSC_VERSION_GE(x,y,z) to determine if the current version is equal to, less than, less than or equal to, greater than or greater than or equal to a given
    version number (x.y.z).

    `PETSC_RELEASE_DATE` is the date the x.y version was released (i.e. the version before any patch releases)

    `PETSC_VERSION_DATE` is the date the x.y.z version was released

    `PETSC_VERSION_GIT` is the last git commit to the repository given in the form vx.y.z-wwwww

    `PETSC_VERSION_DATE_GIT` is the date of the last git commit to the repository

    `PETSC_VERSION_()` is deprecated and will eventually be removed.

    Level: intermediate
M*/

PETSC_EXTERN PetscErrorCode PetscGetArchType(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetHostName(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetUserName(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetProgramName(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscSetProgramName(const char[]);
PETSC_EXTERN PetscErrorCode PetscGetDate(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetVersion(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetVersionNumber(PetscInt *, PetscInt *, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode PetscSortedInt(PetscCount, const PetscInt[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSortedInt64(PetscCount, const PetscInt64[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSortedMPIInt(PetscCount, const PetscMPIInt[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSortedReal(PetscCount, const PetscReal[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSortInt(PetscCount, PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortInt64(PetscCount, PetscInt64[]);
PETSC_EXTERN PetscErrorCode PetscSortCount(PetscCount, PetscCount[]);
PETSC_EXTERN PetscErrorCode PetscSortReverseInt(PetscCount, PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortedRemoveDupsInt(PetscInt *, PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortedCheckDupsInt(PetscCount, const PetscInt[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSortedCheckDupsCount(PetscCount, const PetscCount[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsInt(PetscInt *, PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscCheckDupsInt(PetscInt, const PetscInt[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscFindInt(PetscInt, PetscCount, const PetscInt[], PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFindMPIInt(PetscMPIInt, PetscCount, const PetscMPIInt[], PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFindCount(PetscCount, PetscCount, const PetscCount[], PetscCount *);
PETSC_EXTERN PetscErrorCode PetscSortIntWithPermutation(PetscInt, const PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortStrWithPermutation(PetscInt, const char *[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithArray(PetscCount, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithCountArray(PetscCount, PetscInt[], PetscCount[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithMPIIntArray(PetscCount, PetscInt[], PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithArrayPair(PetscCount, PetscInt[], PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithIntCountArrayPair(PetscCount, PetscInt[], PetscInt[], PetscCount[]);
PETSC_EXTERN PetscErrorCode PetscSortMPIInt(PetscCount, PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsMPIInt(PetscInt *, PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortMPIIntWithArray(PetscCount, PetscMPIInt[], PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortMPIIntWithIntArray(PetscCount, PetscMPIInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithScalarArray(PetscCount, PetscInt[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithDataArray(PetscCount, PetscInt[], void *, size_t, void *);
PETSC_EXTERN PetscErrorCode PetscSortReal(PetscCount, PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscSortRealWithArrayInt(PetscCount, PetscReal[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRealWithPermutation(PetscInt, const PetscReal[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsReal(PetscInt *, PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscFindReal(PetscReal, PetscCount, const PetscReal[], PetscReal, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSortSplit(PetscInt, PetscInt, PetscScalar[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortSplitReal(PetscInt, PetscInt, PetscReal[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscProcessTree(PetscInt, const PetscBool[], const PetscInt[], PetscInt *, PetscInt *[], PetscInt *[], PetscInt *[], PetscInt *[]);
PETSC_EXTERN PetscErrorCode PetscMergeIntArrayPair(PetscInt, const PetscInt[], const PetscInt[], PetscInt, const PetscInt[], const PetscInt[], PetscInt *, PetscInt *[], PetscInt *[]);
PETSC_EXTERN PetscErrorCode PetscMergeIntArray(PetscInt, const PetscInt[], PetscInt, const PetscInt[], PetscInt *, PetscInt *[]);
PETSC_EXTERN PetscErrorCode PetscMergeMPIIntArray(PetscInt, const PetscMPIInt[], PetscInt, const PetscMPIInt[], PetscInt *, PetscMPIInt *[]);
PETSC_EXTERN PetscErrorCode PetscParallelSortedInt(MPI_Comm, PetscInt, const PetscInt[], PetscBool *);

PETSC_EXTERN PetscErrorCode PetscTimSort(PetscInt, void *, size_t, int (*)(const void *, const void *, void *), void *);
PETSC_EXTERN PetscErrorCode PetscIntSortSemiOrdered(PetscInt, PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscMPIIntSortSemiOrdered(PetscInt, PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscRealSortSemiOrdered(PetscInt, PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscTimSortWithArray(PetscInt, void *, size_t, void *, size_t, int (*)(const void *, const void *, void *), void *);
PETSC_EXTERN PetscErrorCode PetscIntSortSemiOrderedWithArray(PetscInt, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscMPIIntSortSemiOrderedWithArray(PetscInt, PetscMPIInt[], PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscRealSortSemiOrderedWithArrayInt(PetscInt, PetscReal[], PetscInt[]);

PETSC_EXTERN PetscErrorCode PetscSetDisplay(void);
PETSC_EXTERN PetscErrorCode PetscGetDisplay(char[], size_t);

/*J
    PetscRandomType - String with the name of a PETSc randomizer

   Level: beginner

   Note:
   To use `PETSCSPRNG` or `PETSCRANDOM123` you must have ./configure PETSc
   with the option `--download-sprng` or `--download-random123`. We recommend the default provided with PETSc.

.seealso: `PetscRandomSetType()`, `PetscRandom`, `PetscRandomCreate()`
J*/
typedef const char *PetscRandomType;
#define PETSCRAND      "rand"
#define PETSCRAND48    "rand48"
#define PETSCSPRNG     "sprng"
#define PETSCRANDER48  "rander48"
#define PETSCRANDOM123 "random123"
#define PETSCCURAND    "curand"

/* Logging support */
PETSC_EXTERN PetscClassId PETSC_RANDOM_CLASSID;

PETSC_EXTERN PetscErrorCode PetscRandomInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscRandomFinalizePackage(void);

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFunctionList PetscRandomList;

PETSC_EXTERN PetscErrorCode PetscRandomRegister(const char[], PetscErrorCode (*)(PetscRandom));
PETSC_EXTERN PetscErrorCode PetscRandomSetType(PetscRandom, PetscRandomType);
PETSC_EXTERN PetscErrorCode PetscRandomSetOptionsPrefix(PetscRandom, const char[]);
PETSC_EXTERN PetscErrorCode PetscRandomSetFromOptions(PetscRandom);
PETSC_EXTERN PetscErrorCode PetscRandomGetType(PetscRandom, PetscRandomType *);
PETSC_EXTERN PetscErrorCode PetscRandomViewFromOptions(PetscRandom, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscRandomView(PetscRandom, PetscViewer);

PETSC_EXTERN PetscErrorCode PetscRandomCreate(MPI_Comm, PetscRandom *);
PETSC_EXTERN PetscErrorCode PetscRandomGetValue(PetscRandom, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscRandomGetValueReal(PetscRandom, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscRandomGetValues(PetscRandom, PetscInt, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscRandomGetValuesReal(PetscRandom, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscRandomGetInterval(PetscRandom, PetscScalar *, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscRandomSetInterval(PetscRandom, PetscScalar, PetscScalar);
PETSC_EXTERN PetscErrorCode PetscRandomSetSeed(PetscRandom, PetscInt64);
PETSC_EXTERN PetscErrorCode PetscRandomGetSeed(PetscRandom, PetscInt64 *);
PETSC_EXTERN PetscErrorCode PetscRandomSeed(PetscRandom);
PETSC_EXTERN PetscErrorCode PetscRandomDestroy(PetscRandom *);

PETSC_EXTERN PetscErrorCode PetscGetFullPath(const char[], char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetRelativePath(const char[], char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetWorkingDirectory(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetRealPath(const char[], char[]);
PETSC_EXTERN PetscErrorCode PetscGetHomeDirectory(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscTestFile(const char[], char, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscTestDirectory(const char[], char, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscMkdir(const char[]);
PETSC_EXTERN PetscErrorCode PetscMkdtemp(char[]);
PETSC_EXTERN PetscErrorCode PetscRMTree(const char[]);

/*MC
   PetscBinaryBigEndian - indicates if values in memory are stored with big endian format

   Synopsis:
   #include <petscsys.h>
   PetscBool PetscBinaryBigEndian(void);

   No Fortran Support

   Level: developer

.seealso: `PetscInitialize()`, `PetscFinalize()`, `PetscInitializeCalled`
M*/
static inline PetscBool PetscBinaryBigEndian(void)
{
  long _petsc_v = 1;
  return ((char *)&_petsc_v)[0] ? PETSC_FALSE : PETSC_TRUE;
}

PETSC_EXTERN PetscErrorCode PetscBinaryRead(int, void *, PetscCount, PetscInt *, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedRead(MPI_Comm, int, void *, PetscInt, PetscInt *, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinaryWrite(int, const void *, PetscCount, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedWrite(MPI_Comm, int, const void *, PetscInt, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinaryOpen(const char[], PetscFileMode, int *);
PETSC_EXTERN PetscErrorCode PetscBinaryClose(int);
PETSC_EXTERN PetscErrorCode PetscSharedTmp(MPI_Comm, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSharedWorkingDirectory(MPI_Comm, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscGetTmp(MPI_Comm, char[], size_t);
PETSC_EXTERN PetscErrorCode PetscFileRetrieve(MPI_Comm, const char[], char[], size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscLs(MPI_Comm, const char[], char[], size_t, PetscBool *);
#if defined(PETSC_USE_SOCKET_VIEWER)
PETSC_EXTERN PetscErrorCode PetscOpenSocket(const char[], int, int *);
#endif

PETSC_EXTERN PetscErrorCode PetscBinarySeek(int, off_t, PetscBinarySeekType, off_t *);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedSeek(MPI_Comm, int, off_t, PetscBinarySeekType, off_t *);
PETSC_EXTERN PetscErrorCode PetscByteSwap(void *, PetscDataType, PetscCount);

PETSC_EXTERN PetscErrorCode PetscSetDebugTerminal(const char[]);
PETSC_EXTERN PetscErrorCode PetscSetDebugger(const char[], PetscBool);
PETSC_EXTERN PetscErrorCode PetscSetDefaultDebugger(void);
PETSC_EXTERN PetscErrorCode PetscSetDebuggerFromString(const char *);
PETSC_EXTERN PetscErrorCode PetscAttachDebugger(void);
PETSC_EXTERN PetscErrorCode PetscStopForDebugger(void);
PETSC_EXTERN PetscErrorCode PetscWaitOnError(void);

PETSC_EXTERN PetscErrorCode PetscGatherNumberOfMessages(MPI_Comm, const PetscMPIInt[], const PetscMPIInt[], PetscMPIInt *);
PETSC_EXTERN PetscErrorCode PetscGatherMessageLengths(MPI_Comm, PetscMPIInt, PetscMPIInt, const PetscMPIInt[], PetscMPIInt *[], PetscMPIInt *[]);
PETSC_EXTERN PetscErrorCode PetscGatherMessageLengths2(MPI_Comm, PetscMPIInt, PetscMPIInt, const PetscMPIInt[], const PetscMPIInt[], PetscMPIInt *[], PetscMPIInt *[], PetscMPIInt *[]);
PETSC_EXTERN PetscErrorCode PetscPostIrecvInt(MPI_Comm, PetscMPIInt, PetscMPIInt, const PetscMPIInt[], const PetscMPIInt[], PetscInt ***, MPI_Request **);
PETSC_EXTERN PetscErrorCode PetscPostIrecvScalar(MPI_Comm, PetscMPIInt, PetscMPIInt, const PetscMPIInt[], const PetscMPIInt[], PetscScalar ***, MPI_Request **);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSided(MPI_Comm, PetscMPIInt, MPI_Datatype, PetscMPIInt, const PetscMPIInt[], const void *, PetscMPIInt *, PetscMPIInt *[], void *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(6, 3);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedF(MPI_Comm, PetscMPIInt, MPI_Datatype, PetscMPIInt, const PetscMPIInt[], const void *, PetscMPIInt *, PetscMPIInt **, void *, PetscMPIInt, PetscErrorCode (*send)(MPI_Comm, const PetscMPIInt[], PetscMPIInt, PetscMPIInt, void *, MPI_Request[], void *), PetscErrorCode (*recv)(MPI_Comm, const PetscMPIInt[], PetscMPIInt, void *, MPI_Request[], void *), void *ctx) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(6, 3);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedFReq(MPI_Comm, PetscMPIInt, MPI_Datatype, PetscMPIInt, const PetscMPIInt[], const void *, PetscMPIInt *, PetscMPIInt **, void *, PetscMPIInt, MPI_Request **, MPI_Request **, PetscErrorCode (*send)(MPI_Comm, const PetscMPIInt[], PetscMPIInt, PetscMPIInt, void *, MPI_Request[], void *), PetscErrorCode (*recv)(MPI_Comm, const PetscMPIInt[], PetscMPIInt, void *, MPI_Request[], void *), void *ctx) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(6, 3);

PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedSetType(MPI_Comm, PetscBuildTwoSidedType);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedGetType(MPI_Comm, PetscBuildTwoSidedType *);

PETSC_DEPRECATED_FUNCTION(3, 24, 0, "PetscSSEIsEnabled()", ) static inline PetscErrorCode PetscSSEIsEnabled(PETSC_UNUSED MPI_Comm comm, PetscBool *lflag, PetscBool *gflag)
{
  if (lflag) *lflag = PETSC_FALSE;
  if (gflag) *gflag = PETSC_FALSE;
  return PETSC_SUCCESS;
}

PETSC_EXTERN MPI_Comm PetscObjectComm(PetscObject);

struct _n_PetscSubcomm {
  MPI_Comm         parent;    /* parent communicator */
  MPI_Comm         dupparent; /* duplicate parent communicator, under which the processors of this subcomm have contiguous rank */
  MPI_Comm         child;     /* the sub-communicator */
  PetscMPIInt      n;         /* num of subcommunicators under the parent communicator */
  PetscMPIInt      color;     /* color of processors belong to this communicator */
  PetscMPIInt     *subsize;   /* size of subcommunicator[color] */
  PetscSubcommType type;
  char            *subcommprefix;
};

static inline MPI_Comm PetscSubcommParent(PetscSubcomm scomm)
{
  return scomm->parent;
}
static inline MPI_Comm PetscSubcommChild(PetscSubcomm scomm)
{
  return scomm->child;
}
static inline MPI_Comm PetscSubcommContiguousParent(PetscSubcomm scomm)
{
  return scomm->dupparent;
}
PETSC_EXTERN PetscErrorCode PetscSubcommCreate(MPI_Comm, PetscSubcomm *);
PETSC_EXTERN PetscErrorCode PetscSubcommDestroy(PetscSubcomm *);
PETSC_EXTERN PetscErrorCode PetscSubcommSetNumber(PetscSubcomm, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSubcommSetType(PetscSubcomm, PetscSubcommType);
PETSC_EXTERN PetscErrorCode PetscSubcommSetTypeGeneral(PetscSubcomm, PetscMPIInt, PetscMPIInt);
PETSC_EXTERN PetscErrorCode PetscSubcommView(PetscSubcomm, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSubcommSetFromOptions(PetscSubcomm);
PETSC_EXTERN PetscErrorCode PetscSubcommSetOptionsPrefix(PetscSubcomm, const char[]);
PETSC_EXTERN PetscErrorCode PetscSubcommGetParent(PetscSubcomm, MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscSubcommGetContiguousParent(PetscSubcomm, MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscSubcommGetChild(PetscSubcomm, MPI_Comm *);

PETSC_EXTERN PetscErrorCode PetscHeapCreate(PetscInt, PetscHeap *);
PETSC_EXTERN PetscErrorCode PetscHeapAdd(PetscHeap, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscHeapPop(PetscHeap, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscHeapPeek(PetscHeap, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscHeapStash(PetscHeap, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscHeapUnstash(PetscHeap);
PETSC_EXTERN PetscErrorCode PetscHeapDestroy(PetscHeap *);
PETSC_EXTERN PetscErrorCode PetscHeapView(PetscHeap, PetscViewer);

PETSC_EXTERN PetscErrorCode PetscProcessPlacementView(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscShmCommGet(MPI_Comm, PetscShmComm *);
PETSC_EXTERN PetscErrorCode PetscShmCommGlobalToLocal(PetscShmComm, PetscMPIInt, PetscMPIInt *);
PETSC_EXTERN PetscErrorCode PetscShmCommLocalToGlobal(PetscShmComm, PetscMPIInt, PetscMPIInt *);
PETSC_EXTERN PetscErrorCode PetscShmCommGetMpiShmComm(PetscShmComm, MPI_Comm *);

/* routines to better support OpenMP multithreading needs of some PETSc third party libraries */
PETSC_EXTERN PetscErrorCode PetscOmpCtrlCreate(MPI_Comm, PetscInt, PetscOmpCtrl *);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlGetOmpComms(PetscOmpCtrl, MPI_Comm *, MPI_Comm *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlDestroy(PetscOmpCtrl *);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlBarrier(PetscOmpCtrl);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlOmpRegionOnMasterBegin(PetscOmpCtrl);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlOmpRegionOnMasterEnd(PetscOmpCtrl);

PETSC_EXTERN PetscErrorCode PetscSegBufferCreate(size_t, PetscCount, PetscSegBuffer *);
PETSC_EXTERN PetscErrorCode PetscSegBufferDestroy(PetscSegBuffer *);
PETSC_EXTERN PetscErrorCode PetscSegBufferGet(PetscSegBuffer, PetscCount, void *);
PETSC_EXTERN PetscErrorCode PetscSegBufferExtractAlloc(PetscSegBuffer, void *);
PETSC_EXTERN PetscErrorCode PetscSegBufferExtractTo(PetscSegBuffer, void *);
PETSC_EXTERN PetscErrorCode PetscSegBufferExtractInPlace(PetscSegBuffer, void *);
PETSC_EXTERN PetscErrorCode PetscSegBufferGetSize(PetscSegBuffer, PetscCount *);
PETSC_EXTERN PetscErrorCode PetscSegBufferUnuse(PetscSegBuffer, PetscCount);

/*MC
  PetscSegBufferGetInts - access an array of `PetscInt` from a `PetscSegBuffer`

  Synopsis:
  #include <petscsys.h>
  PetscErrorCode PetscSegBufferGetInts(PetscSegBuffer seg, size_t count, PetscInt *PETSC_RESTRICT *slot);

  No Fortran Support

  Input Parameters:
+ seg   - `PetscSegBuffer` buffer
- count - number of entries needed

  Output Parameter:
. buf - address of new buffer for contiguous data

  Level: intermediate

  Developer Note:
  Type-safe wrapper to encourage use of PETSC_RESTRICT. Does not use PetscFunctionBegin because the error handling
  prevents the compiler from completely erasing the stub. This is called in inner loops so it has to be as fast as
  possible.

.seealso: `PetscSegBuffer`, `PetscSegBufferGet()`, `PetscInitialize()`, `PetscFinalize()`, `PetscInitializeCalled`
M*/
static inline PetscErrorCode PetscSegBufferGetInts(PetscSegBuffer seg, PetscCount count, PetscInt *PETSC_RESTRICT *slot)
{
  return PetscSegBufferGet(seg, count, (void **)slot);
}

extern PetscOptionsHelpPrinted PetscOptionsHelpPrintedSingleton;
PETSC_EXTERN PetscErrorCode    PetscOptionsHelpPrintedDestroy(PetscOptionsHelpPrinted *);
PETSC_EXTERN PetscErrorCode    PetscOptionsHelpPrintedCreate(PetscOptionsHelpPrinted *);
PETSC_EXTERN PetscErrorCode    PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrinted, const char *, const char *, PetscBool *);

#include <stdarg.h>
PETSC_EXTERN PetscErrorCode PetscVSNPrintf(char *, size_t, const char[], size_t *, va_list);
PETSC_EXTERN PetscErrorCode (*PetscVFPrintf)(FILE *, const char[], va_list);

PETSC_EXTERN PetscSegBuffer PetscCitationsList;

/*@
     PetscCitationsRegister - Register a bibtex item to obtain credit for an implemented algorithm used in the code.

     Not Collective; No Fortran Support

     Input Parameters:
+    cite - the bibtex item, formatted to displayed on multiple lines nicely
-    set - a boolean variable initially set to `PETSC_FALSE`; this is used to insure only a single registration of the citation

     Options Database Key:
.     -citations [filename]   - print out the bibtex entries for the given computation

     Level: intermediate
@*/
static inline PetscErrorCode PetscCitationsRegister(const char cit[], PetscBool *set)
{
  size_t len;
  char  *vstring;

  PetscFunctionBegin;
  if (set && *set) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscStrlen(cit, &len));
  PetscCall(PetscSegBufferGet(PetscCitationsList, (PetscCount)len, &vstring));
  PetscCall(PetscArraycpy(vstring, cit, len));
  if (set) *set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscGoogleDriveAuthorize(MPI_Comm, char[], char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveRefresh(MPI_Comm, const char[], char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveUpload(MPI_Comm, const char[], const char[]);

PETSC_EXTERN PetscErrorCode PetscBoxAuthorize(MPI_Comm, char[], char[], size_t);
PETSC_EXTERN PetscErrorCode PetscBoxRefresh(MPI_Comm, const char[], char[], char[], size_t);
PETSC_EXTERN PetscErrorCode PetscBoxUpload(MPI_Comm, const char[], const char[]);

PETSC_EXTERN PetscErrorCode PetscGlobusGetTransfers(MPI_Comm, const char[], char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGlobusAuthorize(MPI_Comm, char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGlobusUpload(MPI_Comm, const char[], const char[]);

PETSC_EXTERN PetscErrorCode PetscPullJSONValue(const char[], const char[], char[], size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscPushJSONValue(char[], const char[], const char[], size_t);

#if !defined(PETSC_HAVE_MPI_LARGE_COUNT)
  /*
   Cast PetscCount <a> to PetscMPIInt <b>, where <a> is likely used for the 'count' argument in MPI routines.
   It is similar to PetscMPIIntCast() except that here it returns an MPI error code.
*/
  #define PetscMPIIntCast_Internal(a, b) \
    do { \
      *b = 0; \
      if (PetscUnlikely(a > (MPIU_Count)PETSC_MPI_INT_MAX)) return MPI_ERR_COUNT; \
      *b = (PetscMPIInt)a; \
    } while (0)

static inline PetscMPIInt MPIU_Get_count(MPI_Status *status, MPI_Datatype dtype, PetscCount *count)
{
  PetscMPIInt count2, err;

  *count = 0; /* to prevent incorrect warnings of uninitialized variables */
  err    = MPI_Get_count(status, dtype, &count2);
  *count = count2;
  return err;
}

static inline PetscMPIInt MPIU_Send(const void *buf, MPIU_Count count, MPI_Datatype dtype, PetscMPIInt dest, PetscMPIInt tag, MPI_Comm comm)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Send((void *)buf, count2, dtype, dest, tag, comm);
  return err;
}

static inline PetscMPIInt MPIU_Send_init(const void *buf, MPIU_Count count, MPI_Datatype dtype, PetscMPIInt dest, PetscMPIInt tag, MPI_Comm comm, MPI_Request *request)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Send_init((void *)buf, count2, dtype, dest, tag, comm, request);
  return err;
}

static inline PetscMPIInt MPIU_Isend(const void *buf, MPIU_Count count, MPI_Datatype dtype, PetscMPIInt dest, PetscMPIInt tag, MPI_Comm comm, MPI_Request *request)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Isend((void *)buf, count2, dtype, dest, tag, comm, request);
  return err;
}

static inline PetscMPIInt MPIU_Recv(const void *buf, MPIU_Count count, MPI_Datatype dtype, PetscMPIInt source, PetscMPIInt tag, MPI_Comm comm, MPI_Status *status)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Recv((void *)buf, count2, dtype, source, tag, comm, status);
  return err;
}

static inline PetscMPIInt MPIU_Recv_init(const void *buf, MPIU_Count count, MPI_Datatype dtype, PetscMPIInt source, PetscMPIInt tag, MPI_Comm comm, MPI_Request *request)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Recv_init((void *)buf, count2, dtype, source, tag, comm, request);
  return err;
}

static inline PetscMPIInt MPIU_Irecv(const void *buf, MPIU_Count count, MPI_Datatype dtype, PetscMPIInt source, PetscMPIInt tag, MPI_Comm comm, MPI_Request *request)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Irecv((void *)buf, count2, dtype, source, tag, comm, request);
  return err;
}

static inline PetscMPIInt MPIU_Reduce(const void *inbuf, void *outbuf, MPIU_Count count, MPI_Datatype dtype, MPI_Op op, PetscMPIInt root, MPI_Comm comm)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Reduce((void *)inbuf, outbuf, count2, dtype, op, root, comm);
  return err;
}

  #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
static inline PetscMPIInt MPIU_Reduce_local(const void *inbuf, void *inoutbuf, MPIU_Count count, MPI_Datatype dtype, MPI_Op op)
{
  PetscMPIInt count2, err;

  PetscMPIIntCast_Internal(count, &count2);
  err = MPI_Reduce_local((void *)inbuf, inoutbuf, count2, dtype, op);
  return err;
}
  #endif

  #if !defined(PETSC_USE_64BIT_INDICES)
    #define MPIU_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm) MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm)
    #define MPIU_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm)  MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm)
  #else
    #define MPIU_Scatterv(sendbuf, sendcount, displs, sendtype, recvbuf, recvcount, recvtype, root, comm) \
      ((void)PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_SUP, PETSC_ERROR_INITIAL, "Must have MPI 4 support for MPI_Scatterv_c() for this functionality, upgrade your MPI"), MPI_ERR_COUNT)
    #define MPIU_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm) \
      ((void)PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_SUP, PETSC_ERROR_INITIAL, "Must have MPI 4 support for MPI_Scatterv_c() for this functionality, upgrade your MPI"), MPI_ERR_COUNT)
  #endif

#else

  /* on 32 bit systems MPI_Count maybe 64-bit while PetscCount is 32-bit */
  #define PetscCountCast_Internal(a, b) \
    do { \
      *b = 0; \
      if (PetscUnlikely(a > (MPI_Count)PETSC_COUNT_MAX)) return MPI_ERR_COUNT; \
      *b = (PetscMPIInt)a; \
    } while (0)

static inline PetscMPIInt MPIU_Get_count(MPI_Status *status, MPI_Datatype dtype, PetscCount *count)
{
  MPI_Count   count2;
  PetscMPIInt err;

  *count = 0; /* to prevent incorrect warnings of uninitialized variables */
  err    = MPI_Get_count_c(status, dtype, &count2);
  if (err) return err;
  PetscCountCast_Internal(count2, count);
  return MPI_SUCCESS;
}

  #define MPIU_Reduce(inbuf, outbuf, count, dtype, op, root, comm)      MPI_Reduce_c(inbuf, outbuf, (MPI_Count)(count), dtype, op, root, comm)
  #define MPIU_Send(buf, count, dtype, dest, tag, comm)                 MPI_Send_c(buf, (MPI_Count)(count), dtype, dest, tag, comm)
  #define MPIU_Send_init(buf, count, dtype, dest, tag, comm, request)   MPI_Send_init_c(buf, (MPI_Count)(count), dtype, dest, tag, comm, request)
  #define MPIU_Isend(buf, count, dtype, dest, tag, comm, request)       MPI_Isend_c(buf, (MPI_Count)(count), dtype, dest, tag, comm, request)
  #define MPIU_Recv(buf, count, dtype, source, tag, comm, status)       MPI_Recv_c(buf, (MPI_Count)(count), dtype, source, tag, comm, status)
  #define MPIU_Recv_init(buf, count, dtype, source, tag, comm, request) MPI_Recv_init_c(buf, (MPI_Count)(count), dtype, source, tag, comm, request)
  #define MPIU_Irecv(buf, count, dtype, source, tag, comm, request)     MPI_Irecv_c(buf, (MPI_Count)(count), dtype, source, tag, comm, request)
  #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
    #define MPIU_Reduce_local(inbuf, inoutbuf, count, dtype, op) MPI_Reduce_local_c(inbuf, inoutbuf, (MPI_Count)(count), dtype, op)
  #endif

/*MC
  MPIU_Scatterv - A replacement for `MPI_Scatterv()` that can be called with `PetscInt` types when PETSc is built for either 32-bit indices or 64-bit indices.

  Synopsis:
  #include <petscsys.h>
  PetscMPIInt MPIU_Scatterv(const void *sendbuf, const PetscInt sendcounts[], const PetscInt displs[], MPI_Datatype sendtype, void *recvbuf, PetscInt recvcount, MPI_Datatype recvtype, PetscMPIInt root, MPI_Comm comm)

  Collective

  Input Parameters:
+ sendbuf    - address of send buffer
. sendcounts - non-negative `PetscInt` array (of length `comm` group size) specifying the number of elements to send to each MPI process
. displs     - `PetscInt` array (of length `comm` group size). Entry i specifies the displacement (relative to `sendbuf`) from which to take the outgoing data to process i
. sendtype   - data type of `sendbuf` elements
. recvcount  - number of elements in `recvbuf` (non-negative integer)
. recvtype   - data type of `recvbuf` elements
. root       - Rank of the MPI root process, which will dispatch the data to scatter
- comm       - `MPI_Comm` communicator

  Output Parameter:
. recvbuf - the resulting scattered values on this MPI process

  Level: developer

  Notes:
  Should be wrapped with `PetscCallMPI()` for error checking

  This is different than most of the `MPIU_` wrappers in that all the count arguments are in `PetscInt`

.seealso: [](stylePetscCount), `MPI_Allreduce()`, `MPIU_Gatherv()`
M*/

  #if !defined(PETSC_USE_64BIT_INDICES)
    #define MPIU_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm) MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm)
    #define MPIU_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm)  MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm)
  #else
    #define MPIU_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm) MPI_Scatterv_c(sendbuf, (const MPI_Count *)(sendcounts), (const MPI_Aint *)(displs), sendtype, recvbuf, recvcount, recvtype, root, comm)
    #define MPIU_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm)  MPI_Gatherv_c(sendbuf, sendcount, sendtype, recvbuf, (const MPI_Count *)(recvcounts), (const MPI_Aint *)(displs), recvtype, root, comm)
  #endif

#endif

PETSC_EXTERN PetscMPIInt    MPIU_Allreduce_Private(const void *, void *, MPIU_Count, MPI_Datatype, MPI_Op, MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscCheckAllreduceSameLineAndCount_Private(MPI_Comm, const char *, PetscMPIInt, PetscMPIInt);

#if defined(PETSC_USE_DEBUG)
static inline unsigned int PetscStrHash(const char *str)
{
  unsigned int c, hash = 5381;

  while ((c = (unsigned int)*str++)) hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
  return hash;
}
#endif

/*MC
  MPIU_Allreduce - A replacement for `MPI_Allreduce()` that (1) performs single-count `MPIU_INT` operations in `PetscInt64` to detect
                   integer overflows and (2) tries to determine if the call from all the MPI ranks occur in the
                   same place in the PETSc code. This helps to detect bugs where different MPI ranks follow different code paths
                   resulting in inconsistent and incorrect calls to `MPI_Allreduce()`.

  Synopsis:
  #include <petscsys.h>
  PetscMPIInt MPIU_Allreduce(void *indata,void *outdata,PetscCount count,MPI_Datatype dtype, MPI_Op op, MPI_Comm comm);

  Collective

  Input Parameters:
+ a     - pointer to the input data to be reduced
. count - the number of MPI data items in `a` and `b`
. dtype - the MPI datatype, for example `MPI_INT`
. op    - the MPI operation, for example `MPI_SUM`
- comm   - the MPI communicator on which the operation occurs

  Output Parameter:
. b - the reduced values

  Level: developer

  Note:
  Should be wrapped with `PetscCallMPI()` for error checking

.seealso: [](stylePetscCount), `MPI_Allreduce()`
M*/
#if defined(PETSC_USE_DEBUG)
  #define MPIU_Allreduce(a, b, count, dtype, op, comm) \
    PetscMacroReturnStandard( \
    PetscCall(PetscCheckAllreduceSameLineAndCount_Private((comm), __FILE__, (PetscMPIInt)__LINE__, (PetscMPIInt)(count))); \
    PetscCallMPI(MPIU_Allreduce_Private((a), (b), (count), (dtype), (op), (comm)));)
#else
  #define MPIU_Allreduce(a, b, count, dtype, op, comm) MPIU_Allreduce_Private((a), (b), (count), (dtype), (op), (comm))
#endif

/* this is a vile hack */
#if defined(PETSC_HAVE_NECMPI)
  #if !defined(PETSC_NECMPI_VERSION_MAJOR) || !defined(PETSC_NECMPI_VERSION_MINOR) || PETSC_NECMPI_VERSION_MAJOR < 2 || (PETSC_NECMPI_VERSION_MAJOR == 2 && PETSC_NECMPI_VERSION_MINOR < 18)
    #define MPI_Type_free(a) (*(a) = MPI_DATATYPE_NULL, 0);
  #endif
#endif

/*
    List of external packages and queries on it
*/
PETSC_EXTERN PetscErrorCode PetscHasExternalPackage(const char[], PetscBool *);

/* this cannot go here because it may be in a different shared library */
PETSC_EXTERN PetscErrorCode PCMPIServerBegin(void);
PETSC_EXTERN PetscErrorCode PCMPIServerEnd(void);
PETSC_EXTERN PetscBool      PCMPIServerActive;
PETSC_EXTERN PetscBool      PCMPIServerInSolve;
PETSC_EXTERN PetscBool      PCMPIServerUseShmget;
PETSC_EXTERN PetscErrorCode PetscShmgetAllocateArray(size_t, size_t, void **);
PETSC_EXTERN PetscErrorCode PetscShmgetDeallocateArray(void **);
PETSC_EXTERN PetscErrorCode PetscShmgetMapAddresses(MPI_Comm, PetscInt, const void **, void **);
PETSC_EXTERN PetscErrorCode PetscShmgetUnmapAddresses(PetscInt, void **);
PETSC_EXTERN PetscErrorCode PetscShmgetAddressesFinalize(void);

typedef struct {
  PetscInt n;
  void    *addr[3];
} PCMPIServerAddresses;
PETSC_EXTERN PetscCtxDestroyFn PCMPIServerAddressesDestroy;

#define PETSC_HAVE_FORTRAN PETSC_DEPRECATED_MACRO(3, 20, 0, "PETSC_USE_FORTRAN_BINDINGS", ) PETSC_USE_FORTRAN_BINDINGS

PETSC_EXTERN PetscErrorCode PetscBLASSetNumThreads(PetscInt);
PETSC_EXTERN PetscErrorCode PetscBLASGetNumThreads(PetscInt *);

/*MC
   PetscSafePointerPlusOffset - Checks that a pointer is not `NULL` before applying an offset

   Level: beginner

   Note:
   This is needed to avoid errors with undefined-behavior sanitizers such as
   UBSan, assuming PETSc has been configured with `-fsanitize=undefined` as part of the compiler flags
M*/
#define PetscSafePointerPlusOffset(ptr, offset) ((ptr) ? (ptr) + (offset) : NULL)

/* this is required to force PetscDevice to be visible at the system level for the Fortran interface */
#include <petscdevicetypes.h>

#if PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)
PETSC_EXTERN PetscErrorCode PetscStackView(FILE *);
#else
  #define PetscStackView(file) PETSC_SUCCESS
#endif
