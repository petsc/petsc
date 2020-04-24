/*
   This is the main PETSc include file (for C and C++).  It is included by all
   other PETSc include files, so it almost never has to be specifically included.
*/
#if !defined(PETSCSYS_H)
#define PETSCSYS_H
/* ========================================================================== */
/*
   petscconf.h is contained in ${PETSC_ARCH}/include/petscconf.h it is
   found automatically by the compiler due to the -I${PETSC_DIR}/${PETSC_ARCH}/include.
   For --prefix installs the ${PETSC_ARCH}/ does not exist and petscconf.h is in the same
   directory as the other PETSc include files.
*/
#include <petscconf.h>
#include <petscfix.h>

#if defined(PETSC_DESIRE_FEATURE_TEST_MACROS)
/*
   Feature test macros must be included before headers defined by IEEE Std 1003.1-2001
   We only turn these in PETSc source files that require them by setting PETSC_DESIRE_FEATURE_TEST_MACROS
*/
#  if defined(PETSC__POSIX_C_SOURCE_200112L) && !defined(_POSIX_C_SOURCE)
#    define _POSIX_C_SOURCE 200112L
#  endif
#  if defined(PETSC__BSD_SOURCE) && !defined(_BSD_SOURCE)
#    define _BSD_SOURCE
#  endif
#  if defined(PETSC__DEFAULT_SOURCE) && !defined(_DEFAULT_SOURCE)
#    define _DEFAULT_SOURCE
#  endif
#  if defined(PETSC__GNU_SOURCE) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE
#  endif
#endif

#include <petscsystypes.h>

/* ========================================================================== */
/*
   This facilitates using the C version of PETSc from C++ and the C++ version from C.
*/
#if defined(__cplusplus)
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_CXX
#else
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_C
#endif

/* ========================================================================== */
/*
   Since PETSc manages its own extern "C" handling users should never include PETSc include
   files within extern "C". This will generate a compiler error if a user does put the include
   file within an extern "C".
*/
#if defined(__cplusplus)
void assert_never_put_petsc_headers_inside_an_extern_c(int); void assert_never_put_petsc_headers_inside_an_extern_c(double);
#endif

#if defined(__cplusplus)
#  define PETSC_RESTRICT PETSC_CXX_RESTRICT
#else
#  define PETSC_RESTRICT PETSC_C_RESTRICT
#endif

#if defined(__cplusplus)
#  define PETSC_INLINE PETSC_CXX_INLINE
#else
#  define PETSC_INLINE PETSC_C_INLINE
#endif

#define PETSC_STATIC_INLINE static PETSC_INLINE

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES) /* For Win32 shared libraries */
#  define PETSC_DLLEXPORT __declspec(dllexport)
#  define PETSC_DLLIMPORT __declspec(dllimport)
#  define PETSC_VISIBILITY_INTERNAL
#elif defined(PETSC_USE_VISIBILITY_CXX) && defined(__cplusplus)
#  define PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define PETSC_DLLIMPORT __attribute__((visibility ("default")))
#  define PETSC_VISIBILITY_INTERNAL __attribute__((visibility ("hidden")))
#elif defined(PETSC_USE_VISIBILITY_C) && !defined(__cplusplus)
#  define PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define PETSC_DLLIMPORT __attribute__((visibility ("default")))
#  define PETSC_VISIBILITY_INTERNAL __attribute__((visibility ("hidden")))
#else
#  define PETSC_DLLEXPORT
#  define PETSC_DLLIMPORT
#  define PETSC_VISIBILITY_INTERNAL
#endif

#if defined(petsc_EXPORTS)      /* CMake defines this when building the shared library */
#  define PETSC_VISIBILITY_PUBLIC PETSC_DLLEXPORT
#else  /* Win32 users need this to import symbols from petsc.dll */
#  define PETSC_VISIBILITY_PUBLIC PETSC_DLLIMPORT
#endif

/*
    Functions tagged with PETSC_EXTERN in the header files are
  always defined as extern "C" when compiled with C++ so they may be
  used from C and are always visible in the shared libraries
*/
#if defined(__cplusplus)
#  define PETSC_EXTERN extern "C" PETSC_VISIBILITY_PUBLIC
#  define PETSC_EXTERN_TYPEDEF extern "C"
#  define PETSC_INTERN extern "C" PETSC_VISIBILITY_INTERNAL
#else
#  define PETSC_EXTERN extern PETSC_VISIBILITY_PUBLIC
#  define PETSC_EXTERN_TYPEDEF
#  define PETSC_INTERN extern PETSC_VISIBILITY_INTERNAL
#endif

#include <petscversion.h>
#define PETSC_AUTHOR_INFO  "       The PETSc Team\n    petsc-maint@mcs.anl.gov\n https://www.mcs.anl.gov/petsc/\n"

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
#  define MPICH_SKIP_MPICXX 1
#endif
#if !defined(OMPI_SKIP_MPICXX)
#  define OMPI_SKIP_MPICXX 1
#endif
#if defined(PETSC_HAVE_MPIUNI)
#  include <petsc/mpiuni/mpi.h>
#else
#  include <mpi.h>
#endif

/*MC
  PetscDefined - determine whether a boolean macro is defined

  Notes:
  Typical usage is within normal code,

$   if (PetscDefined(USE_DEBUG)) { ... }

  but can also be used in the preprocessor,

$   #if PetscDefined(USE_DEBUG)
$     ...
$   #else

  Either way evaluates true if PETSC_USE_DEBUG is defined (merely defined or defined to 1) or undefined.  This macro
  should not be used if its argument may be defined to a non-empty value other than 1.

  Developer Notes:
  Getting something that works in C and CPP for an arg that may or may not be defined is tricky.  Here, if we have
  "#define PETSC_HAVE_BOOGER 1" we match on the placeholder define, insert the "0," for arg1 and generate the triplet
  (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).  When PETSC_HAVE_BOOGER is not defined, we generate
  a (... 1, 0) pair, and when the last step cherry picks the 2nd arg, we get a zero.

  Our extra expansion via PetscDefined__take_second_expand() is needed with MSVC, which has a nonconforming
  implementation of variadic macros.

  Level: developer
M*/
#if !defined(PETSC_SKIP_VARIADIC_MACROS)
#  define PetscDefined_arg_1    shift,
#  define PetscDefined_arg_     shift,
#  define PetscDefined__take_second_expanded(ignored, val, ...) val
#  define PetscDefined__take_second_expand(args) PetscDefined__take_second_expanded args
#  define PetscDefined__take_second(...) PetscDefined__take_second_expand((__VA_ARGS__))
#  define PetscDefined___(arg1_or_junk) PetscDefined__take_second(arg1_or_junk 1, 0, at_)
#  define PetscDefined__(value) PetscDefined___(PetscDefined_arg_ ## value)
#  define PetscDefined_(d)      PetscDefined__(d)
#  define PetscDefined(d)       PetscDefined_(PETSC_ ## d)
#endif

/*
   Perform various sanity checks that the correct mpi.h is being included at compile time.
   This usually happens because
      * either an unexpected mpi.h is in the default compiler path (i.e. in /usr/include) or
      * an extra include path -I/something (which contains the unexpected mpi.h) is being passed to the compiler
*/
#if defined(PETSC_HAVE_MPIUNI)
#  if !defined(MPIUNI_H)
#    error "PETSc was configured with --with-mpi=0 but now appears to be compiling using a different mpi.h"
#  endif
#elif defined(PETSC_HAVE_I_MPI_NUMVERSION)
#  if !defined(I_MPI_NUMVERSION)
#    error "PETSc was configured with I_MPI but now appears to be compiling using a non-I_MPI mpi.h"
#  elif I_MPI_NUMVERSION != PETSC_HAVE_I_MPI_NUMVERSION
#    error "PETSc was configured with one I_MPI mpi.h version but now appears to be compiling using a different I_MPI mpi.h version"
#  endif
#elif defined(PETSC_HAVE_MVAPICH2_NUMVERSION)
#  if !defined(MVAPICH2_NUMVERSION)
#    error "PETSc was configured with MVAPICH2 but now appears to be compiling using a non-MVAPICH2 mpi.h"
#  elif MVAPICH2_NUMVERSION != PETSC_HAVE_MVAPICH2_NUMVERSION
#    error "PETSc was configured with one MVAPICH2 mpi.h version but now appears to be compiling using a different MVAPICH2 mpi.h version"
#  endif
#elif defined(PETSC_HAVE_MPICH_NUMVERSION)
#  if !defined(MPICH_NUMVERSION) || defined(MVAPICH2_NUMVERSION) || defined(I_MPI_NUMVERSION)
#    error "PETSc was configured with MPICH but now appears to be compiling using a non-MPICH mpi.h"
#  elif (MPICH_NUMVERSION/100000 != PETSC_HAVE_MPICH_NUMVERSION/100000) || (MPICH_NUMVERSION%100000/1000 < PETSC_HAVE_MPICH_NUMVERSION%100000/1000)
#    error "PETSc was configured with one MPICH mpi.h version but now appears to be compiling using a different MPICH mpi.h version"
#  endif
#elif defined(PETSC_HAVE_OMPI_MAJOR_VERSION)
#  if !defined(OMPI_MAJOR_VERSION)
#    error "PETSc was configured with OpenMPI but now appears to be compiling using a non-OpenMPI mpi.h"
#  elif (OMPI_MAJOR_VERSION != PETSC_HAVE_OMPI_MAJOR_VERSION) || (OMPI_MINOR_VERSION != PETSC_HAVE_OMPI_MINOR_VERSION) || (OMPI_RELEASE_VERSION < PETSC_HAVE_OMPI_RELEASE_VERSION)
#    error "PETSc was configured with one OpenMPI mpi.h version but now appears to be compiling using a different OpenMPI mpi.h version"
#  endif
#elif defined(PETSC_HAVE_MSMPI_VERSION)
#  if !defined(MSMPI_VER)
#    error "PETSc was configured with MSMPI but now appears to be compiling using a non-MSMPI mpi.h"
#  elif (MSMPI_VER != PETSC_HAVE_MSMPI_VERSION)
#    error "PETSc was configured with one MSMPI mpi.h version but now appears to be compiling using a different MSMPI mpi.h version"
#  endif
#elif defined(OMPI_MAJOR_VERSION) || defined(MPICH_NUMVERSION) || defined(MSMPI_VER)
#  error "PETSc was configured with undetermined MPI - but now appears to be compiling using any of OpenMPI, MS-MPI or a MPICH variant"
#endif

/*
    Need to put stdio.h AFTER mpi.h for MPICH2 with C++ compiler
    see the top of mpicxx.h in the MPICH2 distribution.
*/
#include <stdio.h>

/* MSMPI on 32bit windows requires this yukky hack - that breaks MPI standard compliance */
#if !defined(MPIAPI)
#define MPIAPI
#endif

/*
   Support for Clang (>=3.2) matching type tag arguments with void* buffer types.
   This allows the compiler to detect cases where the MPI datatype argument passed to a MPI routine
   does not match the actual type of the argument being passed in
*/
#if defined(__has_attribute) && defined(works_with_const_which_is_not_true)
#  if __has_attribute(argument_with_type_tag) && __has_attribute(pointer_with_type_tag) && __has_attribute(type_tag_for_datatype)
#    define PetscAttrMPIPointerWithType(bufno,typeno) __attribute__((pointer_with_type_tag(MPI,bufno,typeno)))
#    define PetscAttrMPITypeTag(type)                 __attribute__((type_tag_for_datatype(MPI,type)))
#    define PetscAttrMPITypeTagLayoutCompatible(type) __attribute__((type_tag_for_datatype(MPI,type,layout_compatible)))
#  endif
#endif
#if !defined(PetscAttrMPIPointerWithType)
#  define PetscAttrMPIPointerWithType(bufno,typeno)
#  define PetscAttrMPITypeTag(type)
#  define PetscAttrMPITypeTagLayoutCompatible(type)
#endif

PETSC_EXTERN MPI_Datatype MPIU_ENUM PetscAttrMPITypeTag(PetscEnum);
PETSC_EXTERN MPI_Datatype MPIU_BOOL PetscAttrMPITypeTag(PetscBool);

/*MC
   MPIU_INT - MPI datatype corresponding to PetscInt

   Notes:
   In MPI calls that require an MPI datatype that matches a PetscInt or array of PetscInt values, pass this value.

   Level: beginner

.seealso: PetscReal, PetscScalar, PetscComplex, PetscInt, MPIU_REAL, MPIU_SCALAR, MPIU_COMPLEX
M*/

#if defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T) /* MPI_INT64_T is not guaranteed to be a macro */
#  define MPIU_INT64 MPI_INT64_T
#  define PetscInt64_FMT PRId64
#elif (PETSC_SIZEOF_LONG_LONG == 8)
#  define MPIU_INT64 MPI_LONG_LONG_INT
#  define PetscInt64_FMT "lld"
#elif defined(PETSC_HAVE___INT64)
#  define MPIU_INT64 MPI_INT64_T
#  define PetscInt64_FMT "ld"
#else
#  error "cannot determine PetscInt64 type"
#endif

PETSC_EXTERN MPI_Datatype MPIU_FORTRANADDR;

#if defined(PETSC_USE_64BIT_INDICES)
#  define MPIU_INT MPIU_INT64
#  define PetscInt_FMT PetscInt64_FMT
#else
#  define MPIU_INT MPI_INT
#  define PetscInt_FMT "d"
#endif

/*
    For the rare cases when one needs to send a size_t object with MPI
*/
PETSC_EXTERN MPI_Datatype MPIU_SIZE_T;

/*
      You can use PETSC_STDOUT as a replacement of stdout. You can also change
    the value of PETSC_STDOUT to redirect all standard output elsewhere
*/
PETSC_EXTERN FILE* PETSC_STDOUT;

/*
      You can use PETSC_STDERR as a replacement of stderr. You can also change
    the value of PETSC_STDERR to redirect all standard error elsewhere
*/
PETSC_EXTERN FILE* PETSC_STDERR;

/*MC
    PetscUnlikely - hints the compiler that the given condition is usually FALSE

    Synopsis:
    #include <petscsys.h>
    PetscBool  PetscUnlikely(PetscBool  cond)

    Not Collective

    Input Parameters:
.   cond - condition or expression

    Notes:
    This returns the same truth value, it is only a hint to compilers that the resulting
    branch is unlikely.

    Level: advanced

.seealso: PetscUnlikelyDebug(), PetscLikely(), CHKERRQ
M*/

/*MC
    PetscLikely - hints the compiler that the given condition is usually TRUE

    Synopsis:
    #include <petscsys.h>
    PetscBool  PetscLikely(PetscBool  cond)

    Not Collective

    Input Parameters:
.   cond - condition or expression

    Notes:
    This returns the same truth value, it is only a hint to compilers that the resulting
    branch is likely.

    Level: advanced

.seealso: PetscUnlikely()
M*/
#if defined(PETSC_HAVE_BUILTIN_EXPECT)
#  define PetscUnlikely(cond)   __builtin_expect(!!(cond),0)
#  define PetscLikely(cond)     __builtin_expect(!!(cond),1)
#else
#  define PetscUnlikely(cond)   (cond)
#  define PetscLikely(cond)     (cond)
#endif

/*MC
    PetscUnlikelyDebug - hints the compiler that the given condition is usually FALSE, eliding the check in optimized mode

    Synopsis:
    #include <petscsys.h>
    PetscBool  PetscUnlikelyDebug(PetscBool  cond)

    Not Collective

    Input Parameters:
.   cond - condition or expression

    Notes:
    This returns the same truth value, it is only a hint to compilers that the resulting
    branch is unlikely.  When compiled in optimized mode, it always returns false.

    Level: advanced

.seealso: PetscUnlikely(), CHKERRQ, SETERRQ
M*/
#define PetscUnlikelyDebug(cond) (PetscDefined(USE_DEBUG) && PetscUnlikely(cond))

/*
    Declare extern C stuff after including external header files
*/

PETSC_EXTERN const char *const PetscBools[];

/*
    Defines elementary mathematics functions and constants.
*/
#include <petscmath.h>

PETSC_EXTERN const char *const PetscCopyModes[];

/*MC
    PETSC_IGNORE - same as NULL, means PETSc will ignore this argument

   Level: beginner

   Note:
   Accepted by many PETSc functions to not set a parameter and instead use
          some default

   Fortran Notes:
   This macro does not exist in Fortran; you must use PETSC_NULL_INTEGER,
          PETSC_NULL_DOUBLE_PRECISION etc

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_DETERMINE

M*/
#define PETSC_IGNORE NULL

/* This is deprecated */
#define PETSC_NULL NULL

/*MC
    PETSC_DECIDE - standard way of passing in integer or floating point parameter
       where you wish PETSc to use the default.

   Level: beginner

.seealso: PETSC_DEFAULT, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_DECIDE -1

/*MC
    PETSC_DETERMINE - standard way of passing in integer or floating point parameter
       where you wish PETSc to compute the required value.

   Level: beginner

   Developer Note:
   I would like to use const PetscInt PETSC_DETERMINE = PETSC_DECIDE; but for
     some reason this is not allowed by the standard even though PETSC_DECIDE is a constant value.

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_IGNORE, VecSetSizes()

M*/
#define PETSC_DETERMINE PETSC_DECIDE

/*MC
    PETSC_DEFAULT - standard way of passing in integer or floating point parameter
       where you wish PETSc to use the default.

   Level: beginner

   Fortran Notes:
   You need to use PETSC_DEFAULT_INTEGER or PETSC_DEFAULT_REAL.

.seealso: PETSC_DECIDE, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_DEFAULT -2

/*MC
    PETSC_COMM_WORLD - the equivalent of the MPI_COMM_WORLD communicator which represents
           all the processes that PETSc knows about.

   Level: beginner

   Notes:
   By default PETSC_COMM_WORLD and MPI_COMM_WORLD are identical unless you wish to
          run PETSc on ONLY a subset of MPI_COMM_WORLD. In that case create your new (smaller)
          communicator, call it, say comm, and set PETSC_COMM_WORLD = comm BEFORE calling
          PetscInitialize(), but after MPI_Init() has been called.

          The value of PETSC_COMM_WORLD should never be USED/accessed before PetscInitialize()
          is called because it may not have a valid value yet.

.seealso: PETSC_COMM_SELF

M*/
PETSC_EXTERN MPI_Comm PETSC_COMM_WORLD;

/*MC
    PETSC_COMM_SELF - This is always MPI_COMM_SELF

   Level: beginner

   Notes:
   Do not USE/access or set this variable before PetscInitialize() has been called.

.seealso: PETSC_COMM_WORLD

M*/
#define PETSC_COMM_SELF MPI_COMM_SELF

PETSC_EXTERN PetscBool PetscBeganMPI;
PETSC_EXTERN PetscBool PetscInitializeCalled;
PETSC_EXTERN PetscBool PetscFinalizeCalled;
PETSC_EXTERN PetscBool PetscViennaCLSynchronize;
PETSC_EXTERN PetscBool PetscCUDASynchronize;

PETSC_EXTERN PetscErrorCode PetscSetHelpVersionFunctions(PetscErrorCode (*)(MPI_Comm),PetscErrorCode (*)(MPI_Comm));
PETSC_EXTERN PetscErrorCode PetscCommDuplicate(MPI_Comm,MPI_Comm*,int*);
PETSC_EXTERN PetscErrorCode PetscCommDestroy(MPI_Comm*);

#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscCUDAInitialize(MPI_Comm);
#endif

/*MC
   PetscMalloc - Allocates memory, One should use PetscNew(), PetscMalloc1() or PetscCalloc1() usually instead of this

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

   It is safe to allocate size 0 and pass the resulting pointer (which may or may not be NULL) to PetscFree().

.seealso: PetscFree(), PetscNew()

M*/
#define PetscMalloc(a,b)  ((*PetscTrMalloc)((a),PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(void**)(b)))

/*MC
   PetscRealloc - Rellocates memory

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscRealloc(size_t m,void **result)

   Not Collective

   Input Parameters:
+  m - number of bytes to allocate
-  result - previous memory

   Output Parameter:
.  result - new memory allocated

   Level: developer

   Notes:
   Memory is always allocated at least double aligned

.seealso: PetscMalloc(), PetscFree(), PetscNew()

M*/
#define PetscRealloc(a,b)  ((*PetscTrRealloc)((a),__LINE__,PETSC_FUNCTION_NAME,__FILE__,(void**)(b)))

/*MC
   PetscAddrAlign - Rounds up an address to PETSC_MEMALIGN alignment

   Synopsis:
    #include <petscsys.h>
   void *PetscAddrAlign(void *addr)

   Not Collective

   Input Parameters:
.  addr - address to align (any pointer type)

   Level: developer

.seealso: PetscMallocAlign()

M*/
#define PetscAddrAlign(a) (void*)((((PETSC_UINTPTR_T)(a))+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

/*MC
   PetscMalloc1 - Allocates an array of memory aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc1(size_t m1,type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate  (may be zero)

   Output Parameter:
.  r1 - memory allocated

   Note:
   This uses the sizeof() of the memory type requested to determine the total memory to be allocated, therefore you should not
         multiply the number of elements requested by the sizeof() the type. For example use
$  PetscInt *id;
$  PetscMalloc1(10,&id);
        not
$  PetscInt *id;
$  PetscMalloc1(10*sizeof(PetscInt),&id);

        Does not zero the memory allocated, use PetscCalloc1() to obtain memory that has been zeroed.

   Level: beginner

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscCalloc1(), PetscMalloc2()

M*/
#define PetscMalloc1(m1,r1) PetscMallocA(1,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1))

/*MC
   PetscCalloc1 - Allocates a cleared (zeroed) array of memory aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc1(size_t m1,type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate in 1st chunk  (may be zero)

   Output Parameter:
.  r1 - memory allocated

   Notes:
   See PetsMalloc1() for more details on usage.

   Level: beginner

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc1(), PetscCalloc2()

M*/
#define PetscCalloc1(m1,r1) PetscMallocA(1,PETSC_TRUE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1))

/*MC
   PetscMalloc2 - Allocates 2 arrays of memory both aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc2(size_t m1,type **r1,size_t m2,type **r2)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
-  m2 - number of elements to allocate in 2nd chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
-  r2 - memory allocated in second chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc1(), PetscCalloc2()

M*/
#define PetscMalloc2(m1,r1,m2,r2) PetscMallocA(2,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2))

/*MC
   PetscCalloc2 - Allocates 2 cleared (zeroed) arrays of memory both aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc2(size_t m1,type **r1,size_t m2,type **r2)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
-  m2 - number of elements to allocate in 2nd chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
-  r2 - memory allocated in second chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscCalloc1(), PetscMalloc2()

M*/
#define PetscCalloc2(m1,r1,m2,r2) PetscMallocA(2,PETSC_TRUE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2))

/*MC
   PetscMalloc3 - Allocates 3 arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc3(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
-  m3 - number of elements to allocate in 3rd chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
-  r3 - memory allocated in third chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc3(), PetscFree3()

M*/
#define PetscMalloc3(m1,r1,m2,r2,m3,r3) PetscMallocA(3,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3))

/*MC
   PetscCalloc3 - Allocates 3 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc3(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
-  m3 - number of elements to allocate in 3rd chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
-  r3 - memory allocated in third chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscCalloc2(), PetscMalloc3(), PetscFree3()

M*/
#define PetscCalloc3(m1,r1,m2,r2,m3,r3) PetscMallocA(3,PETSC_TRUE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3))

/*MC
   PetscMalloc4 - Allocates 4 arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc4(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
-  m4 - number of elements to allocate in 4th chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
-  r4 - memory allocated in fourth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc4(), PetscFree4()

M*/
#define PetscMalloc4(m1,r1,m2,r2,m3,r3,m4,r4) PetscMallocA(4,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4))

/*MC
   PetscCalloc4 - Allocates 4 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

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

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc4(), PetscFree4()

M*/
#define PetscCalloc4(m1,r1,m2,r2,m3,r3,m4,r4) PetscMallocA(4,PETSC_TRUE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4))

/*MC
   PetscMalloc5 - Allocates 5 arrays of memory, all aligned to PETSC_MEMALIGN

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

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc5(), PetscFree5()

M*/
#define PetscMalloc5(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5) PetscMallocA(5,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4),(size_t)(m5)*sizeof(**(r5)),(r5))

/*MC
   PetscCalloc5 - Allocates 5 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

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

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc5(), PetscFree5()

M*/
#define PetscCalloc5(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5) PetscMallocA(5,PETSC_TRUE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4),(size_t)(m5)*sizeof(**(r5)),(r5))

/*MC
   PetscMalloc6 - Allocates 6 arrays of memory, all aligned to PETSC_MEMALIGN

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

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc6(), PetscFree3(), PetscFree4(), PetscFree5(), PetscFree6()

M*/
#define PetscMalloc6(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6) PetscMallocA(6,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4),(size_t)(m5)*sizeof(**(r5)),(r5),(size_t)(m6)*sizeof(**(r6)),(r6))

/*MC
   PetscCalloc6 - Allocates 6 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

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

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscMalloc6(), PetscFree6()

M*/
#define PetscCalloc6(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6) PetscMallocA(6,PETSC_TRUE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4),(size_t)(m5)*sizeof(**(r5)),(r5),(size_t)(m6)*sizeof(**(r6)),(r6))

/*MC
   PetscMalloc7 - Allocates 7 arrays of memory, all aligned to PETSC_MEMALIGN

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

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc7(), PetscFree7()

M*/
#define PetscMalloc7(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6,m7,r7) PetscMallocA(7,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4),(size_t)(m5)*sizeof(**(r5)),(r5),(size_t)(m6)*sizeof(**(r6)),(r6),(size_t)(m7)*sizeof(**(r7)),(r7))

/*MC
   PetscCalloc7 - Allocates 7 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

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

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscMalloc7(), PetscFree7()

M*/
#define PetscCalloc7(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6,m7,r7) PetscMallocA(7,PETSC_TRUE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(m1)*sizeof(**(r1)),(r1),(size_t)(m2)*sizeof(**(r2)),(r2),(size_t)(m3)*sizeof(**(r3)),(r3),(size_t)(m4)*sizeof(**(r4)),(r4),(size_t)(m5)*sizeof(**(r5)),(r5),(size_t)(m6)*sizeof(**(r6)),(r6),(size_t)(m7)*sizeof(**(r7)),(r7))

/*MC
   PetscNew - Allocates memory of a particular type, zeros the memory! Aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscNew(type **result)

   Not Collective

   Output Parameter:
.  result - memory allocated, sized to match pointer type

   Level: beginner

.seealso: PetscFree(), PetscMalloc(), PetscNewLog(), PetscCalloc1(), PetscMalloc1()

M*/
#define PetscNew(b)      PetscCalloc1(1,(b))

/*MC
   PetscNewLog - Allocates memory of a type matching pointer, zeros the memory! Aligned to PETSC_MEMALIGN. Associates the memory allocated
         with the given object using PetscLogObjectMemory().

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscNewLog(PetscObject obj,type **result)

   Not Collective

   Input Parameter:
.  obj - object memory is logged to

   Output Parameter:
.  result - memory allocated, sized to match pointer type

   Level: developer

.seealso: PetscFree(), PetscMalloc(), PetscNew(), PetscLogObjectMemory(), PetscCalloc1(), PetscMalloc1()

M*/
#define PetscNewLog(o,b) (PetscNew((b)) || PetscLogObjectMemory((PetscObject)o,sizeof(**(b))))

/*MC
   PetscFree - Frees memory

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree(void *memory)

   Not Collective

   Input Parameter:
.   memory - memory to free (the pointer is ALWAYS set to NULL upon success)

   Level: beginner

   Notes:
   Do not free memory obtained with PetscMalloc2(), PetscCalloc2() etc, they must be freed with PetscFree2() etc.

   It is safe to call PetscFree() on a NULL pointer.

.seealso: PetscNew(), PetscMalloc(), PetscNewLog(), PetscMalloc1(), PetscCalloc1()

M*/
#define PetscFree(a)   ((*PetscTrFree)((void*)(a),__LINE__,PETSC_FUNCTION_NAME,__FILE__) || ((a) = NULL,0))

/*MC
   PetscFree2 - Frees 2 chunks of memory obtained with PetscMalloc2()

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree2(void *memory1,void *memory2)

   Not Collective

   Input Parameters:
+   memory1 - memory to free
-   memory2 - 2nd memory to free

   Level: developer

   Notes:
    Memory must have been obtained with PetscMalloc2()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree()

M*/
#define PetscFree2(m1,m2)   PetscFreeA(2,__LINE__,PETSC_FUNCTION_NAME,__FILE__,&(m1),&(m2))

/*MC
   PetscFree3 - Frees 3 chunks of memory obtained with PetscMalloc3()

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
    Memory must have been obtained with PetscMalloc3()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3()

M*/
#define PetscFree3(m1,m2,m3)   PetscFreeA(3,__LINE__,PETSC_FUNCTION_NAME,__FILE__,&(m1),&(m2),&(m3))

/*MC
   PetscFree4 - Frees 4 chunks of memory obtained with PetscMalloc4()

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
    Memory must have been obtained with PetscMalloc4()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4()

M*/
#define PetscFree4(m1,m2,m3,m4)   PetscFreeA(4,__LINE__,PETSC_FUNCTION_NAME,__FILE__,&(m1),&(m2),&(m3),&(m4))

/*MC
   PetscFree5 - Frees 5 chunks of memory obtained with PetscMalloc5()

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
    Memory must have been obtained with PetscMalloc5()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5()

M*/
#define PetscFree5(m1,m2,m3,m4,m5)   PetscFreeA(5,__LINE__,PETSC_FUNCTION_NAME,__FILE__,&(m1),&(m2),&(m3),&(m4),&(m5))

/*MC
   PetscFree6 - Frees 6 chunks of memory obtained with PetscMalloc6()

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
    Memory must have been obtained with PetscMalloc6()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5(), PetscMalloc6()

M*/
#define PetscFree6(m1,m2,m3,m4,m5,m6)   PetscFreeA(6,__LINE__,PETSC_FUNCTION_NAME,__FILE__,&(m1),&(m2),&(m3),&(m4),&(m5),&(m6))

/*MC
   PetscFree7 - Frees 7 chunks of memory obtained with PetscMalloc7()

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
    Memory must have been obtained with PetscMalloc7()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5(), PetscMalloc6(),
          PetscMalloc7()

M*/
#define PetscFree7(m1,m2,m3,m4,m5,m6,m7)   PetscFreeA(7,__LINE__,PETSC_FUNCTION_NAME,__FILE__,&(m1),&(m2),&(m3),&(m4),&(m5),&(m6),&(m7))

PETSC_EXTERN PetscErrorCode PetscMallocA(int,PetscBool,int,const char *,const char *,size_t,void *,...);
PETSC_EXTERN PetscErrorCode PetscFreeA(int,int,const char *,const char *,void *,...);
PETSC_EXTERN PetscErrorCode (*PetscTrMalloc)(size_t,PetscBool,int,const char[],const char[],void**);
PETSC_EXTERN PetscErrorCode (*PetscTrFree)(void*,int,const char[],const char[]);
PETSC_EXTERN PetscErrorCode (*PetscTrRealloc)(size_t,int,const char[],const char[],void**);
PETSC_EXTERN PetscErrorCode PetscMallocSetCoalesce(PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocSet(PetscErrorCode (*)(size_t,PetscBool,int,const char[],const char[],void**),PetscErrorCode (*)(void*,int,const char[],const char[]),PetscErrorCode (*)(size_t,int,const char[],const char[], void **));
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
PETSC_EXTERN PetscErrorCode PetscMallocPopMaximumUsage(int,PetscLogDouble*);
PETSC_EXTERN PetscErrorCode PetscMallocSetDebug(PetscBool,PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocGetDebug(PetscBool*,PetscBool*,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscMallocValidate(int,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscMallocViewSet(PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscMallocViewGet(PetscBool*);

PETSC_EXTERN const char *const PetscDataTypes[];
PETSC_EXTERN PetscErrorCode PetscDataTypeToMPIDataType(PetscDataType,MPI_Datatype*);
PETSC_EXTERN PetscErrorCode PetscMPIDataTypeToPetscDataType(MPI_Datatype,PetscDataType*);
PETSC_EXTERN PetscErrorCode PetscDataTypeGetSize(PetscDataType,size_t*);
PETSC_EXTERN PetscErrorCode PetscDataTypeFromString(const char*,PetscDataType*,PetscBool*);

/*
    Basic memory and string operations. These are usually simple wrappers
   around the basic Unix system calls, but a few of them have additional
   functionality and/or error checking.
*/
PETSC_EXTERN PetscErrorCode PetscBitMemcpy(void*,PetscInt,const void*,PetscInt,PetscInt,PetscDataType);
PETSC_EXTERN PetscErrorCode PetscMemcmp(const void*,const void*,size_t,PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscStrlen(const char[],size_t*);
PETSC_EXTERN PetscErrorCode PetscStrToArray(const char[],char,int*,char ***);
PETSC_EXTERN PetscErrorCode PetscStrToArrayDestroy(int,char **);
PETSC_EXTERN PetscErrorCode PetscStrcmp(const char[],const char[],PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscStrgrt(const char[],const char[],PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscStrcasecmp(const char[],const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscStrncmp(const char[],const char[],size_t,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscStrcpy(char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscStrcat(char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscStrlcat(char[],const char[],size_t);
PETSC_EXTERN PetscErrorCode PetscStrncpy(char[],const char[],size_t);
PETSC_EXTERN PetscErrorCode PetscStrchr(const char[],char,char *[]);
PETSC_EXTERN PetscErrorCode PetscStrtolower(char[]);
PETSC_EXTERN PetscErrorCode PetscStrtoupper(char[]);
PETSC_EXTERN PetscErrorCode PetscStrrchr(const char[],char,char *[]);
PETSC_EXTERN PetscErrorCode PetscStrstr(const char[],const char[],char *[]);
PETSC_EXTERN PetscErrorCode PetscStrrstr(const char[],const char[],char *[]);
PETSC_EXTERN PetscErrorCode PetscStrendswith(const char[],const char[],PetscBool*);
PETSC_EXTERN PetscErrorCode PetscStrbeginswith(const char[],const char[],PetscBool*);
PETSC_EXTERN PetscErrorCode PetscStrendswithwhich(const char[],const char *const*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscStrallocpy(const char[],char *[]);
PETSC_EXTERN PetscErrorCode PetscStrArrayallocpy(const char *const*,char***);
PETSC_EXTERN PetscErrorCode PetscStrArrayDestroy(char***);
PETSC_EXTERN PetscErrorCode PetscStrNArrayallocpy(PetscInt,const char *const*,char***);
PETSC_EXTERN PetscErrorCode PetscStrNArrayDestroy(PetscInt,char***);
PETSC_EXTERN PetscErrorCode PetscStrreplace(MPI_Comm,const char[],char[],size_t);

PETSC_EXTERN void PetscStrcmpNoError(const char[],const char[],PetscBool  *);

PETSC_EXTERN PetscErrorCode PetscTokenCreate(const char[],const char,PetscToken*);
PETSC_EXTERN PetscErrorCode PetscTokenFind(PetscToken,char *[]);
PETSC_EXTERN PetscErrorCode PetscTokenDestroy(PetscToken*);

PETSC_EXTERN PetscErrorCode PetscStrInList(const char[],const char[],char,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscEListFind(PetscInt,const char *const*,const char*,PetscInt*,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscEnumFind(const char *const*,const char*,PetscEnum*,PetscBool*);

/*
   These are MPI operations for MPI_Allreduce() etc
*/
PETSC_EXTERN MPI_Op MPIU_MAXSUM_OP;
#if (defined(PETSC_HAVE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
PETSC_EXTERN MPI_Op MPIU_SUM;
#else
#define MPIU_SUM MPI_SUM
#endif
#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
PETSC_EXTERN MPI_Op MPIU_MAX;
PETSC_EXTERN MPI_Op MPIU_MIN;
#else
#define MPIU_MAX MPI_MAX
#define MPIU_MIN MPI_MIN
#endif
PETSC_EXTERN PetscErrorCode PetscMaxSum(MPI_Comm,const PetscInt[],PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode MPIULong_Send(void*,PetscInt,MPI_Datatype,PetscMPIInt,PetscMPIInt,MPI_Comm);
PETSC_EXTERN PetscErrorCode MPIULong_Recv(void*,PetscInt,MPI_Datatype,PetscMPIInt,PetscMPIInt,MPI_Comm);

PETSC_EXTERN const char *const PetscFileModes[];

/*
    Defines PETSc error handling.
*/
#include <petscerror.h>

#define PETSC_SMALLEST_CLASSID  1211211
PETSC_EXTERN PetscClassId PETSC_LARGEST_CLASSID;
PETSC_EXTERN PetscClassId PETSC_OBJECT_CLASSID;
PETSC_EXTERN PetscErrorCode PetscClassIdRegister(const char[],PetscClassId *);
PETSC_EXTERN PetscErrorCode PetscObjectGetId(PetscObject,PetscObjectId*);
PETSC_EXTERN PetscErrorCode PetscObjectCompareId(PetscObject,PetscObjectId,PetscBool*);


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
PETSC_EXTERN PetscErrorCode PetscInitialize(int*,char***,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscInitializeNoPointers(int,char**,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscInitializeNoArguments(void);
PETSC_EXTERN PetscErrorCode PetscInitialized(PetscBool *);
PETSC_EXTERN PetscErrorCode PetscFinalized(PetscBool *);
PETSC_EXTERN PetscErrorCode PetscFinalize(void);
PETSC_EXTERN PetscErrorCode PetscInitializeFortran(void);
PETSC_EXTERN PetscErrorCode PetscGetArgs(int*,char ***);
PETSC_EXTERN PetscErrorCode PetscGetArguments(char ***);
PETSC_EXTERN PetscErrorCode PetscFreeArguments(char **);

PETSC_EXTERN PetscErrorCode PetscEnd(void);
PETSC_EXTERN PetscErrorCode PetscSysInitializePackage(void);

PETSC_EXTERN PetscErrorCode PetscPythonInitialize(const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscPythonFinalize(void);
PETSC_EXTERN PetscErrorCode PetscPythonPrintError(void);
PETSC_EXTERN PetscErrorCode PetscPythonMonitorSet(PetscObject,const char[]);

PETSC_EXTERN PetscErrorCode PetscMonitorCompare(PetscErrorCode (*)(void),void *,PetscErrorCode (*)(void**),PetscErrorCode (*)(void),void *,PetscErrorCode (*)(void**),PetscBool *);


/*
     These are so that in extern C code we can caste function pointers to non-extern C
   function pointers. Since the regular C++ code expects its function pointers to be C++
*/
PETSC_EXTERN_TYPEDEF typedef void (**PetscVoidStarFunction)(void);
PETSC_EXTERN_TYPEDEF typedef void (*PetscVoidFunction)(void);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscErrorCodeFunction)(void);

/*
    Functions that can act on any PETSc object.
*/
PETSC_EXTERN PetscErrorCode PetscObjectDestroy(PetscObject*);
PETSC_EXTERN PetscErrorCode PetscObjectGetComm(PetscObject,MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscObjectGetClassId(PetscObject,PetscClassId *);
PETSC_EXTERN PetscErrorCode PetscObjectGetClassName(PetscObject,const char *[]);
PETSC_EXTERN PetscErrorCode PetscObjectSetType(PetscObject,const char []);
PETSC_EXTERN PetscErrorCode PetscObjectGetType(PetscObject,const char *[]);
PETSC_EXTERN PetscErrorCode PetscObjectSetName(PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectGetName(PetscObject,const char*[]);
PETSC_EXTERN PetscErrorCode PetscObjectSetTabLevel(PetscObject,PetscInt);
PETSC_EXTERN PetscErrorCode PetscObjectGetTabLevel(PetscObject,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscObjectIncrementTabLevel(PetscObject,PetscObject,PetscInt);
PETSC_EXTERN PetscErrorCode PetscObjectReference(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectGetReference(PetscObject,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscObjectDereference(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectGetNewTag(PetscObject,PetscMPIInt *);
PETSC_EXTERN PetscErrorCode PetscObjectCompose(PetscObject,const char[],PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectRemoveReference(PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectQuery(PetscObject,const char[],PetscObject *);
PETSC_EXTERN PetscErrorCode PetscObjectComposeFunction_Private(PetscObject,const char[],void (*)(void));
#define PetscObjectComposeFunction(a,b,d) PetscObjectComposeFunction_Private(a,b,(PetscVoidFunction)(d))
PETSC_EXTERN PetscErrorCode PetscObjectSetFromOptions(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSetUp(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSetPrintedOptions(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectInheritPrintedOptions(PetscObject,PetscObject);
PETSC_EXTERN PetscErrorCode PetscCommGetNewTag(MPI_Comm,PetscMPIInt *);

#include <petscviewertypes.h>
#include <petscoptions.h>

PETSC_EXTERN PetscErrorCode PetscObjectsListGetGlobalNumbering(MPI_Comm,PetscInt,PetscObject*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode PetscMemoryShowUsage(PetscViewer,const char[]);
PETSC_EXTERN PetscErrorCode PetscMemoryView(PetscViewer,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectPrintClassNamePrefixType(PetscObject,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscObjectView(PetscObject,PetscViewer);
#define PetscObjectQueryFunction(obj,name,fptr) PetscObjectQueryFunction_Private((obj),(name),(PetscVoidFunction*)(fptr))
PETSC_EXTERN PetscErrorCode PetscObjectQueryFunction_Private(PetscObject,const char[],void (**)(void));
PETSC_EXTERN PetscErrorCode PetscObjectSetOptionsPrefix(PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectAppendOptionsPrefix(PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectPrependOptionsPrefix(PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectGetOptionsPrefix(PetscObject,const char*[]);
PETSC_EXTERN PetscErrorCode PetscObjectChangeTypeName(PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectRegisterDestroy(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectRegisterDestroyAll(void);
PETSC_EXTERN PetscErrorCode PetscObjectViewFromOptions(PetscObject,PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectName(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectTypeCompare(PetscObject,const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscObjectBaseTypeCompare(PetscObject,const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscObjectTypeCompareAny(PetscObject,PetscBool*,const char[],...);
PETSC_EXTERN PetscErrorCode PetscObjectBaseTypeCompareAny(PetscObject,PetscBool*,const char[],...);
PETSC_EXTERN PetscErrorCode PetscRegisterFinalize(PetscErrorCode (*)(void));
PETSC_EXTERN PetscErrorCode PetscRegisterFinalizeAll(void);

#if defined(PETSC_HAVE_SAWS)
PETSC_EXTERN PetscErrorCode PetscSAWsBlock(void);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsViewOff(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsSetBlock(PetscObject,PetscBool);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsBlock(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsGrantAccess(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsTakeAccess(PetscObject);
PETSC_EXTERN void           PetscStackSAWsGrantAccess(void);
PETSC_EXTERN void           PetscStackSAWsTakeAccess(void);
PETSC_EXTERN PetscErrorCode PetscStackViewSAWs(void);
PETSC_EXTERN PetscErrorCode PetscStackSAWsViewOff(void);

#else
#define PetscSAWsBlock()                        0
#define PetscObjectSAWsViewOff(obj)             0
#define PetscObjectSAWsSetBlock(obj,flg)        0
#define PetscObjectSAWsBlock(obj)               0
#define PetscObjectSAWsGrantAccess(obj)         0
#define PetscObjectSAWsTakeAccess(obj)          0
#define PetscStackViewSAWs()                    0
#define PetscStackSAWsViewOff()                 0
#define PetscStackSAWsTakeAccess()
#define PetscStackSAWsGrantAccess()

#endif

PETSC_EXTERN PetscErrorCode PetscDLOpen(const char[],PetscDLMode,PetscDLHandle *);
PETSC_EXTERN PetscErrorCode PetscDLClose(PetscDLHandle *);
PETSC_EXTERN PetscErrorCode PetscDLSym(PetscDLHandle,const char[],void **);

#if defined(PETSC_USE_DEBUG)
PETSC_EXTERN PetscErrorCode PetscMallocGetStack(void*,PetscStack**);
#endif
PETSC_EXTERN PetscErrorCode PetscObjectsDump(FILE*,PetscBool);

PETSC_EXTERN PetscErrorCode PetscObjectListDestroy(PetscObjectList*);
PETSC_EXTERN PetscErrorCode PetscObjectListFind(PetscObjectList,const char[],PetscObject*);
PETSC_EXTERN PetscErrorCode PetscObjectListReverseFind(PetscObjectList,PetscObject,char**,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscObjectListAdd(PetscObjectList *,const char[],PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectListRemoveReference(PetscObjectList *,const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectListDuplicate(PetscObjectList,PetscObjectList *);

/*
    Dynamic library lists. Lists of names of routines in objects or in dynamic
  link libraries that will be loaded as needed.
*/

#define PetscFunctionListAdd(list,name,fptr) PetscFunctionListAdd_Private((list),(name),(PetscVoidFunction)(fptr))
PETSC_EXTERN PetscErrorCode PetscFunctionListAdd_Private(PetscFunctionList*,const char[],void (*)(void));
PETSC_EXTERN PetscErrorCode PetscFunctionListDestroy(PetscFunctionList*);
#define PetscFunctionListFind(list,name,fptr) PetscFunctionListFind_Private((list),(name),(PetscVoidFunction*)(fptr))
PETSC_EXTERN PetscErrorCode PetscFunctionListFind_Private(PetscFunctionList,const char[],void (**)(void));
PETSC_EXTERN PetscErrorCode PetscFunctionListPrintTypes(MPI_Comm,FILE*,const char[],const char[],const char[],const char[],PetscFunctionList,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscFunctionListDuplicate(PetscFunctionList,PetscFunctionList *);
PETSC_EXTERN PetscErrorCode PetscFunctionListView(PetscFunctionList,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscFunctionListGet(PetscFunctionList,const char ***,int*);

PETSC_EXTERN PetscDLLibrary  PetscDLLibrariesLoaded;
PETSC_EXTERN PetscErrorCode PetscDLLibraryAppend(MPI_Comm,PetscDLLibrary *,const char[]);
PETSC_EXTERN PetscErrorCode PetscDLLibraryPrepend(MPI_Comm,PetscDLLibrary *,const char[]);
PETSC_EXTERN PetscErrorCode PetscDLLibrarySym(MPI_Comm,PetscDLLibrary *,const char[],const char[],void **);
PETSC_EXTERN PetscErrorCode PetscDLLibraryPrintPath(PetscDLLibrary);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRetrieve(MPI_Comm,const char[],char *,size_t,PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscDLLibraryOpen(MPI_Comm,const char[],PetscDLLibrary *);
PETSC_EXTERN PetscErrorCode PetscDLLibraryClose(PetscDLLibrary);

/*
     Useful utility routines
*/
PETSC_EXTERN PetscErrorCode PetscSplitOwnership(MPI_Comm,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSplitOwnershipBlock(MPI_Comm,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSequentialPhaseBegin(MPI_Comm,PetscMPIInt);
PETSC_EXTERN PetscErrorCode PetscSequentialPhaseEnd(MPI_Comm,PetscMPIInt);
PETSC_EXTERN PetscErrorCode PetscBarrier(PetscObject);
PETSC_EXTERN PetscErrorCode PetscMPIDump(FILE*);
PETSC_EXTERN PetscErrorCode PetscGlobalMinMaxInt(MPI_Comm,PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscGlobalMinMaxReal(MPI_Comm,PetscReal[],PetscReal[]);

/*MC
    PetscNot - negates a logical type value and returns result as a PetscBool

    Notes:
    This is useful in cases like
$     int        *a;
$     PetscBool  flag = PetscNot(a)
     where !a would not return a PetscBool because we cannot provide a cast from int to PetscBool in C.

    Level: beginner

    .seealso : PetscBool, PETSC_TRUE, PETSC_FALSE
M*/
#define PetscNot(a) ((a) ? PETSC_FALSE : PETSC_TRUE)

/*MC
    PetscHelpPrintf - Prints help messages.

   Synopsis:
    #include <petscsys.h>
     PetscErrorCode (*PetscHelpPrintf)(const char format[],...);

    Not Collective

    Input Parameters:
.   format - the usual printf() format string

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.


.seealso: PetscFPrintf(), PetscSynchronizedPrintf(), PetscErrorPrintf()
M*/
PETSC_EXTERN PetscErrorCode (*PetscHelpPrintf)(MPI_Comm,const char[],...);

/*
     Defines PETSc profiling.
*/
#include <petsclog.h>

/*
      Simple PETSc parallel IO for ASCII printing
*/
PETSC_EXTERN PetscErrorCode PetscFixFilename(const char[],char[]);
PETSC_EXTERN PetscErrorCode PetscFOpen(MPI_Comm,const char[],const char[],FILE**);
PETSC_EXTERN PetscErrorCode PetscFClose(MPI_Comm,FILE*);
PETSC_EXTERN PetscErrorCode PetscFPrintf(MPI_Comm,FILE*,const char[],...);
PETSC_EXTERN PetscErrorCode PetscPrintf(MPI_Comm,const char[],...);
PETSC_EXTERN PetscErrorCode PetscSNPrintf(char*,size_t,const char [],...);
PETSC_EXTERN PetscErrorCode PetscSNPrintfCount(char*,size_t,const char [],size_t*,...);
PETSC_EXTERN PetscErrorCode PetscFormatRealArray(char[],size_t,const char*,PetscInt,const PetscReal[]);

PETSC_EXTERN PetscErrorCode PetscErrorPrintfDefault(const char [],...);
PETSC_EXTERN PetscErrorCode PetscErrorPrintfNone(const char [],...);
PETSC_EXTERN PetscErrorCode PetscHelpPrintfDefault(MPI_Comm,const char [],...);

PETSC_EXTERN PetscErrorCode PetscFormatConvertGetSize(const char*,size_t*);
PETSC_EXTERN PetscErrorCode PetscFormatConvert(const char*,char *);

#if defined(PETSC_HAVE_POPEN)
PETSC_EXTERN PetscErrorCode PetscPOpen(MPI_Comm,const char[],const char[],const char[],FILE **);
PETSC_EXTERN PetscErrorCode PetscPClose(MPI_Comm,FILE*);
PETSC_EXTERN PetscErrorCode PetscPOpenSetMachine(const char[]);
#endif

PETSC_EXTERN PetscErrorCode PetscSynchronizedPrintf(MPI_Comm,const char[],...);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFPrintf(MPI_Comm,FILE*,const char[],...);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFlush(MPI_Comm,FILE*);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFGets(MPI_Comm,FILE*,size_t,char[]);
PETSC_EXTERN PetscErrorCode PetscStartMatlab(MPI_Comm,const char[],const char[],FILE**);
PETSC_EXTERN PetscErrorCode PetscStartJava(MPI_Comm,const char[],const char[],FILE**);
PETSC_EXTERN PetscErrorCode PetscGetPetscDir(const char*[]);

PETSC_EXTERN PetscClassId PETSC_CONTAINER_CLASSID;
PETSC_EXTERN PetscErrorCode PetscContainerGetPointer(PetscContainer,void **);
PETSC_EXTERN PetscErrorCode PetscContainerSetPointer(PetscContainer,void *);
PETSC_EXTERN PetscErrorCode PetscContainerDestroy(PetscContainer*);
PETSC_EXTERN PetscErrorCode PetscContainerCreate(MPI_Comm,PetscContainer *);
PETSC_EXTERN PetscErrorCode PetscContainerSetUserDestroy(PetscContainer, PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode PetscContainerUserDestroyDefault(void*);

/*
   For use in debuggers
*/
PETSC_EXTERN PetscMPIInt PetscGlobalRank;
PETSC_EXTERN PetscMPIInt PetscGlobalSize;
PETSC_EXTERN PetscErrorCode PetscIntView(PetscInt,const PetscInt[],PetscViewer);
PETSC_EXTERN PetscErrorCode PetscRealView(PetscInt,const PetscReal[],PetscViewer);
PETSC_EXTERN PetscErrorCode PetscScalarView(PetscInt,const PetscScalar[],PetscViewer);

#include <stddef.h>
#include <string.h>             /* for memcpy, memset */
#include <stdlib.h>

#if defined(PETSC_HAVE_XMMINTRIN_H) && !defined(__CUDACC__)
#include <xmmintrin.h>
#endif

/*@C
   PetscMemmove - Copies n bytes, beginning at location b, to the space
   beginning at location a. Copying  between regions that overlap will
   take place correctly. Use PetscMemcpy() if the locations do not overlap

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
.  a - pointer to copy space
-  n - length (in bytes) of space to copy

   Level: intermediate

   Note:
   PetscArraymove() is preferred
   This routine is analogous to memmove().

   Developers Note: This is inlined for performance

.seealso: PetscMemcpy(), PetscMemcmp(), PetscArrayzero(), PetscMemzero(), PetscArraycmp(), PetscArraycpy(), PetscStrallocpy(),
          PetscArraymove()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscMemmove(void *a,const void *b,size_t n)
{
  PetscFunctionBegin;
  if (n > 0 && !a) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy to null pointer");
  if (n > 0 && !b) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy from a null pointer");
#if !defined(PETSC_HAVE_MEMMOVE)
  if (a < b) {
    if (a <= b - n) memcpy(a,b,n);
    else {
      memcpy(a,b,(int)(b - a));
      PetscMemmove(b,b + (int)(b - a),n - (int)(b - a));
    }
  } else {
    if (b <= a - n) memcpy(a,b,n);
    else {
      memcpy(b + n,b + (n - (int)(a - b)),(int)(a - b));
      PetscMemmove(a,b,n - (int)(a - b));
    }
  }
#else
  memmove((char*)(a),(char*)(b),n);
#endif
  PetscFunctionReturn(0);
}

/*@C
   PetscMemcpy - Copies n bytes, beginning at location b, to the space
   beginning at location a. The two memory regions CANNOT overlap, use
   PetscMemmove() in that case.

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
-  n - length (in bytes) of space to copy

   Output Parameter:
.  a - pointer to copy space

   Level: intermediate

   Compile Option:
    PETSC_PREFER_DCOPY_FOR_MEMCPY will cause the BLAS dcopy() routine to be used
                                  for memory copies on double precision values.
    PETSC_PREFER_COPY_FOR_MEMCPY will cause C code to be used
                                  for memory copies on double precision values.
    PETSC_PREFER_FORTRAN_FORMEMCPY will cause Fortran code to be used
                                  for memory copies on double precision values.

   Note:
   Prefer PetscArraycpy()
   This routine is analogous to memcpy().
   Not available from Fortran

   Developer Note: this is inlined for fastest performance

.seealso: PetscMemzero(), PetscMemcmp(), PetscArrayzero(), PetscArraycmp(), PetscArraycpy(), PetscMemmove(), PetscStrallocpy()

@*/
PETSC_STATIC_INLINE PetscErrorCode PetscMemcpy(void *a,const void *b,size_t n)
{
#if defined(PETSC_USE_DEBUG)
  size_t al = (size_t) a,bl = (size_t) b;
  size_t nl = (size_t) n;
  PetscFunctionBegin;
  if (n > 0 && !b) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy from a null pointer");
  if (n > 0 && !a) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy to a null pointer");
#else
  PetscFunctionBegin;
#endif
  if (a != b && n > 0) {
#if defined(PETSC_USE_DEBUG)
    if ((al > bl && (al - bl) < nl) || (bl - al) < nl)  SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Memory regions overlap: either use PetscMemmov()\n\
              or make sure your copy regions and lengths are correct. \n\
              Length (bytes) %ld first address %ld second address %ld",nl,al,bl);
#endif
#if (defined(PETSC_PREFER_DCOPY_FOR_MEMCPY) || defined(PETSC_PREFER_COPY_FOR_MEMCPY) || defined(PETSC_PREFER_FORTRAN_FORMEMCPY))
   if (!(a % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      size_t len = n/sizeof(PetscScalar);
#if defined(PETSC_PREFER_DCOPY_FOR_MEMCPY)
      PetscBLASInt   one = 1,blen;
      PetscErrorCode ierr;
      ierr = PetscBLASIntCast(len,&blen);CHKERRQ(ierr);
      PetscStackCallBLAS("BLAScopy",BLAScopy_(&blen,(PetscScalar *)b,&one,(PetscScalar *)a,&one));
#elif defined(PETSC_PREFER_FORTRAN_FORMEMCPY)
      fortrancopy_(&len,(PetscScalar*)b,(PetscScalar*)a);
#else
      size_t      i;
      PetscScalar *x = (PetscScalar*)b, *y = (PetscScalar*)a;
      for (i=0; i<len; i++) y[i] = x[i];
#endif
    } else {
      memcpy((char*)(a),(char*)(b),n);
    }
#else
    memcpy((char*)(a),(char*)(b),n);
#endif
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscMemzero - Zeros the specified memory.

   Not Collective

   Input Parameters:
+  a - pointer to beginning memory location
-  n - length (in bytes) of memory to initialize

   Level: intermediate

   Compile Option:
   PETSC_PREFER_BZERO - on certain machines (the IBM RS6000) the bzero() routine happens
  to be faster than the memset() routine. This flag causes the bzero() routine to be used.

   Not available from Fortran
   Prefer PetscArrayzero()

   Developer Note: this is inlined for fastest performance

.seealso: PetscMemcpy(), PetscMemcmp(), PetscArrayzero(), PetscArraycmp(), PetscArraycpy(), PetscMemmove(), PetscStrallocpy()
@*/
PETSC_STATIC_INLINE PetscErrorCode  PetscMemzero(void *a,size_t n)
{
  if (n > 0) {
#if defined(PETSC_USE_DEBUG)
    if (!a) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to zero at a null pointer");
#endif
#if defined(PETSC_PREFER_ZERO_FOR_MEMZERO)
    if (!(((long) a) % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      size_t      i,len = n/sizeof(PetscScalar);
      PetscScalar *x = (PetscScalar*)a;
      for (i=0; i<len; i++) x[i] = 0.0;
    } else {
#elif defined(PETSC_PREFER_FORTRAN_FOR_MEMZERO)
    if (!(((long) a) % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      PetscInt len = n/sizeof(PetscScalar);
      fortranzero_(&len,(PetscScalar*)a);
    } else {
#endif
#if defined(PETSC_PREFER_BZERO)
      bzero((char *)a,n);
#else
      memset((char*)a,0,n);
#endif
#if defined(PETSC_PREFER_ZERO_FOR_MEMZERO) || defined(PETSC_PREFER_FORTRAN_FOR_MEMZERO)
    }
#endif
  }
  return 0;
}

/*MC
   PetscArraycmp - Compares two arrays in memory.

   Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscArraycmp(const anytype *str1,const anytype *str2,size_t cnt,PetscBool *e)

   Not Collective

   Input Parameters:
+  str1 - First array
.  str2 - Second array
-  cnt  - Count of the array, not in bytes, but number of entries in the arrays

   Output Parameters:
.   e - PETSC_TRUE if equal else PETSC_FALSE.

   Level: intermediate

   Note:
   This routine is a preferred replacement to PetscMemcmp()
   The arrays must be of the same type

.seealso: PetscMemcpy(), PetscMemcmp(), PetscArrayzero(), PetscMemzero(), PetscArraycpy(), PetscMemmove(), PetscStrallocpy(),
          PetscArraymove()
M*/
#define  PetscArraycmp(str1,str2,cnt,e) ((sizeof(*(str1)) != sizeof(*(str2))) || PetscMemcmp(str1,str2,(size_t)(cnt)*sizeof(*(str1)),e))

/*MC
   PetscArraymove - Copies from one array in memory to another, the arrays may overlap. Use PetscArraycpy() when the arrays
                    do not overlap

   Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscArraymove(anytype *str1,const anytype *str2,size_t cnt)

   Not Collective

   Input Parameters:
+  str1 - First array
.  str2 - Second array
-  cnt  - Count of the array, not in bytes, but number of entries in the arrays

   Level: intermediate

   Note:
   This routine is a preferred replacement to PetscMemmove()
   The arrays must be of the same type

.seealso: PetscMemcpy(), PetscMemcmp(), PetscArrayzero(), PetscMemzero(), PetscArraycpy(), PetscMemmove(), PetscArraycmp(), PetscStrallocpy()
M*/
#define  PetscArraymove(str1,str2,cnt) ((sizeof(*(str1)) != sizeof(*(str2))) || PetscMemmove(str1,str2,(size_t)(cnt)*sizeof(*(str1))))

/*MC
   PetscArraycpy - Copies from one array in memory to another

   Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscArraycpy(anytype *str1,const anytype *str2,size_t cnt)

   Not Collective

   Input Parameters:
+  str1 - First array (destination)
.  str2 - Second array (source)
-  cnt  - Count of the array, not in bytes, but number of entries in the arrays

   Level: intermediate

   Note:
   This routine is a preferred replacement to PetscMemcpy()
   The arrays must be of the same type

.seealso: PetscMemcpy(), PetscMemcmp(), PetscArrayzero(), PetscMemzero(), PetscArraymove(), PetscMemmove(), PetscArraycmp(), PetscStrallocpy()
M*/
#define  PetscArraycpy(str1,str2,cnt) ((sizeof(*(str1)) != sizeof(*(str2))) || PetscMemcpy(str1,str2,(size_t)(cnt)*sizeof(*(str1))))

/*MC
   PetscArrayzero - Zeros an array in memory.

   Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscArrayzero(anytype *str1,size_t cnt)

   Not Collective

   Input Parameters:
+  str1 - array
-  cnt  - Count of the array, not in bytes, but number of entries in the array

   Level: intermediate

   Note:
   This routine is a preferred replacement to PetscMemzero()

.seealso: PetscMemcpy(), PetscMemcmp(), PetscMemzero(), PetscArraycmp(), PetscArraycpy(), PetscMemmove(), PetscStrallocpy(), PetscArraymove()
M*/
#define  PetscArrayzero(str1,cnt) PetscMemzero(str1,(size_t)(cnt)*sizeof(*(str1)))

/*MC
   PetscPrefetchBlock - Prefetches a block of memory

   Synopsis:
    #include <petscsys.h>
    void PetscPrefetchBlock(const anytype *a,size_t n,int rw,int t)

   Not Collective

   Input Parameters:
+  a - pointer to first element to fetch (any type but usually PetscInt or PetscScalar)
.  n - number of elements to fetch
.  rw - 1 if the memory will be written to, otherwise 0 (ignored by many processors)
-  t - temporal locality (PETSC_PREFETCH_HINT_{NTA,T0,T1,T2}), see note

   Level: developer

   Notes:
   The last two arguments (rw and t) must be compile-time constants.

   Adopting Intel's x86/x86-64 conventions, there are four levels of temporal locality.  Not all architectures offer
   equivalent locality hints, but the following macros are always defined to their closest analogue.
+  PETSC_PREFETCH_HINT_NTA - Non-temporal.  Prefetches directly to L1, evicts to memory (skips higher level cache unless it was already there when prefetched).
.  PETSC_PREFETCH_HINT_T0 - Fetch to all levels of cache and evict to the closest level.  Use this when the memory will be reused regularly despite necessary eviction from L1.
.  PETSC_PREFETCH_HINT_T1 - Fetch to level 2 and higher (not L1).
-  PETSC_PREFETCH_HINT_T2 - Fetch to high-level cache only.  (On many systems, T0 and T1 are equivalent.)

   This function does nothing on architectures that do not support prefetch and never errors (even if passed an invalid
   address).

M*/
#define PetscPrefetchBlock(a,n,rw,t) do {                               \
    const char *_p = (const char*)(a),*_end = (const char*)((a)+(n));   \
    for ( ; _p < _end; _p += PETSC_LEVEL1_DCACHE_LINESIZE) PETSC_Prefetch(_p,(rw),(t)); \
  } while (0)

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

#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJPERM)
#define PETSC_USE_FORTRAN_KERNEL_MULTAIJPERM
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
#define PETSC_USE_FORTRAN_KERNEL_MULTAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
#define PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_NORM)
#define PETSC_USE_FORTRAN_KERNEL_NORM
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)
#define PETSC_USE_FORTRAN_KERNEL_MAXPY
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
#define PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
#define PETSC_USE_FORTRAN_KERNEL_RELAXAIJ
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
#  define EXTERN_C_BEGIN extern "C" {
#  define EXTERN_C_END }
#else
#  define EXTERN_C_BEGIN
#  define EXTERN_C_END
#endif

/* --------------------------------------------------------------------*/

/*MC
    MPI_Comm - the basic object used by MPI to determine which processes are involved in a
        communication

   Level: beginner

   Note: This manual page is a place-holder because MPICH does not have a manual page for MPI_Comm

.seealso: PETSC_COMM_WORLD, PETSC_COMM_SELF
M*/

#if defined(PETSC_HAVE_MPIIO)
PETSC_EXTERN PetscErrorCode MPIU_File_write_all(MPI_File,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
PETSC_EXTERN PetscErrorCode MPIU_File_read_all(MPI_File,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
PETSC_EXTERN PetscErrorCode MPIU_File_write_at(MPI_File,MPI_Offset,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
PETSC_EXTERN PetscErrorCode MPIU_File_read_at(MPI_File,MPI_Offset,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
PETSC_EXTERN PetscErrorCode MPIU_File_write_at_all(MPI_File,MPI_Offset,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
PETSC_EXTERN PetscErrorCode MPIU_File_read_at_all(MPI_File,MPI_Offset,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
#endif

/* the following petsc_static_inline require petscerror.h */

/* Limit MPI to 32-bits */
#define PETSC_MPI_INT_MAX  2147483647
#define PETSC_MPI_INT_MIN -2147483647
/* Limit BLAS to 32-bits */
#define PETSC_BLAS_INT_MAX  2147483647
#define PETSC_BLAS_INT_MIN -2147483647

/*@C
    PetscIntCast - casts a PetscInt64 (which is 64 bits in size) to a PetscInt (which may be 32 bits in size), generates an
         error if the PetscInt is not large enough to hold the number.

   Not Collective

   Input Parameter:
.     a - the PetscInt64 value

   Output Parameter:
.     b - the resulting PetscInt value

   Level: advanced

   Notes: If integers needed for the applications are too large to fit in 32 bit ints you can ./configure using --with-64-bit-indices to make PetscInt use 64 bit ints

   Not available from Fortran

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscMPIIntCast(), PetscBLASIntCast()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscIntCast(PetscInt64 a,PetscInt *b)
{
  PetscFunctionBegin;
#if !defined(PETSC_USE_64BIT_INDICES)
  if ((a) > PETSC_MAX_INT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Integer too long for PetscInt, you may need to ./configure using --with-64-bit-indices");
#endif
  *b =  (PetscInt)(a);
  PetscFunctionReturn(0);
}

/*@C
    PetscBLASIntCast - casts a PetscInt (which may be 64 bits in size) to a PetscBLASInt (which may be 32 bits in size), generates an
         error if the PetscBLASInt is not large enough to hold the number.

   Not Collective

   Input Parameter:
.     a - the PetscInt value

   Output Parameter:
.     b - the resulting PetscBLASInt value

   Level: advanced

   Not available from Fortran

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscMPIIntCast(), PetscIntCast()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscBLASIntCast(PetscInt a,PetscBLASInt *b)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES) && !defined(PETSC_HAVE_64BIT_BLAS_INDICES)
  *b = 0;
  if ((a) > PETSC_BLAS_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for BLAS/LAPACK");
#endif
  *b =  (PetscBLASInt)(a);
  PetscFunctionReturn(0);
}

/*@C
    PetscMPIIntCast - casts a PetscInt (which may be 64 bits in size) to a PetscMPIInt (which may be 32 bits in size), generates an
         error if the PetscMPIInt is not large enough to hold the number.

   Not Collective

   Input Parameter:
.     a - the PetscInt value

   Output Parameter:
.     b - the resulting PetscMPIInt value

   Level: advanced

   Not available from Fortran

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscIntCast()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscMPIIntCast(PetscInt a,PetscMPIInt *b)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  *b = 0;
  if ((a) > PETSC_MPI_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for MPI");
#endif
  *b =  (PetscMPIInt)(a);
  PetscFunctionReturn(0);
}

#define PetscInt64Mult(a,b)   ((PetscInt64)(a))*((PetscInt64)(b))

/*@C

   PetscRealIntMultTruncate - Computes the product of a positive PetscReal and a positive PetscInt and truncates the value to slightly less than the maximal possible value

   Not Collective

   Input Parameter:
+     a - the PetscReal value
-     b - the second value

   Returns:
      the result as a PetscInt value

   Use PetscInt64Mult() to compute the product of two PetscInt as a PetscInt64
   Use PetscIntMultTruncate() to compute the product of two positive PetscInt and truncate to fit a PetscInt
   Use PetscIntMultError() to compute the product of two PetscInt if you wish to generate an error if the result will not fit in a PetscInt

   Developers Note:
   We currently assume that PetscInt addition can never overflow, this is obviously wrong but requires many more checks.

   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

   Not available from Fortran

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscInt64Mult()
@*/
PETSC_STATIC_INLINE PetscInt PetscRealIntMultTruncate(PetscReal a,PetscInt b)
{
  PetscInt64 r;

  r  =  (PetscInt64) (a*(PetscReal)b);
  if (r > PETSC_MAX_INT - 100) r = PETSC_MAX_INT - 100;
  return (PetscInt) r;
}

/*@C

   PetscIntMultTruncate - Computes the product of two positive PetscInt and truncates the value to slightly less than the maximal possible value

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Returns:
      the result as a PetscInt value

   Use PetscInt64Mult() to compute the product of two PetscInt as a PetscInt64
   Use PetscRealIntMultTruncate() to compute the product of a PetscReal and a PetscInt and truncate to fit a PetscInt
   Use PetscIntMultError() to compute the product of two PetscInt if you wish to generate an error if the result will not fit in a PetscInt

   Not available from Fortran

   Developers Note: We currently assume that PetscInt addition can never overflow, this is obviously wrong but requires many more checks.

   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscInt64Mult()
@*/
PETSC_STATIC_INLINE PetscInt PetscIntMultTruncate(PetscInt a,PetscInt b)
{
  PetscInt64 r;

  r  =  PetscInt64Mult(a,b);
  if (r > PETSC_MAX_INT - 100) r = PETSC_MAX_INT - 100;
  return (PetscInt) r;
}

/*@C

   PetscIntSumTruncate - Computes the sum of two positive PetscInt and truncates the value to slightly less than the maximal possible value

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Returns:
     the result as a PetscInt value

   Use PetscInt64Mult() to compute the product of two PetscInt as a PetscInt64
   Use PetscRealIntMultTruncate() to compute the product of a PetscReal and a PetscInt and truncate to fit a PetscInt
   Use PetscIntMultError() to compute the product of two PetscInt if you wish to generate an error if the result will not fit in a PetscInt

   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

   Not available from Fortran

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscInt64Mult()
@*/
PETSC_STATIC_INLINE PetscInt PetscIntSumTruncate(PetscInt a,PetscInt b)
{
  PetscInt64 r;

  r  =  ((PetscInt64)a) + ((PetscInt64)b);
  if (r > PETSC_MAX_INT - 100) r = PETSC_MAX_INT - 100;
  return (PetscInt) r;
}

/*@C

   PetscIntMultError - Computes the product of two positive PetscInt and generates an error with overflow.

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Output Parameter:ma
.     result - the result as a PetscInt value, or NULL if you do not want the result, you just want to check if it overflows

   Use PetscInt64Mult() to compute the product of two 32 bit PetscInt and store in a PetscInt64
   Use PetscIntMultTruncate() to compute the product of two PetscInt and truncate it to fit in a PetscInt

   Not available from Fortran

   Developers Note: We currently assume that PetscInt addition does not overflow, this is obviously wrong but requires many more checks.

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscIntMult64(), PetscIntSumError()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscIntMultError(PetscInt a,PetscInt b,PetscInt *result)
{
  PetscInt64 r;

  PetscFunctionBegin;
  r  =  PetscInt64Mult(a,b);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (r > PETSC_MAX_INT) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Product of two integer %d %d overflow, you must ./configure PETSc with --with-64-bit-indices for the case you are running",a,b);
#endif
  if (result) *result = (PetscInt) r;
  PetscFunctionReturn(0);
}

/*@C

   PetscIntSumError - Computes the sum of two positive PetscInt and generates an error with overflow.

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Output Parameter:ma
.     c - the result as a PetscInt value,  or NULL if you do not want the result, you just want to check if it overflows

   Use PetscInt64Mult() to compute the product of two 32 bit PetscInt and store in a PetscInt64
   Use PetscIntMultTruncate() to compute the product of two PetscInt and truncate it to fit in a PetscInt

   Not available from Fortran

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscInt64Mult()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscIntSumError(PetscInt a,PetscInt b,PetscInt *result)
{
  PetscInt64 r;

  PetscFunctionBegin;
  r  =  ((PetscInt64)a) + ((PetscInt64)b);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (r > PETSC_MAX_INT) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sum of two integer %d %d overflow, you must ./configure PETSc with --with-64-bit-indices for the case you are running",a,b);
#endif
  if (result) *result = (PetscInt) r;
  PetscFunctionReturn(0);
}

/*
     The IBM include files define hz, here we hide it so that it may be used as a regular user variable.
*/
#if defined(hz)
#  undef hz
#endif

#include <limits.h>

/* The number of bits in a byte */

#define PETSC_BITS_PER_BYTE CHAR_BIT

/*  For arrays that contain filenames or paths */

#if defined(PETSC_HAVE_SYS_PARAM_H)
#  include <sys/param.h>
#endif
#if defined(PETSC_HAVE_SYS_TYPES_H)
#  include <sys/types.h>
#endif
#if defined(MAXPATHLEN)
#  define PETSC_MAX_PATH_LEN MAXPATHLEN
#elif defined(MAX_PATH)
#  define PETSC_MAX_PATH_LEN MAX_PATH
#elif defined(_MAX_PATH)
#  define PETSC_MAX_PATH_LEN _MAX_PATH
#else
#  define PETSC_MAX_PATH_LEN 4096
#endif

/*MC

    PETSC_VERSION - This manual page provides information about how PETSc documents and uses its version information. This information is available to both C/C++
                    and Fortran compilers when petscsys.h is included.


    The current PETSc version and the API for accessing it are defined in petscversion.h

    The complete version number is given as the triple  PETSC_VERSION_MAJOR.PETSC_VERSION_MINOR.PETSC_VERSION_SUBMINOR (in short hand x.y.z)

    A change in the minor version number (y) indicates possible/likely changes in the PETSc API. Note this is different than with the semantic versioning convention
    where only a change in the major version number (x) indicates a change in the API.

    A subminor greater than zero indicates a patch release. Version x.y.z maintains source and binary compatibility with version x.y.w for all z and w

    Use the macros PETSC_VERSION_EQ(x,y,z), PETSC_VERSION_LT(x,y,z), PETSC_VERSION_LE(x,y,z), PETSC_VERSION_GT(x,y,z),
    PETSC_VERSION_GE(x,y,z) to determine if the current version is equal to, less than, less than or equal to, greater than or greater than or equal to a given
    version number (x.y.z).

    PETSC_RELEASE_DATE is the date the x.y version was released (i.e. the version before any patch releases)

    PETSC_VERSION_DATE is the date the x.y.z version was released

    PETSC_VERSION_GIT is the last git commit to the repository given in the form vx.y.z-wwwww

    PETSC_VERSION_DATE_GIT is the date of the last git commit to the repository

    Level: intermediate

    PETSC_VERSION_() and PETSC_VERSION_PATCH are deprecated and will eventually be removed. For several releases PETSC_VERSION_PATCH is always 0

M*/

/*MC

    UsingFortran - To use PETSc with Fortran you must use both PETSc include files and modules. At the beginning
      of every function and module definition you need something like

$
$#include "petsc/finclude/petscXXX.h"
$         use petscXXX

     You can declare PETSc variables using either of the following.

$    XXX variablename
$    type(tXXX) variablename

    For example,

$#include "petsc/finclude/petscvec.h"
$         use petscvec
$
$    Vec b
$    type(tVec) x

    Level: beginner

M*/

PETSC_EXTERN PetscErrorCode PetscGetArchType(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetHostName(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetUserName(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetProgramName(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscSetProgramName(const char[]);
PETSC_EXTERN PetscErrorCode PetscGetDate(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetVersion(char[], size_t);
PETSC_EXTERN PetscErrorCode PetscGetVersionNumber(PetscInt*,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode PetscSortedInt(PetscInt,const PetscInt[],PetscBool*);
PETSC_EXTERN PetscErrorCode PetscSortedMPIInt(PetscInt,const PetscMPIInt[],PetscBool*);
PETSC_EXTERN PetscErrorCode PetscSortedReal(PetscInt,const PetscReal[],PetscBool*);
PETSC_EXTERN PetscErrorCode PetscSortInt(PetscInt,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortReverseInt(PetscInt,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortedRemoveDupsInt(PetscInt*,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsInt(PetscInt*,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscCheckDupsInt(PetscInt,const PetscInt[],PetscBool*);
PETSC_EXTERN PetscErrorCode PetscFindInt(PetscInt, PetscInt, const PetscInt[], PetscInt*);
PETSC_EXTERN PetscErrorCode PetscFindMPIInt(PetscMPIInt, PetscInt, const PetscMPIInt[], PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSortIntWithPermutation(PetscInt,const PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortStrWithPermutation(PetscInt,const char*[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithArrayPair(PetscInt,PetscInt[],PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortMPIInt(PetscInt,PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsMPIInt(PetscInt*,PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortMPIIntWithArray(PetscMPIInt,PetscMPIInt[],PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortMPIIntWithIntArray(PetscMPIInt,PetscMPIInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithScalarArray(PetscInt,PetscInt[],PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithDataArray(PetscInt,PetscInt[],void*,size_t,void*);
PETSC_EXTERN PetscErrorCode PetscSortReal(PetscInt,PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscSortRealWithArrayInt(PetscInt,PetscReal[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRealWithPermutation(PetscInt,const PetscReal[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsReal(PetscInt*,PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscFindReal(PetscReal,PetscInt,const PetscReal[],PetscReal,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSortSplit(PetscInt,PetscInt,PetscScalar[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortSplitReal(PetscInt,PetscInt,PetscReal[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscProcessTree(PetscInt,const PetscBool [],const PetscInt[],PetscInt*,PetscInt**,PetscInt**,PetscInt**,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscMergeIntArrayPair(PetscInt,const PetscInt[],const PetscInt[],PetscInt,const PetscInt[],const PetscInt[],PetscInt*,PetscInt**,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscMergeIntArray(PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscInt*,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscMergeMPIIntArray(PetscInt,const PetscMPIInt[],PetscInt,const PetscMPIInt[],PetscInt*,PetscMPIInt**);

PETSC_EXTERN PetscErrorCode PetscParallelSortedInt(MPI_Comm, PetscInt, const PetscInt[], PetscBool *);

PETSC_EXTERN PetscErrorCode PetscSetDisplay(void);
PETSC_EXTERN PetscErrorCode PetscGetDisplay(char[],size_t);

/*J
    PetscRandomType - String with the name of a PETSc randomizer

   Level: beginner

   Notes:
   To use SPRNG or RANDOM123 you must have ./configure PETSc
   with the option --download-sprng or --download-random123

.seealso: PetscRandomSetType(), PetscRandom, PetscRandomCreate()
J*/
typedef const char* PetscRandomType;
#define PETSCRAND       "rand"
#define PETSCRAND48     "rand48"
#define PETSCSPRNG      "sprng"
#define PETSCRANDER48   "rander48"
#define PETSCRANDOM123  "random123"

/* Logging support */
PETSC_EXTERN PetscClassId PETSC_RANDOM_CLASSID;

PETSC_EXTERN PetscErrorCode PetscRandomInitializePackage(void);

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFunctionList PetscRandomList;

PETSC_EXTERN PetscErrorCode PetscRandomRegister(const char[],PetscErrorCode (*)(PetscRandom));
PETSC_EXTERN PetscErrorCode PetscRandomSetType(PetscRandom, PetscRandomType);
PETSC_EXTERN PetscErrorCode PetscRandomSetFromOptions(PetscRandom);
PETSC_EXTERN PetscErrorCode PetscRandomGetType(PetscRandom, PetscRandomType*);
PETSC_EXTERN PetscErrorCode PetscRandomViewFromOptions(PetscRandom,PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscRandomView(PetscRandom,PetscViewer);

PETSC_EXTERN PetscErrorCode PetscRandomCreate(MPI_Comm,PetscRandom*);
PETSC_EXTERN PetscErrorCode PetscRandomGetValue(PetscRandom,PetscScalar*);
PETSC_EXTERN PetscErrorCode PetscRandomGetValueReal(PetscRandom,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscRandomGetInterval(PetscRandom,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar);
PETSC_EXTERN PetscErrorCode PetscRandomSetSeed(PetscRandom,unsigned long);
PETSC_EXTERN PetscErrorCode PetscRandomGetSeed(PetscRandom,unsigned long *);
PETSC_EXTERN PetscErrorCode PetscRandomSeed(PetscRandom);
PETSC_EXTERN PetscErrorCode PetscRandomDestroy(PetscRandom*);

PETSC_EXTERN PetscErrorCode PetscGetFullPath(const char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetRelativePath(const char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetWorkingDirectory(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetRealPath(const char[],char[]);
PETSC_EXTERN PetscErrorCode PetscGetHomeDirectory(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscTestFile(const char[],char,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscTestDirectory(const char[],char,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscMkdir(const char[]);
PETSC_EXTERN PetscErrorCode PetscMkdtemp(char[]);
PETSC_EXTERN PetscErrorCode PetscRMTree(const char[]);

PETSC_STATIC_INLINE PetscBool PetscBinaryBigEndian(void) {long _petsc_v = 1; return ((char*)&_petsc_v)[0] ? PETSC_FALSE : PETSC_TRUE;}

PETSC_EXTERN PetscErrorCode PetscBinaryRead(int,void*,PetscInt,PetscInt*,PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedRead(MPI_Comm,int,void*,PetscInt,PetscInt*,PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinaryWrite(int,const void*,PetscInt,PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedWrite(MPI_Comm,int,const void*,PetscInt,PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinaryOpen(const char[],PetscFileMode,int *);
PETSC_EXTERN PetscErrorCode PetscBinaryClose(int);
PETSC_EXTERN PetscErrorCode PetscSharedTmp(MPI_Comm,PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscSharedWorkingDirectory(MPI_Comm,PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscGetTmp(MPI_Comm,char[],size_t);
PETSC_EXTERN PetscErrorCode PetscFileRetrieve(MPI_Comm,const char[],char[],size_t,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscLs(MPI_Comm,const char[],char[],size_t,PetscBool *);
#if defined(PETSC_USE_SOCKET_VIEWER)
PETSC_EXTERN PetscErrorCode PetscOpenSocket(const char[],int,int*);
#endif

PETSC_EXTERN PetscErrorCode PetscBinarySeek(int,off_t,PetscBinarySeekType,off_t*);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedSeek(MPI_Comm,int,off_t,PetscBinarySeekType,off_t*);
PETSC_EXTERN PetscErrorCode PetscByteSwap(void *,PetscDataType,PetscInt);

PETSC_EXTERN PetscErrorCode PetscSetDebugTerminal(const char[]);
PETSC_EXTERN PetscErrorCode PetscSetDebugger(const char[],PetscBool );
PETSC_EXTERN PetscErrorCode PetscSetDefaultDebugger(void);
PETSC_EXTERN PetscErrorCode PetscSetDebuggerFromString(const char*);
PETSC_EXTERN PetscErrorCode PetscAttachDebugger(void);
PETSC_EXTERN PetscErrorCode PetscStopForDebugger(void);

PETSC_EXTERN PetscErrorCode PetscGatherNumberOfMessages(MPI_Comm,const PetscMPIInt[],const PetscMPIInt[],PetscMPIInt*);
PETSC_EXTERN PetscErrorCode PetscGatherMessageLengths(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],PetscMPIInt**,PetscMPIInt**);
PETSC_EXTERN PetscErrorCode PetscGatherMessageLengths2(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],const PetscMPIInt[],PetscMPIInt**,PetscMPIInt**,PetscMPIInt**);
PETSC_EXTERN PetscErrorCode PetscPostIrecvInt(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],const PetscMPIInt[],PetscInt***,MPI_Request**);
PETSC_EXTERN PetscErrorCode PetscPostIrecvScalar(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],const PetscMPIInt[],PetscScalar***,MPI_Request**);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSided(MPI_Comm,PetscMPIInt,MPI_Datatype,PetscMPIInt,const PetscMPIInt*,const void*,PetscMPIInt*,PetscMPIInt**,void*)
  PetscAttrMPIPointerWithType(6,3);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedF(MPI_Comm,PetscMPIInt,MPI_Datatype,PetscMPIInt,const PetscMPIInt[],const void*,PetscMPIInt*,PetscMPIInt**,void*,PetscMPIInt,
                                                    PetscErrorCode (*send)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,PetscMPIInt,void*,MPI_Request[],void*),
                                                    PetscErrorCode (*recv)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,void*,MPI_Request[],void*),void *ctx)
  PetscAttrMPIPointerWithType(6,3);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedFReq(MPI_Comm,PetscMPIInt,MPI_Datatype,PetscMPIInt,const PetscMPIInt[],const void*,PetscMPIInt*,PetscMPIInt**,void*,PetscMPIInt,
                                                       MPI_Request**,MPI_Request**,
                                                       PetscErrorCode (*send)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,PetscMPIInt,void*,MPI_Request[],void*),
                                                       PetscErrorCode (*recv)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,void*,MPI_Request[],void*),void *ctx)
  PetscAttrMPIPointerWithType(6,3);

PETSC_EXTERN const char *const PetscBuildTwoSidedTypes[];
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedSetType(MPI_Comm,PetscBuildTwoSidedType);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedGetType(MPI_Comm,PetscBuildTwoSidedType*);

PETSC_EXTERN PetscErrorCode PetscSSEIsEnabled(MPI_Comm,PetscBool*,PetscBool*);

PETSC_EXTERN MPI_Comm PetscObjectComm(PetscObject);

PETSC_EXTERN const char *const PetscSubcommTypes[];

struct _n_PetscSubcomm {
  MPI_Comm         parent;           /* parent communicator */
  MPI_Comm         dupparent;        /* duplicate parent communicator, under which the processors of this subcomm have contiguous rank */
  MPI_Comm         child;            /* the sub-communicator */
  PetscMPIInt      n;                /* num of subcommunicators under the parent communicator */
  PetscMPIInt      color;            /* color of processors belong to this communicator */
  PetscMPIInt      *subsize;         /* size of subcommunicator[color] */
  PetscSubcommType type;
  char             *subcommprefix;
};

PETSC_STATIC_INLINE MPI_Comm PetscSubcommParent(PetscSubcomm scomm) {return scomm->parent;}
PETSC_STATIC_INLINE MPI_Comm PetscSubcommChild(PetscSubcomm scomm) {return scomm->child;}
PETSC_STATIC_INLINE MPI_Comm PetscSubcommContiguousParent(PetscSubcomm scomm) {return scomm->dupparent;}
PETSC_EXTERN PetscErrorCode PetscSubcommCreate(MPI_Comm,PetscSubcomm*);
PETSC_EXTERN PetscErrorCode PetscSubcommDestroy(PetscSubcomm*);
PETSC_EXTERN PetscErrorCode PetscSubcommSetNumber(PetscSubcomm,PetscInt);
PETSC_EXTERN PetscErrorCode PetscSubcommSetType(PetscSubcomm,PetscSubcommType);
PETSC_EXTERN PetscErrorCode PetscSubcommSetTypeGeneral(PetscSubcomm,PetscMPIInt,PetscMPIInt);
PETSC_EXTERN PetscErrorCode PetscSubcommView(PetscSubcomm,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSubcommSetFromOptions(PetscSubcomm);
PETSC_EXTERN PetscErrorCode PetscSubcommSetOptionsPrefix(PetscSubcomm,const char[]);
PETSC_EXTERN PetscErrorCode PetscSubcommGetParent(PetscSubcomm,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PetscSubcommGetContiguousParent(PetscSubcomm,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PetscSubcommGetChild(PetscSubcomm,MPI_Comm*);

PETSC_EXTERN PetscErrorCode PetscHeapCreate(PetscInt,PetscHeap*);
PETSC_EXTERN PetscErrorCode PetscHeapAdd(PetscHeap,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PetscHeapPop(PetscHeap,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscHeapPeek(PetscHeap,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscHeapStash(PetscHeap,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PetscHeapUnstash(PetscHeap);
PETSC_EXTERN PetscErrorCode PetscHeapDestroy(PetscHeap*);
PETSC_EXTERN PetscErrorCode PetscHeapView(PetscHeap,PetscViewer);

PETSC_EXTERN PetscErrorCode PetscProcessPlacementView(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscShmCommGet(MPI_Comm,PetscShmComm*);
PETSC_EXTERN PetscErrorCode PetscShmCommGlobalToLocal(PetscShmComm,PetscMPIInt,PetscMPIInt*);
PETSC_EXTERN PetscErrorCode PetscShmCommLocalToGlobal(PetscShmComm,PetscMPIInt,PetscMPIInt*);
PETSC_EXTERN PetscErrorCode PetscShmCommGetMpiShmComm(PetscShmComm,MPI_Comm*);

/* routines to better support OpenMP multithreading needs of some PETSc third party libraries */
PETSC_EXTERN PetscErrorCode PetscOmpCtrlCreate(MPI_Comm,PetscInt,PetscOmpCtrl*);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlGetOmpComms(PetscOmpCtrl,MPI_Comm*,MPI_Comm*,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlDestroy(PetscOmpCtrl*);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlBarrier(PetscOmpCtrl);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlOmpRegionOnMasterBegin(PetscOmpCtrl);
PETSC_EXTERN PetscErrorCode PetscOmpCtrlOmpRegionOnMasterEnd(PetscOmpCtrl);

PETSC_EXTERN PetscErrorCode PetscSegBufferCreate(size_t,size_t,PetscSegBuffer*);
PETSC_EXTERN PetscErrorCode PetscSegBufferDestroy(PetscSegBuffer*);
PETSC_EXTERN PetscErrorCode PetscSegBufferGet(PetscSegBuffer,size_t,void*);
PETSC_EXTERN PetscErrorCode PetscSegBufferExtractAlloc(PetscSegBuffer,void*);
PETSC_EXTERN PetscErrorCode PetscSegBufferExtractTo(PetscSegBuffer,void*);
PETSC_EXTERN PetscErrorCode PetscSegBufferExtractInPlace(PetscSegBuffer,void*);
PETSC_EXTERN PetscErrorCode PetscSegBufferGetSize(PetscSegBuffer,size_t*);
PETSC_EXTERN PetscErrorCode PetscSegBufferUnuse(PetscSegBuffer,size_t);


/* Type-safe wrapper to encourage use of PETSC_RESTRICT. Does not use PetscFunctionBegin because the error handling
 * prevents the compiler from completely erasing the stub. This is called in inner loops so it has to be as fast as
 * possible. */
PETSC_STATIC_INLINE PetscErrorCode PetscSegBufferGetInts(PetscSegBuffer seg,size_t count,PetscInt *PETSC_RESTRICT *slot) {return PetscSegBufferGet(seg,count,(void**)slot);}

extern PetscOptionsHelpPrinted PetscOptionsHelpPrintedSingleton;
PETSC_EXTERN PetscErrorCode PetscOptionsHelpPrintedDestroy(PetscOptionsHelpPrinted*);
PETSC_EXTERN PetscErrorCode PetscOptionsHelpPrintedCreate(PetscOptionsHelpPrinted*);
PETSC_EXTERN PetscErrorCode PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrinted,const char*,const char*,PetscBool*);

#include <stdarg.h>
PETSC_EXTERN PetscErrorCode PetscVSNPrintf(char*,size_t,const char[],size_t*,va_list);
PETSC_EXTERN PetscErrorCode (*PetscVFPrintf)(FILE*,const char[],va_list);

PETSC_EXTERN PetscSegBuffer PetscCitationsList;

/*@C
      PetscCitationsRegister - Register a bibtex item to obtain credit for an implemented algorithm used in the code.

     Not Collective - only what is registered on rank 0 of PETSC_COMM_WORLD will be printed

     Input Parameters:
+      cite - the bibtex item, formated to displayed on multiple lines nicely
-      set - a boolean variable initially set to PETSC_FALSE; this is used to insure only a single registration of the citation

   Level: intermediate

   Not available from Fortran

     Options Database:
.     -citations [filename]   - print out the bibtex entries for the given computation
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscCitationsRegister(const char cit[],PetscBool *set)
{
  size_t         len;
  char           *vstring;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (set && *set) PetscFunctionReturn(0);
  ierr = PetscStrlen(cit,&len);CHKERRQ(ierr);
  ierr = PetscSegBufferGet(PetscCitationsList,len,&vstring);CHKERRQ(ierr);
  ierr = PetscArraycpy(vstring,cit,len);CHKERRQ(ierr);
  if (set) *set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscURLShorten(const char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveAuthorize(MPI_Comm,char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveRefresh(MPI_Comm,const char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveUpload(MPI_Comm,const char[],const char []);

PETSC_EXTERN PetscErrorCode PetscBoxAuthorize(MPI_Comm,char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscBoxRefresh(MPI_Comm,const char[],char[],char[],size_t);

PETSC_EXTERN PetscErrorCode PetscGlobusGetTransfers(MPI_Comm,const char[],char[],size_t);

PETSC_EXTERN PetscErrorCode PetscTextBelt(MPI_Comm,const char[],const char[],PetscBool*);
PETSC_EXTERN PetscErrorCode PetscTellMyCell(MPI_Comm,const char[],const char[],PetscBool*);

PETSC_EXTERN PetscErrorCode PetscPullJSONValue(const char[],const char[],char[],size_t,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscPushJSONValue(char[],const char[],const char[],size_t);


#if defined(PETSC_USE_DEBUG)
/*
   Verify that all processes in the communicator have called this from the same line of code
 */
PETSC_EXTERN PetscErrorCode PetscAllreduceBarrierCheck(MPI_Comm,PetscMPIInt,int,const char*,const char *);

/*MC
   MPIU_Allreduce - a PETSc replacement for MPI_Allreduce() that tries to determine if the call from all the MPI processes occur from the
                    same place in the PETSc code. This helps to detect bugs where different MPI processes follow different code paths
                    resulting in inconsistent and incorrect calls to MPI_Allreduce().

   Collective

   Synopsis:
     #include <petscsys.h>
     PetscErrorCode MPIU_Allreduce(void *indata,void *outdata,PetscMPIInt count,MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

   Input Parameters:
+  indata - pointer to the input data to be reduced
.  count - the number of MPI data items in indata and outdata
.  datatype - the MPI datatype, for example MPI_INT
.  op - the MPI operation, for example MPI_SUM
-  comm - the MPI communicator on which the operation occurs

   Output Parameter:
.  outdata - the reduced values

   Notes:
   In optimized mode this directly calls MPI_Allreduce()

   Level: developer

.seealso: MPI_Allreduce()
M*/
#define MPIU_Allreduce(a,b,c,d,e,fcomm) (PetscAllreduceBarrierCheck(fcomm,c,__LINE__,PETSC_FUNCTION_NAME,__FILE__) || MPI_Allreduce(a,b,c,d,e,fcomm))
#else
#define MPIU_Allreduce(a,b,c,d,e,fcomm) MPI_Allreduce(a,b,c,d,e,fcomm)
#endif

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
PETSC_EXTERN PetscErrorCode MPIU_Win_allocate_shared(MPI_Aint,PetscMPIInt,MPI_Info,MPI_Comm,void*,MPI_Win*);
PETSC_EXTERN PetscErrorCode MPIU_Win_shared_query(MPI_Win,PetscMPIInt,MPI_Aint*,PetscMPIInt*,void*);
#endif

/*
    Returned from PETSc functions that are called from MPI, such as related to attributes
*/
PETSC_EXTERN PetscMPIInt PETSC_MPI_ERROR_CLASS;
PETSC_EXTERN PetscMPIInt PETSC_MPI_ERROR_CODE;

#endif
