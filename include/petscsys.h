/*
   This is the main PETSc include file (for C and C++).  It is included by all
   other PETSc include files, so it almost never has to be specifically included.
*/
#if !defined(__PETSCSYS_H)
#define __PETSCSYS_H
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
#  define PETSC_STATIC_INLINE PETSC_CXX_STATIC_INLINE
#else
#  define PETSC_STATIC_INLINE PETSC_C_STATIC_INLINE
#endif

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
#define PETSC_EXTERN extern "C" PETSC_VISIBILITY_PUBLIC
#define PETSC_EXTERN_TYPEDEF extern "C"
#define PETSC_INTERN extern "C" PETSC_VISIBILITY_INTERNAL
#else
#define PETSC_EXTERN extern PETSC_VISIBILITY_PUBLIC
#define PETSC_EXTERN_TYPEDEF
#define PETSC_INTERN extern PETSC_VISIBILITY_INTERNAL
#endif

#include <petscversion.h>
#define PETSC_AUTHOR_INFO  "       The PETSc Team\n    petsc-maint@mcs.anl.gov\n http://www.mcs.anl.gov/petsc/\n"

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
#if !defined(OMPI_WANT_MPI_INTERFACE_WARNING)
#  define OMPI_WANT_MPI_INTERFACE_WARNING 0
#endif
#include <mpi.h>

/*
   Perform various sanity checks that the correct mpi.h is being included at compile time.
   This usually happens because
      * either an unexpected mpi.h is in the default compiler path (i.e. in /usr/include) or
      * an extra include path -I/something (which contains the unexpected mpi.h) is being passed to the compiler
*/
#if defined(PETSC_HAVE_MPIUNI)
#  if !defined(__MPIUNI_H)
#    error "PETSc was configured with --with-mpi=0 but now appears to be compiling using a different mpi.h"
#  endif
#elif defined(PETSC_HAVE_MPICH_NUMVERSION)
#  if !defined(MPICH_NUMVERSION)
#    error "PETSc was configured with MPICH but now appears to be compiling using a non-MPICH mpi.h"
#  elif MPICH_NUMVERSION != PETSC_HAVE_MPICH_NUMVERSION
#    error "PETSc was configured with one MPICH mpi.h version but now appears to be compiling using a different MPICH mpi.h version"
#  endif
#elif defined(PETSC_HAVE_OMPI_MAJOR_VERSION)
#  if !defined(OMPI_MAJOR_VERSION)
#    error "PETSc was configured with OpenMPI but now appears to be compiling using a non-OpenMPI mpi.h"
#  elif (OMPI_MAJOR_VERSION != PETSC_HAVE_OMPI_MAJOR_VERSION) || (OMPI_MINOR_VERSION != PETSC_HAVE_OMPI_MINOR_VERSION) || (OMPI_RELEASE_VERSION != PETSC_HAVE_OMPI_RELEASE_VERSION)
#    error "PETSc was configured with one OpenMPI mpi.h version but now appears to be compiling using a different OpenMPI mpi.h version"
#  endif
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

/*MC
    PetscErrorCode - datatype used for return error code from almost all PETSc functions

    Level: beginner

.seealso: CHKERRQ, SETERRQ
M*/
typedef int PetscErrorCode;

/*MC

    PetscClassId - A unique id used to identify each PETSc class.

    Notes: Use PetscClassIdRegister() to obtain a new value for a new class being created. Usually
         XXXInitializePackage() calls it for each class it defines.

    Developer Notes: Internal integer stored in the _p_PetscObject data structure.
         These are all computed by an offset from the lowest one, PETSC_SMALLEST_CLASSID.

    Level: developer

.seealso: PetscClassIdRegister(), PetscLogEventRegister(), PetscHeaderCreate()
M*/
typedef int PetscClassId;


/*MC
    PetscMPIInt - datatype used to represent 'int' parameters to MPI functions.

    Level: intermediate

    Notes: usually this is the same as PetscInt, but if PETSc was built with --with-64-bit-indices but
           standard C/Fortran integers are 32 bit then this is NOT the same as PetscInt it remains 32 bit

    PetscMPIIntCast(a,&b) checks if the given PetscInt a will fit in a PetscMPIInt, if not it
      generates a PETSC_ERR_ARG_OUTOFRANGE error.

.seealso: PetscBLASInt, PetscInt, PetscMPIIntCast()

M*/
typedef int PetscMPIInt;

/*MC
    PetscEnum - datatype used to pass enum types within PETSc functions.

    Level: intermediate

.seealso: PetscOptionsGetEnum(), PetscOptionsEnum(), PetscBagRegisterEnum()
M*/
typedef enum { ENUM_DUMMY } PetscEnum;
PETSC_EXTERN MPI_Datatype MPIU_ENUM PetscAttrMPITypeTag(PetscEnum);

#if defined(PETSC_HAVE_STDINT_H)
#include <stdint.h>
#endif
#if defined (PETSC_HAVE_INTTYPES_H)
#define __STDC_FORMAT_MACROS /* required for using PRId64 from c++ */
#include <inttypes.h>
# if !defined(PRId64)
# define PRId64 "ld"
# endif
#endif

/*MC
    PetscInt - PETSc type that represents integer - used primarily to
      represent size of arrays and indexing into arrays. Its size can be configured with the option
      --with-64-bit-indices - to be either 32bit or 64bit [default 32 bit ints]

   Level: intermediate

.seealso: PetscScalar, PetscBLASInt, PetscMPIInt
M*/
#if defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T) /* MPI_INT64_T is not guaranteed to be a macro */
typedef int64_t Petsc64bitInt;
# define MPIU_INT64 MPI_INT64_T
# define PetscInt64_FMT PRId64
#elif (PETSC_SIZEOF_LONG_LONG == 8)
typedef long long Petsc64bitInt;
# define MPIU_INT64 MPI_LONG_LONG_INT
# define PetscInt64_FMT "lld"
#elif defined(PETSC_HAVE___INT64)
typedef __int64 Petsc64bitInt;
# define MPIU_INT64 MPI_INT64_T
# define PetscInt64_FMT "ld"
#else
#error "cannot determine Petsc64bitInt type"
#endif
#if defined(PETSC_USE_64BIT_INDICES)
typedef Petsc64bitInt PetscInt;
#define MPIU_INT MPIU_INT64
#define PetscInt_FMT PetscInt64_FMT
#else
typedef int PetscInt;
#define MPIU_INT MPI_INT
#define PetscInt_FMT "d"
#endif

/*MC
    PetscBLASInt - datatype used to represent 'int' parameters to BLAS/LAPACK functions.

    Level: intermediate

    Notes: usually this is the same as PetscInt, but if PETSc was built with --with-64-bit-indices but
           standard C/Fortran integers are 32 bit then this is NOT the same as PetscInt it remains 32 bit
           (except on very rare BLAS/LAPACK implementations that support 64 bit integers see the note below).

    PetscErrorCode PetscBLASIntCast(a,&b) checks if the given PetscInt a will fit in a PetscBLASInt, if not it
      generates a PETSC_ERR_ARG_OUTOFRANGE error

    Installation Notes: The 64bit versions of MATLAB ship with BLAS and LAPACK that use 64 bit integers for sizes etc,
     if you run ./configure with the option
     --with-blas-lapack-lib=[/Applications/MATLAB_R2010b.app/bin/maci64/libmwblas.dylib,/Applications/MATLAB_R2010b.app/bin/maci64/libmwlapack.dylib]
     but you need to also use --known-64-bit-blas-indices.

        MKL also ships with 64 bit integer versions of the BLAS and LAPACK, if you select those you must also ./configure with --known-64-bit-blas-indices

     Developer Notes: Eventually ./configure should automatically determine the size of the integers used by BLAS/LAPACK.

     External packages such as hypre, ML, SuperLU etc do not provide any support for passing 64 bit integers to BLAS/LAPACK so cannot
     be used with PETSc if you have set PetscBLASInt to long int.

.seealso: PetscMPIInt, PetscInt, PetscBLASIntCast()

M*/
#if defined(PETSC_HAVE_64BIT_BLAS_INDICES)
typedef Petsc64bitInt PetscBLASInt;
#else
typedef int PetscBLASInt;
#endif

/*EC

    PetscPrecision - indicates what precision the object is using. This is currently not used.

    Level: advanced

.seealso: PetscObjectSetPrecision()
E*/
typedef enum { PETSC_PRECISION_SINGLE=4,PETSC_PRECISION_DOUBLE=8 } PetscPrecision;
PETSC_EXTERN const char *PetscPrecisions[];

/*
    For the rare cases when one needs to send a size_t object with MPI
*/
#if (PETSC_SIZEOF_SIZE_T) == (PETSC_SIZEOF_INT)
#define MPIU_SIZE_T MPI_UNSIGNED
#elif  (PETSC_SIZEOF_SIZE_T) == (PETSC_SIZEOF_LONG)
#define MPIU_SIZE_T MPI_UNSIGNED_LONG
#elif  (PETSC_SIZEOF_SIZE_T) == (PETSC_SIZEOF_LONG_LONG)
#define MPIU_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "Unknown size for size_t! Send us a bugreport at petsc-maint@mcs.anl.gov"
#endif

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

    Note: This returns the same truth value, it is only a hint to compilers that the resulting
    branch is unlikely.

    Level: advanced

.seealso: PetscLikely(), CHKERRQ
M*/

/*MC
    PetscLikely - hints the compiler that the given condition is usually TRUE

    Synopsis:
    #include <petscsys.h>
    PetscBool  PetscUnlikely(PetscBool  cond)

    Not Collective

    Input Parameters:
.   cond - condition or expression

    Note: This returns the same truth value, it is only a hint to compilers that the resulting
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

/*
    Declare extern C stuff after including external header files
*/


/*
       Basic PETSc constants
*/

/*E
    PetscBool  - Logical variable. Actually an int in C and a logical in Fortran.

   Level: beginner

   Developer Note: Why have PetscBool , why not use bool in C? The problem is that K and R C, C99 and C++ all have different mechanisms for
      boolean values. It is not easy to have a simple macro that that will work properly in all circumstances with all three mechanisms.

.seealso: PETSC_TRUE, PETSC_FALSE, PetscNot()
E*/
typedef enum { PETSC_FALSE,PETSC_TRUE } PetscBool;
PETSC_EXTERN const char *const PetscBools[];
PETSC_EXTERN MPI_Datatype MPIU_BOOL PetscAttrMPITypeTag(PetscBool);

/*
    Defines some elementary mathematics functions and constants.
*/
#include <petscmath.h>

/*E
    PetscCopyMode  - Determines how an array passed to certain functions is copied or retained

   Level: beginner

$   PETSC_COPY_VALUES - the array values are copied into new space, the user is free to reuse or delete the passed in array
$   PETSC_OWN_POINTER - the array values are NOT copied, the object takes ownership of the array and will free it later, the user cannot change or
$                       delete the array. The array MUST have been obtained with PetscMalloc(). Hence this mode cannot be used in Fortran.
$   PETSC_USE_POINTER - the array values are NOT copied, the object uses the array but does NOT take ownership of the array. The user cannot use
                        the array but the user must delete the array after the object is destroyed.

E*/
typedef enum { PETSC_COPY_VALUES, PETSC_OWN_POINTER, PETSC_USE_POINTER} PetscCopyMode;
PETSC_EXTERN const char *const PetscCopyModes[];

/*MC
    PETSC_FALSE - False value of PetscBool

    Level: beginner

    Note: Zero integer

.seealso: PetscBool, PETSC_TRUE
M*/

/*MC
    PETSC_TRUE - True value of PetscBool

    Level: beginner

    Note: Nonzero integer

.seealso: PetscBool, PETSC_FALSE
M*/

/*MC
    PETSC_NULL - standard way of passing in a null or array or pointer. This is deprecated in C/C++ simply use NULL

   Level: beginner

   Notes: accepted by many PETSc functions to not set a parameter and instead use
          some default

          This macro does not exist in Fortran; you must use PETSC_NULL_INTEGER,
          PETSC_NULL_DOUBLE_PRECISION, PETSC_NULL_FUNCTION, PETSC_NULL_OBJECT etc

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_NULL           NULL

/*MC
    PETSC_IGNORE - same as NULL, means PETSc will ignore this argument

   Level: beginner

   Note: accepted by many PETSc functions to not set a parameter and instead use
          some default

   Fortran Notes: This macro does not exist in Fortran; you must use PETSC_NULL_INTEGER,
          PETSC_NULL_DOUBLE_PRECISION etc

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_NULL, PETSC_DETERMINE

M*/
#define PETSC_IGNORE         NULL

/*MC
    PETSC_DECIDE - standard way of passing in integer or floating point parameter
       where you wish PETSc to use the default.

   Level: beginner

.seealso: PETSC_NULL, PETSC_DEFAULT, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_DECIDE  -1

/*MC
    PETSC_DETERMINE - standard way of passing in integer or floating point parameter
       where you wish PETSc to compute the required value.

   Level: beginner

   Developer Note: I would like to use const PetscInt PETSC_DETERMINE = PETSC_DECIDE; but for
     some reason this is not allowed by the standard even though PETSC_DECIDE is a constant value.

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_IGNORE, VecSetSizes()

M*/
#define PETSC_DETERMINE PETSC_DECIDE

/*MC
    PETSC_DEFAULT - standard way of passing in integer or floating point parameter
       where you wish PETSc to use the default.

   Level: beginner

   Fortran Notes: You need to use PETSC_DEFAULT_INTEGER or PETSC_DEFAULT_REAL.

.seealso: PETSC_DECIDE, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_DEFAULT  -2

/*MC
    PETSC_COMM_WORLD - the equivalent of the MPI_COMM_WORLD communicator which represents
           all the processs that PETSc knows about.

   Level: beginner

   Notes: By default PETSC_COMM_WORLD and MPI_COMM_WORLD are identical unless you wish to
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

   Notes: Do not USE/access or set this variable before PetscInitialize() has been called.

.seealso: PETSC_COMM_WORLD

M*/
#define PETSC_COMM_SELF MPI_COMM_SELF

PETSC_EXTERN PetscBool PetscBeganMPI;
PETSC_EXTERN PetscBool PetscInitializeCalled;
PETSC_EXTERN PetscBool PetscFinalizeCalled;
PETSC_EXTERN PetscBool PetscCUSPSynchronize;
PETSC_EXTERN PetscBool PetscViennaCLSynchronize;
PETSC_EXTERN PetscBool PetscCUDASynchronize;

PETSC_EXTERN PetscErrorCode PetscSetHelpVersionFunctions(PetscErrorCode (*)(MPI_Comm),PetscErrorCode (*)(MPI_Comm));
PETSC_EXTERN PetscErrorCode PetscCommDuplicate(MPI_Comm,MPI_Comm*,int*);
PETSC_EXTERN PetscErrorCode PetscCommDestroy(MPI_Comm*);

/*MC
   PetscMalloc - Allocates memory, One should use PetscMalloc1() or PetscCalloc1() usually instead of this

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

  Concepts: memory allocation

M*/
#define PetscMalloc(a,b)  ((*PetscTrMalloc)((a),__LINE__,PETSC_FUNCTION_NAME,__FILE__,(void**)(b)))

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

  Concepts: memory allocation
M*/
#define PetscAddrAlign(a) (void*)((((PETSC_UINTPTR_T)(a))+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

/*MC
   PetscMalloc1 - Allocates an array of memory aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc1(size_t m1,type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate in 1st chunk  (may be zero)

   Output Parameter:
.  r1 - memory allocated in first chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscCalloc1(), PetscMalloc2()

  Concepts: memory allocation

M*/
#define PetscMalloc1(m1,r1) PetscMalloc((m1)*sizeof(**(r1)),r1)

/*MC
   PetscCalloc1 - Allocates a cleared (zeroed) array of memory aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc1(size_t m1,type **r1)

   Not Collective

   Input Parameter:
.  m1 - number of elements to allocate in 1st chunk  (may be zero)

   Output Parameter:
.  r1 - memory allocated in first chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc1(), PetscCalloc2()

  Concepts: memory allocation

M*/
#define PetscCalloc1(m1,r1) (PetscMalloc1((m1),r1) || PetscMemzero(*(r1),(m1)*sizeof(**(r1))))

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

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscMalloc2(m1,r1,m2,r2) (PetscMalloc1((m1),(r1)) || PetscMalloc1((m2),(r2)))
#else
#define PetscMalloc2(m1,r1,m2,r2) ((((m1)+(m2)) ? (*(r2) = 0,PetscMalloc((m1)*sizeof(**(r1))+(m2)*sizeof(**(r2))+(PETSC_MEMALIGN-1),r1)) : 0) \
                                   || (*(void**)(r2) = PetscAddrAlign(*(r1)+(m1)),0) \
                                   || (!(m1) ? (*(r1) = 0,0) : 0) || (!(m2) ? (*(r2) = 0,0) : 0))
#endif

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

  Concepts: memory allocation
M*/
#define PetscCalloc2(m1,r1,m2,r2) (PetscMalloc2((m1),(r1),(m2),(r2)) || PetscMemzero(*(r1),(m1)*sizeof(**(r1))) || PetscMemzero(*(r2),(m2)*sizeof(**(r2))))

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

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscMalloc3(m1,r1,m2,r2,m3,r3) (PetscMalloc1((m1),(r1)) || PetscMalloc1((m2),(r2)) || PetscMalloc1((m3),(r3)))
#else
#define PetscMalloc3(m1,r1,m2,r2,m3,r3) ((((m1)+(m2)+(m3)) ? (*(r2) = 0,*(r3) = 0,PetscMalloc((m1)*sizeof(**(r1))+(m2)*sizeof(**(r2))+(m3)*sizeof(**(r3))+2*(PETSC_MEMALIGN-1),r1)) : 0) \
                                         || (*(void**)(r2) = PetscAddrAlign(*(r1)+(m1)),*(void**)(r3) = PetscAddrAlign(*(r2)+(m2)),0) \
                                         || (!(m1) ? (*(r1) = 0,0) : 0) || (!(m2) ? (*(r2) = 0,0) : 0) || (!(m3) ? (*(r3) = 0,0) : 0))
#endif

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

  Concepts: memory allocation
M*/
#define PetscCalloc3(m1,r1,m2,r2,m3,r3)                                 \
  (PetscMalloc3((m1),(r1),(m2),(r2),(m3),(r3))                          \
   || PetscMemzero(*(r1),(m1)*sizeof(**(r1))) || PetscMemzero(*(r2),(m2)*sizeof(**(r2))) || PetscMemzero(*(r3),(m3)*sizeof(**(r3))))

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

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscMalloc4(m1,r1,m2,r2,m3,r3,m4,r4) (PetscMalloc1((m1),(r1)) || PetscMalloc1((m2),(r2)) || PetscMalloc1((m3),(r3)) || PetscMalloc1((m4),(r4)))
#else
#define PetscMalloc4(m1,r1,m2,r2,m3,r3,m4,r4)                           \
  ((((m1)+(m2)+(m3)+(m4)) ? (*(r2) = 0, *(r3) = 0, *(r4) = 0,PetscMalloc((m1)*sizeof(**(r1))+(m2)*sizeof(**(r2))+(m3)*sizeof(**(r3))+(m4)*sizeof(**(r4))+3*(PETSC_MEMALIGN-1),r1)) : 0) \
   || (*(void**)(r2) = PetscAddrAlign(*(r1)+(m1)),*(void**)(r3) = PetscAddrAlign(*(r2)+(m2)),*(void**)(r4) = PetscAddrAlign(*(r3)+(m3)),0) \
   || (!(m1) ? (*(r1) = 0,0) : 0) || (!(m2) ? (*(r2) = 0,0) : 0) || (!(m3) ? (*(r3) = 0,0) : 0) || (!(m4) ? (*(r4) = 0,0) : 0))
#endif

/*MC
   PetscCalloc4 - Allocates 4 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc4(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4)

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

  Concepts: memory allocation

M*/
#define PetscCalloc4(m1,r1,m2,r2,m3,r3,m4,r4)                           \
  (PetscMalloc4(m1,r1,m2,r2,m3,r3,m4,r4)                                \
   || PetscMemzero(*(r1),(m1)*sizeof(**(r1))) || PetscMemzero(*(r2),(m2)*sizeof(**(r2))) || PetscMemzero(*(r3),(m3)*sizeof(**(r3))) \
   || PetscMemzero(*(r4),(m4)*sizeof(**(r4))))

/*MC
   PetscMalloc5 - Allocates 5 arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc5(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
-  m5 - number of elements to allocate in 5th chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
-  r5 - memory allocated in fifth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc5(), PetscFree5()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscMalloc5(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5) (PetscMalloc1((m1),(r1)) || PetscMalloc1((m2),(r2)) || PetscMalloc1((m3),(r3)) || PetscMalloc1((m4),(r4)) || PetscMalloc1((m5),(r5)))
#else
#define PetscMalloc5(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5)      \
  ((((m1)+(m2)+(m3)+(m4)+(m5)) ? (*(r2) = 0, *(r3) = 0, *(r4) = 0,*(r5) = 0,PetscMalloc((m1)*sizeof(**(r1))+(m2)*sizeof(**(r2))+(m3)*sizeof(**(r3))+(m4)*sizeof(**(r4))+(m5)*sizeof(**(r5))+4*(PETSC_MEMALIGN-1),r1)) : 0) \
   || (*(void**)(r2) = PetscAddrAlign(*(r1)+(m1)),*(void**)(r3) = PetscAddrAlign(*(r2)+(m2)),*(void**)(r4) = PetscAddrAlign(*(r3)+(m3)),*(void**)(r5) = PetscAddrAlign(*(r4)+(m4)),0) \
   || (!(m1) ? (*(r1) = 0,0) : 0) || (!(m2) ? (*(r2) = 0,0) : 0) || (!(m3) ? (*(r3) = 0,0) : 0) || (!(m4) ? (*(r4) = 0,0) : 0) || (!(m5) ? (*(r5) = 0,0) : 0))
#endif

/*MC
   PetscCalloc5 - Allocates 5 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc5(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
-  m5 - number of elements to allocate in 5th chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
-  r5 - memory allocated in fifth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc5(), PetscFree5()

  Concepts: memory allocation

M*/
#define PetscCalloc5(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5)                     \
  (PetscMalloc5(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5)                          \
   || PetscMemzero(*(r1),(m1)*sizeof(**(r1))) || PetscMemzero(*(r2),(m2)*sizeof(**(r2))) || PetscMemzero(*(r3),(m3)*sizeof(**(r3))) \
   || PetscMemzero(*(r4),(m4)*sizeof(**(r4))) || PetscMemzero(*(r5),(m5)*sizeof(**(r5))))

/*MC
   PetscMalloc6 - Allocates 6 arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc6(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
-  m6 - number of elements to allocate in 6th chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
-  r6 - memory allocated in sixth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc6(), PetscFree3(), PetscFree4(), PetscFree5(), PetscFree6()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscMalloc6(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6) (PetscMalloc1((m1),(r1)) || PetscMalloc1((m2),(r2)) || PetscMalloc1((m3),(r3)) || PetscMalloc1((m4),(r4)) || PetscMalloc1((m5),(r5)) || PetscMalloc1((m6),(r6)))
#else
#define PetscMalloc6(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6) \
  ((((m1)+(m2)+(m3)+(m4)+(m5)+(m6)) ? (*(r2) = 0, *(r3) = 0, *(r4) = 0,*(r5) = 0,*(r6) = 0,PetscMalloc((m1)*sizeof(**(r1))+(m2)*sizeof(**(r2))+(m3)*sizeof(**(r3))+(m4)*sizeof(**(r4))+(m5)*sizeof(**(r5))+(m6)*sizeof(**(r6))+5*(PETSC_MEMALIGN-1),r1)) : 0) \
   || (*(void**)(r2) = PetscAddrAlign(*(r1)+(m1)),*(void**)(r3) = PetscAddrAlign(*(r2)+(m2)),*(void**)(r4) = PetscAddrAlign(*(r3)+(m3)),*(void**)(r5) = PetscAddrAlign(*(r4)+(m4)),*(void**)(r6) = PetscAddrAlign(*(r5)+(m5)),0) \
   || (!(m1) ? (*(r1) = 0,0) : 0) || (!(m2) ? (*(r2) = 0,0) : 0) || (!(m3) ? (*(r3) = 0,0) : 0) || (!(m4) ? (*(r4) = 0,0) : 0) || (!(m5) ? (*(r5) = 0,0) : 0) || (!(m6) ? (*(r6) = 0,0) : 0))
#endif

/*MC
   PetscCalloc6 - Allocates 6 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc6(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
-  m6 - number of elements to allocate in 6th chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
-  r6 - memory allocated in sixth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscMalloc6(), PetscFree6()

  Concepts: memory allocation
M*/
#define PetscCalloc6(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6)               \
  (PetscMalloc6(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6)                    \
   || PetscMemzero(*(r1),(m1)*sizeof(**(r1))) || PetscMemzero(*(r2),(m2)*sizeof(**(r2))) || PetscMemzero(*(r3),(m3)*sizeof(**(r3))) \
   || PetscMemzero(*(r4),(m4)*sizeof(**(r4))) || PetscMemzero(*(r5),(m5)*sizeof(**(r5))) || PetscMemzero(*(r6),(m6)*sizeof(**(r6))))

/*MC
   PetscMalloc7 - Allocates 7 arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscMalloc7(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6,size_t m7,type **r7)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
-  m7 - number of elements to allocate in 7th chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
.  r6 - memory allocated in sixth chunk
-  r7 - memory allocated in seventh chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscCalloc7(), PetscFree7()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscMalloc7(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6,m7,r7) (PetscMalloc1((m1),(r1)) || PetscMalloc1((m2),(r2)) || PetscMalloc1((m3),(r3)) || PetscMalloc1((m4),(r4)) || PetscMalloc1((m5),(r5)) || PetscMalloc1((m6),(r6)) || PetscMalloc1((m7),(r7)))
#else
#define PetscMalloc7(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6,m7,r7) \
  ((((m1)+(m2)+(m3)+(m4)+(m5)+(m6)+(m7)) ? (*(r2) = 0, *(r3) = 0, *(r4) = 0,*(r5) = 0,*(r6) = 0,*(r7) = 0,PetscMalloc((m1)*sizeof(**(r1))+(m2)*sizeof(**(r2))+(m3)*sizeof(**(r3))+(m4)*sizeof(**(r4))+(m5)*sizeof(**(r5))+(m6)*sizeof(**(r6))+(m7)*sizeof(**(r7))+6*(PETSC_MEMALIGN-1),r1)) : 0) \
   || (*(void**)(r2) = PetscAddrAlign(*(r1)+(m1)),*(void**)(r3) = PetscAddrAlign(*(r2)+(m2)),*(void**)(r4) = PetscAddrAlign(*(r3)+(m3)),*(void**)(r5) = PetscAddrAlign(*(r4)+(m4)),*(void**)(r6) = PetscAddrAlign(*(r5)+(m5)),*(void**)(r7) = PetscAddrAlign(*(r6)+(m6)),0) \
   || (!(m1) ? (*(r1) = 0,0) : 0) || (!(m2) ? (*(r2) = 0,0) : 0) || (!(m3) ? (*(r3) = 0,0) : 0) || (!(m4) ? (*(r4) = 0,0) : 0) || (!(m5) ? (*(r5) = 0,0) : 0) || (!(m6) ? (*(r6) = 0,0) : 0) || (!(m7) ? (*(r7) = 0,0) : 0))
#endif

/*MC
   PetscCalloc7 - Allocates 7 cleared (zeroed) arrays of memory, all aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscCalloc7(size_t m1,type **r1,size_t m2,type **r2,size_t m3,type **r3,size_t m4,type **r4,size_t m5,type **r5,size_t m6,type **r6,size_t m7,type **r7)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
-  m7 - number of elements to allocate in 7th chunk  (may be zero)

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
.  r6 - memory allocated in sixth chunk
-  r7 - memory allocated in seventh chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscMalloc7(), PetscFree7()

  Concepts: memory allocation
M*/
#define PetscCalloc7(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6,m7,r7)         \
  (PetscMalloc7(m1,r1,m2,r2,m3,r3,m4,r4,m5,r5,m6,r6,m7,r7)              \
   || PetscMemzero(*(r1),(m1)*sizeof(**(r1))) || PetscMemzero(*(r2),(m2)*sizeof(**(r2))) || PetscMemzero(*(r3),(m3)*sizeof(**(r3))) \
   || PetscMemzero(*(r4),(m4)*sizeof(**(r4))) || PetscMemzero(*(r5),(m5)*sizeof(**(r5))) || PetscMemzero(*(r6),(m6)*sizeof(**(r6))) \
   || PetscMemzero(*(r7),(m7)*sizeof(**(r7))))

/*MC
   PetscNew - Allocates memory of a particular type, zeros the memory! Aligned to PETSC_MEMALIGN

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscNew(type **result)

   Not Collective

   Output Parameter:
.  result - memory allocated, sized to match pointer type

   Level: beginner

.seealso: PetscFree(), PetscMalloc(), PetscNewLog()

  Concepts: memory allocation

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

.seealso: PetscFree(), PetscMalloc(), PetscNew(), PetscLogObjectMemory()

  Concepts: memory allocation

M*/
#define PetscNewLog(o,b) (PetscNew((b)) || PetscLogObjectMemory((PetscObject)o,sizeof(**(b))))

/*MC
   PetscFree - Frees memory

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree(void *memory)

   Not Collective

   Input Parameter:
.   memory - memory to free (the pointer is ALWAYS set to 0 upon sucess)

   Level: beginner

   Notes:
   Memory must have been obtained with PetscNew() or PetscMalloc().
   It is safe to call PetscFree() on a NULL pointer.

.seealso: PetscNew(), PetscMalloc(), PetscFreeVoid()

  Concepts: memory allocation

M*/
#define PetscFree(a)   ((*PetscTrFree)((void*)(a),__LINE__,PETSC_FUNCTION_NAME,__FILE__) || ((a) = 0,0))

/*MC
   PetscFreeVoid - Frees memory

   Synopsis:
    #include <petscsys.h>
   void PetscFreeVoid(void *memory)

   Not Collective

   Input Parameter:
.   memory - memory to free

   Level: beginner

   Notes: This is different from PetscFree() in that no error code is returned

.seealso: PetscFree(), PetscNew(), PetscMalloc()

  Concepts: memory allocation

M*/
#define PetscFreeVoid(a) ((*PetscTrFree)((a),__LINE__,PETSC_FUNCTION_NAME,__FILE__),(a) = 0)


/*MC
   PetscFree2 - Frees 2 chunks of memory obtained with PetscMalloc2()

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree2(void *memory1,void *memory2)

   Not Collective

   Input Parameter:
+   memory1 - memory to free
-   memory2 - 2nd memory to free

   Level: developer

   Notes: Memory must have been obtained with PetscMalloc2()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscFree2(m1,m2)   (PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree2(m1,m2)   ((m1) ? ((m2)=0,PetscFree(m1)) : ((m1)=0,PetscFree(m2)))
#endif

/*MC
   PetscFree3 - Frees 3 chunks of memory obtained with PetscMalloc3()

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree3(void *memory1,void *memory2,void *memory3)

   Not Collective

   Input Parameter:
+   memory1 - memory to free
.   memory2 - 2nd memory to free
-   memory3 - 3rd memory to free

   Level: developer

   Notes: Memory must have been obtained with PetscMalloc3()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscFree3(m1,m2,m3)   (PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree3(m1,m2,m3)   ((m1) ? ((m3)=0,(m2)=0,PetscFree(m1)) : ((m2) ? ((m3)=0,(m1)=0,PetscFree(m2)) : ((m2)=0,(m1)=0,PetscFree(m3))))
#endif

/*MC
   PetscFree4 - Frees 4 chunks of memory obtained with PetscMalloc4()

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree4(void *m1,void *m2,void *m3,void *m4)

   Not Collective

   Input Parameter:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
-   m4 - 4th memory to free

   Level: developer

   Notes: Memory must have been obtained with PetscMalloc4()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscFree4(m1,m2,m3,m4)   (PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree4(m1,m2,m3,m4)   ((m1) ? ((m4)=0,(m3)=0,(m2)=0,PetscFree(m1)) : ((m2) ? ((m4)=0,(m3)=0,(m1)=0,PetscFree(m2)) : ((m3) ? ((m4)=0,(m2)=0,(m1)=0,PetscFree(m3)) : ((m3)=0,(m2)=0,(m1)=0,PetscFree(m4)))))
#endif

/*MC
   PetscFree5 - Frees 5 chunks of memory obtained with PetscMalloc5()

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree5(void *m1,void *m2,void *m3,void *m4,void *m5)

   Not Collective

   Input Parameter:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
-   m5 - 5th memory to free

   Level: developer

   Notes: Memory must have been obtained with PetscMalloc5()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscFree5(m1,m2,m3,m4,m5)   (PetscFree(m5) || PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree5(m1,m2,m3,m4,m5)   ((m1) ? ((m5)=0,(m4)=0,(m3)=0,(m2)=0,PetscFree(m1)) : ((m2) ? ((m5)=0,(m4)=0,(m3)=0,(m1)=0,PetscFree(m2)) : ((m3) ? ((m5)=0,(m4)=0,(m2)=0,(m1)=0,PetscFree(m3)) : \
                                     ((m4) ? ((m5)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m4)) : ((m4)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m5))))))
#endif


/*MC
   PetscFree6 - Frees 6 chunks of memory obtained with PetscMalloc6()

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree6(void *m1,void *m2,void *m3,void *m4,void *m5,void *m6)

   Not Collective

   Input Parameter:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
.   m5 - 5th memory to free
-   m6 - 6th memory to free


   Level: developer

   Notes: Memory must have been obtained with PetscMalloc6()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5(), PetscMalloc6()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscFree6(m1,m2,m3,m4,m5,m6)   (PetscFree(m6) || PetscFree(m5) || PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree6(m1,m2,m3,m4,m5,m6)   ((m1) ? ((m6)=0,(m5)=0,(m4)=0,(m3)=0,(m2)=0,PetscFree(m1)) : ((m2) ? ((m6)=0,(m5)=0,(m4)=0,(m3)=0,(m1)=0,PetscFree(m2)) : \
                                        ((m3) ? ((m6)=0,(m5)=0,(m4)=0,(m2)=0,(m1)=0,PetscFree(m3)) : ((m4) ? ((m6)=0,(m5)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m4)) : \
                                        ((m5) ? ((m6)=0,(m4)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m5)) : ((m5)=0,(m4)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m6)))))))
#endif

/*MC
   PetscFree7 - Frees 7 chunks of memory obtained with PetscMalloc7()

   Synopsis:
    #include <petscsys.h>
   PetscErrorCode PetscFree7(void *m1,void *m2,void *m3,void *m4,void *m5,void *m6,void *m7)

   Not Collective

   Input Parameter:
+   m1 - memory to free
.   m2 - 2nd memory to free
.   m3 - 3rd memory to free
.   m4 - 4th memory to free
.   m5 - 5th memory to free
.   m6 - 6th memory to free
-   m7 - 7th memory to free


   Level: developer

   Notes: Memory must have been obtained with PetscMalloc7()

.seealso: PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5(), PetscMalloc6(),
          PetscMalloc7()

  Concepts: memory allocation

M*/
#if !defined(PETSC_USE_MALLOC_COALESCED)
#define PetscFree7(m1,m2,m3,m4,m5,m6,m7)   (PetscFree(m7) || PetscFree(m6) || PetscFree(m5) || PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree7(m1,m2,m3,m4,m5,m6,m7)   ((m1) ? ((m7)=0,(m6)=0,(m5)=0,(m4)=0,(m3)=0,(m2)=0,PetscFree(m1)) : ((m2) ? ((m7)=0,(m6)=0,(m5)=0,(m4)=0,(m3)=0,(m1)=0,PetscFree(m2)) : \
                                           ((m3) ? ((m7)=0,(m6)=0,(m5)=0,(m4)=0,(m2)=0,(m1)=0,PetscFree(m3)) : ((m4) ? ((m7)=0,(m6)=0,(m5)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m4)) : \
                                           ((m5) ? ((m7)=0,(m6)=0,(m4)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m5)) : ((m6) ? ((m7)=0,(m5)=0,(m4)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m6)) : \
                                                   ((m6)=0,(m5)=0,(m4)=0,(m3)=0,(m2)=0,(m1)=0,PetscFree(m7))))))))
#endif

PETSC_EXTERN PetscErrorCode (*PetscTrMalloc)(size_t,int,const char[],const char[],void**);
PETSC_EXTERN PetscErrorCode (*PetscTrFree)(void*,int,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscMallocSet(PetscErrorCode (*)(size_t,int,const char[],const char[],void**),PetscErrorCode (*)(void*,int,const char[],const char[]));
PETSC_EXTERN PetscErrorCode PetscMallocClear(void);

/*
    PetscLogDouble variables are used to contain double precision numbers
  that are not used in the numerical computations, but rather in logging,
  timing etc.
*/
typedef double PetscLogDouble;
#define MPIU_PETSCLOGDOUBLE MPI_DOUBLE

/*
   Routines for tracing memory corruption/bleeding with default PETSc memory allocation
*/
PETSC_EXTERN PetscErrorCode PetscMallocDump(FILE *);
PETSC_EXTERN PetscErrorCode PetscMallocDumpLog(FILE *);
PETSC_EXTERN PetscErrorCode PetscMallocGetCurrentUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocGetMaximumUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMallocDebug(PetscBool);
PETSC_EXTERN PetscErrorCode PetscMallocGetDebug(PetscBool*);
PETSC_EXTERN PetscErrorCode PetscMallocValidate(int,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscMallocSetDumpLog(void);
PETSC_EXTERN PetscErrorCode PetscMallocSetDumpLogThreshold(PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscMallocGetDumpLog(PetscBool*);

/*E
    PetscDataType - Used for handling different basic data types.

   Level: beginner

   Developer comment: It would be nice if we could always just use MPI Datatypes, why can we not?

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscDataTypeToMPIDataType(),
          PetscDataTypeGetSize()

E*/
typedef enum {PETSC_INT = 0,PETSC_DOUBLE = 1,PETSC_COMPLEX = 2, PETSC_LONG = 3 ,PETSC_SHORT = 4,PETSC_FLOAT = 5,
              PETSC_CHAR = 6,PETSC_BIT_LOGICAL = 7,PETSC_ENUM = 8,PETSC_BOOL=9, PETSC___FLOAT128 = 10,PETSC_OBJECT = 11, PETSC_FUNCTION = 12, PETSC_STRING = 12} PetscDataType;
PETSC_EXTERN const char *const PetscDataTypes[];

#if defined(PETSC_USE_COMPLEX)
#define  PETSC_SCALAR  PETSC_COMPLEX
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define  PETSC_SCALAR  PETSC_FLOAT
#elif defined(PETSC_USE_REAL___FLOAT128)
#define  PETSC_SCALAR  PETSC___FLOAT128
#else
#define  PETSC_SCALAR  PETSC_DOUBLE
#endif
#endif
#if defined(PETSC_USE_REAL_SINGLE)
#define  PETSC_REAL  PETSC_FLOAT
#elif defined(PETSC_USE_REAL___FLOAT128)
#define  PETSC_REAL  PETSC___FLOAT128
#else
#define  PETSC_REAL  PETSC_DOUBLE
#endif
#define  PETSC_FORTRANADDR  PETSC_LONG

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
PETSC_EXTERN PetscErrorCode PetscMemmove(void*,void *,size_t);
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
PETSC_EXTERN PetscErrorCode PetscStrncat(char[],const char[],size_t);
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

/*S
    PetscToken - 'Token' used for managing tokenizing strings

  Level: intermediate

.seealso: PetscTokenCreate(), PetscTokenFind(), PetscTokenDestroy()
S*/
typedef struct _p_PetscToken* PetscToken;

PETSC_EXTERN PetscErrorCode PetscTokenCreate(const char[],const char,PetscToken*);
PETSC_EXTERN PetscErrorCode PetscTokenFind(PetscToken,char *[]);
PETSC_EXTERN PetscErrorCode PetscTokenDestroy(PetscToken*);

PETSC_EXTERN PetscErrorCode PetscEListFind(PetscInt,const char *const*,const char*,PetscInt*,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscEnumFind(const char *const*,const char*,PetscEnum*,PetscBool*);

/*
   These are MPI operations for MPI_Allreduce() etc
*/
PETSC_EXTERN MPI_Op PetscMaxSum_Op;
#if (defined(PETSC_HAVE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128)
PETSC_EXTERN MPI_Op MPIU_SUM;
#else
#define MPIU_SUM MPI_SUM
#endif
#if defined(PETSC_USE_REAL___FLOAT128)
PETSC_EXTERN MPI_Op MPIU_MAX;
PETSC_EXTERN MPI_Op MPIU_MIN;
#else
#define MPIU_MAX MPI_MAX
#define MPIU_MIN MPI_MIN
#endif
PETSC_EXTERN PetscErrorCode PetscMaxSum(MPI_Comm,const PetscInt[],PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode MPIULong_Send(void*,PetscInt,MPI_Datatype,PetscMPIInt,PetscMPIInt,MPI_Comm);
PETSC_EXTERN PetscErrorCode MPIULong_Recv(void*,PetscInt,MPI_Datatype,PetscMPIInt,PetscMPIInt,MPI_Comm);

/*S
     PetscObject - any PETSc object, PetscViewer, Mat, Vec, KSP etc

   Level: beginner

   Note: This is the base class from which all PETSc objects are derived from.

.seealso:  PetscObjectDestroy(), PetscObjectView(), PetscObjectGetName(), PetscObjectSetName(), PetscObjectReference(), PetscObjectDereference()
S*/
typedef struct _p_PetscObject* PetscObject;

/*MC
    PetscObjectId - unique integer Id for a PetscObject

    Level: developer

    Notes: Unlike pointer values, object ids are never reused.

.seealso: PetscObjectState, PetscObjectGetId()
M*/
typedef Petsc64bitInt PetscObjectId;

/*MC
    PetscObjectState - integer state for a PetscObject

    Level: developer

    Notes:
    Object state is always-increasing and (for objects that track state) can be used to determine if an object has
    changed since the last time you interacted with it.  It is 64-bit so that it will not overflow for a very long time.

.seealso: PetscObjectId, PetscObjectStateGet(), PetscObjectStateIncrease(), PetscObjectStateSet()
M*/
typedef Petsc64bitInt PetscObjectState;

/*S
     PetscFunctionList - Linked list of functions, possibly stored in dynamic libraries, accessed
      by string name

   Level: advanced

.seealso:  PetscFunctionListAdd(), PetscFunctionListDestroy(), PetscOpFlist
S*/
typedef struct _n_PetscFunctionList *PetscFunctionList;

/*E
  PetscFileMode - Access mode for a file.

  Level: beginner

$  FILE_MODE_READ - open a file at its beginning for reading
$  FILE_MODE_WRITE - open a file at its beginning for writing (will create if the file does not exist)
$  FILE_MODE_APPEND - open a file at end for writing
$  FILE_MODE_UPDATE - open a file for updating, meaning for reading and writing
$  FILE_MODE_APPEND_UPDATE - open a file for updating, meaning for reading and writing, at the end

.seealso: PetscViewerFileSetMode()
E*/
typedef enum {FILE_MODE_READ, FILE_MODE_WRITE, FILE_MODE_APPEND, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE} PetscFileMode;
extern const char *const PetscFileModes[];

/*
    Defines PETSc error handling.
*/
#include <petscerror.h>

#define PETSC_SMALLEST_CLASSID  1211211
PETSC_EXTERN PetscClassId PETSC_LARGEST_CLASSID;
PETSC_EXTERN PetscClassId PETSC_OBJECT_CLASSID;
PETSC_EXTERN PetscErrorCode PetscClassIdRegister(const char[],PetscClassId *);

/*
   Routines that get memory usage information from the OS
*/
PETSC_EXTERN PetscErrorCode PetscMemoryGetCurrentUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMemoryGetMaximumUsage(PetscLogDouble *);
PETSC_EXTERN PetscErrorCode PetscMemorySetGetMaximumUsage(void);
PETSC_EXTERN PetscErrorCode PetscMemoryTrace(const char[]);

PETSC_EXTERN PetscErrorCode PetscInfoAllow(PetscBool ,const char []);
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
PETSC_EXTERN PetscErrorCode PetscObjectSetPrecision(PetscObject,PetscPrecision);
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
PETSC_EXTERN PetscErrorCode PetscObjectTypeCompareAny(PetscObject,PetscBool*,const char[],...);
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

typedef void* PetscDLHandle;
typedef enum {PETSC_DL_DECIDE=0,PETSC_DL_NOW=1,PETSC_DL_LOCAL=2} PetscDLMode;
PETSC_EXTERN PetscErrorCode PetscDLOpen(const char[],PetscDLMode,PetscDLHandle *);
PETSC_EXTERN PetscErrorCode PetscDLClose(PetscDLHandle *);
PETSC_EXTERN PetscErrorCode PetscDLSym(PetscDLHandle,const char[],void **);

#if defined(PETSC_USE_DEBUG)
PETSC_EXTERN PetscErrorCode PetscMallocGetStack(void*,PetscStack**);
#endif
PETSC_EXTERN PetscErrorCode PetscObjectsDump(FILE*,PetscBool);

/*S
     PetscObjectList - Linked list of PETSc objects, each accessable by string name

   Level: developer

   Notes: Used by PetscObjectCompose() and PetscObjectQuery()

.seealso:  PetscObjectListAdd(), PetscObjectListDestroy(), PetscObjectListFind(), PetscObjectCompose(), PetscObjectQuery(), PetscFunctionList
S*/
typedef struct _n_PetscObjectList *PetscObjectList;

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
PETSC_EXTERN PetscErrorCode PetscFunctionListPrintTypes(MPI_Comm,FILE*,const char[],const char[],const char[],const char[],PetscFunctionList,const char[]);
PETSC_EXTERN PetscErrorCode PetscFunctionListDuplicate(PetscFunctionList,PetscFunctionList *);
PETSC_EXTERN PetscErrorCode PetscFunctionListView(PetscFunctionList,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscFunctionListGet(PetscFunctionList,const char ***,int*);

/*S
     PetscDLLibrary - Linked list of dynamics libraries to search for functions

   Level: advanced

.seealso:  PetscDLLibraryOpen()
S*/
typedef struct _n_PetscDLLibrary *PetscDLLibrary;
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

/*
    PetscNot - negates a logical type value and returns result as a PetscBool

    Notes: This is useful in cases like
$     int        *a;
$     PetscBool  flag = PetscNot(a)
     where !a would not return a PetscBool because we cannot provide a cast from int to PetscBool in C.
*/
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

    Concepts: help messages^printing
    Concepts: printing^help messages

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

/* These are used internally by PETSc ASCII IO routines*/
#include <stdarg.h>
PETSC_EXTERN PetscErrorCode PetscVSNPrintf(char*,size_t,const char[],size_t*,va_list);
PETSC_EXTERN PetscErrorCode (*PetscVFPrintf)(FILE*,const char[],va_list);
PETSC_EXTERN PetscErrorCode PetscVFPrintfDefault(FILE*,const char[],va_list);

#if defined(PETSC_HAVE_MATLAB_ENGINE)
PETSC_EXTERN PetscErrorCode PetscVFPrintf_Matlab(FILE*,const char[],va_list);
#endif

#if defined(PETSC_HAVE_CLOSURES)
PETSC_EXTERN PetscErrorCode PetscVFPrintfSetClosure(int (^)(const char*));
#endif

PETSC_EXTERN PetscErrorCode PetscErrorPrintfDefault(const char [],...);
PETSC_EXTERN PetscErrorCode PetscErrorPrintfNone(const char [],...);
PETSC_EXTERN PetscErrorCode PetscHelpPrintfDefault(MPI_Comm,const char [],...);

#if defined(PETSC_HAVE_POPEN)
PETSC_EXTERN PetscErrorCode PetscPOpen(MPI_Comm,const char[],const char[],const char[],FILE **);
PETSC_EXTERN PetscErrorCode PetscPClose(MPI_Comm,FILE*,int*);
PETSC_EXTERN PetscErrorCode PetscPOpenSetMachine(const char[]);
#endif

PETSC_EXTERN PetscErrorCode PetscSynchronizedPrintf(MPI_Comm,const char[],...);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFPrintf(MPI_Comm,FILE*,const char[],...);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFlush(MPI_Comm,FILE*);
PETSC_EXTERN PetscErrorCode PetscSynchronizedFGets(MPI_Comm,FILE*,size_t,char[]);
PETSC_EXTERN PetscErrorCode PetscStartMatlab(MPI_Comm,const char[],const char[],FILE**);
PETSC_EXTERN PetscErrorCode PetscStartJava(MPI_Comm,const char[],const char[],FILE**);
PETSC_EXTERN PetscErrorCode PetscGetPetscDir(const char*[]);

PETSC_EXTERN PetscErrorCode PetscPopUpSelect(MPI_Comm,const char*,const char*,int,const char**,int*);

/*S
     PetscContainer - Simple PETSc object that contains a pointer to any required data

   Level: advanced

.seealso:  PetscObject, PetscContainerCreate()
S*/
PETSC_EXTERN PetscClassId PETSC_CONTAINER_CLASSID;
typedef struct _p_PetscContainer*  PetscContainer;
PETSC_EXTERN PetscErrorCode PetscContainerGetPointer(PetscContainer,void **);
PETSC_EXTERN PetscErrorCode PetscContainerSetPointer(PetscContainer,void *);
PETSC_EXTERN PetscErrorCode PetscContainerDestroy(PetscContainer*);
PETSC_EXTERN PetscErrorCode PetscContainerCreate(MPI_Comm,PetscContainer *);
PETSC_EXTERN PetscErrorCode PetscContainerSetUserDestroy(PetscContainer, PetscErrorCode (*)(void*));

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
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#if defined(PETSC_HAVE_XMMINTRIN_H) && !defined(__CUDACC__)
#include <xmmintrin.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscMemcpy"
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
   This routine is analogous to memcpy().

   Developer Note: this is inlined for fastest performance

  Concepts: memory^copying
  Concepts: copying^memory

.seealso: PetscMemmove()

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

   Developer Note: this is inlined for fastest performance

   Concepts: memory^zeroing
   Concepts: zeroing^memory

.seealso: PetscMemcpy()
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

   Concepts: memory
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
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif

/* --------------------------------------------------------------------*/

/*MC
    MPI_Comm - the basic object used by MPI to determine which processes are involved in a
        communication

   Level: beginner

   Note: This manual page is a place-holder because MPICH does not have a manual page for MPI_Comm

.seealso: PETSC_COMM_WORLD, PETSC_COMM_SELF
M*/

/*MC
    PetscScalar - PETSc type that represents either a double precision real number, a double precision
       complex number, a single precision real number, a long double or an int - if the code is configured
       with --with-scalar-type=real,complex --with-precision=single,double,__float128

   Level: beginner

.seealso: PetscReal, MPIU_SCALAR, PetscInt, MPIU_REAL
M*/

/*MC
    PetscComplex - PETSc type that represents a complex number with precision matching that of PetscReal.

   Synopsis:
   #include <petscsys.h>
   PetscComplex number = 1. + 2.*PETSC_i;

   Level: beginner

   Note:
   Complex numbers are automatically available if PETSc was able to find a working complex implementation

.seealso: PetscReal, PetscComplex, MPIU_COMPLEX, PetscInt, PETSC_i
M*/

/*MC
    PetscReal - PETSc type that represents a real number version of PetscScalar

   Level: beginner

.seealso: PetscScalar
M*/

/*MC
    MPIU_SCALAR - MPI datatype corresponding to PetscScalar

   Level: beginner

    Note: In MPI calls that require an MPI datatype that matches a PetscScalar or array of PetscScalars
          pass this value

.seealso: PetscReal, PetscScalar, MPIU_INT
M*/

#if defined(PETSC_HAVE_MPIIO)
#if !defined(PETSC_WORDS_BIGENDIAN)
PETSC_EXTERN PetscErrorCode MPIU_File_write_all(MPI_File,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
PETSC_EXTERN PetscErrorCode MPIU_File_read_all(MPI_File,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
#else
#define MPIU_File_write_all(a,b,c,d,e) MPI_File_write_all(a,b,c,d,e)
#define MPIU_File_read_all(a,b,c,d,e) MPI_File_read_all(a,b,c,d,e)
#endif
#endif

/* the following petsc_static_inline require petscerror.h */

/* Limit MPI to 32-bits */
#define PETSC_MPI_INT_MAX  2147483647
#define PETSC_MPI_INT_MIN -2147483647
/* Limit BLAS to 32-bits */
#define PETSC_BLAS_INT_MAX  2147483647
#define PETSC_BLAS_INT_MIN -2147483647

#undef __FUNCT__
#define __FUNCT__ "PetscBLASIntCast"
/*@C
    PetscBLASIntCast - casts a PetscInt (which may be 64 bits in size) to a PetscBLASInt (which may be 32 bits in size), generates an
         error if the PetscBLASInt is not large enough to hold the number.

   Not Collective

   Input Parameter:
.     a - the PetscInt value

   Output Parameter:
.     b - the resulting PetscBLASInt value

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscMPIIntCast()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscBLASIntCast(PetscInt a,PetscBLASInt *b)
{
  PetscFunctionBegin;
  *b =  (PetscBLASInt)(a);
#if defined(PETSC_USE_64BIT_INDICES) && !defined(PETSC_HAVE_64BIT_BLAS_INDICES)
  if ((a) > PETSC_BLAS_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for BLAS/LAPACK");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMPIIntCast"
/*@C
    PetscMPIIntCast - casts a PetscInt (which may be 64 bits in size) to a PetscMPIInt (which may be 32 bits in size), generates an
         error if the PetscMPIInt is not large enough to hold the number.

   Not Collective

   Input Parameter:
.     a - the PetscInt value

   Output Parameter:
.     b - the resulting PetscMPIInt value

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscMPIIntCast(PetscInt a,PetscMPIInt *b)
{
  PetscFunctionBegin;
  *b =  (PetscMPIInt)(a);
#if defined(PETSC_USE_64BIT_INDICES)
  if ((a) > PETSC_MPI_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for MPI");
#endif
  PetscFunctionReturn(0);
}

#define PetscIntMult64bit(a,b)   ((Petsc64bitInt)(a))*((Petsc64bitInt)(b))

#undef __FUNCT__
#define __FUNCT__ "PetscRealIntMultTruncate"
/*@C

   PetscRealIntMultTruncate - Computes the product of a positive PetscReal and a positive PetscInt and truncates the value to slightly less than the maximal possible value

   Not Collective

   Input Parameter:
+     a - the PetscReal value
-     b - the second value

   Output Parameter:
.     c - the result as a PetscInt value

   Use PetscIntMult64bit() to compute the product of two PetscInt as a Petsc64bitInt
   Use PetscIntMultTruncate() to compute the product of two positive PetscInt and truncate to fit a PetscInt
   Use PetscIntMultError() to compute the product of two PetscInt if you wish to generate an error if the result will not fit in a PetscInt

   Developers Note: We currently assume that PetscInt addition can never overflow, this is obviously wrong but requires many more checks.

   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscIntMult64()
@*/
PETSC_STATIC_INLINE PetscInt PetscRealIntMultTruncate(PetscReal a,PetscInt b)
{
  Petsc64bitInt r;

  r  =  (Petsc64bitInt) (a*(PetscReal)b);
  if (r > PETSC_MAX_INT - 100) r = PETSC_MAX_INT - 100;
  return (PetscInt) r;
}

#undef __FUNCT__
#define __FUNCT__ "PetscIntMultTruncate"
/*@C

   PetscIntMultTruncate - Computes the product of two positive PetscInt and truncates the value to slightly less than the maximal possible value

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Output Parameter:
.     c - the result as a PetscInt value

   Use PetscIntMult64bit() to compute the product of two PetscInt as a Petsc64bitInt
   Use PetscRealIntMultTruncate() to compute the product of a PetscReal and a PetscInt and truncate to fit a PetscInt
   Use PetscIntMultError() to compute the product of two PetscInt if you wish to generate an error if the result will not fit in a PetscInt

   Developers Note: We currently assume that PetscInt addition can never overflow, this is obviously wrong but requires many more checks.

   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscIntMult64()
@*/
PETSC_STATIC_INLINE PetscInt PetscIntMultTruncate(PetscInt a,PetscInt b)
{
  Petsc64bitInt r;

  r  =  PetscIntMult64bit(a,b);
  if (r > PETSC_MAX_INT - 100) r = PETSC_MAX_INT - 100;
  return (PetscInt) r;
}

#undef __FUNCT__
#define __FUNCT__ "PetscIntSumTruncate"
/*@C

   PetscIntSumTruncate - Computes the sum of two positive PetscInt and truncates the value to slightly less than the maximal possible value

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Output Parameter:
.     c - the result as a PetscInt value

   Use PetscIntMult64bit() to compute the product of two PetscInt as a Petsc64bitInt
   Use PetscRealIntMultTruncate() to compute the product of a PetscReal and a PetscInt and truncate to fit a PetscInt
   Use PetscIntMultError() to compute the product of two PetscInt if you wish to generate an error if the result will not fit in a PetscInt

   This is used where we compute approximate sizes for workspace and need to insure the workspace is index-able.

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscIntMult64()
@*/
PETSC_STATIC_INLINE PetscInt PetscIntSumTruncate(PetscInt a,PetscInt b)
{
  Petsc64bitInt r;

  r  =  ((Petsc64bitInt)a) + ((Petsc64bitInt)b);
  if (r > PETSC_MAX_INT - 100) r = PETSC_MAX_INT - 100;
  return (PetscInt) r;
}

#undef __FUNCT__
#define __FUNCT__ "PetscIntMultError"
/*@C

   PetscIntMultError - Computes the product of two positive PetscInt and generates an error with overflow.

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Output Parameter:ma
.     c - the result as a PetscInt value

   Use PetscIntMult64bit() to compute the product of two 32 bit PetscInt and store in a Petsc64bitInt
   Use PetscIntMultTruncate() to compute the product of two PetscInt and truncate it to fit in a PetscInt

   Developers Note: We currently assume that PetscInt addition can never overflow, this is obviously wrong but requires many more checks.

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscIntMult64()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscIntMultError(PetscInt a,PetscInt b,PetscInt *result)
{
  Petsc64bitInt r;

  PetscFunctionBegin;
  r  =  PetscIntMult64bit(a,b);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (r > PETSC_MAX_INT) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Product of two integer %d %d overflow, you must ./configure PETSc with --with-64-bit-indices for the case you are running",a,b);
#endif
  if (result) *result = (PetscInt) r;
  PetscFunctionReturn(0);
}

 #undef __FUNCT__
#define __FUNCT__ "PetscIntSumError"
/*@C

   PetscIntSumError - Computes the product of two positive PetscInt and generates an error with overflow.

   Not Collective

   Input Parameter:
+     a - the PetscInt value
-     b - the second value

   Output Parameter:ma
.     c - the result as a PetscInt value

   Use PetscIntMult64bit() to compute the product of two 32 bit PetscInt and store in a Petsc64bitInt
   Use PetscIntMultTruncate() to compute the product of two PetscInt and truncate it to fit in a PetscInt

   Level: advanced

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscBLASIntCast(), PetscIntMult64()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscIntSumError(PetscInt a,PetscInt b,PetscInt *result)
{
  Petsc64bitInt r;

  PetscFunctionBegin;
  r  =  ((Petsc64bitInt)a) + ((Petsc64bitInt)b);
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
#undef hz
#endif

/*  For arrays that contain filenames or paths */


#if defined(PETSC_HAVE_LIMITS_H)
#include <limits.h>
#endif
#if defined(PETSC_HAVE_SYS_PARAM_H)
#include <sys/param.h>
#endif
#if defined(PETSC_HAVE_SYS_TYPES_H)
#include <sys/types.h>
#endif
#if defined(MAXPATHLEN)
#  define PETSC_MAX_PATH_LEN     MAXPATHLEN
#elif defined(MAX_PATH)
#  define PETSC_MAX_PATH_LEN     MAX_PATH
#elif defined(_MAX_PATH)
#  define PETSC_MAX_PATH_LEN     _MAX_PATH
#else
#  define PETSC_MAX_PATH_LEN     4096
#endif

/*MC

    UsingFortran - Fortran can be used with PETSc in four distinct approaches

$    1) classic Fortran 77 style
$#include "petsc/finclude/petscXXX.h" to work with material from the XXX component of PETSc
$       XXX variablename
$      You cannot use this approach if you wish to use the Fortran 90 specific PETSc routines
$      which end in F90; such as VecGetArrayF90()
$
$    2) classic Fortran 90 style
$#include "petsc/finclude/petscXXX.h"
$#include "petsc/finclude/petscXXX.h90" to work with material from the XXX component of PETSc
$       XXX variablename
$
$    3) Using Fortran modules
$#include "petsc/finclude/petscXXXdef.h"
$         use petscXXXX
$       XXX variablename
$
$    4) Use Fortran modules and Fortran data types for PETSc types
$#include "petsc/finclude/petscXXXdef.h"
$         use petscXXXX
$       type(XXX) variablename
$      To use this approach you must ./configure PETSc with the additional
$      option --with-fortran-datatypes You cannot use the type(XXX) declaration approach without using Fortran modules

    Finally if you absolutely do not want to use any #include you can use either

$    3a) skip the #include BUT you cannot use any PETSc data type names like Vec, Mat, PetscInt, PetscErrorCode etc
$        and you must declare the variables as integer, for example
$        integer variablename
$
$    4a) skip the #include, you use the object types like type(Vec) type(Mat) but cannot use the data type
$        names like PetscErrorCode, PetscInt etc. again for those you must use integer

   We recommend either 2 or 3. Approaches 2 and 3 provide type checking for most PETSc function calls; 4 has type checking
for only a few PETSc functions.

   Fortran type checking with interfaces is strick, this means you cannot pass a scalar value when an array value
is expected (even though it is legal Fortran). For example when setting a single value in a matrix with MatSetValues()
you cannot have something like
$      PetscInt row,col
$      PetscScalar val
$        ...
$      call MatSetValues(mat,1,row,1,col,val,INSERT_VALUES,ierr)
You must instead have
$      PetscInt row(1),col(1)
$      PetscScalar val(1)
$        ...
$      call MatSetValues(mat,1,row,1,col,val,INSERT_VALUES,ierr)


    See the example src/vec/vec/examples/tutorials/ex20f90.F90 for an example that can use all four approaches

    Developer Notes: The petsc/finclude/petscXXXdef.h contain all the #defines (would be typedefs in C code) these
     automatically include their predecessors; for example petsc/finclude/petscvecdef.h includes petsc/finclude/petscisdef.h

     The petsc/finclude/petscXXXX.h contain all the parameter statements for that package. These automatically include
     their petsc/finclude/petscXXXdef.h file but DO NOT automatically include their predecessors;  for example
     petsc/finclude/petscvec.h does NOT automatically include petsc/finclude/petscis.h

     The petsc/finclude/ftn-custom/petscXXXdef.h90 are not intended to be used directly in code, they define the
     Fortran data type type(XXX) (for example type(Vec)) when PETSc is ./configure with the --with-fortran-datatypes option.

     The petsc/finclude/ftn-custom/petscXXX.h90 (not included directly by code) contain interface definitions for
     the PETSc Fortran stubs that have different bindings then their C version (for example VecGetArrayF90).

     The petsc/finclude/ftn-auto/petscXXX.h90 (not included directly by code) contain interface definitions generated
     automatically by "make allfortranstubs".

     The petsc/finclude/petscXXX.h90 includes the custom petsc/finclude/ftn-custom/petscXXX.h90 and if ./configure
     was run with --with-fortran-interfaces it also includes the petsc/finclude/ftn-auto/petscXXX.h90 These DO NOT automatically
     include their predecessors

    Level: beginner

M*/

PETSC_EXTERN PetscErrorCode PetscGetArchType(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetHostName(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetUserName(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetProgramName(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscSetProgramName(const char[]);
PETSC_EXTERN PetscErrorCode PetscGetDate(char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGetVersion(char[], size_t);

PETSC_EXTERN PetscErrorCode PetscSortInt(PetscInt,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsInt(PetscInt*,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscFindInt(PetscInt, PetscInt, const PetscInt[], PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSortIntWithPermutation(PetscInt,const PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortStrWithPermutation(PetscInt,const char*[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithArrayPair(PetscInt,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSortMPIInt(PetscInt,PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsMPIInt(PetscInt*,PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortMPIIntWithArray(PetscMPIInt,PetscMPIInt[],PetscMPIInt[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithScalarArray(PetscInt,PetscInt[],PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscSortIntWithDataArray(PetscInt,PetscInt[],void*,size_t,void*);
PETSC_EXTERN PetscErrorCode PetscSortReal(PetscInt,PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscSortRealWithPermutation(PetscInt,const PetscReal[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortRemoveDupsReal(PetscInt*,PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscSortSplit(PetscInt,PetscInt,PetscScalar[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscSortSplitReal(PetscInt,PetscInt,PetscReal[],PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscProcessTree(PetscInt,const PetscBool [],const PetscInt[],PetscInt*,PetscInt**,PetscInt**,PetscInt**,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscMergeIntArrayPair(PetscInt,const PetscInt*,const PetscInt*,PetscInt,const PetscInt*,const PetscInt*,PetscInt*,PetscInt**,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscMergeIntArray(PetscInt,const PetscInt*,PetscInt,const PetscInt*,PetscInt*,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscMergeMPIIntArray(PetscInt,const PetscMPIInt[],PetscInt,const PetscMPIInt[],PetscInt*,PetscMPIInt**);

PETSC_EXTERN PetscErrorCode PetscSetDisplay(void);
PETSC_EXTERN PetscErrorCode PetscGetDisplay(char[],size_t);

/*J
    PetscRandomType - String with the name of a PETSc randomizer

   Level: beginner

   Notes: to use the SPRNG you must have ./configure PETSc
   with the option --download-sprng

.seealso: PetscRandomSetType(), PetscRandom, PetscRandomCreate()
J*/
typedef const char* PetscRandomType;
#define PETSCRAND       "rand"
#define PETSCRAND48     "rand48"
#define PETSCSPRNG      "sprng"
#define PETSCRANDER48   "rander48"

/* Logging support */
PETSC_EXTERN PetscClassId PETSC_RANDOM_CLASSID;

PETSC_EXTERN PetscErrorCode PetscRandomInitializePackage(void);

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

  Concepts: random numbers

.seealso:  PetscRandomCreate(), PetscRandomGetValue(), PetscRandomType
S*/
typedef struct _p_PetscRandom*   PetscRandom;

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFunctionList PetscRandomList;

PETSC_EXTERN PetscErrorCode PetscRandomRegister(const char[],PetscErrorCode (*)(PetscRandom));
PETSC_EXTERN PetscErrorCode PetscRandomSetType(PetscRandom, PetscRandomType);
PETSC_EXTERN PetscErrorCode PetscRandomSetFromOptions(PetscRandom);
PETSC_EXTERN PetscErrorCode PetscRandomGetType(PetscRandom, PetscRandomType*);
PETSC_STATIC_INLINE PetscErrorCode PetscRandomViewFromOptions(PetscRandom A,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,obj,name);}
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
PETSC_EXTERN PetscErrorCode PetscRMTree(const char[]);

PETSC_EXTERN PetscErrorCode PetscBinaryRead(int,void*,PetscInt,PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedRead(MPI_Comm,int,void*,PetscInt,PetscDataType);
PETSC_EXTERN PetscErrorCode PetscBinarySynchronizedWrite(MPI_Comm,int,void*,PetscInt,PetscDataType,PetscBool );
PETSC_EXTERN PetscErrorCode PetscBinaryWrite(int,void*,PetscInt,PetscDataType,PetscBool );
PETSC_EXTERN PetscErrorCode PetscBinaryOpen(const char[],PetscFileMode,int *);
PETSC_EXTERN PetscErrorCode PetscBinaryClose(int);
PETSC_EXTERN PetscErrorCode PetscSharedTmp(MPI_Comm,PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscSharedWorkingDirectory(MPI_Comm,PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscGetTmp(MPI_Comm,char[],size_t);
PETSC_EXTERN PetscErrorCode PetscFileRetrieve(MPI_Comm,const char[],char[],size_t,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscLs(MPI_Comm,const char[],char[],size_t,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOpenSocket(const char[],int,int*);

/*
   In binary files variables are stored using the following lengths,
  regardless of how they are stored in memory on any one particular
  machine. Use these rather then sizeof() in computing sizes for
  PetscBinarySeek().
*/
#define PETSC_BINARY_INT_SIZE   (32/8)
#define PETSC_BINARY_FLOAT_SIZE  (32/8)
#define PETSC_BINARY_CHAR_SIZE  (8/8)
#define PETSC_BINARY_SHORT_SIZE  (16/8)
#define PETSC_BINARY_DOUBLE_SIZE  (64/8)
#define PETSC_BINARY_SCALAR_SIZE  sizeof(PetscScalar)

/*E
  PetscBinarySeekType - argument to PetscBinarySeek()

  Level: advanced

.seealso: PetscBinarySeek(), PetscBinarySynchronizedSeek()
E*/
typedef enum {PETSC_BINARY_SEEK_SET = 0,PETSC_BINARY_SEEK_CUR = 1,PETSC_BINARY_SEEK_END = 2} PetscBinarySeekType;
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

/*E
    PetscBuildTwoSidedType - algorithm for setting up two-sided communication

$  PETSC_BUILDTWOSIDED_ALLREDUCE - classical algorithm using an MPI_Allreduce with
$      a buffer of length equal to the communicator size. Not memory-scalable due to
$      the large reduction size. Requires only MPI-1.
$  PETSC_BUILDTWOSIDED_IBARRIER - nonblocking algorithm based on MPI_Issend and MPI_Ibarrier.
$      Proved communication-optimal in Hoefler, Siebert, and Lumsdaine (2010). Requires MPI-3.
$  PETSC_BUILDTWOSIDED_REDSCATTER - similar to above, but use more optimized function
$      that only communicates the part of the reduction that is necessary.  Requires MPI-2.

   Level: developer

.seealso: PetscCommBuildTwoSided(), PetscCommBuildTwoSidedSetType(), PetscCommBuildTwoSidedGetType()
E*/
typedef enum {
  PETSC_BUILDTWOSIDED_NOTSET = -1,
  PETSC_BUILDTWOSIDED_ALLREDUCE = 0,
  PETSC_BUILDTWOSIDED_IBARRIER = 1,
  PETSC_BUILDTWOSIDED_REDSCATTER = 2
  /* Updates here must be accompanied by updates in finclude/petscsys.h and the string array in mpits.c */
} PetscBuildTwoSidedType;
PETSC_EXTERN const char *const PetscBuildTwoSidedTypes[];
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedSetType(MPI_Comm,PetscBuildTwoSidedType);
PETSC_EXTERN PetscErrorCode PetscCommBuildTwoSidedGetType(MPI_Comm,PetscBuildTwoSidedType*);

PETSC_EXTERN PetscErrorCode PetscSSEIsEnabled(MPI_Comm,PetscBool  *,PetscBool  *);

/*E
  InsertMode - Whether entries are inserted or added into vectors or matrices

  Level: beginner

.seealso: VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(),
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()
E*/
 typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES, INSERT_ALL_VALUES, ADD_ALL_VALUES, INSERT_BC_VALUES, ADD_BC_VALUES} InsertMode;

/*MC
    INSERT_VALUES - Put a value into a vector or matrix, overwrites any previous value

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), ADD_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd(), MAX_VALUES

M*/

/*MC
    ADD_VALUES - Adds a value into a vector or matrix, if there previously was no value, just puts the
                value into that location

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), INSERT_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd(), MAX_VALUES

M*/

/*MC
    MAX_VALUES - Puts the maximum of the scattered/gathered value and the current value into each location

    Level: beginner

.seealso: InsertMode, VecScatterBegin(), VecScatterEnd(), ADD_VALUES, INSERT_VALUES

M*/

PETSC_EXTERN MPI_Comm PetscObjectComm(PetscObject);

typedef enum {PETSC_SUBCOMM_GENERAL=0,PETSC_SUBCOMM_CONTIGUOUS=1,PETSC_SUBCOMM_INTERLACED=2} PetscSubcommType;
PETSC_EXTERN const char *const PetscSubcommTypes[];

/*S
   PetscSubcomm - A decomposition of an MPI communicator into subcommunicators

   Notes: After a call to PetscSubcommSetType(), PetscSubcommSetTypeGeneral(), or PetscSubcommSetFromOptions() one may call
$     PetscSubcommChild() returns the associated subcommunicator on this process
$     PetscSubcommContiguousParent() returns a parent communitor but with all child of the same subcommunicator having contiquous rank

   Sample Usage:
       PetscSubcommCreate()
       PetscSubcommSetNumber()
       PetscSubcommSetType(PETSC_SUBCOMM_INTERLACED);
       ccomm = PetscSubcommChild()
       PetscSubcommDestroy()

   Level: advanced

   Concepts: communicator, create

   Notes:
$   PETSC_SUBCOMM_GENERAL - similar to MPI_Comm_split() each process sets the new communicator (color) they will belong to and the order within that communicator
$   PETSC_SUBCOMM_CONTIGUOUS - each new communicator contains a set of process with contiquous ranks in the original MPI communicator
$   PETSC_SUBCOMM_INTERLACED - each new communictor contains a set of processes equally far apart in rank from the others in that new communicator

   Examaple: Consider a communicator with six processes split into 3 subcommunicators.
$     PETSC_SUBCOMM_CONTIGUOUS - the first communicator contains rank 0,1  the second rank 2,3 and the third rank 4,5 in the original ordering of the original communicator
$     PETSC_SUBCOMM_INTERLACED - the first communicator contains rank 0,3, the second 1,4 and the third 2,5

   Developer Notes: This is used in objects such as PCREDUNDANT() to manage the subcommunicators on which the redundant computations
      are performed.


.seealso: PetscSubcommCreate(), PetscSubcommSetNumber(), PetscSubcommSetType(), PetscSubcommView(), PetscSubcommSetFromOptions()

S*/
typedef struct _n_PetscSubcomm* PetscSubcomm;

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

/*S
   PetscSegBuffer - a segmented extendable buffer

   Level: developer

.seealso: PetscSegBufferCreate(), PetscSegBufferGet(), PetscSegBufferExtract(), PetscSegBufferDestroy()
S*/
typedef struct _n_PetscSegBuffer *PetscSegBuffer;
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
PETSC_STATIC_INLINE PetscErrorCode PetscSegBufferGetInts(PetscSegBuffer seg,PetscInt count,PetscInt *PETSC_RESTRICT *slot) {return PetscSegBufferGet(seg,(size_t)count,(void**)slot);}

typedef struct _n_PetscOptionsHelpPrinted *PetscOptionsHelpPrinted;
extern PetscOptionsHelpPrinted PetscOptionsHelpPrintedSingleton;
PETSC_EXTERN PetscErrorCode PetscOptionsHelpPrintedDestroy(PetscOptionsHelpPrinted*);
PETSC_EXTERN PetscErrorCode PetscOptionsHelpPrintedCreate(PetscOptionsHelpPrinted*);
PETSC_EXTERN PetscErrorCode PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrinted,const char*,const char*,PetscBool*);

PETSC_EXTERN PetscSegBuffer PetscCitationsList;
#undef __FUNCT__
#define __FUNCT__ "PetscCitationsRegister"
/*@C
      PetscCitationsRegister - Register a bibtex item to obtain credit for an implemented algorithm used in the code.

     Not Collective - only what is registered on rank 0 of PETSC_COMM_WORLD will be printed

     Input Parameters:
+      cite - the bibtex item, formated to displayed on multiple lines nicely
-      set - a boolean variable initially set to PETSC_FALSE; this is used to insure only a single registration of the citation

   Level: intermediate

     Options Database:
.     -citations [filenmae]   - print out the bibtex entries for the given computation
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
  ierr = PetscMemcpy(vstring,cit,len);CHKERRQ(ierr);
  if (set) *set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscURLShorten(const char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveAuthorize(MPI_Comm,char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveRefresh(MPI_Comm,const char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveUpload(MPI_Comm,const char[],const char []);

PETSC_EXTERN PetscErrorCode PetscBoxAuthorize(MPI_Comm,char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscBoxRefresh(MPI_Comm,const char[],char[],char[],size_t);

PETSC_EXTERN PetscErrorCode PetscTextBelt(MPI_Comm,const char[],const char[],PetscBool*);

PETSC_EXTERN PetscErrorCode PetscPullJSONValue(const char[],const char[],char[],size_t,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscPushJSONValue(char[],const char[],const char[],size_t);


#if defined(PETSC_USE_DEBUG)
/*
   Verify that all processes in the communicator have called this from the same line of code
 */
PETSC_EXTERN PetscErrorCode PetscAllreduceBarrierCheck(MPI_Comm,PetscMPIInt,int,const char*,const char *);
#define MPIU_Allreduce(a,b,c,d,e,fcomm) (PetscAllreduceBarrierCheck(fcomm,c,__LINE__,__FUNCT__,__FILE__) || MPI_Allreduce(a,b,c,d,e,fcomm))
#else
#define MPIU_Allreduce(a,b,c,d,e,fcomm) MPI_Allreduce(a,b,c,d,e,fcomm)
#endif

/* Reset __FUNCT__ in case the user does not define it themselves */
#undef __FUNCT__
#define __FUNCT__ "User provided function"

#endif
