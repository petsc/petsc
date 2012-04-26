/*
   This is the main PETSc include file (for C and C++).  It is included by all
   other PETSc include files, so it almost never has to be specifically included.
*/
#if !defined(__PETSCSYS_H)
#define __PETSCSYS_H
/* ========================================================================== */
/* 
   petscconf.h is contained in ${PETSC_ARCH}/include/petscconf.h it is 
   found automatically by the compiler due to the -I${PETSC_DIR}/${PETSC_ARCH}/include
   in the conf/variables definition of PETSC_INCLUDE. For --prefix installs the ${PETSC_ARCH}/
   does not exist and petscconf.h is in the same directory as the other PETSc include files.
*/
#include "petscconf.h"
#include "petscfix.h"

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
#endif

/* ========================================================================== */
/* 
   This facilitates using the C version of PETSc from C++ and the
   C++ version from C. Use --with-c-support --with-clanguage=c++ with ./configure for the latter)
*/
#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX) && !defined(__cplusplus)
#error "PETSc configured with --with-clanguage=c++ and NOT --with-c-support - it can be used only with a C++ compiler"
#endif

#if defined(__cplusplus)
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_CXX
#else
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_C
#endif

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
#define PETSC_EXTERN extern "C"
#define PETSC_EXTERN_CXX_BEGIN extern "C" {
#define PETSC_EXTERN_CXX_END  }
#else
#define PETSC_EXTERN extern
#define PETSC_EXTERN_CXX_BEGIN
#define PETSC_EXTERN_CXX_END
#endif
/* ========================================================================== */
/* 
   Current PETSc version number and release date. Also listed in
    Web page
    src/docs/tex/manual/intro.tex,
    src/docs/tex/manual/manual.tex.
    src/docs/website/index.html.
*/
#include "petscversion.h"
#define PETSC_AUTHOR_INFO        "       The PETSc Team\n    petsc-maint@mcs.anl.gov\n http://www.mcs.anl.gov/petsc/\n"
#if (PETSC_VERSION_RELEASE == 1)
#define PetscGetVersion(version,len) PetscSNPrintf(version,len,"Petsc Release Version %d.%d.%d, Patch %d, %s ", \
                                         PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, \
					 PETSC_VERSION_PATCH,PETSC_VERSION_PATCH_DATE)
#else
#define PetscGetVersion(version,len) PetscSNPrintf(version,len,"Petsc Development HG revision: %s  HG Date: %s", \
                                        PETSC_VERSION_HG, PETSC_VERSION_DATE_HG)
#endif

/*MC
    PetscGetVersion - Gets the PETSc version information in a string.

    Input Parameter:
.   len - length of the string

    Output Parameter:
.   version - version string

    Level: developer

    Usage:
    char version[256];
    ierr = PetscGetVersion(version,256);CHKERRQ(ierr)

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscGetProgramName()

M*/

/* ========================================================================== */

/*
    Defines the interface to MPI allowing the use of all MPI functions.

    PETSc does not use the C++ binding of MPI at ALL. The following flag
    makes sure the C++ bindings are not included. The C++ bindings REQUIRE
    putting mpi.h before ANY C++ include files, we cannot control this
    with all PETSc users. Users who want to use the MPI C++ bindings can include 
    mpicxx.h directly in their code
*/
#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include "mpi.h"

/*
    Need to put stdio.h AFTER mpi.h for MPICH2 with C++ compiler 
    see the top of mpicxx.h in the MPICH2 distribution.
*/
#include <stdio.h>

/* MSMPI on 32bit windows requires this yukky hack - that breaks MPI standard compliance */
#if !defined(MPIAPI)
#define MPIAPI
#endif


/*MC
    PetscErrorCode - datatype used for return error code from all PETSc functions

    Level: beginner

.seealso: CHKERRQ, SETERRQ
M*/
typedef int PetscErrorCode;

/*MC

    PetscClassId - A unique id used to identify each PETSc class.
         (internal integer in the data structure used for error
         checking). These are all computed by an offset from the lowest
         one, PETSC_SMALLEST_CLASSID. 

    Level: advanced

.seealso: PetscClassIdRegister(), PetscLogEventRegister(), PetscHeaderCreate()
M*/
typedef int PetscClassId;


/*MC
    PetscMPIInt - datatype used to represent 'int' parameters to MPI functions.

    Level: intermediate

    Notes: usually this is the same as PetscInt, but if PETSc was built with --with-64-bit-indices but 
           standard C/Fortran integers are 32 bit then this is NOT the same as PetscInt it remains 32 bit

    PetscMPIIntCheck(a) checks if the given PetscInt a will fit in a PetscMPIInt, if not it generates a 
      PETSC_ERR_ARG_OUTOFRANGE error.

    PetscMPIInt b = PetscMPIIntCast(a) checks if the given PetscInt a will fit in a PetscMPIInt, if not it 
      generates a PETSC_ERR_ARG_OUTOFRANGE error.

.seealso: PetscBLASInt, PetscInt

M*/
typedef int PetscMPIInt;

/*MC
    PetscEnum - datatype used to pass enum types within PETSc functions.

    Level: intermediate

.seealso: PetscOptionsGetEnum(), PetscOptionsEnum(), PetscBagRegisterEnum()
M*/
typedef enum { ENUM_DUMMY } PetscEnum;

/*MC
    PetscInt - PETSc type that represents integer - used primarily to
      represent size of arrays and indexing into arrays. Its size can be configured with the option
      --with-64-bit-indices - to be either 32bit or 64bit [default 32 bit ints]

   Level: intermediate

.seealso: PetscScalar, PetscBLASInt, PetscMPIInt
M*/
#if (PETSC_SIZEOF_LONG_LONG == 8)
typedef long long Petsc64bitInt;
#elif defined(PETSC_HAVE___INT64)
typedef __int64 Petsc64bitInt;
#else
typedef unknown64bit Petsc64bitInt
#endif
#if defined(PETSC_USE_64BIT_INDICES)
typedef Petsc64bitInt PetscInt;
#define MPIU_INT MPI_LONG_LONG_INT
#else
typedef int PetscInt;
#define MPIU_INT MPI_INT
#endif


/*MC
    PetscBLASInt - datatype used to represent 'int' parameters to BLAS/LAPACK functions.

    Level: intermediate

    Notes: usually this is the same as PetscInt, but if PETSc was built with --with-64-bit-indices but
           standard C/Fortran integers are 32 bit then this is NOT the same as PetscInt it remains 32 bit
           (except on very rare BLAS/LAPACK implementations that support 64 bit integers see the note below).

    PetscBLASIntCheck(a) checks if the given PetscInt a will fit in a PetscBLASInt, if not it generates a
      PETSC_ERR_ARG_OUTOFRANGE error.

    PetscBLASInt b = PetscBLASIntCast(a) checks if the given PetscInt a will fit in a PetscBLASInt, if not it
      generates a PETSC_ERR_ARG_OUTOFRANGE error

    Installation Notes: The 64bit versions of MATLAB ship with BLAS and LAPACK that use 64 bit integers for sizes etc,
     if you run ./configure with the option
     --with-blas-lapack-lib=[/Applications/MATLAB_R2010b.app/bin/maci64/libmwblas.dylib,/Applications/MATLAB_R2010b.app/bin/maci64/libmwlapack.dylib]
     but you need to also use --known-64-bit-blas-indices. 

        MKL also ships with 64 bit integer versions of the BLAS and LAPACK, if you select those you must also ./configure with --known-64-bit-blas-indices

     Developer Notes: Eventually ./configure should automatically determine the size of the integers used by BLAS/LAPACK.

     External packages such as hypre, ML, SuperLU etc do not provide any support for passing 64 bit integers to BLAS/LAPACK so cannot
     be used with PETSc if you have set PetscBLASInt to long int.

.seealso: PetscMPIInt, PetscInt

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
extern const char *PetscPrecisions[];

/* 
    For the rare cases when one needs to send a size_t object with MPI
*/
#if (PETSC_SIZEOF_SIZE_T) == (PETSC_SIZEOF_INT)
#define MPIU_SIZE_T MPI_INT
#elif  (PETSC_SIZEOF_SIZE_T) == (PETSC_SIZEOF_LONG)
#define MPIU_SIZE_T MPI_LONG
#elif  (PETSC_SIZEOF_SIZE_T) == (PETSC_SIZEOF_LONG_LONG)
#define MPIU_SIZE_T MPI_LONG_LONG_INT
#else
#error "Unknown size for size_t! Send us a bugreport at petsc-maint@mcs.anl.gov"
#endif


/*
      You can use PETSC_STDOUT as a replacement of stdout. You can also change
    the value of PETSC_STDOUT to redirect all standard output elsewhere
*/
extern FILE* PETSC_STDOUT;

/*
      You can use PETSC_STDERR as a replacement of stderr. You can also change
    the value of PETSC_STDERR to redirect all standard error elsewhere
*/
extern FILE* PETSC_STDERR;

/*MC
    PetscUnlikely - hints the compiler that the given condition is usually FALSE

    Synopsis:
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
    Defines some elementary mathematics functions and constants.
*/
#include "petscmath.h"

/*
    Declare extern C stuff after including external header files
*/

PETSC_EXTERN_CXX_BEGIN

/*
       Basic PETSc constants
*/

/*E
    PetscBool  - Logical variable. Actually an int in C and a logical in Fortran. 

   Level: beginner

   Developer Note: Why have PetscBool , why not use bool in C? The problem is that K and R C, C99 and C++ all have different mechanisms for 
      boolean values. It is not easy to have a simple macro that that will work properly in all circumstances with all three mechanisms.

E*/
typedef enum { PETSC_FALSE,PETSC_TRUE } PetscBool;
extern const char *PetscBools[];

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
extern const char *PetscCopyModes[];

/*MC
    PETSC_FALSE - False value of PetscBool 

    Level: beginner

    Note: Zero integer

.seealso: PetscBool , PETSC_TRUE
M*/

/*MC
    PETSC_TRUE - True value of PetscBool 

    Level: beginner

    Note: Nonzero integer

.seealso: PetscBool , PETSC_FALSE
M*/

/*MC
    PETSC_NULL - standard way of passing in a null or array or pointer

   Level: beginner

   Notes: accepted by many PETSc functions to not set a parameter and instead use
          some default

          This macro does not exist in Fortran; you must use PETSC_NULL_INTEGER, 
          PETSC_NULL_DOUBLE_PRECISION, PETSC_NULL_FUNCTION, PETSC_NULL_OBJECT etc

  Developer Note: Why have PETSC_NULL, why not just use NULL? The problem is that NULL is defined in different include files under 
      different versions of Unix. It is tricky to insure the correct include file is always included.

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_NULL           0

/*MC
    PETSC_IGNORE - same as PETSC_NULL, means PETSc will ignore this argument

   Level: beginner

   Note: accepted by many PETSc functions to not set a parameter and instead use
          some default

   Fortran Notes: This macro does not exist in Fortran; you must use PETSC_NULL_INTEGER, 
          PETSC_NULL_DOUBLE_PRECISION etc

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_NULL, PETSC_DETERMINE

M*/
#define PETSC_IGNORE         PETSC_NULL

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

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_IGNORE, PETSC_NULL, VecSetSizes()

M*/
#define PETSC_DETERMINE PETSC_DECIDE

/*MC
    PETSC_DEFAULT - standard way of passing in integer or floating point parameter
       where you wish PETSc to use the default.

   Level: beginner

   Fortran Notes: You need to use PETSC_DEFAULT_INTEGER or PETSC_DEFAULT_DOUBLE_PRECISION.

.seealso: PETSC_DECIDE, PETSC_NULL, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_DEFAULT  -2

/*MC
    PETSC_COMM_WORLD - the equivalent of the MPI_COMM_WORLD communicator which represents
           all the processs that PETSc knows about. 

   Level: beginner

   Notes: By default PETSC_COMM_WORLD and MPI_COMM_WORLD are identical unless you wish to 
          run PETSc on ONLY a subset of MPI_COMM_WORLD. In that case create your new (smaller)
          communicator, call it, say comm, and set PETSC_COMM_WORLD = comm BEFORE calling
          PetscInitialize()

.seealso: PETSC_COMM_SELF

M*/
extern MPI_Comm PETSC_COMM_WORLD;

/*MC
    PETSC_COMM_SELF - This is always MPI_COMM_SELF

   Level: beginner

.seealso: PETSC_COMM_WORLD

M*/
#define PETSC_COMM_SELF MPI_COMM_SELF

extern  PetscBool  PetscInitializeCalled;
extern  PetscBool  PetscFinalizeCalled;
extern  PetscBool  PetscCUSPSynchronize;

extern PetscErrorCode  PetscSetHelpVersionFunctions(PetscErrorCode (*)(MPI_Comm),PetscErrorCode (*)(MPI_Comm));
extern PetscErrorCode  PetscCommDuplicate(MPI_Comm,MPI_Comm*,int*);
extern PetscErrorCode  PetscCommDestroy(MPI_Comm*);

/*MC
   PetscMalloc - Allocates memory

   Synopsis:
   PetscErrorCode PetscMalloc(size_t m,void **result)

   Not Collective

   Input Parameter:
.  m - number of bytes to allocate

   Output Parameter:
.  result - memory allocated

   Level: beginner

   Notes: Memory is always allocated at least double aligned

          If you request memory of zero size it will allocate no space and assign the pointer to 0; PetscFree() will 
          properly handle not freeing the null pointer.

.seealso: PetscFree(), PetscNew()

  Concepts: memory allocation

M*/
#define PetscMalloc(a,b)  ((a != 0) ? (*PetscTrMalloc)((a),__LINE__,PETSC_FUNCTION_NAME,__FILE__,__SDIR__,(void**)(b)) : (*(b) = 0,0) )

/*MC
   PetscAddrAlign - Rounds up an address to PETSC_MEMALIGN alignment

   Synopsis:
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
   PetscMalloc2 - Allocates 2 chunks of  memory both aligned to PETSC_MEMALIGN

   Synopsis:
   PetscErrorCode PetscMalloc2(size_t m1,type, t1,void **r1,size_t m2,type t2,void **r2)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  t1 - type of first memory elements 
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
-  t2 - type of second memory elements

   Output Parameter:
+  r1 - memory allocated in first chunk
-  r2 - memory allocated in second chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc()

  Concepts: memory allocation

M*/
#if defined(PETSC_USE_DEBUG)
#define PetscMalloc2(m1,t1,r1,m2,t2,r2) (PetscMalloc((m1)*sizeof(t1),r1) || PetscMalloc((m2)*sizeof(t2),r2))
#else
#define PetscMalloc2(m1,t1,r1,m2,t2,r2) ((*(r2) = 0,PetscMalloc((m1)*sizeof(t1)+(m2)*sizeof(t2)+(PETSC_MEMALIGN-1),r1)) || (*(r2) = (t2*)PetscAddrAlign(*(r1)+m1),0))
#endif

/*MC
   PetscMalloc3 - Allocates 3 chunks of  memory  all aligned to PETSC_MEMALIGN

   Synopsis:
   PetscErrorCode PetscMalloc3(size_t m1,type, t1,void **r1,size_t m2,type t2,void **r2,size_t m3,type t3,void **r3)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  t1 - type of first memory elements 
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  t2 - type of second memory elements
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
-  t3 - type of third memory elements

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
-  r3 - memory allocated in third chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree3()

  Concepts: memory allocation

M*/
#if defined(PETSC_USE_DEBUG)
#define PetscMalloc3(m1,t1,r1,m2,t2,r2,m3,t3,r3) (PetscMalloc((m1)*sizeof(t1),r1) || PetscMalloc((m2)*sizeof(t2),r2) || PetscMalloc((m3)*sizeof(t3),r3))
#else
#define PetscMalloc3(m1,t1,r1,m2,t2,r2,m3,t3,r3) ((*(r2) = 0,*(r3) = 0,PetscMalloc((m1)*sizeof(t1)+(m2)*sizeof(t2)+(m3)*sizeof(t3)+2*(PETSC_MEMALIGN-1),r1)) \
                                                  || (*(r2) = (t2*)PetscAddrAlign(*(r1)+m1),*(r3) = (t3*)PetscAddrAlign(*(r2)+m2),0))
#endif

/*MC
   PetscMalloc4 - Allocates 4 chunks of  memory  all aligned to PETSC_MEMALIGN

   Synopsis:
   PetscErrorCode PetscMalloc4(size_t m1,type, t1,void **r1,size_t m2,type t2,void **r2,size_t m3,type t3,void **r3,size_t m4,type t4,void **r4)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  t1 - type of first memory elements 
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  t2 - type of second memory elements
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  t3 - type of third memory elements
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
-  t4 - type of fourth memory elements

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
-  r4 - memory allocated in fourth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree3(), PetscFree4()

  Concepts: memory allocation

M*/
#if defined(PETSC_USE_DEBUG)
#define PetscMalloc4(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4) (PetscMalloc((m1)*sizeof(t1),r1) || PetscMalloc((m2)*sizeof(t2),r2) || PetscMalloc((m3)*sizeof(t3),r3) || PetscMalloc((m4)*sizeof(t4),r4))
#else
#define PetscMalloc4(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4)               \
  ((*(r2) = 0, *(r3) = 0, *(r4) = 0,PetscMalloc((m1)*sizeof(t1)+(m2)*sizeof(t2)+(m3)*sizeof(t3)+(m4)*sizeof(t4)+3*(PETSC_MEMALIGN-1),r1)) \
   || (*(r2) = (t2*)PetscAddrAlign(*(r1)+m1),*(r3) = (t3*)PetscAddrAlign(*(r2)+m2),*(r4) = (t4*)PetscAddrAlign(*(r3)+m3),0))
#endif

/*MC
   PetscMalloc5 - Allocates 5 chunks of  memory all aligned to PETSC_MEMALIGN

   Synopsis:
   PetscErrorCode PetscMalloc5(size_t m1,type, t1,void **r1,size_t m2,type t2,void **r2,size_t m3,type t3,void **r3,size_t m4,type t4,void **r4,size_t m5,type t5,void **r5)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  t1 - type of first memory elements 
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  t2 - type of second memory elements
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  t3 - type of third memory elements
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  t4 - type of fourth memory elements
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
-  t5 - type of fifth memory elements

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
-  r5 - memory allocated in fifth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree3(), PetscFree4(), PetscFree5()

  Concepts: memory allocation

M*/
#if defined(PETSC_USE_DEBUG)
#define PetscMalloc5(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4,m5,t5,r5) (PetscMalloc((m1)*sizeof(t1),r1) || PetscMalloc((m2)*sizeof(t2),r2) || PetscMalloc((m3)*sizeof(t3),r3) || PetscMalloc((m4)*sizeof(t4),r4) || PetscMalloc((m5)*sizeof(t5),r5))
#else
#define PetscMalloc5(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4,m5,t5,r5)      \
  ((*(r2) = 0, *(r3) = 0, *(r4) = 0,*(r5) = 0,PetscMalloc((m1)*sizeof(t1)+(m2)*sizeof(t2)+(m3)*sizeof(t3)+(m4)*sizeof(t4)+(m5)*sizeof(t5)+4*(PETSC_MEMALIGN-1),r1)) \
   || (*(r2) = (t2*)PetscAddrAlign(*(r1)+m1),*(r3) = (t3*)PetscAddrAlign(*(r2)+m2),*(r4) = (t4*)PetscAddrAlign(*(r3)+m3),*(r5) = (t5*)PetscAddrAlign(*(r4)+m4),0))
#endif


/*MC
   PetscMalloc6 - Allocates 6 chunks of  memory all aligned to PETSC_MEMALIGN

   Synopsis:
   PetscErrorCode PetscMalloc6(size_t m1,type, t1,void **r1,size_t m2,type t2,void **r2,size_t m3,type t3,void **r3,size_t m4,type t4,void **r4,size_t m5,type t5,void **r5,size_t m6,type t6,void **r6)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  t1 - type of first memory elements 
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  t2 - type of second memory elements
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  t3 - type of third memory elements
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  t4 - type of fourth memory elements
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  t5 - type of fifth memory elements
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
-  t6 - type of sixth memory elements

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
-  r6 - memory allocated in sixth chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree3(), PetscFree4(), PetscFree5(), PetscFree6()

  Concepts: memory allocation

M*/
#if defined(PETSC_USE_DEBUG)
#define PetscMalloc6(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4,m5,t5,r5,m6,t6,r6) (PetscMalloc((m1)*sizeof(t1),r1) || PetscMalloc((m2)*sizeof(t2),r2) || PetscMalloc((m3)*sizeof(t3),r3) || PetscMalloc((m4)*sizeof(t4),r4) || PetscMalloc((m5)*sizeof(t5),r5) || PetscMalloc((m6)*sizeof(t6),r6))
#else
#define PetscMalloc6(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4,m5,t5,r5,m6,t6,r6) \
  ((*(r2) = 0, *(r3) = 0, *(r4) = 0,*(r5) = 0,*(r6) = 0,PetscMalloc((m1)*sizeof(t1)+(m2)*sizeof(t2)+(m3)*sizeof(t3)+(m4)*sizeof(t4)+(m5)*sizeof(t5)+(m6)*sizeof(t6)+5*(PETSC_MEMALIGN-1),r1)) \
   || (*(r2) = (t2*)PetscAddrAlign(*(r1)+m1),*(r3) = (t3*)PetscAddrAlign(*(r2)+m2),*(r4) = (t4*)PetscAddrAlign(*(r3)+m3),*(r5) = (t5*)PetscAddrAlign(*(r4)+m4),*(r6) = (t6*)PetscAddrAlign(*(r5)+m5),0))
#endif

/*MC
   PetscMalloc7 - Allocates 7 chunks of  memory all aligned to PETSC_MEMALIGN

   Synopsis:
   PetscErrorCode PetscMalloc7(size_t m1,type, t1,void **r1,size_t m2,type t2,void **r2,size_t m3,type t3,void **r3,size_t m4,type t4,void **r4,size_t m5,type t5,void **r5,size_t m6,type t6,void **r6,size_t m7,type t7,void **r7)

   Not Collective

   Input Parameter:
+  m1 - number of elements to allocate in 1st chunk  (may be zero)
.  t1 - type of first memory elements 
.  m2 - number of elements to allocate in 2nd chunk  (may be zero)
.  t2 - type of second memory elements
.  m3 - number of elements to allocate in 3rd chunk  (may be zero)
.  t3 - type of third memory elements
.  m4 - number of elements to allocate in 4th chunk  (may be zero)
.  t4 - type of fourth memory elements
.  m5 - number of elements to allocate in 5th chunk  (may be zero)
.  t5 - type of fifth memory elements
.  m6 - number of elements to allocate in 6th chunk  (may be zero)
.  t6 - type of sixth memory elements
.  m7 - number of elements to allocate in 7th chunk  (may be zero)
-  t7 - type of sixth memory elements

   Output Parameter:
+  r1 - memory allocated in first chunk
.  r2 - memory allocated in second chunk
.  r3 - memory allocated in third chunk
.  r4 - memory allocated in fourth chunk
.  r5 - memory allocated in fifth chunk
.  r6 - memory allocated in sixth chunk
-  r7 - memory allocated in seventh chunk

   Level: developer

.seealso: PetscFree(), PetscNew(), PetscMalloc(), PetscMalloc2(), PetscFree3(), PetscFree4(), PetscFree5(), PetscFree6(), PetscFree7()

  Concepts: memory allocation

M*/
#if defined(PETSC_USE_DEBUG)
#define PetscMalloc7(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4,m5,t5,r5,m6,t6,r6,m7,t7,r7) (PetscMalloc((m1)*sizeof(t1),r1) || PetscMalloc((m2)*sizeof(t2),r2) || PetscMalloc((m3)*sizeof(t3),r3) || PetscMalloc((m4)*sizeof(t4),r4) || PetscMalloc((m5)*sizeof(t5),r5) || PetscMalloc((m6)*sizeof(t6),r6) || PetscMalloc((m7)*sizeof(t7),r7))
#else
#define PetscMalloc7(m1,t1,r1,m2,t2,r2,m3,t3,r3,m4,t4,r4,m5,t5,r5,m6,t6,r6,m7,t7,r7) \
  ((*(r2) = 0, *(r3) = 0, *(r4) = 0,*(r5) = 0,*(r6) = 0,*(r7) = 0,PetscMalloc((m1)*sizeof(t1)+(m2)*sizeof(t2)+(m3)*sizeof(t3)+(m4)*sizeof(t4)+(m5)*sizeof(t5)+(m6)*sizeof(t6)+(m7)*sizeof(t7)+6*(PETSC_MEMALIGN-1),r1)) \
   || (*(r2) = (t2*)PetscAddrAlign(*(r1)+m1),*(r3) = (t3*)PetscAddrAlign(*(r2)+m2),*(r4) = (t4*)PetscAddrAlign(*(r3)+m3),*(r5) = (t5*)PetscAddrAlign(*(r4)+m4),*(r6) = (t6*)PetscAddrAlign(*(r5)+m5),*(r7) = (t7*)PetscAddrAlign(*(r6)+m6),0))
#endif

/*MC
   PetscNew - Allocates memory of a particular type, zeros the memory! Aligned to PETSC_MEMALIGN

   Synopsis:
   PetscErrorCode PetscNew(struct type,((type *))result)

   Not Collective

   Input Parameter:
.  type - structure name of space to be allocated. Memory of size sizeof(type) is allocated

   Output Parameter:
.  result - memory allocated

   Level: beginner

.seealso: PetscFree(), PetscMalloc(), PetscNewLog()

  Concepts: memory allocation

M*/
#define PetscNew(A,b)      (PetscMalloc(sizeof(A),(b)) || PetscMemzero(*(b),sizeof(A)))

/*MC
   PetscNewLog - Allocates memory of a particular type, zeros the memory! Aligned to PETSC_MEMALIGN. Associates the memory allocated 
         with the given object using PetscLogObjectMemory().

   Synopsis:
   PetscErrorCode PetscNewLog(PetscObject obj,struct type,((type *))result)

   Not Collective

   Input Parameter:
+  obj - object memory is logged to
-  type - structure name of space to be allocated. Memory of size sizeof(type) is allocated

   Output Parameter:
.  result - memory allocated

   Level: developer

.seealso: PetscFree(), PetscMalloc(), PetscNew(), PetscLogObjectMemory()

  Concepts: memory allocation

M*/
#define PetscNewLog(o,A,b) (PetscNew(A,b) || ((o) ? PetscLogObjectMemory(o,sizeof(A)) : 0))

/*MC
   PetscFree - Frees memory

   Synopsis:
   PetscErrorCode PetscFree(void *memory)

   Not Collective

   Input Parameter:
.   memory - memory to free (the pointer is ALWAYS set to 0 upon sucess)

   Level: beginner

   Notes: Memory must have been obtained with PetscNew() or PetscMalloc()

.seealso: PetscNew(), PetscMalloc(), PetscFreeVoid()

  Concepts: memory allocation

M*/
#define PetscFree(a)   ((a) && ((*PetscTrFree)((void*)(a),__LINE__,PETSC_FUNCTION_NAME,__FILE__,__SDIR__) || ((a) = 0,0)))

/*MC
   PetscFreeVoid - Frees memory

   Synopsis:
   void PetscFreeVoid(void *memory)

   Not Collective

   Input Parameter:
.   memory - memory to free

   Level: beginner

   Notes: This is different from PetscFree() in that no error code is returned

.seealso: PetscFree(), PetscNew(), PetscMalloc()

  Concepts: memory allocation

M*/
#define PetscFreeVoid(a) ((*PetscTrFree)((a),__LINE__,PETSC_FUNCTION_NAME,__FILE__,__SDIR__),(a) = 0)


/*MC
   PetscFree2 - Frees 2 chunks of memory obtained with PetscMalloc2()

   Synopsis:
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
#if defined(PETSC_USE_DEBUG)
#define PetscFree2(m1,m2)   (PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree2(m1,m2)   ((m2)=0, PetscFree(m1))
#endif

/*MC
   PetscFree3 - Frees 3 chunks of memory obtained with PetscMalloc3()

   Synopsis:
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
#if defined(PETSC_USE_DEBUG)
#define PetscFree3(m1,m2,m3)   (PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree3(m1,m2,m3)   ((m3)=0,(m2)=0,PetscFree(m1))
#endif

/*MC
   PetscFree4 - Frees 4 chunks of memory obtained with PetscMalloc4()

   Synopsis:
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
#if defined(PETSC_USE_DEBUG)
#define PetscFree4(m1,m2,m3,m4)   (PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree4(m1,m2,m3,m4)   ((m4)=0,(m3)=0,(m2)=0,PetscFree(m1))
#endif

/*MC
   PetscFree5 - Frees 5 chunks of memory obtained with PetscMalloc5()

   Synopsis:
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
#if defined(PETSC_USE_DEBUG)
#define PetscFree5(m1,m2,m3,m4,m5)   (PetscFree(m5) || PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree5(m1,m2,m3,m4,m5)   ((m5)=0,(m4)=0,(m3)=0,(m2)=0,PetscFree(m1))
#endif


/*MC
   PetscFree6 - Frees 6 chunks of memory obtained with PetscMalloc6()

   Synopsis:
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
#if defined(PETSC_USE_DEBUG)
#define PetscFree6(m1,m2,m3,m4,m5,m6)   (PetscFree(m6) || PetscFree(m5) || PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree6(m1,m2,m3,m4,m5,m6)   ((m6)=0,(m5)=0,(m4)=0,(m3)=0,(m2)=0,PetscFree(m1))
#endif

/*MC
   PetscFree7 - Frees 7 chunks of memory obtained with PetscMalloc7()

   Synopsis:
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
#if defined(PETSC_USE_DEBUG)
#define PetscFree7(m1,m2,m3,m4,m5,m6,m7)   (PetscFree(m7) || PetscFree(m6) || PetscFree(m5) || PetscFree(m4) || PetscFree(m3) || PetscFree(m2) || PetscFree(m1))
#else
#define PetscFree7(m1,m2,m3,m4,m5,m6,m7)   ((m7)=0,(m6)=0,(m5)=0,(m4)=0,(m3)=0,(m2)=0,PetscFree(m1))
#endif

extern  PetscErrorCode (*PetscTrMalloc)(size_t,int,const char[],const char[],const char[],void**);
extern  PetscErrorCode (*PetscTrFree)(void*,int,const char[],const char[],const char[]);
extern PetscErrorCode   PetscMallocSet(PetscErrorCode (*)(size_t,int,const char[],const char[],const char[],void**),PetscErrorCode (*)(void*,int,const char[],const char[],const char[]));
extern PetscErrorCode   PetscMallocClear(void);

/*
   Routines for tracing memory corruption/bleeding with default PETSc  memory allocation
*/
extern PetscErrorCode    PetscMallocDump(FILE *);
extern PetscErrorCode    PetscMallocDumpLog(FILE *);
extern PetscErrorCode    PetscMallocGetCurrentUsage(PetscLogDouble *);
extern PetscErrorCode    PetscMallocGetMaximumUsage(PetscLogDouble *);
extern PetscErrorCode    PetscMallocDebug(PetscBool);
extern PetscErrorCode    PetscMallocValidate(int,const char[],const char[],const char[]);
extern PetscErrorCode    PetscMallocSetDumpLog(void);
extern PetscErrorCode    PetscMallocGetDumpLog(PetscBool*);

/*E
    PetscDataType - Used for handling different basic data types.

   Level: beginner

   Developer comment: It would be nice if we could always just use MPI Datatypes, why can we not?

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscDataTypeToMPIDataType(),
          PetscDataTypeGetSize()

E*/
typedef enum {PETSC_INT = 0,PETSC_DOUBLE = 1,PETSC_COMPLEX = 2, PETSC_LONG = 3 ,PETSC_SHORT = 4,PETSC_FLOAT = 5,
              PETSC_CHAR = 6,PETSC_BIT_LOGICAL = 7,PETSC_ENUM = 8,PETSC_BOOL=9, PETSC___FLOAT128 = 10} PetscDataType;
extern const char *PetscDataTypes[];

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

extern PetscErrorCode  PetscDataTypeToMPIDataType(PetscDataType,MPI_Datatype*);
extern PetscErrorCode  PetscMPIDataTypeToPetscDataType(MPI_Datatype,PetscDataType*);
extern PetscErrorCode  PetscDataTypeGetSize(PetscDataType,size_t*);

/*
    Basic memory and string operations. These are usually simple wrappers
   around the basic Unix system calls, but a few of them have additional
   functionality and/or error checking.
*/
extern PetscErrorCode    PetscBitMemcpy(void*,PetscInt,const void*,PetscInt,PetscInt,PetscDataType);
extern PetscErrorCode    PetscMemmove(void*,void *,size_t);
extern PetscErrorCode    PetscMemcmp(const void*,const void*,size_t,PetscBool  *);
extern PetscErrorCode    PetscStrlen(const char[],size_t*);
extern PetscErrorCode    PetscStrToArray(const char[],int*,char ***);
extern PetscErrorCode    PetscStrToArrayDestroy(int,char **);
extern PetscErrorCode    PetscStrcmp(const char[],const char[],PetscBool  *);
extern PetscErrorCode    PetscStrgrt(const char[],const char[],PetscBool  *);
extern PetscErrorCode    PetscStrcasecmp(const char[],const char[],PetscBool *);
extern PetscErrorCode    PetscStrncmp(const char[],const char[],size_t,PetscBool *);
extern PetscErrorCode    PetscStrcpy(char[],const char[]);
extern PetscErrorCode    PetscStrcat(char[],const char[]);
extern PetscErrorCode    PetscStrncat(char[],const char[],size_t);
extern PetscErrorCode    PetscStrncpy(char[],const char[],size_t);
extern PetscErrorCode    PetscStrchr(const char[],char,char *[]);
extern PetscErrorCode    PetscStrtolower(char[]);
extern PetscErrorCode    PetscStrrchr(const char[],char,char *[]);
extern PetscErrorCode    PetscStrstr(const char[],const char[],char *[]);
extern PetscErrorCode    PetscStrrstr(const char[],const char[],char *[]);
extern PetscErrorCode    PetscStrendswith(const char[],const char[],PetscBool*);
extern PetscErrorCode    PetscStrendswithwhich(const char[],const char *const*,PetscInt*);
extern PetscErrorCode    PetscStrallocpy(const char[],char *[]);
extern PetscErrorCode    PetscStrreplace(MPI_Comm,const char[],char[],size_t);

/*S
    PetscToken - 'Token' used for managing tokenizing strings

  Level: intermediate

.seealso: PetscTokenCreate(), PetscTokenFind(), PetscTokenDestroy()
S*/
typedef struct _p_PetscToken* PetscToken;

extern PetscErrorCode    PetscTokenCreate(const char[],const char,PetscToken*);
extern PetscErrorCode    PetscTokenFind(PetscToken,char *[]);
extern PetscErrorCode    PetscTokenDestroy(PetscToken*);

/*
   These are  MPI operations for MPI_Allreduce() etc
*/
extern  MPI_Op PetscMaxSum_Op;
#if (defined(PETSC_USE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128)
extern  MPI_Op MPIU_SUM;
#else
#define MPIU_SUM MPI_SUM
#endif
#if defined(PETSC_USE_REAL___FLOAT128)
extern  MPI_Op MPIU_MAX;
extern  MPI_Op MPIU_MIN;
#else
#define MPIU_MAX MPI_MAX
#define MPIU_MIN MPI_MIN
#endif
extern PetscErrorCode  PetscMaxSum(MPI_Comm,const PetscInt[],PetscInt*,PetscInt*);

extern PetscErrorCode MPIULong_Send(void*,PetscInt,MPI_Datatype,PetscMPIInt,PetscMPIInt,MPI_Comm);
extern PetscErrorCode MPIULong_Recv(void*,PetscInt,MPI_Datatype,PetscMPIInt,PetscMPIInt,MPI_Comm);

/*S
     PetscObject - any PETSc object, PetscViewer, Mat, Vec, KSP etc

   Level: beginner

   Note: This is the base class from which all PETSc objects are derived from.

.seealso:  PetscObjectDestroy(), PetscObjectView(), PetscObjectGetName(), PetscObjectSetName(), PetscObjectReference(), PetscObjectDereferenc()
S*/
typedef struct _p_PetscObject* PetscObject;

/*S
     PetscFList - Linked list of functions, possibly stored in dynamic libraries, accessed
      by string name

   Level: advanced

.seealso:  PetscFListAdd(), PetscFListDestroy(), PetscOpFlist
S*/
typedef struct _n_PetscFList *PetscFList;

/*S
     PetscOpFList - Linked list of operations on objects, implemented by functions, possibly stored in dynamic libraries, 
                    accessed by string op name together with string argument types.

   Level: advanced

.seealso:  PetscFList, PetscOpFListAdd(), PetscOpFListFind(), PetscOpFListDestroy()
S*/
typedef struct _n_PetscOpFList *PetscOpFList;

/*E
  PetscFileMode - Access mode for a file.

  Level: beginner

  FILE_MODE_READ - open a file at its beginning for reading

  FILE_MODE_WRITE - open a file at its beginning for writing (will create if the file does not exist)

  FILE_MODE_APPEND - open a file at end for writing

  FILE_MODE_UPDATE - open a file for updating, meaning for reading and writing

  FILE_MODE_APPEND_UPDATE - open a file for updating, meaning for reading and writing, at the end

.seealso: PetscViewerFileSetMode()
E*/
typedef enum {FILE_MODE_READ, FILE_MODE_WRITE, FILE_MODE_APPEND, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE} PetscFileMode;

#include "petscviewer.h"
#include "petscoptions.h"

#define PETSC_SMALLEST_CLASSID  1211211
extern PetscClassId   PETSC_LARGEST_CLASSID;
extern PetscClassId   PETSC_OBJECT_CLASSID;
extern PetscErrorCode PetscClassIdRegister(const char[],PetscClassId *);

/*
   Routines that get memory usage information from the OS
*/
extern PetscErrorCode  PetscMemoryGetCurrentUsage(PetscLogDouble *);
extern PetscErrorCode  PetscMemoryGetMaximumUsage(PetscLogDouble *);
extern PetscErrorCode  PetscMemorySetGetMaximumUsage(void);
extern PetscErrorCode  PetscMemoryShowUsage(PetscViewer,const char[]);

extern PetscErrorCode  PetscInfoAllow(PetscBool ,const char []);
extern PetscErrorCode  PetscGetTime(PetscLogDouble*);
extern PetscErrorCode  PetscGetCPUTime(PetscLogDouble*);
extern PetscErrorCode  PetscSleep(PetscReal);

/* 
   Initialization of PETSc
*/
extern PetscErrorCode  PetscInitialize(int*,char***,const char[],const char[]);
extern PetscErrorCode  PetscInitializeNoPointers(int,char**,const char[],const char[]);
extern PetscErrorCode  PetscInitializeNoArguments(void);
extern PetscErrorCode  PetscInitialized(PetscBool *);
extern PetscErrorCode  PetscFinalized(PetscBool *);
extern PetscErrorCode  PetscFinalize(void);
extern PetscErrorCode  PetscInitializeFortran(void);
extern PetscErrorCode  PetscGetArgs(int*,char ***);
extern PetscErrorCode  PetscGetArguments(char ***);
extern PetscErrorCode  PetscFreeArguments(char **);

extern PetscErrorCode  PetscEnd(void);
extern PetscErrorCode  PetscSysInitializePackage(const char[]);

extern MPI_Comm PETSC_COMM_LOCAL_WORLD;
extern PetscErrorCode  PetscHMPIMerge(PetscMPIInt,PetscErrorCode (*)(void*),void*);
extern PetscErrorCode  PetscHMPISpawn(PetscMPIInt);
extern PetscErrorCode  PetscHMPIFinalize(void);
extern PetscErrorCode  PetscHMPIRun(MPI_Comm,PetscErrorCode (*)(MPI_Comm,void *),void*);
extern PetscErrorCode  PetscHMPIRunCtx(MPI_Comm,PetscErrorCode (*)(MPI_Comm,void*,void *),void*);
extern PetscErrorCode  PetscHMPIFree(MPI_Comm,void*);
extern PetscErrorCode  PetscHMPIMalloc(MPI_Comm,size_t,void**);

extern PetscErrorCode  PetscPythonInitialize(const char[],const char[]);
extern PetscErrorCode  PetscPythonFinalize(void);
extern PetscErrorCode  PetscPythonPrintError(void);
extern PetscErrorCode  PetscPythonMonitorSet(PetscObject,const char[]);

/*
     These are so that in extern C code we can caste function pointers to non-extern C
   function pointers. Since the regular C++ code expects its function pointers to be C++
*/
typedef void (**PetscVoidStarFunction)(void);
typedef void (*PetscVoidFunction)(void); 
typedef PetscErrorCode (*PetscErrorCodeFunction)(void); 

/*
   PetscTryMethod - Queries an object for a method, if it exists then calls it.
              These are intended to be used only inside PETSc functions.

   Level: developer
   
.seealso: PetscUseMethod()
*/
#define  PetscTryMethod(obj,A,B,C) \
  0;{ PetscErrorCode (*f)B, __ierr; \
    __ierr = PetscObjectQueryFunction((PetscObject)obj,A,(PetscVoidStarFunction)&f);CHKERRQ(__ierr); \
    if (f) {__ierr = (*f)C;CHKERRQ(__ierr);}\
  }

/*
   PetscUseMethod - Queries an object for a method, if it exists then calls it, otherwise generates an error.
              These are intended to be used only inside PETSc functions.

   Level: developer
   
.seealso: PetscTryMethod()
*/
#define  PetscUseMethod(obj,A,B,C) \
  0;{ PetscErrorCode (*f)B, __ierr; \
    __ierr = PetscObjectQueryFunction((PetscObject)obj,A,(PetscVoidStarFunction)&f);CHKERRQ(__ierr); \
    if (f) {__ierr = (*f)C;CHKERRQ(__ierr);}\
    else SETERRQ1(((PetscObject)obj)->comm,PETSC_ERR_SUP,"Cannot locate function %s in object",A); \
  }

/*
    Functions that can act on any PETSc object.
*/
extern PetscErrorCode  PetscObjectDestroy(PetscObject*);
extern PetscErrorCode  PetscObjectGetComm(PetscObject,MPI_Comm *);
extern PetscErrorCode  PetscObjectGetClassId(PetscObject,PetscClassId *);
extern PetscErrorCode  PetscObjectGetClassName(PetscObject,const char *[]);
extern PetscErrorCode  PetscObjectSetType(PetscObject,const char []);
extern PetscErrorCode  PetscObjectSetPrecision(PetscObject,PetscPrecision);
extern PetscErrorCode  PetscObjectGetType(PetscObject,const char *[]);
extern PetscErrorCode  PetscObjectSetName(PetscObject,const char[]);
extern PetscErrorCode  PetscObjectGetName(PetscObject,const char*[]);
extern PetscErrorCode  PetscObjectPrintClassNamePrefixType(PetscObject,PetscViewer,const char[]);
extern PetscErrorCode  PetscObjectSetTabLevel(PetscObject,PetscInt);
extern PetscErrorCode  PetscObjectGetTabLevel(PetscObject,PetscInt*);
extern PetscErrorCode  PetscObjectIncrementTabLevel(PetscObject,PetscObject,PetscInt);
extern PetscErrorCode  PetscObjectReference(PetscObject);
extern PetscErrorCode  PetscObjectGetReference(PetscObject,PetscInt*);
extern PetscErrorCode  PetscObjectDereference(PetscObject);
extern PetscErrorCode  PetscObjectGetNewTag(PetscObject,PetscMPIInt *);
extern PetscErrorCode  PetscObjectView(PetscObject,PetscViewer);
extern PetscErrorCode  PetscObjectCompose(PetscObject,const char[],PetscObject);
extern PetscErrorCode  PetscObjectRemoveReference(PetscObject,const char[]);
extern PetscErrorCode  PetscObjectQuery(PetscObject,const char[],PetscObject *);
extern PetscErrorCode  PetscObjectComposeFunction(PetscObject,const char[],const char[],void (*)(void));
extern PetscErrorCode  PetscObjectSetFromOptions(PetscObject);
extern PetscErrorCode  PetscObjectSetUp(PetscObject);
extern PetscErrorCode  PetscCommGetNewTag(MPI_Comm,PetscMPIInt *);
extern PetscErrorCode  PetscObjectAddOptionsHandler(PetscObject,PetscErrorCode (*)(PetscObject,void*),PetscErrorCode (*)(PetscObject,void*),void*);
extern PetscErrorCode  PetscObjectProcessOptionsHandlers(PetscObject);
extern PetscErrorCode  PetscObjectDestroyOptionsHandlers(PetscObject);

/*MC
   PetscObjectComposeFunctionDynamic - Associates a function with a given PETSc object. 
                       
    Synopsis:
    PetscErrorCode PetscObjectComposeFunctionDynamic(PetscObject obj,const char name[],const char fname[],void *ptr)

   Logically Collective on PetscObject

   Input Parameters:
+  obj - the PETSc object; this must be cast with a (PetscObject), for example, 
         PetscObjectCompose((PetscObject)mat,...);
.  name - name associated with the child function
.  fname - name of the function
-  ptr - function pointer (or PETSC_NULL if using dynamic libraries)

   Level: advanced


   Notes:
   To remove a registered routine, pass in a PETSC_NULL rname and fnc().

   PetscObjectComposeFunctionDynamic() can be used with any PETSc object (such as
   Mat, Vec, KSP, SNES, etc.) or any user-provided object. 

   The composed function must be wrapped in a EXTERN_C_BEGIN/END for this to
   work in C++/complex with dynamic link libraries (./configure options --with-shared-libraries --with-dynamic-loading)
   enabled.

   Concepts: objects^composing functions
   Concepts: composing functions
   Concepts: functions^querying
   Concepts: objects^querying
   Concepts: querying objects

.seealso: PetscObjectQueryFunction()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscObjectComposeFunctionDynamic(a,b,c,d) PetscObjectComposeFunction(a,b,c,0)
#else
#define PetscObjectComposeFunctionDynamic(a,b,c,d) PetscObjectComposeFunction(a,b,c,(PetscVoidFunction)(d))
#endif

extern PetscErrorCode  PetscObjectQueryFunction(PetscObject,const char[],void (**)(void));
extern PetscErrorCode  PetscObjectSetOptionsPrefix(PetscObject,const char[]);
extern PetscErrorCode  PetscObjectAppendOptionsPrefix(PetscObject,const char[]);
extern PetscErrorCode  PetscObjectPrependOptionsPrefix(PetscObject,const char[]);
extern PetscErrorCode  PetscObjectGetOptionsPrefix(PetscObject,const char*[]);
extern PetscErrorCode  PetscObjectAMSPublish(PetscObject);
extern PetscErrorCode  PetscObjectUnPublish(PetscObject);
extern PetscErrorCode  PetscObjectChangeTypeName(PetscObject,const char[]);
extern PetscErrorCode  PetscObjectRegisterDestroy(PetscObject);
extern PetscErrorCode  PetscObjectRegisterDestroyAll(void);
extern PetscErrorCode  PetscObjectName(PetscObject);
extern PetscErrorCode  PetscTypeCompare(PetscObject,const char[],PetscBool *);
extern PetscErrorCode  PetscTypeCompareAny(PetscObject,PetscBool*,const char[],...);
extern PetscErrorCode  PetscRegisterFinalize(PetscErrorCode (*)(void));
extern PetscErrorCode  PetscRegisterFinalizeAll(void);

/*
    Defines PETSc error handling.
*/
#include "petscerror.h"

/*S
     PetscOList - Linked list of PETSc objects, each accessable by string name

   Level: developer

   Notes: Used by PetscObjectCompose() and PetscObjectQuery() 

.seealso:  PetscOListAdd(), PetscOListDestroy(), PetscOListFind(), PetscObjectCompose(), PetscObjectQuery(), PetscFList
S*/
typedef struct _n_PetscOList *PetscOList;

extern PetscErrorCode  PetscOListDestroy(PetscOList*);
extern PetscErrorCode  PetscOListFind(PetscOList,const char[],PetscObject*);
extern PetscErrorCode  PetscOListReverseFind(PetscOList,PetscObject,char**,PetscBool*);
extern PetscErrorCode  PetscOListAdd(PetscOList *,const char[],PetscObject);
extern PetscErrorCode  PetscOListRemoveReference(PetscOList *,const char[]);
extern PetscErrorCode  PetscOListDuplicate(PetscOList,PetscOList *);

/*
    Dynamic library lists. Lists of names of routines in objects or in dynamic 
  link libraries that will be loaded as needed.
*/
extern PetscErrorCode  PetscFListAdd(PetscFList*,const char[],const char[],void (*)(void));
extern PetscErrorCode  PetscFListDestroy(PetscFList*);
extern PetscErrorCode  PetscFListFind(PetscFList,MPI_Comm,const char[],PetscBool,void (**)(void));
extern PetscErrorCode  PetscFListPrintTypes(MPI_Comm,FILE*,const char[],const char[],const char[],const char[],PetscFList,const char[]);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define    PetscFListAddDynamic(a,b,p,c) PetscFListAdd(a,b,p,0)
#else
#define    PetscFListAddDynamic(a,b,p,c) PetscFListAdd(a,b,p,(void (*)(void))c)
#endif
extern PetscErrorCode  PetscFListDuplicate(PetscFList,PetscFList *);
extern PetscErrorCode  PetscFListView(PetscFList,PetscViewer);
extern PetscErrorCode  PetscFListConcat(const char [],const char [],char []);
extern PetscErrorCode  PetscFListGet(PetscFList,char ***,int*);

/*
    Multiple dispatch operation function lists. Lists of names of routines with corresponding
    argument type names with function pointers or in dynamic link libraries that will be loaded 
    as needed.  Search on the op name and argument type names.
*/
extern PetscErrorCode  PetscOpFListAdd(MPI_Comm, PetscOpFList*,const char[],PetscVoidFunction, const char[], PetscInt, char*[]);
extern PetscErrorCode  PetscOpFListDestroy(PetscOpFList*);
extern PetscErrorCode  PetscOpFListFind(MPI_Comm, PetscOpFList, PetscVoidFunction*, const char[], PetscInt, char*[]);
extern PetscErrorCode  PetscOpFListView(PetscOpFList,PetscViewer);

/*S
     PetscDLLibrary - Linked list of dynamics libraries to search for functions

   Level: advanced

   --with-shared-libraries --with-dynamic-loading must be used with ./configure to use dynamic libraries

.seealso:  PetscDLLibraryOpen()
S*/
typedef struct _n_PetscDLLibrary *PetscDLLibrary;
extern PetscDLLibrary  PetscDLLibrariesLoaded;
extern PetscErrorCode  PetscDLLibraryAppend(MPI_Comm,PetscDLLibrary *,const char[]);
extern PetscErrorCode  PetscDLLibraryPrepend(MPI_Comm,PetscDLLibrary *,const char[]);
extern PetscErrorCode  PetscDLLibrarySym(MPI_Comm,PetscDLLibrary *,const char[],const char[],void **);
extern PetscErrorCode  PetscDLLibraryPrintPath(PetscDLLibrary);
extern PetscErrorCode  PetscDLLibraryRetrieve(MPI_Comm,const char[],char *,size_t,PetscBool  *);
extern PetscErrorCode  PetscDLLibraryOpen(MPI_Comm,const char[],PetscDLLibrary *);
extern PetscErrorCode  PetscDLLibraryClose(PetscDLLibrary);

/*
  PetscShell support.  Needs to be better documented.  
  Logically it is an extension of PetscDLLXXX, PetscObjectCompose, etc.
*/
#include "petscshell.h"

/*
     Useful utility routines
*/
extern PetscErrorCode  PetscSplitOwnership(MPI_Comm,PetscInt*,PetscInt*);
extern PetscErrorCode  PetscSplitOwnershipBlock(MPI_Comm,PetscInt,PetscInt*,PetscInt*);
extern PetscErrorCode  PetscSequentialPhaseBegin(MPI_Comm,PetscMPIInt);
extern PetscErrorCode  PetscSequentialPhaseEnd(MPI_Comm,PetscMPIInt);
extern PetscErrorCode  PetscBarrier(PetscObject);
extern PetscErrorCode  PetscMPIDump(FILE*);

/*
    PetscNot - negates a logical type value and returns result as a PetscBool 

    Notes: This is useful in cases like 
$     int        *a;
$     PetscBool  flag = PetscNot(a) 
     where !a does not return a PetscBool  because we cannot provide a cast from int to PetscBool  in C.
*/
#define PetscNot(a) ((a) ? PETSC_FALSE : PETSC_TRUE)

/*
    Defines basic graphics available from PETSc. 
*/
#include "petscdraw.h"

/*
    Defines the base data structures for all PETSc objects
*/
#include "petsc-private/petscimpl.h"


/*MC
    PetscErrorPrintf - Prints error messages.

   Synopsis:
     PetscErrorCode (*PetscErrorPrintf)(const char format[],...);

    Not Collective

    Input Parameters:
.   format - the usual printf() format string 

   Options Database Keys:
+    -error_output_stdout - cause error messages to be printed to stdout instead of the
         (default) stderr
-    -error_output_none to turn off all printing of error messages (does not change the way the 
          error is handled.)

   Notes: Use
$     PetscErrorPrintf = PetscErrorPrintfNone; to turn off all printing of error messages (does not change the way the 
$                        error is handled.) and
$     PetscErrorPrintf = PetscErrorPrintfDefault; to turn it back on
$        of you can use your own function

          Use
     PETSC_STDERR = FILE* obtained from a file open etc. to have stderr printed to the file. 
     PETSC_STDOUT = FILE* obtained from a file open etc. to have stdout printed to the file. 

          Use
      PetscPushErrorHandler() to provide your own error handler that determines what kind of messages to print

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

    Concepts: error messages^printing
    Concepts: printing^error messages

.seealso: PetscFPrintf(), PetscSynchronizedPrintf(), PetscHelpPrintf(), PetscPrintf(), PetscErrorHandlerPush(), PetscVFPrintf(), PetscHelpPrintf()
M*/
extern  PetscErrorCode (*PetscErrorPrintf)(const char[],...);

/*MC
    PetscHelpPrintf - Prints help messages.

   Synopsis:
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
extern  PetscErrorCode  (*PetscHelpPrintf)(MPI_Comm,const char[],...);

/*
     Defines PETSc profiling.
*/
#include "petsclog.h"

/*
          For locking, unlocking and destroying AMS memories associated with  PETSc objects. ams.h is included in petscviewer.h
*/
#if defined(PETSC_HAVE_AMS)
extern PetscBool  PetscAMSPublishAll;
#define PetscObjectTakeAccess(obj)  ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_take_access(((PetscObject)(obj))->amem))
#define PetscObjectGrantAccess(obj) ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_grant_access(((PetscObject)(obj))->amem))
#define PetscObjectDepublish(obj)   ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_destroy(((PetscObject)(obj))->amem));((PetscObject)(obj))->amem = -1;
#else
#define PetscObjectTakeAccess(obj)   0
#define PetscObjectGrantAccess(obj)  0
#define PetscObjectDepublish(obj)      0
#endif

/*
      Simple PETSc parallel IO for ASCII printing
*/
extern PetscErrorCode   PetscFixFilename(const char[],char[]);
extern PetscErrorCode   PetscFOpen(MPI_Comm,const char[],const char[],FILE**);
extern PetscErrorCode   PetscFClose(MPI_Comm,FILE*);
extern PetscErrorCode   PetscFPrintf(MPI_Comm,FILE*,const char[],...);
extern PetscErrorCode   PetscPrintf(MPI_Comm,const char[],...);
extern PetscErrorCode   PetscSNPrintf(char*,size_t,const char [],...);
extern PetscErrorCode   PetscSNPrintfCount(char*,size_t,const char [],size_t*,...);

/* These are used internally by PETSc ASCII IO routines*/
#include <stdarg.h>
extern PetscErrorCode   PetscVSNPrintf(char*,size_t,const char[],size_t*,va_list);
extern PetscErrorCode   (*PetscVFPrintf)(FILE*,const char[],va_list);
extern PetscErrorCode   PetscVFPrintfDefault(FILE*,const char[],va_list);

#if defined(PETSC_HAVE_MATLAB_ENGINE)
extern PetscErrorCode  PetscVFPrintf_Matlab(FILE*,const char[],va_list);
#endif

extern PetscErrorCode  PetscErrorPrintfDefault(const char [],...);
extern PetscErrorCode  PetscErrorPrintfNone(const char [],...);
extern PetscErrorCode  PetscHelpPrintfDefault(MPI_Comm,const char [],...);

#if defined(PETSC_HAVE_POPEN)
extern PetscErrorCode   PetscPOpen(MPI_Comm,const char[],const char[],const char[],FILE **);
extern PetscErrorCode   PetscPClose(MPI_Comm,FILE*);
#endif

extern PetscErrorCode   PetscSynchronizedPrintf(MPI_Comm,const char[],...);
extern PetscErrorCode   PetscSynchronizedFPrintf(MPI_Comm,FILE*,const char[],...);
extern PetscErrorCode   PetscSynchronizedFlush(MPI_Comm);
extern PetscErrorCode   PetscSynchronizedFGets(MPI_Comm,FILE*,size_t,char[]);
extern PetscErrorCode   PetscStartMatlab(MPI_Comm,const char[],const char[],FILE**);
extern PetscErrorCode   PetscStartJava(MPI_Comm,const char[],const char[],FILE**);
extern PetscErrorCode   PetscGetPetscDir(const char*[]);

extern PetscErrorCode   PetscPopUpSelect(MPI_Comm,const char*,const char*,int,const char**,int*);

/*S
     PetscContainer - Simple PETSc object that contains a pointer to any required data

   Level: advanced

.seealso:  PetscObject, PetscContainerCreate()
S*/
extern PetscClassId  PETSC_CONTAINER_CLASSID;
typedef struct _p_PetscContainer*  PetscContainer;
extern PetscErrorCode  PetscContainerGetPointer(PetscContainer,void **);
extern PetscErrorCode  PetscContainerSetPointer(PetscContainer,void *);
extern PetscErrorCode  PetscContainerDestroy(PetscContainer*);
extern PetscErrorCode  PetscContainerCreate(MPI_Comm,PetscContainer *);
extern PetscErrorCode  PetscContainerSetUserDestroy(PetscContainer, PetscErrorCode (*)(void*));

/*
   For use in debuggers 
*/
extern  PetscMPIInt PetscGlobalRank;
extern  PetscMPIInt PetscGlobalSize;
extern PetscErrorCode  PetscIntView(PetscInt,const PetscInt[],PetscViewer);
extern PetscErrorCode  PetscRealView(PetscInt,const PetscReal[],PetscViewer);
extern PetscErrorCode  PetscScalarView(PetscInt,const PetscScalar[],PetscViewer);

#if defined(PETSC_HAVE_MEMORY_H)
#include <memory.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>
#endif

#if defined(PETSC_HAVE_XMMINTRIN_H) && !defined(__CUDACC__)
#include <xmmintrin.h>
#endif
#if defined(PETSC_HAVE_STDINT_H)
#include <stdint.h>
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
PETSC_STATIC_INLINE PetscErrorCode  PetscMemcpy(void *a,const void *b,size_t n)
{
#if defined(PETSC_USE_DEBUG)
  unsigned long al = (unsigned long) a,bl = (unsigned long) b;
  unsigned long nl = (unsigned long) n;
  PetscFunctionBegin;
  if (n > 0 && !b) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy from a null pointer");
  if (n > 0 && !a) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy to a null pointer");
#else
  PetscFunctionBegin;
#endif
  if (a != b) {
#if defined(PETSC_USE_DEBUG)
    if ((al > bl && (al - bl) < nl) || (bl - al) < nl)  SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Memory regions overlap: either use PetscMemmov()\n\
              or make sure your copy regions and lengths are correct. \n\
              Length (bytes) %ld first address %ld second address %ld",nl,al,bl);
#endif
#if (defined(PETSC_PREFER_DCOPY_FOR_MEMCPY) || defined(PETSC_PREFER_COPY_FOR_MEMCPY) || defined(PETSC_PREFER_FORTRAN_FORMEMCPY))
   if (!(((long) a) % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      size_t len = n/sizeof(PetscScalar);
#if defined(PETSC_PREFER_DCOPY_FOR_MEMCPY)
      PetscBLASInt one = 1,blen = PetscBLASIntCast(len);
      BLAScopy_(&blen,(PetscScalar *)b,&one,(PetscScalar *)a,&one);
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
    Allows accessing MATLAB Engine
*/
#include "petscmatlab.h"

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
#define PETSC_EXTERN_C extern "C"
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define PETSC_EXTERN_C
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
       with --with-scalar-type=real,complex --with-precision=single,double,longdouble,int,matsingle


   Level: beginner

.seealso: PetscReal, PassiveReal, PassiveScalar, MPIU_SCALAR, PetscInt
M*/

/*MC
    PetscReal - PETSc type that represents a real number version of PetscScalar

   Level: beginner

.seealso: PetscScalar, PassiveReal, PassiveScalar
M*/

/*MC
    PassiveScalar - PETSc type that represents a PetscScalar
   Level: beginner

    This is the same as a PetscScalar except in code that is automatically differentiated it is
   treated as a constant (not an indendent or dependent variable)

.seealso: PetscReal, PassiveReal, PetscScalar
M*/

/*MC
    PassiveReal - PETSc type that represents a PetscReal

   Level: beginner

    This is the same as a PetscReal except in code that is automatically differentiated it is
   treated as a constant (not an indendent or dependent variable)

.seealso: PetscScalar, PetscReal, PassiveScalar
M*/

/*MC
    MPIU_SCALAR - MPI datatype corresponding to PetscScalar

   Level: beginner

    Note: In MPI calls that require an MPI datatype that matches a PetscScalar or array of PetscScalars
          pass this value

.seealso: PetscReal, PassiveReal, PassiveScalar, PetscScalar, MPIU_INT
M*/

#if defined(PETSC_HAVE_MPIIO)
#if !defined(PETSC_WORDS_BIGENDIAN)
extern PetscErrorCode MPIU_File_write_all(MPI_File,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
extern PetscErrorCode MPIU_File_read_all(MPI_File,void*,PetscMPIInt,MPI_Datatype,MPI_Status*);
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
/* On 32 bit systems HDF5 is limited by size of integer, because hsize_t is defined as size_t */
#define PETSC_HDF5_INT_MAX  2147483647
#define PETSC_HDF5_INT_MIN -2147483647

#if defined(PETSC_USE_64BIT_INDICES)
#define PetscMPIIntCheck(a)  if ((a) > PETSC_MPI_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Message too long for MPI")
#define PetscBLASIntCheck(a)  if ((a) > PETSC_BLAS_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for BLAS/LAPACK")
#define PetscMPIIntCast(a) (PetscMPIInt)(a);PetscMPIIntCheck(a)
#define PetscBLASIntCast(a) (PetscBLASInt)(a);PetscBLASIntCheck(a)

#if (PETSC_SIZEOF_SIZE_T == 4)
#define PetscHDF5IntCheck(a)  if ((a) > PETSC_HDF5_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for HDF5")
#define PetscHDF5IntCast(a) (hsize_t)(a);PetscHDF5IntCheck(a)
#else
#define PetscHDF5IntCheck(a)
#define PetscHDF5IntCast(a) a
#endif

#else
#define PetscMPIIntCheck(a) 
#define PetscBLASIntCheck(a) 
#define PetscHDF5IntCheck(a)
#define PetscMPIIntCast(a) a
#define PetscBLASIntCast(a) a
#define PetscHDF5IntCast(a) a
#endif  

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

/* Special support for C++ */
#include "petscsys.hh"

/*MC

    UsingFortran - Fortran can be used with PETSc in four distinct approaches

$    1) classic Fortran 77 style
$#include "finclude/petscXXX.h" to work with material from the XXX component of PETSc
$       XXX variablename
$      You cannot use this approach if you wish to use the Fortran 90 specific PETSc routines
$      which end in F90; such as VecGetArrayF90()
$
$    2) classic Fortran 90 style
$#include "finclude/petscXXX.h" 
$#include "finclude/petscXXX.h90" to work with material from the XXX component of PETSc
$       XXX variablename
$
$    3) Using Fortran modules
$#include "finclude/petscXXXdef.h" 
$         use petscXXXX
$       XXX variablename
$
$    4) Use Fortran modules and Fortran data types for PETSc types
$#include "finclude/petscXXXdef.h" 
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

    Developer Notes: The finclude/petscXXXdef.h contain all the #defines (would be typedefs in C code) these
     automatically include their predecessors; for example finclude/petscvecdef.h includes finclude/petscisdef.h

     The finclude/petscXXXX.h contain all the parameter statements for that package. These automatically include
     their finclude/petscXXXdef.h file but DO NOT automatically include their predecessors;  for example 
     finclude/petscvec.h does NOT automatically include finclude/petscis.h

     The finclude/ftn-custom/petscXXXdef.h90 are not intended to be used directly in code, they define the
     Fortran data type type(XXX) (for example type(Vec)) when PETSc is ./configure with the --with-fortran-datatypes option.

     The finclude/ftn-custom/petscXXX.h90 (not included directly by code) contain interface definitions for
     the PETSc Fortran stubs that have different bindings then their C version (for example VecGetArrayF90).

     The finclude/ftn-auto/petscXXX.h90 (not included directly by code) contain interface definitions generated
     automatically by "make allfortranstubs".

     The finclude/petscXXX.h90 includes the custom finclude/ftn-custom/petscXXX.h90 and if ./configure 
     was run with --with-fortran-interfaces it also includes the finclude/ftn-auto/petscXXX.h90 These DO NOT automatically
     include their predecessors

    Level: beginner

M*/

extern PetscErrorCode  PetscGetArchType(char[],size_t);
extern PetscErrorCode  PetscGetHostName(char[],size_t);
extern PetscErrorCode  PetscGetUserName(char[],size_t);
extern PetscErrorCode  PetscGetProgramName(char[],size_t);
extern PetscErrorCode  PetscSetProgramName(const char[]);
extern PetscErrorCode  PetscGetDate(char[],size_t);

extern PetscErrorCode  PetscSortInt(PetscInt,PetscInt[]);
extern PetscErrorCode  PetscSortRemoveDupsInt(PetscInt*,PetscInt[]);
extern PetscErrorCode  PetscSortIntWithPermutation(PetscInt,const PetscInt[],PetscInt[]);
extern PetscErrorCode  PetscSortStrWithPermutation(PetscInt,const char*[],PetscInt[]);
extern PetscErrorCode  PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[]);
extern PetscErrorCode  PetscSortIntWithArrayPair(PetscInt,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode  PetscSortMPIIntWithArray(PetscMPIInt,PetscMPIInt[],PetscMPIInt[]);
extern PetscErrorCode  PetscSortIntWithScalarArray(PetscInt,PetscInt[],PetscScalar[]);
extern PetscErrorCode  PetscSortReal(PetscInt,PetscReal[]);
extern PetscErrorCode  PetscSortRealWithPermutation(PetscInt,const PetscReal[],PetscInt[]);
extern PetscErrorCode  PetscSortSplit(PetscInt,PetscInt,PetscScalar[],PetscInt[]);
extern PetscErrorCode  PetscSortSplitReal(PetscInt,PetscInt,PetscReal[],PetscInt[]);
extern PetscErrorCode  PetscProcessTree(PetscInt,const PetscBool [],const PetscInt[],PetscInt*,PetscInt**,PetscInt**,PetscInt**,PetscInt**);
extern PetscErrorCode  PetscMergeIntArrayPair(PetscInt,const PetscInt*,const PetscInt*,PetscInt,const PetscInt*,const PetscInt*,PetscInt*,PetscInt**,PetscInt**);

extern PetscErrorCode  PetscSetDisplay(void);
extern PetscErrorCode  PetscGetDisplay(char[],size_t);

/*J
    PetscRandomType - String with the name of a PETSc randomizer
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:myrandcreate()

   Level: beginner

   Notes: to use the SPRNG you must have ./configure PETSc
   with the option --download-sprng

.seealso: PetscRandomSetType(), PetscRandom
J*/
#define PetscRandomType char*
#define PETSCRAND       "rand"
#define PETSCRAND48     "rand48"
#define PETSCSPRNG      "sprng"          

/* Logging support */
extern  PetscClassId PETSC_RANDOM_CLASSID;

extern PetscErrorCode  PetscRandomInitializePackage(const char[]);

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

  Concepts: random numbers

.seealso:  PetscRandomCreate(), PetscRandomGetValue(), PetscRandomType
S*/
typedef struct _p_PetscRandom*   PetscRandom;

/* Dynamic creation and loading functions */
extern PetscFList PetscRandomList;
extern PetscBool  PetscRandomRegisterAllCalled;

extern PetscErrorCode  PetscRandomRegisterAll(const char []);
extern PetscErrorCode  PetscRandomRegister(const char[],const char[],const char[],PetscErrorCode (*)(PetscRandom));
extern PetscErrorCode  PetscRandomRegisterDestroy(void);
extern PetscErrorCode  PetscRandomSetType(PetscRandom, const PetscRandomType);
extern PetscErrorCode  PetscRandomSetFromOptions(PetscRandom);
extern PetscErrorCode  PetscRandomGetType(PetscRandom, const PetscRandomType*);
extern PetscErrorCode  PetscRandomViewFromOptions(PetscRandom,char*);
extern PetscErrorCode  PetscRandomView(PetscRandom,PetscViewer);

/*MC
  PetscRandomRegisterDynamic - Adds a new PetscRandom component implementation

  Synopsis:
  PetscErrorCode PetscRandomRegisterDynamic(const char *name, const char *path, const char *func_name, PetscErrorCode (*create_func)(PetscRandom))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of routine to create method context
- create_func - The creation routine itself

  Notes:
  PetscRandomRegisterDynamic() may be called multiple times to add several user-defined randome number generators

  If dynamic libraries are used, then the fourth input argument (routine_create) is ignored.

  Sample usage:
.vb
    PetscRandomRegisterDynamic("my_rand","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyPetscRandomtorCreate", MyPetscRandomtorCreate);
.ve

  Then, your random type can be chosen with the procedural interface via
.vb
    PetscRandomCreate(MPI_Comm, PetscRandom *);
    PetscRandomSetType(PetscRandom,"my_random_name");
.ve
   or at runtime via the option
.vb
    -random_type my_random_name
.ve

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.

         For an example of the code needed to interface your own random number generator see
         src/sys/random/impls/rand/rand.c
        
  Level: advanced

.keywords: PetscRandom, register
.seealso: PetscRandomRegisterAll(), PetscRandomRegisterDestroy(), PetscRandomRegister()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscRandomRegisterDynamic(a,b,c,d) PetscRandomRegister(a,b,c,0)
#else
#define PetscRandomRegisterDynamic(a,b,c,d) PetscRandomRegister(a,b,c,d)
#endif

extern PetscErrorCode  PetscRandomCreate(MPI_Comm,PetscRandom*);
extern PetscErrorCode  PetscRandomGetValue(PetscRandom,PetscScalar*);
extern PetscErrorCode  PetscRandomGetValueReal(PetscRandom,PetscReal*);
extern PetscErrorCode  PetscRandomGetInterval(PetscRandom,PetscScalar*,PetscScalar*);
extern PetscErrorCode  PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar);
extern PetscErrorCode  PetscRandomSetSeed(PetscRandom,unsigned long);
extern PetscErrorCode  PetscRandomGetSeed(PetscRandom,unsigned long *);
extern PetscErrorCode  PetscRandomSeed(PetscRandom);
extern PetscErrorCode  PetscRandomDestroy(PetscRandom*);

extern PetscErrorCode  PetscGetFullPath(const char[],char[],size_t);
extern PetscErrorCode  PetscGetRelativePath(const char[],char[],size_t);
extern PetscErrorCode  PetscGetWorkingDirectory(char[],size_t);
extern PetscErrorCode  PetscGetRealPath(const char[],char[]);
extern PetscErrorCode  PetscGetHomeDirectory(char[],size_t);
extern PetscErrorCode  PetscTestFile(const char[],char,PetscBool *);
extern PetscErrorCode  PetscTestDirectory(const char[],char,PetscBool *);

extern PetscErrorCode  PetscBinaryRead(int,void*,PetscInt,PetscDataType);
extern PetscErrorCode  PetscBinarySynchronizedRead(MPI_Comm,int,void*,PetscInt,PetscDataType);
extern PetscErrorCode  PetscBinarySynchronizedWrite(MPI_Comm,int,void*,PetscInt,PetscDataType,PetscBool );
extern PetscErrorCode  PetscBinaryWrite(int,void*,PetscInt,PetscDataType,PetscBool );
extern PetscErrorCode  PetscBinaryOpen(const char[],PetscFileMode,int *);
extern PetscErrorCode  PetscBinaryClose(int);
extern PetscErrorCode  PetscSharedTmp(MPI_Comm,PetscBool  *);
extern PetscErrorCode  PetscSharedWorkingDirectory(MPI_Comm,PetscBool  *);
extern PetscErrorCode  PetscGetTmp(MPI_Comm,char[],size_t);
extern PetscErrorCode  PetscFileRetrieve(MPI_Comm,const char[],char[],size_t,PetscBool *);
extern PetscErrorCode  PetscLs(MPI_Comm,const char[],char[],size_t,PetscBool *);
extern PetscErrorCode  PetscOpenSocket(char*,int,int*);
extern PetscErrorCode  PetscWebServe(MPI_Comm,int);

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
extern PetscErrorCode  PetscBinarySeek(int,off_t,PetscBinarySeekType,off_t*);
extern PetscErrorCode  PetscBinarySynchronizedSeek(MPI_Comm,int,off_t,PetscBinarySeekType,off_t*);
extern PetscErrorCode  PetscByteSwap(void *,PetscDataType,PetscInt);

extern PetscErrorCode  PetscSetDebugTerminal(const char[]);
extern PetscErrorCode  PetscSetDebugger(const char[],PetscBool );
extern PetscErrorCode  PetscSetDefaultDebugger(void);
extern PetscErrorCode  PetscSetDebuggerFromString(const char*);
extern PetscErrorCode  PetscAttachDebugger(void);
extern PetscErrorCode  PetscStopForDebugger(void);

extern PetscErrorCode  PetscGatherNumberOfMessages(MPI_Comm,const PetscMPIInt[],const PetscMPIInt[],PetscMPIInt*);
extern PetscErrorCode  PetscGatherMessageLengths(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],PetscMPIInt**,PetscMPIInt**);
extern PetscErrorCode  PetscGatherMessageLengths2(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],const PetscMPIInt[],PetscMPIInt**,PetscMPIInt**,PetscMPIInt**);
extern PetscErrorCode  PetscPostIrecvInt(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],const PetscMPIInt[],PetscInt***,MPI_Request**);
extern PetscErrorCode  PetscPostIrecvScalar(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscMPIInt[],const PetscMPIInt[],PetscScalar***,MPI_Request**);

extern PetscErrorCode  PetscSSEIsEnabled(MPI_Comm,PetscBool  *,PetscBool  *);

/*E
  InsertMode - Whether entries are inserted or added into vectors or matrices

  Level: beginner

.seealso: VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(),
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()
E*/
 typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES, INSERT_ALL_VALUES, ADD_ALL_VALUES} InsertMode;

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

/*S
   PetscSubcomm - Context of MPI subcommunicators, used by PCREDUNDANT

   Level: advanced

   Concepts: communicator, create
S*/
typedef struct _n_PetscSubcomm* PetscSubcomm;

struct _n_PetscSubcomm { 
  MPI_Comm   parent;      /* parent communicator */
  MPI_Comm   dupparent;   /* duplicate parent communicator, under which the processors of this subcomm have contiguous rank */
  MPI_Comm   comm;        /* this communicator */
  PetscInt   n;           /* num of subcommunicators under the parent communicator */
  PetscInt   color;       /* color of processors belong to this communicator */
};

typedef enum {PETSC_SUBCOMM_GENERAL=0,PETSC_SUBCOMM_CONTIGUOUS=1,PETSC_SUBCOMM_INTERLACED=2} PetscSubcommType;
extern const char *PetscSubcommTypes[];

extern PetscErrorCode  PetscSubcommCreate(MPI_Comm,PetscSubcomm*);
extern PetscErrorCode  PetscSubcommDestroy(PetscSubcomm*);
extern PetscErrorCode  PetscSubcommSetNumber(PetscSubcomm,PetscInt);
extern PetscErrorCode  PetscSubcommSetType(PetscSubcomm,PetscSubcommType);
extern PetscErrorCode  PetscSubcommSetTypeGeneral(PetscSubcomm,PetscMPIInt,PetscMPIInt,PetscMPIInt);

#include <petscctable.h>

PETSC_EXTERN_CXX_END

/* Reset __FUNCT__ in case the user does not define it themselves */
#undef __FUNCT__
#define __FUNCT__ "User provided function"

#endif
