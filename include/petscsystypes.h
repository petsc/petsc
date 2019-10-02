#if !defined(PETSCSYSTYPES_H)
#define PETSCSYSTYPES_H

#include <petscconf.h>
#include <petscfix.h>

/*MC
    PetscErrorCode - datatype used for return error code from almost all PETSc functions

    Level: beginner

.seealso: CHKERRQ, SETERRQ
M*/
typedef int PetscErrorCode;

/*MC

    PetscClassId - A unique id used to identify each PETSc class.

    Notes:
    Use PetscClassIdRegister() to obtain a new value for a new class being created. Usually
         XXXInitializePackage() calls it for each class it defines.

    Developer Notes:
    Internal integer stored in the _p_PetscObject data structure.
         These are all computed by an offset from the lowest one, PETSC_SMALLEST_CLASSID.

    Level: developer

.seealso: PetscClassIdRegister(), PetscLogEventRegister(), PetscHeaderCreate()
M*/
typedef int PetscClassId;

/*MC
    PetscMPIInt - datatype used to represent 'int' parameters to MPI functions.

    Level: intermediate

    Notes:
    usually this is the same as PetscInt, but if PETSc was built with --with-64-bit-indices but
           standard C/Fortran integers are 32 bit then this is NOT the same as PetscInt; it remains 32 bit.

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

typedef short PetscShort;
typedef char  PetscChar;
typedef float PetscFloat;

/*MC
  PetscInt - PETSc type that represents an integer, used primarily to
      represent size of arrays and indexing into arrays. Its size can be configured with the option --with-64-bit-indices to be either 32-bit (default) or 64-bit.

  Notes:
  For MPI calls that require datatypes, use MPIU_INT as the datatype for PetscInt. It will automatically work correctly regardless of the size of PetscInt.

  Level: beginner

.seealso: PetscBLASInt, PetscMPIInt, PetscReal, PetscScalar, PetscComplex, PetscInt, MPIU_REAL, MPIU_SCALAR, MPIU_COMPLEX, MPIU_INT
M*/

#if defined(PETSC_HAVE_STDINT_H)
#  include <stdint.h>
#endif
#if defined (PETSC_HAVE_INTTYPES_H)
#  if !defined(__STDC_FORMAT_MACROS)
#    define __STDC_FORMAT_MACROS /* required for using PRId64 from c++ */
#  endif
#  include <inttypes.h>
#  if !defined(PRId64)
#    define PRId64 "ld"
#  endif
#endif

#if defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T) /* MPI_INT64_T is not guaranteed to be a macro */
   typedef int64_t PetscInt64;
#elif (PETSC_SIZEOF_LONG_LONG == 8)
   typedef long long PetscInt64;
#elif defined(PETSC_HAVE___INT64)
   typedef __int64 PetscInt64;
#else
#  error "cannot determine PetscInt64 type"
#endif

#if defined(PETSC_USE_64BIT_INDICES)
   typedef PetscInt64 PetscInt;
#else
   typedef int PetscInt;
#endif

/*MC
   PetscBLASInt - datatype used to represent 'int' parameters to BLAS/LAPACK functions.

   Notes:
    Usually this is the same as PetscInt, but if PETSc was built with --with-64-bit-indices but
           standard C/Fortran integers are 32 bit then this is NOT the same as PetscInt it remains 32 bit
           (except on very rare BLAS/LAPACK implementations that support 64 bit integers see the notes below).

    PetscErrorCode PetscBLASIntCast(a,&b) checks if the given PetscInt a will fit in a PetscBLASInt, if not it
      generates a PETSC_ERR_ARG_OUTOFRANGE error

   Installation Notes:
    ./configure automatically determines the size of the integers used by BLAS/LAPACK except when --with-batch is used
    in that situation one must know (by some other means) if the integers used by BLAS/LAPACK are 64 bit and if so pass the flag --known-64-bit-blas-indice

    MATLAB ships with BLAS and LAPACK that use 64 bit integers, for example if you run ./configure with, the option
     --with-blaslapack-lib=[/Applications/MATLAB_R2010b.app/bin/maci64/libmwblas.dylib,/Applications/MATLAB_R2010b.app/bin/maci64/libmwlapack.dylib]

    MKL ships with both 32 and 64 bit integer versions of the BLAS and LAPACK. If you pass the flag -with-64-bit-blas-indices PETSc will link
    against the 64 bit version, otherwise it use the 32 bit version

    OpenBLAS can be built to use 64 bit integers. The ./configure options --download-openblas -with-64-bit-blas-indices will build a 64 bit integer version

    External packages such as hypre, ML, SuperLU etc do not provide any support for passing 64 bit integers to BLAS/LAPACK so cannot
    be used with PETSc when PETSc links against 64 bit integer BLAS/LAPACK. ./configure will generate an error if you attempt to link PETSc against any of
    these external libraries while using 64 bit integer BLAS/LAPACK.

   Level: intermediate

.seealso: PetscMPIInt, PetscInt, PetscBLASIntCast()

M*/
#if defined(PETSC_HAVE_64BIT_BLAS_INDICES)
   typedef PetscInt64 PetscBLASInt;
#else
   typedef int PetscBLASInt;
#endif

/*E
    PetscBool  - Logical variable. Actually an int in C and a logical in Fortran.

   Level: beginner

   Developer Note:
   Why have PetscBool , why not use bool in C? The problem is that K and R C, C99 and C++ all have different mechanisms for
      boolean values. It is not easy to have a simple macro that that will work properly in all circumstances with all three mechanisms.

.seealso: PETSC_TRUE, PETSC_FALSE, PetscNot()
E*/
typedef enum { PETSC_FALSE,PETSC_TRUE } PetscBool;

/*MC
   PetscReal - PETSc type that represents a real number version of PetscScalar


   Notes:
   For MPI calls that require datatypes, use MPIU_REAL as the datatype for PetscScalar and MPIU_SUM, MPIU_MAX, etc. for operations.
          They will automatically work correctly regardless of the size of PetscReal.

          See PetscScalar for details on how to ./configure the size of PetscReal.

   Level: beginner

.seealso: PetscScalar, PetscComplex, PetscInt, MPIU_REAL, MPIU_SCALAR, MPIU_COMPLEX, MPIU_INT
M*/

#if defined(PETSC_USE_REAL_SINGLE)
   typedef float PetscReal;
#elif defined(PETSC_USE_REAL_DOUBLE)
   typedef double PetscReal;
#elif defined(PETSC_USE_REAL___FLOAT128)
#  if defined(__cplusplus)
     extern "C" {
#  endif
#  include <quadmath.h>
#  if defined(__cplusplus)
     }
#  endif
   typedef __float128 PetscReal;
#elif defined(PETSC_USE_REAL___FP16)
   typedef __fp16 PetscReal;
#endif /* PETSC_USE_REAL_* */

/*MC
   PetscComplex - PETSc type that represents a complex number with precision matching that of PetscReal.

   Synopsis:
   #include <petscsys.h>
   PetscComplex number = 1. + 2.*PETSC_i;

   Notes:
   For MPI calls that require datatypes, use MPIU_COMPLEX as the datatype for PetscComplex and MPIU_SUM etc for operations.
          They will automatically work correctly regardless of the size of PetscComplex.

          See PetscScalar for details on how to ./configure the size of PetscReal

          Complex numbers are automatically available if PETSc was able to find a working complex implementation

   Level: beginner

.seealso: PetscReal, PetscScalar, PetscComplex, PetscInt, MPIU_REAL, MPIU_SCALAR, MPIU_COMPLEX, MPIU_INT, PETSC_i
M*/

#if defined(__cplusplus) && defined(PETSC_HAVE_CXX_COMPLEX) && !defined(PETSC_USE_REAL___FLOAT128)
#  if !defined(PETSC_SKIP_COMPLEX)
     /* C++ support of complex number */
#    define PETSC_HAVE_COMPLEX 1
#    if defined(PETSC_HAVE_CUDA) && __CUDACC_VER_MAJOR__ > 6
       /* complex headers in thrust only available in CUDA 7.0 and above */
#      define petsccomplexlib thrust
#      include <thrust/complex.h>
#    else
#      define petsccomplexlib std
#      include <complex>
#    endif
#    if defined(PETSC_USE_REAL_SINGLE)
       typedef petsccomplexlib::complex<float> PetscComplex;
#    elif defined(PETSC_USE_REAL_DOUBLE)
       typedef petsccomplexlib::complex<double> PetscComplex;
#    elif defined(PETSC_USE_REAL___FLOAT128)
       typedef petsccomplexlib::complex<__float128> PetscComplex; /* Notstandard and not expected to work, use __complex128 */
#    endif  /* PETSC_USE_REAL_ */
#  endif  /* ! PETSC_SKIP_COMPLEX */
#  if !defined(PETSC_SKIP_CXX_COMPLEX_FIX)
#    include <petsccxxcomplexfix.h>
#  endif /* ! PETSC_SKIP_CXX_COMPLEX_FIX */
#elif defined(PETSC_HAVE_C99_COMPLEX) && !defined(PETSC_USE_REAL___FP16)
#  if !defined(PETSC_SKIP_COMPLEX)
#    define PETSC_HAVE_COMPLEX 1
#    include <complex.h>
#    if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL___FP16)
       typedef float _Complex PetscComplex;
#    elif defined(PETSC_USE_REAL_DOUBLE)
       typedef double _Complex PetscComplex;
#    elif defined(PETSC_USE_REAL___FLOAT128)
       typedef __complex128 PetscComplex;
#    endif /* PETSC_USE_REAL_* */
#  endif /* !PETSC_SKIP_COMPLEX */
#elif (defined(PETSC_USE_COMPLEX) && !defined(PETSC_SKIP_COMPLEX))
#  error "PETSc was configured --with-scalar-type=complex, but a language-appropriate complex library is not available"
#endif /* !PETSC_SKIP_COMPLEX */

/*MC
   PetscScalar - PETSc type that represents either a double precision real number, a double precision
       complex number, a single precision real number, a __float128 real or complex or a __fp16 real - if the code is configured
       with --with-scalar-type=real,complex --with-precision=single,double,__float128,__fp16

   Notes:
   For MPI calls that require datatypes, use MPIU_SCALAR as the datatype for PetscScalar and MPIU_SUM, MPIU_MAX etc for operations. They will automatically work correctly regardless of the size of PetscScalar.

   Level: beginner

.seealso: PetscReal, PetscComplex, PetscInt, MPIU_REAL, MPIU_SCALAR, MPIU_COMPLEX, MPIU_INT, PetscRealPart(), PetscImaginaryPart()
M*/

#if (defined(PETSC_USE_COMPLEX) && !defined(PETSC_SKIP_COMPLEX))
   typedef PetscComplex PetscScalar;
#else /* PETSC_USE_COMPLEX */
   typedef PetscReal PetscScalar;
#endif /* PETSC_USE_COMPLEX */

/*E
    PetscCopyMode  - Determines how an array or PetscObject passed to certain functions is copied or retained by the aggregate PetscObject

   Level: beginner

   For the array input:
$   PETSC_COPY_VALUES - the array values are copied into new space, the user is free to reuse or delete the passed in array
$   PETSC_OWN_POINTER - the array values are NOT copied, the object takes ownership of the array and will free it later, the user cannot change or
$                       delete the array. The array MUST have been obtained with PetscMalloc(). Hence this mode cannot be used in Fortran.
$   PETSC_USE_POINTER - the array values are NOT copied, the object uses the array but does NOT take ownership of the array. The user cannot use
$                       the array but the user must delete the array after the object is destroyed.

   For the PetscObject input:
$   PETSC_COPY_VALUES - the input PetscObject is cloned into the aggregate PetscObject; the user is free to reuse/modify the input PetscObject without side effects.
$   PETSC_OWN_POINTER - the input PetscObject is referenced by pointer (with reference count), thus should not be modified by the user. (Modification may cause errors or unintended side-effects in this or a future version of PETSc.)
   For either case above, the input PetscObject should be destroyed by the user when no longer needed (the aggregate object increases its reference count).
$   PETSC_USE_POINTER - invalid for PetscObject inputs.

E*/
typedef enum {PETSC_COPY_VALUES, PETSC_OWN_POINTER, PETSC_USE_POINTER} PetscCopyMode;

/*MC
    PETSC_FALSE - False value of PetscBool

    Level: beginner

    Note:
    Zero integer

.seealso: PetscBool, PETSC_TRUE
M*/

/*MC
    PETSC_TRUE - True value of PetscBool

    Level: beginner

    Note:
    Nonzero integer

.seealso: PetscBool, PETSC_FALSE
M*/

/*MC
    PetscLogDouble - Used for logging times

  Notes:
  Contains double precision numbers that are not used in the numerical computations, but rather in logging, timing etc.

  Level: developer

M*/
typedef double PetscLogDouble;

/*E
    PetscDataType - Used for handling different basic data types.

   Level: beginner

   Notes:
   Use of this should be avoided if one can directly use MPI_Datatype instead.

   Developer comment:
   It would be nice if we could always just use MPI Datatypes, why can we not?

   If you change any values in PetscDatatype make sure you update their usage in
   share/petsc/matlab/PetscBagRead.m

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscDataTypeToMPIDataType(),
          PetscDataTypeGetSize()

E*/
typedef enum {PETSC_DATATYPE_UNKNOWN = 0,
              PETSC_DOUBLE = 1, PETSC_COMPLEX = 2, PETSC_LONG = 3, PETSC_SHORT = 4, PETSC_FLOAT = 5,
              PETSC_CHAR = 6, PETSC_BIT_LOGICAL = 7, PETSC_ENUM = 8, PETSC_BOOL = 9, PETSC___FLOAT128 = 10,
              PETSC_OBJECT = 11, PETSC_FUNCTION = 12, PETSC_STRING = 13, PETSC___FP16 = 14, PETSC_STRUCT = 15,
              PETSC_INT = 16, PETSC_INT64 = 17} PetscDataType;

#if defined(PETSC_USE_REAL_SINGLE)
#  define PETSC_REAL PETSC_FLOAT
#elif defined(PETSC_USE_REAL_DOUBLE)
#  define PETSC_REAL PETSC_DOUBLE
#elif defined(PETSC_USE_REAL___FLOAT128)
#  define PETSC_REAL PETSC___FLOAT128
#elif defined(PETSC_USE_REAL___FP16)
#  define PETSC_REAL PETSC___FP16
#else
#  define PETSC_REAL PETSC_DOUBLE
#endif

#if defined(PETSC_USE_COMPLEX)
#  define PETSC_SCALAR PETSC_COMPLEX
#else
#  define PETSC_SCALAR PETSC_REAL
#endif

#define PETSC_FORTRANADDR PETSC_LONG

/*S
    PetscToken - 'Token' used for managing tokenizing strings

  Level: intermediate

.seealso: PetscTokenCreate(), PetscTokenFind(), PetscTokenDestroy()
S*/
typedef struct _p_PetscToken* PetscToken;

/*S
     PetscObject - any PETSc object, PetscViewer, Mat, Vec, KSP etc

   Level: beginner

   Note:
   This is the base class from which all PETSc objects are derived from.

.seealso:  PetscObjectDestroy(), PetscObjectView(), PetscObjectGetName(), PetscObjectSetName(), PetscObjectReference(), PetscObjectDereference()
S*/
typedef struct _p_PetscObject* PetscObject;

/*MC
    PetscObjectId - unique integer Id for a PetscObject

    Level: developer

    Notes:
    Unlike pointer values, object ids are never reused.

.seealso: PetscObjectState, PetscObjectGetId()
M*/
#if defined(PETSC_USING_F90) && !defined(PETSC_USE_FORTRANKIND) /* compaq F90 */
   typedef int PetscObjectId;
#else
   typedef PetscInt64 PetscObjectId;
#endif

/*MC
    PetscObjectState - integer state for a PetscObject

    Level: developer

    Notes:
    Object state is always-increasing and (for objects that track state) can be used to determine if an object has
    changed since the last time you interacted with it.  It is 64-bit so that it will not overflow for a very long time.

.seealso: PetscObjectId, PetscObjectStateGet(), PetscObjectStateIncrease(), PetscObjectStateSet()
M*/
#if defined(PETSC_USING_F90) && !defined(PETSC_USE_FORTRANKIND) /* compaq F90 */
   typedef int PetscObjectState;
#else
   typedef PetscInt64 PetscObjectState;
#endif

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

typedef void* PetscDLHandle;
typedef enum {PETSC_DL_DECIDE=0,PETSC_DL_NOW=1,PETSC_DL_LOCAL=2} PetscDLMode;

/*S
     PetscObjectList - Linked list of PETSc objects, each accessable by string name

   Level: developer

   Notes:
   Used by PetscObjectCompose() and PetscObjectQuery()

.seealso:  PetscObjectListAdd(), PetscObjectListDestroy(), PetscObjectListFind(), PetscObjectCompose(), PetscObjectQuery(), PetscFunctionList
S*/
typedef struct _n_PetscObjectList *PetscObjectList;

/*S
     PetscDLLibrary - Linked list of dynamics libraries to search for functions

   Level: advanced

.seealso:  PetscDLLibraryOpen()
S*/
typedef struct _n_PetscDLLibrary *PetscDLLibrary;

/*S
     PetscContainer - Simple PETSc object that contains a pointer to any required data

   Level: advanced

.seealso:  PetscObject, PetscContainerCreate()
S*/
typedef struct _p_PetscContainer*  PetscContainer;

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

.seealso:  PetscRandomCreate(), PetscRandomGetValue(), PetscRandomType
S*/
typedef struct _p_PetscRandom*   PetscRandom;

/*
   In binary files variables are stored using the following lengths,
  regardless of how they are stored in memory on any one particular
  machine. Use these rather then sizeof() in computing sizes for
  PetscBinarySeek().
*/
#define PETSC_BINARY_INT_SIZE    (32/8)
#define PETSC_BINARY_FLOAT_SIZE  (32/8)
#define PETSC_BINARY_CHAR_SIZE   (8/8)
#define PETSC_BINARY_SHORT_SIZE  (16/8)
#define PETSC_BINARY_DOUBLE_SIZE (64/8)
#define PETSC_BINARY_SCALAR_SIZE sizeof(PetscScalar)

/*E
  PetscBinarySeekType - argument to PetscBinarySeek()

  Level: advanced

.seealso: PetscBinarySeek(), PetscBinarySynchronizedSeek()
E*/
typedef enum {PETSC_BINARY_SEEK_SET = 0,PETSC_BINARY_SEEK_CUR = 1,PETSC_BINARY_SEEK_END = 2} PetscBinarySeekType;

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

/*E
  InsertMode - Whether entries are inserted or added into vectors or matrices

  Level: beginner

.seealso: VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(),
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()
E*/
 typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES, MIN_VALUES, INSERT_ALL_VALUES, ADD_ALL_VALUES, INSERT_BC_VALUES, ADD_BC_VALUES} InsertMode;

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

/*MC
    MIN_VALUES - Puts the minimal of the scattered/gathered value and the current value into each location

    Level: beginner

.seealso: InsertMode, VecScatterBegin(), VecScatterEnd(), ADD_VALUES, INSERT_VALUES

M*/


/*S
   PetscSubcomm - A decomposition of an MPI communicator into subcommunicators

   Notes:
   After a call to PetscSubcommSetType(), PetscSubcommSetTypeGeneral(), or PetscSubcommSetFromOptions() one may call
$     PetscSubcommChild() returns the associated subcommunicator on this process
$     PetscSubcommContiguousParent() returns a parent communitor but with all child of the same subcommunicator having contiguous rank

   Sample Usage:
       PetscSubcommCreate()
       PetscSubcommSetNumber()
       PetscSubcommSetType(PETSC_SUBCOMM_INTERLACED);
       ccomm = PetscSubcommChild()
       PetscSubcommDestroy()

   Level: advanced

   Notes:
$   PETSC_SUBCOMM_GENERAL - similar to MPI_Comm_split() each process sets the new communicator (color) they will belong to and the order within that communicator
$   PETSC_SUBCOMM_CONTIGUOUS - each new communicator contains a set of process with contiguous ranks in the original MPI communicator
$   PETSC_SUBCOMM_INTERLACED - each new communictor contains a set of processes equally far apart in rank from the others in that new communicator

   Example: Consider a communicator with six processes split into 3 subcommunicators.
$     PETSC_SUBCOMM_CONTIGUOUS - the first communicator contains rank 0,1  the second rank 2,3 and the third rank 4,5 in the original ordering of the original communicator
$     PETSC_SUBCOMM_INTERLACED - the first communicator contains rank 0,3, the second 1,4 and the third 2,5

   Developer Notes:
   This is used in objects such as PCREDUNDANT to manage the subcommunicators on which the redundant computations
      are performed.


.seealso: PetscSubcommCreate(), PetscSubcommSetNumber(), PetscSubcommSetType(), PetscSubcommView(), PetscSubcommSetFromOptions()

S*/
typedef struct _n_PetscSubcomm* PetscSubcomm;
typedef enum {PETSC_SUBCOMM_GENERAL=0,PETSC_SUBCOMM_CONTIGUOUS=1,PETSC_SUBCOMM_INTERLACED=2} PetscSubcommType;

/*S
     PetscHeap - A simple class for managing heaps

   Level: intermediate

.seealso:  PetscHeapCreate(), PetscHeapAdd(), PetscHeapPop(), PetscHeapPeek(), PetscHeapStash(), PetscHeapUnstash(), PetscHeapView(), PetscHeapDestroy()
S*/
typedef struct _PetscHeap *PetscHeap;

typedef struct _n_PetscShmComm* PetscShmComm;
typedef struct _n_PetscOmpCtrl* PetscOmpCtrl;

/*S
   PetscSegBuffer - a segmented extendable buffer

   Level: developer

.seealso: PetscSegBufferCreate(), PetscSegBufferGet(), PetscSegBufferExtract(), PetscSegBufferDestroy()
S*/
typedef struct _n_PetscSegBuffer *PetscSegBuffer;

typedef struct _n_PetscOptionsHelpPrinted *PetscOptionsHelpPrinted;

#endif
