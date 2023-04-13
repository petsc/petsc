/* Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#ifndef PETSCSYSTYPES_H
#define PETSCSYSTYPES_H

#include <petscconf.h>
#include <petscconf_poison.h>
#include <petscfix.h>
#include <petscmacros.h> // PETSC_NODISCARD, PETSC_CPP_VERSION
#include <stddef.h>

/* SUBMANSEC = Sys */

#include <limits.h> // INT_MIN, INT_MAX

#if defined(__clang__) || (PETSC_CPP_VERSION >= 17)
  // clang allows both [[nodiscard]] and __attribute__((warn_unused_result)) on type
  // definitions. GCC, however, does not, so check that we are using C++17 [[nodiscard]]
  // instead of __attribute__((warn_unused_result))
  #define PETSC_ERROR_CODE_NODISCARD PETSC_NODISCARD
#else
  #define PETSC_ERROR_CODE_NODISCARD
#endif

#ifdef PETSC_CLANG_STATIC_ANALYZER
  #undef PETSC_USE_STRICT_PETSCERRORCODE
#endif

#ifdef PETSC_USE_STRICT_PETSCERRORCODE
  #define PETSC_ERROR_CODE_TYPEDEF   typedef
  #define PETSC_ERROR_CODE_ENUM_NAME PetscErrorCode
#else
  #define PETSC_ERROR_CODE_TYPEDEF
  #define PETSC_ERROR_CODE_ENUM_NAME
#endif

/*E
  PetscErrorCode - Datatype used to return PETSc error codes.

  Level: beginner

  Notes:
  Virtually all PETSc functions return an error code. It is the callers responsibility to check
  the value of the returned error code after each PETSc call to determine if any errors
  occurred. A set of convenience macros (e.g. `PetscCall()`, `PetscCallVoid()`) are provided
  for this purpose. Failing to properly check for errors is not supported, as errors may leave
  PETSc in an undetermined state.

  One can retrieve the error string corresponding to a particular error code using
  `PetscErrorMessage()`.

  The user can also configure PETSc with the `--with-strict-petscerrorcode` option to enable
  compiler warnings when the returned error codes are not captured and checked. Users are
  *heavily* encouraged to opt-in to this option, as it will become enabled by default in a
  future release.

  Developer Notes:

  These are the generic error codes. These error codes are used in many different places in the
  PETSc source code. The C-string versions are at defined in `PetscErrorStrings[]` in
  `src/sys/error/err.c`, while the Fortran versions are defined in
  `src/sys/f90-mod/petscerror.h`. Any changes here must also be made in both locations.

.seealso: `PetscErrorMessage()`, `PetscCall()`, `SETERRQ()`
E*/
PETSC_ERROR_CODE_TYPEDEF enum PETSC_ERROR_CODE_NODISCARD {
  PETSC_SUCCESS                   = 0,
  PETSC_ERR_BOOLEAN_MACRO_FAILURE = 1, /* do not use */

  PETSC_ERR_MIN_VALUE = 54, /* should always be one less then the smallest value */

  PETSC_ERR_MEM            = 55, /* unable to allocate requested memory */
  PETSC_ERR_SUP            = 56, /* no support for requested operation */
  PETSC_ERR_SUP_SYS        = 57, /* no support for requested operation on this computer system */
  PETSC_ERR_ORDER          = 58, /* operation done in wrong order */
  PETSC_ERR_SIG            = 59, /* signal received */
  PETSC_ERR_FP             = 72, /* floating point exception */
  PETSC_ERR_COR            = 74, /* corrupted PETSc object */
  PETSC_ERR_LIB            = 76, /* error in library called by PETSc */
  PETSC_ERR_PLIB           = 77, /* PETSc library generated inconsistent data */
  PETSC_ERR_MEMC           = 78, /* memory corruption */
  PETSC_ERR_CONV_FAILED    = 82, /* iterative method (KSP or SNES) failed */
  PETSC_ERR_USER           = 83, /* user has not provided needed function */
  PETSC_ERR_SYS            = 88, /* error in system call */
  PETSC_ERR_POINTER        = 70, /* pointer does not point to valid address */
  PETSC_ERR_MPI_LIB_INCOMP = 87, /* MPI library at runtime is not compatible with MPI user compiled with */

  PETSC_ERR_ARG_SIZ          = 60, /* nonconforming object sizes used in operation */
  PETSC_ERR_ARG_IDN          = 61, /* two arguments not allowed to be the same */
  PETSC_ERR_ARG_WRONG        = 62, /* wrong argument (but object probably ok) */
  PETSC_ERR_ARG_CORRUPT      = 64, /* null or corrupted PETSc object as argument */
  PETSC_ERR_ARG_OUTOFRANGE   = 63, /* input argument, out of range */
  PETSC_ERR_ARG_BADPTR       = 68, /* invalid pointer argument */
  PETSC_ERR_ARG_NOTSAMETYPE  = 69, /* two args must be same object type */
  PETSC_ERR_ARG_NOTSAMECOMM  = 80, /* two args must be same communicators */
  PETSC_ERR_ARG_WRONGSTATE   = 73, /* object in argument is in wrong state, e.g. unassembled mat */
  PETSC_ERR_ARG_TYPENOTSET   = 89, /* the type of the object has not yet been set */
  PETSC_ERR_ARG_INCOMP       = 75, /* two arguments are incompatible */
  PETSC_ERR_ARG_NULL         = 85, /* argument is null that should not be */
  PETSC_ERR_ARG_UNKNOWN_TYPE = 86, /* type name doesn't match any registered type */

  PETSC_ERR_FILE_OPEN       = 65, /* unable to open file */
  PETSC_ERR_FILE_READ       = 66, /* unable to read from file */
  PETSC_ERR_FILE_WRITE      = 67, /* unable to write to file */
  PETSC_ERR_FILE_UNEXPECTED = 79, /* unexpected data in file */

  PETSC_ERR_MAT_LU_ZRPVT = 71, /* detected a zero pivot during LU factorization */
  PETSC_ERR_MAT_CH_ZRPVT = 81, /* detected a zero pivot during Cholesky factorization */

  PETSC_ERR_INT_OVERFLOW   = 84,
  PETSC_ERR_FLOP_COUNT     = 90,
  PETSC_ERR_NOT_CONVERGED  = 91,  /* solver did not converge */
  PETSC_ERR_MISSING_FACTOR = 92,  /* MatGetFactor() failed */
  PETSC_ERR_OPT_OVERWRITE  = 93,  /* attempted to over write options which should not be changed */
  PETSC_ERR_WRONG_MPI_SIZE = 94,  /* example/application run with number of MPI ranks it does not support */
  PETSC_ERR_USER_INPUT     = 95,  /* missing or incorrect user input */
  PETSC_ERR_GPU_RESOURCE   = 96,  /* unable to load a GPU resource, for example cuBLAS */
  PETSC_ERR_GPU            = 97,  /* An error from a GPU call, this may be due to lack of resources on the GPU or a true error in the call */
  PETSC_ERR_MPI            = 98,  /* general MPI error */
  PETSC_ERR_RETURN         = 99,  /* PetscError() incorrectly returned an error code of 0 */
  PETSC_ERR_MAX_VALUE      = 100, /* this is always the one more than the largest error code */

  /*
    do not use, exist purely to make the enum bounds equal that of a regular int (so conversion
    to int in main() is not undefined behavior)
  */
  PETSC_ERR_MIN_SIGNED_BOUND_DO_NOT_USE = INT_MIN,
  PETSC_ERR_MAX_SIGNED_BOUND_DO_NOT_USE = INT_MAX
} PETSC_ERROR_CODE_ENUM_NAME;

#ifndef PETSC_USE_STRICT_PETSCERRORCODE
typedef int PetscErrorCode;

  /*
  Needed so that C++ lambdas can deduce the return type as PetscErrorCode from
  PetscFunctionReturn(PETSC_SUCCESS). Otherwise we get

  error: return type '(unnamed enum at include/petscsystypes.h:50:1)' must match previous
  return type 'int' when lambda expression has unspecified explicit return type
  PetscFunctionReturn(PETSC_SUCCESS);
  ^
*/
  #define PETSC_SUCCESS ((PetscErrorCode)0)
#endif

#undef PETSC_ERROR_CODE_NODISCARD
#undef PETSC_ERROR_CODE_TYPEDEF
#undef PETSC_ERROR_CODE_ENUM_NAME

/*MC
    PetscClassId - A unique id used to identify each PETSc class.

    Level: developer

    Note:
    Use `PetscClassIdRegister()` to obtain a new value for a new class being created. Usually
         XXXInitializePackage() calls it for each class it defines.

    Developer Note:
    Internal integer stored in the `_p_PetscObject` data structure. These are all computed by an offset from the lowest one, `PETSC_SMALLEST_CLASSID`.

.seealso: `PetscClassIdRegister()`, `PetscLogEventRegister()`, `PetscHeaderCreate()`
M*/
typedef int PetscClassId;

/*MC
    PetscMPIInt - datatype used to represent 'int' parameters to MPI functions.

    Level: intermediate

    Notes:
    This is always a 32 bit integer, sometimes it is the same as `PetscInt`, but if PETSc was built with `--with-64-bit-indices` but
           standard C/Fortran integers are 32 bit then this is NOT the same as `PetscInt`; it remains 32 bit.

    `PetscMPIIntCast`(a,&b) checks if the given `PetscInt` a will fit in a `PetscMPIInt`, if not it
      generates a `PETSC_ERR_ARG_OUTOFRANGE` error.

.seealso: `PetscBLASInt`, `PetscInt`, `PetscMPIIntCast()`
M*/
typedef int PetscMPIInt;

/* Limit MPI to 32-bits */
enum {
  PETSC_MPI_INT_MIN = INT_MIN,
  PETSC_MPI_INT_MAX = INT_MAX
};

/*MC
    PetscSizeT - datatype used to represent sizes in memory (like `size_t`)

    Level: intermediate

    Notes:
    This is equivalent to `size_t`, but defined for consistency with Fortran, which lacks a native equivalent of `size_t`.

.seealso: `PetscInt`, `PetscInt64`, `PetscCount`
M*/
typedef size_t PetscSizeT;

/*MC
    PetscCount - signed datatype used to represent counts

    Level: intermediate

    Notes:
    This is equivalent to `ptrdiff_t`, but defined for consistency with Fortran, which lacks a native equivalent of `ptrdiff_t`.

    Use `PetscCount_FMT` to format with `PetscPrintf()`, `printf()`, and related functions.

.seealso: `PetscInt`, `PetscInt64`, `PetscSizeT`
M*/
typedef ptrdiff_t PetscCount;
#define PetscCount_FMT "td"

/*MC
    PetscEnum - datatype used to pass enum types within PETSc functions.

    Level: intermediate

.seealso: `PetscOptionsGetEnum()`, `PetscOptionsEnum()`, `PetscBagRegisterEnum()`
M*/
typedef enum {
  ENUM_DUMMY
} PetscEnum;

typedef short PetscShort;
typedef char  PetscChar;
typedef float PetscFloat;

/*MC
  PetscInt - PETSc type that represents an integer, used primarily to
      represent size of arrays and indexing into arrays. Its size can be configured with the option `--with-64-bit-indices` to be either 32-bit (default) or 64-bit.

  Level: beginner

  Notes:
  For MPI calls that require datatypes, use `MPIU_INT` as the datatype for `PetscInt`. It will automatically work correctly regardless of the size of `PetscInt`.

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`, `MPIU_INT`, `PetscIntCast()`
M*/

#if defined(PETSC_HAVE_STDINT_H)
  #include <stdint.h>
#endif
#if defined(PETSC_HAVE_INTTYPES_H)
  #if !defined(__STDC_FORMAT_MACROS)
    #define __STDC_FORMAT_MACROS /* required for using PRId64 from c++ */
  #endif
  #include <inttypes.h>
  #if !defined(PRId64)
    #define PRId64 "ld"
  #endif
#endif

#if defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T) /* MPI_INT64_T is not guaranteed to be a macro */
typedef int64_t PetscInt64;

  #define PETSC_INT64_MIN INT64_MIN
  #define PETSC_INT64_MAX INT64_MAX

#elif (PETSC_SIZEOF_LONG_LONG == 8)
typedef long long PetscInt64;

  #define PETSC_INT64_MIN LLONG_MIN
  #define PETSC_INT64_MAX LLONG_MAX

#elif defined(PETSC_HAVE___INT64)
typedef __int64 PetscInt64;

  #define PETSC_INT64_MIN INT64_MIN
  #define PETSC_INT64_MAX INT64_MAX

#else
  #error "cannot determine PetscInt64 type"
#endif

#if defined(PETSC_USE_64BIT_INDICES)
typedef PetscInt64 PetscInt;

  #define PETSC_INT_MIN PETSC_INT64_MIN
  #define PETSC_INT_MAX PETSC_INT64_MAX
  #define PetscInt_FMT  PetscInt64_FMT
#else
typedef int       PetscInt;

enum {
  PETSC_INT_MIN = INT_MIN,
  PETSC_INT_MAX = INT_MAX
};

  #define PetscInt_FMT "d"
#endif

#define PETSC_MIN_INT    PETSC_INT_MIN
#define PETSC_MAX_INT    PETSC_INT_MAX
#define PETSC_MAX_UINT16 65535

#if defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T) /* MPI_INT64_T is not guaranteed to be a macro */
  #define MPIU_INT64     MPI_INT64_T
  #define PetscInt64_FMT PRId64
#elif (PETSC_SIZEOF_LONG_LONG == 8)
  #define MPIU_INT64     MPI_LONG_LONG_INT
  #define PetscInt64_FMT "lld"
#elif defined(PETSC_HAVE___INT64)
  #define MPIU_INT64     MPI_INT64_T
  #define PetscInt64_FMT "ld"
#else
  #error "cannot determine PetscInt64 type"
#endif

/*MC
   PetscBLASInt - datatype used to represent 'int' parameters to BLAS/LAPACK functions.

   Level: intermediate

   Notes:
   Usually this is the same as `PetscInt`, but if PETSc was built with `--with-64-bit-indices` but
   standard C/Fortran integers are 32 bit then this may not be the same as `PetscInt`,
   except on some BLAS/LAPACK implementations that support 64 bit integers see the notes below.

   `PetscErrorCode` `PetscBLASIntCast`(a,&b) checks if the given `PetscInt` a will fit in a `PetscBLASInt`, if not it
    generates a `PETSC_ERR_ARG_OUTOFRANGE` error

   Installation Notes:
    ./configure automatically determines the size of the integers used by BLAS/LAPACK except when `--with-batch` is used
    in that situation one must know (by some other means) if the integers used by BLAS/LAPACK are 64 bit and if so pass the flag `--known-64-bit-blas-indices`

    MATLAB ships with BLAS and LAPACK that use 64 bit integers, for example if you run ./configure with, the option
     `--with-blaslapack-lib`=[/Applications/MATLAB_R2010b.app/bin/maci64/libmwblas.dylib,/Applications/MATLAB_R2010b.app/bin/maci64/libmwlapack.dylib]

    MKL ships with both 32 and 64 bit integer versions of the BLAS and LAPACK. If you pass the flag `-with-64-bit-blas-indices` PETSc will link
    against the 64 bit version, otherwise it uses the 32 bit version

    OpenBLAS can be built to use 64 bit integers. The ./configure options `--download-openblas` `-with-64-bit-blas-indices` will build a 64 bit integer version

    External packages such as hypre, ML, SuperLU etc do not provide any support for passing 64 bit integers to BLAS/LAPACK so cannot
    be used with PETSc when PETSc links against 64 bit integer BLAS/LAPACK. ./configure will generate an error if you attempt to link PETSc against any of
    these external libraries while using 64 bit integer BLAS/LAPACK.

.seealso: `PetscMPIInt`, `PetscInt`, `PetscBLASIntCast()`
M*/
#if defined(PETSC_HAVE_64BIT_BLAS_INDICES)
typedef PetscInt64 PetscBLASInt;

  #define PETSC_BLAS_INT_MIN PETSC_INT64_MIN
  #define PETSC_BLAS_INT_MAX PETSC_INT64_MAX
  #define PetscBLASInt_FMT   PetscInt64_FMT
#else
typedef int PetscBLASInt;

enum {
  PETSC_BLAS_INT_MIN = INT_MIN,
  PETSC_BLAS_INT_MAX = INT_MAX
};

  #define PetscBLASInt_FMT "d"
#endif

/*MC
   PetscCuBLASInt - datatype used to represent 'int' parameters to cuBLAS/cuSOLVER functions.

   Level: intermediate

   Notes:
   As of this writing `PetscCuBLASInt` is always the system `int`.

  `PetscErrorCode` `PetscCuBLASIntCast`(a,&b) checks if the given `PetscInt` a will fit in a `PetscCuBLASInt`, if not it
   generates a `PETSC_ERR_ARG_OUTOFRANGE` error

.seealso: `PetscBLASInt`, `PetscMPIInt`, `PetscInt`, `PetscCuBLASIntCast()`
M*/
typedef int PetscCuBLASInt;

enum {
  PETSC_CUBLAS_INT_MIN = INT_MIN,
  PETSC_CUBLAS_INT_MAX = INT_MAX
};

/*MC
   PetscHipBLASInt - datatype used to represent 'int' parameters to hipBLAS/hipSOLVER functions.

   Level: intermediate

   Notes:
   As of this writing `PetscHipBLASInt` is always the system `int`.

   `PetscErrorCode` `PetscHipBLASIntCast`(a,&b) checks if the given `PetscInt` a will fit in a `PetscHipBLASInt`, if not it
   generates a `PETSC_ERR_ARG_OUTOFRANGE` error

.seealso: PetscBLASInt, PetscMPIInt, PetscInt, PetscHipBLASIntCast()
M*/
typedef int PetscHipBLASInt;

enum {
  PETSC_HIPBLAS_INT_MIN = INT_MIN,
  PETSC_HIPBLAS_INT_MAX = INT_MAX
};

/*E
    PetscBool  - Logical variable. Actually an enum in C and a logical in Fortran.

   Level: beginner

   Developer Note:
   Why have `PetscBool`, why not use bool in C? The problem is that K and R C, C99 and C++ all have different mechanisms for
      boolean values. It is not easy to have a simple macro that that will work properly in all circumstances with all three mechanisms.

.seealso: `PETSC_TRUE`, `PETSC_FALSE`, `PetscNot()`, `PetscBool3`
E*/
typedef enum {
  PETSC_FALSE,
  PETSC_TRUE
} PetscBool;
PETSC_EXTERN const char *const PetscBools[];

/*E
    PetscBool3  - Ternary logical variable. Actually an enum in C and a 4 byte integer in Fortran.

   Level: beginner

   Note:
   Should not be used with the if (flg) or if (!flg) syntax.

.seealso: `PETSC_TRUE`, `PETSC_FALSE`, `PetscNot()`, `PETSC_BOOL3_TRUE`, `PETSC_BOOL3_FALSE`, `PETSC_BOOL3_UNKNOWN`
E*/
typedef enum {
  PETSC_BOOL3_FALSE,
  PETSC_BOOL3_TRUE,
  PETSC_BOOL3_UNKNOWN = -1
} PetscBool3;

#define PetscBool3ToBool(a) ((a) == PETSC_BOOL3_TRUE ? PETSC_TRUE : PETSC_FALSE)
#define PetscBoolToBool3(a) ((a) == PETSC_TRUE ? PETSC_BOOL3_TRUE : PETSC_BOOL3_FALSE)

/*MC
   PetscReal - PETSc type that represents a real number version of `PetscScalar`

   Level: beginner

   Notes:
   For MPI calls that require datatypes, use `MPIU_REAL` as the datatype for `PetscReal` and `MPIU_SUM`, `MPIU_MAX`, etc. for operations.
   They will automatically work correctly regardless of the size of `PetscReal`.

   See `PetscScalar` for details on how to ./configure the size of `PetscReal`.

.seealso: `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`, `MPIU_INT`
M*/

#if defined(PETSC_USE_REAL_SINGLE)
typedef float PetscReal;
#elif defined(PETSC_USE_REAL_DOUBLE)
typedef double    PetscReal;
#elif defined(PETSC_USE_REAL___FLOAT128)
  #if defined(__cplusplus)
extern "C" {
  #endif
  #include <quadmath.h>
  #if defined(__cplusplus)
}
  #endif
typedef __float128 PetscReal;
#elif defined(PETSC_USE_REAL___FP16)
typedef __fp16 PetscReal;
#endif /* PETSC_USE_REAL_* */

/*MC
   PetscComplex - PETSc type that represents a complex number with precision matching that of `PetscReal`.

   Synopsis:
   #include <petscsys.h>
   PetscComplex number = 1. + 2.*PETSC_i;

   Level: beginner

   Notes:
   For MPI calls that require datatypes, use `MPIU_COMPLEX` as the datatype for `PetscComplex` and `MPIU_SUM` etc for operations.
          They will automatically work correctly regardless of the size of `PetscComplex`.

          See `PetscScalar` for details on how to ./configure the size of `PetscReal`

          Complex numbers are automatically available if PETSc was able to find a working complex implementation

    PETSc has a 'fix' for complex numbers to support expressions such as `std::complex<PetscReal>` + `PetscInt`, which are not supported by the standard
    C++ library, but are convenient for petsc users. If the C++ compiler is able to compile code in `petsccxxcomplexfix.h` (This is checked by
    configure), we include `petsccxxcomplexfix.h` to provide this convenience.

    If the fix causes conflicts, or one really does not want this fix for a particular C++ file, one can define `PETSC_SKIP_CXX_COMPLEX_FIX`
    at the beginning of the C++ file to skip the fix.

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`, `MPIU_INT`, `PETSC_i`
M*/
#if !defined(PETSC_SKIP_COMPLEX)
  #if defined(PETSC_CLANGUAGE_CXX)
    #if !defined(PETSC_USE_REAL___FP16) && !defined(PETSC_USE_REAL___FLOAT128)
      #if defined(__cplusplus) && defined(PETSC_HAVE_CXX_COMPLEX) /* enable complex for library code */
        #define PETSC_HAVE_COMPLEX 1
      #elif !defined(__cplusplus) && defined(PETSC_HAVE_C99_COMPLEX) && defined(PETSC_HAVE_CXX_COMPLEX) /* User code only - conditional on library code complex support */
        #define PETSC_HAVE_COMPLEX 1
      #endif
    #elif defined(PETSC_USE_REAL___FLOAT128) && defined(PETSC_HAVE_C99_COMPLEX)
      #define PETSC_HAVE_COMPLEX 1
    #endif
  #else /* !PETSC_CLANGUAGE_CXX */
    #if !defined(PETSC_USE_REAL___FP16)
      #if !defined(__cplusplus) && defined(PETSC_HAVE_C99_COMPLEX) /* enable complex for library code */
        #define PETSC_HAVE_COMPLEX 1
      #elif defined(__cplusplus) && defined(PETSC_HAVE_C99_COMPLEX) && defined(PETSC_HAVE_CXX_COMPLEX) /* User code only - conditional on library code complex support */
        #define PETSC_HAVE_COMPLEX 1
      #endif
    #endif
  #endif /* PETSC_CLANGUAGE_CXX */
#endif   /* !PETSC_SKIP_COMPLEX */

#if defined(PETSC_HAVE_COMPLEX)
  #if defined(__cplusplus) /* C++ complex support */
    /* Locate a C++ complex template library */
    #if defined(PETSC_DESIRE_KOKKOS_COMPLEX) /* Defined in petscvec_kokkos.hpp for *.kokkos.cxx files */
      #define petsccomplexlib Kokkos
      #include <Kokkos_Complex.hpp>
    #elif defined(__CUDACC__) || defined(__HIPCC__)
      #define petsccomplexlib thrust
      #include <thrust/complex.h>
    #elif defined(PETSC_USE_REAL___FLOAT128)
      #include <complex.h>
    #else
      #define petsccomplexlib std
      #include <complex>
    #endif

    /* Define PetscComplex based on the precision */
    #if defined(PETSC_USE_REAL_SINGLE)
typedef petsccomplexlib::complex<float> PetscComplex;
    #elif defined(PETSC_USE_REAL_DOUBLE)
typedef petsccomplexlib::complex<double> PetscComplex;
    #elif defined(PETSC_USE_REAL___FLOAT128)
typedef __complex128 PetscComplex;
    #endif

    /* Include a PETSc C++ complex 'fix'. Check PetscComplex manual page for details */
    #if defined(PETSC_HAVE_CXX_COMPLEX_FIX) && !defined(PETSC_SKIP_CXX_COMPLEX_FIX)
      #include <petsccxxcomplexfix.h>
    #endif
  #else /* c99 complex support */
    #include <complex.h>
    #if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL___FP16)
typedef float _Complex PetscComplex;
    #elif defined(PETSC_USE_REAL_DOUBLE)
typedef double _Complex PetscComplex;
    #elif defined(PETSC_USE_REAL___FLOAT128)
typedef __complex128 PetscComplex;
    #endif /* PETSC_USE_REAL_* */
  #endif   /* !__cplusplus */
#endif     /* PETSC_HAVE_COMPLEX */

/*MC
   PetscScalar - PETSc type that represents either a double precision real number, a double precision
       complex number, a single precision real number, a __float128 real or complex or a __fp16 real - if the code is configured
       with `--with-scalar-type`=real,complex `--with-precision`=single,double,__float128,__fp16

   Level: beginner

   Note:
   For MPI calls that require datatypes, use `MPIU_SCALAR` as the datatype for `PetscScalar` and `MPIU_SUM`, etc for operations. They will automatically work correctly regardless of the size of `PetscScalar`.

.seealso: `PetscReal`, `PetscComplex`, `PetscInt`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`, `MPIU_INT`, `PetscRealPart()`, `PetscImaginaryPart()`
M*/

#if defined(PETSC_USE_COMPLEX) && defined(PETSC_HAVE_COMPLEX)
typedef PetscComplex PetscScalar;
#else  /* PETSC_USE_COMPLEX */
typedef PetscReal PetscScalar;
#endif /* PETSC_USE_COMPLEX */

/*E
    PetscCopyMode  - Determines how an array or `PetscObject` passed to certain functions is copied or retained by the aggregate `PetscObject`

   Level: beginner

   Values for array input:
+   `PETSC_COPY_VALUES` - the array values are copied into new space, the user is free to reuse or delete the passed in array
.   `PETSC_OWN_POINTER` - the array values are NOT copied, the object takes ownership of the array and will free it later, the user cannot change or
                          delete the array. The array MUST have been obtained with `PetscMalloc()`. Hence this mode cannot be used in Fortran.
-   `PETSC_USE_POINTER` - the array values are NOT copied, the object uses the array but does NOT take ownership of the array. The user cannot use
                          the array but the user must delete the array after the object is destroyed.

   Values for PetscObject:
+   `PETSC_COPY_VALUES` - the input `PetscObject` is cloned into the aggregate `PetscObject`; the user is free to reuse/modify the input `PetscObject` without side effects.
.   `PETSC_OWN_POINTER` - the input `PetscObject` is referenced by pointer (with reference count), thus should not be modified by the user.
                          increases its reference count).
-   `PETSC_USE_POINTER` - invalid for `PetscObject` inputs.
E*/
typedef enum {
  PETSC_COPY_VALUES,
  PETSC_OWN_POINTER,
  PETSC_USE_POINTER
} PetscCopyMode;
PETSC_EXTERN const char *const PetscCopyModes[];

/*MC
    PETSC_FALSE - False value of `PetscBool`

    Level: beginner

    Note:
    Zero integer

.seealso: `PetscBool`, `PetscBool3`, `PETSC_TRUE`
M*/

/*MC
    PETSC_TRUE - True value of `PetscBool`

    Level: beginner

    Note:
    Nonzero integer

.seealso: `PetscBool`, `PetscBool3`, `PETSC_FALSE`
M*/

/*MC
    PetscLogDouble - Used for logging times

  Level: developer

  Note:
  Contains double precision numbers that are not used in the numerical computations, but rather in logging, timing etc.

M*/
typedef double PetscLogDouble;

/*E
    PetscDataType - Used for handling different basic data types.

   Level: beginner

   Notes:
   Use of this should be avoided if one can directly use `MPI_Datatype` instead.

   `PETSC_INT` is the datatype for a `PetscInt`, regardless of whether it is 4 or 8 bytes.
   `PETSC_REAL`, `PETSC_COMPLEX` and `PETSC_SCALAR` are the datatypes for `PetscReal`, `PetscComplex` and `PetscScalar`, regardless of their sizes.

   Developer Notes:
   It would be nice if we could always just use MPI Datatypes, why can we not?

   If you change any values in `PetscDatatype` make sure you update their usage in
   share/petsc/matlab/PetscBagRead.m and share/petsc/matlab/@PetscOpenSocket/read/write.m

   TODO:
   Add PETSC_INT32 and remove use of improper `PETSC_ENUM`

.seealso: `PetscBinaryRead()`, `PetscBinaryWrite()`, `PetscDataTypeToMPIDataType()`,
          `PetscDataTypeGetSize()`
E*/
typedef enum {
  PETSC_DATATYPE_UNKNOWN = 0,
  PETSC_DOUBLE           = 1,
  PETSC_COMPLEX          = 2,
  PETSC_LONG             = 3,
  PETSC_SHORT            = 4,
  PETSC_FLOAT            = 5,
  PETSC_CHAR             = 6,
  PETSC_BIT_LOGICAL      = 7,
  PETSC_ENUM             = 8,
  PETSC_BOOL             = 9,
  PETSC___FLOAT128       = 10,
  PETSC_OBJECT           = 11,
  PETSC_FUNCTION         = 12,
  PETSC_STRING           = 13,
  PETSC___FP16           = 14,
  PETSC_STRUCT           = 15,
  PETSC_INT              = 16,
  PETSC_INT64            = 17,
  PETSC_COUNT            = 18
} PetscDataType;
PETSC_EXTERN const char *const PetscDataTypes[];

#if defined(PETSC_USE_REAL_SINGLE)
  #define PETSC_REAL PETSC_FLOAT
#elif defined(PETSC_USE_REAL_DOUBLE)
  #define PETSC_REAL PETSC_DOUBLE
#elif defined(PETSC_USE_REAL___FLOAT128)
  #define PETSC_REAL PETSC___FLOAT128
#elif defined(PETSC_USE_REAL___FP16)
  #define PETSC_REAL PETSC___FP16
#else
  #define PETSC_REAL PETSC_DOUBLE
#endif

#if defined(PETSC_USE_COMPLEX)
  #define PETSC_SCALAR PETSC_COMPLEX
#else
  #define PETSC_SCALAR PETSC_REAL
#endif

#define PETSC_FORTRANADDR PETSC_LONG

/*S
    PetscToken - 'Token' used for managing tokenizing strings

  Level: intermediate

.seealso: `PetscTokenCreate()`, `PetscTokenFind()`, `PetscTokenDestroy()`
S*/
typedef struct _p_PetscToken *PetscToken;

/*S
     PetscObject - any PETSc object, `PetscViewer`, `Mat`, `Vec`, `KSP` etc

   Level: beginner

   Notes:
   This is the base class from which all PETSc objects are derived from.

   In certain situations one can cast an object, for example a `Vec`, to a `PetscObject` with (`PetscObject`)vec

.seealso: `PetscObjectDestroy()`, `PetscObjectView()`, `PetscObjectGetName()`, `PetscObjectSetName()`, `PetscObjectReference()`, `PetscObjectDereference()`
S*/
typedef struct _p_PetscObject *PetscObject;

/*MC
    PetscObjectId - unique integer Id for a `PetscObject`

    Level: developer

    Note:
    Unlike pointer values, object ids are never reused so one may save a `PetscObjectId` and compare it to one obtained later from a `PetscObject` to determine
    if the objects are the same. Never compare two object pointer values.

.seealso: `PetscObjectState`, `PetscObjectGetId()`
M*/
typedef PetscInt64 PetscObjectId;

/*MC
    PetscObjectState - integer state for a `PetscObject`

    Level: developer

    Notes:
    Object state is always-increasing and (for objects that track state) can be used to determine if an object has
    changed since the last time you interacted with it.  It is 64-bit so that it will not overflow for a very long time.

.seealso: `PetscObjectId`, `PetscObjectStateGet()`, `PetscObjectStateIncrease()`, `PetscObjectStateSet()`
M*/
typedef PetscInt64 PetscObjectState;

/*S
     PetscFunctionList - Linked list of functions, possibly stored in dynamic libraries, accessed
      by string name

   Level: advanced

.seealso: `PetscFunctionListAdd()`, `PetscFunctionListDestroy()`
S*/
typedef struct _n_PetscFunctionList *PetscFunctionList;

/*E
  PetscFileMode - Access mode for a file.

  Values:
+  `FILE_MODE_UNDEFINED` - initial invalid value
.  `FILE_MODE_READ` - open a file at its beginning for reading
.  `FILE_MODE_WRITE` - open a file at its beginning for writing (will create if the file does not exist)
.  `FILE_MODE_APPEND` - open a file at end for writing
.  `FILE_MODE_UPDATE` - open a file for updating, meaning for reading and writing
-  `FILE_MODE_APPEND_UPDATE` - open a file for updating, meaning for reading and writing, at the end

  Level: beginner

.seealso: `PetscViewerFileSetMode()`
E*/
typedef enum {
  FILE_MODE_UNDEFINED = -1,
  FILE_MODE_READ      = 0,
  FILE_MODE_WRITE,
  FILE_MODE_APPEND,
  FILE_MODE_UPDATE,
  FILE_MODE_APPEND_UPDATE
} PetscFileMode;
PETSC_EXTERN const char *const PetscFileModes[];

typedef void *PetscDLHandle;
typedef enum {
  PETSC_DL_DECIDE = 0,
  PETSC_DL_NOW    = 1,
  PETSC_DL_LOCAL  = 2
} PetscDLMode;

/*S
     PetscObjectList - Linked list of PETSc objects, each accessible by string name

   Level: developer

   Note:
   Used by `PetscObjectCompose()` and `PetscObjectQuery()`

.seealso: `PetscObjectListAdd()`, `PetscObjectListDestroy()`, `PetscObjectListFind()`, `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscFunctionList`
S*/
typedef struct _n_PetscObjectList *PetscObjectList;

/*S
     PetscDLLibrary - Linked list of dynamic libraries to search for functions

   Level: developer

.seealso: `PetscDLLibraryOpen()`
S*/
typedef struct _n_PetscDLLibrary *PetscDLLibrary;

/*S
     PetscContainer - Simple PETSc object that contains a pointer to any required data

   Level: advanced

   Note:
   This is useful to attach arbitrary data to a `PetscObject` with `PetscObjectCompose()` and `PetscObjectQuery()`

.seealso: `PetscObject`, `PetscContainerCreate()`, `PetscObjectCompose()`, `PetscObjectQuery()`
S*/
typedef struct _p_PetscContainer *PetscContainer;

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

.seealso: `PetscRandomCreate()`, `PetscRandomGetValue()`, `PetscRandomType`
S*/
typedef struct _p_PetscRandom *PetscRandom;

/*
   In binary files variables are stored using the following lengths,
  regardless of how they are stored in memory on any one particular
  machine. Use these rather then sizeof() in computing sizes for
  PetscBinarySeek().
*/
#define PETSC_BINARY_INT_SIZE    (32 / 8)
#define PETSC_BINARY_FLOAT_SIZE  (32 / 8)
#define PETSC_BINARY_CHAR_SIZE   (8 / 8)
#define PETSC_BINARY_SHORT_SIZE  (16 / 8)
#define PETSC_BINARY_DOUBLE_SIZE (64 / 8)
#define PETSC_BINARY_SCALAR_SIZE sizeof(PetscScalar)

/*E
  PetscBinarySeekType - argument to `PetscBinarySeek()`

  Values:
+  `PETSC_BINARY_SEEK_SET` - offset is an absolute location in the file
.  `PETSC_BINARY_SEEK_CUR` - offset is an offset from the current location of the file pointer
-  `PETSC_BINARY_SEEK_END` - offset is an offset from the end of the file

  Level: advanced

.seealso: `PetscBinarySeek()`, `PetscBinarySynchronizedSeek()`
E*/
typedef enum {
  PETSC_BINARY_SEEK_SET = 0,
  PETSC_BINARY_SEEK_CUR = 1,
  PETSC_BINARY_SEEK_END = 2
} PetscBinarySeekType;

/*E
    PetscBuildTwoSidedType - algorithm for setting up two-sided communication for use with `PetscSF`

   Values:
+  `PETSC_BUILDTWOSIDED_ALLREDUCE` - classical algorithm using an `MPI_Allreduce()` with
      a buffer of length equal to the communicator size. Not memory-scalable due to
      the large reduction size. Requires only an MPI-1 implementation.
.  `PETSC_BUILDTWOSIDED_IBARRIER` - nonblocking algorithm based on `MPI_Issend()` and `MPI_Ibarrier()`.
      Proved communication-optimal in Hoefler, Siebert, and Lumsdaine (2010). Requires an MPI-3 implementation.
-  `PETSC_BUILDTWOSIDED_REDSCATTER` - similar to above, but use more optimized function
      that only communicates the part of the reduction that is necessary.  Requires an MPI-2 implementation.

   Level: developer

.seealso: `PetscCommBuildTwoSided()`, `PetscCommBuildTwoSidedSetType()`, `PetscCommBuildTwoSidedGetType()`
E*/
typedef enum {
  PETSC_BUILDTWOSIDED_NOTSET     = -1,
  PETSC_BUILDTWOSIDED_ALLREDUCE  = 0,
  PETSC_BUILDTWOSIDED_IBARRIER   = 1,
  PETSC_BUILDTWOSIDED_REDSCATTER = 2
  /* Updates here must be accompanied by updates in finclude/petscsys.h and the string array in mpits.c */
} PetscBuildTwoSidedType;
PETSC_EXTERN const char *const PetscBuildTwoSidedTypes[];

/* NOTE: If you change this, you must also change the values in src/vec/f90-mod/petscvec.h */
/*E
  InsertMode - How the entries are combined with the current values in the vectors or matrices

  Values:
+  `NOT_SET_VALUES` - do not actually use the values
.  `INSERT_VALUES` - replace the current values with the provided values, unless the index is marked as constrained by the `PetscSection`
.  `ADD_VALUES` - add the values to the current values, unless the index is marked as constrained by the `PetscSection`
.  `MAX_VALUES` - use the maximum of each current value and provided value
.  `MIN_VALUES` - use the minimum of each current value and provided value
.  `INSERT_ALL_VALUES` - insert, even if indices that are not marked as constrained by the `PetscSection`
.  `ADD_ALL_VALUES` - add, even if indices that are not marked as constrained by the `PetscSection`
.  `INSERT_BC_VALUES` - insert, but ignore indices that are not marked as constrained by the `PetscSection`
-  `ADD_BC_VALUES` - add, but ignore indices that are not marked as constrained by the `PetscSection`

  Level: beginner

  Note:
  The `PetscSection` that determines the effects of the `InsertMode` values can be obtained by the `Vec` object with `VecGetDM()`
  and `DMGetLocalSection()`.

  Not all options are supported for all operations or PETSc object types.

.seealso: `VecSetValues()`, `MatSetValues()`, `VecSetValue()`, `VecSetValuesBlocked()`,
          `VecSetValuesLocal()`, `VecSetValuesBlockedLocal()`, `MatSetValuesBlocked()`,
          `MatSetValuesBlockedLocal()`, `MatSetValuesLocal()`, `VecScatterBegin()`, `VecScatterEnd()`
E*/
typedef enum {
  NOT_SET_VALUES,
  INSERT_VALUES,
  ADD_VALUES,
  MAX_VALUES,
  MIN_VALUES,
  INSERT_ALL_VALUES,
  ADD_ALL_VALUES,
  INSERT_BC_VALUES,
  ADD_BC_VALUES
} InsertMode;

/*MC
    INSERT_VALUES - Put a value into a vector or matrix, overwrites any previous value

    Level: beginner

.seealso: `InsertMode`, `VecSetValues()`, `MatSetValues()`, `VecSetValue()`, `VecSetValuesBlocked()`,
          `VecSetValuesLocal()`, `VecSetValuesBlockedLocal()`, `MatSetValuesBlocked()`, `ADD_VALUES`,
          `MatSetValuesBlockedLocal()`, `MatSetValuesLocal()`, `VecScatterBegin()`, `VecScatterEnd()`, `MAX_VALUES`
M*/

/*MC
    ADD_VALUES - Adds a value into a vector or matrix, if there previously was no value, just puts the
                value into that location

    Level: beginner

.seealso: `InsertMode`, `VecSetValues()`, `MatSetValues()`, `VecSetValue()`, `VecSetValuesBlocked()`,
          `VecSetValuesLocal()`, `VecSetValuesBlockedLocal()`, `MatSetValuesBlocked()`, `INSERT_VALUES`,
          `MatSetValuesBlockedLocal()`, `MatSetValuesLocal()`, `VecScatterBegin()`, `VecScatterEnd()`, `MAX_VALUES`
M*/

/*MC
    MAX_VALUES - Puts the maximum of the scattered/gathered value and the current value into each location

    Level: beginner

.seealso: `InsertMode`, `VecScatterBegin()`, `VecScatterEnd()`, `ADD_VALUES`, `INSERT_VALUES`
M*/

/*MC
    MIN_VALUES - Puts the minimal of the scattered/gathered value and the current value into each location

    Level: beginner

.seealso: `InsertMode`, `VecScatterBegin()`, `VecScatterEnd()`, `ADD_VALUES`, `INSERT_VALUES`
M*/

/*S
   PetscSubcomm - A decomposition of an MPI communicator into subcommunicators

   Values:
+   `PETSC_SUBCOMM_GENERAL` - similar to `MPI_Comm_split()` each process sets the new communicator (color) they will belong to and the order within that communicator
.   `PETSC_SUBCOMM_CONTIGUOUS` - each new communicator contains a set of process with contiguous ranks in the original MPI communicator
-   `PETSC_SUBCOMM_INTERLACED` - each new communictor contains a set of processes equally far apart in rank from the others in that new communicator

   Sample Usage:
.vb
       PetscSubcommCreate()
       PetscSubcommSetNumber()
       PetscSubcommSetType(PETSC_SUBCOMM_INTERLACED);
       ccomm = PetscSubcommChild()
       PetscSubcommDestroy()
.ve

   Example:
   Consider a communicator with six processes split into 3 subcommunicators.
.vb
   PETSC_SUBCOMM_CONTIGUOUS - the first communicator contains rank 0,1  the second rank 2,3 and the third rank 4,5 in the original ordering of the original communicator
   PETSC_SUBCOMM_INTERLACED - the first communicator contains rank 0,3, the second 1,4 and the third 2,5
.ve

   Level: advanced

   Note:
   After a call to `PetscSubcommSetType()`, `PetscSubcommSetTypeGeneral()`, or `PetscSubcommSetFromOptions()` one may call
.vb
     PetscSubcommChild() returns the associated subcommunicator on this process
     PetscSubcommContiguousParent() returns a parent communitor but with all child of the same subcommunicator having contiguous rank
.ve

   Developer Note:
   This is used in objects such as `PCREDUNDANT` to manage the subcommunicators on which the redundant computations
   are performed.

.seealso: `PetscSubcommCreate()`, `PetscSubcommSetNumber()`, `PetscSubcommSetType()`, `PetscSubcommView()`, `PetscSubcommSetFromOptions()`
S*/
typedef struct _n_PetscSubcomm *PetscSubcomm;
typedef enum {
  PETSC_SUBCOMM_GENERAL    = 0,
  PETSC_SUBCOMM_CONTIGUOUS = 1,
  PETSC_SUBCOMM_INTERLACED = 2
} PetscSubcommType;
PETSC_EXTERN const char *const PetscSubcommTypes[];

/*S
     PetscHeap - A simple class for managing heaps

   Level: intermediate

.seealso: `PetscHeapCreate()`, `PetscHeapAdd()`, `PetscHeapPop()`, `PetscHeapPeek()`, `PetscHeapStash()`, `PetscHeapUnstash()`, `PetscHeapView()`, `PetscHeapDestroy()`
S*/
typedef struct _PetscHeap *PetscHeap;

typedef struct _n_PetscShmComm *PetscShmComm;
typedef struct _n_PetscOmpCtrl *PetscOmpCtrl;

/*S
   PetscSegBuffer - a segmented extendable buffer

   Level: developer

.seealso: `PetscSegBufferCreate()`, `PetscSegBufferGet()`, `PetscSegBufferExtract()`, `PetscSegBufferDestroy()`
S*/
typedef struct _n_PetscSegBuffer *PetscSegBuffer;

typedef struct _n_PetscOptionsHelpPrinted *PetscOptionsHelpPrinted;
#endif
