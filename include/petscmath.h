/*
   
      PETSc mathematics include file. Defines certain basic mathematical 
    constants and functions for working with single and double precision
    floating point numbers as well as complex and integers.

    This file is included by petscsys.h and should not be used directly.

*/

#if !defined(__PETSCMATH_H)
#define __PETSCMATH_H
#include <math.h>
PETSC_EXTERN_CXX_BEGIN

extern  MPI_Datatype  MPIU_2SCALAR;
extern  MPI_Datatype  MPIU_2INT;

/*

     Defines operations that are different for complex and real numbers;
   note that one cannot really mix the use of complex and real in the same 
   PETSc program. All PETSc objects in one program are built around the object
   PetscScalar which is either always a real or a complex.

*/

#define PetscExpPassiveScalar(a) PetscExpScalar()
#if defined(PETSC_USE_REAL_SINGLE)
#define MPIU_REAL   MPI_FLOAT
typedef float PetscReal;
#define PetscSqrtReal(a)    sqrt(a)
#elif defined(PETSC_USE_REAL_DOUBLE)
#define MPIU_REAL   MPI_DOUBLE
typedef double PetscReal;
#define PetscSqrtReal(a)    sqrt(a)
#elif defined(PETSC_USE_REAL___FLOAT128)
#define MPIU_REAL MPIU___FLOAT128
typedef __float128 PetscReal;
#define PetscSqrtReal(a)    sqrtq(a)
#endif /* PETSC_USE_REAL_* */

/*
    Complex number definitions
 */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_CLANGUAGE_CXX)
/* C++ support of complex number */
#include <complex>

#define PetscRealPart(a)      (a).real()
#define PetscImaginaryPart(a) (a).imag()
#define PetscAbsScalar(a)     std::abs(a)
#define PetscConj(a)          std::conj(a)
#define PetscSqrtScalar(a)    std::sqrt(a)
#define PetscPowScalar(a,b)   std::pow(a,b)
#define PetscExpScalar(a)     std::exp(a)
#define PetscLogScalar(a)     std::log(a)
#define PetscSinScalar(a)     std::sin(a)
#define PetscCosScalar(a)     std::cos(a)

#if defined(PETSC_USE_REAL_SINGLE)
typedef std::complex<float> PetscScalar;
#elif defined(PETSC_USE_REAL_DOUBLE)
typedef std::complex<double> PetscScalar;
#endif /* PETSC_USE_REAL_* */

#else /* PETSC_CLANGUAGE_CXX */
/*  C support of complex numbers: Requires C99 compliant compiler*/
#include <complex.h>

#if defined(PETSC_USE_REAL_SINGLE)
typedef float complex PetscScalar;

#define PetscRealPart(a)      crealf(a)
#define PetscImaginaryPart(a) cimagf(a)
#define PetscAbsScalar(a)     cabsf(a)
#define PetscConj(a)          conjf(a)
#define PetscSqrtScalar(a)    csqrtf(a)
#define PetscPowScalar(a,b)   cpowf(a,b)
#define PetscExpScalar(a)     cexpf(a)
#define PetscLogScalar(a)     clogf(a)
#define PetscSinScalar(a)     csinf(a)
#define PetscCosScalar(a)     ccosf(a)

#elif defined(PETSC_USE_REAL_DOUBLE)
typedef double complex PetscScalar;

#define PetscRealPart(a)      creal(a)
#define PetscImaginaryPart(a) cimag(a)
#define PetscAbsScalar(a)     cabs(a)
#define PetscConj(a)          conj(a)
#define PetscSqrtScalar(a)    csqrt(a)
#define PetscPowScalar(a,b)   cpow(a,b)
#define PetscExpScalar(a)     cexp(a)
#define PetscLogScalar(a)     clog(a)
#define PetscSinScalar(a)     csin(a)
#define PetscCosScalar(a)     ccos(a)

#endif /* PETSC_USE_REAL_* */
#endif /* PETSC_CLANGUAGE_CXX */

#if defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
#define MPIU_C_DOUBLE_COMPLEX MPI_C_DOUBLE_COMPLEX
#define MPIU_C_COMPLEX MPI_C_COMPLEX
#else
extern MPI_Datatype  MPIU_C_DOUBLE_COMPLEX;
extern MPI_Datatype  MPIU_C_COMPLEX;
#endif /* PETSC_HAVE_MPI_C_DOUBLE_COMPLEX */

#if defined(PETSC_USE_REAL_SINGLE)
#define MPIU_SCALAR MPIU_C_COMPLEX
#elif defined(PETSC_USE_REAL_DOUBLE)
#define MPIU_SCALAR MPIU_C_DOUBLE_COMPLEX
#endif /* PETSC_USE_REAL_* */

/*
    real number definitions
 */
#else /* PETSC_USE_COMPLEX */
#if defined(PETSC_USE_REAL_SINGLE)
#define MPIU_SCALAR           MPI_FLOAT
typedef float PetscScalar;
#elif defined(PETSC_USE_REAL_DOUBLE)
#define MPIU_SCALAR           MPI_DOUBLE
typedef double PetscScalar;
#elif defined(PETSC_USE_REAL___FLOAT128)
extern MPI_Datatype MPIU___FLOAT128;
#define MPIU_SCALAR MPIU___FLOAT128
typedef __float128 PetscScalar;
#endif /* PETSC_USE_REAL_* */
#define PetscRealPart(a)      (a)
#define PetscImaginaryPart(a) ((PetscReal)0.)
PETSC_STATIC_INLINE PetscReal PetscAbsScalar(PetscScalar a) {return a < 0.0 ? -a : a;}
#define PetscConj(a)          (a)
#if !defined(PETSC_USE_REAL___FLOAT128)
#define PetscSqrtScalar(a)    sqrt(a)
#define PetscPowScalar(a,b)   pow(a,b)
#define PetscExpScalar(a)     exp(a)
#define PetscLogScalar(a)     log(a)
#define PetscSinScalar(a)     sin(a)
#define PetscCosScalar(a)     cos(a)
#else /* PETSC_USE_REAL___FLOAT128 */
#include <quadmath.h>
#define PetscSqrtScalar(a)    sqrtq(a)
#define PetscPowScalar(a,b)   powq(a,b)
#define PetscExpScalar(a)     expq(a)
#define PetscLogScalar(a)     logq(a)
#define PetscSinScalar(a)     sinq(a)
#define PetscCosScalar(a)     cosq(a)
#endif /* PETSC_USE_REAL___FLOAT128 */

#endif /* PETSC_USE_COMPLEX */

#define PetscSign(a) (((a) >= 0) ? ((a) == 0 ? 0 : 1) : -1)
#define PetscAbs(a)  (((a) >= 0) ? (a) : -(a))

/* --------------------------------------------------------------------------*/

/*
   Certain objects may be created using either single or double precision.
   This is currently not used.
*/
typedef enum { PETSC_SCALAR_DOUBLE,PETSC_SCALAR_SINGLE, PETSC_SCALAR_LONG_DOUBLE } PetscScalarPrecision;

/* PETSC_i is the imaginary number, i */
extern  PetscScalar  PETSC_i;

/*MC
   PetscMin - Returns minimum of two numbers

   Synopsis:
   type PetscMin(type v1,type v2)

   Not Collective

   Input Parameter:
+  v1 - first value to find minimum of
-  v2 - second value to find minimum of

   
   Notes: type can be integer or floating point value

   Level: beginner


.seealso: PetscMin(), PetscClipInterval(), PetscAbsInt(), PetscAbsReal(), PetscSqr()

M*/
#define PetscMin(a,b)   (((a)<(b)) ?  (a) : (b))

/*MC
   PetscMax - Returns maxium of two numbers

   Synopsis:
   type max PetscMax(type v1,type v2)

   Not Collective

   Input Parameter:
+  v1 - first value to find maximum of
-  v2 - second value to find maximum of

   Notes: type can be integer or floating point value

   Level: beginner

.seealso: PetscMin(), PetscClipInterval(), PetscAbsInt(), PetscAbsReal(), PetscSqr()

M*/
#define PetscMax(a,b)   (((a)<(b)) ?  (b) : (a))

/*MC
   PetscClipInterval - Returns a number clipped to be within an interval

   Synopsis:
   type clip PetscClipInterval(type x,type a,type b)

   Not Collective

   Input Parameter:
+  x - value to use if within interval (a,b)
.  a - lower end of interval
-  b - upper end of interval

   Notes: type can be integer or floating point value

   Level: beginner

.seealso: PetscMin(), PetscMax(), PetscAbsInt(), PetscAbsReal(), PetscSqr()

M*/
#define PetscClipInterval(x,a,b)   (PetscMax((a),PetscMin((x),(b))))

/*MC
   PetscAbsInt - Returns the absolute value of an integer

   Synopsis:
   int abs PetscAbsInt(int v1)

   Not Collective

   Input Parameter:
.   v1 - the integer

   Level: beginner

.seealso: PetscMax(), PetscMin(), PetscAbsReal(), PetscSqr()

M*/
#define PetscAbsInt(a)  (((a)<0)   ? -(a) : (a))

/*MC
   PetscAbsReal - Returns the absolute value of an real number

   Synopsis:
   Real abs PetscAbsReal(PetscReal v1)

   Not Collective

   Input Parameter:
.   v1 - the double 


   Level: beginner

.seealso: PetscMax(), PetscMin(), PetscAbsInt(), PetscSqr()

M*/
#define PetscAbsReal(a) (((a)<0)   ? -(a) : (a))

/*MC
   PetscSqr - Returns the square of a number

   Synopsis:
   type sqr PetscSqr(type v1)

   Not Collective

   Input Parameter:
.   v1 - the value

   Notes: type can be integer or floating point value

   Level: beginner

.seealso: PetscMax(), PetscMin(), PetscAbsInt(), PetscAbsReal()

M*/
#define PetscSqr(a)     ((a)*(a))

/* ----------------------------------------------------------------------------*/
/*
     Basic constants 
*/
#if defined(PETSC_USE_REAL___FLOAT128)
#define PETSC_PI                 M_PIq
#elif defined(M_PI)
#define PETSC_PI                 M_PI
#else
#define PETSC_PI                 3.14159265358979323846264338327950288419716939937510582
#endif

#if !defined(PETSC_USE_64BIT_INDICES)
#define PETSC_MAX_INT            2147483647
#define PETSC_MIN_INT            (-PETSC_MAX_INT - 1)
#else
#define PETSC_MAX_INT            9223372036854775807L
#define PETSC_MIN_INT            (-PETSC_MAX_INT - 1)
#endif

#if defined(PETSC_USE_REAL_SINGLE)
#  define PETSC_MAX_REAL                3.40282346638528860e+38F
#  define PETSC_MIN_REAL                -PETSC_MAX_REAL
#  define PETSC_MACHINE_EPSILON         1.19209290e-07F
#  define PETSC_SQRT_MACHINE_EPSILON    3.45266983e-04F
#  define PETSC_SMALL                   1.e-5
#elif defined(PETSC_USE_REAL_DOUBLE)
#  define PETSC_MAX_REAL                1.7976931348623157e+308
#  define PETSC_MIN_REAL                -PETSC_MAX_REAL
#  define PETSC_MACHINE_EPSILON         2.2204460492503131e-16
#  define PETSC_SQRT_MACHINE_EPSILON    1.490116119384766e-08
#  define PETSC_SMALL                   1.e-10
#elif defined(PETSC_USE_REAL___FLOAT128)
#  define PETSC_MAX_REAL                FLT128_MAX
#  define PETSC_MIN_REAL                -FLT128_MAX
#  define PETSC_MACHINE_EPSILON         FLT128_EPSILON
#  define PETSC_SQRT_MACHINE_EPSILON    1.38777878078e-17
#  define PETSC_SMALL                   1.e-20
#endif

#if defined PETSC_HAVE_ADIC
/* Use MPI_Allreduce when ADIC is not available. */
extern PetscErrorCode  PetscGlobalMax(MPI_Comm, const PetscReal*,PetscReal*);
extern PetscErrorCode  PetscGlobalMin(MPI_Comm, const PetscReal*,PetscReal*);
extern PetscErrorCode  PetscGlobalSum(MPI_Comm, const PetscScalar*,PetscScalar*);
#endif

extern PetscErrorCode PetscIsInfOrNanScalar(PetscScalar);
extern PetscErrorCode PetscIsInfOrNanReal(PetscReal);

/* ----------------------------------------------------------------------------*/
/*
    PetscLogDouble variables are used to contain double precision numbers
  that are not used in the numerical computations, but rather in logging,
  timing etc.
*/
typedef double PetscLogDouble;
#define MPIU_PETSCLOGDOUBLE MPI_DOUBLE

#define PassiveReal   PetscReal
#define PassiveScalar PetscScalar

/*
    These macros are currently hardwired to match the regular data types, so there is no support for a different
    MatScalar from PetscScalar. We left the MatScalar in the source just in case we use it again.
 */
#define MPIU_MATSCALAR MPIU_SCALAR
typedef PetscScalar MatScalar;
typedef PetscReal MatReal;


PETSC_EXTERN_CXX_END
#endif
