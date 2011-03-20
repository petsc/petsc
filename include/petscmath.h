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
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
typedef std::complex<long double> PetscScalar;
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

#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
typedef long double complex PetscScalar;

#define PetscRealPart(a)      creall(a)
#define PetscImaginaryPart(a) cimagl(a)
#define PetscAbsScalar(a)     cabsl(a)
#define PetscConj(a)          conjl(a)
#define PetscSqrtScalar(a)    csqrtl(a)
#define PetscPowScalar(a,b)   cpowl(a,b)
#define PetscExpScalar(a)     cexpl(a)
#define PetscLogScalar(a)     clogl(a)
#define PetscSinScalar(a)     csinl(a)
#define PetscCosScalar(a)     ccosl(a)

#endif /* PETSC_USE_REAL_* */
#endif /* PETSC_CLANGUAGE_CXX */

#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
extern  MPI_Datatype  MPI_C_DOUBLE_COMPLEX;
extern  MPI_Datatype  MPI_C_COMPLEX;
#endif /* PETSC_HAVE_MPI_C_DOUBLE_COMPLEX */

#if defined(PETSC_USE_REAL_SINGLE)
#define MPIU_SCALAR MPI_C_COMPLEX
#elif defined(PETSC_USE_REAL_DOUBLE)
#define MPIU_SCALAR MPI_C_DOUBLE_COMPLEX
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#define MPIU_SCALAR error
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
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#define MPIU_SCALAR           MPI_LONG_DOUBLE
typedef long double PetscScalar;
#elif defined(PETSC_USE_REAL___FLOAT128)
extern MPI_Datatype MPIU___FLOAT128;
#define MPIU_SCALAR MPIU___FLOAT128
typedef __float128 PetscScalar;
#endif /* PETSC_USE_REAL_* */
#define PetscRealPart(a)      (a)
#define PetscImaginaryPart(a) (0.)
#define PetscAbsScalar(a)     (((a)<0.0)   ? -(a) : (a))
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

#if defined(PETSC_USE_REAL_SINGLE)
#define MPIU_REAL   MPI_FLOAT
typedef float PetscReal;
#elif defined(PETSC_USE_REAL_DOUBLE)
#define MPIU_REAL   MPI_DOUBLE
typedef double PetscReal;
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#define MPIU_REAL   MPI_LONG_DOUBLE
typedef long double PetscReal;
#elif defined(PETSC_USE_REAL___FLOAT128)
#define MPIU_REAL MPIU___FLOAT128
typedef __float128 PetscReal;
#endif /* PETSC_USE_REAL_* */

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


.seealso: PetscMin(), PetscAbsInt(), PetscAbsReal(), PetscSqr()

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

.seealso: PetscMin(), PetscAbsInt(), PetscAbsReal(), PetscSqr()

M*/
#define PetscMax(a,b)   (((a)<(b)) ?  (b) : (a))

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
#define PETSC_PI                 3.14159265358979323846264
#endif


#define PETSC_MAX_INT            2147483647
#define PETSC_MIN_INT            -2147483647

#if defined(PETSC_USE_REAL_SINGLE)
#if defined(MAXFLOAT)
#  define PETSC_REAL_MAX                 MAXFLOAT
#else
#  define PETSC_REAL_MAX                1.e30
#endif
#  define PETSC_MIN_REAL                -PETSC_REAL_MAX
#  define PETSC_MACHINE_EPSILON         1.e-7
#  define PETSC_SQRT_MACHINE_EPSILON    3.e-4
#  define PETSC_SMALL                   1.e-5
#elif defined(PETSC_USE_REAL_DOUBLE)
#  define PETSC_REAL_MAX                1.e300
#  define PETSC_MIN_REAL                -PETSC_REAL_MAX
#  define PETSC_MACHINE_EPSILON         1.e-14
#  define PETSC_SQRT_MACHINE_EPSILON    1.e-7
#  define PETSC_SMALL                   1.e-10
#elif defined(PETSC_USE_REAL___FLOAT128)
#  define PETSC_REAL_MAX                FLT128_MAX
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

/*MC
      PetscIsInfOrNan - Returns 1 if the input double has an infinity for Not-a-number (Nan) value, otherwise 0.

    Input Parameter:
.     a - the double


     Notes: uses the C99 standard isinf() and isnan() on systems where they exist.
      Otherwises uses ( (a - a) != 0.0), note that some optimizing compiles compile
      out this form, thus removing the check.

     Level: beginner

M*/
#if defined(PETSC_USE_REAL___FLOAT128)
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanScalar(PetscScalar a) {
  return isinfq(PetscAbsScalar(a)) || isnanq(PetscAbsScalar(a));
}
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanReal(PetscReal a) {
  return isinfq(a) || isnanq(a);
}
#elif defined(PETSC_HAVE_ISINF) && defined(PETSC_HAVE_ISNAN)
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanScalar(PetscScalar a) {
  return isinf(PetscAbsScalar(a)) || isnan(PetscAbsScalar(a));
}
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanReal(PetscReal a) {
  return isinf(a) || isnan(a);
}
#elif defined(PETSC_HAVE__FINITE) && defined(PETSC_HAVE__ISNAN)
#if defined(PETSC_HAVE_FLOAT_H)
#include "float.h"  /* Microsoft Windows defines _finite() in float.h */
#endif
#if defined(PETSC_HAVE_IEEEFP_H)
#include "ieeefp.h"  /* Solaris prototypes these here */
#endif
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanScalar(PetscScalar a) {
  return !_finite(PetscAbsScalar(a)) || _isnan(PetscAbsScalar(a));
}
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanReal(PetscReal a) {
  return !_finite(a) || _isnan(a);
}
#else
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanScalar(PetscScalar a) {
  return  ((a - a) != 0.0);
}
PETSC_STATIC_INLINE PetscErrorCode PetscIsInfOrNanReal(PetscReal a) {
  return ((a - a) != 0.0);
}
#endif


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
