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

extern  MPI_Datatype PETSC_DLLEXPORT MPIU_2SCALAR;
extern  MPI_Datatype PETSC_DLLEXPORT MPIU_2INT;
/*

     Defines operations that are different for complex and real numbers;
   note that one cannot really mix the use of complex and real in the same 
   PETSc program. All PETSc objects in one program are built around the object
   PetscScalar which is either always a double or a complex.

*/

#define PetscExpPassiveScalar(a) PetscExpScalar()

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_CLANGUAGE_CXX)
/*
   C++ support of complex numbers: Original support
*/
#include <complex>

#if defined(PETSC_USE_SCALAR_SINGLE)
/*
    For d double and c single complex defines the following operations
       d == c
       c == d
       d != c
       c != d
       d / c
       c /d
       d * c
       c * d
       d - c
       c - d
       d + c
       c + d
*/
namespace std
{
  template<typename _Tp>
    inline bool
    operator==(const double& __x, const complex<_Tp>& __y)
    { return __x == __y.real() && _Tp() == __y.imag(); }
  template<typename _Tp>
    inline bool
    operator==(const complex<_Tp>& __x, const double& __y)
    { return __x.real() == __y && __x.imag() == _Tp(); }
  template<typename _Tp>
    inline bool
    operator!=(const complex<_Tp>& __x, const double& __y)
    { return __x.real() != __y || __x.imag() != _Tp(); }
  template<typename _Tp>
    inline bool
    operator!=(const double& __x, const complex<_Tp>& __y)
    { return __x != __y.real() || _Tp() != __y.imag(); }
  template<typename _Tp>
    inline complex<_Tp>
    operator/(const complex<_Tp>& __x, const double& __y)
    {
      complex<_Tp> __r = __x;
      __r /= ((float)__y);
      return __r;
    }
  template<typename _Tp>
    inline complex<_Tp>
    operator/(const double& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = (float)__x;
      __r /= __y;
      return __r;
    }
  template<typename _Tp>
    inline complex<_Tp>
    operator*(const complex<_Tp>& __x, const double& __y)
    {
      complex<_Tp> __r = __x;
      __r *= ((float)__y);
      return __r;
    }
  template<typename _Tp>
    inline complex<_Tp>
    operator*(const double& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = (float)__x;
      __r *= __y;
      return __r;
    }
  template<typename _Tp>
    inline complex<_Tp>
    operator-(const complex<_Tp>& __x, const double& __y)
    {
      complex<_Tp> __r = __x;
      __r -= ((float)__y);
      return __r;
    }
  template<typename _Tp>
    inline complex<_Tp>
    operator-(const double& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = (float)__x;
      __r -= __y;
      return __r;
    }
  template<typename _Tp>
    inline complex<_Tp>
    operator+(const complex<_Tp>& __x, const double& __y)
    {
      complex<_Tp> __r = __x;
      __r += ((float)__y);
      return __r;
    }
  template<typename _Tp>
    inline complex<_Tp>
    operator+(const double& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = (float)__x;
      __r += __y;
      return __r;
    }
}
#endif



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

#if defined(PETSC_USE_SCALAR_SINGLE)
typedef std::complex<float> PetscScalar;
#elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
typedef std::complex<long double> PetscScalar;
#elif defined(PETSC_USE_SCALAR_INT)
typedef std::complex<int> PetscScalar;
#else
typedef std::complex<double> PetscScalar;
#endif
#else
#include <complex.h>

/* 
   C support of complex numbers: Warning it needs a 
   C90 compliant compiler to work...
 */

#if defined(PETSC_USE_SCALAR_SINGLE)
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
#elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
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

#else
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
#endif
#endif

#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
extern  MPI_Datatype PETSC_DLLEXPORT MPI_C_DOUBLE_COMPLEX;
extern  MPI_Datatype PETSC_DLLEXPORT MPI_C_COMPLEX;
#endif

#if defined(PETSC_USE_SCALAR_SINGLE)
#define MPIU_SCALAR         MPI_C_COMPLEX
#else
#define MPIU_SCALAR         MPI_C_DOUBLE_COMPLEX
#endif
#if defined(PETSC_USE_SCALAR_MAT_SINGLE)
#define MPIU_MATSCALAR        ??Notdone
#else
#define MPIU_MATSCALAR      MPI_C_DOUBLE_COMPLEX
#endif


/* Compiling for real numbers only */
#else
#  if defined(PETSC_USE_SCALAR_SINGLE)
#    define MPIU_SCALAR           MPI_FLOAT
#  elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
#    define MPIU_SCALAR           MPI_LONG_DOUBLE
#  elif defined(PETSC_USE_SCALAR_INT)
#    define MPIU_SCALAR           MPI_INT
#  elif defined(PETSC_USE_SCALAR_QD_DD)
#    define MPIU_SCALAR           MPIU_QD_DD
#  else
#    define MPIU_SCALAR           MPI_DOUBLE
#  endif
#  if defined(PETSC_USE_SCALAR_MAT_SINGLE) || defined(PETSC_USE_SCALAR_SINGLE)
#    define MPIU_MATSCALAR        MPI_FLOAT
#  elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
#    define MPIU_MATSCALAR        MPI_LONG_DOUBLE
#  elif defined(PETSC_USE_SCALAR_INT)
#    define MPIU_MATSCALAR        MPI_INT
#  elif defined(PETSC_USE_SCALAR_QD_DD)
#    define MPIU_MATSCALAR        MPIU_QD_DD
#  else
#    define MPIU_MATSCALAR        MPI_DOUBLE
#  endif
#  define PetscRealPart(a)      (a)
#  define PetscImaginaryPart(a) (0.)
#  define PetscAbsScalar(a)     (((a)<0.0)   ? -(a) : (a))
#  define PetscConj(a)          (a)
#  define PetscSqrtScalar(a)    sqrt(a)
#  define PetscPowScalar(a,b)   pow(a,b)
#  define PetscExpScalar(a)     exp(a)
#  define PetscLogScalar(a)     log(a)
#  define PetscSinScalar(a)     sin(a)
#  define PetscCosScalar(a)     cos(a)

#  if defined(PETSC_USE_SCALAR_SINGLE)
  typedef float PetscScalar;
#  elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
  typedef long double PetscScalar;
#  elif defined(PETSC_USE_SCALAR_INT)
  typedef int PetscScalar;
#  elif defined(PETSC_USE_SCALAR_QD_DD)
#  include "qd/dd_real.h"
  typedef dd_real PetscScalar;
#  else
  typedef double PetscScalar;
#  endif
#endif

#if defined(PETSC_USE_SCALAR_SINGLE)
#  define MPIU_REAL   MPI_FLOAT
#elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
#  define MPIU_REAL   MPI_LONG_DOUBLE
#elif defined(PETSC_USE_SCALAR_INT)
#  define MPIU_REAL   MPI_INT
#elif defined(PETSC_USE_SCALAR_QD_DD)
#  define MPIU_REAL   MPIU_QD_DD
#else
#  define MPIU_REAL   MPI_DOUBLE
#endif

#if defined(PETSC_USE_SCALAR_QD_DD)
extern  MPI_Datatype PETSC_DLLEXPORT MPIU_QD_DD;
#endif

#define PetscSign(a) (((a) >= 0) ? ((a) == 0 ? 0 : 1) : -1)
#define PetscAbs(a)  (((a) >= 0) ? (a) : -(a))
/*
       Allows compiling PETSc so that matrix values are stored in 
   single precision but all other objects still use double
   precision. This does not work for complex numbers in that case
   it remains double

          EXPERIMENTAL! NOT YET COMPLETELY WORKING
*/

#if defined(PETSC_USE_SCALAR_MAT_SINGLE)
typedef float MatScalar;
#else
typedef PetscScalar MatScalar;
#endif

#if defined(PETSC_USE_SCALAR_SINGLE)
  typedef float PetscReal;
#elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
  typedef long double PetscReal;
#elif defined(PETSC_USE_SCALAR_INT)
  typedef int PetscReal;
#elif defined(PETSC_USE_SCALAR_QD_DD)
  typedef dd_real PetscReal;
#else 
  typedef double PetscReal;
#endif

#if defined(PETSC_USE_COMPLEX)
typedef PetscReal MatReal;
#elif defined(PETSC_USE_SCALAR_MAT_SINGLE) || defined(PETSC_USE_SCALAR_SINGLE)
typedef float MatReal;
#else
typedef PetscReal MatReal;
#endif


/* --------------------------------------------------------------------------*/

/*
   Certain objects may be created using either single
  or double precision.
*/
typedef enum { PETSC_SCALAR_DOUBLE,PETSC_SCALAR_SINGLE, PETSC_SCALAR_LONG_DOUBLE, PETSC_SCALAR_QD_DD } PetscScalarPrecision;

/* PETSC_i is the imaginary number, i */
extern  PetscScalar PETSC_DLLEXPORT PETSC_i;

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
     Basic constants - These should be done much better
*/
#define PETSC_PI                 3.14159265358979323846264
#define PETSC_DEGREES_TO_RADIANS 0.01745329251994
#define PETSC_MAX_INT            2147483647
#define PETSC_MIN_INT            -2147483647

#if defined(PETSC_USE_SCALAR_SINGLE)
#  define PETSC_MAX                     1.e30
#  define PETSC_MIN                    -1.e30
#  define PETSC_MACHINE_EPSILON         1.e-7
#  define PETSC_SQRT_MACHINE_EPSILON    3.e-4
#  define PETSC_SMALL                   1.e-5
#elif defined(PETSC_USE_SCALAR_INT)
#  define PETSC_MAX                     PETSC_MAX_INT
#  define PETSC_MIN                     PETSC_MIN_INT
#  define PETSC_MACHINE_EPSILON         1
#  define PETSC_SQRT_MACHINE_EPSILON    1
#  define PETSC_SMALL                   0
#elif defined(PETSC_USE_SCALAR_QD_DD)
#  define PETSC_MAX                     1.e300
#  define PETSC_MIN                    -1.e300 
#  define PETSC_MACHINE_EPSILON         1.e-30
#  define PETSC_SQRT_MACHINE_EPSILON    1.e-15
#  define PETSC_SMALL                   1.e-25
#else
#  define PETSC_MAX                     1.e300
#  define PETSC_MIN                    -1.e300
#  define PETSC_MACHINE_EPSILON         1.e-14
#  define PETSC_SQRT_MACHINE_EPSILON    1.e-7
#  define PETSC_SMALL                   1.e-10
#endif

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGlobalMax(PetscReal*,PetscReal*,MPI_Comm);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGlobalMin(PetscReal*,PetscReal*,MPI_Comm);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGlobalSum(PetscScalar*,PetscScalar*,MPI_Comm);

/*MC
      PetscIsInfOrNan - Returns 1 if the input double has an infinity for Not-a-number (Nan) value, otherwise 0.

    Input Parameter:
.     a - the double


     Notes: uses the C99 standard isinf() and isnan() on systems where they exist.
      Otherwises uses ( (a - a) != 0.0), note that some optimizing compiles compile
      out this form, thus removing the check.

     Level: beginner

M*/
#if defined(PETSC_HAVE_ISINF) && defined(PETSC_HAVE_ISNAN)
#define PetscIsInfOrNanScalar(a) (isinf(PetscAbsScalar(a)) || isnan(PetscAbsScalar(a)))
#define PetscIsInfOrNanReal(a) (isinf(a) || isnan(a))
#elif defined(PETSC_HAVE__FINITE) && defined(PETSC_HAVE__ISNAN)
#if defined(PETSC_HAVE_FLOAT_H)
#include "float.h"  /* windows defines _finite() in float.h */
#endif
#if defined(PETSC_HAVE_IEEEFP_H)
#include "ieeefp.h"  /* Solaris prototypes these here */
#endif
#define PetscIsInfOrNanScalar(a) (!_finite(PetscAbsScalar(a)) || _isnan(PetscAbsScalar(a)))
#define PetscIsInfOrNanReal(a) (!_finite(a) || _isnan(a))
#else
#define PetscIsInfOrNanScalar(a) ((a - a) != 0.0)
#define PetscIsInfOrNanReal(a) ((a - a) != 0.0)
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


PETSC_EXTERN_CXX_END
#endif
