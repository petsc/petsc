/* $Id: petscmath.h,v 1.21 2001/04/10 19:37:48 bsmith Exp bsmith $ */
/*
   
      PETSc mathematics include file. Defines certain basic mathematical 
    constants and functions for working with single and double precision
    floating point numbers as well as complex and integers.

    This file is included by petsc.h and should not be used directly.

*/

#if !defined(__PETSCMATH_H)
#define __PETSCMATH_H
#include <math.h>

/*

     Defines operations that are different for complex and real numbers;
   note that one cannot really mix the use of complex and real in the same 
   PETSc program. All PETSc objects in one program are built around the object
   Scalar which is either always a double or a complex.

*/
#if defined(PETSC_USE_COMPLEX)

#if defined (PETSC_HAVE_STD_COMPLEX)
#include <complex>
#elif defined(PETSC_HAVE_NONSTANDARD_COMPLEX_H)
#include PETSC_HAVE_NONSTANDARD_COMPLEX_H
#else
#include <complex.h>
#endif

extern  MPI_Datatype        MPIU_COMPLEX;
#define MPIU_SCALAR         MPIU_COMPLEX
#if defined(PETSC_USE_MAT_SINGLE)
#define MPIU_MATSCALAR        ??Notdone
#else
#define MPIU_MATSCALAR      MPIU_COMPLEX
#endif

#if defined (PETSC_HAVE_STD_COMPLEX)
#define PetscRealPart(a)        (a).real()
#define PetscImaginaryPart(a)   (a).imag()
#define PetscAbsScalar(a)   std::abs(a)
#define PetscConj(a)        std::conj(a)
#define PetscSqrtScalar(a)  std::sqrt(a)
#define PetscPowScalar(a,b) std::pow(a,b)
#define PetscExpScalar(a)   std::exp(a)
#define PetscSinScalar(a)   std::sin(a)
#define PetscCosScalar(a)   std::cos(a)
#else
#define PetscRealPart(a)        real(a)
#define PetscImaginaryPart(a)   imag(a)
#define PetscAbsScalar(a)   abs(a)
#define PetscConj(a)        conj(a)
#define PetscSqrtScalar(a)  sqrt(a)
#define PetscPowScalar(a,b) pow(a,b)
#define PetscExpScalar(a)   exp(a)
#define PetscSinScalar(a)   sin(a)
#define PetscCosScalar(a)   cos(a)
#endif
/*
  The new complex class for GNU C++ is based on templates and is not backward
  compatible with all previous complex class libraries.
*/
#if defined(PETSC_HAVE_STD_COMPLEX)
#define Scalar             std::complex<double>
#elif defined(PETSC_HAVE_TEMPLATED_COMPLEX)
#define Scalar             complex<double>
#else
#define Scalar             complex
#endif

/* Compiling for real numbers only */
#else
#  define MPIU_SCALAR           MPI_DOUBLE
#  if defined(PETSC_USE_MAT_SINGLE)
#    define MPIU_MATSCALAR        MPI_FLOAT
#  else
#    define MPIU_MATSCALAR        MPI_DOUBLE
#  endif
#  define PetscRealPart(a)      (a)
#  define PetscImaginaryPart(a) (a)
#  define PetscAbsScalar(a)     (((a)<0.0)   ? -(a) : (a))
#  define PetscConj(a)          (a)
#  define PetscSqrtScalar(a)    sqrt(a)
#  define PetscPowScalar(a,b)   pow(a,b)
#  define PetscExpScalar(a)     exp(a)
#  define PetscSinScalar(a)     sin(a)
#  define PetscCosScalar(a)     cos(a)

#  if defined(PETSC_USE_SINGLE)
#    define Scalar            float
#  else
#    define Scalar            double
#  endif
#endif

/*
       Allows compiling PETSc so that matrix values are stored in 
   single precision but all other objects still use double
   precision. This does not work for complex numbers in that case
   it remains double

          EXPERIMENTAL! NOT YET COMPLETELY WORKING
*/
#if defined(PETSC_USE_COMPLEX)

#define MatScalar Scalar 
#define MatReal   double

#elif defined(PETSC_USE_MAT_SINGLE)

#define MatScalar float
#define MatReal   float

#else

#define MatScalar Scalar
#define MatReal   double

#endif

#if defined(PETSC_USE_SINGLE)
#define PetscReal float
#else 
#define PetscReal double
#endif

/* --------------------------------------------------------------------------*/

/*
   Certain objects may be created using either single
  or double precision.
*/
typedef enum { SCALAR_DOUBLE,SCALAR_SINGLE } ScalarPrecision;

/* PETSC_i is the imaginary number, i */
extern  Scalar            PETSC_i;

#define PetscMin(a,b)      (((a)<(b)) ? (a) : (b))
#define PetscMax(a,b)      (((a)<(b)) ? (b) : (a))
#define PetscAbsInt(a)     (((a)<0)   ? -(a) : (a))
#define PetscAbsDouble(a)  (((a)<0)   ? -(a) : (a))

/* ----------------------------------------------------------------------------*/
/*
     Basic constants
*/
#define PETSC_PI                 3.14159265358979323846264
#define PETSC_DEGREES_TO_RADIANS 0.01745329251994
#define PETSC_MAX                1.e300
#define PETSC_MIN                -1.e300
#define PETSC_MAX_INT            1000000000;
#define PETSC_MIN_INT            -1000000000;

/* ----------------------------------------------------------------------------*/
/*
    PetscLogDouble variables are used to contain double precision numbers
  that are not used in the numerical computations, but rather in logging,
  timing etc.
*/
typedef double PetscLogDouble;
/*
      Once PETSc is compiling with a ADIC enhanced version of MPI
   we will create a new MPI_Datatype for the inactive double variables.
*/
#if defined(AD_DERIV_H)
/* extern  MPI_Datatype  MPIU_PETSCLOGDOUBLE; */
#else
#if !defined(USING_MPIUNI)
#define MPIU_PETSCLOGDOUBLE MPI_DOUBLE
#endif
#endif

#define InactiveDouble double
#define InactiveScalar Scalar

#define PETSCMAP1_a(a,b)  a ## _ ## b
#define PETSCMAP1_b(a,b)  PETSCMAP1_a(a,b)
#define PETSCMAP1(a)      PETSCMAP1_b(a,Scalar)
#endif
