/* $Id: petscmath.h,v 1.1 1997/08/29 20:00:06 bsmith Exp bsmith $ */
/*
   
      PETSc mathematics include file. Defines certain basic mathematical 
    constants and functions for working with single and double precision
    floating point numbers as well as complex and integers.

*/
#if !defined(__PETSCMATH_PACKAGE)
#define __PETSCMAT_PACKAGE

#include "petsc.h"

/*
    Defines operations that are different for complex and real numbers
*/
#if defined(PETSC_COMPLEX)
#if defined(HAVE_NONSTANDARD_COMPLEX_H)
#include HAVE_NONSTANDARD_COMPLEX_H
#else
#include <complex.h>
#endif
extern  MPI_Datatype      MPIU_COMPLEX;
#define MPIU_SCALAR       MPIU_COMPLEX
#define PetscReal(a)      real(a)
#define PetscImaginary(a) imag(a)
#define PetscAbsScalar(a) abs(a)
#define PetscConj(a)      conj(a)
/*
  The new complex class for GNU C++ is based on templates and is not backward
  compatible with all previous complex class libraries.
*/
#if defined(USES_TEMPLATED_COMPLEX)
#define Scalar            complex<double>
#else
#define Scalar            complex
#endif

/* Compiling for real numbers only */
#else
#define MPIU_SCALAR       MPI_DOUBLE
#define PetscReal(a)      (a)
#define PetscImaginary(a) (a)
#define PetscAbsScalar(a) ( ((a)<0.0)   ? -(a) : (a) )
#define Scalar            double
#define PetscConj(a)      (a)
#endif

/* --------------------------------------------------------------------------*/

/*
   Certain objects may be created using either single
  or double precision.
*/
typedef enum { SCALAR_DOUBLE, SCALAR_SINGLE } ScalarPrecision;

/* PETSC_i is the imaginary number, i */
extern  Scalar            PETSC_i;

#define PetscMin(a,b)      ( ((a)<(b)) ? (a) : (b) )
#define PetscMax(a,b)      ( ((a)<(b)) ? (b) : (a) )
#define PetscAbsInt(a)     ( ((a)<0)   ? -(a) : (a) )
#define PetscAbsDouble(a)  ( ((a)<0)   ? -(a) : (a) )

/* ----------------------------------------------------------------------------*/
/*
     Basic constants
*/
#define PETSC_PI                 3.14159265358979323846264
#define PETSC_DEGREES_TO_RADIANS 0.01745329251994
#define PETSC_MAX                1.e300
#define PETSC_MIN                -1.e300

/* ----------------------------------------------------------------------------*/
/*
    PLogDouble variables are used to contain double precision numbers
  that are not used in the numerical computations, but rather in logging,
  timing etc.
*/
typedef double PLogDouble;
/*
      Once PETSc is compiling with a ADIC enhanced version of MPI
   we will create a new MPI_Datatype for the inactive double variables.
*/
#if defined(AD_DERIV_H)
/* extern  MPI_Datatype  MPIU_PLOGDOUBLE; */
#else
#if !defined(PETSC_USING_MPIUNI)
#define MPIU_PLOGDOUBLE MPI_DOUBLE
#endif
#endif


#endif
