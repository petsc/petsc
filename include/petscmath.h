/*
   
      PETSc mathematics include file. Defines certain basic mathematical 
    constants and functions for working with single and double precision
    floating point numbers as well as complex and integers.

    This file is included by petsc.h and should not be used directly.

*/

#if !defined(__PETSCMATH_H)
#define __PETSCMATH_H
#include <math.h>
PETSC_EXTERN_CXX_BEGIN

extern  MPI_Datatype        MPIU_2SCALAR;
extern  MPI_Datatype        MPIU_2INT;
/*

     Defines operations that are different for complex and real numbers;
   note that one cannot really mix the use of complex and real in the same 
   PETSc program. All PETSc objects in one program are built around the object
   PetscScalar which is either always a double or a complex.

*/

#define PetscExpPassiveScalar(a) PetscExpScalar()

#if defined(PETSC_USE_COMPLEX)

/*
   PETSc now only supports std::complex
*/
#include <complex>

extern  MPI_Datatype        MPIU_COMPLEX;
#define MPIU_SCALAR         MPIU_COMPLEX
#if defined(PETSC_USE_MAT_SINGLE)
#define MPIU_MATSCALAR        ??Notdone
#else
#define MPIU_MATSCALAR      MPIU_COMPLEX
#endif

#define PetscRealPart(a)        (a).real()
#define PetscImaginaryPart(a)   (a).imag()
#define PetscAbsScalar(a)   std::abs(a)
#define PetscConj(a)        std::conj(a)
#define PetscSqrtScalar(a)  std::sqrt(a)
#define PetscPowScalar(a,b) std::pow(a,b)
#define PetscExpScalar(a)   std::exp(a)
#define PetscSinScalar(a)   std::sin(a)
#define PetscCosScalar(a)   std::cos(a)

typedef std::complex<double> PetscScalar;

/* Compiling for real numbers only */
#else
#  if defined(PETSC_USE_SINGLE)
#    define MPIU_SCALAR           MPI_FLOAT
#  else
#    define MPIU_SCALAR           MPI_DOUBLE
#  endif
#  if defined(PETSC_USE_MAT_SINGLE) || defined(PETSC_USE_SINGLE)
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
  typedef float PetscScalar;
#  else
  typedef double PetscScalar;
#  endif
#endif

#if defined(PETSC_USE_SINGLE)
#  define MPIU_REAL   MPI_FLOAT
#else
#  define MPIU_REAL   MPI_DOUBLE
#endif

#define PetscSign(a) (((a) >= 0) ? ((a) == 0 ? 0 : 1) : -1)
#define PetscAbs(a)  (((a) >= 0) ? a : -a)
/*
       Allows compiling PETSc so that matrix values are stored in 
   single precision but all other objects still use double
   precision. This does not work for complex numbers in that case
   it remains double

          EXPERIMENTAL! NOT YET COMPLETELY WORKING
*/

#if defined(PETSC_USE_MAT_SINGLE)
typedef float MatScalar;
#else
typedef PetscScalar MatScalar;
#endif

#if defined(PETSC_USE_COMPLEX)
typedef double MatReal;
#elif defined(PETSC_USE_MAT_SINGLE) || defined(PETSC_USE_SINGLE)
typedef float MatReal;
#else
typedef double MatReal;
#endif

#if defined(PETSC_USE_SINGLE)
  typedef float PetscReal;
#else 
  typedef double PetscReal;
#endif

/* --------------------------------------------------------------------------*/

/*
   Certain objects may be created using either single
  or double precision.
*/
typedef enum { PETSC_SCALAR_DOUBLE,PETSC_SCALAR_SINGLE } PetscScalarPrecision;

/* PETSC_i is the imaginary number, i */
extern  PetscScalar       PETSC_i;

/*MC
   PetscMin - Returns minimum of two numbers

   Input Parameter:
+  v1 - first value to find minimum of
-  v2 - second value to find minimum of

   Synopsis:
   type PetscMin(type v1,type v2)

   Notes: type can be integer or floating point value

   Level: beginner


.seealso: PetscMin(), PetscAbsInt(), PetscAbsReal(), PetscSqr()

M*/
#define PetscMin(a,b)   (((a)<(b)) ?  (a) : (b))

/*MC
   PetscMax - Returns maxium of two numbers

   Input Parameter:
+  v1 - first value to find maximum of
-  v2 - second value to find maximum of

   Synopsis:
   type max PetscMax(type v1,type v2)

   Notes: type can be integer or floating point value

   Level: beginner

.seealso: PetscMin(), PetscAbsInt(), PetscAbsReal(), PetscSqr()

M*/
#define PetscMax(a,b)   (((a)<(b)) ?  (b) : (a))

/*MC
   PetscAbsInt - Returns the absolute value of an integer

   Input Parameter:
.   v1 - the integer

   Synopsis:
   int abs PetscAbsInt(int v1)


   Level: beginner

.seealso: PetscMax(), PetscMin(), PetscAbsReal(), PetscSqr()

M*/
#define PetscAbsInt(a)  (((a)<0)   ? -(a) : (a))

/*MC
   PetscAbsReal - Returns the absolute value of an real number

   Input Parameter:
.   v1 - the double 

   Synopsis:
   int abs PetscAbsReal(PetscReal v1)


   Level: beginner

.seealso: PetscMax(), PetscMin(), PetscAbsInt(), PetscSqr()

M*/
#define PetscAbsReal(a) (((a)<0)   ? -(a) : (a))

/*MC
   PetscSqr - Returns the square of a number

   Input Parameter:
.   v1 - the value

   Synopsis:
   type sqr PetscSqr(type v1)

   Notes: type can be integer or floating point value

   Level: beginner

.seealso: PetscMax(), PetscMin(), PetscAbsInt(), PetscAbsReal()

M*/
#define PetscSqr(a)     ((a)*(a))

/* ----------------------------------------------------------------------------*/
/*
     Basic constants
*/
#define PETSC_PI                 3.14159265358979323846264
#define PETSC_DEGREES_TO_RADIANS 0.01745329251994
#define PETSC_MAX_INT            1000000000
#define PETSC_MIN_INT            -1000000000

#if defined(PETSC_USE_SINGLE)
#  define PETSC_MAX                     1.e30
#  define PETSC_MIN                    -1.e30
#  define PETSC_MACHINE_EPSILON         1.e-7
#  define PETSC_SQRT_MACHINE_EPSILON    3.e-4
#  define PETSC_SMALL                   1.e-5
#else
#  define PETSC_MAX                     1.e300
#  define PETSC_MIN                    -1.e300
#  define PETSC_MACHINE_EPSILON         1.e-14
#  define PETSC_SQRT_MACHINE_EPSILON    1.e-7
#  define PETSC_SMALL                   1.e-10
#endif

EXTERN PetscErrorCode PetscGlobalMax(PetscReal*,PetscReal*,MPI_Comm);
EXTERN PetscErrorCode PetscGlobalMin(PetscReal*,PetscReal*,MPI_Comm);
EXTERN PetscErrorCode PetscGlobalSum(PetscScalar*,PetscScalar*,MPI_Comm);


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
#if !defined(_petsc_mpi_uni)
#define MPIU_PETSCLOGDOUBLE MPI_DOUBLE
#endif
#endif

#define PassiveReal   PetscReal
#define PassiveScalar PetscScalar

#define PETSCMAP1_a(a,b)  a ## _ ## b
#define PETSCMAP1_b(a,b)  PETSCMAP1_a(a,b)
#define PETSCMAP1(a)      PETSCMAP1_b(a,PetscScalar)

PETSC_EXTERN_CXX_END
#endif
