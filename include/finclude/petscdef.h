!
!  $Id: petscdef.h,v 1.8 1998/12/03 19:40:24 balay Exp bsmith $;
!
!  Base include file for Fortran use of the PETSc package
!
#include "petscconf.h"
#include "mpif.h"
!
#define MPI_Comm integer
!
#define PetscTruth    integer
#define PetscDataType integer

!
! The real*8,complex*16 notatiton is used so that the 
! PETSc double/complex variables are not affected by 
! compiler options like -r4,-r8, sometimes invoked 
! by the user. NAG compiler does not like integer*4,real*8
!
! ???? All integers should also be changed to PetscFortranInt ?????
!

#if (SIZEOF_VOIDP == 8)
#define PetscOffset        integer*8
#define PetscFortranAddr   integer*8
#elif defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
#define PetscOffset        integer
#define PetscFortranAddr   integer
#else
#define PetscOffset        integer*4
#define PetscFortranAddr   integer*4
#endif

#if (SIZEOF_INT == 8)
#define PetscFortranInt integer*8
#elif defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
#define PetscFortranInt integer
#else
#define PetscFortranInt integer*4
#endif

#if defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
#define PetscFortranDouble  double precision
#define PetscFortranComplex complex (KIND=SELECTED_REAL_KIND(14))
#else
#define PetscFortranDouble  real*8
#define PetscFortranComplex complex*16
#endif
!
!     Flags
!
      integer   PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE
      integer   PETSC_DEFAULT_INTEGER,PETSC_DETERMINE
      integer   PETSC_FP_TRAP_OFF, PETSC_FP_TRAP_ON

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1)
      parameter (PETSC_DEFAULT_INTEGER = -2,PETSC_DETERMINE = -1)
      parameter (PETSC_FP_TRAP_OFF = 0, PETSC_FP_TRAP_ON = 1) 


      PetscFortranDouble PETSC_DEFAULT_DOUBLE_PRECISION

      parameter (PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)

!
!     Default Viewers.
!     Some more viewers, which are initialized using common-block
!     are declared in finclude/petsc.h
!
      PetscFortranAddr VIEWER_DRAW_WORLD
      PetscFortranAddr VIEWER_DRAW_WORLD_0,VIEWER_DRAW_WORLD_1
      PetscFortranAddr VIEWER_DRAW_WORLD_2,VIEWER_DRAW_SELF
      PetscFortranAddr VIEWER_SOCKET_WORLD
!
!     The numbers used below should match those in 
!     src/fortran/custom/zpetsc.h
!
      parameter (VIEWER_DRAW_WORLD_0 = -4) 
      parameter (VIEWER_DRAW_WORLD_1 = -5)
      parameter (VIEWER_DRAW_WORLD_2 = -6)
      parameter (VIEWER_DRAW_SELF = -7)
      parameter (VIEWER_SOCKET_WORLD = -8)
      parameter (VIEWER_DRAW_WORLD = VIEWER_DRAW_WORLD_0)


!
!     PETSc DataTypes
!
      integer PETSC_INT, PETSC_DOUBLE, PETSC_SHORT, PETSC_FLOAT
      integer PETSC_COMPLEX, PETSC_CHAR, PETSC_LOGICAL

      parameter (PETSC_INT=0, PETSC_DOUBLE=1, PETSC_SHORT=2)
      parameter (PETSC_FLOAT=3, PETSC_COMPLEX=4, PETSC_CHAR=5)
      parameter (PETSC_LOGICAL=6)

#if defined(USE_PETSC_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif     

!
! ------------------------------------------------------------------------
!     PETSc mathematics include file. Defines certain basic mathematical 
!    constants and functions for working with single and double precision
!    floating point numbers as well as complex and integers.
!
!
!     Representation of complex i
!
      PetscFortranComplex PETSC_i
      parameter (PETSC_i = (0.0d0,1.0d0))
!
!     Macro for templating between real and complex
!
#if defined(USE_PETSC_COMPLEX)
#define Scalar       PetscFortranComplex
!
! F90 uses real(), conjg() when KIND parameter is used.
!
#if defined (HAVE_NAGF90) || defined (HAVE_IRIXF90) || defined (PARCH_freebsd)
#define PetscReal(a) real(a)
#define PetscConj(a) conjg(a)
#else
#define PetscReal(a) dreal(a)
#define PetscConj(a) dconjg(a)
#endif
#define MPIU_SCALAR   MPI_DOUBLE_COMPLEX
#else
#define Scalar       PetscFortranDouble
#define PetscReal(a) a
#define PetscConj(a) a
#define MPIU_SCALAR  MPI_DOUBLE_PRECISION
#endif
!
!    Allows the matrix Fortran Kernels to work with single precision
!    matrix data structures
!
#if defined(USE_PETSC_COMPLEX)

#define MatScalar Scalar 

#elif defined(USE_MAT_SINGLE)

#define MatScalar real*4

#else

#define MatScalar Scalar

#endif
!
!
!     Basic constants
! 
      PetscFortranDouble PETSC_PI,PETSC_DEGREES_TO_RADIANS
      PetscFortranDouble PETSC_MAX,PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0)
      parameter (PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0)
      parameter (PETSC_MAX = 1.d300, PETSC_MIN = -1.d300)


!
!     Declare PETSC_NULL_OBJECT
!
#define PETSC_NULL_OBJECT PETSC_NULL_INTEGER
!
!     PLogDouble variables are used to contain double precision numbers
!     that are not used in the numerical computations, but rather in logging,
!     timing etc.
!
#define PetscObject PetscFortranAddr
#define PLogDouble  PetscFortranDouble



! ----------------------------------------------------------------------------
