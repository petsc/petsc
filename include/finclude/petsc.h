!
!  $Id: petsc.h,v 1.68 1998/04/07 00:54:59 balay Exp balay $;
!
!  Base include file for Fortran use of the PETSc package
!
#define MPI_Comm integer
!
#include "mpif.h"

#define PetscTruth    integer
#define PetscDataType integer 

#if defined(HAVE_64BITS)
#define PetscOffset integer*8
#else
#define PetscOffset integer
#endif
#if defined(HAVE_64BITS) && !defined(USE_POINTER_CONVERSION)
#define PetscFortranAddr   integer*8
#else
#define PetscFortranAddr   integer
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


      double precision PETSC_DEFAULT_DOUBLE_PRECISION

      parameter (PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)

!
!     Default Viewers
!
      PetscFortranAddr VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF
      PetscFortranAddr VIEWER_STDOUT_WORLD, VIEWER_DRAWX_WORLD
      PetscFortranAddr VIEWER_DRAWX_WORLD_0,VIEWER_DRAWX_WORLD_1
      PetscFortranAddr VIEWER_DRAWX_WORLD_2,VIEWER_DRAWX_SELF
      PetscFortranAddr VIEWER_MATLAB_WORLD
!
!     The numbers used below should match those in 
!     src/fortran/custom/zpetsc.h
!
      parameter (VIEWER_DRAWX_WORLD_0 = -4) 
      parameter (VIEWER_DRAWX_WORLD_1 = -5)
      parameter (VIEWER_DRAWX_WORLD_2 = -6)
      parameter (VIEWER_DRAWX_SELF = -7)
      parameter (VIEWER_MATLAB_WORLD = -8)
      parameter (VIEWER_DRAWX_WORLD = VIEWER_DRAWX_WORLD_0)

!
!     Fortran Null
!
      integer        PETSC_NULL
      character*(80) PETSC_NULL_CHARACTER

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
! PETSc world communicator
!
      MPI_Comm PETSC_COMM_WORLD, PETSC_COMM_SELF

!
!     A string should be in a different common block
!  
      common /petscfortran1/ PETSC_NULL_CHARACTER
      common /petscfortran2/ PETSC_NULL
      common /petscfortran3/ VIEWER_STDOUT_SELF,VIEWER_STDERR_SELF
      common /petscfortran4/ VIEWER_STDOUT_WORLD
      common /petscfortran5/ PETSC_COMM_WORLD,PETSC_COMM_SELF
!
!     Macros for error checking
!
#if defined(USE_PETSC_DEBUG)
#define SETERRA(n,p,s) call MPI_Abort(PETSC_COMM_WORLD,n)
#define CHKERRA(n) if (n .ne. 0) call MPI_Abort(PETSC_COMM_WORLD,n)
#else
#define SETERRA(n,p,s)   
#define CHKERRA(n)     
#endif
!
!
!     End of base Fortran include file for the PETSc package
!
! ------------------------------------------------------------------------
!     PETSc mathematics include file. Defines certain basic mathematical 
!    constants and functions for working with single and double precision
!    floating point numbers as well as complex and integers.
!
!
! Nag f90 compiler chokes on double complex
!
#if !defined (HAVE_NAGF90)
!     Representation of complex i
!
      double complex PETSC_i
      parameter (PETSC_i = (0,1.0))
#endif
!
!     Macro for templating between real and complex
!
#if defined(USE_PETSC_COMPLEX)
!
! Some Antique IRIX- f90 Compilers require explicit
! prototype for  dconjg() 
!
#if defined(PARCH_IRIX) || defined (PARCH_IRIX5)
      external dconjg
      double complex dconjg
#endif

#define PetscReal(a)  real(a)
#define PetscConj(a)  dconjg(a)
#define Scalar        double complex
#define MPIU_SCALAR   MPI_DOUBLE_COMPLEX
#else
#define PetscReal(a) a
#define PetscConj(a) a
#define Scalar       double precision
#define MPIU_SCALAR  MPI_DOUBLE_PRECISION
#endif

! ----------------------------------------------------------------------------
!
!     Basic constants
!
      double precision PETSC_PI,PETSC_DEGREES_TO_RADIANS
      double precision PETSC_MAX,PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0)
      parameter (PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0)
      parameter (PETSC_MAX = 1.d300, PETSC_MIN = -1.d300)

! ----------------------------------------------------------------------------
!
!    PLogDouble variables are used to contain double precision numbers
!  that are not used in the numerical computations, but rather in logging,
!  timing etc.
!
#define PetscObject PetscFortranAddr
#define PLogDouble  double precision


