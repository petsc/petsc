!
!  $Id: petscdef.h,v 1.1 1998/07/10 20:17:15 bsmith Exp bsmith $;
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

#if (SIZEOF_VOIDP == 8)
#define PetscOffset        integer*8
#define PetscFortranAddr   integer*8
#else
#define PetscOffset        integer
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
! PETSc world communicator
!
      MPI_Comm PETSC_COMM_WORLD, PETSC_COMM_SELF

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
! ------------------------------------------------------------------------
!     PETSc mathematics include file. Defines certain basic mathematical 
!    constants and functions for working with single and double precision
!    floating point numbers as well as complex and integers.
!
!
!     Representation of complex i
!
#if defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
      complex (KIND=SELECTED_REAL_KIND(14)) PETSC_i
#else
      double complex PETSC_i
#endif
      parameter (PETSC_i = (0.0d0,1.0d0))
!
!     Macro for templating between real and complex
!
#if defined(USE_PETSC_COMPLEX)
#if defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
#define Scalar       complex (KIND=SELECTED_REAL_KIND(14))
#else
#define Scalar       double complex
#endif
!
! F90 uses real(), conjg() when KIND parameter is used.
!
#if defined (HAVE_NAGF90) || defined (HAVE_IRIXF90)
#define PetscReal(a) real(a)
#define PetscConj(a) conjg(a)
#else
#define PetscReal(a) dreal(a)
#define PetscConj(a) dconjg(a)
#endif
#define MPIU_SCALAR   MPI_DOUBLE_COMPLEX
#else
#define PetscReal(a) a
#define PetscConj(a) a
#define Scalar       double precision
#define MPIU_SCALAR  MPI_DOUBLE_PRECISION
#endif

!
!     Basic constants
! 
      double precision PETSC_PI,PETSC_DEGREES_TO_RADIANS
      double precision PETSC_MAX,PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0)
      parameter (PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0)
      parameter (PETSC_MAX = 1.d300, PETSC_MIN = -1.d300)

!
!     Fortran Null
!
      character*(80)   PETSC_NULL_CHARACTER
      integer          PETSC_NULL_INTEGER
      double precision PETSC_NULL_DOUBLE
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
#define PLogDouble  double precision



! ----------------------------------------------------------------------------
