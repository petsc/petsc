!
!  $Id: petsc.h,v 1.57 1998/03/23 23:46:47 balay Exp balay $;
!
!  Base include file for Fortran use of the PETSc package
!
#define MPI_Comm integer
!
#include "mpif.h"

#define PetscTruth    integer
#define PetscDataType integer 

!
!     Flags
!
      integer   PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE
      integer   PETSC_DEFAULT_INTEGER,PETSC_DETERMINE
      integer   PETSC_FP_TRAP_OFF, PETSC_FP_TRAP_ON
      double precision    PETSC_DEFAULT_DOUBLE_PRECISION

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT_INTEGER = -2,PETSC_DETERMINE = -1,
     *           PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)
      parameter (PETSC_FP_TRAP_OFF = 0, PETSC_FP_TRAP_ON = 1) 

!
!     Default Viewers
!
      integer   VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF,
     *          VIEWER_STDOUT_WORLD, VIEWER_DRAWX_WORLD,
     *          VIEWER_DRAWX_WORLD_0,VIEWER_DRAWX_WORLD_1,
     *          VIEWER_DRAWX_WORLD_2,VIEWER_DRAWX_SELF,
     *          VIEWER_MATLAB_WORLD
!
!     The numbers used below should match those in 
!     src/fortran/custom/zpetsc.h
!
      parameter (VIEWER_DRAWX_WORLD_0 = -4, 
     *           VIEWER_DRAWX_WORLD_1 = -5,
     *           VIEWER_DRAWX_WORLD_2 = -6, 
     *           VIEWER_DRAWX_SELF = -7,
     *           VIEWER_MATLAB_WORLD = -8, 
     *           VIEWER_DRAWX_WORLD = VIEWER_DRAWX_WORLD_0)

!
!     Fortran Null
!
      integer        PETSC_NULL
      character*(80) PETSC_NULL_CHARACTER

      integer PETSC_INT, PETSC_DOUBLE, PETSC_SHORT, PETSC_FLOAT,
     *        PETSC_COMPLEX, PETSC_CHAR, PETSC_LOGICAL

      parameter (PETSC_INT=0, PETSC_DOUBLE=1, PETSC_SHORT=2, 
     *           PETSC_FLOAT=3, PETSC_COMPLEX=4, PETSC_CHAR=5, 
     *           PETSC_LOGICAL=6)

#if defined(USE_PETSC_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif

!
! PETSc world communicator
!
      MPI_Comm PETSC_COMM_WORLD, PETSC_COMM_SELF

      common   /petscfortran/  PETSC_NULL,
     *         VIEWER_STDOUT_SELF,VIEWER_STDERR_SELF,
     *         VIEWER_STDOUT_WORLD,PETSC_NULL_CHARACTER,
     *         PETSC_COMM_WORLD,PETSC_COMM_SELF
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
!     Prototypes for functions which return a value.
!
      external PetscGetTime,PetscGetCPUTime,PetscGetFlops
      double precision PetscGetTime,PetscGetCPUTime,PetscGetFlops
!
!
!
#if defined(HAVE_64BITS)
#define PetscOffset integer*8
#define PETScAddr   integer*8
#else
#define PetscOffset integer
#define PETScAddr   integer
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
!
!     Representation of complex i
!
      double complex PETSC_i
      parameter (PETSC_i = (0,1.0))
!
!     Macro for templating between real and complex
!
#if defined(USE_PETSC_COMPLEX)
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
      double precision PETSC_PI,PETSC_DEGREES_TO_RADIANS,
     &                 PETSC_MAX,PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0,
     &           PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0,
     &           PETSC_MAX = 1.d300, PETSC_MIN = -1.d300)

! ----------------------------------------------------------------------------
!
!    PLogDouble variables are used to contain double precision numbers
!  that are not used in the numerical computations, but rather in logging,
!  timing etc.
!
#define PLogDouble double precision


