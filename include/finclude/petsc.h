C
C  $Id: petsc.h,v 1.49 1997/09/15 16:27:42 bsmith Exp bsmith $;
C
C  Base include file for Fortran use of the PETSc package
C
#define MPI_Comm integer
C
#include "mpif.h"

#define PetscTruth integer

C
C     Flags
C
      integer   PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE
      integer   PETSC_DEFAULT_INTEGER
      integer   PETSC_FP_TRAP_OFF, PETSC_FP_TRAP_ON
      double precision    PETSC_DEFAULT_DOUBLE_PRECISION

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT_INTEGER = -2,
     *           PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)
      parameter (PETSC_FP_TRAP_OFF = 0, PETSC_FP_TRAP_ON = 1) 

C
C     Default Viewers
C
      integer   VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF,
     *          VIEWER_STDOUT_WORLD, VIEWER_DRAWX_WORLD,
     *          VIEWER_DRAWX_WORLD_0,VIEWER_DRAWX_WORLD_1,
     *          VIEWER_DRAWX_WORLD_2,VIEWER_DRAWX_SELF,
     *          VIEWER_MATLAB_WORLD
C
C     The numbers used below should match those in 
C     src/fortran/custom/zpetsc.h
C
      parameter (VIEWER_DRAWX_WORLD_0 = -4, 
     *           VIEWER_DRAWX_WORLD_1 = -5,
     *           VIEWER_DRAWX_WORLD_2 = -6, 
     *           VIEWER_DRAWX_SELF = -7,
     *           VIEWER_MATLAB_WORLD = -8, 
     *           VIEWER_DRAWX_WORLD = VIEWER_DRAWX_WORLD_0)

C
C     Fortran Null
C
      integer        PETSC_NULL
      character*(80) PETSC_NULL_CHARACTER

C
C PETSc world communicator
C
      MPI_Comm PETSC_COMM_WORLD, PETSC_COMM_SELF

      common   /petscfortran/  PETSC_NULL,
     *         VIEWER_STDOUT_SELF,VIEWER_STDERR_SELF,
     *         VIEWER_STDOUT_WORLD,PETSC_NULL_CHARACTER,
     *         PETSC_COMM_WORLD,PETSC_COMM_SELF
C
C     Macros for error checking
C
#if defined(PETSC_DEBUG)
#define SETERRA(n,p,s) call MPI_Abort(PETSC_COMM_WORLD,n)
#define CHKERRA(n) if (n .ne. 0) call MPI_Abort(PETSC_COMM_WORLD,n)
#else
#define SETERRA(n,p,s)   
#define CHKERRA(n)     
#endif
C
C     Prototypes for functions which return a value.
C
      external PetscGetTime,PetscGetCPUTime,PetscGetFlops
      double precision PetscGetTime,PetscGetCPUTime,PetscGetFlops
C
C
C     End of base Fortran include file for the PETSc package
C
C ------------------------------------------------------------------------
C     PETSc mathematics include file. Defines certain basic mathematical 
C    constants and functions for working with single and double precision
C    floating point numbers as well as complex and integers.
C
C
C     Macro for templating between real and complex
C
#if defined(PETSC_COMPLEX)
#define PetscReal(a)  real(a)
#define PetscConj(a)  dconjg(a)
#define Scalar        double complex
#define DoubleComplex double complex
#define MPIU_SCALAR   MPI_DOUBLE_COMPLEX
C
C     Representation of complex i
C
      DoubleComplex PETSC_i
      parameter (PETSC_i = (0,1.0))
#else
#define PetscReal(a) a
#define PetscConj(a) a
#define Scalar       double precision
#define MPIU_SCALAR  MPI_DOUBLE_PRECISION
C
C     This is to allow compiling complex codes with real numbers
C
      double precision PETSC_i
      parameter (PETSC_i = 0.d0)
#endif

C ----------------------------------------------------------------------------
C
C     Basic constants
C
      double precision PETSC_PI,PETSC_DEGREES_TO_RADIANS,
     &                 PETSC_MAX,PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0,
     &           PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0,
     &           PETSC_MAX = 1.d300, PETSC_MIN = -1.d300)

C ----------------------------------------------------------------------------
C
C    PLogDouble variables are used to contain double precision numbers
C  that are not used in the numerical computations, but rather in logging,
C  timing etc.
C
#define PLogDouble double precision


