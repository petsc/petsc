C
C  $Id: petsc.h,v 1.20 1996/04/10 04:31:16 bsmith Exp curfman $;
C
C  Base include file for Fortran use of the PETSc package
C
#define MPI_Comm integer
C
#include "mpif.h"

C
C     Flags
C
      integer   PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE, PETSC_DEFAULT
      integer   FP_TRAP_OFF, FP_TRAP_ON, FP_TRAP_ALWAYS

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT = -2)
      parameter (FP_TRAP_OFF = 0, FP_TRAP_ON = 1, FP_TRAP_ALWAYS = 2)

C
C     Default Viewers
C
      integer   STDOUT_VIEWER_SELF, STDERR_VIEWER_SELF,
     *          STDOUT_VIEWER_WORLD

C
C     Random numbers
C
      integer   RANDOM_DEFAULT, RANDOM_DEFAULT_REAL,
     *          RANDOM_DEFAULT_IMAGINARY     

      parameter (RANDOM_DEFAULT=0, RANDOM_DEFAULT_REAL=1,
     *           RANDOM_DEFAULT_IMAGINARY=2)     

C
C     Fortran Null
C
      integer        PETSC_NULL
      character*(80) PETSC_NULL_CHAR

      common   /petscfortran/  PETSC_NULL,
     *         STDOUT_VIEWER_SELF,STDERR_VIEWER_SELF,
     *         STDOUT_VIEWER_WORLD,PETSC_NULL_CHAR
C
C     Macro for templating between real and complex
C
#if defined(PETSC_COMPLEX)
#if defined(PARCH_t3d)
#define Scalar  complex
#else
#define Scalar  double complex
#endif
#else
#if defined(PARCH_t3d)
#define Scalar  real
#else
#define Scalar  double precision
#endif
#endif
C
C     Macros for error checking
C
#if defined(PETSC_DEBUG)
#define SETERRA(n,s)   call MPI_Abort(MPI_COMM_WORLD,n)
#define CHKERRA(n)     if (n .ne. 0) call MPI_Abort(MPI_COMM_WORLD,n)
#else
#define SETERRA(n,s)   
#define CHKERRA(n)     
#endif
C
C     Prototypes for functions which return a value.
C
      external PetscGetTime, PetscGetFlops

C     On the Cray T3D this should be "real" not "double precision"!
#if defined(PARCH_t3d)
      real PetscGetTime, PetscGetFlops
#else
      double precision PetscGetTime, PetscGetFlops
#endif

C
C     Random number object
#define PetscRandom integer
C
C     
C     End of base Fortran include file for the PETSc package

