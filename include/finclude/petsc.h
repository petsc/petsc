C
C  $Id: petsc.h,v 1.23 1996/04/23 21:52:02 balay Exp bsmith $;
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
      integer   PETSC_FP_TRAP_OFF, PETSC_FP_TRAP_ON, 
      integer   PETSC_FP_TRAP_ALWAYS

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT = -2)
      parameter (PETSC_FP_TRAP_OFF = 0, PETSC_FP_TRAP_ON = 1, 
     *           PETSC_FP_TRAP_ALWAYS = 2)

C
C     Default Viewers
C
      integer   VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF,
     *          VIEWER_STDOUT_WORLD

C
C     Fortran Null
C
      integer        PETSC_NULL
      character*(80) PETSC_NULL_CHAR

      common   /petscfortran/  PETSC_NULL,
     *         VIEWER_STDOUT_SELF,VIEWER_STDERR_SELF,
     *         VIEWER_STDOUT_WORLD,PETSC_NULL_CHAR
C
C     Macro for templating between real and complex
C
#if defined(PETSC_COMPLEX)
#define PetscReal(a) real(a)
#if defined(PARCH_t3d)
#define Scalar  complex
#else
#define Scalar  double complex
#endif
#else
#define PetscReal(a) a
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
C     End of base Fortran include file for the PETSc package

