C
C  $Id: petsc.h,v 1.27 1996/07/22 15:32:19 bsmith Exp bsmith $;
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
      integer   PETSC_FP_TRAP_OFF, PETSC_FP_TRAP_ON

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT = -2)
      parameter (PETSC_FP_TRAP_OFF = 0, PETSC_FP_TRAP_ON = 1) 

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
C     real on the Cray T3d is actually double precision
C
#if defined(PARCH_t3d)
#define Double real
#else
#define Double double precision
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
      Double PetscGetTime, PetscGetFlops
C     
C     End of base Fortran include file for the PETSc package

