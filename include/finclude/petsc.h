C
C  $Id: petsc.h,v 1.30 1996/08/27 20:29:05 curfman Exp curfman $;
C
C  Base include file for Fortran use of the PETSc package
C
#define MPI_Comm integer
C
#include "mpif.h"

#define PetscTruth integer

C
C     real on the Cray T3d is actually double precision
C
#if defined(PARCH_t3d)
#define Double real
#else
#define Double double precision
#endif
C
C     Flags
C
      integer   PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE
      integer   PETSC_DEFAULT_INTEGER
      integer   PETSC_FP_TRAP_OFF, PETSC_FP_TRAP_ON
      Double    PETSC_DEFAULT_DOUBLE_PRECISION

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT_INTEGER = -2,
     *           PETSC_DEFAULT_DOUBLE_PRECISION = -2.0d0)
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
      character*(80) PETSC_NULL_CHARACTER

C
C PETSc world communicator
C
      MPI_Comm PETSC_COMM_WORLD

      common   /petscfortran/  PETSC_NULL,
     *         VIEWER_STDOUT_SELF,VIEWER_STDERR_SELF,
     *         VIEWER_STDOUT_WORLD,PETSC_NULL_CHARACTER,
     *         PETSC_COMM_WORLD
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
#define SETERRA(n,s)   call MPI_Abort(PETSC_COMM_WORLD,n)
#define CHKERRA(n)     if (n .ne. 0) call MPI_Abort(PETSC_COMM_WORLD,n)
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

