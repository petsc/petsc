C
C      Base include file for Fortran use of the PETSc package
C
#define MPI_Comm integer
C
#include "mpif.h"

C  Various flags
      integer   PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE, PETSC_DEFAULT
      integer   FP_TRAP_OFF, FP_TRAP_ON, FP_TRAP_ALWAYS

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT = -2)
      parameter (FP_TRAP_OFF = 0, FP_TRAP_ON = 1, FP_TRAP_ALWAYS = 2)

C  Default Viewers
      integer   STDOUT_VIEWER_SELF, STDERR_VIEWER_SELF,
     *          STDOUT_VIEWER_WORLD

C  Miscellaneous
      integer   PetscInt(1), PetscNull
      double precision PetscDouble(1)

      common   /fortranview/  PetscDouble, PetscInt, PetscNull,
     *         STDOUT_VIEWER_SELF,STDERR_VIEWER_SELF,STDOUT_VIEWER_WORLD


C  Macros for error checking
#if defined(PETSC_DEBUG)
#define SETERRA(n,s)   call MPI_Abort(MPI_COMM_WORLD,n)
#define CHKERRA(n)     if (n .ne. 0) call MPI_Abort(MPI_COMM_WORLD,n)
#else
#define SETERRA(n,s)   
#define CHKERRA(n)     
#endif
C
C      End of base Fortran include file for the PETSc package

