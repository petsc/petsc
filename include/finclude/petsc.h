C
C      Base include file for Fortran use of the PETSc package
C
#define MPI_Comm integer
C
#include "mpif.h"

      integer PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE, PETSC_DEFAULT
      integer FP_TRAP_OFF, FP_TRAP_ON, FP_TRAP_ALWAYS

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT = -2)
      parameter (FP_TRAP_OFF = 0, FP_TRAP_ON = 1, FP_TRAP_ALWAYS = 2)

      integer STDOUT_VIEWER_SELF,STDERR_VIEWER_SELF,
     *        STDOUT_VIEWER_WORLD
      common /fortranview/ STDOUT_VIEWER_SELF,STDERR_VIEWER_SELF,
     *                     STDOUT_VIEWER_WORLD

C #if defined(PETSC_DEBUG)
C #define SETERRQ(n,s)     call PetscError(s,n)
C #define SETERRA(n,s)   {int _ierr = PetscError(s,n)
C                         call MPI_Abort(MPI_COMM_WORLD,_ierr);}
C #define CHKERRQ(n)       if (n .ne. 0) call SETERRQ(n,0)
C #define CHKERRA(n)       if (n .ne. 0) call SETERRA(n,0)
C #endif
C
C      End of base Fortran include file for the PETSc package

