
C      Include file for for Fortran use of the PETSc package
C
#include "mpif.h"

      integer PETSC_TRUE, PETSC_FALSE, PETSC_DECIDE, PETSC_DEFAULT
      integer FP_TRAP_OFF, FP_TRAP_ON, FP_TRAP_ALWAYS

      parameter (PETSC_TRUE = 1, PETSC_FALSE = 0, PETSC_DECIDE = -1,
     *           PETSC_DEFAULT = -2)
      parameter (FP_TRAP_OFF = 0, FP_TRAP_ON = 1, FP_TRAP_ALWAYS = 2)

      integer STDOUT_VIEWER_SELF,STDERR_VIEWER_SELF,
     *        STDOUT_VIEWER_WORLD
      common /fortransucks/ STDOUT_VIEWER_SELF,STDERR_VIEWER_SELF,
     *                      STDOUT_VIEWER_WORLD
C
C      End of Fortran include file for the PETSc package

