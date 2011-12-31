!
!  Include file for Fortran use of the MG preconditioner in PETSc
!
#include "finclude/petscpcmgdef.h"

!
!
      PetscEnum PC_MG_MULTIPLICATIVE
      PetscEnum PC_MG_ADDITIVE
      PetscEnum PC_MG_FULL
      PetscEnum PC_MG_KASKADE
      PetscEnum PC_MG_CASCADE
      parameter (PC_MG_MULTIPLICATIVE=0,PC_MG_ADDITIVE=1)
      parameter (PC_MG_FULL=2,PC_MG_KASKADE=3)
      parameter (PC_MG_CASCADE=3)

! PCMGCycleType
      PetscEnum PC_MG_CYCLE_V
      PetscEnum PC_MG_CYCLE_W
      parameter (PC_MG_CYCLE_V = 1,PC_MG_CYCLE_W = 2)

      PetscEnum PC_EXOTIC_FACE
      PetscEnum PC_EXOTIC_WIREBASKET
      parameter (PC_EXOTIC_FACE=0,PC_EXOTIC_WIREBASKET=1)

      external  PCMGDEFAULTRESIDUAL

!
!     End of Fortran include file for the  MG include file in PETSc

