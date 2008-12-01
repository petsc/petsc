!
!  Include file for Fortran use of the MG preconditioner in PETSc
!
#include "finclude/petscmgdef.h"

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

!
!  Other defines
!
      PetscEnum PC_MG_V_CYCLE
      PetscEnum PC_MG_W_CYCLE
      parameter (PC_MG_V_CYCLE=1,PC_MG_W_CYCLE=2)

      PetscEnum PC_EXOTIC_FACE
      PetscEnum PC_EXOTIC_WIREBASKET
      parameter (PC_EXOTIC_FACE=0,PC_EXOTIC_WIREBASKET=1)

      external  PCMGDEFAULTRESIDUAL

!PETSC_DEC_ATTRIBUTES(PCMGDEFAULTRESIDUAL,'_PCMGDEFAULTRESIDUAL')

!
!     End of Fortran include file for the  MG include file in PETSc

