!
!
!  Include file for Fortran use of the MG preconditioner in PETSc
!
#if !defined (__PETSCMG_H)
#define __PETSCMG_H

#define PCMGType PetscEnum

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!
      PetscEnum PC_MG_MULTIPLICATIVE,PC_MG_ADDITIVE,PC_MG_FULL,
      PetscEnum PC_MG_KASKADE,PC_MG_CASCADE
      parameter (PC_MG_MULTIPLICATIVE=0,PC_MG_ADDITIVE=1)
      parameter (PC_MG_FULL=2,PC_MG_KASKADE=3)
      parameter (PC_MG_CASCADE=3)

!
!  Other defines
!
      PetscEnum PC_MG_V_CYCLE,PC_MG_W_CYCLE
      parameter (PC_MG_V_CYCLE=1,PC_MG_W_CYCLE=2)

      external  PCMGDEFAULTRESIDUAL
!PETSC_DEC_ATTRIBUTES(PCMGDEFAULTRESIDUAL,'_PCMGDEFAULTRESIDUAL')

!
!     End of Fortran include file for the  MG include file in PETSc

#endif
