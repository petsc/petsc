!
!
!  Include file for Fortran use of the MG preconditioner in PETSc
!
#if !defined (__PETSCMG_H)
#define __PETSCMG_H

#define MGType PetscEnum

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!
      PetscEnum MGMULTIPLICATIVE,MGADDITIVE,MGFULL,MGKASKADE,MGCASCADE
      parameter (MGMULTIPLICATIVE=0,MGADDITIVE=1,MGFULL=2,MGKASKADE=3)
      parameter (MGCASCADE=3)

!
!  Other defines
!
      PetscEnum MG_V_CYCLE,MG_W_CYCLE
      parameter (MG_V_CYCLE=1,MG_W_CYCLE=2)

      external  PCMGDEFAULTRESIDUAL
!PETSC_DEC_ATTRIBUTES(PCMGDEFAULTRESIDUAL,'_PCMGDEFAULTRESIDUAL')

!
!     End of Fortran include file for the  MG include file in PETSc

#endif
