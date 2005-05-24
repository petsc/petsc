!
!
!  Include file for Fortran use of the PC (preconditioner) package in PETSc
!
#if !defined (__PETSCPC_H)
#define __PETSCPC_H

#define PC PetscFortranAddr
#define PCSide PetscEnum
#define PCASMType PetscEnum
#define PCCompositeType PetscEnum
#define PCType character*(80)
!
!  Various preconditioners
!
#define PCNONE 'none'
#define PCJACOBI 'jacobi'
#define PCSOR 'sor'
#define PCLU 'lu'
#define PCSHELL 'shell'
#define PCBJACOBI 'bjacobi'
#define PCMG 'mg'
#define PCEISENSTAT 'eisenstat'
#define PCILU 'ilu'
#define PCICC 'icc'
#define PCASM 'asm'
#define PCKSP 'ksp'
#define PCCOMPOSITE 'composite'
#define PCREDUNDANT 'redundant'
#define PCSPAI 'spai'
#define PCMILU 'milu'
#define PCNN 'nn'
#define PCCHOLESKY 'cholesky'
#define PCSAMG 'samg'
#define PCPBJACOBI 'pbjacobi'
#define PCMAT 'mat'
#define PCHYPRE 'hypre'
#define PCFIELDSPLIT 'fieldsplit'
#define PCML 'ml'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  PCSide
!
      PetscEnum PC_LEFT,PC_RIGHT,PC_SYMMETRIC 
      parameter (PC_LEFT=0,PC_RIGHT=1,PC_SYMMETRIC=2)

      PetscEnum USE_PRECONDITIONER_MATRIX,USE_TRUE_MATRIX
      parameter (USE_PRECONDITIONER_MATRIX=0,USE_TRUE_MATRIX=1)

!
! PCASMType
!
      PetscEnum PC_ASM_BASIC,PC_ASM_RESTRICT,PC_ASM_INTERPOLATE
      PetscEnum PC_ASM_NONE

      parameter (PC_ASM_BASIC = 3,PC_ASM_RESTRICT = 1)
      parameter (PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0)
!
! PCCompositeType
!
      PetscEnum PC_COMPOSITE_ADDITIVE,PC_COMPOSITE_MULTIPLICATIVE
      PetscEnum PC_COMPOSITE_SPECIAL
      parameter (PC_COMPOSITE_ADDITIVE=0,PC_COMPOSITE_MULTIPLICATIVE=1)
      parameter (PC_COMPOSITE_SPECIAL=2)
!
!  End of Fortran include file for the PC package in PETSc

#endif
