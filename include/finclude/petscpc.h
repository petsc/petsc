!
!  $Id: petscpc.h,v 1.34 2000/08/01 20:58:48 bsmith Exp balay $;
!
!  Include file for Fortran use of the PC (preconditioner) package in PETSc
!
#if !defined (__PETSCPC_H)
#define __PETSCPC_H

#define PC PetscFortranAddr
#define MatNullSpace PetscFortranAddr
#define PCSide integer
#define PCASMType integer
#define PCCompositeType integer
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
#define PCSLES 'sles'
#define PCCOMPOSITE 'composite'
#define PCREDUNDANT 'redundant'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  PCSide
!
      integer PC_LEFT,PC_RIGHT,PC_SYMMETRIC 
      parameter (PC_LEFT=0,PC_RIGHT=1,PC_SYMMETRIC=2)

      integer USE_PRECONDITIONER_MATRIX,USE_TRUE_MATRIX
      parameter (USE_PRECONDITIONER_MATRIX=0,USE_TRUE_MATRIX=1)

!
! PCASMType
!
      integer PC_ASM_BASIC,PC_ASM_RESTRICT,PC_ASM_INTERPOLATE
      integer PC_ASM_NONE

      parameter (PC_ASM_BASIC = 3,PC_ASM_RESTRICT = 1)
      parameter (PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0)
!
! PCCompositeType
!
      integer PC_COMPOSITE_ADDITIVE,PC_COMPOSITE_MULTIPLICATIVE
      parameter (PC_COMPOSITE_ADDITIVE=0,PC_COMPOSITE_MULTIPLICATIVE=1)
!
!  End of Fortran include file for the PC package in PETSc

#endif
