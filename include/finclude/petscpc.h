!
!  $Id: pc.h,v 1.25 1998/03/27 21:17:55 balay Exp balay $;
!
!  Include file for Fortran use of the PC (preconditioner) package in PETSc
!
#define PC           PetscFortranAddr
#define PCNullSpace  PetscFortranAddr
#define PCSide       integer
#define PCBGSType    integer
#define PCASMType    integer
!
!  Various preconditioners
!
#define PCNONE      'none'
#define PCJACOBI    'jacobi'
#define PCSOR       'sor'
#define PCLU        'lu'
#define PCSHELL     'shell'
#define PCBJACOBI   'bjacobi'
#define PCMG        'mg'
#define PCEISENSTAT 'eisenstat'
#define PCILU       'ilu'
#define PCICC       'icc'
#define PCASM       'asm'
#define PCBGS       'bgs'
#define PCSLES      'sles'
#define PCCOMPOSITE 'composite'

!
!  PCSide
!
      integer PC_LEFT, PC_RIGHT, PC_SYMMETRIC 

      parameter (PC_LEFT=0, PC_RIGHT=1, PC_SYMMETRIC=2)


!
!  PCBGSType
!
      integer PCBGS_FORWARD_SWEEP,PCBGS_SYMMETRIC_SWEEP
      
      parameter (PCBGS_FORWARD_SWEEP=1,PCBGS_SYMMETRIC_SWEEP=2)

      integer USE_PRECONDITIONER_MATRIX, USE_TRUE_MATRIX
      parameter (USE_PRECONDITIONER_MATRIX=0, USE_TRUE_MATRIX=1)

!
! PCASMType
!
      integer PC_ASM_BASIC, PC_ASM_RESTRICT, PC_ASM_INTERPOLATE
      integer PC_ASM_NONE

      parameter (PC_ASM_BASIC = 3, PC_ASM_RESTRICT = 1)
      parameter (PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0)
!
!  End of Fortran include file for the PC package in PETSc

