C
C  $Id: pc.h,v 1.20 1996/09/28 13:46:45 bsmith Exp balay $;
C
C  Include file for Fortran use of the PC (preconditioner) package in PETSc
C
#define PC           integer
#define PCType       integer
#define PCNullSpace  integer
#define PCSide       integer
#define PCBGSType    integer
#define PCASMType    integer
C
C  Various preconditioners
C
      integer PCNONE, PCJACOBI, PCSOR, PCLU, PCSHELL, PCBJACOBI, PCMG,
     *        PCEISENSTAT, PCILU, PCICC, PCASM, PCBGS, PCNEW

      parameter (PCNONE = 0, PCJACOBI = 1, PCSOR = 2, PCLU = 3, 
     *           PCSHELL = 4, PCBJACOBI = 5, PCMG = 6,
     *           PCEISENSTAT = 7, PCILU = 8, PCICC = 9, PCASM = 10,
     *           PCBGS = 11, PCNEW = 12)

C
C  PCSide
C
      integer PC_LEFT, PC_RIGHT, PC_SYMMETRIC 

      parameter (PC_LEFT=0, PC_RIGHT=1, PC_SYMMETRIC=2)


C
C  PCBGSType
C
      integer PCBGS_FORWARD_SWEEP,PCBGS_SYMMETRIC_SWEEP
      
      parameter (PCBGS_FORWARD_SWEEP=1,PCBGS_SYMMETRIC_SWEEP=2)

      integer USE_PRECONDITIONER_MATRIX, USE_TRUE_MATRIX
      parameter (USE_PRECONDITIONER_MATRIX=0, USE_TRUE_MATRIX=1)

C
C PCASMType
C
      integer PC_ASM_BASIC, PC_ASM_RESTRICT, PC_ASM_INTERPOLATE,
     *        PC_ASM_NONE 
      parameter (PC_ASM_BASIC = 3, PC_ASM_RESTRICT = 1, 
     *           PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0)
C
C  End of Fortran include file for the PC package in PETSc

