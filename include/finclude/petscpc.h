C
C  $Id: pc.h,v 1.16 1996/04/16 04:56:29 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the PC (preconditioner) package in PETSc
C
#define PC           integer
#define PCType       integer
#define MatStructure integer
#define PCNullSpace  integer
#define PCSide       integer
#define PCBGSType    integer
C
C  Various preconditioners
C
      integer PCNONE, PCJACOBI, PCSOR, PCLU, PCSHELL, PCBJACOBI, PCMG,
     *        PCEISENSTAT, PCILU, PCICC, PCASM, PCBGS, PCSPAI

      parameter (PCNONE = 0, PCJACOBI = 1, PCSOR = 2, PCLU = 3, 
     *           PCSHELL = 4, PCBJACOBI = 5, PCMG = 6,
     *           PCEISENSTAT = 7, PCILU = 8, PCICC = 9, PCASM = 10,
     *           PCBGS = 11, PCSPAI = 12)

C
C  PCSide
C
      integer PC_LEFT, PC_RIGHT, PC_SYMMETRIC 

      parameter (PC_LEFT=0, PC_RIGHT=1, PC_SYMMETRIC=2)

C
C  Flags for PCSetOperators()
C
      integer SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,
     *        SAME_PRECONDITIONER

      parameter (SAME_NONZERO_PATTERN = 0,
     *           DIFFERENT_NONZERO_PATTERN = 1,
     *           SAME_PRECONDITIONER = 2)


C
C  PCBGSType
C
      integer PCBGS_FORWARD_SWEEP,PCBGS_SYMMETRIC_SWEEP
      
      parameter (PCBGS_FORWARD_SWEEP=1,PCBGS_SYMMETRIC_SWEEP=2)

      integer USE_PRECONDITIONER_MATRIX, USE_TRUE_MATRIX
      parameter (USE_PRECONDITIONER_MATRIX=0, USE_TRUE_MATRIX=1)


C
C  End of Fortran include file for the PC package in PETSc

