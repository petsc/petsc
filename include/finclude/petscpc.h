C
C  Include file for Fortran use of the PC (preconditioner) package in PETSc
C
#define PC           integer
#define PCType       integer
#define MatStructure integer
C
C  Various preconditioners
C
      integer PCNONE, PCJACOBI, PCSOR, PCLU, PCSHELL, PCBJACOBI, PCMG,
     *        PCEISENSTAT, PCILU, PCICC, PCSPAI

      parameter (PCNONE = 0, PCJACOBI = 1, PCSOR = 2, PCLU = 3, 
     *           PCSHELL = 4, PCBJACOBI = 5, PCMG = 6,
     *           PCEISENSTAT = 7, PCILU = 8, PCICC = 9, PCSPAI = 10)
C
C  Flags for PCSetOperators()
C
      integer SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER

      parameter (SAME_NONZERO_PATTERN = 0,
     *           DIFFERENT_NONZERO_PATTERN = 1,
     *           SAME_PRECONDITIONER = 2)
C
C  End of Fortran include file for the PC package in PETSc

