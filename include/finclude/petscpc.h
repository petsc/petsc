C
C      Include file for Fortran use of the PC package in PETSc
C
#define PC           integer
#define PCMethod     integer
#define MatStructure integer
C
      integer PCNONE, PCJACOBI, PCSOR, PCLU, PCSHELL, PCBJACOBI, PCMG,
     *        PCEISENSTAT, PCILU, PCICC, PCSPAI

      parameter (PCNONE = 0, PCJACOBI = 1, PCSOR = 2, PCLU = 3, 
     *           PCSHELL = 4, PCBJACOBI = 5, PCMG = 6,
     *           PCEISENSTAT = 7, PCILU = 8, PCICC = 9, PCSPAI = 10)

      integer ALLMAT_DIFFERENT_NONZERO_PATTERN,MAT_SAME_NONZERO_PATTERN, 
     *        PMAT_SAME_NONZERO_PATTERN,ALLMAT_SAME_NONZERO_PATTERN

      parameter (ALLMAT_DIFFERENT_NONZERO_PATTERN = 0,
     *           MAT_SAME_NONZERO_PATTERN = 1,
     *           PMAT_SAME_NONZERO_PATTERN = 2,
     *           ALLMAT_SAME_NONZERO_PATTERN = 3)
C
C      End of Fortran include file for the PC package in PETSc

