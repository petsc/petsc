
C      Include file for for Fortran use of the KSP package
C
      integer KSPRICHARDSON, KSPCHEBYCHEV, KSPCG, KSPGMRES, 
     *         KSPTCQMR, KSPBCGS, KSPCGS, KSPTFQMR, KSPCR, KSPLSQR,
     *         KSPPREONLY, KSPQCG

      parameter (KSPRICHARDSON = 0, KSPCHEBYCHEV = 1, KSPCG = 2,
     *           KSPGMRES = 3,KSPTCQMR = 4, KSPBCGS = 5, KSPCGS = 6,
     *           KSPTFQMR = 7, KSPCR = 8, KSPLSQR = 9, KSPPREONLY = 10,
     *           KSPQCG = 11)
C
C      End of Fortran include file for the KSP package

