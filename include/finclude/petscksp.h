C
C  "$Id: petsc.h,v 1.18 1996/02/12 20:26:22 bsmith Exp $";
C
C  Include file for Fortran use of the KSP package in PETSc
C
#define KSP       integer
#define KSPType   integer
C
C  Various Krylov subspace methods
C
      integer KSPRICHARDSON, KSPCHEBYCHEV, KSPCG, KSPGMRES, 
     *         KSPTCQMR, KSPBCGS, KSPCGS, KSPTFQMR, KSPCR, KSPLSQR,
     *         KSPPREONLY, KSPQCG

      parameter (KSPRICHARDSON = 0, KSPCHEBYCHEV = 1, KSPCG = 2,
     *           KSPGMRES = 3,KSPTCQMR = 4, KSPBCGS = 5, KSPCGS = 6,
     *           KSPTFQMR = 7, KSPCR = 8, KSPLSQR = 9, KSPPREONLY = 10,
     *           KSPQCG = 11)
C
C  End of Fortran include file for the KSP package in PETSc

