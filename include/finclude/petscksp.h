C
C  "$Id: ksp.h,v 1.7 1996/02/12 20:26:59 bsmith Exp balay $";
C
C  Include file for Fortran use of the KSP package in PETSc
C
#define KSP       integer
#define KSPType   integer
#define CGType    integer
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
C  CG Types
C
      integer CG_SYMMETRIC,CG_HERMITIAN

      parameter (CG_SYMMETRIC=1, CG_HERMITIAN=2)
C
C  End of Fortran include file for the KSP package in PETSc

