C
C  "$Id: ksp.h,v 1.8 1996/04/15 23:36:29 balay Exp bsmith $";
C
C  Include file for Fortran use of the KSP package in PETSc
C
#define KSP          integer
#define KSPType      integer
#define KSPCGType    integer
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
      integer KSP_CG_SYMMETRIC,KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=1, KSP_CG_HERMITIAN=2)
C
C  End of Fortran include file for the KSP package in PETSc

