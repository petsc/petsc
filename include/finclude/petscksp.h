C
C  "$Id: ksp.h,v 1.11 1997/11/13 20:51:40 balay Exp bsmith $";
C
C  Include file for Fortran use of the KSP package in PETSc
C
#define KSP          integer
#define KSPCGType    integer
C
C  Various Krylov subspace methods
C
#define KSPRICHARDSON 'richardson'
#define KSPCHEBYCHEV  'chebychev'
#define KSPCG         'cg'
#define KSPGMRES      'gmres'
#define KSPTCQMR      'tcqmr'
#define KSPBCGS       'bcgs'
#define KSPCGS        'cgs'
#define KSPTFQMR      'tfqmr'
#define KSPCR         'cr'
#define KSPLSQR       'lsqr'
#define KSPPREONLY    'preonly'
#define KSPQCG        'qcg'
#define KSPTRLS       'trls'

C
C  CG Types
C
      integer KSP_CG_SYMMETRIC,KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=1, KSP_CG_HERMITIAN=2)
C
C  End of Fortran include file for the KSP package in PETSc

