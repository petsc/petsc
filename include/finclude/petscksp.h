!
!  "$Id: ksp.h,v 1.12 1998/03/06 00:21:29 bsmith Exp balay $";
!
!  Include file for Fortran use of the KSP package in PETSc
!
#define KSP          integer
#define KSPCGType    integer
!
!  Various Krylov subspace methods
!
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

!
!  CG Types
!
      integer KSP_CG_SYMMETRIC,KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=1, KSP_CG_HERMITIAN=2)
!
!  End of Fortran include file for the KSP package in PETSc

