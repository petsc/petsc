!
!  "$Id: ksp.h,v 1.16 1998/12/03 04:07:13 bsmith Exp bsmith $";
!
!  Include file for Fortran use of the KSP package in PETSc
!
#define KSP          PetscFortranAddr
#define KSPCGType    integer
#define KSPType      character*(80)
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



