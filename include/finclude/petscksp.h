!
!  "$Id: ksp.h,v 1.18 1999/03/24 18:07:05 balay Exp balay $";
!
!  Include file for Fortran use of the KSP package in PETSc
!
#if !defined (__KSP_H)
#define __KSP_H

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
#define KSPBICG       'bicg'

#endif
!
!  CG Types
!
      integer KSP_CG_SYMMETRIC,KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=1, KSP_CG_HERMITIAN=2)
!
!  End of Fortran include file for the KSP package in PETSc



