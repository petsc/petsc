!
!  "$Id: ksp.h,v 1.22 1999/11/24 21:56:02 bsmith Exp bsmith $";
!
!  Include file for Fortran use of the KSP package in PETSc
!
#if !defined (__KSP_H)
#define __KSP_H

#define KSP                PetscFortranAddr
#define KSPCGType          integer
#define KSPType            character*(80)
#define KSPConvergedReason integer 
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


#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  CG Types
!
      integer KSP_CG_SYMMETRIC,KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=1, KSP_CG_HERMITIAN=2)

      integer KSP_CONVERGED_RTOL,KSP_CONVERGED_ATOL
      integer KSP_DIVERGED_ITS,KSP_DIVERGED_DTOL
      integer KSP_DIVERGED_BREAKDOWN,KSP_CONVERGED_ITERATING

      parameter (KSP_CONVERGED_RTOL      = 2)
      parameter (KSP_CONVERGED_ATOL      = 3)

      parameter (KSP_DIVERGED_ITS        = -3)
      parameter (KSP_DIVERGED_DTOL       = -4)
      parameter (KSP_DIVERGED_BREAKDOWN  = -5)

      parameter (KSP_CONVERGED_ITERATING = 0)
!
!
!   Possible arguments to KSPSetMonitor()
!
      external KSPDEFAULTCONVERGED

      external KSPDEFAULTMONITOR
      external KSPTRUEMONITOR
      external KSPLGMONITOR
      external KSPLGTRUEMONITOR
      external KSPVECVIEWMONITOR
      external KSPSINGULARVALUEMONITOR

!  End of Fortran include file for the KSP package in PETSc

#endif
