!
!  "$Id: petscksp.h,v 1.32 2001/04/10 22:37:56 balay Exp $";
!
!  Include file for Fortran use of the KSP package in PETSc
!
#if !defined (__PETSCKSP_H)
#define __PETSCKSP_H

#define KSP PetscFortranAddr
#define KSPType character*(80)
#define KSPCGType integer
#define KSPConvergedReason integer 
#define KSPNormType integer
!
!  Various Krylov subspace methods
!
#define KSPRICHARDSON 'richardson'
#define KSPCHEBYCHEV 'chebychev'
#define KSPCG 'cg'
#define KSPGMRES 'gmres'
#define KSPTCQMR 'tcqmr'
#define KSPBCGS 'bcgs'
#define KSPCGS 'cgs'
#define KSPTFQMR 'tfqmr'
#define KSPCR 'cr'
#define KSPLSQR 'lsqr'
#define KSPPREONLY 'preonly'
#define KSPQCG 'qcg'
#define KSPBICG 'bicg'
#define KSPFGMRES 'fgmres'
#define KSPMINRES 'minres'
#define KSPSYMMLQ 'symmlq'
#endif


#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  CG Types
!
      integer KSP_CG_SYMMETRIC,KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=1,KSP_CG_HERMITIAN=2)

      integer KSP_CONVERGED_RTOL,KSP_CONVERGED_ATOL
      integer KSP_CONVERGED_ITS
      integer KSP_DIVERGED_ITS,KSP_DIVERGED_DTOL
      integer KSP_DIVERGED_BREAKDOWN,KSP_CONVERGED_ITERATING
      integer KSP_CONVERGED_QCG_NEG_CURVE
      integer KSP_CONVERGED_QCG_CONSTRAINED
      integer KSP_CONVERGED_STEP_LENGTH
      integer KSP_DIVERGED_BREAKDOWN_BICG
      integer KSP_DIVERGED_NONSYMMETRIC
      integer KSP_DIVERGED_INDEFINITE_PC

      parameter (KSP_CONVERGED_RTOL      = 2)
      parameter (KSP_CONVERGED_ATOL      = 3)
      parameter (KSP_CONVERGED_ITS       = 4)
      parameter (KSP_CONVERGED_QCG_NEG_CURVE = 5)
      parameter (KSP_CONVERGED_QCG_CONSTRAINED = 6)
      parameter (KSP_CONVERGED_STEP_LENGTH = 7)

      parameter (KSP_DIVERGED_ITS        = -3)
      parameter (KSP_DIVERGED_DTOL       = -4)
      parameter (KSP_DIVERGED_BREAKDOWN  = -5)
      parameter (KSP_DIVERGED_BREAKDOWN_BICG = -6)
      parameter (KSP_DIVERGED_NONSYMMETRIC = -7)
      parameter (KSP_DIVERGED_INDEFINITE_PC = -8)

      parameter (KSP_CONVERGED_ITERATING = 0)
!
!  Possible arguments to KSPSetNormType()
!
      integer KSP_NO_NORM
      integer KSP_PRECONDITIONED_NORM
      integer KSP_UNPRECONDITIONED_NORM
      integer KSP_NATURAL_NORM 
      
      parameter (KSP_NO_NORM=0)
      parameter (KSP_PRECONDITIONED_NORM=1)
      parameter (KSP_UNPRECONDITIONED_NORM=2)
      parameter (KSP_NATURAL_NORM=3) 
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
      external KSPGMRESKRYLOVMONITOR

!PETSC_DEC_ATTRIBUTES(KSPDEFAULTCONVERGED,'_KSPDEFAULTCONVERGED')
!PETSC_DEC_ATTRIBUTES(KSPDEFAULTMONITOR,'_KSPDEFAULTMONITOR')
!PETSC_DEC_ATTRIBUTES(KSPTRUEMONITOR,'_KSPTRUEMONITOR')
!PETSC_DEC_ATTRIBUTES(KSPLGMONITOR,'_KSPLGMONITOR')
!PETSC_DEC_ATTRIBUTES(KSPLGTRUEMONITOR,'_KSPLGTRUEMONITOR')
!PETSC_DEC_ATTRIBUTES(KSPVECVIEWMONITOR,'_KSPVECVIEWMONITOR')
!PETSC_DEC_ATTRIBUTES(KSPSINGULARVALUEMONITOR,'_KSPSINGULARVALUEMONITOR')
!PETSC_DEC_ATTRIBUTES(KSPGMRESKRYLOVMONITOR,'_KSPGMRESKRYLOVMONITOR')

!
!  End of Fortran include file for the KSP package in PETSc
!

#endif

