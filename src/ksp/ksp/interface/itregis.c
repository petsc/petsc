#include <petsc/private/kspimpl.h>  /*I "petscksp.h" I*/

PETSC_EXTERN PetscErrorCode KSPCreate_Richardson(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_Chebyshev(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GROPPCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECGRR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPELCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEPRCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECG2(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CGNE(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_NASH(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_STCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GLTR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_TCQMR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEFCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_BCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_IBCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_QMRCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEBCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGSR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_BCGSL(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_TFQMR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_LSQR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PREONLY(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_QCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_BiCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FGMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEFGMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_MINRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_SYMMLQ(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_LGMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_LCD(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GCR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEGCR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PGMRES(KSP);
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode KSPCreate_DGMRES(KSP);
#endif
PETSC_EXTERN PetscErrorCode KSPCreate_TSIRM(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CGLS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FETIDP(KSP);
#if defined(PETSC_HAVE_HPDDM)
PETSC_EXTERN PetscErrorCode KSPCreate_HPDDM(KSP);
#endif

/*@C
  KSPRegisterAll - Registers all of the Krylov subspace methods in the KSP package.

  Not Collective

  Level: advanced
@*/
PetscErrorCode KSPRegisterAll(void)
{
  PetscFunctionBegin;
  if (KSPRegisterAllCalled) PetscFunctionReturn(0);
  KSPRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(KSPRegister(KSPCG,          KSPCreate_CG));
  CHKERRQ(KSPRegister(KSPGROPPCG,     KSPCreate_GROPPCG));
  CHKERRQ(KSPRegister(KSPPIPECG,      KSPCreate_PIPECG));
  CHKERRQ(KSPRegister(KSPPIPECGRR,    KSPCreate_PIPECGRR));
  CHKERRQ(KSPRegister(KSPPIPELCG,     KSPCreate_PIPELCG));
  CHKERRQ(KSPRegister(KSPPIPEPRCG,    KSPCreate_PIPEPRCG));
  CHKERRQ(KSPRegister(KSPPIPECG2,     KSPCreate_PIPECG2));
  CHKERRQ(KSPRegister(KSPCGNE,        KSPCreate_CGNE));
  CHKERRQ(KSPRegister(KSPNASH,        KSPCreate_NASH));
  CHKERRQ(KSPRegister(KSPSTCG,        KSPCreate_STCG));
  CHKERRQ(KSPRegister(KSPGLTR,        KSPCreate_GLTR));
  CHKERRQ(KSPRegister(KSPRICHARDSON,  KSPCreate_Richardson));
  CHKERRQ(KSPRegister(KSPCHEBYSHEV,   KSPCreate_Chebyshev));
  CHKERRQ(KSPRegister(KSPGMRES,       KSPCreate_GMRES));
  CHKERRQ(KSPRegister(KSPTCQMR,       KSPCreate_TCQMR));
  CHKERRQ(KSPRegister(KSPFCG  ,       KSPCreate_FCG));
  CHKERRQ(KSPRegister(KSPPIPEFCG,     KSPCreate_PIPEFCG));
  CHKERRQ(KSPRegister(KSPBCGS,        KSPCreate_BCGS));
  CHKERRQ(KSPRegister(KSPIBCGS,       KSPCreate_IBCGS));
  CHKERRQ(KSPRegister(KSPQMRCGS,      KSPCreate_QMRCGS));
  CHKERRQ(KSPRegister(KSPFBCGS,       KSPCreate_FBCGS));
  CHKERRQ(KSPRegister(KSPPIPEBCGS,    KSPCreate_PIPEBCGS));
  CHKERRQ(KSPRegister(KSPFBCGSR,      KSPCreate_FBCGSR));
  CHKERRQ(KSPRegister(KSPBCGSL,       KSPCreate_BCGSL));
  CHKERRQ(KSPRegister(KSPCGS,         KSPCreate_CGS));
  CHKERRQ(KSPRegister(KSPTFQMR,       KSPCreate_TFQMR));
  CHKERRQ(KSPRegister(KSPCR,          KSPCreate_CR));
  CHKERRQ(KSPRegister(KSPPIPECR,      KSPCreate_PIPECR));
  CHKERRQ(KSPRegister(KSPLSQR,        KSPCreate_LSQR));
  CHKERRQ(KSPRegister(KSPPREONLY,     KSPCreate_PREONLY));
  CHKERRQ(KSPRegister(KSPQCG,         KSPCreate_QCG));
  CHKERRQ(KSPRegister(KSPBICG,        KSPCreate_BiCG));
  CHKERRQ(KSPRegister(KSPFGMRES,      KSPCreate_FGMRES));
  CHKERRQ(KSPRegister(KSPPIPEFGMRES,  KSPCreate_PIPEFGMRES));
  CHKERRQ(KSPRegister(KSPMINRES,      KSPCreate_MINRES));
  CHKERRQ(KSPRegister(KSPSYMMLQ,      KSPCreate_SYMMLQ));
  CHKERRQ(KSPRegister(KSPLGMRES,      KSPCreate_LGMRES));
  CHKERRQ(KSPRegister(KSPLCD,         KSPCreate_LCD));
  CHKERRQ(KSPRegister(KSPGCR,         KSPCreate_GCR));
  CHKERRQ(KSPRegister(KSPPIPEGCR,     KSPCreate_PIPEGCR));
  CHKERRQ(KSPRegister(KSPPGMRES,      KSPCreate_PGMRES));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(KSPRegister(KSPDGMRES,      KSPCreate_DGMRES));
#endif
  CHKERRQ(KSPRegister(KSPTSIRM,       KSPCreate_TSIRM));
  CHKERRQ(KSPRegister(KSPCGLS,        KSPCreate_CGLS));
  CHKERRQ(KSPRegister(KSPFETIDP,      KSPCreate_FETIDP));
#if defined(PETSC_HAVE_HPDDM)
  CHKERRQ(KSPRegister(KSPHPDDM,       KSPCreate_HPDDM));
#endif
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitoRegisterAll - Registers all of the Krylov subspace monitors in the KSP package.

  Not Collective

  Level: advanced
@*/
PetscErrorCode KSPMonitorRegisterAll(void)
{
  PetscFunctionBegin;
  if (KSPMonitorRegisterAllCalled) PetscFunctionReturn(0);
  KSPMonitorRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(KSPMonitorRegister("preconditioned_residual",       PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorResidual,           NULL, NULL));
  CHKERRQ(KSPMonitorRegister("preconditioned_residual",       PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorResidualDraw,       NULL, NULL));
  CHKERRQ(KSPMonitorRegister("preconditioned_residual",       PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorResidualDrawLG,     KSPMonitorResidualDrawLGCreate, NULL));
  CHKERRQ(KSPMonitorRegister("preconditioned_residual_short", PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorResidualShort,      NULL, NULL));
  CHKERRQ(KSPMonitorRegister("preconditioned_residual_range", PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorResidualRange,      NULL, NULL));
  CHKERRQ(KSPMonitorRegister("true_residual",                 PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorTrueResidual,       NULL, NULL));
  CHKERRQ(KSPMonitorRegister("true_residual",                 PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorTrueResidualDraw,   NULL, NULL));
  CHKERRQ(KSPMonitorRegister("true_residual",                 PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorTrueResidualDrawLG, KSPMonitorTrueResidualDrawLGCreate, NULL));
  CHKERRQ(KSPMonitorRegister("true_residual_max",             PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorTrueResidualMax,    NULL, NULL));
  CHKERRQ(KSPMonitorRegister("error",                         PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorError,              NULL, NULL));
  CHKERRQ(KSPMonitorRegister("error",                         PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorErrorDraw,          NULL, NULL));
  CHKERRQ(KSPMonitorRegister("error",                         PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorErrorDrawLG,        KSPMonitorErrorDrawLGCreate, NULL));
  CHKERRQ(KSPMonitorRegister("solution",                      PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorSolution,           NULL, NULL));
  CHKERRQ(KSPMonitorRegister("solution",                      PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorSolutionDraw,       NULL, NULL));
  CHKERRQ(KSPMonitorRegister("solution",                      PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorSolutionDrawLG,     KSPMonitorSolutionDrawLGCreate, NULL));
  CHKERRQ(KSPMonitorRegister("singular_value",                PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorSingularValue,      KSPMonitorSingularValueCreate, NULL));
  CHKERRQ(KSPMonitorRegister("lsqr_residual",                 PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPLSQRMonitorResidual,       NULL, NULL));
  CHKERRQ(KSPMonitorRegister("lsqr_residual",                 PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPLSQRMonitorResidualDrawLG, KSPLSQRMonitorResidualDrawLGCreate, NULL));
  PetscFunctionReturn(0);
}
