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

  PetscCall(KSPRegister(KSPCG,          KSPCreate_CG));
  PetscCall(KSPRegister(KSPGROPPCG,     KSPCreate_GROPPCG));
  PetscCall(KSPRegister(KSPPIPECG,      KSPCreate_PIPECG));
  PetscCall(KSPRegister(KSPPIPECGRR,    KSPCreate_PIPECGRR));
  PetscCall(KSPRegister(KSPPIPELCG,     KSPCreate_PIPELCG));
  PetscCall(KSPRegister(KSPPIPEPRCG,    KSPCreate_PIPEPRCG));
  PetscCall(KSPRegister(KSPPIPECG2,     KSPCreate_PIPECG2));
  PetscCall(KSPRegister(KSPCGNE,        KSPCreate_CGNE));
  PetscCall(KSPRegister(KSPNASH,        KSPCreate_NASH));
  PetscCall(KSPRegister(KSPSTCG,        KSPCreate_STCG));
  PetscCall(KSPRegister(KSPGLTR,        KSPCreate_GLTR));
  PetscCall(KSPRegister(KSPRICHARDSON,  KSPCreate_Richardson));
  PetscCall(KSPRegister(KSPCHEBYSHEV,   KSPCreate_Chebyshev));
  PetscCall(KSPRegister(KSPGMRES,       KSPCreate_GMRES));
  PetscCall(KSPRegister(KSPTCQMR,       KSPCreate_TCQMR));
  PetscCall(KSPRegister(KSPFCG  ,       KSPCreate_FCG));
  PetscCall(KSPRegister(KSPPIPEFCG,     KSPCreate_PIPEFCG));
  PetscCall(KSPRegister(KSPBCGS,        KSPCreate_BCGS));
  PetscCall(KSPRegister(KSPIBCGS,       KSPCreate_IBCGS));
  PetscCall(KSPRegister(KSPQMRCGS,      KSPCreate_QMRCGS));
  PetscCall(KSPRegister(KSPFBCGS,       KSPCreate_FBCGS));
  PetscCall(KSPRegister(KSPPIPEBCGS,    KSPCreate_PIPEBCGS));
  PetscCall(KSPRegister(KSPFBCGSR,      KSPCreate_FBCGSR));
  PetscCall(KSPRegister(KSPBCGSL,       KSPCreate_BCGSL));
  PetscCall(KSPRegister(KSPCGS,         KSPCreate_CGS));
  PetscCall(KSPRegister(KSPTFQMR,       KSPCreate_TFQMR));
  PetscCall(KSPRegister(KSPCR,          KSPCreate_CR));
  PetscCall(KSPRegister(KSPPIPECR,      KSPCreate_PIPECR));
  PetscCall(KSPRegister(KSPLSQR,        KSPCreate_LSQR));
  PetscCall(KSPRegister(KSPPREONLY,     KSPCreate_PREONLY));
  PetscCall(KSPRegister(KSPNONE,        KSPCreate_PREONLY));
  PetscCall(KSPRegister(KSPQCG,         KSPCreate_QCG));
  PetscCall(KSPRegister(KSPBICG,        KSPCreate_BiCG));
  PetscCall(KSPRegister(KSPFGMRES,      KSPCreate_FGMRES));
  PetscCall(KSPRegister(KSPPIPEFGMRES,  KSPCreate_PIPEFGMRES));
  PetscCall(KSPRegister(KSPMINRES,      KSPCreate_MINRES));
  PetscCall(KSPRegister(KSPSYMMLQ,      KSPCreate_SYMMLQ));
  PetscCall(KSPRegister(KSPLGMRES,      KSPCreate_LGMRES));
  PetscCall(KSPRegister(KSPLCD,         KSPCreate_LCD));
  PetscCall(KSPRegister(KSPGCR,         KSPCreate_GCR));
  PetscCall(KSPRegister(KSPPIPEGCR,     KSPCreate_PIPEGCR));
  PetscCall(KSPRegister(KSPPGMRES,      KSPCreate_PGMRES));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(KSPRegister(KSPDGMRES,      KSPCreate_DGMRES));
#endif
  PetscCall(KSPRegister(KSPTSIRM,       KSPCreate_TSIRM));
  PetscCall(KSPRegister(KSPCGLS,        KSPCreate_CGLS));
  PetscCall(KSPRegister(KSPFETIDP,      KSPCreate_FETIDP));
#if defined(PETSC_HAVE_HPDDM)
  PetscCall(KSPRegister(KSPHPDDM,       KSPCreate_HPDDM));
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

  PetscCall(KSPMonitorRegister("preconditioned_residual",       PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorResidual,           NULL, NULL));
  PetscCall(KSPMonitorRegister("preconditioned_residual",       PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorResidualDraw,       NULL, NULL));
  PetscCall(KSPMonitorRegister("preconditioned_residual",       PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorResidualDrawLG,     KSPMonitorResidualDrawLGCreate, NULL));
  PetscCall(KSPMonitorRegister("preconditioned_residual_short", PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorResidualShort,      NULL, NULL));
  PetscCall(KSPMonitorRegister("preconditioned_residual_range", PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorResidualRange,      NULL, NULL));
  PetscCall(KSPMonitorRegister("true_residual",                 PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorTrueResidual,       NULL, NULL));
  PetscCall(KSPMonitorRegister("true_residual",                 PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorTrueResidualDraw,   NULL, NULL));
  PetscCall(KSPMonitorRegister("true_residual",                 PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorTrueResidualDrawLG, KSPMonitorTrueResidualDrawLGCreate, NULL));
  PetscCall(KSPMonitorRegister("true_residual_max",             PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorTrueResidualMax,    NULL, NULL));
  PetscCall(KSPMonitorRegister("error",                         PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorError,              NULL, NULL));
  PetscCall(KSPMonitorRegister("error",                         PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorErrorDraw,          NULL, NULL));
  PetscCall(KSPMonitorRegister("error",                         PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorErrorDrawLG,        KSPMonitorErrorDrawLGCreate, NULL));
  PetscCall(KSPMonitorRegister("solution",                      PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorSolution,           NULL, NULL));
  PetscCall(KSPMonitorRegister("solution",                      PETSCVIEWERDRAW,  PETSC_VIEWER_DEFAULT, KSPMonitorSolutionDraw,       NULL, NULL));
  PetscCall(KSPMonitorRegister("solution",                      PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPMonitorSolutionDrawLG,     KSPMonitorSolutionDrawLGCreate, NULL));
  PetscCall(KSPMonitorRegister("singular_value",                PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPMonitorSingularValue,      KSPMonitorSingularValueCreate, NULL));
  PetscCall(KSPMonitorRegister("lsqr_residual",                 PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, KSPLSQRMonitorResidual,       NULL, NULL));
  PetscCall(KSPMonitorRegister("lsqr_residual",                 PETSCVIEWERDRAW,  PETSC_VIEWER_DRAW_LG, KSPLSQRMonitorResidualDrawLG, KSPLSQRMonitorResidualDrawLGCreate, NULL));
  PetscFunctionReturn(0);
}
