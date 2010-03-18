#ifndef _COMPAT_PETSC_SNES_H
#define _COMPAT_PETSC_SNES_H

#include "private/snesimpl.h"

#if !defined(SNESPICARD)
#define SNESPICARD "picard"
#endif

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define SNESPICARD "picard"
#endif

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
/**/
#if PETSC_VERSION_(2,3,2)
typedef struct _p_MatSNESMFCtx* MatMFFD;
#define MATMFFD_COOKIE MATSNESMFCTX_COOKIE
#define MatCreateSNESMF(snes,J) MatCreateSNESMF((snes),(snes)->vec_func,(J))
#define MatMFFDSetFromOptions MatSNESMFSetFromOptions
#define MatMFFDComputeJacobian MatSNESMFComputeJacobian
#endif
/**/
#undef __FUNCT__
#define __FUNCT__ "MatMFFDSetOptionsPrefix"
static PETSC_UNUSED
PetscErrorCode MatMFFDSetOptionsPrefix_Compat(Mat mat, const char prefix[])
{
  MatMFFD        mfctx = mat ? (MatMFFD)mat->data : PETSC_NULL ;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(mfctx,MATMFFD_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)mfctx,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatMFFDSetOptionsPrefix MatMFFDSetOptionsPrefix_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "SNESSetFunction"
static PETSC_UNUSED
PetscErrorCode SNESSetFunction_Compat(SNES snes,Vec r,PetscErrorCode (*func)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(r,VEC_COOKIE,2);
  PetscCheckSameComm(snes,1,r,2);
  ierr = PetscObjectCompose((PetscObject)snes, "__vec_fun__", (PetscObject)r);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,r,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define SNESSetFunction SNESSetFunction_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "SNESSolve"
static PETSC_UNUSED
PetscErrorCode SNESSolve_Compat(SNES snes,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscCheckSameComm(snes,1,x,3);
  if (b) PetscValidHeaderSpecific(b,VEC_COOKIE,2);
  if (b) PetscCheckSameComm(snes,1,b,2);
  ierr = PetscObjectCompose((PetscObject)snes, "__vec_sol__", (PetscObject)x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,b,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define SNESSolve SNESSolve_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "SNESSetConvergenceTest"
static PETSC_UNUSED
PetscErrorCode SNESSetConvergenceTest_Compat(SNES snes,
					     PetscErrorCode (*converge)(SNES,PetscInt,PetscReal,PetscReal,PetscReal,
									SNESConvergedReason*,void*),
					     void *cctx,
					     PetscErrorCode (*destroy)(void*))
{
  PetscContainer container = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (destroy) {
    ierr = PetscContainerCreate(((PetscObject)snes)->comm,&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,cctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,destroy);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)snes,"SNESConvTestCtx",(PetscObject)container);CHKERRQ(ierr);
  if (container) { ierr = PetscContainerDestroy(container);CHKERRQ(ierr); }
  ierr = SNESSetConvergenceTest(snes,converge,cctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define SNESSetConvergenceTest SNESSetConvergenceTest_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "SNESSetConvergenceHistory"
PETSC_STATIC_INLINE PetscErrorCode
SNESSetConvergenceHistory_Compat(SNES snes, PetscReal a[],PetscInt its[],PetscInt na,PetscTruth reset)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = SNESSetConvergenceHistory(snes,a,its,na,reset);CHKERRQ(ierr);
  snes->conv_hist_len = 0;
  PetscFunctionReturn(0);
}
#define SNESSetConvergenceHistory SNESSetConvergenceHistory_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "SNESGetNumberFunctionEvals"
static PETSC_UNUSED
PetscErrorCode SNESGetNumberFunctionEvals(SNES snes, PetscInt *nfuncs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(nfuncs,2);
  *nfuncs = snes->nfuncs;
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetParams"
static PETSC_UNUSED
PetscErrorCode SNESLineSearchSetParams_Compat(SNES snes,PetscReal alpha,PetscReal maxstep)
{
  PetscReal steptol;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = SNESGetTolerances(snes,PETSC_NULL,PETSC_NULL,&steptol,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchSetParams(snes,alpha,maxstep,steptol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define SNESLineSearchSetParams SNESLineSearchSetParams_Compat
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetParams"
static PETSC_UNUSED
PetscErrorCode SNESLineSearchGetParams_Compat(SNES snes,PetscReal *alpha,PetscReal *maxstep)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = SNESLineSearchGetParams(snes,alpha,maxstep,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define SNESLineSearchGetParams SNESLineSearchGetParams_Compat
#endif

#if PETSC_VERSION_(2,3,2)
#undef __FUNCT__
#define __FUNCT__ "SNESKSPSetUseEW"
static PETSC_UNUSED
PetscErrorCode SNESKSPSetUseEW(SNES snes,PetscTruth flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->ksp_ewconv = flag;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESKSPGetUseEW"
static PETSC_UNUSED
PetscErrorCode SNESKSPGetUseEW(SNES snes, PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(flag,2);
  *flag = snes->ksp_ewconv;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESKSPGetParametersEW"
static PETSC_UNUSED
PetscErrorCode SNESKSPGetParametersEW(SNES snes,PetscInt *version,PetscReal *rtol_0,PetscReal *rtol_max,
				      PetscReal *gamma,PetscReal *alpha,PetscReal *alpha2,PetscReal *threshold)
{
  SNES_KSP_EW_ConvCtx *kctx;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");
  if(version)   *version   = kctx->version;
  if(rtol_0)    *rtol_0    = kctx->rtol_0;
  if(rtol_max)  *rtol_max  = kctx->rtol_max;
  if(gamma)     *gamma     = kctx->gamma;
  if(alpha)     *alpha     = kctx->alpha;
  if(alpha2)    *alpha2    = kctx->alpha2;
  if(threshold) *threshold = kctx->threshold;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESKSPSetParametersEW"
static PETSC_UNUSED
PetscErrorCode SNESKSPSetParametersEW(SNES snes,PetscInt version,PetscReal rtol_0,PetscReal rtol_max,
				      PetscReal gamma,PetscReal alpha,PetscReal alpha2,PetscReal threshold)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = SNES_KSP_SetParametersEW(snes,version,rtol_0,rtol_max,gamma,alpha,alpha2,threshold);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
#endif

#if PETSC_VERSION_(2,3,2)
#define SNES_CONVERGED_ITS ((SNESConvergedReason)5)
#undef __FUNCT__
#define __FUNCT__ "SNESDefaultConverged"
static PETSC_UNUSED
PetscErrorCode SNESDefaultConverged(SNES snes,PetscInt it,
				    PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,
				    SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(reason,6);
  ierr = SNESConverged_LS(snes,it,xnorm,pnorm,fnorm,reason,dummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESSkipConverged"
static PETSC_UNUSED
PetscErrorCode SNESSkipConverged(SNES snes,PetscInt it,
				 PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,
				 SNESConvergedReason *reason,void *dummy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(reason,6);
  *reason = SNES_CONVERGED_ITERATING;
  PetscFunctionReturn(0);
}
#endif

#if PETSC_VERSION_(2,3,2)
#define SNESGetNonlinearStepFailures SNESGetNumberUnsuccessfulSteps
#define SNESSetMaxNonlinearStepFailures SNESSetMaximumUnsuccessfulSteps
#define SNESGetMaxNonlinearStepFailures SNESGetMaximumUnsuccessfulSteps
#define SNESGetLinearSolveIterations SNESGetNumberLinearIterations
#endif

#if PETSC_VERSION_(2,3,2)
#define SNESMonitorSet SNESSetMonitor
#define SNESMonitorCancel SNESClearMonitor
#define SNESMonitorDefault SNESDefaultMonitor
#define SNESMonitorResidual SNESVecViewResidualMonitor
#define SNESMonitorSolution SNESVecViewMonitor
#define SNESMonitorSolutionUpdate SNESVecViewUpdateMonitor
#define SNESMonitorLG SNESLGMonitor
#endif

#endif /* _COMPAT_PETSC_SNES_H */
