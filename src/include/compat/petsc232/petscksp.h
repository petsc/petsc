#ifndef _PETSC_COMPAT_KSP_H
#define _PETSC_COMPAT_KSP_H

#include "private/kspimpl.h"

#define KSPNASH "nash"
#define KSPGLTR "gltr"

#define PCFactorGetMatrix      PCGetFactoredMatrix
#define PCApplyTransposeExists PCHasApplyTranspose


#define KSP_NORM_NO		  KSP_NO_NORM
#define KSP_NORM_PRECONDITIONED   KSP_PRECONDITIONED_NORM
#define KSP_NORM_UNPRECONDITIONED KSP_UNPRECONDITIONED_NORM
#define KSP_NORM_NATURAL          KSP_NATURAL_NORM

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedCreate_232"
static PETSC_UNUSED
PetscErrorCode KSPDefaultConvergedCreate_232(void **ctx)
{
  PetscFunctionBegin;
  if (ctx) *ctx = PETSC_NULL;
  PetscFunctionReturn(0);
}
#define KSPDefaultConvergedCreate KSPDefaultConvergedCreate_232

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedDestroy_232"
static PETSC_UNUSED
PetscErrorCode KSPDefaultConvergedDestroy_232(void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#define KSPDefaultConvergedDestroy KSPDefaultConvergedDestroy_232

#undef __FUNCT__
#define __FUNCT__ "KSPSkipConverged_232"
static PETSC_UNUSED
PetscErrorCode KSPSkipConverged_232(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason, void *dummy)
{
  PetscInt       maxits;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(reason,4);
  *reason = KSP_CONVERGED_ITERATING;
  ierr = KSPGetTolerances(ksp,PETSC_NULL,PETSC_NULL,PETSC_NULL,&maxits);CHKERRQ(ierr);
  if (n >= maxits) *reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(0);
}
#define KSPSkipConverged KSPSkipConverged_232

#undef __FUNCT__
#define __FUNCT__ "KSPSetConvergenceTest_232"
static PETSC_UNUSED
PetscErrorCode KSPSetConvergenceTest_232(KSP ksp,
					 PetscErrorCode (*converge)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),
					 void *cctx,
					 PetscErrorCode (*destroy)(void*))
{
  PetscContainer container = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (destroy) {
    ierr = PetscContainerCreate(((PetscObject)ksp)->comm,&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,cctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,destroy);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)ksp,"KSPConvTestCtx",(PetscObject)container);CHKERRQ(ierr);
  if (container) { ierr = PetscContainerDestroy(container);CHKERRQ(ierr); }
  ierr = KSPSetConvergenceTest(ksp,converge,cctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define KSPSetConvergenceTest KSPSetConvergenceTest_232

#undef __FUNCT__
#define __FUNCT__ "KSPSetNormType_232"
static PETSC_UNUSED
PetscErrorCode KSPSetNormType_232(KSP ksp, KSPNormType normtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = KSPSetNormType(ksp, normtype);CHKERRQ(ierr);
  if (normtype == KSP_NORM_NO) {
    ierr = KSPSetConvergenceTest(ksp, KSPSkipConverged, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define KSPSetNormType KSPSetNormType_232

#undef __FUNCT__
#define __FUNCT__ "KSPGetNormType_232"
static PETSC_UNUSED
PetscErrorCode KSPGetNormType_232(KSP ksp, KSPNormType *normtype) {
  if (ksp)  *normtype = ksp->normtype;
  else      *normtype = KSP_PRECONDITIONED_NORM;
  return 0;
}
#define KSPGetNormType KSPGetNormType_232


#define KSP_CONVERGED_CG_NEG_CURVE   KSP_CONVERGED_STCG_NEG_CURVE
#define KSP_CONVERGED_CG_CONSTRAINED KSP_CONVERGED_STCG_CONSTRAINED

#define KSPMonitorSet KSPSetMonitor
#define KSPMonitorCancel KSPClearMonitor
#define KSPMonitorDefault KSPDefaultMonitor
#define KSPMonitorTrueResidualNorm KSPTrueMonitor
#define KSPMonitorSolution KSPVecViewMonitor
#define KSPMonitorLG KSPLGMonitor

#endif /* _PETSC_COMPAT_KSP_H */
