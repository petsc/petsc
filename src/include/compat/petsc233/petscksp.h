#ifndef _PETSC_COMPAT_KSP_H
#define _PETSC_COMPAT_KSP_H

#define KSPNASH "nash"
#define KSPGLTR "gltr"

#define PCFactorGetMatrix      PCGetFactoredMatrix
#define PCApplyTransposeExists PCHasApplyTranspose

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedCreate_233"
static PETSC_UNUSED
PetscErrorCode KSPDefaultConvergedCreate_233(void **ctx)
{
  PetscFunctionBegin;
  if (ctx) *ctx = PETSC_NULL;
  PetscFunctionReturn(0);
}
#define KSPDefaultConvergedCreate KSPDefaultConvergedCreate_233

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedDestroy_233"
static PETSC_UNUSED
PetscErrorCode KSPDefaultConvergedDestroy_233(void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#define KSPDefaultConvergedDestroy KSPDefaultConvergedDestroy_233

#undef __FUNCT__
#define __FUNCT__ "KSPSkipConverged_233"
static PETSC_UNUSED
PetscErrorCode KSPSkipConverged_233(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason, void *dummy)
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
#define KSPSkipConverged KSPSkipConverged_233

#undef __FUNCT__
#define __FUNCT__ "KSPSetConvergenceTest_233"
static PETSC_UNUSED
PetscErrorCode KSPSetConvergenceTest_233(KSP ksp,
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
#define KSPSetConvergenceTest KSPSetConvergenceTest_233

#undef __FUNCT__
#define __FUNCT__ "KSPSetNormType_233"
static PETSC_UNUSED
PetscErrorCode KSPSetNormType_233(KSP ksp, KSPNormType normtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = KSPSetNormType(ksp, normtype);CHKERRQ(ierr);
  if (normtype == KSP_NORM_NO) {
    ierr = KSPSetConvergenceTest(ksp, KSPSkipConverged, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define KSPSetNormType KSPSetNormType_233

#endif /* _PETSC_COMPAT_KSP_H */
