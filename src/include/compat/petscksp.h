#ifndef _COMPAT_PETSC_KSP_H
#define _COMPAT_PETSC_KSP_H

#include "private/kspimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define KSPSetPCSide KSPSetPreconditionerSide
#define KSPGetPCSide KSPGetPreconditionerSide
#endif

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define KSPBROYDEN "broyden"
#define KSPGCR     "gcr"
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define KSPNASH  "nash"
#define KSPGLTR  "gltr"
#define KSPIBCGS "ibcgs"
#endif

#if PETSC_VERSION_(2,3,2)
#define KSP_NORM_NO		  KSP_NO_NORM
#define KSP_NORM_PRECONDITIONED   KSP_PRECONDITIONED_NORM
#define KSP_NORM_UNPRECONDITIONED KSP_UNPRECONDITIONED_NORM
#define KSP_NORM_NATURAL          KSP_NATURAL_NORM
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedCreate"
static PETSC_UNUSED
PetscErrorCode KSPDefaultConvergedCreate_Compat(void **ctx)
{
  PetscFunctionBegin;
  if (ctx) *ctx = PETSC_NULL;
  PetscFunctionReturn(0);
}
#define KSPDefaultConvergedCreate KSPDefaultConvergedCreate_Compat

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedDestroy"
static PETSC_UNUSED
PetscErrorCode KSPDefaultConvergedDestroy_Compat(void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#define KSPDefaultConvergedDestroy KSPDefaultConvergedDestroy_Compat

#undef __FUNCT__
#define __FUNCT__ "KSPSkipConverged"
static PETSC_UNUSED
PetscErrorCode KSPSkipConverged_Compat(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason, void *dummy)
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
#define KSPSkipConverged KSPSkipConverged_Compat

#undef __FUNCT__
#define __FUNCT__ "KSPSetConvergenceTest"
static PETSC_UNUSED
PetscErrorCode KSPSetConvergenceTest_Compat(KSP ksp,
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
#define KSPSetConvergenceTest KSPSetConvergenceTest_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "KSPSetNormType"
static PETSC_UNUSED
PetscErrorCode KSPSetNormType_Compat(KSP ksp, KSPNormType normtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = KSPSetNormType(ksp, normtype);CHKERRQ(ierr);
  if (normtype != KSP_NORM_NO) PetscFunctionReturn(0);
  ierr = KSPSetConvergenceTest(ksp,KSPSkipConverged,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define KSPSetNormType KSPSetNormType_Compat
#endif

#if PETSC_VERSION_(2,3,2)
#undef __FUNCT__
#define __FUNCT__ "KSPGetNormType"
static PETSC_UNUSED
PetscErrorCode KSPGetNormType_Compat(KSP ksp, KSPNormType *normtype) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(normtype,2);
  *normtype = ksp->normtype;
  PetscFunctionReturn(0);
}
#define KSPGetNormType KSPGetNormType_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "KSPSetUseFischerGuess"
static PETSC_UNUSED
PetscErrorCode KSPSetUseFischerGuess_Compat(KSP ksp,
					    PetscInt model,PetscInt size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  SETERRQ(PETSC_ERR_SUP,"KSPSetUseFischerGuess()"
	  " not available in this PETSc version");
  PetscFunctionReturn(0);
}
#define KSPSetUseFischerGuess KSPSetUseFischerGuess_Compat
#endif


#if PETSC_VERSION_(2,3,2)
#define KSP_CONVERGED_CG_NEG_CURVE   KSP_CONVERGED_STCG_NEG_CURVE
#define KSP_CONVERGED_CG_CONSTRAINED KSP_CONVERGED_STCG_CONSTRAINED
#endif

#if PETSC_VERSION_(2,3,2)
#define KSPMonitorSet KSPSetMonitor
#define KSPMonitorCancel KSPClearMonitor
#define KSPMonitorDefault KSPDefaultMonitor
#define KSPMonitorTrueResidualNorm KSPTrueMonitor
#define KSPMonitorSolution KSPVecViewMonitor
#define KSPMonitorLG KSPLGMonitor
#endif

#endif /* _COMPAT_PETSC_KSP_H */
