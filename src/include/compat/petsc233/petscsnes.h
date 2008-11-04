#ifndef _PETSC_COMPAT_SNES_H
#define _PETSC_COMPAT_SNES_H

#include "private/snesimpl.h"

#undef __FUNCT__
#define __FUNCT__ "SNESSetFunction_233"
static PETSC_UNUSED
PetscErrorCode SNESSetFunction_233(SNES snes,Vec r,PetscErrorCode (*func)(SNES,Vec,Vec,void*),void *ctx)
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
#define SNESSetFunction SNESSetFunction_233

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_233"
static PETSC_UNUSED
PetscErrorCode SNESSolve_233(SNES snes,Vec b,Vec x)
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
#define SNESSolve SNESSolve_233

#undef __FUNCT__
#define __FUNCT__ "SNESSetConvergenceTest_233"
static PETSC_UNUSED
PetscErrorCode SNESSetConvergenceTest_233(SNES snes,
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
#define SNESSetConvergenceTest SNESSetConvergenceTest_233

#undef __FUNCT__  
#define __FUNCT__ "SNESSetConvergenceHistory_233"
PETSC_STATIC_INLINE PetscErrorCode
SNESSetConvergenceHistory_233(SNES snes, PetscReal a[],PetscInt its[],PetscInt na,PetscTruth reset)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = SNESSetConvergenceHistory(snes,a,its,na,reset);CHKERRQ(ierr);
  snes->conv_hist_len = 0;
  PetscFunctionReturn(0);
}
#define SNESSetConvergenceHistory SNESSetConvergenceHistory_233

#undef __FUNCT__
#define __FUNCT__ "SNESGetNumberFunctionEvals_233"
static PETSC_UNUSED
PetscErrorCode SNESGetNumberFunctionEvals_233(SNES snes, PetscInt *nfuncs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(nfuncs,2);
  *nfuncs = snes->nfuncs;
  PetscFunctionReturn(0);
}
#define SNESGetNumberFunctionEvals SNESGetNumberFunctionEvals_233

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetParams_233"
static PETSC_UNUSED
PetscErrorCode SNESLineSearchSetParams_233(SNES snes,PetscReal alpha,PetscReal maxstep)
{
  PetscReal steptol;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = SNESGetTolerances(snes,PETSC_NULL,PETSC_NULL,&steptol,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchSetParams(snes,alpha,maxstep,steptol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define SNESLineSearchSetParams SNESLineSearchSetParams_233

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchGetParams_233"
static PETSC_UNUSED
PetscErrorCode SNESLineSearchGetParams_233(SNES snes,PetscReal *alpha,PetscReal *maxstep)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = SNESLineSearchGetParams(snes,alpha,maxstep,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define SNESLineSearchGetParams SNESLineSearchGetParams_233


#endif /* _PETSC_COMPAT_SNES_H */
