#ifndef _COMPAT_PETSC_SNES_H
#define _COMPAT_PETSC_SNES_H

#include "private/snesimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define SNESKSPONLY "ksponly"
#define SNESPICARD  "picard"
#define SNESVI      "vi"
#define SNES_DIVERGED_LINE_SEARCH SNES_DIVERGED_LS_FAILURE
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "SNESReset"
static PetscErrorCode SNESReset_Compat(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  /*if (snes->ops->reset) {
    ierr = (*snes->ops->reset)(snes);CHKERRQ(ierr);
  }*/
  if (snes->ksp) {ierr = KSPReset(snes->ksp);CHKERRQ(ierr);}
  if (snes->vec_rhs) {ierr = VecDestroy(snes->vec_rhs);CHKERRQ(ierr);}
  if (snes->vec_sol) {ierr = VecDestroy(snes->vec_sol);CHKERRQ(ierr);}
  if (snes->vec_sol_update) {ierr = VecDestroy(snes->vec_sol_update);CHKERRQ(ierr);}
  if (snes->vec_func) {ierr = VecDestroy(snes->vec_func);CHKERRQ(ierr);}
  if (snes->jacobian) {ierr = MatDestroy(snes->jacobian);CHKERRQ(ierr);}
  if (snes->jacobian_pre) {ierr = MatDestroy(snes->jacobian_pre);CHKERRQ(ierr);}
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  if (snes->vwork) {ierr = VecDestroyVecs(snes->nvwork,&snes->vwork);CHKERRQ(ierr);}
  snes->nwork = snes->nvwork = 0;
  snes->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
#define SNESReset SNESReset_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "SNESSetDM"
static PetscErrorCode SNESSetDM(SNES snes,DM dm)
{
  KSP            ksp;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(dm,DM_COOKIE,2);
  ierr = PetscObjectCompose((PetscObject)snes, "__DM__",(PetscObject)dm);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "SNESGetDM"
static PetscErrorCode SNESGetDM(SNES snes,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(dm,2);
  ierr = PetscObjectQuery((PetscObject)snes, "__DM__",(PetscObject*)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "MatMFFDSetOptionsPrefix"
static PetscErrorCode MatMFFDSetOptionsPrefix_Compat(Mat mat, const char prefix[])
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

#endif /* _COMPAT_PETSC_SNES_H */
