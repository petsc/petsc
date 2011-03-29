#ifndef _COMPAT_PETSC_KSP_H
#define _COMPAT_PETSC_KSP_H

#include "private/kspimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define KSPNGMRES  "ngmres"
#define KSPSPECEST "specest"
#define KSPSetPCSide KSPSetPreconditionerSide
#define KSPGetPCSide KSPGetPreconditionerSide
#define KSP_NORM_NONE KSP_NORM_NO
#endif

#if (PETSC_VERSION_(3,0,0))
#define KSPBROYDEN "broyden"
#define KSPGCR     "gcr"
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "KSPReset"
static PetscErrorCode KSPReset_Compat(KSP ksp)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->guess) {ierr = KSPFischerGuessDestroy(ksp->guess);CHKERRQ(ierr);}
  if (ksp->pc) {ierr = PCReset(ksp->pc);CHKERRQ(ierr);}
  if (ksp->vec_rhs) {ierr = VecDestroy(ksp->vec_rhs);CHKERRQ(ierr);}
  if (ksp->vec_sol) {ierr = VecDestroy(ksp->vec_sol);CHKERRQ(ierr);}
  if (ksp->diagonal) {ierr = VecDestroy(ksp->diagonal);CHKERRQ(ierr);}
  if (ksp->truediagonal) {ierr = VecDestroy(ksp->truediagonal);CHKERRQ(ierr);}
  if (ksp->nullsp) {ierr = MatNullSpaceDestroy(&ksp->nullsp);CHKERRQ(ierr);}
  /*if (ksp->ops->reset) {ierr = (*ksp->ops->reset)(ksp);CHKERRQ(ierr);}*/
  ksp->guess = PETSC_NULL;
  ksp->vec_rhs = 0;
  ksp->vec_sol = 0;
  ksp->diagonal = 0;
  ksp->truediagonal = 0;
  ksp->nullsp = 0;
  ksp->setupcalled = 0;
  PetscFunctionReturn(0);
}
#define KSPReset KSPReset_Compat
#endif

#endif /* _COMPAT_PETSC_KSP_H */
