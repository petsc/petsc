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
#define __FUNCT__ "KSPSetDM"
static PetscErrorCode KSPSetDM(KSP ksp,DM dm)
{
  PC             pc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidHeaderSpecific(dm,DM_COOKIE,2);
  ierr = PetscObjectCompose((PetscObject)ksp, "__DM__",(PetscObject)dm);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetDM(pc,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "KSPGetDM"
static PetscErrorCode KSPGetDM(KSP ksp,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(dm,2);
  ierr = PetscObjectQuery((PetscObject)ksp, "__DM__",(PetscObject*)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "KSPReset"
static PetscErrorCode KSPReset_Compat(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  SETERRQ(PETSC_ERR_SUP,"not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define KSPReset KSPReset_Compat
#endif

#endif /* _COMPAT_PETSC_KSP_H */
