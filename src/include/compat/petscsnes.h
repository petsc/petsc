#ifndef _COMPAT_PETSC_SNES_H
#define _COMPAT_PETSC_SNES_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#include "private/snesimpl.h"
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define SNESKSPONLY "ksponly"
#define SNESPICARD  "picard"
#define SNESVI      "vi"
#define SNESNGMRES  "ngmres"
#define SNES_DIVERGED_LINE_SEARCH SNES_DIVERGED_LS_FAILURE
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "SNESVISetVariableBounds"
static PetscErrorCode  
SNESVISetVariableBounds(SNES snes, Vec xl, Vec xu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(xl,VEC_COOKIE,1);
  PetscValidHeaderSpecific(xu,VEC_COOKIE,1);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "SNESSetComputeInitialGuess"
static PetscErrorCode  SNESSetComputeInitialGuess(SNES snes,
                                                  PetscErrorCode (*func)(SNES,Vec,void*),
                                                  void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "SNESMonitor_Compat"
static PetscErrorCode
SNESMonitor_Compat(SNES snes, PetscInt its, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SNESMonitor(snes,its,rnorm);
  PetscFunctionReturn(0);
}
#undef  SNESMonitor
#define SNESMonitor SNESMonitor_Compat


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

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "SNESReset"
static PetscErrorCode SNESReset_Compat(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define SNESReset SNESReset_Compat
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
