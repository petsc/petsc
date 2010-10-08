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
