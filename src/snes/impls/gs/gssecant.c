#include <petsc-private/snesimpl.h>      /*I "petscsnes.h"  I*/
#include <petscdm.h>

#undef __FUNCT__
#define __FUNCT__ "GSDestroy_Private"
PetscErrorCode GSDestroy_Private(ISColoring coloring)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeGSDefaultSecant"
PETSC_EXTERN PetscErrorCode SNESComputeGSDefaultSecant(SNES snes,Vec X,Vec B,void *ctx)
{

  PetscErrorCode ierr;
  PetscInt       i,j,k,ncolors;
  DM             dm;
  PetscBool      flg;
  ISColoring     coloring;
  MatColoring    mc;
  Vec            W,G,F;
  PetscScalar    h=1e-6;
  IS             *coloris;
  PetscScalar    f,g,x,w,d;
  const PetscInt *idx;
  PetscInt       size;
  PetscReal      atol,rtol,stol;
  PetscInt       its;
  PetscErrorCode (*func)(SNES,Vec,Vec,void*);
  void           *fctx;
  PetscContainer colorcontainer;

  PetscFunctionBegin;
  if (snes->nwork < 3) {
    ierr = SNESSetWorkVecs(snes,3);CHKERRQ(ierr);
  }
  W = snes->work[0];
  G = snes->work[1];
  F = snes->work[2];
  ierr = SNESGSGetTolerances(snes,&atol,&rtol,&stol,&its);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,NULL,&func,&fctx);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)snes,"SNESGSColoring",(PetscObject*)&colorcontainer);CHKERRQ(ierr);
  if (!colorcontainer) {
    /* create the coloring */
    ierr = DMHasColoring(dm,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMCreateColoring(dm,IS_COLORING_GLOBAL,&coloring);CHKERRQ(ierr);
    } else {
      if (!snes->jacobian) {ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);}
      ierr = MatColoringCreate(snes->jacobian,&mc);CHKERRQ(ierr);
      ierr = MatColoringSetDistance(mc,1);CHKERRQ(ierr);
      ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
      ierr = MatColoringApply(mc,&coloring);CHKERRQ(ierr);
      ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
    }
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)snes),&colorcontainer);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(colorcontainer,(void *)coloring);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(colorcontainer,(PetscErrorCode (*)(void *))GSDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)snes,"SNESGSColoring",(PetscObject)colorcontainer);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&colorcontainer);CHKERRQ(ierr);
  } else {
    ierr = PetscContainerGetPointer(colorcontainer,(void **)&coloring);CHKERRQ(ierr);
  }
  ierr = ISColoringGetIS(coloring,&ncolors,&coloris);CHKERRQ(ierr);
  for (i=0;i<ncolors;i++) {
    for (k=0;k<its;k++) {
      ierr = (*func)(snes,X,F,fctx);CHKERRQ(ierr);
      if (B) {ierr = VecAXPY(F,-1.0,B);CHKERRQ(ierr);}
      ierr = ISGetIndices(coloris[i],&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(coloris[i],&size);CHKERRQ(ierr);
      ierr = VecCopy(X,W);CHKERRQ(ierr);
      for (j=0;j<size;j++) {
        ierr = VecSetValue(W,idx[j],h,ADD_VALUES);CHKERRQ(ierr);
      }
      ierr = (*func)(snes,W,G,fctx);CHKERRQ(ierr);
      if (B) {ierr = VecAXPY(G,-1.0,B);CHKERRQ(ierr);}
      for (j=0;j<size;j++) {
        ierr = VecGetValues(F,1,&idx[j],&f);CHKERRQ(ierr);
        ierr = VecGetValues(X,1,&idx[j],&x);CHKERRQ(ierr);
        ierr = VecGetValues(G,1,&idx[j],&g);CHKERRQ(ierr);
        ierr = VecGetValues(W,1,&idx[j],&w);CHKERRQ(ierr);
        d = (x*g-w*f) / PetscRealPart(g-f);
        ierr = VecSetValue(X,idx[j],d,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISColoringRestoreIS(coloring,&coloris);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
