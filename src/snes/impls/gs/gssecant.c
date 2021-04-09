#include <../src/snes/impls/gs/gsimpl.h>

PETSC_EXTERN PetscErrorCode SNESComputeNGSDefaultSecant(SNES snes,Vec X,Vec B,void *ctx)
{
  PetscErrorCode    ierr;
  SNES_NGS          *gs = (SNES_NGS*)snes->data;
  PetscInt          i,j,k,ncolors;
  DM                dm;
  PetscBool         flg;
  ISColoring        coloring = gs->coloring;
  MatColoring       mc;
  Vec               W,G,F;
  PetscScalar       h=gs->h;
  IS                *coloris;
  PetscScalar       f,g,x,w,d;
  PetscReal         dxt,xt,ft,ft1=0;
  const PetscInt    *idx;
  PetscInt          size,s;
  PetscReal         atol,rtol,stol;
  PetscInt          its;
  PetscErrorCode    (*func)(SNES,Vec,Vec,void*);
  void              *fctx;
  PetscBool         mat = gs->secant_mat,equal,isdone,alldone;
  PetscScalar       *xa,*wa;
  const PetscScalar *fa,*ga;

  PetscFunctionBegin;
  if (snes->nwork < 3) {
    ierr = SNESSetWorkVecs(snes,3);CHKERRQ(ierr);
  }
  W = snes->work[0];
  G = snes->work[1];
  F = snes->work[2];
  ierr = VecGetOwnershipRange(X,&s,NULL);CHKERRQ(ierr);
  ierr = SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&its);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,NULL,&func,&fctx);CHKERRQ(ierr);
  if (!coloring) {
    /* create the coloring */
    ierr = DMHasColoring(dm,&flg);CHKERRQ(ierr);
    if (flg && !mat) {
      ierr = DMCreateColoring(dm,IS_COLORING_GLOBAL,&coloring);CHKERRQ(ierr);
    } else {
      if (!snes->jacobian) {ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);}
      ierr = MatColoringCreate(snes->jacobian,&mc);CHKERRQ(ierr);
      ierr = MatColoringSetDistance(mc,1);CHKERRQ(ierr);
      ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
      ierr = MatColoringApply(mc,&coloring);CHKERRQ(ierr);
      ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
    }
    gs->coloring = coloring;
  }
  ierr = ISColoringGetIS(coloring,PETSC_USE_POINTER,&ncolors,&coloris);CHKERRQ(ierr);
  ierr = VecEqual(X,snes->vec_sol,&equal);CHKERRQ(ierr);
  if (equal && snes->normschedule == SNES_NORM_ALWAYS) {
    /* assume that the function is already computed */
    ierr = VecCopy(snes->vec_func,F);CHKERRQ(ierr);
  } else {
    ierr = PetscLogEventBegin(SNES_NGSFuncEval,snes,X,B,0);CHKERRQ(ierr);
    ierr = (*func)(snes,X,F,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_NGSFuncEval,snes,X,B,0);CHKERRQ(ierr);
    if (B) {ierr = VecAXPY(F,-1.0,B);CHKERRQ(ierr);}
  }
  for (i=0;i<ncolors;i++) {
    ierr = ISGetIndices(coloris[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(coloris[i],&size);CHKERRQ(ierr);
    ierr = VecCopy(X,W);CHKERRQ(ierr);
    ierr = VecGetArray(W,&wa);CHKERRQ(ierr);
    for (j=0;j<size;j++) {
      wa[idx[j]-s] += h;
    }
    ierr = VecRestoreArray(W,&wa);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SNES_NGSFuncEval,snes,X,B,0);CHKERRQ(ierr);
    ierr = (*func)(snes,W,G,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_NGSFuncEval,snes,X,B,0);CHKERRQ(ierr);
    if (B) {ierr = VecAXPY(G,-1.0,B);CHKERRQ(ierr);}
    for (k=0;k<its;k++) {
      dxt = 0.;
      xt = 0.;
      ft = 0.;
      ierr = VecGetArray(W,&wa);CHKERRQ(ierr);
      ierr = VecGetArray(X,&xa);CHKERRQ(ierr);
      ierr = VecGetArrayRead(F,&fa);CHKERRQ(ierr);
      ierr = VecGetArrayRead(G,&ga);CHKERRQ(ierr);
      for (j=0;j<size;j++) {
        f = fa[idx[j]-s];
        x = xa[idx[j]-s];
        g = ga[idx[j]-s];
        w = wa[idx[j]-s];
        if (PetscAbsScalar(g-f) > atol) {
          /* This is equivalent to d = x - (h*f) / PetscRealPart(g-f) */
          d = (x*g-w*f) / PetscRealPart(g-f);
        } else {
          d = x;
        }
        dxt += PetscRealPart(PetscSqr(d-x));
        xt += PetscRealPart(PetscSqr(x));
        ft += PetscRealPart(PetscSqr(f));
        xa[idx[j]-s] = d;
      }
      ierr = VecRestoreArray(X,&xa);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(F,&fa);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(G,&ga);CHKERRQ(ierr);
      ierr = VecRestoreArray(W,&wa);CHKERRQ(ierr);

      if (k == 0) ft1 = PetscSqrtReal(ft);
      if (k<its-1) {
        isdone = PETSC_FALSE;
        if (stol*PetscSqrtReal(xt) > PetscSqrtReal(dxt)) isdone = PETSC_TRUE;
        if (PetscSqrtReal(ft) < atol) isdone = PETSC_TRUE;
        if (rtol*ft1 > PetscSqrtReal(ft)) isdone = PETSC_TRUE;
        ierr = MPIU_Allreduce(&isdone,&alldone,1,MPIU_BOOL,MPI_BAND,PetscObjectComm((PetscObject)snes));CHKERRMPI(ierr);
        if (alldone) break;
      }
      if (i < ncolors-1 || k < its-1) {
        ierr = PetscLogEventBegin(SNES_NGSFuncEval,snes,X,B,0);CHKERRQ(ierr);
        ierr = (*func)(snes,X,F,fctx);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(SNES_NGSFuncEval,snes,X,B,0);CHKERRQ(ierr);
        if (B) {ierr = VecAXPY(F,-1.0,B);CHKERRQ(ierr);}
      }
      if (k<its-1) {
        ierr = VecSwap(X,W);CHKERRQ(ierr);
        ierr = VecSwap(F,G);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISColoringRestoreIS(coloring,PETSC_USE_POINTER,&coloris);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
