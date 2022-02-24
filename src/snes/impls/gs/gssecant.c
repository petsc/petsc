#include <../src/snes/impls/gs/gsimpl.h>

PETSC_EXTERN PetscErrorCode SNESComputeNGSDefaultSecant(SNES snes,Vec X,Vec B,void *ctx)
{
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
    CHKERRQ(SNESSetWorkVecs(snes,3));
  }
  W = snes->work[0];
  G = snes->work[1];
  F = snes->work[2];
  CHKERRQ(VecGetOwnershipRange(X,&s,NULL));
  CHKERRQ(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&its));
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(SNESGetFunction(snes,NULL,&func,&fctx));
  if (!coloring) {
    /* create the coloring */
    CHKERRQ(DMHasColoring(dm,&flg));
    if (flg && !mat) {
      CHKERRQ(DMCreateColoring(dm,IS_COLORING_GLOBAL,&coloring));
    } else {
      if (!snes->jacobian) CHKERRQ(SNESSetUpMatrices(snes));
      CHKERRQ(MatColoringCreate(snes->jacobian,&mc));
      CHKERRQ(MatColoringSetDistance(mc,1));
      CHKERRQ(MatColoringSetFromOptions(mc));
      CHKERRQ(MatColoringApply(mc,&coloring));
      CHKERRQ(MatColoringDestroy(&mc));
    }
    gs->coloring = coloring;
  }
  CHKERRQ(ISColoringGetIS(coloring,PETSC_USE_POINTER,&ncolors,&coloris));
  CHKERRQ(VecEqual(X,snes->vec_sol,&equal));
  if (equal && snes->normschedule == SNES_NORM_ALWAYS) {
    /* assume that the function is already computed */
    CHKERRQ(VecCopy(snes->vec_func,F));
  } else {
    CHKERRQ(PetscLogEventBegin(SNES_NGSFuncEval,snes,X,B,0));
    CHKERRQ((*func)(snes,X,F,fctx));
    CHKERRQ(PetscLogEventEnd(SNES_NGSFuncEval,snes,X,B,0));
    if (B) CHKERRQ(VecAXPY(F,-1.0,B));
  }
  for (i=0;i<ncolors;i++) {
    CHKERRQ(ISGetIndices(coloris[i],&idx));
    CHKERRQ(ISGetLocalSize(coloris[i],&size));
    CHKERRQ(VecCopy(X,W));
    CHKERRQ(VecGetArray(W,&wa));
    for (j=0;j<size;j++) {
      wa[idx[j]-s] += h;
    }
    CHKERRQ(VecRestoreArray(W,&wa));
    CHKERRQ(PetscLogEventBegin(SNES_NGSFuncEval,snes,X,B,0));
    CHKERRQ((*func)(snes,W,G,fctx));
    CHKERRQ(PetscLogEventEnd(SNES_NGSFuncEval,snes,X,B,0));
    if (B) CHKERRQ(VecAXPY(G,-1.0,B));
    for (k=0;k<its;k++) {
      dxt = 0.;
      xt = 0.;
      ft = 0.;
      CHKERRQ(VecGetArray(W,&wa));
      CHKERRQ(VecGetArray(X,&xa));
      CHKERRQ(VecGetArrayRead(F,&fa));
      CHKERRQ(VecGetArrayRead(G,&ga));
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
      CHKERRQ(VecRestoreArray(X,&xa));
      CHKERRQ(VecRestoreArrayRead(F,&fa));
      CHKERRQ(VecRestoreArrayRead(G,&ga));
      CHKERRQ(VecRestoreArray(W,&wa));

      if (k == 0) ft1 = PetscSqrtReal(ft);
      if (k<its-1) {
        isdone = PETSC_FALSE;
        if (stol*PetscSqrtReal(xt) > PetscSqrtReal(dxt)) isdone = PETSC_TRUE;
        if (PetscSqrtReal(ft) < atol) isdone = PETSC_TRUE;
        if (rtol*ft1 > PetscSqrtReal(ft)) isdone = PETSC_TRUE;
        CHKERRMPI(MPIU_Allreduce(&isdone,&alldone,1,MPIU_BOOL,MPI_BAND,PetscObjectComm((PetscObject)snes)));
        if (alldone) break;
      }
      if (i < ncolors-1 || k < its-1) {
        CHKERRQ(PetscLogEventBegin(SNES_NGSFuncEval,snes,X,B,0));
        CHKERRQ((*func)(snes,X,F,fctx));
        CHKERRQ(PetscLogEventEnd(SNES_NGSFuncEval,snes,X,B,0));
        if (B) CHKERRQ(VecAXPY(F,-1.0,B));
      }
      if (k<its-1) {
        CHKERRQ(VecSwap(X,W));
        CHKERRQ(VecSwap(F,G));
      }
    }
  }
  CHKERRQ(ISColoringRestoreIS(coloring,PETSC_USE_POINTER,&coloris));
  PetscFunctionReturn(0);
}
