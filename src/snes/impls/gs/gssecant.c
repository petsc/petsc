#include <../src/snes/impls/gs/gsimpl.h>

PETSC_EXTERN PetscErrorCode SNESComputeNGSDefaultSecant(SNES snes, Vec X, Vec B, void *ctx)
{
  SNES_NGS       *gs = (SNES_NGS *)snes->data;
  PetscInt        i, j, k, ncolors;
  DM              dm;
  PetscBool       flg;
  ISColoring      coloring = gs->coloring;
  MatColoring     mc;
  Vec             W, G, F;
  PetscScalar     h = gs->h;
  IS             *coloris;
  PetscScalar     f, g, x, w, d;
  PetscReal       dxt, xt, ft, ft1 = 0;
  const PetscInt *idx;
  PetscInt        size, s;
  PetscReal       atol, rtol, stol;
  PetscInt        its;
  PetscErrorCode (*func)(SNES, Vec, Vec, void *);
  void              *fctx;
  PetscBool          mat = gs->secant_mat, equal, isdone, alldone;
  PetscScalar       *xa, *wa;
  const PetscScalar *fa, *ga;

  PetscFunctionBegin;
  if (snes->nwork < 3) PetscCall(SNESSetWorkVecs(snes, 3));
  W = snes->work[0];
  G = snes->work[1];
  F = snes->work[2];
  PetscCall(VecGetOwnershipRange(X, &s, NULL));
  PetscCall(SNESNGSGetTolerances(snes, &atol, &rtol, &stol, &its));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(SNESGetFunction(snes, NULL, &func, &fctx));
  if (!coloring) {
    /* create the coloring */
    PetscCall(DMHasColoring(dm, &flg));
    if (flg && !mat) {
      PetscCall(DMCreateColoring(dm, IS_COLORING_GLOBAL, &coloring));
    } else {
      if (!snes->jacobian) PetscCall(SNESSetUpMatrices(snes));
      PetscCall(MatColoringCreate(snes->jacobian, &mc));
      PetscCall(MatColoringSetDistance(mc, 1));
      PetscCall(MatColoringSetFromOptions(mc));
      PetscCall(MatColoringApply(mc, &coloring));
      PetscCall(MatColoringDestroy(&mc));
    }
    gs->coloring = coloring;
  }
  PetscCall(ISColoringGetIS(coloring, PETSC_USE_POINTER, &ncolors, &coloris));
  PetscCall(VecEqual(X, snes->vec_sol, &equal));
  if (equal && snes->normschedule == SNES_NORM_ALWAYS) {
    /* assume that the function is already computed */
    PetscCall(VecCopy(snes->vec_func, F));
  } else {
    PetscCall(PetscLogEventBegin(SNES_NGSFuncEval, snes, X, B, 0));
    PetscCall((*func)(snes, X, F, fctx));
    PetscCall(PetscLogEventEnd(SNES_NGSFuncEval, snes, X, B, 0));
    if (B) PetscCall(VecAXPY(F, -1.0, B));
  }
  for (i = 0; i < ncolors; i++) {
    PetscCall(ISGetIndices(coloris[i], &idx));
    PetscCall(ISGetLocalSize(coloris[i], &size));
    PetscCall(VecCopy(X, W));
    PetscCall(VecGetArray(W, &wa));
    for (j = 0; j < size; j++) wa[idx[j] - s] += h;
    PetscCall(VecRestoreArray(W, &wa));
    PetscCall(PetscLogEventBegin(SNES_NGSFuncEval, snes, X, B, 0));
    PetscCall((*func)(snes, W, G, fctx));
    PetscCall(PetscLogEventEnd(SNES_NGSFuncEval, snes, X, B, 0));
    if (B) PetscCall(VecAXPY(G, -1.0, B));
    for (k = 0; k < its; k++) {
      dxt = 0.;
      xt  = 0.;
      ft  = 0.;
      PetscCall(VecGetArray(W, &wa));
      PetscCall(VecGetArray(X, &xa));
      PetscCall(VecGetArrayRead(F, &fa));
      PetscCall(VecGetArrayRead(G, &ga));
      for (j = 0; j < size; j++) {
        f = fa[idx[j] - s];
        x = xa[idx[j] - s];
        g = ga[idx[j] - s];
        w = wa[idx[j] - s];
        if (PetscAbsScalar(g - f) > atol) {
          /* This is equivalent to d = x - (h*f) / PetscRealPart(g-f) */
          d = (x * g - w * f) / PetscRealPart(g - f);
        } else {
          d = x;
        }
        dxt += PetscRealPart(PetscSqr(d - x));
        xt += PetscRealPart(PetscSqr(x));
        ft += PetscRealPart(PetscSqr(f));
        xa[idx[j] - s] = d;
      }
      PetscCall(VecRestoreArray(X, &xa));
      PetscCall(VecRestoreArrayRead(F, &fa));
      PetscCall(VecRestoreArrayRead(G, &ga));
      PetscCall(VecRestoreArray(W, &wa));

      if (k == 0) ft1 = PetscSqrtReal(ft);
      if (k < its - 1) {
        isdone = PETSC_FALSE;
        if (stol * PetscSqrtReal(xt) > PetscSqrtReal(dxt)) isdone = PETSC_TRUE;
        if (PetscSqrtReal(ft) < atol) isdone = PETSC_TRUE;
        if (rtol * ft1 > PetscSqrtReal(ft)) isdone = PETSC_TRUE;
        PetscCall(MPIU_Allreduce(&isdone, &alldone, 1, MPIU_BOOL, MPI_BAND, PetscObjectComm((PetscObject)snes)));
        if (alldone) break;
      }
      if (i < ncolors - 1 || k < its - 1) {
        PetscCall(PetscLogEventBegin(SNES_NGSFuncEval, snes, X, B, 0));
        PetscCall((*func)(snes, X, F, fctx));
        PetscCall(PetscLogEventEnd(SNES_NGSFuncEval, snes, X, B, 0));
        if (B) PetscCall(VecAXPY(F, -1.0, B));
      }
      if (k < its - 1) {
        PetscCall(VecSwap(X, W));
        PetscCall(VecSwap(F, G));
      }
    }
  }
  PetscCall(ISColoringRestoreIS(coloring, PETSC_USE_POINTER, &coloris));
  PetscFunctionReturn(PETSC_SUCCESS);
}
