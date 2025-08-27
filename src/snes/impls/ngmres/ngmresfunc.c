#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>

PetscErrorCode SNESNGMRESGetAdditiveLineSearch_Private(SNES snes, SNESLineSearch *linesearch)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  if (!ngmres->additive_linesearch) {
    const char *optionsprefix;
    PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
    PetscCall(SNESLineSearchCreate(PetscObjectComm((PetscObject)snes), &ngmres->additive_linesearch));
    PetscCall(SNESLineSearchSetSNES(ngmres->additive_linesearch, snes));
    PetscCall(SNESLineSearchSetType(ngmres->additive_linesearch, SNESLINESEARCHSECANT));
    PetscCall(SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch, "snes_ngmres_additive_"));
    PetscCall(SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch, optionsprefix));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)ngmres->additive_linesearch, (PetscObject)snes, 1));
  }
  *linesearch = ngmres->additive_linesearch;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESNGMRESUpdateSubspace_Private(SNES snes, PetscInt ivec, PetscInt l, Vec F, PetscReal fnorm, Vec X)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  Vec         *Fdot   = ngmres->Fdot;
  Vec         *Xdot   = ngmres->Xdot;

  PetscFunctionBegin;
  PetscCheck(ivec <= l, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "Cannot update vector %" PetscInt_FMT " with space size %" PetscInt_FMT "!", ivec, l);
  PetscCall(VecCopy(F, Fdot[ivec]));
  PetscCall(VecCopy(X, Xdot[ivec]));

  ngmres->fnorms[ivec] = fnorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESNGMRESFormCombinedSolution_Private(SNES snes, PetscInt ivec, PetscInt l, Vec XM, Vec FM, PetscReal fMnorm, Vec X, Vec XA, Vec FA)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  PetscInt     i, j;
  Vec         *Fdot       = ngmres->Fdot;
  Vec         *Xdot       = ngmres->Xdot;
  PetscScalar *beta       = ngmres->beta;
  PetscScalar *xi         = ngmres->xi;
  PetscScalar  alph_total = 0.;
  PetscReal    nu;
  Vec          Y = snes->vec_sol_update;
  PetscBool    changed_y, changed_w;

  PetscFunctionBegin;
  nu = fMnorm * fMnorm;

  /* construct the right-hand side and xi factors */
  if (l > 0) {
    PetscCall(VecMDotBegin(FM, l, Fdot, xi));
    PetscCall(VecMDotBegin(Fdot[ivec], l, Fdot, beta));
    PetscCall(VecMDotEnd(FM, l, Fdot, xi));
    PetscCall(VecMDotEnd(Fdot[ivec], l, Fdot, beta));
    for (i = 0; i < l; i++) {
      Q(i, ivec) = beta[i];
      Q(ivec, i) = beta[i];
    }
  } else {
    Q(0, 0) = ngmres->fnorms[ivec] * ngmres->fnorms[ivec];
  }

  for (i = 0; i < l; i++) beta[i] = nu - xi[i];

  /* construct h */
  for (j = 0; j < l; j++) {
    for (i = 0; i < l; i++) H(i, j) = Q(i, j) - xi[i] - xi[j] + nu;
  }
  if (l == 1) {
    /* simply set alpha[0] = beta[0] / H[0, 0] */
    if (H(0, 0) != 0.) beta[0] = beta[0] / H(0, 0);
    else beta[0] = 0.;
  } else {
    PetscCall(PetscBLASIntCast(l, &ngmres->m));
    PetscCall(PetscBLASIntCast(l, &ngmres->n));
    ngmres->info  = 0;
    ngmres->rcond = -1.;
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
    PetscCallBLAS("LAPACKgelss", LAPACKgelss_(&ngmres->m, &ngmres->n, &ngmres->nrhs, ngmres->h, &ngmres->lda, ngmres->beta, &ngmres->ldb, ngmres->s, &ngmres->rcond, &ngmres->rank, ngmres->work, &ngmres->lwork, ngmres->rwork, &ngmres->info));
#else
    PetscCallBLAS("LAPACKgelss", LAPACKgelss_(&ngmres->m, &ngmres->n, &ngmres->nrhs, ngmres->h, &ngmres->lda, ngmres->beta, &ngmres->ldb, ngmres->s, &ngmres->rcond, &ngmres->rank, ngmres->work, &ngmres->lwork, &ngmres->info));
#endif
    PetscCall(PetscFPTrapPop());
    PetscCheck(ngmres->info >= 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_LIB, "Bad argument to GELSS");
    PetscCheck(ngmres->info <= 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_LIB, "SVD failed to converge");
  }
  for (i = 0; i < l; i++) PetscCheck(!PetscIsInfOrNanScalar(beta[i]), PetscObjectComm((PetscObject)snes), PETSC_ERR_LIB, "SVD generated inconsistent output");
  alph_total = 0.;
  for (i = 0; i < l; i++) alph_total += beta[i];

  PetscCall(VecAXPBY(XA, 1.0 - alph_total, 0.0, XM));
  PetscCall(VecMAXPY(XA, l, beta, Xdot));
  /* check the validity of the step */
  PetscCall(VecWAXPY(Y, -1.0, X, XA));
  PetscCall(SNESLineSearchPostCheck(snes->linesearch, X, Y, XA, &changed_y, &changed_w));
  if (!ngmres->approxfunc) {
    if (snes->npc && snes->npcside == PC_LEFT) {
      PetscCall(SNESApplyNPC(snes, XA, NULL, FA));
    } else {
      PetscCall(SNESComputeFunction(snes, XA, FA));
    }
  } else {
    PetscCall(VecAXPBY(FA, 1.0 - alph_total, 0.0, FM));
    PetscCall(VecMAXPY(FA, l, beta, Fdot));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESNGMRESNorms_Private(SNES snes, PetscInt l, Vec X, Vec F, Vec XM, Vec FM, Vec XA, Vec FA, Vec D, PetscReal *dnorm, PetscReal *dminnorm, PetscReal *xMnorm, PetscReal *fMnorm, PetscReal *yMnorm, PetscReal *xAnorm, PetscReal *fAnorm, PetscReal *yAnorm)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  PetscReal    dcurnorm, dmin = -1.0;
  Vec         *Xdot = ngmres->Xdot;
  PetscInt     i;

  PetscFunctionBegin;
  if (xMnorm) PetscCall(VecNormBegin(XM, NORM_2, xMnorm));
  if (fMnorm) PetscCall(VecNormBegin(FM, NORM_2, fMnorm));
  if (yMnorm) {
    PetscCall(VecWAXPY(D, -1.0, XM, X));
    PetscCall(VecNormBegin(D, NORM_2, yMnorm));
  }
  if (xAnorm) PetscCall(VecNormBegin(XA, NORM_2, xAnorm));
  if (fAnorm) PetscCall(VecNormBegin(FA, NORM_2, fAnorm));
  if (yAnorm) {
    PetscCall(VecWAXPY(D, -1.0, XA, X));
    PetscCall(VecNormBegin(D, NORM_2, yAnorm));
  }
  if (dnorm) {
    PetscCall(VecWAXPY(D, -1.0, XM, XA));
    PetscCall(VecNormBegin(D, NORM_2, dnorm));
  }
  if (dminnorm) {
    for (i = 0; i < l; i++) {
      PetscCall(VecWAXPY(D, -1.0, XA, Xdot[i]));
      PetscCall(VecNormBegin(D, NORM_2, &ngmres->xnorms[i]));
    }
  }
  if (xMnorm) PetscCall(VecNormEnd(XM, NORM_2, xMnorm));
  if (fMnorm) PetscCall(VecNormEnd(FM, NORM_2, fMnorm));
  if (yMnorm) PetscCall(VecNormEnd(D, NORM_2, yMnorm));
  if (xAnorm) PetscCall(VecNormEnd(XA, NORM_2, xAnorm));
  if (fAnorm) PetscCall(VecNormEnd(FA, NORM_2, fAnorm));
  if (yAnorm) PetscCall(VecNormEnd(D, NORM_2, yAnorm));
  if (dnorm) PetscCall(VecNormEnd(D, NORM_2, dnorm));
  if (dminnorm) {
    for (i = 0; i < l; i++) {
      PetscCall(VecNormEnd(D, NORM_2, &ngmres->xnorms[i]));
      dcurnorm = ngmres->xnorms[i];
      if ((dcurnorm < dmin) || (dmin < 0.0)) dmin = dcurnorm;
    }
    *dminnorm = dmin;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESNGMRESSelect_Private(SNES snes, PetscInt k_restart, Vec XM, Vec FM, PetscReal xMnorm, PetscReal fMnorm, PetscReal yMnorm, PetscReal objM, Vec XA, Vec FA, PetscReal xAnorm, PetscReal fAnorm, PetscReal yAnorm, PetscReal objA, PetscReal dnorm, PetscReal objmin, PetscReal dminnorm, Vec X, Vec F, Vec Y, PetscReal *xnorm, PetscReal *fnorm, PetscReal *ynorm)
{
  SNES_NGMRES         *ngmres = (SNES_NGMRES *)snes->data;
  SNESLineSearchReason lssucceed;
  PetscBool            selectA;

  PetscFunctionBegin;
  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    /* X = X + \lambda(XA - X) */
    if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "obj(X_A) = %e, ||F_A||_2 = %e, obj(X_M) = %e, ||F_M||_2 = %e\n", (double)objA, (double)fAnorm, (double)objM, (double)fMnorm));
    /* Test if is XA - XM is a descent direction: we want < F(XM), XA - XM > not positive
       If positive, GMRES will be restarted see https://epubs.siam.org/doi/pdf/10.1137/110835530 */
    PetscCall(VecCopy(FM, F));
    PetscCall(VecCopy(XM, X));
    PetscCall(VecWAXPY(Y, -1.0, XA, X));                        /* minus sign since linesearch expects to find Xnew = X - lambda * Y */
    PetscCall(VecDotRealPart(FM, Y, &ngmres->descent_ls_test)); /* this is actually < F(XM), XM - XA > */
    *fnorm = fMnorm;
    if (ngmres->descent_ls_test < 0) { /* XA - XM is not a descent direction, select XM */
      *xnorm = xMnorm;
      *fnorm = fMnorm;
      *ynorm = yMnorm;
      PetscCall(VecWAXPY(Y, -1.0, X, XM));
      PetscCall(VecCopy(FM, F));
      PetscCall(VecCopy(XM, X));
    } else {
      PetscCall(SNESNGMRESGetAdditiveLineSearch_Private(snes, &ngmres->additive_linesearch));
      PetscCall(SNESLineSearchApply(ngmres->additive_linesearch, X, F, fnorm, Y));
      PetscCall(SNESLineSearchGetReason(ngmres->additive_linesearch, &lssucceed));
      PetscCall(SNESLineSearchGetNorms(ngmres->additive_linesearch, xnorm, fnorm, ynorm));
      if (lssucceed) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(PETSC_SUCCESS);
        }
      }
    }
    if (ngmres->monitor) {
      PetscReal        objT = *fnorm;
      SNESObjectiveFn *objective;

      PetscCall(SNESGetObjective(snes, &objective, NULL));
      if (objective) PetscCall(SNESComputeObjective(snes, X, &objT));
      PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "Additive solution: objective = %e\n", (double)objT));
    }
  } else if (ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
    /* Conditions for choosing the accelerated answer:
          Criterion A -- the objective function isn't increased above the minimum by too much
          Criterion B -- the choice of x^A isn't too close to some other choice
    */
    selectA = (PetscBool)(/* A */ (objA < ngmres->gammaA * objmin) && /* B */ (ngmres->epsilonB * dnorm < dminnorm || objA < ngmres->deltaB * objmin));

    if (selectA) {
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "picked X_A, obj(X_A) = %e, ||F_A||_2 = %e, obj(X_M) = %e, ||F_M||_2 = %e\n", (double)objA, (double)fAnorm, (double)objM, (double)fMnorm));
      /* copy it over */
      *xnorm = xAnorm;
      *fnorm = fAnorm;
      *ynorm = yAnorm;
      PetscCall(VecCopy(FA, F));
      PetscCall(VecCopy(XA, X));
    } else {
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "picked X_M, obj(X_A) = %e, ||F_A||_2 = %e, obj(X_M) = %e, ||F_M||_2 = %e\n", (double)objA, (double)fAnorm, (double)objM, (double)fMnorm));
      *xnorm = xMnorm;
      *fnorm = fMnorm;
      *ynorm = yMnorm;
      PetscCall(VecWAXPY(Y, -1.0, X, XM));
      PetscCall(VecCopy(FM, F));
      PetscCall(VecCopy(XM, X));
    }
  } else { /* none */
    *xnorm = xAnorm;
    *fnorm = fAnorm;
    *ynorm = yAnorm;
    PetscCall(VecCopy(FA, F));
    PetscCall(VecCopy(XA, X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESNGMRESSelectRestart_Private(SNES snes, PetscInt l, PetscReal obj, PetscReal objM, PetscReal objA, PetscReal dnorm, PetscReal objmin, PetscReal dminnorm, PetscBool *selectRestart)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  *selectRestart = PETSC_FALSE;
  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    if (ngmres->descent_ls_test < 0) { /* XA - XM is not a descent direction */
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "ascent restart: %e > 0\n", (double)-ngmres->descent_ls_test));
      *selectRestart = PETSC_TRUE;
    }
  } else if (ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
    /* difference stagnation restart */
    if (ngmres->epsilonB * dnorm > dminnorm && objA > ngmres->deltaB * objmin && l > 0) {
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "difference restart: %e > %e\n", (double)(ngmres->epsilonB * dnorm), (double)dminnorm));
      *selectRestart = PETSC_TRUE;
    }
    /* residual stagnation restart */
    if (objA > ngmres->gammaC * objmin) {
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "residual restart: %e > %e\n", (double)objA, (double)(ngmres->gammaC * objmin)));
      *selectRestart = PETSC_TRUE;
    }

    /* F_M stagnation restart */
    if (ngmres->restart_fm_rise && objM > obj) {
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "F_M rise restart: %e > %e\n", (double)objM, (double)obj));
      *selectRestart = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
