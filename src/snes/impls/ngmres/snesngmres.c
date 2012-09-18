#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>

const char *const SNESNGMRESRestartTypes[] = {"NONE","PERIODIC","DIFFERENCE","SNESNGMRESRestartType","SNES_NGMRES_RESTART_",0};
const char *const SNESNGMRESSelectTypes[] = {"NONE","DIFFERENCE","LINESEARCH","SNESNGMRESSelectType","SNES_NGMRES_SELECT_",0};

#undef __FUNCT__
#define __FUNCT__ "SNESReset_NGMRES"
PetscErrorCode SNESReset_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->Fdot);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->Xdot);CHKERRQ(ierr);
  ierr = SNESLineSearchDestroy(&ngmres->additive_linesearch);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NGMRES"
PetscErrorCode SNESDestroy_NGMRES(SNES snes)
{
  PetscErrorCode ierr;
  SNES_NGMRES *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  ierr = SNESReset_NGMRES(snes);CHKERRQ(ierr);
  ierr = PetscFree5(ngmres->h, ngmres->beta, ngmres->xi, ngmres->fnorms, ngmres->q);CHKERRQ(ierr);
  ierr = PetscFree(ngmres->s);CHKERRQ(ierr);
  ierr = PetscFree(ngmres->xnorms);CHKERRQ(ierr);
#if PETSC_USE_COMPLEX
  ierr = PetscFree(ngmres->rwork);
#endif
  ierr = PetscFree(ngmres->work);
  ierr = PetscFree(snes->data);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NGMRES"
PetscErrorCode SNESSetUp_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES *) snes->data;
  const char     *optionsprefix;
  PetscInt       msize,hsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes,5);CHKERRQ(ierr);
  if (!ngmres->Xdot) {ierr = VecDuplicateVecs(snes->vec_sol,ngmres->msize,&ngmres->Xdot);CHKERRQ(ierr);}
  if (!ngmres->Fdot) {ierr = VecDuplicateVecs(snes->vec_sol,ngmres->msize,&ngmres->Fdot);CHKERRQ(ierr);}
  if (!ngmres->setup_called) {
    msize         = ngmres->msize;  /* restart size */
    hsize         = msize * msize;

    /* explicit least squares minimization solve */
    ierr = PetscMalloc5(hsize,PetscScalar,&ngmres->h,
                        msize,PetscScalar,&ngmres->beta,
                        msize,PetscScalar,&ngmres->xi,
                        msize,PetscReal,  &ngmres->fnorms,
                        hsize,PetscScalar,&ngmres->q);CHKERRQ(ierr);
    if (ngmres->singlereduction) {
      ierr = PetscMalloc(msize*sizeof(PetscReal),&ngmres->xnorms);CHKERRQ(ierr);
    }
    ngmres->nrhs = 1;
    ngmres->lda = msize;
    ngmres->ldb = msize;
    ierr = PetscMalloc(msize*sizeof(PetscScalar),&ngmres->s);CHKERRQ(ierr);
    ierr = PetscMemzero(ngmres->h,   hsize*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(ngmres->q,   hsize*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(ngmres->xi,  msize*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(ngmres->beta,msize*sizeof(PetscScalar));CHKERRQ(ierr);
    ngmres->lwork = 12*msize;
#if PETSC_USE_COMPLEX
    ierr = PetscMalloc(sizeof(PetscReal)*ngmres->lwork,&ngmres->rwork);
#endif
    ierr = PetscMalloc(sizeof(PetscScalar)*ngmres->lwork,&ngmres->work);
  }

  /* linesearch setup */
  ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);

  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    ierr = SNESLineSearchCreate(((PetscObject)snes)->comm, &ngmres->additive_linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSNES(ngmres->additive_linesearch, snes);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(ngmres->additive_linesearch, SNESLINESEARCHL2);CHKERRQ(ierr);
    ierr = SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch, "additive_");CHKERRQ(ierr);
    ierr = SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch, optionsprefix);CHKERRQ(ierr);
    ierr = SNESLineSearchSetFromOptions(ngmres->additive_linesearch);CHKERRQ(ierr);
  }

  ngmres->setup_called = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NGMRES"
PetscErrorCode SNESSetFromOptions_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscErrorCode ierr;
  PetscBool      debug;
  SNESLineSearch linesearch;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NGMRES options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_ngmres_select_type","Select type","SNESNGMRESSetSelectType",SNESNGMRESSelectTypes,
                          (PetscEnum)ngmres->select_type,(PetscEnum*)&ngmres->select_type,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_ngmres_restart_type","Restart type","SNESNGMRESSetRestartType",SNESNGMRESRestartTypes,
                          (PetscEnum)ngmres->restart_type,(PetscEnum*)&ngmres->restart_type,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_anderson", "Use Anderson mixing storage",        "SNES", ngmres->anderson,  &ngmres->anderson, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_m",         "Number of directions",               "SNES", ngmres->msize,  &ngmres->msize, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_restart",   "Iterations before forced restart",   "SNES", ngmres->restart_periodic,  &ngmres->restart_periodic, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_restart_it","Tolerance iterations before restart","SNES", ngmres->restart_it,  &ngmres->restart_it, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_monitor",  "Monitor actions of NGMRES",          "SNES", ngmres->monitor ? PETSC_TRUE: PETSC_FALSE, &debug, PETSC_NULL);CHKERRQ(ierr);
  if (debug) {
    ngmres->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-snes_ngmres_gammaA",   "Residual selection constant",   "SNES", ngmres->gammaA, &ngmres->gammaA, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_gammaC", "  Residual restart constant",     "SNES", ngmres->gammaC, &ngmres->gammaC, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_epsilonB", "Difference selection constant", "SNES", ngmres->epsilonB, &ngmres->epsilonB, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_deltaB",   "Difference residual selection constant", "SNES", ngmres->deltaB, &ngmres->deltaB, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_single_reduction", "Aggregate reductions",  "SNES", ngmres->singlereduction, &ngmres->singlereduction, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ngmres->gammaA > ngmres->gammaC) && (ngmres->gammaC > 2.)) ngmres->gammaC = ngmres->gammaA;

  /* set the default type of the line search if the user hasn't already. */
  if (!snes->linesearch) {
    ierr = SNESGetSNESLineSearch(snes, &linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_NGMRES"
PetscErrorCode SNESView_NGMRES(SNES snes, PetscViewer viewer)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {

    ierr = PetscViewerASCIIPrintf(viewer, "  Number of stored past updates: %d\n", ngmres->msize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Residual selection: gammaA=%1.0e, gammaC=%1.0e\n", ngmres->gammaA, ngmres->gammaC);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Difference restart: epsilonB=%1.0e, deltaB=%1.0e\n", ngmres->epsilonB, ngmres->deltaB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NGMRES"

PetscErrorCode SNESSolve_NGMRES(SNES snes)
{
  SNES_NGMRES        *ngmres = (SNES_NGMRES *) snes->data;
  /* present solution, residual, and preconditioned residual */
  Vec                 X, F, B, D, Y;

  /* candidate linear combination answers */
  Vec                 XA, FA, XM, FM, FPC;

  /* previous iterations to construct the subspace */
  Vec                 *Fdot = ngmres->Fdot;
  Vec                 *Xdot = ngmres->Xdot;

  /* coefficients and RHS to the minimization problem */
  PetscScalar         *beta = ngmres->beta;
  PetscScalar         *xi   = ngmres->xi;
  PetscReal           fnorm, fMnorm, fAnorm;
  PetscReal           nu;
  PetscScalar         alph_total = 0.;
  PetscInt            i, j, k, k_restart, l, ivec, restart_count = 0;

  /* solution selection data */
  PetscBool           selectA, selectRestart;
  PetscReal           dnorm, dminnorm = 0.0, dcurnorm;
  PetscReal           fminnorm,xnorm,ynorm;

  SNESConvergedReason reason;
  PetscBool           lssucceed,changed_y,changed_w;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* variable initialization */
  snes->reason  = SNES_CONVERGED_ITERATING;
  X             = snes->vec_sol;
  F             = snes->vec_func;
  B             = snes->vec_rhs;
  XA            = snes->vec_sol_update;
  FA            = snes->work[0];
  D             = snes->work[1];

  /* work for the line search */
  Y             = snes->work[2];
  XM            = snes->work[3];
  FM            = snes->work[4];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  /* initialization */

  /* r = F(x) */
  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
  } else {
    snes->vec_func_init_set = PETSC_FALSE;
  }

  if (!snes->norm_init_set) {
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(fnorm)) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_FP, "Infinite or not-a-number generated in function evaluation");
  } else {
    fnorm = snes->norm_init;
    snes->norm_init_set = PETSC_FALSE;
  }
  fminnorm = fnorm;
  /* nu = (r, r) */
  nu = fnorm*fnorm;

  /* q_{00} = nu  */
  Q(0,0) = nu;
  ngmres->fnorms[0] = fnorm;
  /* Fdot[0] = F */
  ierr = VecCopy(X, Xdot[0]);CHKERRQ(ierr);
  ierr = VecCopy(F, Fdot[0]);CHKERRQ(ierr);

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes, fnorm, 0);
  ierr = SNESMonitor(snes, 0, fnorm);CHKERRQ(ierr);
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  k_restart = 1;
  l = 1;
  for (k=1; k < snes->max_its+1; k++) {
    /* select which vector of the stored subspace will be updated */
    ivec = k_restart % ngmres->msize; /* replace the last used part of the subspace */

    /* Computation of x^M */
    if (snes->pc && snes->pcside == PC_RIGHT) {
      ierr = VecCopy(X, XM);CHKERRQ(ierr);
      ierr = SNESSetInitialFunction(snes->pc, F);CHKERRQ(ierr);
      ierr = SNESSetInitialFunctionNorm(snes->pc, fnorm);CHKERRQ(ierr);
      ierr = SNESSolve(snes->pc, B, XM);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = SNESGetFunction(snes->pc, &FPC, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
      ierr = VecCopy(FPC, FM);CHKERRQ(ierr);
      ierr = SNESGetFunctionNorm(snes->pc, &fMnorm);CHKERRQ(ierr);
    } else {
      /* no preconditioner -- just take gradient descent with line search */
      ierr = VecCopy(F, Y);CHKERRQ(ierr);
      ierr = VecCopy(F, FM);CHKERRQ(ierr);
      ierr = VecCopy(X, XM);CHKERRQ(ierr);
      fMnorm = fnorm;
      ierr = SNESLineSearchApply(snes->linesearch,XM,FM,&fMnorm,Y);CHKERRQ(ierr);
      ierr = SNESLineSearchGetSuccess(snes->linesearch, &lssucceed);CHKERRQ(ierr);
      if (!lssucceed) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(0);
        }
      }
    }

    /* r = F(x) */
    nu = fMnorm*fMnorm;
    if (fminnorm > fMnorm) fminnorm = fMnorm;  /* the minimum norm is now of F^M */

    /* construct the right hand side and xi factors */
    ierr = VecMDot(FM, l, Fdot, xi);CHKERRQ(ierr);

    for (i = 0; i < l; i++) {
      beta[i] = nu - xi[i];
    }

    /* construct h */
    for (j = 0; j < l; j++) {
      for (i = 0; i < l; i++) {
        H(i, j) = Q(i, j) - xi[i] - xi[j] + nu;
      }
    }

    if (l == 1) {
      /* simply set alpha[0] = beta[0] / H[0, 0] */
      if (H(0, 0) != 0.) {
        beta[0] = beta[0] / H(0, 0);
      } else {
        beta[0] = 0.;
      }
    } else {
#ifdef PETSC_MISSING_LAPACK_GELSS
      SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_SUP, "NGMRES with LS requires the LAPACK GELSS routine.");
#else
    ngmres->m = PetscBLASIntCast(l);
    ngmres->n = PetscBLASIntCast(l);
    ngmres->info = PetscBLASIntCast(0);
    ngmres->rcond = -1.;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#ifdef PETSC_USE_COMPLEX
    LAPACKgelss_(&ngmres->m,
                 &ngmres->n,
                 &ngmres->nrhs,
                 ngmres->h,
                 &ngmres->lda,
                 ngmres->beta,
                 &ngmres->ldb,
                 ngmres->s,
                 &ngmres->rcond,
                 &ngmres->rank,
                 ngmres->work,
                 &ngmres->lwork,
                 ngmres->rwork,
                 &ngmres->info);
#else
    LAPACKgelss_(&ngmres->m,
                 &ngmres->n,
                 &ngmres->nrhs,
                 ngmres->h,
                 &ngmres->lda,
                 ngmres->beta,
                 &ngmres->ldb,
                 ngmres->s,
                 &ngmres->rcond,
                 &ngmres->rank,
                 ngmres->work,
                 &ngmres->lwork,
                 &ngmres->info);
#endif
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (ngmres->info < 0) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_LIB,"Bad argument to GELSS");
    if (ngmres->info > 0) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_LIB,"SVD failed to converge");
#endif
    }

    for (i=0;i<l;i++) {
      if (PetscIsInfOrNanScalar(beta[i])) {
        SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_LIB,"SVD generated inconsistent output");
      }
    }

    alph_total = 0.;
    for (i = 0; i < l; i++) {
      alph_total += beta[i];
    }

    ierr = VecCopy(XM, XA);CHKERRQ(ierr);
    ierr = VecScale(XA, 1. - alph_total);CHKERRQ(ierr);

    ierr = VecMAXPY(XA, l, beta, Xdot);CHKERRQ(ierr);

    /* check the validity of the step */
    ierr = VecCopy(XA,Y);CHKERRQ(ierr);
    ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr);
    ierr = SNESLineSearchPostCheck(snes->linesearch,X,Y,XA,&changed_y,&changed_w);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, XA, FA);CHKERRQ(ierr);

    /* differences for selection and restart */
    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE || ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
      if (ngmres->singlereduction) {
        dminnorm = -1.0;
        ierr=VecCopy(XA, D);CHKERRQ(ierr);
        ierr=VecAXPY(D, -1.0, XM);CHKERRQ(ierr);
        for (i=0;i<l;i++) {
          ierr=VecAXPY(Xdot[i],-1.0,XA);CHKERRQ(ierr);
        }
        ierr = VecNormBegin(FA, NORM_2, &fAnorm);CHKERRQ(ierr);
        ierr = VecNormBegin(D, NORM_2, &dnorm);CHKERRQ(ierr);
        for (i=0;i<l;i++) {
          ierr = VecNormBegin(Xdot[i], NORM_2, &ngmres->xnorms[i]);CHKERRQ(ierr);
        }
        ierr = VecNormEnd(FA, NORM_2, &fAnorm);CHKERRQ(ierr);
        ierr = VecNormEnd(D, NORM_2, &dnorm);CHKERRQ(ierr);
        for (i=0;i<l;i++) {
          ierr = VecNormEnd(Xdot[i], NORM_2, &ngmres->xnorms[i]);CHKERRQ(ierr);
        }
        for (i=0;i<l;i++) {
          dcurnorm = ngmres->xnorms[i];
          if ((dcurnorm < dminnorm) || (dminnorm < 0.0)) dminnorm = dcurnorm;
          ierr=VecAXPY(Xdot[i],1.0,XA);CHKERRQ(ierr);
        }
      } else {
        ierr=VecCopy(XA, D);CHKERRQ(ierr);
        ierr=VecAXPY(D, -1.0, XM);CHKERRQ(ierr);
        ierr=VecNormBegin(D, NORM_2, &dnorm);CHKERRQ(ierr);
        ierr=VecNormBegin(FA, NORM_2, &fAnorm);CHKERRQ(ierr);
        ierr=VecNormEnd(D, NORM_2, &dnorm);CHKERRQ(ierr);
        ierr=VecNormEnd(FA, NORM_2, &fAnorm);CHKERRQ(ierr);
        dminnorm = -1.0;
        for (i=0;i<l;i++) {
          ierr=VecCopy(XA, D);CHKERRQ(ierr);
          ierr=VecAXPY(D, -1.0, Xdot[i]);CHKERRQ(ierr);
          ierr=VecNorm(D, NORM_2, &dcurnorm);CHKERRQ(ierr);
          if ((dcurnorm < dminnorm) || (dminnorm < 0.0)) dminnorm = dcurnorm;
        }
      }
    } else {
      ierr = VecNorm(FA, NORM_2, &fAnorm);CHKERRQ(ierr);
    }
    if (PetscIsInfOrNanReal(fAnorm)) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_FP, "Infinite or not-a-number generated in function evaluation");
    /* combination (additive) or selection (multiplicative) of the N-GMRES solution */
    if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
      /* X = X + \lambda(XA - X) */
      if (ngmres->monitor) {
        ierr = PetscViewerASCIIPrintf(ngmres->monitor, "||F_A||_2 = %e, ||F_M||_2 = %e\n", fAnorm, fMnorm);CHKERRQ(ierr);
      }
      ierr = VecCopy(FM, F);CHKERRQ(ierr);
      ierr = VecCopy(XM, X);CHKERRQ(ierr);
      ierr = VecCopy(XA, Y);CHKERRQ(ierr);
      ierr = VecAYPX(Y, -1.0, X);CHKERRQ(ierr);
      fnorm = fMnorm;
      ierr = SNESLineSearchApply(ngmres->additive_linesearch,X,F,&fnorm,Y);CHKERRQ(ierr);
      ierr = SNESLineSearchGetSuccess(ngmres->additive_linesearch, &lssucceed);CHKERRQ(ierr);
      if (!lssucceed) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(0);
        }
      }
      if (ngmres->monitor) {
        ierr = PetscViewerASCIIPrintf(ngmres->monitor, "Additive solution: ||F||_2 = %e\n", fnorm);CHKERRQ(ierr);
      }
    } else if (ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
      selectA = PETSC_TRUE;
      /* Conditions for choosing the accelerated answer */
      /* Criterion A -- the norm of the function isn't increased above the minimum by too much */
      if (fAnorm >= ngmres->gammaA*fminnorm) {
        selectA = PETSC_FALSE;
      }
      /* Criterion B -- the choice of x^A isn't too close to some other choice */
      if (ngmres->epsilonB*dnorm<dminnorm || PetscSqrtReal(fnorm)<ngmres->deltaB*PetscSqrtReal(fminnorm)) {
      } else {
        selectA=PETSC_FALSE;
      }
      if (selectA) {
        if (ngmres->monitor) {
          ierr = PetscViewerASCIIPrintf(ngmres->monitor, "picked X_A, ||F_A||_2 = %e, ||F_M||_2 = %e\n", fAnorm, fMnorm);CHKERRQ(ierr);
        }
        /* copy it over */
        fnorm = fAnorm;
        nu = fnorm*fnorm;
        ierr = VecCopy(FA, F);CHKERRQ(ierr);
        ierr = VecCopy(XA, X);CHKERRQ(ierr);
      } else {
        if (ngmres->monitor) {
          ierr = PetscViewerASCIIPrintf(ngmres->monitor, "picked X_M, ||F_A||_2 = %e, ||F_M||_2 = %e\n", fAnorm, fMnorm);CHKERRQ(ierr);
        }
        fnorm = fMnorm;
        nu = fnorm*fnorm;
        ierr = VecCopy(XM, Y);CHKERRQ(ierr);
        ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr);
        ierr = VecCopy(FM, F);CHKERRQ(ierr);
        ierr = VecCopy(XM, X);CHKERRQ(ierr);
      }
    } else { /* none */
      fnorm = fAnorm;
      nu = fnorm*fnorm;
      ierr = VecCopy(FA, F);CHKERRQ(ierr);
      ierr = VecCopy(XA, X);CHKERRQ(ierr);
    }

    selectRestart = PETSC_FALSE;
    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE) {
      /* difference stagnation restart */
      if ((ngmres->epsilonB*dnorm > dminnorm) && (PetscSqrtReal(fAnorm) > ngmres->deltaB*PetscSqrtReal(fminnorm))) {
        if (ngmres->monitor) {
          ierr = PetscViewerASCIIPrintf(ngmres->monitor, "difference restart: %e > %e\n", ngmres->epsilonB*dnorm, dminnorm);CHKERRQ(ierr);
        }
        selectRestart = PETSC_TRUE;
      }
      /* residual stagnation restart */
      if (PetscSqrtReal(fAnorm) > ngmres->gammaC*PetscSqrtReal(fminnorm)) {
        if (ngmres->monitor) {
          ierr = PetscViewerASCIIPrintf(ngmres->monitor, "residual restart: %e > %e\n", PetscSqrtReal(fAnorm), ngmres->gammaC*PetscSqrtReal(fminnorm));CHKERRQ(ierr);
        }
        selectRestart = PETSC_TRUE;
      }
      /* if the restart conditions persist for more than restart_it iterations, restart. */
      if (selectRestart) {
        restart_count++;
      } else {
        restart_count = 0;
      }
    } else if (ngmres->restart_type == SNES_NGMRES_RESTART_PERIODIC) {
      if (k_restart > ngmres->restart_periodic) {
        if (ngmres->monitor) ierr = PetscViewerASCIIPrintf(ngmres->monitor, "periodic restart after %D iterations\n", k_restart);CHKERRQ(ierr);
        restart_count = ngmres->restart_it;
      }
    }
    /* restart after restart conditions have persisted for a fixed number of iterations */
    if (restart_count >= ngmres->restart_it) {
      if (ngmres->monitor){
        ierr = PetscViewerASCIIPrintf(ngmres->monitor, "Restarted at iteration %d\n", k_restart);CHKERRQ(ierr);
      }
      restart_count = 0;
      k_restart = 1;
      l = 1;
      /* q_{00} = nu */
      if (ngmres->anderson) {
        ngmres->fnorms[0] = fMnorm;
        nu = fMnorm*fMnorm;
        Q(0,0) = nu;
        /* Fdot[0] = F */
        ierr = VecCopy(XM, Xdot[0]);CHKERRQ(ierr);
        ierr = VecCopy(FM, Fdot[0]);CHKERRQ(ierr);
      } else {
        ngmres->fnorms[0] = fnorm;
        nu = fnorm*fnorm;
        Q(0,0) = nu;
        /* Fdot[0] = F */
        ierr = VecCopy(X, Xdot[0]);CHKERRQ(ierr);
        ierr = VecCopy(F, Fdot[0]);CHKERRQ(ierr);
      }
    } else {
      /* select the current size of the subspace */
      if (l < ngmres->msize) l++;
      k_restart++;
      /* place the current entry in the list of previous entries */
      if (ngmres->anderson) {
        ierr = VecCopy(FM, Fdot[ivec]);CHKERRQ(ierr);
        ierr = VecCopy(XM, Xdot[ivec]);CHKERRQ(ierr);
        ngmres->fnorms[ivec] = fMnorm;
        if (fminnorm > fMnorm) fminnorm = fMnorm;  /* the minimum norm is now of FM */
        xi[ivec] = fMnorm*fMnorm;
        for (i = 0; i < l; i++) {
          Q(i, ivec) = xi[i];
          Q(ivec, i) = xi[i];
        }
      } else {
        ierr = VecCopy(F, Fdot[ivec]);CHKERRQ(ierr);
        ierr = VecCopy(X, Xdot[ivec]);CHKERRQ(ierr);
        ngmres->fnorms[ivec] = fnorm;
        if (fminnorm > fnorm) fminnorm = fnorm;  /* the minimum norm is now of FA */
        ierr = VecMDot(F, l, Fdot, xi);CHKERRQ(ierr);
        for (i = 0; i < l; i++) {
          Q(i, ivec) = xi[i];
          Q(ivec, i) = xi[i];
        }
      }
    }

    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = k;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes, snes->norm, snes->iter);
    ierr = SNESMonitor(snes, snes->iter, snes->norm);CHKERRQ(ierr);
    ierr = VecNormBegin(Y,NORM_2,&ynorm);CHKERRQ(ierr);
    ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);
    ierr = VecNormEnd(Y,NORM_2,&ynorm);CHKERRQ(ierr);
    ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESSetRestartType"
/*@
    SNESNGMRESSetRestartType - Sets the restart type for SNESNGMRES.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   rtype - restart type

    Options Database:
+   -snes_ngmres_restart_type<difference,periodic,none> - set the restart type
-   -snes_ngmres_restart[30] - sets the number of iterations before restart for periodic

    Level: intermediate

    SNESNGMRESRestartTypes:
+   SNES_NGMRES_RESTART_NONE - never restart
.   SNES_NGMRES_RESTART_DIFFERENCE - restart based upon difference criteria
-   SNES_NGMRES_RESTART_PERIODIC - restart after a fixed number of iterations

    Notes:
    The default line search used is the L2 line search and it requires two additional function evaluations.

.keywords: SNES, SNESNGMRES, restart, type, set SNESLineSearch
@*/
PetscErrorCode SNESNGMRESSetRestartType(SNES snes, SNESNGMRESRestartType rtype) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESNGMRESSetRestartType_C",(SNES,SNESNGMRESRestartType),(snes,rtype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESSetSelectType"
/*@
    SNESNGMRESSetSelectType - Sets the selection type for SNESNGMRES.  This determines how the candidate solution and
    combined solution are used to create the next iterate.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   stype - selection type

    Options Database:
.   -snes_ngmres_select_type<difference,none,linesearch>

    Level: intermediate

    SNESNGMRESSelectTypes:
+   SNES_NGMRES_SELECT_NONE - choose the combined solution all the time
.   SNES_NGMRES_SELECT_DIFFERENCE - choose based upon the selection criteria
-   SNES_NGMRES_SELECT_LINESEARCH - choose based upon line search combination

    Notes:
    The default line search used is the L2 line search and it requires two additional function evaluations.

.keywords: SNES, SNESNGMRES, selection, type, set SNESLineSearch
@*/

PetscErrorCode SNESNGMRESSetSelectType(SNES snes, SNESNGMRESSelectType stype) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESNGMRESSetSelectType_C",(SNES,SNESNGMRESSelectType),(snes,stype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESSetSelectType_NGMRES"

PetscErrorCode SNESNGMRESSetSelectType_NGMRES(SNES snes, SNESNGMRESSelectType stype) {
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  PetscFunctionBegin;
  ngmres->select_type = stype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESSetRestartType_NGMRES"

PetscErrorCode SNESNGMRESSetRestartType_NGMRES(SNES snes, SNESNGMRESRestartType rtype) {
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  PetscFunctionBegin;
  ngmres->restart_type = rtype;
  PetscFunctionReturn(0);
}
EXTERN_C_END


/*MC
  SNESNGMRES - The Nonlinear Generalized Minimum Residual method.

   Level: beginner

   Options Database:
+  -snes_ngmres_select_type<difference,none,linesearch> - choose the select between candidate and combined solution
+  -snes_ngmres_restart_type<difference,none,periodic> - choose the restart conditions
.  -snes_ngmres_anderson         - Use Anderson mixing NGMRES variant which combines candidate solutions instead of actual solutions
.  -snes_ngmres_m                - Number of stored previous solutions and residuals
.  -snes_ngmres_restart_it       - Number of iterations the restart conditions hold before restart
.  -snes_ngmres_gammaA           - Residual tolerance for solution select between the candidate and combination
.  -snes_ngmres_gammaC           - Residual tolerance for restart
.  -snes_ngmres_epsilonB         - Difference tolerance between subsequent solutions triggering restart
.  -snes_ngmres_deltaB           - Difference tolerance between residuals triggering restart
.  -snes_ngmres_monitor          - Prints relevant information about the ngmres iteration
.  -snes_linesearch_type <basic,basicnonorms,quadratic,critical> - Line search type used for the default smoother
-  -additive_snes_linesearch_type - linesearch type used to select between the candidate and combined solution with additive select type

   Notes:

   The N-GMRES method combines m previous solutions into a minimum-residual solution by solving a small linearized
   optimization problem at each iteration.

   References:

   "Krylov Subspace Acceleration of Nonlinear Multigrid with Application to Recirculating Flows", C. W. Oosterlee and T. Washio,
   SIAM Journal on Scientific Computing, 21(5), 2000.

   This is also the same as the algorithm called Anderson acceleration introduced in "D. G. Anderson. Iterative procedures for nonlinear integral equations.
   J. Assoc. Comput. Mach., 12:547–560, 1965."

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NGMRES"
PetscErrorCode SNESCreate_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NGMRES;
  snes->ops->setup          = SNESSetUp_NGMRES;
  snes->ops->setfromoptions = SNESSetFromOptions_NGMRES;
  snes->ops->view           = SNESView_NGMRES;
  snes->ops->solve          = SNESSolve_NGMRES;
  snes->ops->reset          = SNESReset_NGMRES;

  snes->usespc          = PETSC_TRUE;
  snes->usesksp         = PETSC_FALSE;

  ierr = PetscNewLog(snes, SNES_NGMRES, &ngmres);CHKERRQ(ierr);
  snes->data = (void*) ngmres;
  ngmres->msize = 30;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  ngmres->anderson   = PETSC_FALSE;

  ngmres->additive_linesearch = PETSC_NULL;

  ngmres->restart_it = 2;
  ngmres->restart_periodic = 30;
  ngmres->gammaA     = 2.0;
  ngmres->gammaC     = 2.0;
  ngmres->deltaB     = 0.9;
  ngmres->epsilonB   = 0.1;

  ngmres->restart_type   = SNES_NGMRES_RESTART_DIFFERENCE;
  ngmres->select_type    = SNES_NGMRES_SELECT_DIFFERENCE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESNGMRESSetSelectType_C","SNESNGMRESSetSelectType_NGMRES", SNESNGMRESSetSelectType_NGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESNGMRESSetRestartType_C","SNESNGMRESSetRestartType_NGMRES", SNESNGMRESSetRestartType_NGMRES);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
