/* Defines the basic SNES object */
#include <../src/snes/impls/ngmres/snesngmres.h>
#include <petscblaslapack.h>




#undef __FUNCT__
#define __FUNCT__ "SNESReset_NGMRES"
PetscErrorCode SNESReset_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->Fdot);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->Xdot);CHKERRQ(ierr);
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork, &snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NGMRES"
PetscErrorCode SNESDestroy_NGMRES(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NGMRES(snes);CHKERRQ(ierr);
  if (snes->data) {
    SNES_NGMRES * ngmres = (SNES_NGMRES *)snes->data;
    ierr = PetscFree5(ngmres->h, ngmres->beta, ngmres->xi, ngmres->fnorms, ngmres->q);CHKERRQ(ierr);
    ierr = PetscFree(ngmres->s);CHKERRQ(ierr);
#if PETSC_USE_COMPLEX
    ierr = PetscFree(ngmres->rwork);
#endif
    ierr = PetscFree(ngmres->work);
  }
  ierr = PetscFree(snes->data);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NGMRES"
PetscErrorCode SNESSetUp_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES *) snes->data;
  PetscInt       msize,hsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  msize         = ngmres->msize;  /* restart size */
  hsize         = msize * msize;


  /* explicit least squares minimization solve */
  ierr = PetscMalloc5(hsize,PetscScalar,&ngmres->h,
                      msize,PetscScalar,&ngmres->beta,
                      msize,PetscScalar,&ngmres->xi,
                      msize,PetscReal,  &ngmres->fnorms,
                      hsize,PetscScalar,&ngmres->q);CHKERRQ(ierr);
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

  ierr = VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->Xdot);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->Fdot);CHKERRQ(ierr);
  ierr = SNESDefaultGetWork(snes, 5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NGMRES"
PetscErrorCode SNESSetFromOptions_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscErrorCode ierr;
  PetscBool      debug;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NGMRES options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_m",         "Number of directions",               "SNES", ngmres->msize,  &ngmres->msize, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_restart_it","Tolerance iterations before restart","SNES", ngmres->restart_it,  &ngmres->restart_it, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_monitor",  "Monitor actions of NGMRES",          "SNES", ngmres->monitor ? PETSC_TRUE: PETSC_FALSE, &debug, PETSC_NULL);CHKERRQ(ierr);
  if (debug) {
    ngmres->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-snes_ngmres_gammaA",   "Residual selection constant",   "SNES", ngmres->gammaA, &ngmres->gammaA, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_gammaC", "  Residual restart constant",     "SNES", ngmres->gammaC, &ngmres->gammaC, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_epsilonB", "Difference selection constant", "SNES", ngmres->epsilonB, &ngmres->epsilonB, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_deltaB",   "Difference residual selection constant", "SNES", ngmres->deltaB, &ngmres->deltaB, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ngmres->gammaA > ngmres->gammaC) && (ngmres->gammaC > 2.)) ngmres->gammaC = ngmres->gammaA;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_NGMRES"
PetscErrorCode SNESView_NGMRES(SNES snes, PetscViewer viewer)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscBool      iascii;
  const char     *cstr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    cstr = SNESLineSearchTypeName(snes->ls_type);
    ierr = PetscViewerASCIIPrintf(viewer, "  line search variant: %s\n",cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Number of stored past updates: %d\n", ngmres->msize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Residual selection: gammaA=%1.0e, gammaC=%1.0e\n", ngmres->gammaA, ngmres->gammaC);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Difference restart: epsilonB=%1.0e, deltaB=%1.0e\n", ngmres->epsilonB, ngmres->deltaB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType_NGMRES"
PetscErrorCode  SNESLineSearchSetType_NGMRES(SNES snes, SNESLineSearchType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (type) {
  case SNES_LS_BASIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNo,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_BASIC_NONORMS:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_QUADRATIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchQuadraticSecant,PETSC_NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"Unknown line search type.");
    break;
  }
  snes->ls_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NGMRES"

PetscErrorCode SNESSolve_NGMRES(SNES snes)
{
  SNES_NGMRES        *ngmres = (SNES_NGMRES *) snes->data;
  /* present solution, residual, and preconditioned residual */
  Vec                 X, F, B, D, G, W, Y;

  /* candidate linear combination answers */
  Vec                 XA, FA;

  /* previous iterations to construct the subspace */
  Vec                 *Fdot = ngmres->Fdot;
  Vec                 *Xdot = ngmres->Xdot;

  /* coefficients and RHS to the minimization problem */
  PetscScalar         *beta = ngmres->beta;
  PetscScalar         *xi   = ngmres->xi;
  PetscReal           fnorm, fAnorm, gnorm, ynorm, xnorm = 0.0;
  PetscReal           nu;
  PetscScalar         alph_total = 0.;
  PetscScalar         qentry;
  PetscInt            i, j, k, k_restart, l, ivec, restart_count = 0;

  /* solution selection data */
  PetscBool           selectA, selectRestart;
  PetscReal           dnorm, dminnorm, dcurnorm;
  PetscReal           fminnorm;

  SNESConvergedReason reason;
  PetscBool           lssucceed;
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
  G             = snes->work[3];
  W             = snes->work[4];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  /* initialization */

  /* r = F(x) */
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }

  /* nu = (r, r) */
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
  fminnorm = fnorm;
  nu = fnorm*fnorm;
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP, "Infinite or not-a-number generated in function evaluation");

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
  for (k=1; k<snes->max_its; k++) {
    /* select which vector of the stored subspace will be updated */
    ivec = k_restart % ngmres->msize; /* replace the last used part of the subspace */

    /* Computation of x^M */
    if (!snes->pc) {
      /* no preconditioner -- just take gradient descent with line search */
      ierr = VecCopy(F, Y);CHKERRQ(ierr);
      ierr = VecScale(Y, -1.0);CHKERRQ(ierr);
      ierr = (*snes->ops->linesearch)(snes,snes->lsP,X,F,Y,fnorm,xnorm,G,W,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
      if (!lssucceed) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(0);
        }
      }
      fnorm = gnorm;
      ierr = VecCopy(G, F);CHKERRQ(ierr);
      ierr = VecCopy(W, X);CHKERRQ(ierr);
    } else {
      ierr = SNESSolve(snes->pc, B, X);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    }

    /* r = F(x) */
    nu = fnorm*fnorm;
    if (fminnorm > fnorm) fminnorm = fnorm;  /* the minimum norm is now of F^M */

    /* construct the right hand side and xi factors */
    for (i = 0; i < l; i++) {
      ierr = VecDot(Fdot[i], F, &xi[i]);CHKERRQ(ierr);
      beta[i] = nu - xi[i];
    }

    /* construct h */
    for (j = 0; j < l; j++) {
      for (i = 0; i < l; i++) {
        H(i, j) = Q(i, j) - xi[i] - xi[j] + nu;
      }
    }

    if(l == 1) {
      /* simply set alpha[0] = beta[0] / H[0, 0] */
      beta[0] = beta[0] / H(0, 0);
    } else {
#ifdef PETSC_MISSING_LAPACK_GELSS
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "NGMRES with LS requires the LAPACK GELSS routine.");
#else
    ngmres->m = PetscBLASIntCast(l);
    ngmres->n = PetscBLASIntCast(l);
    ngmres->info = PetscBLASIntCast(0);
    ngmres->rcond = -1.;
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
    if (ngmres->info < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GELSS");
    if (ngmres->info > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SVD failed to converge");
#endif
    }

    alph_total = 0.;
    for (i = 0; i < l; i++) {
      alph_total += beta[i];
    }

    ierr = VecCopy(X, XA);CHKERRQ(ierr);
    ierr = VecScale(XA, 1. - alph_total);CHKERRQ(ierr);

    for(i=0;i<l;i++){
      ierr= VecAXPY(XA, beta[i], Xdot[i]);CHKERRQ(ierr);
    }
    ierr = SNESComputeFunction(snes, XA, FA);CHKERRQ(ierr);
    ierr = VecNorm(FA, NORM_2, &fAnorm);CHKERRQ(ierr);

    selectA = PETSC_TRUE;
    /* Conditions for choosing the accelerated answer */

    /* Criterion A -- the norm of the function isn't increased above the minimum by too much */
    if (fAnorm >= ngmres->gammaA*fminnorm) {
      selectA = PETSC_FALSE;
    }

    /* Criterion B -- the choice of x^A isn't too close to some other choice */
    ierr=VecCopy(XA, D);CHKERRQ(ierr);
    ierr=VecAXPY(D, -1.0, X);CHKERRQ(ierr);
    ierr=VecNorm(D, NORM_2, &dnorm);CHKERRQ(ierr);
    dminnorm = -1.0;
    for(i=0;i<l;i++) {
      ierr=VecCopy(XA, D);CHKERRQ(ierr);
      ierr=VecAXPY(D, -1.0, Xdot[i]);CHKERRQ(ierr);
      ierr=VecNorm(D, NORM_2, &dcurnorm);CHKERRQ(ierr);
      if((dcurnorm < dminnorm) || (dminnorm < 0.0)) dminnorm = dcurnorm;
    }
    if (ngmres->epsilonB*dnorm<dminnorm || PetscSqrtReal(fnorm)<ngmres->deltaB*PetscSqrtReal(fminnorm)) {
    } else {
      selectA=PETSC_FALSE;
    }


    if (selectA) {
      if (ngmres->monitor) {
        ierr = PetscViewerASCIIPrintf(ngmres->monitor, "picked X_A, ||F_A||_2 = %e, ||F_M||_2 = %e\n", fAnorm, fnorm);CHKERRQ(ierr);
      }
      /* copy it over */
      fnorm = fAnorm;
      nu = fnorm*fnorm;
      ierr = VecCopy(FA, F);CHKERRQ(ierr);
      ierr = VecCopy(XA, X);CHKERRQ(ierr);
    } else {
      if (ngmres->monitor) {
        ierr = PetscViewerASCIIPrintf(ngmres->monitor, "picked X_M, ||F_A||_2 = %e, ||F_M||_2 = %e\n", fAnorm, fnorm);CHKERRQ(ierr);
      }
    }

    selectRestart = PETSC_FALSE;

    /* difference stagnation restart */
    if((ngmres->epsilonB*dnorm > dminnorm) && (PetscSqrtReal(fAnorm) > ngmres->deltaB*PetscSqrtReal(fminnorm))) {
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

    /* restart after restart conditions have persisted for a fixed number of iterations */
    if (restart_count >= ngmres->restart_it) {
      if (ngmres->monitor){
        ierr = PetscViewerASCIIPrintf(ngmres->monitor, "Restarted at iteration %d\n", k_restart);CHKERRQ(ierr);
      }
      restart_count = 0;
      k_restart = 1;
      l = 1;
      /* q_{00} = nu */
      ngmres->fnorms[0] = fnorm;
      nu = fnorm*fnorm;
      Q(0,0) = nu;
      /* Fdot[0] = F */
      ierr = VecCopy(X, Xdot[0]);CHKERRQ(ierr);
      ierr = VecCopy(F, Fdot[0]);CHKERRQ(ierr);
    } else {
      /* select the current size of the subspace */
      if (l < ngmres->msize) l++;
      k_restart++;
      /* place the current entry in the list of previous entries */
      ierr = VecCopy(F, Fdot[ivec]);CHKERRQ(ierr);
      ierr = VecCopy(X, Xdot[ivec]);CHKERRQ(ierr);
      ngmres->fnorms[ivec] = fnorm;
      if (fminnorm > fnorm) fminnorm = fnorm;  /* the minimum norm is now of r^A */
      for (i = 0; i < l; i++) {
        ierr = VecDot(F, Fdot[i], &qentry);CHKERRQ(ierr);
        Q(i, ivec) = qentry;
        Q(ivec, i) = qentry;
      }
    }

    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = k;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes, snes->norm, snes->iter);
    ierr = SNESMonitor(snes, snes->iter, snes->norm);CHKERRQ(ierr);

    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*MC
  SNESNGMRES - The Nonlinear Generalized Minimum Residual (NGMRES) method of Oosterlee and Washio.

   Level: beginner

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
  ngmres->msize = 10;

  ngmres->restart_it = 2;
  ngmres->gammaA     = 2.0;
  ngmres->gammaC     = 2.0;
  ngmres->deltaB     = 0.9;
  ngmres->epsilonB   = 0.1;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetType_C","SNESLineSearchSetType_NGMRES",SNESLineSearchSetType_NGMRES);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(snes, SNES_LS_QUADRATIC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
