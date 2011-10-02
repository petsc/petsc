#include <private/snesimpl.h>

typedef struct {
  Vec * dX;           /* The change in X */
  Vec * dF;           /* The change in F */
  PetscInt m;         /* the number of kept previous steps */
  PetscScalar * alpha;
  PetscScalar * beta;
  PetscScalar * rho;
} QNContext;

#undef __FUNCT__
#define __FUNCT__ "LBGFSApplyJinv_Private"
PetscErrorCode LBGFSApplyJinv_Private(SNES snes, PetscInt it, Vec g, Vec z) {

  PetscErrorCode ierr;
  
  QNContext * qn = (QNContext *)snes->data;

  Vec * dX = qn->dX;
  Vec * dF = qn->dF;

  PetscScalar * alpha = qn->alpha;
  PetscScalar * beta = qn->beta;
  PetscScalar * rho = qn->rho;

  PetscInt k, i;
  PetscInt m = qn->m;
  PetscScalar t;
  PetscInt l = m;

  PetscFunctionBegin;

  if (it < m) l = it;

  ierr = VecCopy(g, z);CHKERRQ(ierr);

  /* outward recursion starting at iteration k's update and working back */
  for (i = 0; i < l; i++) {
    k = (it - i - 1) % m;
    /* k = (it + i - l) % m; */ 
    ierr = VecDot(dX[k], z, &t);CHKERRQ(ierr);
    alpha[k] = t*rho[k];
    ierr = VecAXPY(z, -alpha[k], dF[k]);CHKERRQ(ierr);
  }

  /* inner application of the initial inverse jacobian approximation */
  /* right now it's just the identity. Nothing needs to go here. */

  /* inward recursion starting at the first update and working forward*/
  for (i = 0; i < l; i++) {
    /* k = (it - i - 1) % m; */ 
    k = (it + i - l) % m;
    ierr = VecDot(dF[k], z, &t);CHKERRQ(ierr);
    beta[k] = rho[k]*t;
    ierr = VecAXPY(z, (alpha[k] - beta[k]), dX[k]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_QN"
static PetscErrorCode SNESSolve_QN(SNES snes)
{

  PetscErrorCode ierr;
  QNContext * qn = (QNContext*) snes->data;

  Vec X, Xold;
  Vec F, Fold, G;
  Vec W, Y;

  PetscInt i, k;

  PetscReal fnorm, xnorm = 0, ynorm, gnorm;
  PetscInt m = qn->m;
  PetscBool lssucceed;

  PetscScalar rhosc;
  
  Vec * dX = qn->dX;
  Vec * dF = qn->dF;
  PetscScalar * rho = qn->rho;

  /* basically just a regular newton's method except for the application of the jacobian */
  PetscFunctionBegin;

  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];
  Xold          = snes->work[3];
  Fold          = snes->work[4];

  snes->reason = SNES_CONVERGED_ITERATING;

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
   snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);
  ierr = VecCopy(F, Fold);CHKERRQ(ierr);
  ierr = VecCopy(X, Xold);CHKERRQ(ierr);
  for(i = 0; i < snes->max_its; i++) {
    /* general purpose update */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /* apply the current iteration of the approximate jacobian */
    ierr = LBGFSApplyJinv_Private(snes, i, F, Y);CHKERRQ(ierr);

    /* line search for lambda */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = fnorm;
    ierr = (*snes->ops->linesearch)(snes,snes->lsP,X,F,Y,fnorm,xnorm,G,W,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
         snes->reason = SNES_DIVERGED_LINE_SEARCH;
        /* PetscBool  ismin;
         ierr = SNESLSCheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
         */
        break;
      }
    }
    /* Update function and solution vectors */
    fnorm = gnorm;
    ierr = VecCopy(G,F);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);

    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,snes->iter);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* set parameter for default relative tolerance convergence test */
    ierr = (*snes->ops->converged)(snes,i+1,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);

    /* set the differences */
    k = i % m;
    ierr = VecCopy(F, dF[k]);CHKERRQ(ierr);
    ierr = VecAXPY(dF[k], -1.0, Fold);CHKERRQ(ierr);
    ierr = VecCopy(X, dX[k]);CHKERRQ(ierr);
    ierr = VecAXPY(dX[k], -1.0, Xold);CHKERRQ(ierr);
    ierr = VecDot(dX[k], dF[k], &rhosc);CHKERRQ(ierr);
    rho[k] = 1. / rhosc;
    ierr = VecCopy(F, Fold);CHKERRQ(ierr);
    ierr = VecCopy(X, Xold);CHKERRQ(ierr);
  }
  if (i == snes->max_its) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", snes->max_its);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_QN"
static PetscErrorCode SNESSetUp_QN(SNES snes)
{
  QNContext * qn = (QNContext *)snes->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDuplicateVecs(snes->vec_sol, qn->m, &qn->dX);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(snes->vec_sol, qn->m, &qn->dF);CHKERRQ(ierr);
  ierr = PetscMalloc3(qn->m, PetscScalar, &qn->alpha, qn->m, PetscScalar, &qn->beta, qn->m, PetscScalar, &qn->rho);CHKERRQ(ierr);
  ierr = SNESDefaultGetWork(snes,5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESReset_QN"
static PetscErrorCode SNESReset_QN(SNES snes)
{
  PetscErrorCode ierr;
  QNContext * qn;
  PetscFunctionBegin;
  if (snes->data) {
    qn = (QNContext *)snes->data;
    if (qn->dX) {
      ierr = VecDestroyVecs(qn->m, &qn->dX);CHKERRQ(ierr);
    }
    if (qn->dF) {
      ierr = VecDestroyVecs(qn->m, &qn->dF);CHKERRQ(ierr);
    }
    ierr = PetscFree3(qn->alpha, qn->beta, qn->rho);CHKERRQ(ierr);
  }
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_QN"
static PetscErrorCode SNESDestroy_QN(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESReset_QN(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_QN"
static PetscErrorCode SNESSetFromOptions_QN(SNES snes)
{

  PetscErrorCode ierr;
  QNContext * qn;

  PetscFunctionBegin;

  qn = (QNContext *)snes->data;

  ierr = PetscOptionsHead("SNES QN options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_qn_m", "Number of past states saved for L-Broyden methods", "SNES", qn->m, &qn->m, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESQN - Limited-Memory Quasi-Newton methods for the solution of nonlinear systems.

      Options Database:

+     -snes_qn_m - Number of past states saved for the L-Broyden methods.

      Notes: This implements the L-BFGS algorithm for the solution of F(x) = 0 using previous change in F(x) and x to
      form the approximate inverse Jacobian using a series of multiplicative rank-one updates.  This will eventually be
      generalized to implement several limited-memory Broyden methods.

      References:
      
      L-Broyden Methods: a generalization of the L-BFGS method to the limited memory Broyden family, M. B. Reed,
      International Journal of Computer Mathematics, vol. 86, 2009.
      

      Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_QN"
PetscErrorCode  SNESCreate_QN(SNES snes)
{
  
  PetscErrorCode ierr;
  QNContext * qn;  

  PetscFunctionBegin;
  snes->ops->setup           = SNESSetUp_QN;
  snes->ops->solve           = SNESSolve_QN;
  snes->ops->destroy         = SNESDestroy_QN;
  snes->ops->setfromoptions  = SNESSetFromOptions_QN;
  snes->ops->view            = 0;
  snes->ops->reset           = SNESReset_QN;

  snes->usespc          = PETSC_TRUE;
  snes->usesksp         = PETSC_FALSE;

  ierr = PetscNewLog(snes, QNContext, &qn);CHKERRQ(ierr);
  snes->data = (void *) qn;
  qn->m = 100;
  qn->dX = PETSC_NULL;
  qn->dF = PETSC_NULL;

  snes->ops->linesearchcubic     = SNESLineSearchCubic;
  snes->ops->linesearchquadratic = SNESLineSearchQuadratic;
  snes->ops->linesearchno        = SNESLineSearchNo;
  snes->ops->linesearchnonorms   = SNESLineSearchNoNorms;

  ierr = SNESLineSearchSetType(snes, SNES_LS_CUBIC);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
