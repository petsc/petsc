
#include <private/snesimpl.h>

typedef struct {
  PetscReal lambda;   /* The default step length for the update */
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
  ierr = VecScale(z, 1.0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QNLineSearchQuadratic"
PetscErrorCode QNLineSearchQuadratic(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec G,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscInt       i;
  PetscReal      alphas[3] = {0.0, 0.5, 1.0};
  PetscReal      norms[3];
  PetscReal      alpha,a,b;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  norms[0]  = fnorm;
  /* Calculate trial solutions */
  for(i=1; i < 3; ++i) {
    /* Calculate X^{n+1} = (1 - \alpha) X^n + \alpha Y */
    ierr = VecCopy(X, W);CHKERRQ(ierr);
    ierr = VecAXPBY(W, alphas[i], 1 - alphas[i], Y);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &norms[i]);CHKERRQ(ierr);
  }
  for(i = 0; i < 3; ++i) {
    norms[i] = PetscSqr(norms[i]);
  }
  /* Fit a quadratic:
       If we have x_{0,1,2} = 0, x_1, x_2 which generate norms y_{0,1,2}
       a = (x_1 y_2 - x_2 y_1 + (x_2 - x_1) y_0)/(x^2_2 x_1 - x_2 x^2_1)
       b = (x^2_1 y_2 - x^2_2 y_1 + (x^2_2 - x^2_1) y_0)/(x_2 x^2_1 - x^2_2 x_1)
       c = y_0
       x_min = -b/2a

       If we let x_{0,1,2} = 0, 0.5, 1.0
       a = 2 y_2 - 4 y_1 + 2 y_0
       b =  -y_2 + 4 y_1 - 3 y_0
       c =   y_0
  */
  a = (alphas[1]*norms[2] - alphas[2]*norms[1] + (alphas[2] - alphas[1])*norms[0])/(PetscSqr(alphas[2])*alphas[1] - alphas[2]*PetscSqr(alphas[1]));
  b = (PetscSqr(alphas[1])*norms[2] - PetscSqr(alphas[2])*norms[1] + (PetscSqr(alphas[2]) - PetscSqr(alphas[1]))*norms[0])/(alphas[2]*PetscSqr(alphas[1]) - PetscSqr(alphas[2])*alphas[1]);
  /* Check for positive a (concave up) */
  if (a >= 0.0) {
    alpha = -b/(2.0*a);
    alpha = PetscMin(alpha, alphas[2]);
    alpha = PetscMax(alpha, alphas[0]);
  } else {
    alpha = 1.0;
  }
  ierr = VecAXPBY(X, alpha, 1 - alpha, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  if (alpha != 1.0) {
    ierr = VecNorm(F, NORM_2, gnorm);CHKERRQ(ierr);
  } else {
    *gnorm = PetscSqrtReal(norms[2]); 
  }
  *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_QN"
static PetscErrorCode SNESSolve_QN(SNES snes)
{

  PetscErrorCode ierr;
  QNContext * qn = (QNContext*) snes->data;

  Vec x, xold;
  Vec f, fold;
  Vec w, y;

  PetscInt i, k;

  PetscReal fnorm, xnorm;
  PetscInt m = qn->m;
  PetscBool ls_OK;

  PetscScalar rhosc;
  
  Vec * dX = qn->dX;
  Vec * dF = qn->dF;
  PetscScalar * rho = qn->rho;

  /* basically just a regular newton's method except for the application of the jacobian */
  PetscFunctionBegin;

  x = snes->vec_sol;
  xold = snes->vec_sol_update; /* dX_k */
  w = snes->work[1];
  f = snes->vec_func;
  fold = snes->work[0];
  y = snes->work[2];

  snes->reason = SNES_CONVERGED_ITERATING;

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(f, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
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
  ierr = VecCopy(f, fold);CHKERRQ(ierr);
  ierr = VecCopy(x, xold);CHKERRQ(ierr);
  for(i = 0; i < snes->max_its; i++) {
    /* general purpose update */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /* apply the current iteration of the approximate jacobian */
    ierr = LBGFSApplyJinv_Private(snes, i, f, y);CHKERRQ(ierr);

    /* line search for lambda */
    ierr = VecAYPX(y,-1.0,x);CHKERRQ(ierr);
    ierr = QNLineSearchQuadratic(snes, PETSC_NULL, x, f, y, fnorm, xnorm, 0, w,&xnorm, &fnorm, &ls_OK);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, x, f);CHKERRQ(ierr);
    ierr = VecNorm(f, NORM_2, &fnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,fnorm,i+1);
    ierr = SNESMonitor(snes,i+1,fnorm);CHKERRQ(ierr);  
    /* set parameter for default relative tolerance convergence test */
    ierr = (*snes->ops->converged)(snes,i+1,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);

    /* set the differences */
    k = i % m;
    ierr = VecCopy(f, dF[k]);CHKERRQ(ierr);
    ierr = VecAXPY(dF[k], -1.0, fold);CHKERRQ(ierr);
    ierr = VecCopy(x, dX[k]);CHKERRQ(ierr);
    ierr = VecAXPY(dX[k], -1.0, xold);CHKERRQ(ierr);
    ierr = VecDot(dX[k], dF[k], &rhosc);CHKERRQ(ierr);
    rho[k] = 1. / rhosc;
    ierr = VecCopy(f, fold);CHKERRQ(ierr);
    ierr = VecCopy(x, xold);CHKERRQ(ierr);
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
  ierr = SNESDefaultGetWork(snes,3);CHKERRQ(ierr);
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
  ierr = PetscOptionsReal("-snes_ls_damping", "Damping parameter", "SNES", qn->lambda, &qn->lambda, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_qn_m", "Number of past states saved for L-Broyden methods", "SNES", qn->m, &qn->m, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESQN - Limited-Memory Quasi-Newton methods for the solution of nonlinear systems.

      Options Database:

+     -snes_qn_m - Number of past states saved for the L-Broyden methods.
+     -snes_ls_damping - The damping parameter on the update.

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
  qn->lambda = 1.;
  qn->dX = PETSC_NULL;
  qn->dF = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
