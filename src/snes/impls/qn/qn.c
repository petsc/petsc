
#include <private/snesimpl.h>

typedef struct {
  PetscScalar lambda;   /* The default step length for the update */
  Vec * dX;  /* The change in X */
  Vec * dF;  /* The change in F */
  PetscInt m;     /* the number of kept previous steps */
} QNContext;

#undef __FUNCT__
#define __FUNCT__ "LBGFSApplyJinv_Private"
PetscErrorCode LBGFSApplyJinv_Private(SNES snes, PetscInt it, Vec g, Vec z) {

  PetscErrorCode ierr;
  
  QNContext * qn = (QNContext *)snes->data;

  Vec * dX = qn->dX;
  Vec * dF = qn->dF;

  PetscInt k, i;
  PetscInt m = qn->m;
  PetscScalar * alpha = PETSC_NULL;
  PetscScalar * beta = PETSC_NULL;
  PetscScalar * rho = PETSC_NULL;
  PetscScalar t;
  PetscInt l = m;

  PetscFunctionBegin;

  if (it < m) l = it+1;
  ierr = PetscMalloc3(m, PetscScalar, &alpha, m, PetscScalar, &beta, m, PetscScalar, &rho);CHKERRQ(ierr);

  /* precalculate alpha, beta, rho corresponding to the normal indices*/
  for (i = 0; i < l; i++) {
    ierr = VecDot(dX[i], dF[i], &t);CHKERRQ(ierr);
    rho[i] = 1. / t;
    beta[i] = 0;
    alpha[i] = 0;
  }

  ierr = VecCopy(g, z);CHKERRQ(ierr);

  /* outward recursion starting at iteration k's update and working back */
  for (i = 0; i < l; i++) {
    k = (it - i) % m; 
    ierr = VecDot(dX[k], z, &t);CHKERRQ(ierr);
    alpha[k] = t / rho[k];
    ierr = PetscPrintf(PETSC_COMM_WORLD, " %d: %e ", k, -alpha[k]);CHKERRQ(ierr);
    ierr = VecAXPY(z, -alpha[k], dF[k]);CHKERRQ(ierr);
  }

  /* inner application of the initial inverse jacobian approximation */
  /* right now it's just the identity. Nothing needs to go here. */

  /* inward recursion starting at the first update and working forward*/
  for (i = l - 1; i >= 0; i--) {
    k = (it - i) % m;
    ierr = VecDot(dF[k], z, &t);CHKERRQ(ierr);
    beta[k] = rho[k]*t;
    ierr = PetscPrintf(PETSC_COMM_WORLD, " %d: %e %e", k, alpha[k] - beta[k], rho[k]);CHKERRQ(ierr);
    ierr = VecAXPY(z, (alpha[k] - beta[k]), dX[k]);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\n", k);CHKERRQ(ierr);
  ierr = PetscFree3(alpha, beta, rho);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_QN"
static PetscErrorCode SNESSolve_QN(SNES snes)
{

  PetscErrorCode ierr;
  QNContext * qn = (QNContext*) snes->data;

  Vec x;
  Vec f;
  Vec p, pold;

  PetscInt i, j, k, l;

  PetscReal fnorm;
  PetscScalar gdot;
  PetscInt m = qn->m;
  
  Vec * W = qn->dX;
  Vec * V = qn->dF;

  /* basically just a regular newton's method except for the application of the jacobian */
  PetscFunctionBegin;

  x = snes->vec_sol;
  f = snes->vec_func;
  p = snes->vec_sol_update;
  pold = snes->work[0];

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
  ierr = VecCopy(f, pold);CHKERRQ(ierr);
  ierr = VecAXPY(x, -1.0, pold);CHKERRQ(ierr);
  for(i = 0; i < snes->max_its; i++) {

    ierr = SNESComputeFunction(snes, x, f);CHKERRQ(ierr);
    ierr = VecNorm(f, NORM_2, &fnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,fnorm,i+1);
    ierr = SNESMonitor(snes,i+1,fnorm);CHKERRQ(ierr);
    
    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;
    /* test convergence */
    ierr = (*snes->ops->converged)(snes,i+1,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
    k = (i) % m;
    l = m;
    if (i < l) l = i;
    ierr = VecCopy(f, p);CHKERRQ(ierr);
    for (j=0; j<k; j++) {                                     /* p = product_{j<i} [I+v(j)w(j)^T]*p */
      ierr = VecDot(W[j],p,&gdot);CHKERRQ(ierr);
      ierr = VecAXPY(p,gdot,V[j]);CHKERRQ(ierr);
    }
    ierr = VecCopy(pold,W[k]);CHKERRQ(ierr);                  /* w[i] = pold   */
    ierr = VecAXPY(pold,-1.0,p);CHKERRQ(ierr);                /* v[i] =         p         */
    ierr = VecDot(W[k],pold,&gdot);CHKERRQ(ierr);             /*        ----------------- */
    ierr = VecCopy(p,V[k]);CHKERRQ(ierr);                     /*         w[i]'*(Pold - p) */
    ierr = VecScale(V[k],1.0/gdot);CHKERRQ(ierr);
    
    ierr = VecDot(W[k],p,&gdot);CHKERRQ(ierr);                /* p = (I + v[i]*w[i]')*p   */
    ierr = VecAXPY(p,gdot,V[k]);CHKERRQ(ierr);
    ierr = VecCopy(p,pold);CHKERRQ(ierr);
    
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Setting %d\n", k);CHKERRQ(ierr);
    ierr = VecAXPY(x, -1.0, p);CHKERRQ(ierr);
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
  ierr = PetscOptionsReal("-snes_qn_lambda", "SOR mixing parameter", "SNES", qn->lambda, &qn->lambda, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_qn_m", "Number of past states saved for L-Broyden methods", "SNES", qn->m, &qn->m, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESQN - Limited-Memory Quasi-Newton methods for the solution of nonlinear systems.


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

  ierr = PetscNewLog(snes, QNContext, &qn);CHKERRQ(ierr);
  snes->data = (void *) qn;
  qn->m = 10;
  qn->lambda = 1.;
  qn->dX = PETSC_NULL;
  qn->dF = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
