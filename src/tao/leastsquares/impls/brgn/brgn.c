#include <../src/tao/leastsquares/impls/brgn/brgn.h> /*I "petsctao.h" I*/

#define BRGN_REGULARIZATION_USER    0
#define BRGN_REGULARIZATION_L2PROX  1
#define BRGN_REGULARIZATION_L2PURE  2
#define BRGN_REGULARIZATION_L1DICT  3
#define BRGN_REGULARIZATION_LM      4
#define BRGN_REGULARIZATION_TYPES   5

static const char *BRGN_REGULARIZATION_TABLE[64] = {"user","l2prox","l2pure","l1dict","lm"};

static PetscErrorCode GNHessianProd(Mat H,Vec in,Vec out)
{
  TAO_BRGN              *gn;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(H,&gn));
  PetscCall(MatMult(gn->subsolver->ls_jac,in,gn->r_work));
  PetscCall(MatMultTranspose(gn->subsolver->ls_jac,gn->r_work,out));
  switch (gn->reg_type) {
  case BRGN_REGULARIZATION_USER:
    PetscCall(MatMult(gn->Hreg,in,gn->x_work));
    PetscCall(VecAXPY(out,gn->lambda,gn->x_work));
    break;
  case BRGN_REGULARIZATION_L2PURE:
    PetscCall(VecAXPY(out,gn->lambda,in));
    break;
  case BRGN_REGULARIZATION_L2PROX:
    PetscCall(VecAXPY(out,gn->lambda,in));
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /* out = out + lambda*D'*(diag.*(D*in)) */
    if (gn->D) {
      PetscCall(MatMult(gn->D,in,gn->y));/* y = D*in */
    } else {
      PetscCall(VecCopy(in,gn->y));
    }
    PetscCall(VecPointwiseMult(gn->y_work,gn->diag,gn->y));   /* y_work = diag.*(D*in), where diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3 */
    if (gn->D) {
      PetscCall(MatMultTranspose(gn->D,gn->y_work,gn->x_work)); /* x_work = D'*(diag.*(D*in)) */
    } else {
      PetscCall(VecCopy(gn->y_work,gn->x_work));
    }
    PetscCall(VecAXPY(out,gn->lambda,gn->x_work));
    break;
  case BRGN_REGULARIZATION_LM:
    PetscCall(VecPointwiseMult(gn->x_work,gn->damping,in));
    PetscCall(VecAXPY(out,1,gn->x_work));
    break;
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode ComputeDamping(TAO_BRGN *gn)
{
  const PetscScalar *diag_ary;
  PetscScalar       *damping_ary;
  PetscInt          i,n;

  PetscFunctionBegin;
  /* update damping */
  PetscCall(VecGetArray(gn->damping,&damping_ary));
  PetscCall(VecGetArrayRead(gn->diag,&diag_ary));
  PetscCall(VecGetLocalSize(gn->damping,&n));
  for (i=0; i<n; i++) {
    damping_ary[i] = PetscClipInterval(diag_ary[i],PETSC_SQRT_MACHINE_EPSILON,PetscSqrtReal(PETSC_MAX_REAL));
  }
  PetscCall(VecScale(gn->damping,gn->lambda));
  PetscCall(VecRestoreArray(gn->damping,&damping_ary));
  PetscCall(VecRestoreArrayRead(gn->diag,&diag_ary));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBRGNGetDampingVector(Tao tao,Vec *d)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCheck(gn->reg_type == BRGN_REGULARIZATION_LM,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Damping vector is only available if regularization type is lm.");
  *d = gn->damping;
  PetscFunctionReturn(0);
}

static PetscErrorCode GNObjectiveGradientEval(Tao tao,Vec X,PetscReal *fcn,Vec G,void *ptr)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ptr;
  PetscInt              K;                    /* dimension of D*X */
  PetscScalar           yESum;
  PetscReal             f_reg;

  PetscFunctionBegin;
  /* compute objective *fcn*/
  /* compute first term 0.5*||ls_res||_2^2 */
  PetscCall(TaoComputeResidual(tao,X,tao->ls_res));
  PetscCall(VecDot(tao->ls_res,tao->ls_res,fcn));
  *fcn *= 0.5;
  /* compute gradient G */
  PetscCall(TaoComputeResidualJacobian(tao,X,tao->ls_jac,tao->ls_jac_pre));
  PetscCall(MatMultTranspose(tao->ls_jac,tao->ls_res,G));
  /* add the regularization contribution */
  switch (gn->reg_type) {
  case BRGN_REGULARIZATION_USER:
    PetscCall((*gn->regularizerobjandgrad)(tao,X,&f_reg,gn->x_work,gn->reg_obj_ctx));
    *fcn += gn->lambda*f_reg;
    PetscCall(VecAXPY(G,gn->lambda,gn->x_work));
    break;
  case BRGN_REGULARIZATION_L2PURE:
    /* compute f = f + lambda*0.5*xk'*xk */
    PetscCall(VecDot(X,X,&f_reg));
    *fcn += gn->lambda*0.5*f_reg;
    /* compute G = G + lambda*xk */
    PetscCall(VecAXPY(G,gn->lambda,X));
    break;
  case BRGN_REGULARIZATION_L2PROX:
    /* compute f = f + lambda*0.5*(xk - xkm1)'*(xk - xkm1) */
    PetscCall(VecAXPBYPCZ(gn->x_work,1.0,-1.0,0.0,X,gn->x_old));
    PetscCall(VecDot(gn->x_work,gn->x_work,&f_reg));
    *fcn += gn->lambda*0.5*f_reg;
    /* compute G = G + lambda*(xk - xkm1) */
    PetscCall(VecAXPBYPCZ(G,gn->lambda,-gn->lambda,1.0,X,gn->x_old));
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /* compute f = f + lambda*sum(sqrt(y.^2+epsilon^2) - epsilon), where y = D*x*/
    if (gn->D) {
      PetscCall(MatMult(gn->D,X,gn->y));/* y = D*x */
    } else {
      PetscCall(VecCopy(X,gn->y));
    }
    PetscCall(VecPointwiseMult(gn->y_work,gn->y,gn->y));
    PetscCall(VecShift(gn->y_work,gn->epsilon*gn->epsilon));
    PetscCall(VecSqrtAbs(gn->y_work));  /* gn->y_work = sqrt(y.^2+epsilon^2) */
    PetscCall(VecSum(gn->y_work,&yESum));
    PetscCall(VecGetSize(gn->y,&K));
    *fcn += gn->lambda*(yESum - K*gn->epsilon);
    /* compute G = G + lambda*D'*(y./sqrt(y.^2+epsilon^2)),where y = D*x */
    PetscCall(VecPointwiseDivide(gn->y_work,gn->y,gn->y_work)); /* reuse y_work = y./sqrt(y.^2+epsilon^2) */
    if (gn->D) {
      PetscCall(MatMultTranspose(gn->D,gn->y_work,gn->x_work));
    } else {
      PetscCall(VecCopy(gn->y_work,gn->x_work));
    }
    PetscCall(VecAXPY(G,gn->lambda,gn->x_work));
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GNComputeHessian(Tao tao,Vec X,Mat H,Mat Hpre,void *ptr)
{
  TAO_BRGN       *gn = (TAO_BRGN *)ptr;
  PetscInt       i,n,cstart,cend;
  PetscScalar    *cnorms,*diag_ary;

  PetscFunctionBegin;
  PetscCall(TaoComputeResidualJacobian(tao,X,tao->ls_jac,tao->ls_jac_pre));
  if (gn->mat_explicit) {
    PetscCall(MatTransposeMatMult(tao->ls_jac, tao->ls_jac, MAT_REUSE_MATRIX, PETSC_DEFAULT, &gn->H));
  }

  switch (gn->reg_type) {
  case BRGN_REGULARIZATION_USER:
    PetscCall((*gn->regularizerhessian)(tao,X,gn->Hreg,gn->reg_hess_ctx));
    if (gn->mat_explicit) PetscCall(MatAXPY(gn->H, 1.0, gn->Hreg, DIFFERENT_NONZERO_PATTERN));
    break;
  case BRGN_REGULARIZATION_L2PURE:
    if (gn->mat_explicit) PetscCall(MatShift(gn->H, gn->lambda));
    break;
  case BRGN_REGULARIZATION_L2PROX:
    if (gn->mat_explicit) PetscCall(MatShift(gn->H, gn->lambda));
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /* calculate and store diagonal matrix as a vector: diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3* --> diag = epsilon^2 ./ sqrt(y.^2+epsilon^2).^3,where y = D*x */
    if (gn->D) {
      PetscCall(MatMult(gn->D,X,gn->y));/* y = D*x */
    } else {
      PetscCall(VecCopy(X,gn->y));
    }
    PetscCall(VecPointwiseMult(gn->y_work,gn->y,gn->y));
    PetscCall(VecShift(gn->y_work,gn->epsilon*gn->epsilon));
    PetscCall(VecCopy(gn->y_work,gn->diag));                  /* gn->diag = y.^2+epsilon^2 */
    PetscCall(VecSqrtAbs(gn->y_work));                        /* gn->y_work = sqrt(y.^2+epsilon^2) */
    PetscCall(VecPointwiseMult(gn->diag,gn->y_work,gn->diag));/* gn->diag = sqrt(y.^2+epsilon^2).^3 */
    PetscCall(VecReciprocal(gn->diag));
    PetscCall(VecScale(gn->diag,gn->epsilon*gn->epsilon));
    if (gn->mat_explicit) PetscCall(MatDiagonalSet(gn->H, gn->diag, ADD_VALUES));
    break;
  case BRGN_REGULARIZATION_LM:
    /* compute diagonal of J^T J */
    PetscCall(MatGetSize(gn->parent->ls_jac,NULL,&n));
    PetscCall(PetscMalloc1(n,&cnorms));
    PetscCall(MatGetColumnNorms(gn->parent->ls_jac,NORM_2,cnorms));
    PetscCall(MatGetOwnershipRangeColumn(gn->parent->ls_jac,&cstart,&cend));
    PetscCall(VecGetArray(gn->diag,&diag_ary));
    for (i = 0; i < cend-cstart; i++) {
      diag_ary[i] = cnorms[cstart+i] * cnorms[cstart+i];
    }
    PetscCall(VecRestoreArray(gn->diag,&diag_ary));
    PetscCall(PetscFree(cnorms));
    PetscCall(ComputeDamping(gn));
    if (gn->mat_explicit) PetscCall(MatDiagonalSet(gn->H, gn->damping, ADD_VALUES));
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GNHookFunction(Tao tao,PetscInt iter, void *ctx)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ctx;

  PetscFunctionBegin;
  /* Update basic tao information from the subsolver */
  gn->parent->nfuncs = tao->nfuncs;
  gn->parent->ngrads = tao->ngrads;
  gn->parent->nfuncgrads = tao->nfuncgrads;
  gn->parent->nhess = tao->nhess;
  gn->parent->niter = tao->niter;
  gn->parent->ksp_its = tao->ksp_its;
  gn->parent->ksp_tot_its = tao->ksp_tot_its;
  gn->parent->fc = tao->fc;
  PetscCall(TaoGetConvergedReason(tao,&gn->parent->reason));
  /* Update the solution vectors */
  if (iter == 0) {
    PetscCall(VecSet(gn->x_old,0.0));
  } else {
    PetscCall(VecCopy(tao->solution,gn->x_old));
    PetscCall(VecCopy(tao->solution,gn->parent->solution));
  }
  /* Update the gradient */
  PetscCall(VecCopy(tao->gradient,gn->parent->gradient));

  /* Update damping parameter for LM */
  if (gn->reg_type == BRGN_REGULARIZATION_LM) {
    if (iter > 0) {
      if (gn->fc_old > tao->fc) {
        gn->lambda = gn->lambda * gn->downhill_lambda_change;
      } else {
        /* uphill step */
        gn->lambda = gn->lambda * gn->uphill_lambda_change;
      }
    }
    gn->fc_old = tao->fc;
  }

  /* Call general purpose update function */
  if (gn->parent->ops->update) PetscCall((*gn->parent->ops->update)(gn->parent,gn->parent->niter,gn->parent->user_update));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoSolve(gn->subsolver));
  /* Update basic tao information from the subsolver */
  tao->nfuncs = gn->subsolver->nfuncs;
  tao->ngrads = gn->subsolver->ngrads;
  tao->nfuncgrads = gn->subsolver->nfuncgrads;
  tao->nhess = gn->subsolver->nhess;
  tao->niter = gn->subsolver->niter;
  tao->ksp_its = gn->subsolver->ksp_its;
  tao->ksp_tot_its = gn->subsolver->ksp_tot_its;
  PetscCall(TaoGetConvergedReason(gn->subsolver,&tao->reason));
  /* Update vectors */
  PetscCall(VecCopy(gn->subsolver->solution,tao->solution));
  PetscCall(VecCopy(gn->subsolver->gradient,tao->gradient));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BRGN(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  TaoLineSearch         ls;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"least-squares problems with regularizer: ||f(x)||^2 + lambda*g(x), g(x) = ||xk-xkm1||^2 or ||Dx||_1 or user defined function.");
  PetscCall(PetscOptionsBool("-tao_brgn_mat_explicit","switches the Hessian construction to be an explicit matrix rather than MATSHELL","",gn->mat_explicit,&gn->mat_explicit,NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_regularizer_weight","regularizer weight (default 1e-4)","",gn->lambda,&gn->lambda,NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_l1_smooth_epsilon","L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)","",gn->epsilon,&gn->epsilon,NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_lm_downhill_lambda_change","Factor to decrease trust region by on downhill steps","",gn->downhill_lambda_change,&gn->downhill_lambda_change,NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_lm_uphill_lambda_change","Factor to increase trust region by on uphill steps","",gn->uphill_lambda_change,&gn->uphill_lambda_change,NULL));
  PetscCall(PetscOptionsEList("-tao_brgn_regularization_type","regularization type", "",BRGN_REGULARIZATION_TABLE,BRGN_REGULARIZATION_TYPES,BRGN_REGULARIZATION_TABLE[gn->reg_type],&gn->reg_type,NULL));
  PetscOptionsHeadEnd();
  /* set unit line search direction as the default when using the lm regularizer */
  if (gn->reg_type == BRGN_REGULARIZATION_LM) {
    PetscCall(TaoGetLineSearch(gn->subsolver,&ls));
    PetscCall(TaoLineSearchSetType(ls,TAOLINESEARCHUNIT));
  }
  PetscCall(TaoSetFromOptions(gn->subsolver));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BRGN(Tao tao,PetscViewer viewer)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoView(gn->subsolver,viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscBool             is_bnls,is_bntr,is_bntl;
  PetscInt              i,n,N,K; /* dict has size K*N*/

  PetscFunctionBegin;
  PetscCheck(tao->ls_res,PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetResidualRoutine() must be called before setup!");
  PetscCall(PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNLS,&is_bnls));
  PetscCall(PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNTR,&is_bntr));
  PetscCall(PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNTL,&is_bntl));
  PetscCheck((!is_bnls && !is_bntr && !is_bntl) || tao->ls_jac,PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetResidualJacobianRoutine() must be called before setup!");
  if (!tao->gradient) {
    PetscCall(VecDuplicate(tao->solution,&tao->gradient));
  }
  if (!gn->x_work) {
    PetscCall(VecDuplicate(tao->solution,&gn->x_work));
  }
  if (!gn->r_work) {
    PetscCall(VecDuplicate(tao->ls_res,&gn->r_work));
  }
  if (!gn->x_old) {
    PetscCall(VecDuplicate(tao->solution,&gn->x_old));
    PetscCall(VecSet(gn->x_old,0.0));
  }

  if (BRGN_REGULARIZATION_L1DICT == gn->reg_type) {
    if (!gn->y) {
      if (gn->D) {
        PetscCall(MatGetSize(gn->D,&K,&N)); /* Shell matrices still must have sizes defined. K = N for identity matrix, K=N-1 or N for gradient matrix */
        PetscCall(MatCreateVecs(gn->D,NULL,&gn->y));
      } else {
        PetscCall(VecDuplicate(tao->solution,&gn->y)); /* If user does not setup dict matrix, use identiy matrix, K=N */
      }
      PetscCall(VecSet(gn->y,0.0));
    }
    if (!gn->y_work) {
      PetscCall(VecDuplicate(gn->y,&gn->y_work));
    }
    if (!gn->diag) {
      PetscCall(VecDuplicate(gn->y,&gn->diag));
      PetscCall(VecSet(gn->diag,0.0));
    }
  }
  if (BRGN_REGULARIZATION_LM == gn->reg_type) {
    if (!gn->diag) {
      PetscCall(MatCreateVecs(tao->ls_jac,&gn->diag,NULL));
    }
    if (!gn->damping) {
      PetscCall(MatCreateVecs(tao->ls_jac,&gn->damping,NULL));
    }
  }

  if (!tao->setupcalled) {
    /* Hessian setup */
    if (gn->mat_explicit) {
      PetscCall(TaoComputeResidualJacobian(tao,tao->solution,tao->ls_jac,tao->ls_jac_pre));
      PetscCall(MatTransposeMatMult(tao->ls_jac, tao->ls_jac, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &gn->H));
    } else {
      PetscCall(VecGetLocalSize(tao->solution,&n));
      PetscCall(VecGetSize(tao->solution,&N));
      PetscCall(MatCreate(PetscObjectComm((PetscObject)tao),&gn->H));
      PetscCall(MatSetSizes(gn->H,n,n,N,N));
      PetscCall(MatSetType(gn->H,MATSHELL));
      PetscCall(MatSetOption(gn->H, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(MatShellSetOperation(gn->H,MATOP_MULT,(void (*)(void))GNHessianProd));
      PetscCall(MatShellSetContext(gn->H,gn));
    }
    PetscCall(MatSetUp(gn->H));
    /* Subsolver setup,include initial vector and dictionary D */
    PetscCall(TaoSetUpdate(gn->subsolver,GNHookFunction,gn));
    PetscCall(TaoSetSolution(gn->subsolver,tao->solution));
    if (tao->bounded) PetscCall(TaoSetVariableBounds(gn->subsolver,tao->XL,tao->XU));
    PetscCall(TaoSetResidualRoutine(gn->subsolver,tao->ls_res,tao->ops->computeresidual,tao->user_lsresP));
    PetscCall(TaoSetJacobianResidualRoutine(gn->subsolver,tao->ls_jac,tao->ls_jac,tao->ops->computeresidualjacobian,tao->user_lsjacP));
    PetscCall(TaoSetObjectiveAndGradient(gn->subsolver,NULL,GNObjectiveGradientEval,gn));
    PetscCall(TaoSetHessian(gn->subsolver,gn->H,gn->H,GNComputeHessian,gn));
    /* Propagate some options down */
    PetscCall(TaoSetTolerances(gn->subsolver,tao->gatol,tao->grtol,tao->gttol));
    PetscCall(TaoSetMaximumIterations(gn->subsolver,tao->max_it));
    PetscCall(TaoSetMaximumFunctionEvaluations(gn->subsolver,tao->max_funcs));
    for (i=0; i<tao->numbermonitors; ++i) {
      PetscCall(TaoSetMonitor(gn->subsolver,tao->monitor[i],tao->monitorcontext[i],tao->monitordestroy[i]));
      PetscCall(PetscObjectReference((PetscObject)(tao->monitorcontext[i])));
    }
    PetscCall(TaoSetUp(gn->subsolver));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&tao->gradient));
    PetscCall(VecDestroy(&gn->x_work));
    PetscCall(VecDestroy(&gn->r_work));
    PetscCall(VecDestroy(&gn->x_old));
    PetscCall(VecDestroy(&gn->diag));
    PetscCall(VecDestroy(&gn->y));
    PetscCall(VecDestroy(&gn->y_work));
  }
  PetscCall(VecDestroy(&gn->damping));
  PetscCall(VecDestroy(&gn->diag));
  PetscCall(MatDestroy(&gn->H));
  PetscCall(MatDestroy(&gn->D));
  PetscCall(MatDestroy(&gn->Hreg));
  PetscCall(TaoDestroy(&gn->subsolver));
  gn->parent = NULL;
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*MC
  TAOBRGN - Bounded Regularized Gauss-Newton method for solving nonlinear least-squares
            problems with bound constraints. This algorithm is a thin wrapper around TAOBNTL
            that constructs the Gauss-Newton problem with the user-provided least-squares
            residual and Jacobian. The algorithm offers an L2-norm ("l2pure"), L2-norm proximal point ("l2prox")
            regularizer, and L1-norm dictionary regularizer ("l1dict"), where we approximate the
            L1-norm ||x||_1 by sum_i(sqrt(x_i^2+epsilon^2)-epsilon) with a small positive number epsilon.
            Also offered is the "lm" regularizer which uses a scaled diagonal of J^T J.
            With the "lm" regularizer, BRGN is a Levenberg-Marquardt optimizer.
            The user can also provide own regularization function.

  Options Database Keys:
+ -tao_brgn_regularization_type - regularization type ("user", "l2prox", "l2pure", "l1dict", "lm") (default "l2prox")
. -tao_brgn_regularizer_weight  - regularizer weight (default 1e-4)
- -tao_brgn_l1_smooth_epsilon   - L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BRGN(Tao tao)
{
  TAO_BRGN       *gn;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(tao,&gn));

  tao->ops->destroy = TaoDestroy_BRGN;
  tao->ops->setup = TaoSetUp_BRGN;
  tao->ops->setfromoptions = TaoSetFromOptions_BRGN;
  tao->ops->view = TaoView_BRGN;
  tao->ops->solve = TaoSolve_BRGN;

  tao->data = gn;
  gn->reg_type = BRGN_REGULARIZATION_L2PROX;
  gn->lambda = 1e-4;
  gn->epsilon = 1e-6;
  gn->downhill_lambda_change = 1./5.;
  gn->uphill_lambda_change = 1.5;
  gn->parent = tao;

  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao),&gn->subsolver));
  PetscCall(TaoSetType(gn->subsolver,TAOBNLS));
  PetscCall(TaoSetOptionsPrefix(gn->subsolver,"tao_brgn_subsolver_"));
  PetscFunctionReturn(0);
}

/*@
  TaoBRGNGetSubsolver - Get the pointer to the subsolver inside BRGN

  Collective on Tao

  Level: advanced

  Input Parameters:
+  tao - the Tao solver context
-  subsolver - the Tao sub-solver context
@*/
PetscErrorCode TaoBRGNGetSubsolver(Tao tao,Tao *subsolver)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  *subsolver = gn->subsolver;
  PetscFunctionReturn(0);
}

/*@
  TaoBRGNSetRegularizerWeight - Set the regularizer weight for the Gauss-Newton least-squares algorithm

  Collective on Tao

  Input Parameters:
+  tao - the Tao solver context
-  lambda - L1-norm regularizer weight

  Level: beginner
@*/
PetscErrorCode TaoBRGNSetRegularizerWeight(Tao tao,PetscReal lambda)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;

  /* Initialize lambda here */

  PetscFunctionBegin;
  gn->lambda = lambda;
  PetscFunctionReturn(0);
}

/*@
  TaoBRGNSetL1SmoothEpsilon - Set the L1-norm smooth approximation parameter for L1-regularized least-squares algorithm

  Collective on Tao

  Input Parameters:
+  tao - the Tao solver context
-  epsilon - L1-norm smooth approximation parameter

  Level: advanced
@*/
PetscErrorCode TaoBRGNSetL1SmoothEpsilon(Tao tao,PetscReal epsilon)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;

  /* Initialize epsilon here */

  PetscFunctionBegin;
  gn->epsilon = epsilon;
  PetscFunctionReturn(0);
}

/*@
   TaoBRGNSetDictionaryMatrix - bind the dictionary matrix from user application context to gn->D, for compressed sensing (with least-squares problem)

   Input Parameters:
+  tao  - the Tao context
-  dict - the user specified dictionary matrix.  We allow to set a null dictionary, which means identity matrix by default

    Level: advanced
@*/
PetscErrorCode TaoBRGNSetDictionaryMatrix(Tao tao,Mat dict)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (dict) {
    PetscValidHeaderSpecific(dict,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,dict,2);
    PetscCall(PetscObjectReference((PetscObject)dict));
  }
  PetscCall(MatDestroy(&gn->D));
  gn->D = dict;
  PetscFunctionReturn(0);
}

/*@C
   TaoBRGNSetRegularizerObjectiveAndGradientRoutine - Sets the user-defined regularizer call-back
   function into the algorithm.

   Input Parameters:
+ tao - the Tao context
. func - function pointer for the regularizer value and gradient evaluation
- ctx - user context for the regularizer

   Level: advanced
@*/
PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(Tao tao,PetscErrorCode (*func)(Tao,Vec,PetscReal *,Vec,void*),void *ctx)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (ctx) {
    gn->reg_obj_ctx = ctx;
  }
  if (func) {
    gn->regularizerobjandgrad = func;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoBRGNSetRegularizerHessianRoutine - Sets the user-defined regularizer call-back
   function into the algorithm.

   Input Parameters:
+ tao - the Tao context
. Hreg - user-created matrix for the Hessian of the regularization term
. func - function pointer for the regularizer Hessian evaluation
- ctx - user context for the regularizer Hessian

   Level: advanced
@*/
PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(Tao tao,Mat Hreg,PetscErrorCode (*func)(Tao,Vec,Mat,void*),void *ctx)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (Hreg) {
    PetscValidHeaderSpecific(Hreg,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,Hreg,2);
  } else SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONG,"NULL Hessian detected! User must provide valid Hessian for the regularizer.");
  if (ctx) {
    gn->reg_hess_ctx = ctx;
  }
  if (func) {
    gn->regularizerhessian = func;
  }
  if (Hreg) {
    PetscCall(PetscObjectReference((PetscObject)Hreg));
    PetscCall(MatDestroy(&gn->Hreg));
    gn->Hreg = Hreg;
  }
  PetscFunctionReturn(0);
}
