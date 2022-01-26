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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(H,&gn);CHKERRQ(ierr);
  ierr = MatMult(gn->subsolver->ls_jac,in,gn->r_work);CHKERRQ(ierr);
  ierr = MatMultTranspose(gn->subsolver->ls_jac,gn->r_work,out);CHKERRQ(ierr);
  switch (gn->reg_type) {
  case BRGN_REGULARIZATION_USER:
    ierr = MatMult(gn->Hreg,in,gn->x_work);CHKERRQ(ierr);
    ierr = VecAXPY(out,gn->lambda,gn->x_work);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L2PURE:
    ierr = VecAXPY(out,gn->lambda,in);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L2PROX:
    ierr = VecAXPY(out,gn->lambda,in);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /* out = out + lambda*D'*(diag.*(D*in)) */
    if (gn->D) {
      ierr = MatMult(gn->D,in,gn->y);CHKERRQ(ierr);/* y = D*in */
    } else {
      ierr = VecCopy(in,gn->y);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(gn->y_work,gn->diag,gn->y);CHKERRQ(ierr);   /* y_work = diag.*(D*in), where diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3 */
    if (gn->D) {
      ierr = MatMultTranspose(gn->D,gn->y_work,gn->x_work);CHKERRQ(ierr); /* x_work = D'*(diag.*(D*in)) */
    } else {
      ierr = VecCopy(gn->y_work,gn->x_work);CHKERRQ(ierr);
    }
    ierr = VecAXPY(out,gn->lambda,gn->x_work);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_LM:
    ierr = VecPointwiseMult(gn->x_work,gn->damping,in);CHKERRQ(ierr);
    ierr = VecAXPY(out,1,gn->x_work);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode ComputeDamping(TAO_BRGN *gn)
{
  const PetscScalar *diag_ary;
  PetscScalar       *damping_ary;
  PetscInt          i,n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* update damping */
  ierr = VecGetArray(gn->damping,&damping_ary);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gn->diag,&diag_ary);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gn->damping,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    damping_ary[i] = PetscClipInterval(diag_ary[i],PETSC_SQRT_MACHINE_EPSILON,PetscSqrtReal(PETSC_MAX_REAL));
  }
  ierr = VecScale(gn->damping,gn->lambda);CHKERRQ(ierr);
  ierr = VecRestoreArray(gn->damping,&damping_ary);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(gn->diag,&diag_ary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBRGNGetDampingVector(Tao tao,Vec *d)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCheckFalse(gn->reg_type != BRGN_REGULARIZATION_LM,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Damping vector is only available if regularization type is lm.");
  *d = gn->damping;
  PetscFunctionReturn(0);
}

static PetscErrorCode GNObjectiveGradientEval(Tao tao,Vec X,PetscReal *fcn,Vec G,void *ptr)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ptr;
  PetscInt              K;                    /* dimension of D*X */
  PetscScalar           yESum;
  PetscErrorCode        ierr;
  PetscReal             f_reg;

  PetscFunctionBegin;
  /* compute objective *fcn*/
  /* compute first term 0.5*||ls_res||_2^2 */
  ierr = TaoComputeResidual(tao,X,tao->ls_res);CHKERRQ(ierr);
  ierr = VecDot(tao->ls_res,tao->ls_res,fcn);CHKERRQ(ierr);
  *fcn *= 0.5;
  /* compute gradient G */
  ierr = TaoComputeResidualJacobian(tao,X,tao->ls_jac,tao->ls_jac_pre);CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->ls_jac,tao->ls_res,G);CHKERRQ(ierr);
  /* add the regularization contribution */
  switch (gn->reg_type) {
  case BRGN_REGULARIZATION_USER:
    ierr = (*gn->regularizerobjandgrad)(tao,X,&f_reg,gn->x_work,gn->reg_obj_ctx);CHKERRQ(ierr);
    *fcn += gn->lambda*f_reg;
    ierr = VecAXPY(G,gn->lambda,gn->x_work);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L2PURE:
    /* compute f = f + lambda*0.5*xk'*xk */
    ierr = VecDot(X,X,&f_reg);CHKERRQ(ierr);
    *fcn += gn->lambda*0.5*f_reg;
    /* compute G = G + lambda*xk */
    ierr = VecAXPY(G,gn->lambda,X);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L2PROX:
    /* compute f = f + lambda*0.5*(xk - xkm1)'*(xk - xkm1) */
    ierr = VecAXPBYPCZ(gn->x_work,1.0,-1.0,0.0,X,gn->x_old);CHKERRQ(ierr);
    ierr = VecDot(gn->x_work,gn->x_work,&f_reg);CHKERRQ(ierr);
    *fcn += gn->lambda*0.5*f_reg;
    /* compute G = G + lambda*(xk - xkm1) */
    ierr = VecAXPBYPCZ(G,gn->lambda,-gn->lambda,1.0,X,gn->x_old);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /* compute f = f + lambda*sum(sqrt(y.^2+epsilon^2) - epsilon), where y = D*x*/
    if (gn->D) {
      ierr = MatMult(gn->D,X,gn->y);CHKERRQ(ierr);/* y = D*x */
    } else {
      ierr = VecCopy(X,gn->y);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(gn->y_work,gn->y,gn->y);CHKERRQ(ierr);
    ierr = VecShift(gn->y_work,gn->epsilon*gn->epsilon);CHKERRQ(ierr);
    ierr = VecSqrtAbs(gn->y_work);CHKERRQ(ierr);  /* gn->y_work = sqrt(y.^2+epsilon^2) */
    ierr = VecSum(gn->y_work,&yESum);CHKERRQ(ierr);
    ierr = VecGetSize(gn->y,&K);CHKERRQ(ierr);
    *fcn += gn->lambda*(yESum - K*gn->epsilon);
    /* compute G = G + lambda*D'*(y./sqrt(y.^2+epsilon^2)),where y = D*x */
    ierr = VecPointwiseDivide(gn->y_work,gn->y,gn->y_work);CHKERRQ(ierr); /* reuse y_work = y./sqrt(y.^2+epsilon^2) */
    if (gn->D) {
      ierr = MatMultTranspose(gn->D,gn->y_work,gn->x_work);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(gn->y_work,gn->x_work);CHKERRQ(ierr);
    }
    ierr = VecAXPY(G,gn->lambda,gn->x_work);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GNComputeHessian(Tao tao,Vec X,Mat H,Mat Hpre,void *ptr)
{
  TAO_BRGN       *gn = (TAO_BRGN *)ptr;
  PetscInt       i,n,cstart,cend;
  PetscScalar    *cnorms,*diag_ary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoComputeResidualJacobian(tao,X,tao->ls_jac,tao->ls_jac_pre);CHKERRQ(ierr);
  if (gn->mat_explicit) {
    ierr = MatTransposeMatMult(tao->ls_jac, tao->ls_jac, MAT_REUSE_MATRIX, PETSC_DEFAULT, &gn->H);CHKERRQ(ierr);
  }

  switch (gn->reg_type) {
  case BRGN_REGULARIZATION_USER:
    ierr = (*gn->regularizerhessian)(tao,X,gn->Hreg,gn->reg_hess_ctx);CHKERRQ(ierr);
    if (gn->mat_explicit) {
      ierr = MatAXPY(gn->H, 1.0, gn->Hreg, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    break;
  case BRGN_REGULARIZATION_L2PURE:
    if (gn->mat_explicit) {
      ierr = MatShift(gn->H, gn->lambda);CHKERRQ(ierr);
    }
    break;
  case BRGN_REGULARIZATION_L2PROX:
    if (gn->mat_explicit) {
      ierr = MatShift(gn->H, gn->lambda);CHKERRQ(ierr);
    }
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /* calculate and store diagonal matrix as a vector: diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3* --> diag = epsilon^2 ./ sqrt(y.^2+epsilon^2).^3,where y = D*x */
    if (gn->D) {
      ierr = MatMult(gn->D,X,gn->y);CHKERRQ(ierr);/* y = D*x */
    } else {
      ierr = VecCopy(X,gn->y);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(gn->y_work,gn->y,gn->y);CHKERRQ(ierr);
    ierr = VecShift(gn->y_work,gn->epsilon*gn->epsilon);CHKERRQ(ierr);
    ierr = VecCopy(gn->y_work,gn->diag);CHKERRQ(ierr);                  /* gn->diag = y.^2+epsilon^2 */
    ierr = VecSqrtAbs(gn->y_work);CHKERRQ(ierr);                        /* gn->y_work = sqrt(y.^2+epsilon^2) */
    ierr = VecPointwiseMult(gn->diag,gn->y_work,gn->diag);CHKERRQ(ierr);/* gn->diag = sqrt(y.^2+epsilon^2).^3 */
    ierr = VecReciprocal(gn->diag);CHKERRQ(ierr);
    ierr = VecScale(gn->diag,gn->epsilon*gn->epsilon);CHKERRQ(ierr);
    if (gn->mat_explicit) {
      ierr = MatDiagonalSet(gn->H, gn->diag, ADD_VALUES);CHKERRQ(ierr);
    }
    break;
  case BRGN_REGULARIZATION_LM:
    /* compute diagonal of J^T J */
    ierr = MatGetSize(gn->parent->ls_jac,NULL,&n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&cnorms);CHKERRQ(ierr);
    ierr = MatGetColumnNorms(gn->parent->ls_jac,NORM_2,cnorms);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(gn->parent->ls_jac,&cstart,&cend);CHKERRQ(ierr);
    ierr = VecGetArray(gn->diag,&diag_ary);CHKERRQ(ierr);
    for (i = 0; i < cend-cstart; i++) {
      diag_ary[i] = cnorms[cstart+i] * cnorms[cstart+i];
    }
    ierr = VecRestoreArray(gn->diag,&diag_ary);CHKERRQ(ierr);
    ierr = PetscFree(cnorms);CHKERRQ(ierr);
    ierr = ComputeDamping(gn);CHKERRQ(ierr);
    if (gn->mat_explicit) {
      ierr = MatDiagonalSet(gn->H, gn->damping, ADD_VALUES);CHKERRQ(ierr);
    }
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GNHookFunction(Tao tao,PetscInt iter, void *ctx)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ctx;
  PetscErrorCode        ierr;

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
  ierr = TaoGetConvergedReason(tao,&gn->parent->reason);CHKERRQ(ierr);
  /* Update the solution vectors */
  if (iter == 0) {
    ierr = VecSet(gn->x_old,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(tao->solution,gn->x_old);CHKERRQ(ierr);
    ierr = VecCopy(tao->solution,gn->parent->solution);CHKERRQ(ierr);
  }
  /* Update the gradient */
  ierr = VecCopy(tao->gradient,gn->parent->gradient);CHKERRQ(ierr);

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
  if (gn->parent->ops->update) {
    ierr = (*gn->parent->ops->update)(gn->parent,gn->parent->niter,gn->parent->user_update);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = TaoSolve(gn->subsolver);CHKERRQ(ierr);
  /* Update basic tao information from the subsolver */
  tao->nfuncs = gn->subsolver->nfuncs;
  tao->ngrads = gn->subsolver->ngrads;
  tao->nfuncgrads = gn->subsolver->nfuncgrads;
  tao->nhess = gn->subsolver->nhess;
  tao->niter = gn->subsolver->niter;
  tao->ksp_its = gn->subsolver->ksp_its;
  tao->ksp_tot_its = gn->subsolver->ksp_tot_its;
  ierr = TaoGetConvergedReason(gn->subsolver,&tao->reason);CHKERRQ(ierr);
  /* Update vectors */
  ierr = VecCopy(gn->subsolver->solution,tao->solution);CHKERRQ(ierr);
  ierr = VecCopy(gn->subsolver->gradient,tao->gradient);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BRGN(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  TaoLineSearch         ls;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"least-squares problems with regularizer: ||f(x)||^2 + lambda*g(x), g(x) = ||xk-xkm1||^2 or ||Dx||_1 or user defined function.");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_brgn_mat_explicit","switches the Hessian construction to be an explicit matrix rather than MATSHELL","",gn->mat_explicit,&gn->mat_explicit,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_regularizer_weight","regularizer weight (default 1e-4)","",gn->lambda,&gn->lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_l1_smooth_epsilon","L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)","",gn->epsilon,&gn->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_lm_downhill_lambda_change","Factor to decrease trust region by on downhill steps","",gn->downhill_lambda_change,&gn->downhill_lambda_change,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_lm_uphill_lambda_change","Factor to increase trust region by on uphill steps","",gn->uphill_lambda_change,&gn->uphill_lambda_change,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_brgn_regularization_type","regularization type", "",BRGN_REGULARIZATION_TABLE,BRGN_REGULARIZATION_TYPES,BRGN_REGULARIZATION_TABLE[gn->reg_type],&gn->reg_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  /* set unit line search direction as the default when using the lm regularizer */
  if (gn->reg_type == BRGN_REGULARIZATION_LM) {
    ierr = TaoGetLineSearch(gn->subsolver,&ls);CHKERRQ(ierr);
    ierr = TaoLineSearchSetType(ls,TAOLINESEARCHUNIT);CHKERRQ(ierr);
  }
  ierr = TaoSetFromOptions(gn->subsolver);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BRGN(Tao tao,PetscViewer viewer)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = TaoView(gn->subsolver,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;
  PetscBool             is_bnls,is_bntr,is_bntl;
  PetscInt              i,n,N,K; /* dict has size K*N*/

  PetscFunctionBegin;
  PetscCheckFalse(!tao->ls_res,PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetResidualRoutine() must be called before setup!");
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNLS,&is_bnls);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNTR,&is_bntr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNTL,&is_bntl);CHKERRQ(ierr);
  PetscCheckFalse((is_bnls || is_bntr || is_bntl) && !tao->ls_jac,PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetResidualJacobianRoutine() must be called before setup!");
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  if (!gn->x_work) {
    ierr = VecDuplicate(tao->solution,&gn->x_work);CHKERRQ(ierr);
  }
  if (!gn->r_work) {
    ierr = VecDuplicate(tao->ls_res,&gn->r_work);CHKERRQ(ierr);
  }
  if (!gn->x_old) {
    ierr = VecDuplicate(tao->solution,&gn->x_old);CHKERRQ(ierr);
    ierr = VecSet(gn->x_old,0.0);CHKERRQ(ierr);
  }

  if (BRGN_REGULARIZATION_L1DICT == gn->reg_type) {
    if (!gn->y) {
      if (gn->D) {
        ierr = MatGetSize(gn->D,&K,&N);CHKERRQ(ierr); /* Shell matrices still must have sizes defined. K = N for identity matrix, K=N-1 or N for gradient matrix */
        ierr = MatCreateVecs(gn->D,NULL,&gn->y);CHKERRQ(ierr);
      } else {
        ierr = VecDuplicate(tao->solution,&gn->y);CHKERRQ(ierr); /* If user does not setup dict matrix, use identiy matrix, K=N */
      }
      ierr = VecSet(gn->y,0.0);CHKERRQ(ierr);
    }
    if (!gn->y_work) {
      ierr = VecDuplicate(gn->y,&gn->y_work);CHKERRQ(ierr);
    }
    if (!gn->diag) {
      ierr = VecDuplicate(gn->y,&gn->diag);CHKERRQ(ierr);
      ierr = VecSet(gn->diag,0.0);CHKERRQ(ierr);
    }
  }
  if (BRGN_REGULARIZATION_LM == gn->reg_type) {
    if (!gn->diag) {
      ierr = MatCreateVecs(tao->ls_jac,&gn->diag,NULL);CHKERRQ(ierr);
    }
    if (!gn->damping) {
      ierr = MatCreateVecs(tao->ls_jac,&gn->damping,NULL);CHKERRQ(ierr);
    }
  }

  if (!tao->setupcalled) {
    /* Hessian setup */
    if (gn->mat_explicit) {
      ierr = TaoComputeResidualJacobian(tao,tao->solution,tao->ls_jac,tao->ls_jac_pre);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(tao->ls_jac, tao->ls_jac, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &gn->H);CHKERRQ(ierr);
    } else {
      ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
      ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
      ierr = MatCreate(PetscObjectComm((PetscObject)tao),&gn->H);CHKERRQ(ierr);
      ierr = MatSetSizes(gn->H,n,n,N,N);CHKERRQ(ierr);
      ierr = MatSetType(gn->H,MATSHELL);CHKERRQ(ierr);
      ierr = MatSetOption(gn->H, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatShellSetOperation(gn->H,MATOP_MULT,(void (*)(void))GNHessianProd);CHKERRQ(ierr);
      ierr = MatShellSetContext(gn->H,gn);CHKERRQ(ierr);
    }
    ierr = MatSetUp(gn->H);CHKERRQ(ierr);
    /* Subsolver setup,include initial vector and dictionary D */
    ierr = TaoSetUpdate(gn->subsolver,GNHookFunction,gn);CHKERRQ(ierr);
    ierr = TaoSetSolution(gn->subsolver,tao->solution);CHKERRQ(ierr);
    if (tao->bounded) {
      ierr = TaoSetVariableBounds(gn->subsolver,tao->XL,tao->XU);CHKERRQ(ierr);
    }
    ierr = TaoSetResidualRoutine(gn->subsolver,tao->ls_res,tao->ops->computeresidual,tao->user_lsresP);CHKERRQ(ierr);
    ierr = TaoSetJacobianResidualRoutine(gn->subsolver,tao->ls_jac,tao->ls_jac,tao->ops->computeresidualjacobian,tao->user_lsjacP);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradient(gn->subsolver,NULL,GNObjectiveGradientEval,gn);CHKERRQ(ierr);
    ierr = TaoSetHessian(gn->subsolver,gn->H,gn->H,GNComputeHessian,gn);CHKERRQ(ierr);
    /* Propagate some options down */
    ierr = TaoSetTolerances(gn->subsolver,tao->gatol,tao->grtol,tao->gttol);CHKERRQ(ierr);
    ierr = TaoSetMaximumIterations(gn->subsolver,tao->max_it);CHKERRQ(ierr);
    ierr = TaoSetMaximumFunctionEvaluations(gn->subsolver,tao->max_funcs);CHKERRQ(ierr);
    for (i=0; i<tao->numbermonitors; ++i) {
      ierr = TaoSetMonitor(gn->subsolver,tao->monitor[i],tao->monitorcontext[i],tao->monitordestroy[i]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)(tao->monitorcontext[i]));CHKERRQ(ierr);
    }
    ierr = TaoSetUp(gn->subsolver);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&tao->gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->x_work);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->r_work);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->x_old);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->diag);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->y);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->y_work);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&gn->damping);CHKERRQ(ierr);
  ierr = VecDestroy(&gn->diag);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->H);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->D);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->Hreg);CHKERRQ(ierr);
  ierr = TaoDestroy(&gn->subsolver);CHKERRQ(ierr);
  gn->parent = NULL;
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&gn);CHKERRQ(ierr);

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

  ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&gn->subsolver);CHKERRQ(ierr);
  ierr = TaoSetType(gn->subsolver,TAOBNLS);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(gn->subsolver,"tao_brgn_subsolver_");CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (dict) {
    PetscValidHeaderSpecific(dict,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,dict,2);
    ierr = PetscObjectReference((PetscObject)dict);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&gn->D);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

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
    ierr = PetscObjectReference((PetscObject)Hreg);CHKERRQ(ierr);
    ierr = MatDestroy(&gn->Hreg);CHKERRQ(ierr);
    gn->Hreg = Hreg;
  }
  PetscFunctionReturn(0);
}
