#include <../src/tao/leastsquares/impls/brgn/brgn.h>

/* Old code
static PetscErrorCode GNHessianProd(Mat H, Vec in, Vec out)
{
  TAO_BRGN              *gn;
  PetscErrorCode        ierr;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(H, &gn);CHKERRQ(ierr);
  ierr = MatMult(gn->subsolver->ls_jac, in, gn->r_work);CHKERRQ(ierr);
  ierr = MatMultTranspose(gn->subsolver->ls_jac, gn->r_work, out);CHKERRQ(ierr);
  ierr = VecAXPY(out, gn->lambda, in);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GNObjectiveGradientEval(Tao tao, Vec X, PetscReal *fcn, Vec G, void *ptr)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ptr;
  PetscScalar           xnorm2;
  PetscErrorCode        ierr;
  
  PetscFunctionBegin;
  ierr = TaoComputeResidual(tao, X, tao->ls_res);CHKERRQ(ierr);
  ierr = VecDotBegin(tao->ls_res, tao->ls_res, fcn);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(gn->x_work, 1.0, -1.0, 0.0, X, gn->x_old);CHKERRQ(ierr);
  ierr = VecDotBegin(gn->x_work, gn->x_work, &xnorm2);CHKERRQ(ierr);
  ierr = VecDotEnd(tao->ls_res, tao->ls_res, fcn);CHKERRQ(ierr);
  ierr = VecDotEnd(gn->x_work, gn->x_work, &xnorm2);CHKERRQ(ierr);
  *fcn = 0.5*(*fcn) + 0.5*gn->lambda*xnorm2;
  
  ierr = TaoComputeResidualJacobian(tao, X, tao->ls_jac, tao->ls_jac_pre);CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->ls_jac, tao->ls_res, G);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(G, gn->lambda, -gn->lambda, 1.0, X, gn->x_old);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
*/

static PetscErrorCode GNHessianProd(Mat H, Vec in, Vec out)
{
  TAO_BRGN              *gn;
  PetscInt              N = 3; /* dimension of X hack, to change size 3 to VecGetSize() from Vec X*/
  PetscScalar           epsilon = 1e-6; /*XH: added epsilon to approximate L1-norm with sqrt(x^2+epsilon^2)-epsilon*/
  PetscErrorCode        ierr;
  Vec                   xE, tmp;          /* xE: vector sqrt(x.^2 + epsilon^2) that is reused multiple times */  
  
  PetscFunctionBegin;
  /* Allocate vectors */

  /* XH: Use x_work or r_work instead and do not create new vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&xE);CHKERRQ(ierr); 
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&tmp);CHKERRQ(ierr); 
    
  ierr = MatShellGetContext(H, &gn);CHKERRQ(ierr);
  ierr = MatMult(gn->subsolver->ls_jac, in, gn->r_work);CHKERRQ(ierr);
  ierr = MatMultTranspose(gn->subsolver->ls_jac, gn->r_work, out);CHKERRQ(ierr);

  /* XH: Use the sparse diagonal vector diag that has already been computed.  Code should just be the VecAXPY() */

  /* out = out +  lambda*epsilon^2*(in./xE.^3) */
  /* xE = sqrt(x.^2+epsilon^2). Should we reuse code/result of xE from GNObjectiveGradientEval()?*/
  ierr = VecPointwiseMult(xE, gn->x_old, gn->x_old);CHKERRQ(ierr);  /* hack, todo: change to X, how to add X? */
  ierr = VecShift(xE, epsilon*epsilon);CHKERRQ(ierr);
  ierr = VecCopy(xE, tmp);CHKERRQ(ierr);                 /* tmp = xE.^2 */
  ierr = VecSqrtAbs(xE);CHKERRQ(ierr);
  ierr = VecPointwiseMult(tmp, tmp, xE);CHKERRQ(ierr);   /* tmp = xE.^3 */
  ierr = VecPointwiseDivide(tmp, in, tmp);CHKERRQ(ierr); /* tmp = in./xE.^3 */
  ierr = VecAXPY(out, gn->lambda * epsilon * epsilon, tmp);CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&xE);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GNObjectiveGradientEval(Tao tao, Vec X, PetscReal *fcn, Vec G, void *ptr)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ptr;
  PetscInt              N = 3; /* dimension of X hack, to change size 3 to VecGetSize() from Vec X*/
  PetscScalar           xESum, epsilon = 1e-6;/* epsilon to approximate L1-norm with sqrt(x^2+epsilon^2)-epsilon*/
  PetscErrorCode        ierr;
  Vec                   xE;          /* xE: vector sqrt(x.^2 + epsilon^2) that is reused multiple times */  
  
  PetscFunctionBegin;
  /* Allocate vectors */

  /* XH: Do no create vectors; use x_work or r_work */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&xE);CHKERRQ(ierr); 

  ierr = TaoComputeResidual(tao, X, tao->ls_res);CHKERRQ(ierr);
  ierr = VecDotBegin(tao->ls_res, tao->ls_res, fcn);CHKERRQ(ierr);
  ierr = VecDotEnd(tao->ls_res, tao->ls_res, fcn);CHKERRQ(ierr);

  /* Compute xE = sqrt(x.^2+epsilon^2) */

  /* XH: Use epsilon from the gn structure */
  /* XH: Use VecGetSize() rather than N */

  ierr = VecPointwiseMult(xE, X, X);CHKERRQ(ierr);
  ierr = VecShift(xE, epsilon*epsilon);CHKERRQ(ierr);
  ierr = VecSqrtAbs(xE);CHKERRQ(ierr);

  ierr = VecSum(xE,&xESum);CHKERRQ(ierr);CHKERRQ(ierr);
  *fcn = 0.5*(*fcn) + gn->lambda*(xESum - N*epsilon);
  
  ierr = TaoComputeResidualJacobian(tao, X, tao->ls_jac, tao->ls_jac_pre);CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->ls_jac, tao->ls_res, G);CHKERRQ(ierr);
  /* G = G + lambda*(X./xE) */
  ierr = VecPointwiseDivide(xE, X, xE);CHKERRQ(ierr); /* reuse xE = X./xE */
  ierr = VecAXPY(G, gn->lambda, xE);CHKERRQ(ierr); 

  /* Free PETSc data structures */
  ierr = VecDestroy(&xE);CHKERRQ(ierr);  

  PetscFunctionReturn(0);
}


static PetscErrorCode GNComputeHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{ 
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoComputeResidualJacobian(tao, X, tao->ls_jac, tao->ls_jac_pre);CHKERRQ(ierr);

  /* XH: Calculate and store diagonal matrix as a vector; diag in the structure */
  PetscFunctionReturn(0);
}

static PetscErrorCode GNHookFunction(Tao tao, PetscInt iter)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->user_update;
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
  ierr = TaoGetConvergedReason(tao, &gn->parent->reason);CHKERRQ(ierr);
  /* Update the solution vectors */
  if (iter == 0) {
    ierr = VecSet(gn->x_old, 0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(tao->solution, gn->x_old);CHKERRQ(ierr);
    ierr = VecCopy(tao->solution, gn->parent->solution);CHKERRQ(ierr);
  }
  /* Update the gradient */
  ierr = VecCopy(tao->gradient, gn->parent->gradient);CHKERRQ(ierr);
  /* Call general purpose update function */
  if (gn->parent->ops->update) {
    ierr = (*gn->parent->ops->update)(gn->parent, gn->parent->niter);CHKERRQ(ierr);
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
  ierr = TaoGetConvergedReason(gn->subsolver, &tao->reason);CHKERRQ(ierr);
  /* Update vectors */
  ierr = VecCopy(gn->subsolver->solution, tao->solution);CHKERRQ(ierr);
  ierr = VecCopy(gn->subsolver->gradient, tao->gradient);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BRGN(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  /* XH: Read an option to change epsilon in the structure; follow the lambda function below */

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Gauss-Newton method for least-squares problems using Tikhonov regularization");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_lambda", "Tikhonov regularization factor", "", gn->lambda, &gn->lambda, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetFromOptions(gn->subsolver);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BRGN(Tao tao, PetscViewer viewer)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = TaoView(gn->subsolver, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;
  PetscBool             is_bnls, is_bntr, is_bntl;
  PetscInt              i, nx, Nx;

  PetscFunctionBegin;
  if (!tao->ls_res) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetResidualRoutine() must be called before setup!");
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver, TAOBNLS, &is_bnls);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver, TAOBNTR, &is_bntr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver, TAOBNTL, &is_bntl);CHKERRQ(ierr);
  if ((is_bnls || is_bntr || is_bntl) && !tao->ls_jac) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetResidualJacobianRoutine() must be called before setup!");
  if (!tao->gradient){
    ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  }
  if (!gn->x_work){
    ierr = VecDuplicate(tao->solution, &gn->x_work);CHKERRQ(ierr);
  }
  if (!gn->r_work){
    ierr = VecDuplicate(tao->ls_res, &gn->r_work);CHKERRQ(ierr);
  }
  if (!gn->x_old) {
    ierr = VecDuplicate(tao->solution, &gn->x_old);CHKERRQ(ierr);
    ierr = VecSet(gn->x_old, 0.0);CHKERRQ(ierr);
  }

  /* XH: Create diag vector */

  if (!tao->setupcalled) {
    /* Hessian setup */
    ierr = VecGetLocalSize(tao->solution, &nx);CHKERRQ(ierr);
    ierr = VecGetSize(tao->solution, &Nx);CHKERRQ(ierr);
    ierr = MatSetSizes(gn->H, nx, nx, Nx, Nx);CHKERRQ(ierr);
    ierr = MatSetType(gn->H, MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(gn->H);CHKERRQ(ierr);
    ierr = MatShellSetOperation(gn->H, MATOP_MULT, (void (*)(void))GNHessianProd);CHKERRQ(ierr);
    ierr = MatShellSetContext(gn->H, (void*)gn);CHKERRQ(ierr);
    /* Subsolver setup */
    ierr = TaoSetUpdate(gn->subsolver, GNHookFunction, (void*)gn);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(gn->subsolver, tao->solution);CHKERRQ(ierr);
    if (tao->bounded) {
      ierr = TaoSetVariableBounds(gn->subsolver, tao->XL, tao->XU);CHKERRQ(ierr);
    }
    ierr = TaoSetResidualRoutine(gn->subsolver, tao->ls_res, tao->ops->computeresidual, tao->user_lsresP);CHKERRQ(ierr);
    ierr = TaoSetJacobianResidualRoutine(gn->subsolver, tao->ls_jac, tao->ls_jac, tao->ops->computeresidualjacobian, tao->user_lsjacP);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine(gn->subsolver, GNObjectiveGradientEval, (void*)gn);CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine(gn->subsolver, gn->H, gn->H, GNComputeHessian, (void*)gn);CHKERRQ(ierr);
    /* Propagate some options down */
    ierr = TaoSetTolerances(gn->subsolver, tao->gatol, tao->grtol, tao->gttol);CHKERRQ(ierr);
    ierr = TaoSetMaximumIterations(gn->subsolver, tao->max_it);CHKERRQ(ierr);
    ierr = TaoSetMaximumFunctionEvaluations(gn->subsolver, tao->max_funcs);CHKERRQ(ierr);
    for (i=0; i<tao->numbermonitors; ++i) {
      ierr = TaoSetMonitor(gn->subsolver, tao->monitor[i], tao->monitorcontext[i], tao->monitordestroy[i]);CHKERRQ(ierr);
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

    /* XH: Destroy diagonal vector */
  }
  ierr = MatDestroy(&gn->H);CHKERRQ(ierr);
  ierr = TaoDestroy(&gn->subsolver);CHKERRQ(ierr);
  gn->parent = NULL;
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  TAOBRGN - Bounded Regularized Gauss-Newton method for solving nonlinear least-squares 
            problems with bound constraints. This algorithm is a thin wrapper around TAOBNTL 
            that constructs the Guass-Newton problem with the user-provided least-squares 
            residual and Jacobian. The problem is regularized with an L2-norm proximal point 
            term.

  Options Database Keys:
  + -tao_bqnk_max_cg_its - maximum number of bounded conjugate-gradient iterations taken in each Newton loop
  . -tao_bqnk_init_type - trust radius initialization method ("constant", "direction", "interpolation")
  . -tao_bqnk_update_type - trust radius update method ("step", "direction", "interpolation")
  - -tao_bqnk_as_type - active-set estimation method ("none", "bertsekas")

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
  
  /* XH: initialize the epsilon */

  tao->data = (void*)gn;
  gn->lambda = 1e-4;
  gn->parent = tao;
  
  ierr = MatCreate(PetscObjectComm((PetscObject)tao), &gn->H);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(gn->H, "tao_brgn_hessian_");CHKERRQ(ierr);
  
  ierr = TaoCreate(PetscObjectComm((PetscObject)tao), &gn->subsolver);CHKERRQ(ierr);
  ierr = TaoSetType(gn->subsolver, TAOBNLS);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(gn->subsolver, "tao_brgn_subsolver_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TaoBRGNGetSubsolver - Get the pointer to the subsolver inside BRGN

  Collective on Tao

  Level: developer
  
  Input Parameters:
+  tao - the Tao solver context
-  subsolver - the Tao sub-solver context
@*/
PetscErrorCode TaoBRGNGetSubsolver(Tao tao, Tao *subsolver)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  
  PetscFunctionBegin;
  *subsolver = gn->subsolver;
  PetscFunctionReturn(0);
}

/*@C
  TaoBRGNSetTikhonovLambda - Set the Tikhonov regularization factor for the Gauss-Newton least-squares algorithm

  Collective on Tao

  Level: developer
  
  Input Parameters:
+  tao - the Tao solver context
-  lambda - Tikhonov regularization factor
@*/
PetscErrorCode TaoBRGNSetTikhonovLambda(Tao tao, PetscReal lambda)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  
  /* Initialize epsilon here */

  PetscFunctionBegin;
  gn->lambda = lambda;
  PetscFunctionReturn(0);
}

/* XH: Add a routine to TaoBRGNSetEpsilon; follow the SetTikhonovLambda function including the comment */
/* XH: Look for BRGNSetTikhonovLambda in the rest of the code; it will appear in a header file somewhere and add TaoBRGNSetEpsilon to that header with the same format */
/* XH: Need to add a line to the ftn-custom for the TaoBRGNSetEpsilon function */
