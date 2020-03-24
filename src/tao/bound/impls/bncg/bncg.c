#include <petsctaolinesearch.h>
#include <../src/tao/bound/impls/bncg/bncg.h>
#include <petscksp.h>

#define CG_GradientDescent      0
#define CG_HestenesStiefel      1
#define CG_FletcherReeves       2
#define CG_PolakRibierePolyak   3
#define CG_PolakRibierePlus     4
#define CG_DaiYuan              5
#define CG_HagerZhang           6
#define CG_DaiKou               7
#define CG_KouDai               8
#define CG_SSML_BFGS            9
#define CG_SSML_DFP             10
#define CG_SSML_BROYDEN         11
#define CG_PCGradientDescent    12
#define CGTypes                 13

static const char *CG_Table[64] = {"gd", "hs", "fr", "pr", "prp", "dy", "hz", "dk", "kd", "ssml_bfgs", "ssml_dfp", "ssml_brdn", "pcgd"};

#define CG_AS_NONE       0
#define CG_AS_BERTSEKAS  1
#define CG_AS_SIZE       2

static const char *CG_AS_TYPE[64] = {"none", "bertsekas"};

PetscErrorCode TaoBNCGSetRecycleFlag(Tao tao, PetscBool recycle)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;

  PetscFunctionBegin;
  cg->recycle = recycle;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGEstimateActiveSet(Tao tao, PetscInt asType)
{
  PetscErrorCode               ierr;
  TAO_BNCG                     *cg = (TAO_BNCG *)tao->data;

  PetscFunctionBegin;
  ierr = ISDestroy(&cg->inactive_old);CHKERRQ(ierr);
  if (cg->inactive_idx) {
    ierr = ISDuplicate(cg->inactive_idx, &cg->inactive_old);CHKERRQ(ierr);
    ierr = ISCopy(cg->inactive_idx, cg->inactive_old);CHKERRQ(ierr);
  }
  switch (asType) {
  case CG_AS_NONE:
    ierr = ISDestroy(&cg->inactive_idx);CHKERRQ(ierr);
    ierr = VecWhichInactive(tao->XL, tao->solution, cg->unprojected_gradient, tao->XU, PETSC_TRUE, &cg->inactive_idx);CHKERRQ(ierr);
    ierr = ISDestroy(&cg->active_idx);CHKERRQ(ierr);
    ierr = ISComplementVec(cg->inactive_idx, tao->solution, &cg->active_idx);CHKERRQ(ierr);
    break;

  case CG_AS_BERTSEKAS:
    /* Use gradient descent to estimate the active set */
    ierr = VecCopy(cg->unprojected_gradient, cg->W);CHKERRQ(ierr);
    ierr = VecScale(cg->W, -1.0);CHKERRQ(ierr);
    ierr = TaoEstimateActiveBounds(tao->solution, tao->XL, tao->XU, cg->unprojected_gradient, cg->W, cg->work, cg->as_step, &cg->as_tol,
                                   &cg->active_lower, &cg->active_upper, &cg->active_fixed, &cg->active_idx, &cg->inactive_idx);CHKERRQ(ierr);
    break;

  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGBoundStep(Tao tao, PetscInt asType, Vec step)
{
  PetscErrorCode               ierr;
  TAO_BNCG                     *cg = (TAO_BNCG *)tao->data;

  PetscFunctionBegin;
  switch (asType) {
  case CG_AS_NONE:
    ierr = VecISSet(step, cg->active_idx, 0.0);CHKERRQ(ierr);
    break;

  case CG_AS_BERTSEKAS:
    ierr = TaoBoundStep(tao->solution, tao->XL, tao->XU, cg->active_lower, cg->active_upper, cg->active_fixed, 1.0, step);CHKERRQ(ierr);
    break;

  default:
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_BNCG(Tao tao)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode               ierr;
  PetscReal                    step=1.0,gnorm,gnorm2, resnorm;
  PetscInt                     nDiff;

  PetscFunctionBegin;
  /*   Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);

  /* Project the initial point onto the feasible region */
  ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);

  if (nDiff > 0 || !cg->recycle){
    ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &cg->f, cg->unprojected_gradient);CHKERRQ(ierr);
  }
  ierr = VecNorm(cg->unprojected_gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(cg->f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  /* Estimate the active set and compute the projected gradient */
  ierr = TaoBNCGEstimateActiveSet(tao, cg->as_type);CHKERRQ(ierr);

  /* Project the gradient and calculate the norm */
  ierr = VecCopy(cg->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
  ierr = VecISSet(tao->gradient, cg->active_idx, 0.0);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  gnorm2 = gnorm*gnorm;

  /* Initialize counters */
  tao->niter = 0;
  cg->ls_fails = cg->descent_error = 0;
  cg->resets = -1;
  cg->skipped_updates = cg->pure_gd_steps = 0;
  cg->iter_quad = 0;

  /* Convergence test at the starting point. */
  tao->reason = TAO_CONTINUE_ITERATING;

  ierr = VecFischer(tao->solution, cg->unprojected_gradient, tao->XL, tao->XU, cg->W);CHKERRQ(ierr);
  ierr = VecNorm(cg->W, NORM_2, &resnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
  ierr = TaoLogConvergenceHistory(tao, cg->f, resnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, cg->f, resnorm, 0.0, step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  /* Calculate initial direction. */
  if (!cg->recycle) {
    /* We are not recycling a solution/history from a past TaoSolve */
    ierr = TaoBNCGResetUpdate(tao, gnorm2);CHKERRQ(ierr);
  }
  /* Initial gradient descent step. Scaling by 1.0 also does a decent job for some problems. */
  while(1) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    ierr = TaoBNCGConductIteration(tao, gnorm);CHKERRQ(ierr);
    if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BNCG(Tao tao)
{
  TAO_BNCG         *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  if (!tao->stepdirection) {
    ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);
  }
  if (!cg->W) {
    ierr = VecDuplicate(tao->solution,&cg->W);CHKERRQ(ierr);
  }
  if (!cg->work) {
    ierr = VecDuplicate(tao->solution,&cg->work);CHKERRQ(ierr);
  }
  if (!cg->sk) {
    ierr = VecDuplicate(tao->solution,&cg->sk);CHKERRQ(ierr);
  }
  if (!cg->yk) {
    ierr = VecDuplicate(tao->gradient,&cg->yk);CHKERRQ(ierr);
  }
  if (!cg->X_old) {
    ierr = VecDuplicate(tao->solution,&cg->X_old);CHKERRQ(ierr);
  }
  if (!cg->G_old) {
    ierr = VecDuplicate(tao->gradient,&cg->G_old);CHKERRQ(ierr);
  }
  if (cg->diag_scaling){
    ierr = VecDuplicate(tao->solution,&cg->d_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution,&cg->y_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution,&cg->g_work);CHKERRQ(ierr);
  }
  if (!cg->unprojected_gradient) {
    ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient);CHKERRQ(ierr);
  }
  if (!cg->unprojected_gradient_old) {
    ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient_old);CHKERRQ(ierr);
  }
  ierr = MatLMVMAllocate(cg->B, cg->sk, cg->yk);CHKERRQ(ierr);
  if (cg->pc){
    ierr = MatLMVMSetJ0(cg->B, cg->pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BNCG(Tao tao)
{
  TAO_BNCG       *cg = (TAO_BNCG*) tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&cg->W);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->work);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->X_old);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->G_old);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->unprojected_gradient_old);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->g_work);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->d_work);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->y_work);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->sk);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->yk);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&cg->active_lower);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_upper);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_fixed);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->inactive_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->inactive_old);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->new_inactives);CHKERRQ(ierr);
  ierr = MatDestroy(&cg->B);CHKERRQ(ierr);
  if (cg->pc) {
    ierr = MatDestroy(&cg->pc);CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BNCG(PetscOptionItems *PetscOptionsObject,Tao tao)
{
    TAO_BNCG       *cg = (TAO_BNCG*)tao->data;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
    ierr = PetscOptionsHead(PetscOptionsObject,"Nonlinear Conjugate Gradient method for unconstrained optimization");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-tao_bncg_type","cg formula", "", CG_Table, CGTypes, CG_Table[cg->cg_type], &cg->cg_type,NULL);CHKERRQ(ierr);
    if (cg->cg_type != CG_SSML_BFGS){
      cg->alpha = -1.0; /* Setting defaults for non-BFGS methods. User can change it below. */
    }
    if (CG_GradientDescent == cg->cg_type){
      cg->cg_type = CG_PCGradientDescent;
      /* Set scaling equal to none or, at best, scalar scaling. */
      cg->unscaled_restart = PETSC_TRUE;
      cg->diag_scaling = PETSC_FALSE;
    }
    ierr = PetscOptionsEList("-tao_bncg_as_type","active set estimation method", "", CG_AS_TYPE, CG_AS_SIZE, CG_AS_TYPE[cg->cg_type], &cg->cg_type,NULL);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-tao_bncg_hz_eta","(developer) cutoff tolerance for HZ", "", cg->hz_eta,&cg->hz_eta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_eps","(developer) cutoff value for restarts", "", cg->epsilon,&cg->epsilon,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_dk_eta","(developer) cutoff tolerance for DK", "", cg->dk_eta,&cg->dk_eta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_xi","(developer) Parameter in the KD method", "", cg->xi,&cg->xi,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_theta", "(developer) update parameter for the Broyden method", "", cg->theta, &cg->theta, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_hz_theta", "(developer) parameter for the HZ (2006) method", "", cg->hz_theta, &cg->hz_theta, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_alpha","(developer) parameter for the scalar scaling","",cg->alpha,&cg->alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_bfgs_scale", "(developer) update parameter for bfgs/brdn CG methods", "", cg->bfgs_scale, &cg->bfgs_scale, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_dfp_scale", "(developer) update parameter for bfgs/brdn CG methods", "", cg->dfp_scale, &cg->dfp_scale, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_diag_scaling","Enable diagonal Broyden-like preconditioning","",cg->diag_scaling,&cg->diag_scaling,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_dynamic_restart","(developer) use dynamic restarts as in HZ, DK, KD","",cg->use_dynamic_restart,&cg->use_dynamic_restart,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_unscaled_restart","(developer) use unscaled gradient restarts","",cg->unscaled_restart,&cg->unscaled_restart,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_zeta", "(developer) Free parameter for the Kou-Dai method", "", cg->zeta, &cg->zeta, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tao_bncg_min_quad", "(developer) Number of iterations with approximate quadratic behavior needed for restart", "", cg->min_quad, &cg->min_quad, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tao_bncg_min_restart_num", "(developer) Number of iterations between restarts (times dimension)", "", cg->min_restart_num, &cg->min_restart_num, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_recycle","enable recycling the existing solution, gradient, and diagonal scaling vector at the start of a new TaoSolve() call","",cg->recycle,&cg->recycle,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_spaced_restart","(developer) Enable regular steepest descent restarting every fixed number of iterations","",cg->spaced_restart,&cg->spaced_restart,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_no_scaling","Disable all scaling except in restarts","",cg->no_scaling,&cg->no_scaling,NULL);CHKERRQ(ierr);
    if (cg->no_scaling){
      cg->diag_scaling = PETSC_FALSE;
      cg->alpha = -1.0;
    }
    if (cg->alpha == -1.0 && cg->cg_type == CG_KouDai && !cg->diag_scaling){ /* Some more default options that appear to be good. */
      cg->neg_xi = PETSC_TRUE;
    }
    ierr = PetscOptionsBool("-tao_bncg_neg_xi","(developer) Use negative xi when it might be a smaller descent direction than necessary","",cg->neg_xi,&cg->neg_xi,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_as_tol", "(developer) initial tolerance used when estimating actively bounded variables","",cg->as_tol,&cg->as_tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_as_step", "(developer) step length used when estimating actively bounded variables","",cg->as_step,&cg->as_step,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_delta_min", "(developer) minimum scaling factor used for scaled gradient restarts","",cg->delta_min,&cg->delta_min,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_delta_max", "(developer) maximum scaling factor used for scaled gradient restarts","",cg->delta_max,&cg->delta_max,NULL);CHKERRQ(ierr);

   ierr = PetscOptionsTail();CHKERRQ(ierr);
   ierr = MatSetFromOptions(cg->B);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BNCG(Tao tao, PetscViewer viewer)
{
  PetscBool      isascii;
  TAO_BNCG       *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "CG Type: %s\n", CG_Table[cg->cg_type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Skipped Stepdirection Updates: %i\n", cg->skipped_updates);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Scaled gradient steps: %i\n", cg->resets);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Pure gradient steps: %i\n", cg->pure_gd_steps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Not a descent direction: %i\n", cg->descent_error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Line search fails: %i\n", cg->ls_fails);CHKERRQ(ierr);
    if (cg->diag_scaling){
      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
      if (isascii) {
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
        ierr = MatView(cg->B, viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGComputeScalarScaling(PetscReal yty, PetscReal yts, PetscReal sts, PetscReal *scale, PetscReal alpha)
{
  PetscReal            a, b, c, sig1, sig2;

  PetscFunctionBegin;
  *scale = 0.0;

  if (1.0 == alpha){
    *scale = yts/yty;
  } else if (0.0 == alpha){
    *scale = sts/yts;
  }
  else if (-1.0 == alpha) *scale = 1.0;
  else {
    a = yty;
    b = yts;
    c = sts;
    a *= alpha;
    b *= -(2.0*alpha - 1.0);
    c *= alpha - 1.0;
    sig1 = (-b + PetscSqrtReal(b*b - 4.0*a*c))/(2.0*a);
    sig2 = (-b - PetscSqrtReal(b*b - 4.0*a*c))/(2.0*a);
    /* accept the positive root as the scalar */
    if (sig1 > 0.0) {
      *scale = sig1;
    } else if (sig2 > 0.0) {
      *scale = sig2;
    } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_CONV_FAILED, "Cannot find positive scalar");
    }
  }
  PetscFunctionReturn(0);
}

/*MC
  TAOBNCG - Bound-constrained Nonlinear Conjugate Gradient method.

  Options Database Keys:
+ -tao_bncg_recycle - enable recycling the latest calculated gradient vector in subsequent TaoSolve() calls (currently disabled)
. -tao_bncg_eta <r> - restart tolerance
. -tao_bncg_type <taocg_type> - cg formula
. -tao_bncg_as_type <none,bertsekas> - active set estimation method
. -tao_bncg_as_tol <r> - tolerance used in Bertsekas active-set estimation
. -tao_bncg_as_step <r> - trial step length used in Bertsekas active-set estimation
. -tao_bncg_eps <r> - cutoff used for determining whether or not we restart based on steplength each iteration, as well as determining whether or not we continue using the last stepdirection. Defaults to machine precision.
. -tao_bncg_theta <r> - convex combination parameter for the Broyden method
. -tao_bncg_hz_eta <r> - cutoff tolerance for the beta term in the HZ, DK methods
. -tao_bncg_dk_eta <r> - cutoff tolerance for the beta term in the HZ, DK methods
. -tao_bncg_xi <r> - Multiplicative constant of the gamma term in the KD method
. -tao_bncg_hz_theta <r> - Multiplicative constant of the theta term for the HZ method
. -tao_bncg_bfgs_scale <r> - Scaling parameter of the bfgs contribution to the scalar Broyden method
. -tao_bncg_dfp_scale <r> - Scaling parameter of the dfp contribution to the scalar Broyden method
. -tao_bncg_diag_scaling <b> - Whether or not to use diagonal initialization/preconditioning for the CG methods. Default True.
. -tao_bncg_dynamic_restart <b> - use dynamic restart strategy in the HZ, DK, KD methods
. -tao_bncg_unscaled_restart <b> - whether or not to scale the gradient when doing gradient descent restarts
. -tao_bncg_zeta <r> - Scaling parameter in the KD method
. -tao_bncg_delta_min <r> - Minimum bound for rescaling during restarted gradient descent steps
. -tao_bncg_delta_max <r> - Maximum bound for rescaling during restarted gradient descent steps
. -tao_bncg_min_quad <i> - Number of quadratic-like steps in a row necessary to do a dynamic restart
. -tao_bncg_min_restart_num <i> - This number, x, makes sure there is a gradient descent step every x*n iterations, where n is the dimension of the problem
. -tao_bncg_spaced_restart <b> - whether or not to do gradient descent steps every x*n iterations
. -tao_bncg_no_scaling <b> - If true, eliminates all scaling, including defaults.
- -tao_bncg_neg_xi <b> - Whether or not to use negative xi in the KD method under certain conditions

  Notes:
    CG formulas are:
+ "gd" - Gradient Descent
. "fr" - Fletcher-Reeves
. "pr" - Polak-Ribiere-Polyak
. "prp" - Polak-Ribiere-Plus
. "hs" - Hestenes-Steifel
. "dy" - Dai-Yuan
. "ssml_bfgs" - Self-Scaling Memoryless BFGS
. "ssml_dfp"  - Self-Scaling Memoryless DFP
. "ssml_brdn" - Self-Scaling Memoryless Broyden
. "hz" - Hager-Zhang (CG_DESCENT 5.3)
. "dk" - Dai-Kou (2013)
- "kd" - Kou-Dai (2015)

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_BNCG(Tao tao)
{
  TAO_BNCG       *cg;
  const char     *morethuente_type = TAOLINESEARCHMT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetUp_BNCG;
  tao->ops->solve = TaoSolve_BNCG;
  tao->ops->view = TaoView_BNCG;
  tao->ops->setfromoptions = TaoSetFromOptions_BNCG;
  tao->ops->destroy = TaoDestroy_BNCG;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  /*  Note: nondefault values should be used for nonlinear conjugate gradient  */
  /*  method.  In particular, gtol should be less that 0.5; the value used in  */
  /*  Nocedal and Wright is 0.10.  We use the default values for the  */
  /*  linesearch because it seems to work better. */
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch, tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);

  ierr = PetscNewLog(tao,&cg);CHKERRQ(ierr);
  tao->data = (void*)cg;
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)tao), &cg->B);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)cg->B, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(cg->B, "tao_bncg_");CHKERRQ(ierr);
  ierr = MatSetType(cg->B, MATLMVMDIAGBROYDEN);CHKERRQ(ierr);

  cg->pc = NULL;

  cg->dk_eta = 0.5;
  cg->hz_eta = 0.4;
  cg->dynamic_restart = PETSC_FALSE;
  cg->unscaled_restart = PETSC_FALSE;
  cg->no_scaling = PETSC_FALSE;
  cg->delta_min = 1e-7;
  cg->delta_max = 100;
  cg->theta = 1.0;
  cg->hz_theta = 1.0;
  cg->dfp_scale = 1.0;
  cg->bfgs_scale = 1.0;
  cg->zeta = 0.1;
  cg->min_quad = 6;
  cg->min_restart_num = 6; /* As in CG_DESCENT and KD2015*/
  cg->xi = 1.0;
  cg->neg_xi = PETSC_TRUE;
  cg->spaced_restart = PETSC_FALSE;
  cg->tol_quad = 1e-8;
  cg->as_step = 0.001;
  cg->as_tol = 0.001;
  cg->eps_23 = PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0); /* Just a little tighter*/
  cg->as_type = CG_AS_BERTSEKAS;
  cg->cg_type = CG_SSML_BFGS;
  cg->recycle = PETSC_FALSE;
  cg->alpha = 1.0;
  cg->diag_scaling = PETSC_TRUE;
  PetscFunctionReturn(0);
}


PetscErrorCode TaoBNCGResetUpdate(Tao tao, PetscReal gnormsq)
{
   TAO_BNCG          *cg = (TAO_BNCG*)tao->data;
   PetscErrorCode    ierr;
   PetscReal         scaling;

   PetscFunctionBegin;
   ++cg->resets;
   scaling = 2.0 * PetscMax(1.0, PetscAbsScalar(cg->f)) / PetscMax(gnormsq, cg->eps_23);
   scaling = PetscMin(cg->delta_max, PetscMax(cg->delta_min, scaling));
   if (cg->unscaled_restart) {
     scaling = 1.0;
     ++cg->pure_gd_steps;
   }
   ierr = VecAXPBY(tao->stepdirection, -scaling, 0.0, tao->gradient);CHKERRQ(ierr);
   /* Also want to reset our diagonal scaling with each restart */
   if (cg->diag_scaling) {
     ierr = MatLMVMReset(cg->B, PETSC_FALSE);CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
 }

PetscErrorCode TaoBNCGCheckDynamicRestart(Tao tao, PetscReal stepsize, PetscReal gd, PetscReal gd_old, PetscBool *dynrestart, PetscReal fold)
{
   TAO_BNCG          *cg = (TAO_BNCG*)tao->data;
   PetscReal         quadinterp;

   PetscFunctionBegin;
   if (cg->f < cg->min_quad/10) {
     *dynrestart = PETSC_FALSE;
     PetscFunctionReturn(0);
   } /* just skip this since this strategy doesn't work well for functions near zero */
   quadinterp = 2.0*(cg->f - fold)/(stepsize*(gd + gd_old));
   if (PetscAbs(quadinterp - 1.0) < cg->tol_quad) ++cg->iter_quad;
   else {
     cg->iter_quad = 0;
     *dynrestart = PETSC_FALSE;
   }
   if (cg->iter_quad >= cg->min_quad) {
     cg->iter_quad = 0;
     *dynrestart = PETSC_TRUE;
   }
   PetscFunctionReturn(0);
 }

PETSC_INTERN PetscErrorCode TaoBNCGStepDirectionUpdate(Tao tao, PetscReal gnorm2, PetscReal step, PetscReal fold, PetscReal gnorm2_old, PetscReal dnorm, PetscBool pcgd_fallback)
{
  TAO_BNCG          *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode    ierr;
  PetscReal         gamma = 1.0, tau_k, beta;
  PetscReal         tmp = 1.0, ynorm, ynorm2 = 1.0, snorm = 1.0, dk_yk=1.0, gd;
  PetscReal         gkp1_yk, gd_old, tau_bfgs, tau_dfp, gkp1D_yk, gtDg;
  PetscInt          dim;
  PetscBool         cg_restart = PETSC_FALSE;
  PetscFunctionBegin;

  /* Local curvature check to see if we need to restart */
  if (tao->niter >= 1 || cg->recycle){
    ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient);CHKERRQ(ierr);
    ierr = VecNorm(cg->yk, NORM_2, &ynorm);CHKERRQ(ierr);
    ynorm2 = ynorm*ynorm;
    ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk);CHKERRQ(ierr);
    if (step*dnorm < PETSC_MACHINE_EPSILON || step*dk_yk < PETSC_MACHINE_EPSILON){
      cg_restart = PETSC_TRUE;
      ++cg->skipped_updates;
    }
    if (cg->spaced_restart) {
      ierr = VecGetSize(tao->gradient, &dim);CHKERRQ(ierr);
      if (tao->niter % (dim*cg->min_restart_num)) cg_restart = PETSC_TRUE;
    }
  }
  /* If the user wants regular restarts, do it every 6n iterations, where n=dimension */
  if (cg->spaced_restart){
    ierr = VecGetSize(tao->gradient, &dim);CHKERRQ(ierr);
    if (0 == tao->niter % (6*dim)) cg_restart = PETSC_TRUE;
  }
  /* Compute the diagonal scaling vector if applicable */
  if (cg->diag_scaling) {
    ierr = MatLMVMUpdate(cg->B, tao->solution, tao->gradient);CHKERRQ(ierr);
  }

  /* A note on diagonal scaling (to be added to paper):
   For the FR, PR, PRP, and DY methods, the diagonally scaled versions
   must be derived as a preconditioned CG method rather than as
   a Hessian initialization like in the Broyden methods. */

  /* In that case, one writes the objective function as 
   f(x) \equiv f(Ay). Gradient evaluations yield g(x_k) = A g(Ay_k) = A g(x_k). 
   Furthermore, the direction d_k \equiv (x_k - x_{k-1})/step according to 
   HZ (2006) becomes A^{-1} d_k, such that d_k^T g_k remains the 
   same under preconditioning. Note that A is diagonal, such that A^T = A. */

  /* This yields questions like what the dot product d_k^T y_k 
   should look like. HZ mistakenly treats that as the same under 
   preconditioning, but that is not necessarily true. */

  /* Observe y_k \equiv g_k - g_{k-1}, and under the P.C. transformation, 
   we get d_k^T y_k = (d_k^T A_k^{-T} A_k g_k - d_k^T A_k^{-T} A_{k-1} g_{k-1}), 
   yielding d_k^T y_k = d_k^T g_k - d_k^T (A_k^{-T} A_{k-1} g_{k-1}), which is 
   NOT the same if our preconditioning matrix is updated between iterations. 
   This same issue is found when considering dot products of the form g_{k+1}^T y_k. */

  /* Compute CG step direction */
  if (cg_restart) {
    ierr = TaoBNCGResetUpdate(tao, gnorm2);CHKERRQ(ierr);
  } else if (pcgd_fallback) {
    /* Just like preconditioned CG */
    ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
    ierr = VecAXPBY(tao->stepdirection, -1.0, 0.0, cg->g_work);CHKERRQ(ierr);
  } else if (ynorm2 > PETSC_MACHINE_EPSILON) {
    switch (cg->cg_type) {
    case CG_PCGradientDescent:
      if (!cg->diag_scaling){
        if (!cg->no_scaling){
        cg->sts = step*step*dnorm*dnorm;
        ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, cg->sts, &tau_k, cg->alpha);CHKERRQ(ierr);
        } else {
          tau_k = 1.0;
          ++cg->pure_gd_steps;
        }
        ierr = VecAXPBY(tao->stepdirection, -tau_k, 0.0, tao->gradient);CHKERRQ(ierr);
      } else {
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = VecAXPBY(tao->stepdirection, -1.0, 0.0, cg->g_work);CHKERRQ(ierr);
      }
      break;

    case CG_HestenesStiefel:
      /* Classic Hestenes-Stiefel method, modified with scalar and diagonal preconditioning. */
      if (!cg->diag_scaling){
        cg->sts = step*step*dnorm*dnorm;
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
        ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, cg->sts, &tau_k, cg->alpha);CHKERRQ(ierr);
        beta = tau_k*gkp1_yk/dk_yk;
        ierr = VecAXPBY(tao->stepdirection, -tau_k, beta, tao->gradient);CHKERRQ(ierr);
      } else {
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = VecDot(cg->yk, cg->g_work, &gkp1_yk);CHKERRQ(ierr);
        beta = gkp1_yk/dk_yk;
        ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work);CHKERRQ(ierr);
      }
      break;

    case CG_FletcherReeves:
      ierr = VecDot(cg->G_old, cg->G_old, &gnorm2_old);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient);CHKERRQ(ierr);
      ierr = VecNorm(cg->yk, NORM_2, &ynorm);CHKERRQ(ierr);
      ynorm2 = ynorm*ynorm;
      ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk);CHKERRQ(ierr);
      if (!cg->diag_scaling){
        ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, step*step*dnorm*dnorm, &tau_k, cg->alpha);CHKERRQ(ierr);
        beta = tau_k*gnorm2/gnorm2_old;
        ierr = VecAXPBY(tao->stepdirection, -tau_k, beta, tao->gradient);CHKERRQ(ierr);
      } else {
        ierr = VecDot(cg->G_old, cg->g_work, &gnorm2_old);CHKERRQ(ierr); /* Before it's updated */
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = VecDot(tao->gradient, cg->g_work, &tmp);CHKERRQ(ierr);
        beta = tmp/gnorm2_old;
        ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work);CHKERRQ(ierr);
      }
      break;

    case CG_PolakRibierePolyak:
      snorm = step*dnorm;
      if (!cg->diag_scaling){
        ierr = VecDot(cg->G_old, cg->G_old, &gnorm2_old);CHKERRQ(ierr);
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
        ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, snorm*snorm, &tau_k, cg->alpha);CHKERRQ(ierr);
        beta = tau_k*gkp1_yk/gnorm2_old;
        ierr = VecAXPBY(tao->stepdirection, -tau_k, beta, tao->gradient);CHKERRQ(ierr);
      } else {
        ierr = VecDot(cg->G_old, cg->g_work, &gnorm2_old);CHKERRQ(ierr);
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = VecDot(cg->g_work, cg->yk, &gkp1_yk);CHKERRQ(ierr);
        beta = gkp1_yk/gnorm2_old;
        ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work);CHKERRQ(ierr);
      }
      break;

    case CG_PolakRibierePlus:
      ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient);CHKERRQ(ierr);
      ierr = VecNorm(cg->yk, NORM_2, &ynorm);CHKERRQ(ierr);
      ynorm2 = ynorm*ynorm;
      if (!cg->diag_scaling){
        ierr = VecDot(cg->G_old, cg->G_old, &gnorm2_old);CHKERRQ(ierr);
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
        ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, snorm*snorm, &tau_k, cg->alpha);CHKERRQ(ierr);
        beta = tau_k*gkp1_yk/gnorm2_old;
        beta = PetscMax(beta, 0.0);
        ierr = VecAXPBY(tao->stepdirection, -tau_k, beta, tao->gradient);CHKERRQ(ierr);
      } else {
        ierr = VecDot(cg->G_old, cg->g_work, &gnorm2_old);CHKERRQ(ierr); /* Old gtDg */
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = VecDot(cg->g_work, cg->yk, &gkp1_yk);CHKERRQ(ierr);
        beta = gkp1_yk/gnorm2_old;
        beta = PetscMax(beta, 0.0);
        ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work);CHKERRQ(ierr);
      }
      break;

    case CG_DaiYuan:
      /* Dai, Yu-Hong, and Yaxiang Yuan. "A nonlinear conjugate gradient method with a strong global convergence property."
         SIAM Journal on optimization 10, no. 1 (1999): 177-182. */
      if (!cg->diag_scaling){
        ierr = VecDot(tao->stepdirection, tao->gradient, &gd);CHKERRQ(ierr);
        ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
        ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, cg->yts, &tau_k, cg->alpha);CHKERRQ(ierr);
        beta = tau_k*gnorm2/(gd - gd_old);
        ierr = VecAXPBY(tao->stepdirection, -tau_k, beta, tao->gradient);CHKERRQ(ierr);
      } else {
        ierr = MatMult(cg->B, tao->stepdirection, cg->d_work);CHKERRQ(ierr);
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = VecDot(cg->g_work, tao->gradient, &gtDg);CHKERRQ(ierr);
        ierr = VecDot(tao->stepdirection, cg->G_old, &gd_old);CHKERRQ(ierr);
        ierr = VecDot(cg->d_work, cg->g_work, &dk_yk);CHKERRQ(ierr);
        dk_yk = dk_yk - gd_old;
        beta = gtDg/dk_yk;
        ierr = VecScale(cg->d_work, beta);CHKERRQ(ierr);
        ierr = VecWAXPY(tao->stepdirection, -1.0, cg->g_work, cg->d_work);CHKERRQ(ierr);
      }
      break;

    case CG_HagerZhang:
      /* Hager, William W., and Hongchao Zhang. "Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent."
         ACM Transactions on Mathematical Software (TOMS) 32, no. 1 (2006): 113-137. */
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution);CHKERRQ(ierr);
      snorm = dnorm*step;
      cg->yts = step*dk_yk;
      if (cg->use_dynamic_restart){
        ierr = TaoBNCGCheckDynamicRestart(tao, step, gd, gd_old, &cg->dynamic_restart, fold);CHKERRQ(ierr);
      }
      if (cg->dynamic_restart){
        ierr = TaoBNCGResetUpdate(tao, gnorm2);CHKERRQ(ierr);
      } else {
        if (!cg->diag_scaling){
          ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
          ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha);CHKERRQ(ierr);
          /* Supplying cg->alpha = -1.0 will give the CG_DESCENT 5.3 special case of tau_k = 1.0 */
          tmp = gd/dk_yk;
          beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk));
          /* Bound beta as in CG_DESCENT 5.3, as implemented, with the third comparison from DK 2013 */
          beta = PetscMax(PetscMax(beta, cg->hz_eta*tau_k*gd_old/(dnorm*dnorm)), cg->dk_eta*tau_k*gd/(dnorm*dnorm));
          /* d <- -t*g + beta*t*d */
          ierr = VecAXPBY(tao->stepdirection, -tau_k, beta, tao->gradient);CHKERRQ(ierr);
        } else {
          /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
          cg->yty = ynorm2;
          cg->sts = snorm*snorm;
          /* Apply the diagonal scaling to all my vectors */
          ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
          ierr = MatSolve(cg->B, cg->yk, cg->y_work);CHKERRQ(ierr);
          ierr = MatSolve(cg->B, tao->stepdirection, cg->d_work);CHKERRQ(ierr);
          /* Construct the constant ytDgkp1 */
          ierr = VecDot(cg->yk, cg->g_work, &gkp1_yk);CHKERRQ(ierr);
          /* Construct the constant for scaling Dkyk in the update */
          tmp = gd/dk_yk;
          ierr = VecDot(cg->yk, cg->y_work, &tau_k);CHKERRQ(ierr);
          tau_k = -tau_k*gd/(dk_yk*dk_yk);
          /* beta is the constant which adds the dk contribution */
          beta = gkp1_yk/dk_yk + cg->hz_theta*tau_k; /* HZ; (1.15) from DK 2013 */
          /* From HZ2013, modified to account for diagonal scaling*/
          ierr = VecDot(cg->G_old, cg->d_work, &gd_old);CHKERRQ(ierr);
          ierr = VecDot(tao->stepdirection, cg->g_work, &gd);CHKERRQ(ierr);
          beta = PetscMax(PetscMax(beta, cg->hz_eta*gd_old/(dnorm*dnorm)), cg->dk_eta*gd/(dnorm*dnorm));
          /* Do the update */
          ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work);CHKERRQ(ierr);
        }
      }
      break;

    case CG_DaiKou:
      /* Dai, Yu-Hong, and Cai-Xia Kou. "A nonlinear conjugate gradient algorithm with an optimal property and an improved Wolfe line search." 
         SIAM Journal on Optimization 23, no. 1 (2013): 296-320. */
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution);CHKERRQ(ierr);
      snorm = step*dnorm;
      cg->yts = dk_yk*step;
      if (!cg->diag_scaling){
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
        ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha);CHKERRQ(ierr);
        /* Use cg->alpha = -1.0 to get tau_k = 1.0 as in CG_DESCENT 5.3 */
        tmp = gd/dk_yk;
        beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk) + gd/(dnorm*dnorm)) - step*gd/dk_yk;
        beta = PetscMax(PetscMax(beta, cg->hz_eta*tau_k*gd_old/(dnorm*dnorm)), cg->dk_eta*tau_k*gd/(dnorm*dnorm));
        /* d <- -t*g + beta*t*d */
        ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, 0.0, beta, tao->gradient, cg->yk);CHKERRQ(ierr);
      } else {
        /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
        cg->yty = ynorm2;
        cg->sts = snorm*snorm;
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = MatSolve(cg->B, cg->yk, cg->y_work);CHKERRQ(ierr);
        ierr = MatSolve(cg->B, tao->stepdirection, cg->d_work);CHKERRQ(ierr);
        /* Construct the constant ytDgkp1 */
        ierr = VecDot(cg->yk, cg->g_work, &gkp1_yk);CHKERRQ(ierr);
        ierr = VecDot(cg->yk, cg->y_work, &tau_k);CHKERRQ(ierr);
        tau_k = tau_k*gd/(dk_yk*dk_yk);
        tmp = gd/dk_yk;
        /* beta is the constant which adds the dk contribution */
        beta = gkp1_yk/dk_yk - step*tmp - tau_k;
        /* Update this for the last term in beta */
        ierr = VecDot(cg->y_work, tao->stepdirection, &dk_yk);CHKERRQ(ierr);
        beta += tmp*dk_yk/(dnorm*dnorm); /* projection of y_work onto dk */
        ierr = VecDot(tao->stepdirection, cg->g_work, &gd);CHKERRQ(ierr);
        ierr = VecDot(cg->G_old, cg->d_work, &gd_old);CHKERRQ(ierr);
        beta = PetscMax(PetscMax(beta, cg->hz_eta*gd_old/(dnorm*dnorm)), cg->dk_eta*gd/(dnorm*dnorm));
        /* Do the update */
        ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work);CHKERRQ(ierr);
      }
      break;

    case CG_KouDai:
      /* Kou, Cai-Xia, and Yu-Hong Dai. "A modified self-scaling memoryless Broyden–Fletcher–Goldfarb–Shanno method for unconstrained optimization."
         Journal of Optimization Theory and Applications 165, no. 1 (2015): 209-224. */
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution);CHKERRQ(ierr);
      snorm = step*dnorm;
      cg->yts = dk_yk*step;
      if (cg->use_dynamic_restart){
        ierr = TaoBNCGCheckDynamicRestart(tao, step, gd, gd_old, &cg->dynamic_restart, fold);CHKERRQ(ierr);
      }
      if (cg->dynamic_restart){
        ierr = TaoBNCGResetUpdate(tao, gnorm2);CHKERRQ(ierr);
      } else {
        if (!cg->diag_scaling){
          ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
          ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha);CHKERRQ(ierr);
          beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk)) - step*gd/dk_yk;
          if (beta < cg->zeta*tau_k*gd/(dnorm*dnorm)) /* 0.1 is KD's zeta parameter */
          {
            beta = cg->zeta*tau_k*gd/(dnorm*dnorm);
            gamma = 0.0;
          } else {
            if (gkp1_yk < 0 && cg->neg_xi) gamma = -1.0*gd/dk_yk;
            /* This seems to be very effective when there's no tau_k scaling. 
               This guarantees a large descent step every iteration, going through DK 2015 Lemma 3.1's proof but allowing for negative xi */
            else {
              gamma = cg->xi*gd/dk_yk;
            }
          }
          /* d <- -t*g + beta*t*d + t*tmp*yk */
          ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, gamma*tau_k, beta, tao->gradient, cg->yk);CHKERRQ(ierr);
        } else {
          /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
          cg->yty = ynorm2;
          cg->sts = snorm*snorm;
          ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
          ierr = MatSolve(cg->B, cg->yk, cg->y_work);CHKERRQ(ierr);
          /* Construct the constant ytDgkp1 */
          ierr = VecDot(cg->yk, cg->g_work, &gkp1D_yk);CHKERRQ(ierr);
          /* Construct the constant for scaling Dkyk in the update */
          gamma = gd/dk_yk;
          /* tau_k = -ytDy/(ytd)^2 * gd */
          ierr = VecDot(cg->yk, cg->y_work, &tau_k);CHKERRQ(ierr);
          tau_k = tau_k*gd/(dk_yk*dk_yk);
          /* beta is the constant which adds the d_k contribution */
          beta = gkp1D_yk/dk_yk - step*gamma - tau_k;
          /* Here is the requisite check */
          ierr = VecDot(tao->stepdirection, cg->g_work, &tmp);CHKERRQ(ierr);
          if (cg->neg_xi){
            /* modified KD implementation */
            if (gkp1D_yk/dk_yk < 0) gamma = -1.0*gd/dk_yk;
            else {
              gamma = cg->xi*gd/dk_yk;
            }
            if (beta < cg->zeta*tmp/(dnorm*dnorm)){
              beta = cg->zeta*tmp/(dnorm*dnorm);
              gamma = 0.0;
            }
          } else { /* original KD 2015 implementation */
            if (beta < cg->zeta*tmp/(dnorm*dnorm)) {
              beta = cg->zeta*tmp/(dnorm*dnorm);
              gamma = 0.0;
            } else {
              gamma = cg->xi*gd/dk_yk;
            }
          }
          /* Do the update in two steps */
          ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work);CHKERRQ(ierr);
          ierr = VecAXPY(tao->stepdirection, gamma, cg->y_work);CHKERRQ(ierr);
        }
      }
      break;

    case CG_SSML_BFGS:
      /* Perry, J. M. "A class of conjugate gradient algorithms with a two-step variable-metric memory."
         Discussion Papers 269 (1977). */
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution);CHKERRQ(ierr);
      snorm = step*dnorm;
      cg->yts = dk_yk*step;
      cg->yty = ynorm2;
      cg->sts = snorm*snorm;
      if (!cg->diag_scaling){
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
        ierr = TaoBNCGComputeScalarScaling(cg->yty, cg->yts, cg->sts, &tau_k, cg->alpha);CHKERRQ(ierr);
        tmp = gd/dk_yk;
        beta = tau_k*(gkp1_yk/dk_yk - cg->yty*gd/(dk_yk*dk_yk)) - step*tmp;
        /* d <- -t*g + beta*t*d + t*tmp*yk */
        ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, tmp*tau_k, beta, tao->gradient, cg->yk);CHKERRQ(ierr);
      } else {
        /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = MatSolve(cg->B, cg->yk, cg->y_work);CHKERRQ(ierr);
        /* compute scalar gamma */
        ierr = VecDot(cg->g_work, cg->yk, &gkp1_yk);CHKERRQ(ierr);
        ierr = VecDot(cg->y_work, cg->yk, &tmp);CHKERRQ(ierr);
        gamma = gd/dk_yk;
        /* Compute scalar beta */
        beta = (gkp1_yk/dk_yk - gd*tmp/(dk_yk*dk_yk)) - step*gd/dk_yk;
        /* Compute stepdirection d_kp1 = gamma*Dkyk + beta*dk - Dkgkp1 */
        ierr = VecAXPBYPCZ(tao->stepdirection, -1.0, gamma, beta, cg->g_work, cg->y_work);CHKERRQ(ierr);
      }
      break;

    case CG_SSML_DFP:
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution);CHKERRQ(ierr);
      snorm = step*dnorm;
      cg->yts = dk_yk*step;
      cg->yty = ynorm2;
      cg->sts = snorm*snorm;
      if (!cg->diag_scaling){
        /* Instead of a regular convex combination, we will solve a quadratic formula. */
        ierr = TaoBNCGComputeScalarScaling(cg->yty, cg->yts, cg->sts, &tau_k, cg->alpha);CHKERRQ(ierr);
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
        tau_k = cg->dfp_scale*tau_k;
        tmp = tau_k*gkp1_yk/cg->yty;
        beta = -step*gd/dk_yk;
        /* d <- -t*g + beta*d + tmp*yk */
        ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, tmp, beta, tao->gradient, cg->yk);CHKERRQ(ierr);
      } else {
        /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless DFP step */
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = MatSolve(cg->B, cg->yk, cg->y_work);CHKERRQ(ierr);
        /* compute scalar gamma */
        ierr = VecDot(cg->g_work, cg->yk, &gkp1_yk);CHKERRQ(ierr);
        ierr = VecDot(cg->y_work, cg->yk, &tmp);CHKERRQ(ierr);
        gamma = (gkp1_yk/tmp);
        /* Compute scalar beta */
        beta = -step*gd/dk_yk;
        /* Compute stepdirection d_kp1 = gamma*Dkyk + beta*dk - Dkgkp1 */
        ierr = VecAXPBYPCZ(tao->stepdirection, -1.0, gamma, beta, cg->g_work, cg->y_work);CHKERRQ(ierr);
      }
      break;

    case CG_SSML_BROYDEN:
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution);CHKERRQ(ierr);
      snorm = step*dnorm;
      cg->yts = step*dk_yk;
      cg->yty = ynorm2;
      cg->sts = snorm*snorm;
      if (!cg->diag_scaling){
        /* Instead of a regular convex combination, we will solve a quadratic formula. */
        ierr = TaoBNCGComputeScalarScaling(cg->yty, step*dk_yk, snorm*snorm, &tau_bfgs, cg->bfgs_scale);CHKERRQ(ierr);
        ierr = TaoBNCGComputeScalarScaling(cg->yty, step*dk_yk, snorm*snorm, &tau_dfp, cg->dfp_scale);CHKERRQ(ierr);
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk);CHKERRQ(ierr);
        tau_k = cg->theta*tau_bfgs + (1.0-cg->theta)*tau_dfp;
        /* If bfgs_scale = 1.0, it should reproduce the bfgs tau_bfgs. If bfgs_scale = 0.0,
           it should reproduce the tau_dfp scaling. Same with dfp_scale.   */
        tmp = cg->theta*tau_bfgs*gd/dk_yk + (1-cg->theta)*tau_dfp*gkp1_yk/cg->yty;
        beta = cg->theta*tau_bfgs*(gkp1_yk/dk_yk - cg->yty*gd/(dk_yk*dk_yk)) - step*gd/dk_yk;
        /* d <- -t*g + beta*d + tmp*yk */
        ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, tmp, beta, tao->gradient, cg->yk);CHKERRQ(ierr);
      } else {
        /* We have diagonal scaling enabled */
        ierr = MatSolve(cg->B, tao->gradient, cg->g_work);CHKERRQ(ierr);
        ierr = MatSolve(cg->B, cg->yk, cg->y_work);CHKERRQ(ierr);
        /* compute scalar gamma */
        ierr = VecDot(cg->g_work, cg->yk, &gkp1_yk);CHKERRQ(ierr);
        ierr = VecDot(cg->y_work, cg->yk, &tmp);CHKERRQ(ierr);
        gamma = cg->theta*gd/dk_yk + (1-cg->theta)*(gkp1_yk/tmp);
        /* Compute scalar beta */
        beta = cg->theta*(gkp1_yk/dk_yk - gd*tmp/(dk_yk*dk_yk)) - step*gd/dk_yk;
        /* Compute stepdirection dkp1 = gamma*Dkyk + beta*dk - Dkgkp1 */
        ierr = VecAXPBYPCZ(tao->stepdirection, -1.0, gamma, beta, cg->g_work, cg->y_work);CHKERRQ(ierr);
      }
      break;

    default:
      break;

    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoBNCGConductIteration(Tao tao, PetscReal gnorm)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode               ierr;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    step=1.0,gnorm2,gd,dnorm=0.0;
  PetscReal                    gnorm2_old,f_old,resnorm, gnorm_old;
  PetscBool                    pcgd_fallback = PETSC_FALSE;

  PetscFunctionBegin;
  /* We are now going to perform a line search along the direction. */
  /* Store solution and gradient info before it changes */
  ierr = VecCopy(tao->solution, cg->X_old);CHKERRQ(ierr);
  ierr = VecCopy(tao->gradient, cg->G_old);CHKERRQ(ierr);
  ierr = VecCopy(cg->unprojected_gradient, cg->unprojected_gradient_old);CHKERRQ(ierr);

  gnorm_old = gnorm;
  gnorm2_old = gnorm_old*gnorm_old;
  f_old = cg->f;
  /* Perform bounded line search. If we are recycling a solution from a previous */
  /* TaoSolve, then we want to immediately skip to calculating a new direction rather than performing a linesearch */
  if (!(cg->recycle && 0 == tao->niter)){
    /* Above logic: the below code happens every iteration, except for the first iteration of a recycled TaoSolve */
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0);CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &cg->f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

    /*  Check linesearch failure */
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      ++cg->ls_fails;
      if (cg->cg_type == CG_GradientDescent){
        /* Nothing left to do but fail out of the optimization */
        step = 0.0;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      } else {
        /* Restore previous point, perform preconditioned GD and regular GD steps at the last good point */
        ierr = VecCopy(cg->X_old, tao->solution);CHKERRQ(ierr);
        ierr = VecCopy(cg->G_old, tao->gradient);CHKERRQ(ierr);
        ierr = VecCopy(cg->unprojected_gradient_old, cg->unprojected_gradient);CHKERRQ(ierr);
        gnorm = gnorm_old;
        gnorm2 = gnorm2_old;
        cg->f = f_old;

        /* Fall back on preconditioned CG (so long as you're not already using it) */
        if (cg->cg_type != CG_PCGradientDescent && cg->diag_scaling){
          pcgd_fallback = PETSC_TRUE;
          ierr = TaoBNCGStepDirectionUpdate(tao, gnorm2, step, f_old, gnorm2_old, dnorm, pcgd_fallback);CHKERRQ(ierr);

          ierr = TaoBNCGResetUpdate(tao, gnorm2);CHKERRQ(ierr);
          ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);

          ierr = TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0);CHKERRQ(ierr);
          ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &cg->f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
          ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

          pcgd_fallback = PETSC_FALSE;
          if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER){
            /* Going to perform a regular gradient descent step. */
            ++cg->ls_fails;
            step = 0.0;
          }
        }
        /* Fall back on the scaled gradient step */
        if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER){
          ++cg->ls_fails;
          ierr = TaoBNCGResetUpdate(tao, gnorm2);CHKERRQ(ierr);
          ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);
          ierr = TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0);CHKERRQ(ierr);
          ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &cg->f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
          ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
        }

        if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER){
          /* Nothing left to do but fail out of the optimization */
          ++cg->ls_fails;
          step = 0.0;
          tao->reason = TAO_DIVERGED_LS_FAILURE;
        } else {
          /* One of the fallbacks worked. Set them both back equal to false. */
          pcgd_fallback = PETSC_FALSE;
        }
      }
    }
    /* Convergence test for line search failure */
    if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

    /* Standard convergence test */
    ierr = VecFischer(tao->solution, cg->unprojected_gradient, tao->XL, tao->XU, cg->W);CHKERRQ(ierr);
    ierr = VecNorm(cg->W, NORM_2, &resnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
    ierr = TaoLogConvergenceHistory(tao, cg->f, resnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao, tao->niter, cg->f, resnorm, 0.0, step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
    if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  }
  /* Assert we have an updated step and we need at least one more iteration. */
  /* Calculate the next direction */
  /* Estimate the active set at the new solution */
  ierr = TaoBNCGEstimateActiveSet(tao, cg->as_type);CHKERRQ(ierr);
  /* Compute the projected gradient and its norm */
  ierr = VecCopy(cg->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
  ierr = VecISSet(tao->gradient, cg->active_idx, 0.0);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  gnorm2 = gnorm*gnorm;

  /* Calculate some quantities used in the StepDirectionUpdate. */
  ierr = VecNorm(tao->stepdirection, NORM_2, &dnorm);CHKERRQ(ierr);
  /* Update the step direction. */
  ierr = TaoBNCGStepDirectionUpdate(tao, gnorm2, step, f_old, gnorm2_old, dnorm, pcgd_fallback);CHKERRQ(ierr);
  ++tao->niter;
  ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);

  if (cg->cg_type != CG_GradientDescent) {
    /* Figure out which previously active variables became inactive this iteration */
    ierr = ISDestroy(&cg->new_inactives);CHKERRQ(ierr);
    if (cg->inactive_idx && cg->inactive_old) {
      ierr = ISDifference(cg->inactive_idx, cg->inactive_old, &cg->new_inactives);CHKERRQ(ierr);
    }
    /* Selectively reset the CG step those freshly inactive variables */
    if (cg->new_inactives) {
      ierr = VecGetSubVector(tao->stepdirection, cg->new_inactives, &cg->inactive_step);CHKERRQ(ierr);
      ierr = VecGetSubVector(cg->unprojected_gradient, cg->new_inactives, &cg->inactive_grad);CHKERRQ(ierr);
      ierr = VecCopy(cg->inactive_grad, cg->inactive_step);CHKERRQ(ierr);
      ierr = VecScale(cg->inactive_step, -1.0);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(tao->stepdirection, cg->new_inactives, &cg->inactive_step);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(cg->unprojected_gradient, cg->new_inactives, &cg->inactive_grad);CHKERRQ(ierr);
    }
    /* Verify that this is a descent direction */
    ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
    ierr = VecNorm(tao->stepdirection, NORM_2, &dnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(gd) || (gd/(dnorm*dnorm) <= -1e10 || gd/(dnorm*dnorm) >= -1e-10)) {
      /* Not a descent direction, so we reset back to projected gradient descent */
      ierr = TaoBNCGResetUpdate(tao, gnorm2);CHKERRQ(ierr);
      ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);
      ++cg->descent_error;
    } else {
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGSetH0(Tao tao, Mat H0)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)H0);CHKERRQ(ierr);
  cg->pc = H0;
  PetscFunctionReturn(0);
}
