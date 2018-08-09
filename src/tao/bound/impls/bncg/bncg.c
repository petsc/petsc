#include <petsctaolinesearch.h>
#include <../src/tao/bound/impls/bncg/bncg.h>

#define CG_GradientDescent      0
#define CG_HestenesStiefel      1
#define CG_FletcherReeves       2
#define CG_PolakRibiere         3
#define CG_PolakRibierePlus     4
#define CG_DaiYuan              5
#define CG_HagerZhang           6
#define CG_DaiKou               7
#define CG_KouDai               8
#define CG_SSML_BFGS            9
#define CG_SSML_DFP             10
#define CG_SSML_BROYDEN         11
#define CG_BFGS_Mod             12
#define CG_Types                13

static const char *CG_Table[64] = {"gd", "hs", "fr", "pr", "prp", "dy", "hz", "dk", "kd", "ssml_bfgs", "ssml_dfp", "ssml_brdn", "ssml_bfgs_mod"};

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
  PetscReal                    step=1.0,gnorm,gnorm2;
  PetscInt                     nDiff;

  PetscFunctionBegin;
  /*   Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);

  /* Project the initial point onto the feasible region */
  ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &cg->f, cg->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecNorm(cg->unprojected_gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(cg->f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");

  /* Estimate the active set and compute the projected gradient */
  ierr = TaoBNCGEstimateActiveSet(tao, cg->as_type);CHKERRQ(ierr);

  /* Project the gradient and calculate the norm */
  ierr = VecCopy(cg->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
  ierr = VecISSet(tao->gradient, cg->active_idx, 0.0);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  gnorm2 = gnorm*gnorm;

  /* Initialize counters */
  tao->niter = 0;
  cg->ls_fails = cg->broken_ortho = cg->descent_error = 0;
  cg->resets = -1;
  cg->iter_quad = 0;

  /* Convergence test at the starting point. */
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao, cg->f, gnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, cg->f, gnorm, 0.0, step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Assert that we have not converged.  Calculate initial direction. */
  /* This is where recycling goes.  The outcome of this code must be */
  /* the direction that we will use. */
  if (cg->recycle) {
  }
  /* Initial gradient descent step. Scaling by 1.0 also does a decent job for some problems.  */
  ierr = TaoBNCGResetUpdate(tao, gnorm2);

  while(1) {
    ierr = TaoBNCGConductIteration(tao, gnorm); CHKERRQ(ierr);
    if (cg->use_steffenson) ierr = TaoBNCGSteffensonAcceleration(tao); CHKERRQ(ierr);
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
    ierr = VecDuplicate(tao->solution,&cg->g_work);CHKERRQ(ierr);
  }
  if (cg->use_steffenson){
    if (!cg->steffnm1) ierr = VecDuplicate(tao->solution, &cg->steffnm1); CHKERRQ(ierr);
    if (!cg->steffn) ierr = VecDuplicate(tao->solution, &cg->steffn); CHKERRQ(ierr);
    if (!cg->steffnp1) ierr = VecDuplicate(tao->solution, &cg->steffnp1); CHKERRQ(ierr);
    if (!cg->steffva) ierr = VecDuplicate(tao->solution, &cg->steffva); CHKERRQ(ierr);
    if (!cg->steffvatmp) ierr = VecDuplicate(tao->solution, &cg->steffvatmp); CHKERRQ(ierr);
  }
  if (cg->diag_scaling){
    ierr = VecDuplicate(tao->gradient,&cg->invD);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->gradient,&cg->invDnew);CHKERRQ(ierr);
    ierr = VecSet(cg->invDnew, 1.0);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->gradient,&cg->bfgs_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->gradient,&cg->dfp_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->gradient,&cg->U);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->gradient,&cg->V);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->gradient,&cg->W);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution,&cg->d_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution,&cg->y_work);CHKERRQ(ierr);

  }
  if (!cg->unprojected_gradient) {
    ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient);CHKERRQ(ierr);
  }
  if (!cg->unprojected_gradient_old) {
    ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient_old);CHKERRQ(ierr);
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
  }
  ierr = ISDestroy(&cg->active_lower);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_upper);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_fixed);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->inactive_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->inactive_old);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->new_inactives);CHKERRQ(ierr);
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
    ierr = PetscOptionsReal("-tao_bncg_eta","cutoff tolerance for HZ", "", cg->eta,&cg->eta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_xi","Parameter in KD, HZ, and DK methods", "", cg->xi,&cg->xi,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_gamma", "stabilization term for multiple CG methods", "", cg->gamma, &cg->gamma, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_theta", "update parameter for some CG methods", "", cg->theta, &cg->theta, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_hz_theta", "parameter for the HZ (2006) method", "", cg->hz_theta, &cg->hz_theta, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_rho","(developer) update limiter in the J0 scaling","",cg->rho,&cg->rho,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_alpha","(developer) convex ratio in the J0 scaling","",cg->alpha,&cg->alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_beta","(developer) exponential factor in the diagonal J0 scaling","",cg->beta,&cg->beta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_bfgs_scale", "(developer) update parameter for bfgs/brdn CG methods", "", cg->bfgs_scale, &cg->bfgs_scale, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_dfp_scale", "(developer) update parameter for bfgs/brdn CG methods", "", cg->dfp_scale, &cg->dfp_scale, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_diag_scaling","Enable diagonal Broyden-like preconditioning","",cg->diag_scaling,&cg->diag_scaling,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_inv_sig","(developer) test parameter to invert the sigma scaling","",cg->inv_sig,&cg->inv_sig,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_dynamic_restart","use dynamic restarts as in HZ, DK, KD","",cg->use_dynamic_restart,&cg->use_dynamic_restart,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_use_steffenson","(incomplete) use vector-Steffenson acceleration","",cg->use_steffenson,&cg->use_steffenson,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_zeta", "Free parameter for the Kou-Dai method", "", cg->zeta, &cg->zeta, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tao_bncg_min_quad", "(developer) Number of iterations with approximate quadratic behavior needed for restart", "", cg->min_quad, &cg->min_quad, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tao_bncg_min_restart_num", "Number of iterations between restarts (times dimension)", "", cg->min_restart_num, &cg->min_restart_num, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_rho","(developer) descent direction tolerance", "", cg->rho,&cg->rho,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_pow","(developer) descent direction exponent", "", cg->pow,&cg->pow,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-tao_bncg_type","cg formula", "", CG_Table, CG_Types, CG_Table[cg->cg_type], &cg->cg_type,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-tao_bncg_as_type","active set estimation method", "", CG_AS_TYPE, CG_AS_SIZE, CG_AS_TYPE[cg->cg_type], &cg->cg_type,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_recycle","enable recycling the existing solution and gradient at the start of a new solve","",cg->recycle,&cg->recycle,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_spaced_restart","Enable regular steepest descent restarting every ixed number of iterations","",cg->spaced_restart,&cg->spaced_restart,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_neg_xi","(Developer) Use negative xi when it might be a smaller descent direction than necessary","",cg->neg_xi,&cg->neg_xi,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_as_tol", "initial tolerance used when estimating actively bounded variables","",cg->as_tol,&cg->as_tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_as_step", "step length used when estimating actively bounded variables","",cg->as_step,&cg->as_step,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsTail();CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer, "Resets: %i\n", cg->resets);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Broken ortho: %i\n", cg->broken_ortho);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Not a descent dir.: %i\n", cg->descent_error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Line search fails: %i\n", cg->ls_fails);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGComputeScalarScaling(PetscReal yty, PetscReal yts, PetscReal sts, PetscReal *scale, PetscReal alpha)
{
  PetscReal            a, b, c, sig1, sig2;

  PetscFunctionBegin;
  *scale = 0.0;

  if (alpha == 1.0){
    *scale = yts/yty;
  } else if (alpha == 0.5) {
    *scale = sts/yty;
  }
  else if (alpha == 0.0){
    *scale = sts/yts;
  }
  else if (alpha == -1.0) *scale = 1.0;
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
     TAOBNCG -   Bound-constrained Nonlinear Conjugate Gradient method.

   Options Database Keys:
+      -tao_bncg_recycle - enable recycling the latest calculated gradient vector in subsequent TaoSolve() calls
.      -tao_bncg_eta <r> - restart tolerance
.      -tao_bncg_type <taocg_type> - cg formula
.      -tao_bncg_as_type <none,bertsekas> - active set estimation method
.      -tao_bncg_as_tol <r> - tolerance used in Bertsekas active-set estimation
.      -tao_bncg_as_step <r> - trial step length used in Bertsekas active-set estimation

  Notes:
     CG formulas are:
         "fr" - Fletcher-Reeves
         "pr" - Polak-Ribiere
         "prp" - Polak-Ribiere-Plus
         "hs" - Hestenes-Steifel
         "dy" - Dai-Yuan
         "gd" - Gradient Descent
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

  cg->pow = 2.1;
  cg->eta = 0.5;
  cg->dynamic_restart = PETSC_FALSE;
  cg->unscaled_restart = PETSC_FALSE;
  cg->theta = 1.0;
  cg->hz_theta = 1.0;
  cg->dfp_scale = 1.0;
  cg->bfgs_scale = 1.0;
  cg->gamma = 0.4;
  cg->zeta = 0.5;
  cg->min_quad = 3;
  cg->min_restart_num = 6; /* As in CG_DESCENT and KD2015*/
  cg->xi = 1.0;
  cg->neg_xi = PETSC_FALSE;
  cg->spaced_restart = PETSC_FALSE;
  cg->tol_quad = 1e-8;
  cg->as_step = 0.001;
  cg->as_tol = 0.001;
  cg->epsilon = PETSC_MACHINE_EPSILON;
  cg->eps_23 = PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0);
  cg->as_type = CG_AS_BERTSEKAS;
  cg->cg_type = CG_SSML_BFGS;
  cg->recycle = PETSC_FALSE;
  cg->alpha = 1.0;
  cg->rho = 1.0;
  cg->beta = 0.5; /* Default a la Alp */
  cg->diag_scaling = PETSC_TRUE;
  cg->inv_sig = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGComputeDiagScaling(Tao tao, PetscReal yts, PetscReal yty){
  TAO_BNCG          *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode    ierr;
  PetscScalar       ytDy, ytDs, stDs;
  PetscReal         sigma;

  PetscFunctionBegin;
  /*  W = Hy */
  ierr = VecPointwiseMult(cg->W, cg->invD, cg->yk);CHKERRQ(ierr);
  ierr = VecDot(cg->W, cg->yk, &ytDy);CHKERRQ(ierr);
  /* Compute Hadamard product V = s*s^T */
  ierr = VecPointwiseMult(cg->V, cg->sk, cg->sk);CHKERRQ(ierr);
  /* Compute the BFGS contribution bfgs_work */
  /* first construct sDy - Hadamard product */
  ierr = VecPointwiseMult(cg->bfgs_work, cg->sk, cg->W); CHKERRQ(ierr);
  /* Now assemble the BFGS component in there - denom of yts added later */
  ierr = VecAXPBY(cg->bfgs_work, ytDy/(yts), -2.0, cg->V); CHKERRQ(ierr);
  /* Start assembling the new inverse diagonal, the pure DFP component - denom of ytDy added later */
  /* Compute Hadamard product U = (Hy)*(Hy)^T  */
  ierr = VecPointwiseMult(cg->U, cg->W, cg->W); CHKERRQ(ierr);
  /* D_{k+1} = D_k + V/yts + (1-theta)*BFGS + theta*DFP */
  ierr = VecCopy(cg->invD, cg->invDnew); CHKERRQ(ierr);
  /* The factors I left out in BFGS and DFP  */
  ierr = VecAXPBYPCZ(cg->invDnew, (1-cg->theta)/yts, -cg->theta/(ytDy), 1.0, cg->bfgs_work, cg->U); CHKERRQ(ierr);
  ierr = VecAXPY(cg->invDnew, 1/cg->yts, cg->V);
  /*  Ensure positive definite */
  ierr = VecAbs(cg->invDnew); CHKERRQ(ierr);
  /* Start with re-scaling on the newly computed diagonal */
  /* Compute y^T H^{2*beta} y */
  /* Note that VecPow has special cases tabulated for +-1.0, +-0.5, 0.0, and 2.0 */
  ierr = VecCopy(cg->invDnew, cg->work);CHKERRQ(ierr);
  ierr = VecPow(cg->work, 2.0*cg->beta);CHKERRQ(ierr);
  ierr = VecPointwiseMult(cg->work, cg->work, cg->yk);CHKERRQ(ierr);
  ierr = VecDot(cg->yk, cg->work, &ytDy);CHKERRQ(ierr);
  /* Compute y^T H^{2*beta - 1} s */
  ierr = VecCopy(cg->invDnew, cg->work);CHKERRQ(ierr);
  ierr = VecPow(cg->work, 2.0*cg->beta - 1.0);CHKERRQ(ierr);
  ierr = VecPointwiseMult(cg->work, cg->work, cg->sk);CHKERRQ(ierr);
  ierr = VecDot(cg->yk, cg->work, &ytDs);CHKERRQ(ierr);
  /* Compute s^T H^{2*beta - 2} s */
  ierr = VecCopy(cg->invDnew, cg->work);CHKERRQ(ierr);
  ierr = VecPow(cg->work, 2.0*cg->beta - 2.0);CHKERRQ(ierr);
  ierr = VecPointwiseMult(cg->work, cg->work, cg->sk);CHKERRQ(ierr);
  ierr = VecDot(cg->sk, cg->work, &stDs);CHKERRQ(ierr);
  /* Compute the diagonal scaling */
  sigma = 0.0;
  ierr = TaoBNCGComputeScalarScaling(ytDy, ytDs, stDs, &sigma, cg->alpha);

  /*  If Q has small values, then Q^(r_beta - 1) */
  /*  can have very large values.  Hence, ys_sum */
  /*  and ss_sum can be infinity.  In this case, */
  /*  sigma can either be not-a-number or infinity. */

  if (PetscIsInfOrNanReal(sigma)) {
    /*  sigma is not-a-number; skip rescaling */
  } else if (!sigma) {
    /*  sigma is zero; this is a bad case; skip rescaling */
  } else {
    /*  sigma is positive */
    ierr = VecScale(cg->invDnew, sigma);CHKERRQ(ierr);
  }

  /* Combine the old diagonal and the new diagonal using a convex limiter */
  if (cg->rho == 1.0) {
    ierr = VecCopy(cg->invDnew, cg->invD);CHKERRQ(ierr);
  } else if (cg->rho) {
    ierr = VecAXPBY(cg->invD, 1.0-cg->rho, cg->rho, cg->invDnew);CHKERRQ(ierr);
  }
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
   scaling = PetscMin(100.0, PetscMax(1e-7, scaling));
   if (cg->unscaled_restart) scaling = 1.0;
   ierr = VecAXPBY(tao->stepdirection, -scaling, 0.0, tao->gradient);CHKERRQ(ierr);
   /* Also want to reset our diagonal scaling with each restart */
   if (cg->diag_scaling) {
     ierr = VecSet(cg->invD, 1.0);CHKERRQ(ierr);
   }


   PetscFunctionReturn(0);
 }

PetscErrorCode TaoBNCGCheckDynamicRestart(Tao tao, PetscReal stepsize, PetscReal gd, PetscReal gd_old, PetscBool *dynrestart, PetscReal fold)
 {
   TAO_BNCG          *cg = (TAO_BNCG*)tao->data;
   PetscReal         quadinterp;

   PetscFunctionBegin;
   if (cg->f < cg->min_quad/10) {*dynrestart = PETSC_FALSE; PetscFunctionReturn(0);} /* just skip this since this strategy doesn't work well for functions near zero */
   quadinterp = 2*(cg->f - fold)/(stepsize*(gd + gd_old));
   if (PetscAbs(quadinterp - 1.0) < cg->tol_quad) cg->iter_quad++;
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

PETSC_INTERN PetscErrorCode TaoBNCGStepDirectionUpdate(Tao tao, PetscReal gnorm2, PetscReal step, PetscReal fold, PetscReal gnorm2_old, PetscBool cg_restart, PetscReal dnorm, PetscReal ginner){
  TAO_BNCG          *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode    ierr;
  PetscReal         gamma, tau_k, beta;
  PetscReal         tmp, ynorm, ynorm2, snorm, dk_yk, gd;
  PetscReal         gkp1_yk, gd_old, tau_bfgs, tau_dfp, gkp1D_yk;
  PetscInt          dim;

  PetscFunctionBegin;

  /* Want to implement P.C. versions eventually  */
  /* Compute CG step */
  if (cg_restart) {
    beta = 0.0;
    ++cg->resets;
    ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient);CHKERRQ(ierr);
  } else {
    switch (cg->cg_type) {
    case CG_GradientDescent:
      beta = 0.0;
      ierr = VecAXPBY(tao->stepdirection, -1.0, 0.0, tao->gradient);CHKERRQ(ierr);
      break;

    case CG_HestenesStiefel:
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
      ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
      ynorm2 = ynorm*ynorm;
      ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
      //tau_k = step*dk_yk/ynorm2;
      //beta = tau_k*(gnorm2 - ginner) / (gd - gd_old);
      beta = (gnorm2 - ginner) / (gd - gd_old);
      ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient);CHKERRQ(ierr);
      break;
    case CG_FletcherReeves:
      beta = gnorm2 / gnorm2_old;
      ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient);CHKERRQ(ierr);
      break;
    case CG_PolakRibiere:
      beta = (gnorm2 - ginner) / gnorm2_old;
      ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient);CHKERRQ(ierr);
      break;
    case CG_PolakRibierePlus:
      beta = PetscMax((gnorm2-ginner)/gnorm2_old, 0.0);
      ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient);CHKERRQ(ierr);
      break;
    case CG_DaiYuan:
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
      beta = gnorm2 / (gd - gd_old);
      ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient);CHKERRQ(ierr);
      break;
    case CG_SSML_BFGS:
      ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
      ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
      ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution); CHKERRQ(ierr);
      ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
      ierr = VecNorm(cg->sk, NORM_2, &snorm); CHKERRQ(ierr);
      ynorm2 = ynorm*ynorm;
      ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
      ierr = VecDot(cg->yk, cg->sk, &cg->yts); CHKERRQ(ierr);
      cg->yty = ynorm2;
      cg->sts = snorm*snorm;
      /* TODO: Should only need one of dk_yk and yts */
      if (ynorm2 < cg->epsilon){
        /* The gradient hasn't changed, so we should stay in the same direction as before. Don't update it or anything.*/
        if (snorm < cg->eps_23){
          /* We're making no progress. Scaled gradient descent step.*/
          ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
        }
      } else {
        if (!cg->diag_scaling){
          ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk); CHKERRQ(ierr);
          ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha); CHKERRQ(ierr);
          tmp = gd/dk_yk;
          beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk) - (step/tau_k)*gd/dk_yk);
          /* d <- -t*g + beta*t*d + t*tmp*yk */
          ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, tmp*tau_k, beta, tao->gradient, cg->yk); CHKERRQ(ierr);
        }
        else if (snorm < cg->epsilon || cg->yts < cg->epsilon) {
          /* We're making no progress. Scaled gradient descent step and reset diagonal scaling */
          ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
        } else {
          /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
          /* Compute the invD vector  */
          ierr = TaoBNCGComputeDiagScaling(tao, cg->yts, ynorm2); CHKERRQ(ierr);
          /* Apply the invD scaling to all my vectors */
          ierr = VecPointwiseMult(cg->g_work, cg->invD, tao->gradient); CHKERRQ(ierr);
          ierr = VecPointwiseMult(cg->d_work, cg->invD, tao->stepdirection); CHKERRQ(ierr);
          ierr = VecPointwiseMult(cg->y_work, cg->invD, cg->yk); CHKERRQ(ierr);
          /* Construct the constant ytDgkp1 */
          ierr = VecDot(cg->yk, cg->g_work, &gkp1_yk); CHKERRQ(ierr);
          /* Construct the constant for scaling Dkyk in the update */
          tmp = gd/dk_yk;
          /* tau_bfgs can be a parameter we tune via command line in future versions. For now, just setting to one. May instead put this inside the ComputeDiagonalScaling function... */
          cg->tau_bfgs = 1.0;
          /* tau_k = -ytDy/(ytd)^2 * gd */
          ierr = VecDot(cg->yk, cg->y_work, &tau_k);
          tau_k = -tau_k*gd/(dk_yk*dk_yk);
          /* beta is the constant which adds the dk contribution */
          beta = cg->tau_bfgs*gkp1_yk/dk_yk - step*tmp + tau_k;
          /* Do the update in two steps */
          ierr = VecAXPBY(tao->stepdirection, -cg->tau_bfgs, beta, cg->g_work); CHKERRQ(ierr);
          ierr = VecAXPY(tao->stepdirection, tmp, cg->y_work); CHKERRQ(ierr);
        }}
      break;
    case CG_SSML_DFP:

            ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
            ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
            ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution); CHKERRQ(ierr);
            ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
            ierr = VecNorm(cg->sk, NORM_2, &snorm); CHKERRQ(ierr);
            ynorm2 = ynorm*ynorm;
            ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk); CHKERRQ(ierr);
            ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
            if (ynorm < cg->epsilon){
              /* The gradient hasn't changed, so we should stay in the same direction as before. Don't update it or anything.*/
              if (snorm < cg->epsilon){
                /* We're making no progress. Gradient descent step. Not scaled by tau_k since it's' is 0 or NaN in this case.*/
                ierr = TaoBNCGResetUpdate(tao, gnorm2);
              }
        } else {

          tau_k = cg->dfp_scale*snorm*snorm/(step*dk_yk);
          tmp = tau_k*gkp1_yk/ynorm2;
          beta = -step*gd/dk_yk;

          /* d <- -t*g + beta*d + tmp*yk */
          ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, tmp, beta, tao->gradient, cg->yk); CHKERRQ(ierr);
        }
        break;
      case CG_SSML_BROYDEN:

        ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
        ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
        ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution); CHKERRQ(ierr);
        ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
        ierr = VecNorm(cg->sk, NORM_2, &snorm); CHKERRQ(ierr);
        ynorm2 = ynorm*ynorm;
        ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk); CHKERRQ(ierr);
        ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);

        if (ynorm < cg->epsilon){
          /* The gradient hasn't changed, so we should stay in the same direction as before. Don't update it or anything.*/
          if (snorm < cg->epsilon){
            /* We're making no progress. Gradient descent step.*/
            ierr = TaoBNCGResetUpdate(tao, gnorm2);
          }
        } else {
          /* Instead of a regular convex combination, we will solve a quadratic formula. */
          ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, snorm*snorm, &tau_bfgs, cg->bfgs_scale);
          ierr = TaoBNCGComputeScalarScaling(ynorm2, step*dk_yk, snorm*snorm, &tau_dfp, cg->dfp_scale);
          tau_k = cg->theta*tau_bfgs + (1.0-cg->theta)*tau_dfp;

          /* Used for the gradient */
          /* If bfgs_scale = 1.0, it should reproduce the bfgs tau_bfgs. If bfgs_scale = 0.0, it should reproduce the tau_dfp scaling. Same with dfp_scale.   */
          tmp = cg->theta*tau_bfgs*gd/dk_yk + (1-cg->theta)*tau_dfp*gkp1_yk/ynorm2;
          beta = cg->theta*tau_bfgs*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk)) - step*gd/dk_yk;
          /* d <- -t*g + beta*d + tmp*yk */
          ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, tmp, beta, tao->gradient, cg->yk); CHKERRQ(ierr);
        }
        break;

      case CG_HagerZhang:
        /* Their 2006 paper. Comes from deleting the y_k term from SSML_BFGS, introducing a theta parameter, and using a cutoff for beta. See DK 2013 pg. 315 for a review of CG_DESCENT 5.3 */
        ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
        ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
        ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
        ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution); CHKERRQ(ierr);
        ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
        ierr = VecNorm(cg->sk, NORM_2, &snorm); CHKERRQ(ierr);
        ynorm2 = ynorm*ynorm;
        ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
        ierr = VecDot(cg->yk, cg->sk, &cg->yts); CHKERRQ(ierr);
        if (cg->use_dynamic_restart){
          ierr = TaoBNCGCheckDynamicRestart(tao, step, gd, gd_old, &cg->dynamic_restart, fold); CHKERRQ(ierr);
        }
        if (ynorm2 < cg->epsilon){
         /* The gradient hasn't changed, so we should stay in the same direction as before. Don't update it or anything.*/
          if (snorm < cg->eps_23){
            /* We're making no progress. Scaled gradient descent step.*/
            ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
          }
        } else if (cg->dynamic_restart){
          ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
        } else if (tao->niter % (cg->min_restart_num*dim) == 0 && cg->spaced_restart){
          ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
        } else {
          if (!cg->diag_scaling){
            ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk); CHKERRQ(ierr);
            ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha); CHKERRQ(ierr);
            /* Supplying cg->alpha = -1.0 will give the CG_DESCENT 5.3 special case of tau_k = 1.0 */
            tmp = gd/dk_yk;
            beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk));
            /* Bound beta as in CG_DESCENT 5.3, as implemented, with the third comparison from DK 2013 */
            beta = PetscMax(PetscMax(beta, 0.4*tau_k*gd_old/(dnorm*dnorm)), 0.5*tau_k*gd/(dnorm*dnorm));
            /* d <- -t*g + beta*t*d */
            ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, 0.0, beta, tao->gradient, cg->yk); CHKERRQ(ierr);
          }
          else if (snorm < cg->epsilon || cg->yts < cg->epsilon) {
            /* We're making no progress. Scaled gradient descent step and reset diagonal scaling */
            ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
          } else {
            /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
            cg->yty = ynorm2;
            cg->sts = snorm*snorm;
            /* Compute the invD vector  */
            ierr = TaoBNCGComputeDiagScaling(tao, cg->yts, ynorm2); CHKERRQ(ierr);
            /* Apply the invD scaling to all my vectors */
            ierr = VecPointwiseMult(cg->g_work, cg->invD, tao->gradient); CHKERRQ(ierr);
            ierr = VecPointwiseMult(cg->d_work, cg->invD, tao->stepdirection); CHKERRQ(ierr);
            ierr = VecPointwiseMult(cg->y_work, cg->invD, cg->yk); CHKERRQ(ierr);
            /* Construct the constant ytDgkp1 */
            ierr = VecDot(cg->yk, cg->g_work, &gkp1_yk); CHKERRQ(ierr);
            /* Construct the constant for scaling Dkyk in the update */
            tmp = gd/dk_yk;
            /* tau_bfgs can be a parameter we tune via command line in future versions. For now, just setting to one. May instead put this inside the ComputeDiagonalScaling function... */
            cg->tau_bfgs = 1.0;
            ierr = VecDot(cg->yk, cg->y_work, &tau_k);
            tau_k = -tau_k*gd/(dk_yk*dk_yk);
            /* beta is the constant which adds the dk contribution */
            beta = cg->tau_bfgs*gkp1_yk/dk_yk + cg->hz_theta*tau_k; /* HZ; (1.15) from DK 2013 */
            /* From HZ2013, modified to account for diagonal scaling*/
            ierr = VecDot(cg->G_old, cg->d_work, &gd_old);
            ierr = VecDot(tao->stepdirection, cg->g_work, &gd);
            beta = PetscMax(PetscMax(beta, 0.4*gd_old/(dnorm*dnorm)), 0.5*gd/(dnorm*dnorm));
            /* Do the update */
            ierr = VecAXPBY(tao->stepdirection, -cg->tau_bfgs, beta, cg->g_work); CHKERRQ(ierr);
          }}
        break;
      case CG_DaiKou:
        /* 2013 paper.  */
        ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
        ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
        ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
        ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution); CHKERRQ(ierr);
        ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
        ierr = VecNorm(cg->sk, NORM_2, &snorm); CHKERRQ(ierr);
        ynorm2 = ynorm*ynorm;
        ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
        ierr = VecDot(cg->yk, cg->sk, &cg->yts); CHKERRQ(ierr);
        /* TODO: Should only need one of dk_yk and yts */
        if (ynorm2 < cg->epsilon){
         /* The gradient hasn't changed, so we should stay in the same direction as before. Don't update it or anything.*/
          if (snorm < cg->eps_23){
            /* We're making no progress. Scaled gradient descent step.*/
            ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
          }
        } else {
            if (!cg->diag_scaling){
              ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk); CHKERRQ(ierr);
              ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha);
              /* Use cg->alpha = -1.0 to get tau_k = 1.0 as in CG_DESCENT 5.3 */
              tmp = gd/dk_yk;
              beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk) - (step/tau_k)*gd/dk_yk + gd/(dnorm*dnorm));
              beta = PetscMax(PetscMax(beta, 0.4*tau_k*gd_old/(dnorm*dnorm)), 0.5*tau_k*gd/(dnorm*dnorm)); 
              /* d <- -t*g + beta*t*d */
              ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, 0.0, beta, tao->gradient, cg->yk); CHKERRQ(ierr);
            }
         else if (snorm < cg->epsilon || cg->yts < cg->epsilon) {
            /* We're making no progress. Scaled gradient descent step and reset diagonal scaling */
            ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
          } else {
            /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
            cg->yty = ynorm2;
            cg->sts = snorm*snorm;
            /* Compute the invD vector  */
            ierr = TaoBNCGComputeDiagScaling(tao, cg->yts, ynorm2); CHKERRQ(ierr);
            /* Apply the invD scaling to all my vectors */
            ierr = VecPointwiseMult(cg->g_work, cg->invD, tao->gradient); CHKERRQ(ierr);
            ierr = VecPointwiseMult(cg->d_work, cg->invD, tao->stepdirection); CHKERRQ(ierr);
            ierr = VecPointwiseMult(cg->y_work, cg->invD, cg->yk); CHKERRQ(ierr);
            /* Construct the constant ytDgkp1 */
            ierr = VecDot(cg->yk, cg->g_work, &gkp1_yk); CHKERRQ(ierr);
            /* tau_bfgs can be a parameter we tune via command line in future versions. For now, just setting to one. May instead put this inside the ComputeDiagonalScaling function... */
            cg->tau_bfgs = 1.0;
            ierr = VecDot(cg->yk, cg->y_work, &tau_k);
            tau_k = tau_k*gd/(dk_yk*dk_yk);
            tmp = gd/dk_yk;
            /* beta is the constant which adds the dk contribution */
            beta = cg->tau_bfgs*gkp1_yk/dk_yk - step*tmp - tau_k;

            /* Update this for the last term in beta */
            ierr = VecDot(cg->y_work, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
            beta += tmp*dk_yk/(dnorm*dnorm); /* projection of y_work onto dk */
            ierr = VecDot(tao->stepdirection, cg->g_work, &gd);
            ierr = VecDot(cg->G_old, cg->d_work, &gd_old);
            beta = PetscMax(PetscMax(beta, 0.4*gd_old/(dnorm*dnorm)), 0.5*gd/(dnorm*dnorm));
            /* TODO: need to change these hardcoded constants into user parameters */
            /* Do the update */
            ierr = VecAXPBY(tao->stepdirection, -cg->tau_bfgs, beta, cg->g_work); CHKERRQ(ierr);
         }}
        break;

      case CG_KouDai:

        ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
        ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
        ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
        ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution); CHKERRQ(ierr);
        ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
        ierr = VecNorm(cg->sk, NORM_2, &snorm); CHKERRQ(ierr);
        ynorm2 = ynorm*ynorm;
        ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
        ierr = VecDot(cg->yk, cg->sk, &cg->yts); CHKERRQ(ierr);
        ierr = VecGetSize(tao->gradient, &dim); CHKERRQ(ierr);
        /* TODO: Should only need one of dk_yk and yts */
        if (cg->use_dynamic_restart){
          ierr = TaoBNCGCheckDynamicRestart(tao, step, gd, gd_old, &cg->dynamic_restart, fold); CHKERRQ(ierr);
        }
        if (ynorm2 < cg->epsilon){
         /* The gradient hasn't changed, so we should stay in the same direction as before. Don't update it or anything.*/
          if (snorm < cg->eps_23){
            /* We're making no progress. Scaled gradient descent step.*/
            ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
          }
        } else if (cg->dynamic_restart){
          ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
        } else if (tao->niter % (cg->min_restart_num*dim) == 0 && cg->spaced_restart){
          ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
        } else {
            if (!cg->diag_scaling){
              ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk); CHKERRQ(ierr);
              ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha); CHKERRQ(ierr);
              beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk)) - step*gd/dk_yk;
              if (beta < cg->zeta*tau_k*gd/(dnorm*dnorm)) /* 0.1 is KD's zeta parameter */
              {
                beta = cg->zeta*tau_k*gd/(dnorm*dnorm);
                gamma = 0.0;
              } else {
                if (gkp1_yk < 0 && cg->neg_xi) gamma = -1.0*gd/dk_yk;
                /* This seems to be very effective when there's no tau_k scaling... this guarantees a large descent step every iteration, going through DK 2015 Lemma 3.1's proof but allowing for negative xi */
                else gamma = cg->xi*gd/dk_yk;
              }
              /* d <- -t*g + beta*t*d + t*tmp*yk */
              ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, gamma*tau_k, beta, tao->gradient, cg->yk); CHKERRQ(ierr);
            }
            else if (snorm < cg->epsilon || cg->yts < cg->epsilon) {
              /* We're making no progress. Scaled gradient descent step and reset diagonal scaling */
              ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
            } else {
              /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
              cg->yty = ynorm2;
              cg->sts = snorm*snorm;
              /* Compute the invD vector  */
              ierr = TaoBNCGComputeDiagScaling(tao, cg->yts, ynorm2); CHKERRQ(ierr);
              /* Apply the invD scaling to all my vectors */
              ierr = VecPointwiseMult(cg->g_work, cg->invD, tao->gradient); CHKERRQ(ierr);
              /* ierr = VecPointwiseMult(cg->d_work, cg->invD, tao->stepdirection); CHKERRQ(ierr); */
              ierr = VecPointwiseMult(cg->y_work, cg->invD, cg->yk); CHKERRQ(ierr);
              /* Construct the constant ytDgkp1 */
              ierr = VecDot(cg->yk, cg->g_work, &gkp1D_yk); CHKERRQ(ierr);
              /* Construct the constant for scaling Dkyk in the update */
              gamma = gd/dk_yk;
              /* tau_k = -ytDy/(ytd)^2 * gd */
              ierr = VecDot(cg->yk, cg->y_work, &tau_k);
              tau_k = tau_k*gd/(dk_yk*dk_yk);
              /* beta is the constant which adds the d_k contribution */
              beta = gkp1D_yk/dk_yk - step*gamma - tau_k;
              /* Here is the requisite check */
              ierr = VecDot(tao->stepdirection, cg->g_work, &tmp);
              if (cg->neg_xi){
                /* modified KD implementation */
                if (gkp1D_yk/dk_yk < 0) gamma = -1.0*gd/dk_yk;
                else gamma = cg->xi*gd/dk_yk;
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
              ierr = VecAXPBY(tao->stepdirection, -1.0, beta, cg->g_work); CHKERRQ(ierr);
              ierr = VecAXPY(tao->stepdirection, gamma, cg->y_work); CHKERRQ(ierr);
            }}
        break;
      case CG_BFGS_Mod:
        ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
        ierr = VecWAXPY(cg->yk, -1.0, cg->G_old, tao->gradient); CHKERRQ(ierr);
        ierr = VecWAXPY(cg->sk, -1.0, cg->X_old, tao->solution); CHKERRQ(ierr);
        ierr = VecNorm(cg->yk, NORM_2, &ynorm); CHKERRQ(ierr);
        ierr = VecNorm(cg->sk, NORM_2, &snorm); CHKERRQ(ierr);
        ynorm2 = ynorm*ynorm;
        ierr = VecDot(cg->yk, tao->stepdirection, &dk_yk); CHKERRQ(ierr);
        ierr = VecDot(cg->yk, cg->sk, &cg->yts); CHKERRQ(ierr);
        /* TODO: Should only need one of dk_yk and yts */
        if (ynorm2 < cg->epsilon){
         /* The gradient hasn't changed, so we should stay in the same direction as before. Don't update it or anything.*/
          if (snorm < cg->eps_23){
            /* We're making no progress. Scaled gradient descent step.*/
            ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
          }
        } else {
            if (!cg->diag_scaling){
              ierr = VecDot(cg->yk, tao->gradient, &gkp1_yk); CHKERRQ(ierr);
              ierr = TaoBNCGComputeScalarScaling(ynorm2, cg->yts, snorm*snorm, &tau_k, cg->alpha); CHKERRQ(ierr);
              tmp = gd/dk_yk;
              beta = tau_k*(gkp1_yk/dk_yk - ynorm2*gd/(dk_yk*dk_yk) - (step/tau_k)*gd/dk_yk);
              /* d <- -t*g + beta*t*d + t*tmp*yk */
              ierr = VecAXPBYPCZ(tao->stepdirection, -tau_k, tmp*tau_k, beta, tao->gradient, cg->yk); CHKERRQ(ierr);
            }
         else if (snorm < cg->epsilon || cg->yts < cg->epsilon) {
            /* We're making no progress. Scaled gradient descent step and reset diagonal scaling */
            ierr = TaoBNCGResetUpdate(tao, gnorm2); CHKERRQ(ierr);
          } else {
            /* We have diagonal scaling enabled and are taking a diagonally-scaled memoryless BFGS step */
            cg->yty = ynorm2;
            cg->sts = snorm*snorm;
            /* Compute the invD vector  */
            ierr = TaoBNCGComputeDiagScaling(tao, cg->yts, ynorm2); CHKERRQ(ierr);
            /* Apply the invD scaling to all my vectors */
            ierr = VecPointwiseMult(cg->g_work, cg->invD, tao->gradient); CHKERRQ(ierr);
            ierr = VecPointwiseMult(cg->d_work, cg->invD, tao->stepdirection); CHKERRQ(ierr);
            ierr = VecPointwiseMult(cg->y_work, cg->invD, cg->yk); CHKERRQ(ierr);
            /* Construct the constant ytDgkp1 */
            ierr = VecDot(cg->yk, cg->g_work, &gkp1_yk); CHKERRQ(ierr);
            /* Construct the constant for scaling Dkyk in the update */
            tmp = gd/dk_yk;
            /* tau_bfgs can be a parameter we tune via command line in future versions. For now, just setting to one. May instead put this inside the ComputeDiagonalScaling function... */
            cg->tau_bfgs = 1.0;
            /* tau_k = -ytDy/(ytd)^2 * gd */
            ierr = VecDot(cg->yk, cg->y_work, &tau_k);
            tau_k = -tau_k*gd/(dk_yk*dk_yk);
            /* beta is the constant which adds the dk contribution */
            beta = cg->tau_bfgs*gkp1_yk/dk_yk - step*tmp + tau_k;
            /* Do the update in two steps */
            ierr = VecAXPBY(tao->stepdirection, -cg->tau_bfgs, beta, cg->g_work); CHKERRQ(ierr);
            ierr = VecAXPY(tao->stepdirection, tmp, cg->y_work); CHKERRQ(ierr);
         }}
      default:
        beta = 0.0;
        break;
      }
    }
    PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoBNCGSteffensonAcceleration(Tao tao){
  TAO_BNCG          *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode    ierr;
  PetscReal         mag1, mag2;
  PetscReal         resnorm;
  PetscReal         steff_f;
  PetscFunctionBegin;
  if (tao->niter > 2 && tao->niter % 2 == 0){
    ierr = VecCopy(cg->steffnp1, cg->steffnm1); CHKERRQ(ierr); /* X_np1 to X_nm1 since it's been two iterations*/
    ierr = VecCopy(cg->X_old, cg->steffn); CHKERRQ(ierr); /* Get X_n */
    ierr = VecCopy(tao->solution, cg->steffnp1); CHKERRQ(ierr);

    /* Begin step 4 */
    ierr = VecCopy(cg->steffnm1, cg->W); CHKERRQ(ierr);
    ierr = VecAXPBY(cg->W, 1.0, -1.0, cg->steffn); CHKERRQ(ierr);
    ierr = VecDot(cg->W, cg->W, &mag1);
    ierr = VecAXPBYPCZ(cg->W, -1.0, 1.0, -1.0, cg->steffn, cg->steffnp1); CHKERRQ(ierr);
    ierr = VecDot(cg->W, cg->W, &mag2);
    ierr = VecCopy(cg->steffnm1, cg->steffva); CHKERRQ(ierr);
    ierr = VecAXPY(cg->steffva, -mag1/mag2, cg->W); CHKERRQ(ierr);

    ierr = TaoComputeObjectiveAndGradient(tao, cg->steffva, &steff_f, cg->g_work); CHKERRQ(ierr);

    /* Check if the accelerated point has converged*/
    ierr = VecFischer(cg->steffva, cg->g_work, tao->XL, tao->XU, cg->W);CHKERRQ(ierr);
    ierr = VecNorm(cg->W, NORM_2, &resnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
    //ierr = TaoLogConvergenceHistory(tao, cg->f, resnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
    //ierr = TaoMonitor(tao, tao->niter, cg->f, resnorm, 0.0, step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "va f: %20.19e\t va gnorm: %20.19e\n", steff_f, resnorm);

  } else if (tao->niter == 2){
    ierr = VecCopy(tao->solution, cg->steffnp1); CHKERRQ(ierr);
    mag1 = cg->sts; /* = |x1 - x0|^2 */
    ierr = VecCopy(cg->steffnm1, cg->steffvatmp); CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(cg->steffva, 1.0, -1.0, 0.0, cg->steffnp1, cg->steffn);
    ierr = VecAXPBYPCZ(cg->steffva, 1.0, -1.0, 1.0, cg->steffnm1, cg->steffn);
    ierr = VecNorm(cg->steffva, NORM_2, &mag2); CHKERRQ(ierr);
    mag2 = mag2*mag2;
    ierr = VecAXPBY(cg->steffva, 1.0, -mag1/mag2, cg->steffvatmp); CHKERRQ(ierr);
    // finished step 2
  } else if (tao->niter == 1){
    ierr = VecCopy(cg->X_old, cg->steffnm1); CHKERRQ(ierr);
    ierr = VecCopy(tao->solution, cg->steffn); CHKERRQ(ierr);
  }
  /* Now have step 2 done of method */

  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoBNCGConductIteration(Tao tao, PetscReal gnorm)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode               ierr;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    step=1.0,gnorm2,gd,ginner,dnorm;
  PetscReal                    gnorm2_old,f_old,resnorm, gnorm_old;
  PetscBool                    cg_restart, gd_fallback = PETSC_FALSE;

  PetscFunctionBegin;
  /* We are now going to perform a line search along the direction.  */
  ++tao->niter;

  /* Store solution and gradient info before it changes */
  ierr = VecCopy(tao->solution, cg->X_old);CHKERRQ(ierr);
  ierr = VecCopy(tao->gradient, cg->G_old);CHKERRQ(ierr);
  ierr = VecCopy(cg->unprojected_gradient, cg->unprojected_gradient_old);CHKERRQ(ierr);

  gnorm_old = gnorm;
  gnorm2_old = gnorm_old*gnorm_old;
  f_old = cg->f;
  /* Perform bounded line search */
  ierr = TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0);CHKERRQ(ierr);
  ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &cg->f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
  ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

  /*  Check linesearch failure */
  if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
    ++cg->ls_fails;

    if (cg->cg_type == CG_GradientDescent || gd_fallback){
      /* Nothing left to do but fail out of the optimization */
      step = 0.0;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
    } else {
      /* Restore previous point */
      ierr = VecCopy(cg->X_old, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(cg->G_old, tao->gradient);CHKERRQ(ierr);
      ierr = VecCopy(cg->unprojected_gradient_old, cg->unprojected_gradient);CHKERRQ(ierr);

      gnorm = gnorm_old;
      gnorm2 = gnorm2_old;
      cg->f = f_old;

      /* Fall back on the scaled gradient step */
      ierr = TaoBNCGResetUpdate(tao, gnorm2);
      ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);

      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0);CHKERRQ(ierr);
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &cg->f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER){
        /* Nothing left to do but fail out of the optimization */
        cg->ls_fails++;
        step = 0.0;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      }
    }
  }
  /* Convergence test for line search failure */
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Standard convergence test */
  /* Make sure convergence test is the same. */
  ierr = VecFischer(tao->solution, cg->unprojected_gradient, tao->XL, tao->XU, cg->W);CHKERRQ(ierr);
  ierr = VecNorm(cg->W, NORM_2, &resnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
  ierr = TaoLogConvergenceHistory(tao, cg->f, resnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, cg->f, resnorm, 0.0, step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Assert we have an updated step and we need at least one more iteration. */
  /* Calculate the next direction */
  /* Estimate the active set at the new solution */
  ierr = TaoBNCGEstimateActiveSet(tao, cg->as_type);CHKERRQ(ierr);
  /* Compute the projected gradient and its norm */
  ierr = VecCopy(cg->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
  ierr = VecISSet(tao->gradient, cg->active_idx, 0.0);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  gnorm2 = gnorm*gnorm;

  /* Check restart conditions for using steepest descent */
  cg_restart = PETSC_FALSE;
  ierr = VecDot(tao->gradient, cg->G_old, &ginner);CHKERRQ(ierr);
  ierr = VecNorm(tao->stepdirection, NORM_2, &dnorm);CHKERRQ(ierr);

  ierr = TaoBNCGStepDirectionUpdate(tao, gnorm2, step, f_old, gnorm2_old, cg_restart, dnorm, ginner); CHKERRQ(ierr);

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
    ierr = VecNorm(tao->stepdirection, NORM_2, &dnorm);
    /* TODO: potentially remove the third check */
    if (gd >= 0 || PetscIsInfOrNanReal(gd) || gd >= -cg->epsilon) {
      /* Not a descent direction, so we reset back to projected gradient descent */
      PetscPrintf(PETSC_COMM_SELF, "gd is small or positive: %20.19e\n", gd);
      ierr = TaoBNCGResetUpdate(tao, gnorm2);
      ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);
      ++cg->descent_error;
      gd_fallback = PETSC_TRUE;
    } else {
      gd_fallback = PETSC_FALSE;
    }
  }

  PetscFunctionReturn(0);
}
