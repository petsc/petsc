#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/snesimpl.h>

PetscBool         TaoRegisterAllCalled = PETSC_FALSE;
PetscFunctionList TaoList              = NULL;

PetscClassId TAO_CLASSID;

PetscLogEvent TAO_Solve;
PetscLogEvent TAO_ObjectiveEval;
PetscLogEvent TAO_GradientEval;
PetscLogEvent TAO_ObjGradEval;
PetscLogEvent TAO_HessianEval;
PetscLogEvent TAO_JacobianEval;
PetscLogEvent TAO_ConstraintsEval;

const char *TaoSubSetTypes[] = {"subvec", "mask", "matrixfree", "TaoSubSetType", "TAO_SUBSET_", NULL};

struct _n_TaoMonitorDrawCtx {
  PetscViewer viewer;
  PetscInt    howoften; /* when > 0 uses iteration % howoften, when negative only final solution plotted */
};

static PetscErrorCode KSPPreSolve_TAOEW_Private(KSP ksp, Vec b, Vec x, void *ctx)
{
  Tao  tao          = (Tao)ctx;
  SNES snes_ewdummy = tao->snes_ewdummy;

  PetscFunctionBegin;
  if (!snes_ewdummy) PetscFunctionReturn(PETSC_SUCCESS);
  /* populate snes_ewdummy struct values used in KSPPreSolve_SNESEW */
  snes_ewdummy->vec_func = b;
  snes_ewdummy->rtol     = tao->gttol;
  snes_ewdummy->iter     = tao->niter;
  PetscCall(VecNorm(b, NORM_2, &snes_ewdummy->norm));
  PetscCall(KSPPreSolve_SNESEW(ksp, b, x, snes_ewdummy));
  snes_ewdummy->vec_func = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPPostSolve_TAOEW_Private(KSP ksp, Vec b, Vec x, void *ctx)
{
  Tao  tao          = (Tao)ctx;
  SNES snes_ewdummy = tao->snes_ewdummy;

  PetscFunctionBegin;
  if (!snes_ewdummy) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(KSPPostSolve_SNESEW(ksp, b, x, snes_ewdummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUpEW_Private(Tao tao)
{
  SNESKSPEW  *kctx;
  const char *ewprefix;

  PetscFunctionBegin;
  if (!tao->ksp) PetscFunctionReturn(PETSC_SUCCESS);
  if (tao->ksp_ewconv) {
    if (!tao->snes_ewdummy) PetscCall(SNESCreate(PetscObjectComm((PetscObject)tao), &tao->snes_ewdummy));
    tao->snes_ewdummy->ksp_ewconv = PETSC_TRUE;
    PetscCall(KSPSetPreSolve(tao->ksp, KSPPreSolve_TAOEW_Private, tao));
    PetscCall(KSPSetPostSolve(tao->ksp, KSPPostSolve_TAOEW_Private, tao));

    PetscCall(KSPGetOptionsPrefix(tao->ksp, &ewprefix));
    kctx = (SNESKSPEW *)tao->snes_ewdummy->kspconvctx;
    PetscCall(SNESEWSetFromOptions_Private(kctx, PETSC_FALSE, PetscObjectComm((PetscObject)tao), ewprefix));
  } else PetscCall(SNESDestroy(&tao->snes_ewdummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoParametersInitialize - Sets all the parameters in `tao` to their default value (when `TaoCreate()` was called) if they
  currently contain default values. Default values are the parameter values when the object's type is set.

  Collective

  Input Parameter:
. tao - the `Tao` object

  Level: developer

  Developer Note:
  This is called by all the `TaoCreate_XXX()` routines.

.seealso: [](ch_snes), `Tao`, `TaoSolve()`, `TaoDestroy()`,
          `PetscObjectParameterSetDefault()`
@*/
PetscErrorCode TaoParametersInitialize(Tao tao)
{
  PetscObjectParameterSetDefault(tao, max_it, 10000);
  PetscObjectParameterSetDefault(tao, max_funcs, PETSC_UNLIMITED);
  PetscObjectParameterSetDefault(tao, gatol, PetscDefined(USE_REAL_SINGLE) ? 1e-5 : 1e-8);
  PetscObjectParameterSetDefault(tao, grtol, PetscDefined(USE_REAL_SINGLE) ? 1e-5 : 1e-8);
  PetscObjectParameterSetDefault(tao, crtol, PetscDefined(USE_REAL_SINGLE) ? 1e-5 : 1e-8);
  PetscObjectParameterSetDefault(tao, catol, PetscDefined(USE_REAL_SINGLE) ? 1e-5 : 1e-8);
  PetscObjectParameterSetDefault(tao, gttol, 0.0);
  PetscObjectParameterSetDefault(tao, steptol, 0.0);
  PetscObjectParameterSetDefault(tao, fmin, PETSC_NINFINITY);
  PetscObjectParameterSetDefault(tao, trust0, PETSC_INFINITY);
  return PETSC_SUCCESS;
}

/*@
  TaoCreate - Creates a Tao solver

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newtao - the new `Tao` context

  Options Database Key:
. -tao_type - select which method Tao should use

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoDestroy()`, `TaoSetFromOptions()`, `TaoSetType()`
@*/
PetscErrorCode TaoCreate(MPI_Comm comm, Tao *newtao)
{
  Tao tao;

  PetscFunctionBegin;
  PetscAssertPointer(newtao, 2);
  PetscCall(TaoInitializePackage());
  PetscCall(TaoLineSearchInitializePackage());

  PetscCall(PetscHeaderCreate(tao, TAO_CLASSID, "Tao", "Optimization solver", "Tao", comm, TaoDestroy, TaoView));
  tao->ops->convergencetest = TaoDefaultConvergenceTest;

  tao->hist_reset = PETSC_TRUE;
  PetscCall(TaoResetStatistics(tao));
  *newtao = tao;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSolve - Solves an optimization problem min F(x) s.t. l <= x <= u

  Collective

  Input Parameter:
. tao - the `Tao` context

  Level: beginner

  Notes:
  The user must set up the `Tao` object  with calls to `TaoSetSolution()`, `TaoSetObjective()`, `TaoSetGradient()`, and (if using 2nd order method) `TaoSetHessian()`.

  You should call `TaoGetConvergedReason()` or run with `-tao_converged_reason` to determine if the optimization algorithm actually succeeded or
  why it failed.

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoSetObjective()`, `TaoSetGradient()`, `TaoSetHessian()`, `TaoGetConvergedReason()`, `TaoSetUp()`
 @*/
PetscErrorCode TaoSolve(Tao tao)
{
  static PetscBool set = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscCitationsRegister("@TechReport{tao-user-ref,\n"
                                   "title   = {Toolkit for Advanced Optimization (TAO) Users Manual},\n"
                                   "author  = {Todd Munson and Jason Sarich and Stefan Wild and Steve Benson and Lois Curfman McInnes},\n"
                                   "Institution = {Argonne National Laboratory},\n"
                                   "Year   = 2014,\n"
                                   "Number = {ANL/MCS-TM-322 - Revision 3.5},\n"
                                   "url    = {https://www.mcs.anl.gov/research/projects/tao/}\n}\n",
                                   &set));
  tao->header_printed = PETSC_FALSE;
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoResetStatistics(tao));
  if (tao->linesearch) PetscCall(TaoLineSearchReset(tao->linesearch));

  PetscCall(PetscLogEventBegin(TAO_Solve, tao, 0, 0, 0));
  PetscTryTypeMethod(tao, solve);
  PetscCall(PetscLogEventEnd(TAO_Solve, tao, 0, 0, 0));

  PetscCall(VecViewFromOptions(tao->solution, (PetscObject)tao, "-tao_view_solution"));

  tao->ntotalits += tao->niter;

  if (tao->printreason) {
    PetscViewer viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
    PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject)tao)->tablevel));
    if (tao->reason > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  TAO %s solve converged due to %s iterations %" PetscInt_FMT "\n", ((PetscObject)tao)->prefix ? ((PetscObject)tao)->prefix : "", TaoConvergedReasons[tao->reason], tao->niter));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  TAO %s solve did not converge due to %s iteration %" PetscInt_FMT "\n", ((PetscObject)tao)->prefix ? ((PetscObject)tao)->prefix : "", TaoConvergedReasons[tao->reason], tao->niter));
    }
    PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)tao)->tablevel));
  }
  PetscCall(TaoViewFromOptions(tao, NULL, "-tao_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetUp - Sets up the internal data structures for the later use
  of a Tao solver

  Collective

  Input Parameter:
. tao - the `Tao` context

  Level: advanced

  Note:
  The user will not need to explicitly call `TaoSetUp()`, as it will
  automatically be called in `TaoSolve()`.  However, if the user
  desires to call it explicitly, it should come after `TaoCreate()`
  and any TaoSetSomething() routines, but before `TaoSolve()`.

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoSolve()`
@*/
PetscErrorCode TaoSetUp(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (tao->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoSetUpEW_Private(tao));
  PetscCheck(tao->solution, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Must call TaoSetSolution");
  PetscTryTypeMethod(tao, setup);
  tao->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoDestroy - Destroys the `Tao` context that was created with `TaoCreate()`

  Collective

  Input Parameter:
. tao - the `Tao` context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoSolve()`
@*/
PetscErrorCode TaoDestroy(Tao *tao)
{
  PetscFunctionBegin;
  if (!*tao) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*tao, TAO_CLASSID, 1);
  if (--((PetscObject)*tao)->refct > 0) {
    *tao = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod(*tao, destroy);
  PetscCall(KSPDestroy(&(*tao)->ksp));
  PetscCall(SNESDestroy(&(*tao)->snes_ewdummy));
  PetscCall(TaoLineSearchDestroy(&(*tao)->linesearch));

  if ((*tao)->ops->convergencedestroy) {
    PetscCall((*(*tao)->ops->convergencedestroy)((*tao)->cnvP));
    if ((*tao)->jacobian_state_inv) PetscCall(MatDestroy(&(*tao)->jacobian_state_inv));
  }
  PetscCall(VecDestroy(&(*tao)->solution));
  PetscCall(VecDestroy(&(*tao)->gradient));
  PetscCall(VecDestroy(&(*tao)->ls_res));

  if ((*tao)->gradient_norm) {
    PetscCall(PetscObjectDereference((PetscObject)(*tao)->gradient_norm));
    PetscCall(VecDestroy(&(*tao)->gradient_norm_tmp));
  }

  PetscCall(VecDestroy(&(*tao)->XL));
  PetscCall(VecDestroy(&(*tao)->XU));
  PetscCall(VecDestroy(&(*tao)->IL));
  PetscCall(VecDestroy(&(*tao)->IU));
  PetscCall(VecDestroy(&(*tao)->DE));
  PetscCall(VecDestroy(&(*tao)->DI));
  PetscCall(VecDestroy(&(*tao)->constraints));
  PetscCall(VecDestroy(&(*tao)->constraints_equality));
  PetscCall(VecDestroy(&(*tao)->constraints_inequality));
  PetscCall(VecDestroy(&(*tao)->stepdirection));
  PetscCall(MatDestroy(&(*tao)->hessian_pre));
  PetscCall(MatDestroy(&(*tao)->hessian));
  PetscCall(MatDestroy(&(*tao)->ls_jac));
  PetscCall(MatDestroy(&(*tao)->ls_jac_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian));
  PetscCall(MatDestroy(&(*tao)->jacobian_state_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian_state));
  PetscCall(MatDestroy(&(*tao)->jacobian_state_inv));
  PetscCall(MatDestroy(&(*tao)->jacobian_design));
  PetscCall(MatDestroy(&(*tao)->jacobian_equality));
  PetscCall(MatDestroy(&(*tao)->jacobian_equality_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian_inequality));
  PetscCall(MatDestroy(&(*tao)->jacobian_inequality_pre));
  PetscCall(ISDestroy(&(*tao)->state_is));
  PetscCall(ISDestroy(&(*tao)->design_is));
  PetscCall(VecDestroy(&(*tao)->res_weights_v));
  PetscCall(TaoMonitorCancel(*tao));
  if ((*tao)->hist_malloc) PetscCall(PetscFree4((*tao)->hist_obj, (*tao)->hist_resid, (*tao)->hist_cnorm, (*tao)->hist_lits));
  if ((*tao)->res_weights_n) {
    PetscCall(PetscFree((*tao)->res_weights_rows));
    PetscCall(PetscFree((*tao)->res_weights_cols));
    PetscCall(PetscFree((*tao)->res_weights_w));
  }
  PetscCall(PetscHeaderDestroy(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoKSPSetUseEW - Sets `SNES` to use Eisenstat-Walker method {cite}`ew96`for computing relative tolerance for linear solvers.

  Logically Collective

  Input Parameters:
+ tao  - Tao context
- flag - `PETSC_TRUE` or `PETSC_FALSE`

  Level: advanced

  Note:
  See `SNESKSPSetUseEW()` for customization details.

.seealso: [](ch_tao), `Tao`, `SNESKSPSetUseEW()`
@*/
PetscErrorCode TaoKSPSetUseEW(Tao tao, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  tao->ksp_ewconv = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetFromOptions - Sets various Tao parameters from the options database

  Collective

  Input Parameter:
. tao - the `Tao` solver context

  Options Database Keys:
+ -tao_type <type>             - The algorithm that Tao uses (lmvm, nls, etc.)
. -tao_gatol <gatol>           - absolute error tolerance for ||gradient||
. -tao_grtol <grtol>           - relative error tolerance for ||gradient||
. -tao_gttol <gttol>           - reduction of ||gradient|| relative to initial gradient
. -tao_max_it <max>            - sets maximum number of iterations
. -tao_max_funcs <max>         - sets maximum number of function evaluations
. -tao_fmin <fmin>             - stop if function value reaches fmin
. -tao_steptol <tol>           - stop if trust region radius less than <tol>
. -tao_trust0 <t>              - initial trust region radius
. -tao_view_solution           - view the solution at the end of the optimization process
. -tao_monitor                 - prints function value and residual norm at each iteration
. -tao_monitor_short           - same as `-tao_monitor`, but truncates very small values
. -tao_monitor_constraint_norm - prints objective value, gradient, and constraint norm at each iteration
. -tao_monitor_globalization   - prints information about the globalization at each iteration
. -tao_monitor_solution        - prints solution vector at each iteration
. -tao_monitor_ls_residual     - prints least-squares residual vector at each iteration
. -tao_monitor_step            - prints step vector at each iteration
. -tao_monitor_gradient        - prints gradient vector at each iteration
. -tao_monitor_solution_draw   - graphically view solution vector at each iteration
. -tao_monitor_step_draw       - graphically view step vector at each iteration
. -tao_monitor_gradient_draw   - graphically view gradient at each iteration
. -tao_monitor_cancel          - cancels all monitors (except those set with command line)
. -tao_fd_gradient             - use gradient computed with finite differences
. -tao_fd_hessian              - use hessian computed with finite differences
. -tao_mf_hessian              - use matrix-free Hessian computed with finite differences
. -tao_view                    - prints information about the Tao after solving
- -tao_converged_reason        - prints the reason Tao stopped iterating

  Level: beginner

  Note:
  To see all options, run your program with the `-help` option or consult the
  user's manual. Should be called after `TaoCreate()` but before `TaoSolve()`

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoSolve()`
@*/
PetscErrorCode TaoSetFromOptions(Tao tao)
{
  TaoType     default_type = TAOLMVM;
  char        type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer monviewer;
  PetscBool   flg, found;
  MPI_Comm    comm;
  PetscReal   catol, crtol, gatol, grtol, gttol;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));

  if (((PetscObject)tao)->type_name) default_type = ((PetscObject)tao)->type_name;

  PetscObjectOptionsBegin((PetscObject)tao);
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-tao_type", "Tao Solver type", "TaoSetType", TaoList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(TaoSetType(tao, type));
  } else if (!((PetscObject)tao)->type_name) {
    PetscCall(TaoSetType(tao, default_type));
  }

  /* Tao solvers do not set the prefix, set it here if not yet done
     We do it after SetType since solver may have been changed */
  if (tao->linesearch) {
    const char *prefix;
    PetscCall(TaoLineSearchGetOptionsPrefix(tao->linesearch, &prefix));
    if (!prefix) PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, ((PetscObject)tao)->prefix));
  }

  catol = tao->catol;
  crtol = tao->crtol;
  PetscCall(PetscOptionsReal("-tao_catol", "Stop if constraints violations within", "TaoSetConstraintTolerances", tao->catol, &catol, NULL));
  PetscCall(PetscOptionsReal("-tao_crtol", "Stop if relative constraint violations within", "TaoSetConstraintTolerances", tao->crtol, &crtol, NULL));
  PetscCall(TaoSetConstraintTolerances(tao, catol, crtol));

  gatol = tao->gatol;
  grtol = tao->grtol;
  gttol = tao->gttol;
  PetscCall(PetscOptionsReal("-tao_gatol", "Stop if norm of gradient less than", "TaoSetTolerances", tao->gatol, &gatol, NULL));
  PetscCall(PetscOptionsReal("-tao_grtol", "Stop if norm of gradient divided by the function value is less than", "TaoSetTolerances", tao->grtol, &grtol, NULL));
  PetscCall(PetscOptionsReal("-tao_gttol", "Stop if the norm of the gradient is less than the norm of the initial gradient times tol", "TaoSetTolerances", tao->gttol, &gttol, NULL));
  PetscCall(TaoSetTolerances(tao, gatol, grtol, gttol));

  PetscCall(PetscOptionsInt("-tao_max_it", "Stop if iteration number exceeds", "TaoSetMaximumIterations", tao->max_it, &tao->max_it, &flg));
  if (flg) PetscCall(TaoSetMaximumIterations(tao, tao->max_it));

  PetscCall(PetscOptionsInt("-tao_max_funcs", "Stop if number of function evaluations exceeds", "TaoSetMaximumFunctionEvaluations", tao->max_funcs, &tao->max_funcs, &flg));
  if (flg) PetscCall(TaoSetMaximumFunctionEvaluations(tao, tao->max_funcs));

  PetscCall(PetscOptionsReal("-tao_fmin", "Stop if function less than", "TaoSetFunctionLowerBound", tao->fmin, &tao->fmin, NULL));
  PetscCall(PetscOptionsBoundedReal("-tao_steptol", "Stop if step size or trust region radius less than", "", tao->steptol, &tao->steptol, NULL, 0));
  PetscCall(PetscOptionsReal("-tao_trust0", "Initial trust region radius", "TaoSetInitialTrustRegionRadius", tao->trust0, &tao->trust0, &flg));
  if (flg) PetscCall(TaoSetInitialTrustRegionRadius(tao, tao->trust0));

  PetscCall(PetscOptionsDeprecated("-tao_solution_monitor", "-tao_monitor_solution", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_gradient_monitor", "-tao_monitor_gradient", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_stepdirection_monitor", "-tao_monitor_step", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_residual_monitor", "-tao_monitor_residual", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_smonitor", "-tao_monitor_short", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_cmonitor", "-tao_monitor_constraint_norm", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_gmonitor", "-tao_monitor_globalization", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_draw_solution", "-tao_monitor_solution_draw", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_draw_gradient", "-tao_monitor_gradient_draw", "3.21", NULL));
  PetscCall(PetscOptionsDeprecated("-tao_draw_step", "-tao_monitor_step_draw", "3.21", NULL));

  PetscCall(PetscOptionsString("-tao_monitor_solution", "View solution vector after each iteration", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorSolution, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  PetscCall(PetscOptionsBool("-tao_converged_reason", "Print reason for Tao converged", "TaoSolve", tao->printreason, &tao->printreason, NULL));
  PetscCall(PetscOptionsString("-tao_monitor_gradient", "View gradient vector for each iteration", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorGradient, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  PetscCall(PetscOptionsString("-tao_monitor_step", "View step vector after each iteration", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorStep, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  PetscCall(PetscOptionsString("-tao_monitor_residual", "View least-squares residual vector after each iteration", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorResidual, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  PetscCall(PetscOptionsString("-tao_monitor", "Use the default convergence monitor", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorDefault, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  PetscCall(PetscOptionsString("-tao_monitor_globalization", "Use the convergence monitor with extra globalization info", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorGlobalization, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  PetscCall(PetscOptionsString("-tao_monitor_short", "Use the short convergence monitor", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorDefaultShort, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  PetscCall(PetscOptionsString("-tao_monitor_constraint_norm", "Use the default convergence monitor with constraint norm", "TaoMonitorSet", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(comm, monfilename, &monviewer));
    PetscCall(TaoMonitorSet(tao, TaoMonitorConstraintNorm, monviewer, (PetscErrorCode (*)(void **))PetscViewerDestroy));
  }

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsDeprecated("-tao_cancelmonitors", "-tao_monitor_cancel", "3.21", NULL));
  PetscCall(PetscOptionsBool("-tao_monitor_cancel", "cancel all monitors and call any registered destroy routines", "TaoMonitorCancel", flg, &flg, NULL));
  if (flg) PetscCall(TaoMonitorCancel(tao));

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-tao_monitor_solution_draw", "Plot solution vector at each iteration", "TaoMonitorSet", flg, &flg, NULL));
  if (flg) {
    TaoMonitorDrawCtx drawctx;
    PetscInt          howoften = 1;
    PetscCall(TaoMonitorDrawCtxCreate(PetscObjectComm((PetscObject)tao), NULL, NULL, PETSC_DECIDE, PETSC_DECIDE, 300, 300, howoften, &drawctx));
    PetscCall(TaoMonitorSet(tao, TaoMonitorSolutionDraw, drawctx, (PetscErrorCode (*)(void **))TaoMonitorDrawCtxDestroy));
  }

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-tao_monitor_step_draw", "Plots step at each iteration", "TaoMonitorSet", flg, &flg, NULL));
  if (flg) PetscCall(TaoMonitorSet(tao, TaoMonitorStepDraw, NULL, NULL));

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-tao_monitor_gradient_draw", "plots gradient at each iteration", "TaoMonitorSet", flg, &flg, NULL));
  if (flg) {
    TaoMonitorDrawCtx drawctx;
    PetscInt          howoften = 1;
    PetscCall(TaoMonitorDrawCtxCreate(PetscObjectComm((PetscObject)tao), NULL, NULL, PETSC_DECIDE, PETSC_DECIDE, 300, 300, howoften, &drawctx));
    PetscCall(TaoMonitorSet(tao, TaoMonitorGradientDraw, drawctx, (PetscErrorCode (*)(void **))TaoMonitorDrawCtxDestroy));
  }
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-tao_fd_gradient", "compute gradient using finite differences", "TaoDefaultComputeGradient", flg, &flg, NULL));
  if (flg) PetscCall(TaoSetGradient(tao, NULL, TaoDefaultComputeGradient, NULL));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-tao_fd_hessian", "compute Hessian using finite differences", "TaoDefaultComputeHessian", flg, &flg, NULL));
  if (flg) {
    Mat H;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)tao), &H));
    PetscCall(MatSetType(H, MATAIJ));
    PetscCall(TaoSetHessian(tao, H, H, TaoDefaultComputeHessian, NULL));
    PetscCall(MatDestroy(&H));
  }
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-tao_mf_hessian", "compute matrix-free Hessian using finite differences", "TaoDefaultComputeHessianMFFD", flg, &flg, NULL));
  if (flg) {
    Mat H;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)tao), &H));
    PetscCall(TaoSetHessian(tao, H, H, TaoDefaultComputeHessianMFFD, NULL));
    PetscCall(MatDestroy(&H));
  }
  PetscCall(PetscOptionsBool("-tao_recycle_history", "enable recycling/re-using information from the previous TaoSolve() call for some algorithms", "TaoSetRecycleHistory", flg, &flg, &found));
  if (found) PetscCall(TaoSetRecycleHistory(tao, flg));
  PetscCall(PetscOptionsEnum("-tao_subset_type", "subset type", "", TaoSubSetTypes, (PetscEnum)tao->subset_type, (PetscEnum *)&tao->subset_type, NULL));

  if (tao->ksp) {
    PetscCall(PetscOptionsBool("-tao_ksp_ew", "Use Eisentat-Walker linear system convergence test", "TaoKSPSetUseEW", tao->ksp_ewconv, &tao->ksp_ewconv, NULL));
    PetscCall(TaoKSPSetUseEW(tao, tao->ksp_ewconv));
  }

  PetscTryTypeMethod(tao, setfromoptions, PetscOptionsObject);

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)tao, PetscOptionsObject));
  PetscOptionsEnd();

  if (tao->linesearch) PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoViewFromOptions - View a `Tao` object based on values in the options database

  Collective

  Input Parameters:
+ A    - the  `Tao` context
. obj  - Optional object that provides the prefix for the options database
- name - command line option

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoView`, `PetscObjectViewFromOptions()`, `TaoCreate()`
@*/
PetscErrorCode TaoViewFromOptions(Tao A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, TAO_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoView - Prints information about the `Tao` object

  Collective

  Input Parameters:
+ tao    - the `Tao` context
- viewer - visualization context

  Options Database Key:
. -tao_view - Calls `TaoView()` at the end of `TaoSolve()`

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

.seealso: [](ch_tao), `Tao`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode TaoView(Tao tao, PetscViewer viewer)
{
  PetscBool isascii, isstring;
  TaoType   type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)tao)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(tao, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)tao, viewer));

    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(tao, view, viewer);
    if (tao->linesearch) PetscCall(TaoLineSearchView(tao->linesearch, viewer));
    if (tao->ksp) {
      PetscCall(KSPView(tao->ksp, viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "total KSP iterations: %" PetscInt_FMT "\n", tao->ksp_tot_its));
    }

    if (tao->XL || tao->XU) PetscCall(PetscViewerASCIIPrintf(viewer, "Active Set subset type: %s\n", TaoSubSetTypes[tao->subset_type]));

    PetscCall(PetscViewerASCIIPrintf(viewer, "convergence tolerances: gatol=%g,", (double)tao->gatol));
    PetscCall(PetscViewerASCIIPrintf(viewer, " grtol=%g,", (double)tao->grtol));
    PetscCall(PetscViewerASCIIPrintf(viewer, " steptol=%g,", (double)tao->steptol));
    PetscCall(PetscViewerASCIIPrintf(viewer, " gttol=%g\n", (double)tao->gttol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Residual in Function/Gradient:=%g\n", (double)tao->residual));

    if (tao->constrained) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "convergence tolerances:"));
      PetscCall(PetscViewerASCIIPrintf(viewer, " catol=%g,", (double)tao->catol));
      PetscCall(PetscViewerASCIIPrintf(viewer, " crtol=%g\n", (double)tao->crtol));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Residual in Constraints:=%g\n", (double)tao->cnorm));
    }

    if (tao->trust < tao->steptol) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "convergence tolerances: steptol=%g\n", (double)tao->steptol));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Final trust region radius:=%g\n", (double)tao->trust));
    }

    if (tao->fmin > -1.e25) PetscCall(PetscViewerASCIIPrintf(viewer, "convergence tolerances: function minimum=%g\n", (double)tao->fmin));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Objective value=%g\n", (double)tao->fc));

    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of iterations=%" PetscInt_FMT ",          ", tao->niter));
    PetscCall(PetscViewerASCIIPrintf(viewer, "              (max: %" PetscInt_FMT ")\n", tao->max_it));

    if (tao->nfuncs > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function evaluations=%" PetscInt_FMT ",", tao->nfuncs));
      if (tao->max_funcs == PETSC_UNLIMITED) PetscCall(PetscViewerASCIIPrintf(viewer, "                (max: unlimited)\n"));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "                (max: %" PetscInt_FMT ")\n", tao->max_funcs));
    }
    if (tao->ngrads > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "total number of gradient evaluations=%" PetscInt_FMT ",", tao->ngrads));
      if (tao->max_funcs == PETSC_UNLIMITED) PetscCall(PetscViewerASCIIPrintf(viewer, "                (max: unlimited)\n"));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "                (max: %" PetscInt_FMT ")\n", tao->max_funcs));
    }
    if (tao->nfuncgrads > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function/gradient evaluations=%" PetscInt_FMT ",", tao->nfuncgrads));
      if (tao->max_funcs == PETSC_UNLIMITED) PetscCall(PetscViewerASCIIPrintf(viewer, "    (max: unlimited)\n"));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "    (max: %" PetscInt_FMT ")\n", tao->max_funcs));
    }
    if (tao->nhess > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of Hessian evaluations=%" PetscInt_FMT "\n", tao->nhess));
    if (tao->nconstraints > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of constraint function evaluations=%" PetscInt_FMT "\n", tao->nconstraints));
    if (tao->njac > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of Jacobian evaluations=%" PetscInt_FMT "\n", tao->njac));

    if (tao->reason > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Solution converged: "));
      switch (tao->reason) {
      case TAO_CONVERGED_GATOL:
        PetscCall(PetscViewerASCIIPrintf(viewer, " ||g(X)|| <= gatol\n"));
        break;
      case TAO_CONVERGED_GRTOL:
        PetscCall(PetscViewerASCIIPrintf(viewer, " ||g(X)||/|f(X)| <= grtol\n"));
        break;
      case TAO_CONVERGED_GTTOL:
        PetscCall(PetscViewerASCIIPrintf(viewer, " ||g(X)||/||g(X0)|| <= gttol\n"));
        break;
      case TAO_CONVERGED_STEPTOL:
        PetscCall(PetscViewerASCIIPrintf(viewer, " Steptol -- step size small\n"));
        break;
      case TAO_CONVERGED_MINF:
        PetscCall(PetscViewerASCIIPrintf(viewer, " Minf --  f < fmin\n"));
        break;
      case TAO_CONVERGED_USER:
        PetscCall(PetscViewerASCIIPrintf(viewer, " User Terminated\n"));
        break;
      default:
        PetscCall(PetscViewerASCIIPrintf(viewer, " %d\n", tao->reason));
        break;
      }
    } else if (tao->reason == TAO_CONTINUE_ITERATING) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Solver never run\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Solver failed: "));
      switch (tao->reason) {
      case TAO_DIVERGED_MAXITS:
        PetscCall(PetscViewerASCIIPrintf(viewer, " Maximum Iterations\n"));
        break;
      case TAO_DIVERGED_NAN:
        PetscCall(PetscViewerASCIIPrintf(viewer, " NAN or Inf encountered\n"));
        break;
      case TAO_DIVERGED_MAXFCN:
        PetscCall(PetscViewerASCIIPrintf(viewer, " Maximum Function Evaluations\n"));
        break;
      case TAO_DIVERGED_LS_FAILURE:
        PetscCall(PetscViewerASCIIPrintf(viewer, " Line Search Failure\n"));
        break;
      case TAO_DIVERGED_TR_REDUCTION:
        PetscCall(PetscViewerASCIIPrintf(viewer, " Trust Region too small\n"));
        break;
      case TAO_DIVERGED_USER:
        PetscCall(PetscViewerASCIIPrintf(viewer, " User Terminated\n"));
        break;
      default:
        PetscCall(PetscViewerASCIIPrintf(viewer, " %d\n", tao->reason));
        break;
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(TaoGetType(tao, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " %-3.3s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetRecycleHistory - Sets the boolean flag to enable/disable re-using
  iterate information from the previous `TaoSolve()`. This feature is disabled by
  default.

  Logically Collective

  Input Parameters:
+ tao     - the `Tao` context
- recycle - boolean flag

  Options Database Key:
. -tao_recycle_history <true,false> - reuse the history

  Level: intermediate

  Notes:
  For conjugate gradient methods (`TAOBNCG`), this re-uses the latest search direction
  from the previous `TaoSolve()` call when computing the first search direction in a
  new solution. By default, CG methods set the first search direction to the
  negative gradient.

  For quasi-Newton family of methods (`TAOBQNLS`, `TAOBQNKLS`, `TAOBQNKTR`, `TAOBQNKTL`), this re-uses
  the accumulated quasi-Newton Hessian approximation from the previous `TaoSolve()`
  call. By default, QN family of methods reset the initial Hessian approximation to
  the identity matrix.

  For any other algorithm, this setting has no effect.

.seealso: [](ch_tao), `Tao`, `TaoGetRecycleHistory()`, `TAOBNCG`, `TAOBQNLS`, `TAOBQNKLS`, `TAOBQNKTR`, `TAOBQNKTL`
@*/
PetscErrorCode TaoSetRecycleHistory(Tao tao, PetscBool recycle)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, recycle, 2);
  tao->recycle = recycle;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetRecycleHistory - Retrieve the boolean flag for re-using iterate information
  from the previous `TaoSolve()`. This feature is disabled by default.

  Logically Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. recycle - boolean flag

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoSetRecycleHistory()`, `TAOBNCG`, `TAOBQNLS`, `TAOBQNKLS`, `TAOBQNKTR`, `TAOBQNKTL`
@*/
PetscErrorCode TaoGetRecycleHistory(Tao tao, PetscBool *recycle)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(recycle, 2);
  *recycle = tao->recycle;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetTolerances - Sets parameters used in `TaoSolve()` convergence tests

  Logically Collective

  Input Parameters:
+ tao   - the `Tao` context
. gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by this factor

  Options Database Keys:
+ -tao_gatol <gatol> - Sets gatol
. -tao_grtol <grtol> - Sets grtol
- -tao_gttol <gttol> - Sets gttol

  Stopping Criteria\:
.vb
  ||g(X)||                            <= gatol
  ||g(X)|| / |f(X)|                   <= grtol
  ||g(X)|| / ||g(X0)||                <= gttol
.ve

  Level: beginner

  Notes:
  Use `PETSC_CURRENT` to leave one or more tolerances unchanged.

  Use `PETSC_DETERMINE` to set one or more tolerances to their values when the `tao`object's type was set

  Fortran Note:
  Use `PETSC_CURRENT_REAL` or `PETSC_DETERMINE_REAL`

.seealso: [](ch_tao), `Tao`, `TaoConvergedReason`, `TaoGetTolerances()`
@*/
PetscErrorCode TaoSetTolerances(Tao tao, PetscReal gatol, PetscReal grtol, PetscReal gttol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, gatol, 2);
  PetscValidLogicalCollectiveReal(tao, grtol, 3);
  PetscValidLogicalCollectiveReal(tao, gttol, 4);

  if (gatol == (PetscReal)PETSC_DETERMINE) {
    tao->gatol = tao->default_gatol;
  } else if (gatol != (PetscReal)PETSC_CURRENT) {
    PetscCheck(gatol >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Negative gatol not allowed");
    tao->gatol = gatol;
  }

  if (grtol == (PetscReal)PETSC_DETERMINE) {
    tao->grtol = tao->default_grtol;
  } else if (grtol != (PetscReal)PETSC_CURRENT) {
    PetscCheck(grtol >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Negative grtol not allowed");
    tao->grtol = grtol;
  }

  if (gttol == (PetscReal)PETSC_DETERMINE) {
    tao->gttol = tao->default_gttol;
  } else if (gttol != (PetscReal)PETSC_CURRENT) {
    PetscCheck(gttol >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Negative gttol not allowed");
    tao->gttol = gttol;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetConstraintTolerances - Sets constraint tolerance parameters used in `TaoSolve()` convergence tests

  Logically Collective

  Input Parameters:
+ tao   - the `Tao` context
. catol - absolute constraint tolerance, constraint norm must be less than `catol` for used for `gatol` convergence criteria
- crtol - relative constraint tolerance, constraint norm must be less than `crtol` for used for `gatol`, `gttol` convergence criteria

  Options Database Keys:
+ -tao_catol <catol> - Sets catol
- -tao_crtol <crtol> - Sets crtol

  Level: intermediate

  Notes:
  Use `PETSC_CURRENT` to leave one or tolerance unchanged.

  Use `PETSC_DETERMINE` to set one or more tolerances to their values when the `tao` object's type was set

  Fortran Note:
  Use `PETSC_CURRENT_REAL` or `PETSC_DETERMINE_REAL`

.seealso: [](ch_tao), `Tao`, `TaoConvergedReason`, `TaoGetTolerances()`, `TaoGetConstraintTolerances()`, `TaoSetTolerances()`
@*/
PetscErrorCode TaoSetConstraintTolerances(Tao tao, PetscReal catol, PetscReal crtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, catol, 2);
  PetscValidLogicalCollectiveReal(tao, crtol, 3);

  if (catol == (PetscReal)PETSC_DETERMINE) {
    tao->catol = tao->default_catol;
  } else if (catol != (PetscReal)PETSC_CURRENT) {
    PetscCheck(catol >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Negative catol not allowed");
    tao->catol = catol;
  }

  if (crtol == (PetscReal)PETSC_DETERMINE) {
    tao->crtol = tao->default_crtol;
  } else if (crtol != (PetscReal)PETSC_CURRENT) {
    PetscCheck(crtol >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Negative crtol not allowed");
    tao->crtol = crtol;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetConstraintTolerances - Gets constraint tolerance parameters used in `TaoSolve()` convergence tests

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ catol - absolute constraint tolerance, constraint norm must be less than `catol` for used for `gatol` convergence criteria
- crtol - relative constraint tolerance, constraint norm must be less than `crtol` for used for `gatol`, `gttol` convergence criteria

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoConvergedReasons`,`TaoGetTolerances()`, `TaoSetTolerances()`, `TaoSetConstraintTolerances()`
@*/
PetscErrorCode TaoGetConstraintTolerances(Tao tao, PetscReal *catol, PetscReal *crtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (catol) *catol = tao->catol;
  if (crtol) *crtol = tao->crtol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetFunctionLowerBound - Sets a bound on the solution objective value.
  When an approximate solution with an objective value below this number
  has been found, the solver will terminate.

  Logically Collective

  Input Parameters:
+ tao  - the Tao solver context
- fmin - the tolerance

  Options Database Key:
. -tao_fmin <fmin> - sets the minimum function value

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoConvergedReason`, `TaoSetTolerances()`
@*/
PetscErrorCode TaoSetFunctionLowerBound(Tao tao, PetscReal fmin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, fmin, 2);
  tao->fmin = fmin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetFunctionLowerBound - Gets the bound on the solution objective value.
  When an approximate solution with an objective value below this number
  has been found, the solver will terminate.

  Not Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. fmin - the minimum function value

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoConvergedReason`, `TaoSetFunctionLowerBound()`
@*/
PetscErrorCode TaoGetFunctionLowerBound(Tao tao, PetscReal *fmin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(fmin, 2);
  *fmin = tao->fmin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetMaximumFunctionEvaluations - Sets a maximum number of function evaluations allowed for a `TaoSolve()`.

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` solver context
- nfcn - the maximum number of function evaluations (>=0), use `PETSC_UNLIMITED` to have no bound

  Options Database Key:
. -tao_max_funcs <nfcn> - sets the maximum number of function evaluations

  Level: intermediate

  Note:
  Use `PETSC_DETERMINE` to use the default maximum number of function evaluations that was set when the object type was set.

  Developer Note:
  Deprecated support for an unlimited number of function evaluations by passing a negative value.

.seealso: [](ch_tao), `Tao`, `TaoSetTolerances()`, `TaoSetMaximumIterations()`
@*/
PetscErrorCode TaoSetMaximumFunctionEvaluations(Tao tao, PetscInt nfcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveInt(tao, nfcn, 2);
  if (nfcn == PETSC_DETERMINE) {
    tao->max_funcs = tao->default_max_funcs;
  } else if (nfcn == PETSC_UNLIMITED || nfcn < 0) {
    tao->max_funcs = PETSC_UNLIMITED;
  } else {
    PetscCheck(nfcn >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Maximum number of function evaluations  must be positive");
    tao->max_funcs = nfcn;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetMaximumFunctionEvaluations - Gets a maximum number of function evaluations allowed for a `TaoSolve()`

  Logically Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. nfcn - the maximum number of function evaluations

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoSetMaximumFunctionEvaluations()`, `TaoGetMaximumIterations()`
@*/
PetscErrorCode TaoGetMaximumFunctionEvaluations(Tao tao, PetscInt *nfcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(nfcn, 2);
  *nfcn = tao->max_funcs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetCurrentFunctionEvaluations - Get current number of function evaluations used by a `Tao` object

  Not Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. nfuncs - the current number of function evaluations (maximum between gradient and function evaluations)

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoSetMaximumFunctionEvaluations()`, `TaoGetMaximumFunctionEvaluations()`, `TaoGetMaximumIterations()`
@*/
PetscErrorCode TaoGetCurrentFunctionEvaluations(Tao tao, PetscInt *nfuncs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(nfuncs, 2);
  *nfuncs = PetscMax(tao->nfuncs, tao->nfuncgrads);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetMaximumIterations - Sets a maximum number of iterates to be used in `TaoSolve()`

  Logically Collective

  Input Parameters:
+ tao    - the `Tao` solver context
- maxits - the maximum number of iterates (>=0), use `PETSC_UNLIMITED` to have no bound

  Options Database Key:
. -tao_max_it <its> - sets the maximum number of iterations

  Level: intermediate

  Note:
  Use `PETSC_DETERMINE` to use the default maximum number of iterations that was set when the object's type was set.

  Developer Note:
  DeprAlso accepts the deprecated negative values to indicate no limit

.seealso: [](ch_tao), `Tao`, `TaoSetTolerances()`, `TaoSetMaximumFunctionEvaluations()`
@*/
PetscErrorCode TaoSetMaximumIterations(Tao tao, PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveInt(tao, maxits, 2);
  if (maxits == PETSC_DETERMINE) {
    tao->max_it = tao->default_max_it;
  } else if (maxits == PETSC_UNLIMITED) {
    tao->max_it = PETSC_INT_MAX;
  } else {
    PetscCheck(maxits > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Maximum number of iterations must be positive");
    tao->max_it = maxits;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetMaximumIterations - Gets a maximum number of iterates that will be used

  Not Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. maxits - the maximum number of iterates

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoSetMaximumIterations()`, `TaoGetMaximumFunctionEvaluations()`
@*/
PetscErrorCode TaoGetMaximumIterations(Tao tao, PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(maxits, 2);
  *maxits = tao->max_it;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetInitialTrustRegionRadius - Sets the initial trust region radius.

  Logically Collective

  Input Parameters:
+ tao    - a `Tao` optimization solver
- radius - the trust region radius

  Options Database Key:
. -tao_trust0 <t0> - sets initial trust region radius

  Level: intermediate

  Note:
  Use `PETSC_DETERMINE` to use the default radius that was set when the object's type was set.

.seealso: [](ch_tao), `Tao`, `TaoGetTrustRegionRadius()`, `TaoSetTrustRegionTolerance()`, `TAONTR`
@*/
PetscErrorCode TaoSetInitialTrustRegionRadius(Tao tao, PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, radius, 2);
  if (radius == PETSC_DETERMINE) {
    tao->trust0 = tao->default_trust0;
  } else {
    PetscCheck(radius > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Radius must be positive");
    tao->trust0 = radius;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetInitialTrustRegionRadius - Gets the initial trust region radius.

  Not Collective

  Input Parameter:
. tao - a `Tao` optimization solver

  Output Parameter:
. radius - the trust region radius

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoSetInitialTrustRegionRadius()`, `TaoGetCurrentTrustRegionRadius()`, `TAONTR`
@*/
PetscErrorCode TaoGetInitialTrustRegionRadius(Tao tao, PetscReal *radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(radius, 2);
  *radius = tao->trust0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetCurrentTrustRegionRadius - Gets the current trust region radius.

  Not Collective

  Input Parameter:
. tao - a `Tao` optimization solver

  Output Parameter:
. radius - the trust region radius

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoSetInitialTrustRegionRadius()`, `TaoGetInitialTrustRegionRadius()`, `TAONTR`
@*/
PetscErrorCode TaoGetCurrentTrustRegionRadius(Tao tao, PetscReal *radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(radius, 2);
  *radius = tao->trust;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetTolerances - gets the current values of some tolerances used for the convergence testing of `TaoSolve()`

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by a this factor

  Level: intermediate

  Note:
  `NULL` can be used as an argument if not all tolerances values are needed

.seealso: [](ch_tao), `Tao`, `TaoSetTolerances()`
@*/
PetscErrorCode TaoGetTolerances(Tao tao, PetscReal *gatol, PetscReal *grtol, PetscReal *gttol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (gatol) *gatol = tao->gatol;
  if (grtol) *grtol = tao->grtol;
  if (gttol) *gttol = tao->gttol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetKSP - Gets the linear solver used by the optimization solver.

  Not Collective

  Input Parameter:
. tao - the `Tao` solver

  Output Parameter:
. ksp - the `KSP` linear solver used in the optimization solver

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `KSP`
@*/
PetscErrorCode TaoGetKSP(Tao tao, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(ksp, 2);
  *ksp = tao->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetLinearSolveIterations - Gets the total number of linear iterations
  used by the `Tao` solver

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. lits - number of linear iterations

  Level: intermediate

  Note:
  This counter is reset to zero for each successive call to `TaoSolve()`

.seealso: [](ch_tao), `Tao`, `TaoGetKSP()`
@*/
PetscErrorCode TaoGetLinearSolveIterations(Tao tao, PetscInt *lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(lits, 2);
  *lits = tao->ksp_tot_its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetLineSearch - Gets the line search used by the optimization solver.

  Not Collective

  Input Parameter:
. tao - the `Tao` solver

  Output Parameter:
. ls - the line search used in the optimization solver

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchType`
@*/
PetscErrorCode TaoGetLineSearch(Tao tao, TaoLineSearch *ls)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(ls, 2);
  *ls = tao->linesearch;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoAddLineSearchCounts - Adds the number of function evaluations spent
  in the line search to the running total.

  Input Parameters:
. tao - the `Tao` solver

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoGetLineSearch()`, `TaoLineSearchApply()`
@*/
PetscErrorCode TaoAddLineSearchCounts(Tao tao)
{
  PetscBool flg;
  PetscInt  nfeval, ngeval, nfgeval;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (tao->linesearch) {
    PetscCall(TaoLineSearchIsUsingTaoRoutines(tao->linesearch, &flg));
    if (!flg) {
      PetscCall(TaoLineSearchGetNumberFunctionEvaluations(tao->linesearch, &nfeval, &ngeval, &nfgeval));
      tao->nfuncs += nfeval;
      tao->ngrads += ngeval;
      tao->nfuncgrads += nfgeval;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetSolution - Returns the vector with the current solution from the `Tao` object

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. X - the current solution

  Level: intermediate

  Note:
  The returned vector will be the same object that was passed into `TaoSetSolution()`

.seealso: [](ch_tao), `Tao`, `TaoSetSolution()`, `TaoSolve()`
@*/
PetscErrorCode TaoGetSolution(Tao tao, Vec *X)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(X, 2);
  *X = tao->solution;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoResetStatistics - Initialize the statistics collected by the `Tao` object.
  These statistics include the iteration number, residual norms, and convergence status.
  This routine gets called before solving each optimization problem.

  Collective

  Input Parameter:
. tao - the `Tao` context

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoSolve()`
@*/
PetscErrorCode TaoResetStatistics(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  tao->niter        = 0;
  tao->nfuncs       = 0;
  tao->nfuncgrads   = 0;
  tao->ngrads       = 0;
  tao->nhess        = 0;
  tao->njac         = 0;
  tao->nconstraints = 0;
  tao->ksp_its      = 0;
  tao->ksp_tot_its  = 0;
  tao->reason       = TAO_CONTINUE_ITERATING;
  tao->residual     = 0.0;
  tao->cnorm        = 0.0;
  tao->step         = 0.0;
  tao->lsflag       = PETSC_FALSE;
  if (tao->hist_reset) tao->hist_len = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetUpdate - Sets the general-purpose update function called
  at the beginning of every iteration of the optimization algorithm. Called after the new solution and the gradient
  is determined, but before the Hessian is computed (if applicable).

  Logically Collective

  Input Parameters:
+ tao  - The `Tao` solver
. func - The function
- ctx  - The update function context

  Calling sequence of `func`:
+ tao - The optimizer context
. it  - The current iteration index
- ctx - The update context

  Level: advanced

  Notes:
  Users can modify the gradient direction or any other vector associated to the specific solver used.
  The objective function value is always recomputed after a call to the update hook.

.seealso: [](ch_tao), `Tao`, `TaoSolve()`
@*/
PetscErrorCode TaoSetUpdate(Tao tao, PetscErrorCode (*func)(Tao tao, PetscInt it, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  tao->ops->update = func;
  tao->user_update = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetConvergenceTest - Sets the function that is to be used to test
  for convergence of the iterative minimization solution.  The new convergence
  testing routine will replace Tao's default convergence test.

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` object
. conv - the routine to test for convergence
- ctx  - [optional] context for private data for the convergence routine
        (may be `NULL`)

  Calling sequence of `conv`:
+ tao - the `Tao` object
- ctx - [optional] convergence context

  Level: advanced

  Note:
  The new convergence testing routine should call `TaoSetConvergedReason()`.

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoSetConvergedReason()`, `TaoGetSolutionStatus()`, `TaoGetTolerances()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoSetConvergenceTest(Tao tao, PetscErrorCode (*conv)(Tao, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  tao->ops->convergencetest = conv;
  tao->cnvP                 = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorSet - Sets an additional function that is to be used at every
  iteration of the solver to display the iteration's
  progress.

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` solver context
. func - monitoring routine
. ctx  - [optional] user-defined context for private data for the monitor routine (may be `NULL`)
- dest - [optional] function to destroy the context when the `Tao` is destroyed

  Calling sequence of `func`:
+ tao - the `Tao` solver context
- ctx - [optional] monitoring context

  Calling sequence of `dest`:
. ctx - monitoring context

  Level: intermediate

  Notes:
  See `TaoSetFromOptions()` for a monitoring options.

  Several different monitoring routines may be set by calling
  `TaoMonitorSet()` multiple times; all will be called in the
  order in which they were set.

  Fortran Notes:
  Only one monitor function may be set

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoMonitorDefault()`, `TaoMonitorCancel()`, `TaoSetDestroyRoutine()`, `TaoView()`
@*/
PetscErrorCode TaoMonitorSet(Tao tao, PetscErrorCode (*func)(Tao, void *), void *ctx, PetscErrorCode (*dest)(void **))
{
  PetscInt  i;
  PetscBool identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCheck(tao->numbermonitors < MAXTAOMONITORS, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Cannot attach another monitor -- max=%d", MAXTAOMONITORS);

  for (i = 0; i < tao->numbermonitors; i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))func, ctx, dest, (PetscErrorCode (*)(void))tao->monitor[i], tao->monitorcontext[i], tao->monitordestroy[i], &identical));
    if (identical) PetscFunctionReturn(PETSC_SUCCESS);
  }
  tao->monitor[tao->numbermonitors]        = func;
  tao->monitorcontext[tao->numbermonitors] = (void *)ctx;
  tao->monitordestroy[tao->numbermonitors] = dest;
  ++tao->numbermonitors;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoMonitorCancel - Clears all the monitor functions for a `Tao` object.

  Logically Collective

  Input Parameter:
. tao - the `Tao` solver context

  Options Database Key:
. -tao_monitor_cancel - cancels all monitors that have been hardwired
    into a code by calls to `TaoMonitorSet()`, but does not cancel those
    set via the options database

  Level: advanced

  Note:
  There is no way to clear one specific monitor from a `Tao` object.

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefault()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorCancel(Tao tao)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  for (i = 0; i < tao->numbermonitors; i++) {
    if (tao->monitordestroy[i]) PetscCall((*tao->monitordestroy[i])(&tao->monitorcontext[i]));
  }
  tao->numbermonitors = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoMonitorDefault - Default routine for monitoring progress of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context or `NULL`

  Options Database Key:
. -tao_monitor - turn on default monitoring

  Level: advanced

  Note:
  This monitor prints the function value and gradient
  norm at each iteration.

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefaultShort()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorDefault(Tao tao, void *ctx)
{
  PetscInt    its, tabs;
  PetscReal   fct, gnorm;
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  its   = tao->niter;
  fct   = tao->fc;
  gnorm = tao->residual;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  if (its == 0 && ((PetscObject)tao)->prefix && !tao->header_printed) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Iteration information for %s solve.\n", ((PetscObject)tao)->prefix));
    tao->header_printed = PETSC_TRUE;
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " TAO,", its));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Function value: %g,", (double)fct));
  if (gnorm >= PETSC_INFINITY) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual: Inf \n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual: %g \n", (double)gnorm));
  }
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoMonitorGlobalization - Default routine for monitoring progress of `TaoSolve()` with extra detail on the globalization method.

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context or `NULL`

  Options Database Key:
. -tao_monitor_globalization - turn on monitoring with globalization information

  Level: advanced

  Note:
  This monitor prints the function value and gradient norm at each
  iteration, as well as the step size and trust radius. Note that the
  step size and trust radius may be the same for some algorithms.

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefaultShort()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorGlobalization(Tao tao, void *ctx)
{
  PetscInt    its, tabs;
  PetscReal   fct, gnorm, stp, tr;
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  its   = tao->niter;
  fct   = tao->fc;
  gnorm = tao->residual;
  stp   = tao->step;
  tr    = tao->trust;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  if (its == 0 && ((PetscObject)tao)->prefix && !tao->header_printed) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Iteration information for %s solve.\n", ((PetscObject)tao)->prefix));
    tao->header_printed = PETSC_TRUE;
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " TAO,", its));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Function value: %g,", (double)fct));
  if (gnorm >= PETSC_INFINITY) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual: Inf,"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual: %g,", (double)gnorm));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Step: %g,  Trust: %g\n", (double)stp, (double)tr));
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoMonitorDefaultShort - Routine for monitoring progress of `TaoSolve()` that displays fewer digits than `TaoMonitorDefault()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context of type `PETSCVIEWERASCII`

  Options Database Key:
. -tao_monitor_short - turn on default short monitoring

  Level: advanced

  Note:
  Same as `TaoMonitorDefault()` except
  it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often
  different on different machines; by using this routine different
  machines will usually generate the same output.

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefault()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorDefaultShort(Tao tao, void *ctx)
{
  PetscInt    its, tabs;
  PetscReal   fct, gnorm;
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  its   = tao->niter;
  fct   = tao->fc;
  gnorm = tao->residual;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "iter = %3" PetscInt_FMT ",", its));
  PetscCall(PetscViewerASCIIPrintf(viewer, " Function value %g,", (double)fct));
  if (gnorm >= PETSC_INFINITY) {
    PetscCall(PetscViewerASCIIPrintf(viewer, " Residual: Inf \n"));
  } else if (gnorm > 1.e-6) {
    PetscCall(PetscViewerASCIIPrintf(viewer, " Residual: %g \n", (double)gnorm));
  } else if (gnorm > 1.e-11) {
    PetscCall(PetscViewerASCIIPrintf(viewer, " Residual: < 1.0e-6 \n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, " Residual: < 1.0e-11 \n"));
  }
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoMonitorConstraintNorm - same as `TaoMonitorDefault()` except
  it prints the norm of the constraint function.

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context or `NULL`

  Options Database Key:
. -tao_monitor_constraint_norm - monitor the constraints

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefault()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorConstraintNorm(Tao tao, void *ctx)
{
  PetscInt    its, tabs;
  PetscReal   fct, gnorm;
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  its   = tao->niter;
  fct   = tao->fc;
  gnorm = tao->residual;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "iter = %" PetscInt_FMT ",", its));
  PetscCall(PetscViewerASCIIPrintf(viewer, " Function value: %g,", (double)fct));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual: %g ", (double)gnorm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Constraint: %g \n", (double)tao->cnorm));
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorSolution - Views the solution at each iteration of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context or `NULL`

  Options Database Key:
. -tao_monitor_solution - view the solution

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefaultShort()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorSolution(Tao tao, void *ctx)
{
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(VecView(tao->solution, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorGradient - Views the gradient at each iteration of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context or `NULL`

  Options Database Key:
. -tao_monitor_gradient - view the gradient at each iteration

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefaultShort()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorGradient(Tao tao, void *ctx)
{
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(VecView(tao->gradient, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorStep - Views the step-direction at each iteration of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context or `NULL`

  Options Database Key:
. -tao_monitor_step - view the step vector at each iteration

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefaultShort()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorStep(Tao tao, void *ctx)
{
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(VecView(tao->stepdirection, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorSolutionDraw - Plots the solution at each iteration of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `TaoMonitorDraw` context

  Options Database Key:
. -tao_monitor_solution_draw - draw the solution at each iteration

  Level: advanced

  Note:
  The context created by `TaoMonitorDrawCtxCreate()`, along with `TaoMonitorSolutionDraw()`, and `TaoMonitorDrawCtxDestroy()`
  are passed to `TaoMonitorSet()` to monitor the solution graphically.

.seealso: [](ch_tao), `Tao`, `TaoMonitorSolution()`, `TaoMonitorSet()`, `TaoMonitorGradientDraw()`, `TaoMonitorDrawCtxCreate()`,
          `TaoMonitorDrawCtxDestroy()`
@*/
PetscErrorCode TaoMonitorSolutionDraw(Tao tao, void *ctx)
{
  TaoMonitorDrawCtx ictx = (TaoMonitorDrawCtx)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (!(((ictx->howoften > 0) && (!(tao->niter % ictx->howoften))) || ((ictx->howoften == -1) && tao->reason))) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecView(tao->solution, ictx->viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorGradientDraw - Plots the gradient at each iteration of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - `PetscViewer` context

  Options Database Key:
. -tao_monitor_gradient_draw - draw the gradient at each iteration

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMonitorGradient()`, `TaoMonitorSet()`, `TaoMonitorSolutionDraw()`
@*/
PetscErrorCode TaoMonitorGradientDraw(Tao tao, void *ctx)
{
  TaoMonitorDrawCtx ictx = (TaoMonitorDrawCtx)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (!(((ictx->howoften > 0) && (!(tao->niter % ictx->howoften))) || ((ictx->howoften == -1) && tao->reason))) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecView(tao->gradient, ictx->viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorStepDraw - Plots the step direction at each iteration of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - the `PetscViewer` context

  Options Database Key:
. -tao_monitor_step_draw - draw the step direction at each iteration

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMonitorSet()`, `TaoMonitorSolutionDraw`
@*/
PetscErrorCode TaoMonitorStepDraw(Tao tao, void *ctx)
{
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(VecView(tao->stepdirection, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorResidual - Views the least-squares residual at each iteration of `TaoSolve()`

  Collective

  Input Parameters:
+ tao - the `Tao` context
- ctx - the `PetscViewer` context or `NULL`

  Options Database Key:
. -tao_monitor_ls_residual - view the residual at each iteration

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMonitorDefaultShort()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitorResidual(Tao tao, void *ctx)
{
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(VecView(tao->ls_res, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoDefaultConvergenceTest - Determines whether the solver should continue iterating
  or terminate.

  Collective

  Input Parameters:
+ tao   - the `Tao` context
- dummy - unused dummy context

  Level: developer

  Notes:
  This routine checks the residual in the optimality conditions, the
  relative residual in the optimity conditions, the number of function
  evaluations, and the function value to test convergence.  Some
  solvers may use different convergence routines.

.seealso: [](ch_tao), `Tao`, `TaoSetTolerances()`, `TaoGetConvergedReason()`, `TaoSetConvergedReason()`
@*/
PetscErrorCode TaoDefaultConvergenceTest(Tao tao, void *dummy)
{
  PetscInt           niter = tao->niter, nfuncs = PetscMax(tao->nfuncs, tao->nfuncgrads);
  PetscInt           max_funcs = tao->max_funcs;
  PetscReal          gnorm = tao->residual, gnorm0 = tao->gnorm0;
  PetscReal          f = tao->fc, steptol = tao->steptol, trradius = tao->step;
  PetscReal          gatol = tao->gatol, grtol = tao->grtol, gttol = tao->gttol;
  PetscReal          catol = tao->catol, crtol = tao->crtol;
  PetscReal          fmin = tao->fmin, cnorm = tao->cnorm;
  TaoConvergedReason reason = tao->reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  if (PetscIsInfOrNanReal(f)) {
    PetscCall(PetscInfo(tao, "Failed to converged, function value is Inf or NaN\n"));
    reason = TAO_DIVERGED_NAN;
  } else if (f <= fmin && cnorm <= catol) {
    PetscCall(PetscInfo(tao, "Converged due to function value %g < minimum function value %g\n", (double)f, (double)fmin));
    reason = TAO_CONVERGED_MINF;
  } else if (gnorm <= gatol && cnorm <= catol) {
    PetscCall(PetscInfo(tao, "Converged due to residual norm ||g(X)||=%g < %g\n", (double)gnorm, (double)gatol));
    reason = TAO_CONVERGED_GATOL;
  } else if (f != 0 && PetscAbsReal(gnorm / f) <= grtol && cnorm <= crtol) {
    PetscCall(PetscInfo(tao, "Converged due to residual ||g(X)||/|f(X)| =%g < %g\n", (double)(gnorm / f), (double)grtol));
    reason = TAO_CONVERGED_GRTOL;
  } else if (gnorm0 != 0 && ((gttol == 0 && gnorm == 0) || gnorm / gnorm0 < gttol) && cnorm <= crtol) {
    PetscCall(PetscInfo(tao, "Converged due to relative residual norm ||g(X)||/||g(X0)|| = %g < %g\n", (double)(gnorm / gnorm0), (double)gttol));
    reason = TAO_CONVERGED_GTTOL;
  } else if (max_funcs != PETSC_UNLIMITED && nfuncs > max_funcs) {
    PetscCall(PetscInfo(tao, "Exceeded maximum number of function evaluations: %" PetscInt_FMT " > %" PetscInt_FMT "\n", nfuncs, max_funcs));
    reason = TAO_DIVERGED_MAXFCN;
  } else if (tao->lsflag != 0) {
    PetscCall(PetscInfo(tao, "Tao Line Search failure.\n"));
    reason = TAO_DIVERGED_LS_FAILURE;
  } else if (trradius < steptol && niter > 0) {
    PetscCall(PetscInfo(tao, "Trust region/step size too small: %g < %g\n", (double)trradius, (double)steptol));
    reason = TAO_CONVERGED_STEPTOL;
  } else if (niter >= tao->max_it) {
    PetscCall(PetscInfo(tao, "Exceeded maximum number of iterations: %" PetscInt_FMT " > %" PetscInt_FMT "\n", niter, tao->max_it));
    reason = TAO_DIVERGED_MAXITS;
  } else {
    reason = TAO_CONTINUE_ITERATING;
  }
  tao->reason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetOptionsPrefix - Sets the prefix used for searching for all
  Tao options in the database.

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
- p   - the prefix string to prepend to all Tao option requests

  Level: advanced

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  For example, to distinguish between the runtime options for two
  different Tao solvers, one could call
.vb
      TaoSetOptionsPrefix(tao1,"sys1_")
      TaoSetOptionsPrefix(tao2,"sys2_")
.ve

  This would enable use of different options for each system, such as
.vb
      -sys1_tao_method blmvm -sys1_tao_grtol 1.e-3
      -sys2_tao_method lmvm  -sys2_tao_grtol 1.e-4
.ve

.seealso: [](ch_tao), `Tao`, `TaoSetFromOptions()`, `TaoAppendOptionsPrefix()`, `TaoGetOptionsPrefix()`
@*/
PetscErrorCode TaoSetOptionsPrefix(Tao tao, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao, p));
  if (tao->linesearch) PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, p));
  if (tao->ksp) PetscCall(KSPSetOptionsPrefix(tao->ksp, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoAppendOptionsPrefix - Appends to the prefix used for searching for all Tao options in the database.

  Logically Collective

  Input Parameters:
+ tao - the `Tao` solver context
- p   - the prefix string to prepend to all `Tao` option requests

  Level: advanced

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is automatically the hyphen.

.seealso: [](ch_tao), `Tao`, `TaoSetFromOptions()`, `TaoSetOptionsPrefix()`, `TaoGetOptionsPrefix()`
@*/
PetscErrorCode TaoAppendOptionsPrefix(Tao tao, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)tao, p));
  if (tao->linesearch) PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)tao->linesearch, p));
  if (tao->ksp) PetscCall(KSPAppendOptionsPrefix(tao->ksp, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetOptionsPrefix - Gets the prefix used for searching for all
  Tao options in the database

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. p - pointer to the prefix string used is returned

  Fortran Notes:
  Pass in a string 'prefix' of sufficient length to hold the prefix.

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoSetFromOptions()`, `TaoSetOptionsPrefix()`, `TaoAppendOptionsPrefix()`
@*/
PetscErrorCode TaoGetOptionsPrefix(Tao tao, const char *p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tao, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetType - Sets the `TaoType` for the minimization solver.

  Collective

  Input Parameters:
+ tao  - the `Tao` solver context
- type - a known method

  Options Database Key:
. -tao_type <type> - Sets the method; use -help for a list
   of available methods (for instance, "-tao_type lmvm" or "-tao_type tron")

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoGetType()`, `TaoType`
@*/
PetscErrorCode TaoSetType(Tao tao, TaoType type)
{
  PetscErrorCode (*create_xxx)(Tao);
  PetscBool issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);

  PetscCall(PetscObjectTypeCompare((PetscObject)tao, type, &issame));
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoList, type, (void (**)(void))&create_xxx));
  PetscCheck(create_xxx, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested Tao type %s", type);

  /* Destroy the existing solver information */
  PetscTryTypeMethod(tao, destroy);
  PetscCall(KSPDestroy(&tao->ksp));
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));
  tao->ops->setup          = NULL;
  tao->ops->solve          = NULL;
  tao->ops->view           = NULL;
  tao->ops->setfromoptions = NULL;
  tao->ops->destroy        = NULL;

  tao->setupcalled = PETSC_FALSE;

  PetscCall((*create_xxx)(tao));
  PetscCall(PetscObjectChangeTypeName((PetscObject)tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegister - Adds a method to the Tao package for minimization.

  Not Collective, No Fortran Support

  Input Parameters:
+ sname - name of a new user-defined solver
- func  - routine to Create method context

  Example Usage:
.vb
   TaoRegister("my_solver", MySolverCreate);
.ve

  Then, your solver can be chosen with the procedural interface via
$     TaoSetType(tao, "my_solver")
  or at runtime via the option
$     -tao_type my_solver

  Level: advanced

  Note:
  `TaoRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](ch_tao), `Tao`, `TaoSetType()`, `TaoRegisterAll()`, `TaoRegisterDestroy()`
@*/
PetscErrorCode TaoRegister(const char sname[], PetscErrorCode (*func)(Tao))
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegisterDestroy - Frees the list of minimization solvers that were
  registered by `TaoRegister()`.

  Not Collective

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoRegisterAll()`, `TaoRegister()`
@*/
PetscErrorCode TaoRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoList));
  TaoRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetIterationNumber - Gets the number of `TaoSolve()` iterations completed
  at this time.

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. iter - iteration number

  Notes:
  For example, during the computation of iteration 2 this would return 1.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoGetLinearSolveIterations()`, `TaoGetResidualNorm()`, `TaoGetObjective()`
@*/
PetscErrorCode TaoGetIterationNumber(Tao tao, PetscInt *iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(iter, 2);
  *iter = tao->niter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetResidualNorm - Gets the current value of the norm of the residual (gradient)
  at this time.

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. value - the current value

  Level: intermediate

  Developer Notes:
  This is the 2-norm of the residual, we cannot use `TaoGetGradientNorm()` because that has
  a different meaning. For some reason `Tao` sometimes calls the gradient the residual.

.seealso: [](ch_tao), `Tao`, `TaoGetLinearSolveIterations()`, `TaoGetIterationNumber()`, `TaoGetObjective()`
@*/
PetscErrorCode TaoGetResidualNorm(Tao tao, PetscReal *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(value, 2);
  *value = tao->residual;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetIterationNumber - Sets the current iteration number.

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
- iter - iteration number

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoGetLinearSolveIterations()`
@*/
PetscErrorCode TaoSetIterationNumber(Tao tao, PetscInt iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveInt(tao, iter, 2);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)tao));
  tao->niter = iter;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetTotalIterationNumber - Gets the total number of `TaoSolve()` iterations
  completed. This number keeps accumulating if multiple solves
  are called with the `Tao` object.

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. iter - number of iterations

  Level: intermediate

  Note:
  The total iteration count is updated after each solve, if there is a current
  `TaoSolve()` in progress then those iterations are not included in the count

.seealso: [](ch_tao), `Tao`, `TaoGetLinearSolveIterations()`
@*/
PetscErrorCode TaoGetTotalIterationNumber(Tao tao, PetscInt *iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(iter, 2);
  *iter = tao->ntotalits;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetTotalIterationNumber - Sets the current total iteration number.

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
- iter - the iteration number

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoGetLinearSolveIterations()`
@*/
PetscErrorCode TaoSetTotalIterationNumber(Tao tao, PetscInt iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveInt(tao, iter, 2);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)tao));
  tao->ntotalits = iter;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetConvergedReason - Sets the termination flag on a `Tao` object

  Logically Collective

  Input Parameters:
+ tao    - the `Tao` context
- reason - the `TaoConvergedReason`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoConvergedReason`
@*/
PetscErrorCode TaoSetConvergedReason(Tao tao, TaoConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(tao, reason, 2);
  tao->reason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetConvergedReason - Gets the reason the `TaoSolve()` was stopped.

  Not Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. reason - value of `TaoConvergedReason`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoConvergedReason`, `TaoSetConvergenceTest()`, `TaoSetTolerances()`
@*/
PetscErrorCode TaoGetConvergedReason(Tao tao, TaoConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(reason, 2);
  *reason = tao->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetSolutionStatus - Get the current iterate, objective value,
  residual, infeasibility, and termination from a `Tao` object

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ its    - the current iterate number (>=0)
. f      - the current function value
. gnorm  - the square of the gradient norm, duality gap, or other measure indicating distance from optimality.
. cnorm  - the infeasibility of the current solution with regard to the constraints.
. xdiff  - the step length or trust region radius of the most recent iterate.
- reason - The termination reason, which can equal `TAO_CONTINUE_ITERATING`

  Level: intermediate

  Notes:
  Tao returns the values set by the solvers in the routine `TaoMonitor()`.

  If any of the output arguments are set to `NULL`, no corresponding value will be returned.

.seealso: [](ch_tao), `TaoMonitor()`, `TaoGetConvergedReason()`
@*/
PetscErrorCode TaoGetSolutionStatus(Tao tao, PetscInt *its, PetscReal *f, PetscReal *gnorm, PetscReal *cnorm, PetscReal *xdiff, TaoConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (its) *its = tao->niter;
  if (f) *f = tao->fc;
  if (gnorm) *gnorm = tao->residual;
  if (cnorm) *cnorm = tao->cnorm;
  if (reason) *reason = tao->reason;
  if (xdiff) *xdiff = tao->step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetType - Gets the current `TaoType` being used in the `Tao` object

  Not Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. type - the `TaoType`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoType`, `TaoSetType()`
@*/
PetscErrorCode TaoGetType(Tao tao, TaoType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)tao)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitor - Monitor the solver and the current solution.  This
  routine will record the iteration number and residual statistics,
  and call any monitors specified by the user.

  Input Parameters:
+ tao        - the `Tao` context
. its        - the current iterate number (>=0)
. f          - the current objective function value
. res        - the gradient norm, square root of the duality gap, or other measure indicating distance from optimality.  This measure will be recorded and
          used for some termination tests.
. cnorm      - the infeasibility of the current solution with regard to the constraints.
- steplength - multiple of the step direction added to the previous iterate.

  Options Database Key:
. -tao_monitor - Use the default monitor, which prints statistics to standard output

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoGetConvergedReason()`, `TaoMonitorDefault()`, `TaoMonitorSet()`
@*/
PetscErrorCode TaoMonitor(Tao tao, PetscInt its, PetscReal f, PetscReal res, PetscReal cnorm, PetscReal steplength)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  tao->fc       = f;
  tao->residual = res;
  tao->cnorm    = cnorm;
  tao->step     = steplength;
  if (!its) {
    tao->cnorm0 = cnorm;
    tao->gnorm0 = res;
  }
  PetscCall(VecLockReadPush(tao->solution));
  for (i = 0; i < tao->numbermonitors; i++) PetscCall((*tao->monitor[i])(tao, tao->monitorcontext[i]));
  PetscCall(VecLockReadPop(tao->solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetConvergenceHistory - Sets the array used to hold the convergence history.

  Logically Collective

  Input Parameters:
+ tao   - the `Tao` solver context
. obj   - array to hold objective value history
. resid - array to hold residual history
. cnorm - array to hold constraint violation history
. lits  - integer array holds the number of linear iterations for each Tao iteration
. na    - size of `obj`, `resid`, and `cnorm`
- reset - `PETSC_TRUE` indicates each new minimization resets the history counter to zero,
           else it continues storing new values for new minimizations after the old ones

  Level: intermediate

  Notes:
  If set, `Tao` will fill the given arrays with the indicated
  information at each iteration.  If 'obj','resid','cnorm','lits' are
  *all* `NULL` then space (using size `na`, or 1000 if `na` is `PETSC_DECIDE`) is allocated for the history.
  If not all are `NULL`, then only the non-`NULL` information categories
  will be stored, the others will be ignored.

  Any convergence information after iteration number 'na' will not be stored.

  This routine is useful, e.g., when running a code for purposes
  of accurate performance monitoring, when no I/O should be done
  during the section of code that is being timed.

.seealso: [](ch_tao), `TaoGetConvergenceHistory()`
@*/
PetscErrorCode TaoSetConvergenceHistory(Tao tao, PetscReal obj[], PetscReal resid[], PetscReal cnorm[], PetscInt lits[], PetscInt na, PetscBool reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (obj) PetscAssertPointer(obj, 2);
  if (resid) PetscAssertPointer(resid, 3);
  if (cnorm) PetscAssertPointer(cnorm, 4);
  if (lits) PetscAssertPointer(lits, 5);

  if (na == PETSC_DECIDE || na == PETSC_CURRENT) na = 1000;
  if (!obj && !resid && !cnorm && !lits) {
    PetscCall(PetscCalloc4(na, &obj, na, &resid, na, &cnorm, na, &lits));
    tao->hist_malloc = PETSC_TRUE;
  }

  tao->hist_obj   = obj;
  tao->hist_resid = resid;
  tao->hist_cnorm = cnorm;
  tao->hist_lits  = lits;
  tao->hist_max   = na;
  tao->hist_reset = reset;
  tao->hist_len   = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoGetConvergenceHistory - Gets the arrays used that hold the convergence history.

  Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ obj   - array used to hold objective value history
. resid - array used to hold residual history
. cnorm - array used to hold constraint violation history
. lits  - integer array used to hold linear solver iteration count
- nhist - size of `obj`, `resid`, `cnorm`, and `lits`

  Level: advanced

  Notes:
  This routine must be preceded by calls to `TaoSetConvergenceHistory()`
  and `TaoSolve()`, otherwise it returns useless information.

  This routine is useful, e.g., when running a code for purposes
  of accurate performance monitoring, when no I/O should be done
  during the section of code that is being timed.

  Fortran Notes:
  The calling sequence is
.vb
   call TaoGetConvergenceHistory(Tao tao, PetscInt nhist, PetscErrorCode ierr)
.ve
  In other words this gets the current number of entries in the history. Access the history through the array you passed to `TaoSetConvergenceHistory()`

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoSetConvergenceHistory()`
@*/
PetscErrorCode TaoGetConvergenceHistory(Tao tao, PetscReal **obj, PetscReal **resid, PetscReal **cnorm, PetscInt **lits, PetscInt *nhist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (obj) *obj = tao->hist_obj;
  if (cnorm) *cnorm = tao->hist_cnorm;
  if (resid) *resid = tao->hist_resid;
  if (lits) *lits = tao->hist_lits;
  if (nhist) *nhist = tao->hist_len;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetApplicationContext - Sets the optional user-defined context for a `Tao` solver.

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
- usrP - optional user context

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoGetApplicationContext()`
@*/
PetscErrorCode TaoSetApplicationContext(Tao tao, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  tao->user = usrP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetApplicationContext - Gets the user-defined context for a `Tao` solver

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. usrP - user context

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoSetApplicationContext()`
@*/
PetscErrorCode TaoGetApplicationContext(Tao tao, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(usrP, 2);
  *(void **)usrP = tao->user;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetGradientNorm - Sets the matrix used to define the norm that measures the size of the gradient in some of the `Tao` algorithms

  Collective

  Input Parameters:
+ tao - the `Tao` context
- M   - matrix that defines the norm

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoGetGradientNorm()`, `TaoGradientNorm()`
@*/
PetscErrorCode TaoSetGradientNorm(Tao tao, Mat M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(M, MAT_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)M));
  PetscCall(MatDestroy(&tao->gradient_norm));
  PetscCall(VecDestroy(&tao->gradient_norm_tmp));
  tao->gradient_norm = M;
  PetscCall(MatCreateVecs(M, NULL, &tao->gradient_norm_tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetGradientNorm - Returns the matrix used to define the norm used for measuring the size of the gradient in some of the `Tao` algorithms

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. M - gradient norm

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoSetGradientNorm()`, `TaoGradientNorm()`
@*/
PetscErrorCode TaoGetGradientNorm(Tao tao, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(M, 2);
  *M = tao->gradient_norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGradientNorm - Compute the norm using the `NormType`, the user has selected

  Collective

  Input Parameters:
+ tao      - the `Tao` context
. gradient - the gradient
- type     - the norm type

  Output Parameter:
. gnorm - the gradient norm

  Level: advanced

  Note:
  If `TaoSetGradientNorm()` has been set and `type` is `NORM_2` then the norm provided with `TaoSetGradientNorm()` is used.

  Developer Notes:
  Should be named `TaoComputeGradientNorm()`.

  The usage is a bit confusing, with `TaoSetGradientNorm()` plus `NORM_2` resulting in the computation of the user provided
  norm, perhaps a refactorization is in order.

.seealso: [](ch_tao), `Tao`, `TaoSetGradientNorm()`, `TaoGetGradientNorm()`
@*/
PetscErrorCode TaoGradientNorm(Tao tao, Vec gradient, NormType type, PetscReal *gnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(gradient, VEC_CLASSID, 2);
  PetscValidLogicalCollectiveEnum(tao, type, 3);
  PetscAssertPointer(gnorm, 4);
  if (tao->gradient_norm) {
    PetscScalar gnorms;

    PetscCheck(type == NORM_2, PetscObjectComm((PetscObject)gradient), PETSC_ERR_ARG_WRONG, "Norm type must be NORM_2 if an inner product for the gradient norm is set.");
    PetscCall(MatMult(tao->gradient_norm, gradient, tao->gradient_norm_tmp));
    PetscCall(VecDot(gradient, tao->gradient_norm_tmp, &gnorms));
    *gnorm = PetscRealPart(PetscSqrtScalar(gnorms));
  } else {
    PetscCall(VecNorm(gradient, type, gnorm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorDrawCtxCreate - Creates the monitor context for `TaoMonitorSolutionDraw()`

  Collective

  Input Parameters:
+ comm     - the communicator to share the context
. host     - the name of the X Windows host that will display the monitor
. label    - the label to put at the top of the display window
. x        - the horizontal coordinate of the lower left corner of the window to open
. y        - the vertical coordinate of the lower left corner of the window to open
. m        - the width of the window
. n        - the height of the window
- howoften - how many `Tao` iterations between displaying the monitor information

  Output Parameter:
. ctx - the monitor context

  Options Database Keys:
+ -tao_monitor_solution_draw - use `TaoMonitorSolutionDraw()` to monitor the solution
- -tao_draw_solution_initial - show initial guess as well as current solution

  Level: intermediate

  Note:
  The context this creates, along with `TaoMonitorSolutionDraw()`, and `TaoMonitorDrawCtxDestroy()`
  are passed to `TaoMonitorSet()`.

.seealso: [](ch_tao), `Tao`, `TaoMonitorSet()`, `TaoMonitorDefault()`, `VecView()`, `TaoMonitorDrawCtx()`
@*/
PetscErrorCode TaoMonitorDrawCtxCreate(MPI_Comm comm, const char host[], const char label[], int x, int y, int m, int n, PetscInt howoften, TaoMonitorDrawCtx *ctx)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(ctx));
  PetscCall(PetscViewerDrawOpen(comm, host, label, x, y, m, n, &(*ctx)->viewer));
  PetscCall(PetscViewerSetFromOptions((*ctx)->viewer));
  (*ctx)->howoften = howoften;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoMonitorDrawCtxDestroy - Destroys the monitor context for `TaoMonitorSolutionDraw()`

  Collective

  Input Parameter:
. ictx - the monitor context

  Level: intermediate

  Note:
  This is passed to `TaoMonitorSet()` as the final argument, along with `TaoMonitorSolutionDraw()`, and the context
  obtained with `TaoMonitorDrawCtxCreate()`.

.seealso: [](ch_tao), `Tao`, `TaoMonitorSet()`, `TaoMonitorDefault()`, `VecView()`, `TaoMonitorSolutionDraw()`
@*/
PetscErrorCode TaoMonitorDrawCtxDestroy(TaoMonitorDrawCtx *ictx)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&(*ictx)->viewer));
  PetscCall(PetscFree(*ictx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
