#include <petsc/private/linesearchimpl.h> /*I "petscsnes.h" I*/

PetscBool         SNESLineSearchRegisterAllCalled = PETSC_FALSE;
PetscFunctionList SNESLineSearchList              = NULL;

PetscClassId  SNESLINESEARCH_CLASSID;
PetscLogEvent SNESLINESEARCH_Apply;

/*@
  SNESLineSearchMonitorCancel - Clears all the monitor functions for a `SNESLineSearch` object.

  Logically Collective

  Input Parameter:
. ls - the `SNESLineSearch` context

  Options Database Key:
. -snes_linesearch_monitor_cancel - cancels all monitors that have been hardwired
    into a code by calls to `SNESLineSearchMonitorSet()`, but does not cancel those
    set via the options database

  Level: advanced

  Notes:
  There is no way to clear one specific monitor from a `SNESLineSearch` object.

  This does not clear the monitor set with `SNESLineSearchSetDefaultMonitor()` use `SNESLineSearchSetDefaultMonitor`(`ls`,`NULL`) to cancel it
  that one.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchMonitorDefault()`, `SNESLineSearchMonitorSet()`
@*/
PetscErrorCode SNESLineSearchMonitorCancel(SNESLineSearch ls)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, SNESLINESEARCH_CLASSID, 1);
  for (i = 0; i < ls->numbermonitors; i++) {
    if (ls->monitordestroy[i]) PetscCall((*ls->monitordestroy[i])(&ls->monitorcontext[i]));
  }
  ls->numbermonitors = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchMonitor - runs the user provided monitor routines, if they exist

  Collective

  Input Parameter:
. ls - the linesearch object

  Level: developer

  Note:
  This routine is called by the `SNESLineSearch` implementations.
  It does not typically need to be called by the user.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchMonitorSet()`
@*/
PetscErrorCode SNESLineSearchMonitor(SNESLineSearch ls)
{
  PetscInt i, n = ls->numbermonitors;

  PetscFunctionBegin;
  for (i = 0; i < n; i++) PetscCall((*ls->monitorftns[i])(ls, ls->monitorcontext[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchMonitorSet - Sets an ADDITIONAL function that is to be used at every
  iteration of the nonlinear solver to display the iteration's
  progress.

  Logically Collective

  Input Parameters:
+ ls             - the `SNESLineSearch` context
. f              - the monitor function
. mctx           - [optional] user-defined context for private data for the monitor routine (use `NULL` if no context is desired)
- monitordestroy - [optional] routine that frees monitor context (may be `NULL`), see `PetscCtxDestroyFn` for the calling sequence

  Calling sequence of `f`:
+ ls   - the `SNESLineSearch` context
- mctx - [optional] user-defined context for private data for the monitor routine

  Level: intermediate

  Note:
  Several different monitoring routines may be set by calling
  `SNESLineSearchMonitorSet()` multiple times; all will be called in the
  order in which they were set.

  Fortran Note:
  Only a single monitor function can be set for each `SNESLineSearch` object

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchMonitorDefault()`, `SNESLineSearchMonitorCancel()`, `PetscCtxDestroyFn`
@*/
PetscErrorCode SNESLineSearchMonitorSet(SNESLineSearch ls, PetscErrorCode (*f)(SNESLineSearch ls, void *mctx), void *mctx, PetscCtxDestroyFn *monitordestroy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, SNESLINESEARCH_CLASSID, 1);
  for (PetscInt i = 0; i < ls->numbermonitors; i++) {
    PetscBool identical;

    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))(PetscVoidFn *)f, mctx, monitordestroy, (PetscErrorCode (*)(void))(PetscVoidFn *)ls->monitorftns[i], ls->monitorcontext[i], ls->monitordestroy[i], &identical));
    if (identical) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(ls->numbermonitors < MAXSNESLSMONITORS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many monitors set");
  ls->monitorftns[ls->numbermonitors]      = f;
  ls->monitordestroy[ls->numbermonitors]   = monitordestroy;
  ls->monitorcontext[ls->numbermonitors++] = mctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchMonitorSolutionUpdate - Monitors each update of the function value the linesearch tries

  Collective

  Input Parameters:
+ ls - the `SNESLineSearch` object
- vf - the context for the monitor, in this case it is an `PetscViewerAndFormat`

  Options Database Key:
. -snes_linesearch_monitor_solution_update [viewer:filename:format] - view each update tried by line search routine

  Level: developer

  This is not normally called directly but is passed to `SNESLineSearchMonitorSet()`

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchMonitorSet()`, `SNESMonitorSolution()`
@*/
PetscErrorCode SNESLineSearchMonitorSolutionUpdate(SNESLineSearch ls, PetscViewerAndFormat *vf)
{
  PetscViewer viewer = vf->viewer;
  Vec         Y, W, G;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(ls, NULL, NULL, &Y, &W, &G));
  PetscCall(PetscViewerPushFormat(viewer, vf->format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LineSearch attempted update to solution \n"));
  PetscCall(VecView(Y, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LineSearch attempted new solution \n"));
  PetscCall(VecView(W, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LineSearch attempted updated function value\n"));
  PetscCall(VecView(G, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchCreate - Creates a `SNESLineSearch` context.

  Logically Collective

  Input Parameter:
. comm - MPI communicator for the line search (typically from the associated `SNES` context).

  Output Parameter:
. outlinesearch - the new line search context

  Level: developer

  Note:
  The preferred calling sequence is to use `SNESGetLineSearch()` to acquire the `SNESLineSearch` instance
  already associated with the `SNES`.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `LineSearchDestroy()`, `SNESGetLineSearch()`
@*/
PetscErrorCode SNESLineSearchCreate(MPI_Comm comm, SNESLineSearch *outlinesearch)
{
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  PetscAssertPointer(outlinesearch, 2);
  PetscCall(SNESInitializePackage());

  PetscCall(PetscHeaderCreate(linesearch, SNESLINESEARCH_CLASSID, "SNESLineSearch", "Linesearch", "SNESLineSearch", comm, SNESLineSearchDestroy, SNESLineSearchView));
  linesearch->vec_sol_new  = NULL;
  linesearch->vec_func_new = NULL;
  linesearch->vec_sol      = NULL;
  linesearch->vec_func     = NULL;
  linesearch->vec_update   = NULL;

  linesearch->lambda       = 1.0;
  linesearch->fnorm        = 1.0;
  linesearch->ynorm        = 1.0;
  linesearch->xnorm        = 1.0;
  linesearch->result       = SNES_LINESEARCH_SUCCEEDED;
  linesearch->norms        = PETSC_TRUE;
  linesearch->keeplambda   = PETSC_FALSE;
  linesearch->damping      = 1.0;
  linesearch->maxlambda    = 1.0;
  linesearch->minlambda    = 1e-12;
  linesearch->rtol         = 1e-8;
  linesearch->atol         = 1e-15;
  linesearch->ltol         = 1e-8;
  linesearch->precheckctx  = NULL;
  linesearch->postcheckctx = NULL;
  linesearch->max_it       = 1;
  linesearch->setupcalled  = PETSC_FALSE;
  linesearch->monitor      = NULL;
  *outlinesearch           = linesearch;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetUp - Prepares the line search for being applied by allocating
  any required vectors.

  Collective

  Input Parameter:
. linesearch - The `SNESLineSearch` instance.

  Level: advanced

  Note:
  For most cases, this needn't be called by users or outside of `SNESLineSearchApply()`.
  The only current case where this is called outside of this is for the VI
  solvers, which modify the solution and work vectors before the first call
  of `SNESLineSearchApply()`, requiring the `SNESLineSearch` work vectors to be
  allocated upfront.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchReset()`
@*/
PetscErrorCode SNESLineSearchSetUp(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  if (!((PetscObject)linesearch)->type_name) PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC));
  if (!linesearch->setupcalled) {
    if (!linesearch->vec_sol_new) PetscCall(VecDuplicate(linesearch->vec_sol, &linesearch->vec_sol_new));
    if (!linesearch->vec_func_new) PetscCall(VecDuplicate(linesearch->vec_sol, &linesearch->vec_func_new));
    PetscTryTypeMethod(linesearch, setup);
    if (!linesearch->ops->snesfunc) PetscCall(SNESLineSearchSetFunction(linesearch, SNESComputeFunction));
    linesearch->lambda      = linesearch->damping;
    linesearch->setupcalled = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchReset - Undoes the `SNESLineSearchSetUp()` and deletes any `Vec`s or `Mat`s allocated by the line search.

  Collective

  Input Parameter:
. linesearch - The `SNESLineSearch` instance.

  Level: developer

  Note:
  Usually only called by `SNESReset()`

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchSetUp()`
@*/
PetscErrorCode SNESLineSearchReset(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  PetscTryTypeMethod(linesearch, reset);

  PetscCall(VecDestroy(&linesearch->vec_sol_new));
  PetscCall(VecDestroy(&linesearch->vec_func_new));

  PetscCall(VecDestroyVecs(linesearch->nwork, &linesearch->work));

  linesearch->nwork       = 0;
  linesearch->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchSetFunction - Sets the function evaluation used by the `SNES` line search
  `

  Input Parameters:
+ linesearch - the `SNESLineSearch` context
- func       - function evaluation routine, this is usually the function provided with `SNESSetFunction()`

  Calling sequence of `func`:
+ snes - the `SNES` with which the `SNESLineSearch` context is associated with
. x    - the input vector
- f    - the computed value of the function

  Level: developer

  Note:
  By default the `SNESLineSearch` uses the function provided by `SNESSetFunction()` so this is rarely needed

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESSetFunction()`
@*/
PetscErrorCode SNESLineSearchSetFunction(SNESLineSearch linesearch, PetscErrorCode (*func)(SNES snes, Vec x, Vec f))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  linesearch->ops->snesfunc = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchSetPreCheck - Sets a function that is called after the initial search direction has been computed but
  before the line search routine has been applied. Allows adjusting the result of (usually a linear solve) that
  determined the search direction.

  Logically Collective

  Input Parameters:
+ linesearch - the `SNESLineSearch` context
. func       - [optional] function evaluation routine
- ctx        - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Calling sequence of `func`:
+ ls        - the `SNESLineSearch` context
. x         - the current solution
. d         - the current search direction
. changed_d - indicates if the search direction has been changed
- ctx       - the context passed to `SNESLineSearchSetPreCheck()`

  Level: intermediate

  Note:
  Use `SNESLineSearchSetPostCheck()` to change the step after the line search is complete.

  Use `SNESVISetVariableBounds()` and `SNESVISetComputeVariableBounds()` to cause `SNES` to automatically control the ranges of variables allowed.

.seealso: [](ch_snes), `SNES`, `SNESGetLineSearch()`, `SNESLineSearchPreCheck()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchGetPreCheck()`,
          `SNESVISetVariableBounds()`, `SNESVISetComputeVariableBounds()`, `SNESSetFunctionDomainError()`, `SNESSetJacobianDomainError()`

@*/
PetscErrorCode SNESLineSearchSetPreCheck(SNESLineSearch linesearch, PetscErrorCode (*func)(SNESLineSearch ls, Vec x, Vec d, PetscBool *changed_d, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) linesearch->ops->precheck = func;
  if (ctx) linesearch->precheckctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchGetPreCheck - Gets the pre-check function for the line search routine.

  Input Parameter:
. linesearch - the `SNESLineSearch` context

  Output Parameters:
+ func - [optional] function evaluation routine,  for calling sequence see `SNESLineSearchSetPreCheck()`
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchPreCheck()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchSetPreCheck()`, `SNESLineSearchSetPostCheck()`
@*/
PetscErrorCode SNESLineSearchGetPreCheck(SNESLineSearch linesearch, PetscErrorCode (**func)(SNESLineSearch, Vec, Vec, PetscBool *, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) *func = linesearch->ops->precheck;
  if (ctx) *ctx = linesearch->precheckctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchSetPostCheck - Sets a user function that is called after the line search has been applied to determine the step
  direction and length. Allows the user a chance to change or override the decision of the line search routine

  Logically Collective

  Input Parameters:
+ linesearch - the `SNESLineSearch` context
. func       - [optional] function evaluation routine
- ctx        - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Calling sequence of `func`:
+ ls        - the `SNESLineSearch` context
. x         - the current solution
. d         - the current search direction
. w         - $ w = x + lambda*d $ for some lambda
. changed_d - indicates if the search direction `d` has been changed
. changed_w - indicates `w` has been changed
- ctx       - the context passed to `SNESLineSearchSetPreCheck()`

  Level: intermediate

  Notes:
  Use `SNESLineSearchSetPreCheck()` to change the step before the line search is completed.
  The calling sequence of the callback does not contain the current scaling factor. To access the value, use `SNESLineSearchGetLambda()`.

  Use `SNESVISetVariableBounds()` and `SNESVISetComputeVariableBounds()` to cause `SNES` to automatically control the ranges of variables allowed.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchPostCheck()`, `SNESLineSearchSetPreCheck()`, `SNESLineSearchGetPreCheck()`, `SNESLineSearchGetPostCheck()`,
          `SNESVISetVariableBounds()`, `SNESVISetComputeVariableBounds()`, `SNESSetFunctionDomainError()`, `SNESSetJacobianDomainError()`
@*/
PetscErrorCode SNESLineSearchSetPostCheck(SNESLineSearch linesearch, PetscErrorCode (*func)(SNESLineSearch ls, Vec x, Vec d, Vec w, PetscBool *changed_d, PetscBool *changed_w, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) linesearch->ops->postcheck = func;
  if (ctx) linesearch->postcheckctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchGetPostCheck - Gets the post-check function for the line search routine.

  Input Parameter:
. linesearch - the `SNESLineSearch` context

  Output Parameters:
+ func - [optional] function evaluation routine, see for the calling sequence `SNESLineSearchSetPostCheck()`
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchGetPreCheck()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchPostCheck()`, `SNESLineSearchSetPreCheck()`
@*/
PetscErrorCode SNESLineSearchGetPostCheck(SNESLineSearch linesearch, PetscErrorCode (**func)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) *func = linesearch->ops->postcheck;
  if (ctx) *ctx = linesearch->postcheckctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchPreCheck - Prepares the line search for being applied.

  Logically Collective

  Input Parameters:
+ linesearch - The linesearch instance.
. X          - The current solution
- Y          - The step direction

  Output Parameter:
. changed - Indicator that the precheck routine has changed `Y`

  Level: advanced

  Note:
  This calls any function provided with `SNESLineSearchSetPreCheck()` and is called automatically inside the line search routines

  Developer Note:
  The use of `PetscObjectGetState()` would eliminate the need for the `changed` argument to be provided

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchPostCheck()`, `SNESLineSearchSetPreCheck()`, `SNESLineSearchGetPreCheck()`, `SNESLineSearchSetPostCheck()`,
          `SNESLineSearchGetPostCheck()`
@*/
PetscErrorCode SNESLineSearchPreCheck(SNESLineSearch linesearch, Vec X, Vec Y, PetscBool *changed)
{
  PetscFunctionBegin;
  *changed = PETSC_FALSE;
  if (linesearch->ops->precheck) {
    PetscUseTypeMethod(linesearch, precheck, X, Y, changed, linesearch->precheckctx);
    PetscValidLogicalCollectiveBool(linesearch, *changed, 4);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchPostCheck - Hook to modify step direction or updated solution after a successful linesearch

  Logically Collective

  Input Parameters:
+ linesearch - The line search context
. X          - The last solution
. Y          - The step direction
- W          - The updated solution, `W = X - lambda * Y` for some lambda

  Output Parameters:
+ changed_Y - Indicator if the direction `Y` has been changed.
- changed_W - Indicator if the new candidate solution `W` has been changed.

  Level: developer

  Note:
  This calls any function provided with `SNESLineSearchSetPostCheck()` and is called automatically inside the line search routines

  Developer Note:
  The use of `PetscObjectGetState()` would eliminate the need for the `changed_Y` and `changed_W` arguments to be provided

.seealso: [](ch_snes), `SNES`, `SNESGetLineSearch()`, `SNESLineSearchPreCheck()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchSetPrecheck()`, `SNESLineSearchGetPrecheck()`
@*/
PetscErrorCode SNESLineSearchPostCheck(SNESLineSearch linesearch, Vec X, Vec Y, Vec W, PetscBool *changed_Y, PetscBool *changed_W)
{
  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  if (linesearch->ops->postcheck) {
    PetscUseTypeMethod(linesearch, postcheck, X, Y, W, changed_Y, changed_W, linesearch->postcheckctx);
    PetscValidLogicalCollectiveBool(linesearch, *changed_Y, 5);
    PetscValidLogicalCollectiveBool(linesearch, *changed_W, 6);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchPreCheckPicard - Implements a correction that is sometimes useful to improve the convergence rate of Picard iteration {cite}`hindmarsh1996time`

  Logically Collective

  Input Parameters:
+ linesearch - the line search context
. X          - base state for this step
- ctx        - context for this function

  Input/Output Parameter:
. Y - correction, possibly modified

  Output Parameter:
. changed - flag indicating that `Y` was modified

  Options Database Keys:
+ -snes_linesearch_precheck_picard       - activate this routine
- -snes_linesearch_precheck_picard_angle - angle

  Level: advanced

  Notes:
  This function should be passed to `SNESLineSearchSetPreCheck()`

  The justification for this method involves the linear convergence of a Picard iteration
  so the Picard linearization should be provided in place of the "Jacobian"  {cite}`hindmarsh1996time`. This correction
  is generally not useful when using a Newton linearization.

  Developer Note:
  The use of `PetscObjectGetState()` would eliminate the need for the `changed` argument to be provided

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESSetPicard()`, `SNESGetLineSearch()`, `SNESLineSearchSetPreCheck()`, `SNESLineSearchSetPostCheck()`
@*/
PetscErrorCode SNESLineSearchPreCheckPicard(SNESLineSearch linesearch, Vec X, Vec Y, PetscBool *changed, void *ctx)
{
  PetscReal   angle = *(PetscReal *)linesearch->precheckctx;
  Vec         Ylast;
  PetscScalar dot;
  PetscInt    iter;
  PetscReal   ynorm, ylastnorm, theta, angle_radians;
  SNES        snes;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(PetscObjectQuery((PetscObject)snes, "SNESLineSearchPreCheckPicard_Ylast", (PetscObject *)&Ylast));
  if (!Ylast) {
    PetscCall(VecDuplicate(Y, &Ylast));
    PetscCall(PetscObjectCompose((PetscObject)snes, "SNESLineSearchPreCheckPicard_Ylast", (PetscObject)Ylast));
    PetscCall(PetscObjectDereference((PetscObject)Ylast));
  }
  PetscCall(SNESGetIterationNumber(snes, &iter));
  if (iter < 2) {
    PetscCall(VecCopy(Y, Ylast));
    *changed = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecDot(Y, Ylast, &dot));
  PetscCall(VecNorm(Y, NORM_2, &ynorm));
  PetscCall(VecNorm(Ylast, NORM_2, &ylastnorm));
  if (ynorm == 0. || ylastnorm == 0.) {
    *changed = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Compute the angle between the vectors Y and Ylast, clip to keep inside the domain of acos() */
  theta         = PetscAcosReal((PetscReal)PetscClipInterval(PetscAbsScalar(dot) / (ynorm * ylastnorm), -1.0, 1.0));
  angle_radians = angle * PETSC_PI / 180.;
  if (PetscAbsReal(theta) < angle_radians || PetscAbsReal(theta - PETSC_PI) < angle_radians) {
    /* Modify the step Y */
    PetscReal alpha, ydiffnorm;
    PetscCall(VecAXPY(Ylast, -1.0, Y));
    PetscCall(VecNorm(Ylast, NORM_2, &ydiffnorm));
    alpha = (ydiffnorm > .001 * ylastnorm) ? ylastnorm / ydiffnorm : 1000.0;
    PetscCall(VecCopy(Y, Ylast));
    PetscCall(VecScale(Y, alpha));
    PetscCall(PetscInfo(snes, "Angle %14.12e degrees less than threshold %14.12e, corrected step by alpha=%14.12e\n", (double)(theta * 180 / PETSC_PI), (double)angle, (double)alpha));
    *changed = PETSC_TRUE;
  } else {
    PetscCall(PetscInfo(snes, "Angle %14.12e degrees exceeds threshold %14.12e, no correction applied\n", (double)(theta * 180 / PETSC_PI), (double)angle));
    PetscCall(VecCopy(Y, Ylast));
    *changed = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchApply - Computes the line-search update.

  Collective

  Input Parameter:
. linesearch - The line search context

  Input/Output Parameters:
+ X     - The current solution, on output the new solution
. F     - The current function value, on output the new function value at the solution value `X`
. fnorm - The current norm of `F`, on output the new norm of `F`
- Y     - The current search direction, on output the direction determined by the linesearch, i.e. `Xnew = Xold - lambda*Y`

  Options Database Keys:
+ -snes_linesearch_type                - basic (or equivalently none), bt, secant, cp, nleqerr, bisection, shell
. -snes_linesearch_monitor [:filename] - Print progress of line searches
. -snes_linesearch_damping             - The linesearch damping parameter, default is 1.0 (no damping)
. -snes_linesearch_norms               - Turn on/off the linesearch norms computation (SNESLineSearchSetComputeNorms())
. -snes_linesearch_keeplambda          - Keep the previous `lambda` as the initial guess
- -snes_linesearch_max_it              - The number of iterations for iterative line searches

  Level: intermediate

  Notes:
  This is typically called from within a `SNESSolve()` implementation in order to
  help with convergence of the nonlinear method.  Various `SNES` types use line searches
  in different ways, but the overarching theme is that a line search is used to determine
  an optimal damping parameter (that is `lambda`) of a step at each iteration of the method. Each
  application of the line search may invoke `SNESComputeFunction()` several times, and
  therefore may be fairly expensive.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchCreate()`, `SNESLineSearchGetLambda()`, `SNESLineSearchPreCheck()`, `SNESLineSearchPostCheck()`, `SNESSolve()`, `SNESComputeFunction()`, `SNESLineSearchSetComputeNorms()`,
          `SNESLineSearchType`, `SNESLineSearchSetType()`
@*/
PetscErrorCode SNESLineSearchApply(SNESLineSearch linesearch, Vec X, Vec F, PetscReal *fnorm, Vec Y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 5);

  linesearch->result = SNES_LINESEARCH_SUCCEEDED;

  linesearch->vec_sol    = X;
  linesearch->vec_update = Y;
  linesearch->vec_func   = F;

  PetscCall(SNESLineSearchSetUp(linesearch));

  if (!linesearch->keeplambda) linesearch->lambda = linesearch->damping; /* set the initial guess to lambda */

  if (fnorm) linesearch->fnorm = *fnorm;
  else PetscCall(VecNorm(F, NORM_2, &linesearch->fnorm));

  PetscCall(PetscLogEventBegin(SNESLINESEARCH_Apply, linesearch, X, F, Y));

  PetscUseTypeMethod(linesearch, apply);

  PetscCall(PetscLogEventEnd(SNESLINESEARCH_Apply, linesearch, X, F, Y));

  if (fnorm) *fnorm = linesearch->fnorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchDestroy - Destroys the line search instance.

  Collective

  Input Parameter:
. linesearch - The line search context

  Level: developer

  Note:
  The line search in `SNES` is automatically called on `SNESDestroy()` so this call is rarely needed

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchCreate()`, `SNESLineSearchReset()`, `SNESDestroy()`
@*/
PetscErrorCode SNESLineSearchDestroy(SNESLineSearch *linesearch)
{
  PetscFunctionBegin;
  if (!*linesearch) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*linesearch, SNESLINESEARCH_CLASSID, 1);
  if (--((PetscObject)*linesearch)->refct > 0) {
    *linesearch = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*linesearch));
  PetscCall(SNESLineSearchReset(*linesearch));
  PetscTryTypeMethod(*linesearch, destroy);
  PetscCall(PetscViewerDestroy(&(*linesearch)->monitor));
  PetscCall(SNESLineSearchMonitorCancel(*linesearch));
  PetscCall(PetscHeaderDestroy(linesearch));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetDefaultMonitor - Turns on/off printing useful information and debugging output about the line search.

  Logically Collective

  Input Parameters:
+ linesearch - the linesearch object
- viewer     - an `PETSCVIEWERASCII` `PetscViewer` or `NULL` to turn off monitor

  Options Database Key:
. -snes_linesearch_monitor [:filename] - enables the monitor

  Level: intermediate

  Developer Notes:
  This monitor is implemented differently than the other line search monitors that are set with
  `SNESLineSearchMonitorSet()` since it is called in many locations of the line search routines to display aspects of the
  line search that are not visible to the other monitors.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `PETSCVIEWERASCII`, `SNESGetLineSearch()`, `SNESLineSearchGetDefaultMonitor()`, `PetscViewer`, `SNESLineSearchSetMonitor()`,
          `SNESLineSearchMonitorSetFromOptions()`
@*/
PetscErrorCode SNESLineSearchSetDefaultMonitor(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&linesearch->monitor));
  linesearch->monitor = viewer;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetDefaultMonitor - Gets the `PetscViewer` instance for the default line search monitor that is turned on with `SNESLineSearchSetDefaultMonitor()`

  Logically Collective

  Input Parameter:
. linesearch - the line search context

  Output Parameter:
. monitor - monitor context

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchSetDefaultMonitor()`, `PetscViewer`
@*/
PetscErrorCode SNESLineSearchGetDefaultMonitor(SNESLineSearch linesearch, PetscViewer *monitor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  *monitor = linesearch->monitor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated in the options database

  Collective

  Input Parameters:
+ ls           - `SNESLineSearch` object to monitor
. name         - the monitor type
. help         - message indicating what monitoring is done
. manual       - manual page for the monitor
. monitor      - the monitor function, must use `PetscViewerAndFormat` as its context
- monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the `SNESLineSearch` or `PetscViewer`

  Calling sequence of `monitor`:
+ ls - `SNESLineSearch` object being monitored
- vf - a `PetscViewerAndFormat` struct that provides the `PetscViewer` and `PetscViewerFormat` being used

  Calling sequence of `monitorsetup`:
+ ls - `SNESLineSearch` object being monitored
- vf - a `PetscViewerAndFormat` struct that provides the `PetscViewer` and `PetscViewerFormat` being used

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetMonitor()`, `PetscOptionsCreateViewer()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode SNESLineSearchMonitorSetFromOptions(SNESLineSearch ls, const char name[], const char help[], const char manual[], PetscErrorCode (*monitor)(SNESLineSearch ls, PetscViewerAndFormat *vf), PetscErrorCode (*monitorsetup)(SNESLineSearch ls, PetscViewerAndFormat *vf))
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsCreateViewer(PetscObjectComm((PetscObject)ls), ((PetscObject)ls)->options, ((PetscObject)ls)->prefix, name, &viewer, &format, &flg));
  if (flg) {
    PetscViewerAndFormat *vf;
    PetscCall(PetscViewerAndFormatCreate(viewer, format, &vf));
    PetscCall(PetscViewerDestroy(&viewer));
    if (monitorsetup) PetscCall((*monitorsetup)(ls, vf));
    PetscCall(SNESLineSearchMonitorSet(ls, (PetscErrorCode (*)(SNESLineSearch, void *))monitor, vf, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetFromOptions - Sets options for the line search

  Logically Collective

  Input Parameter:
. linesearch - a `SNESLineSearch` line search context

  Options Database Keys:
+ -snes_linesearch_type <type>                                      - basic (or equivalently none), `bt`, `secant`, `cp`, `nleqerr`, `bisection`, `shell`
. -snes_linesearch_order <order>                                    - 1, 2, 3.  Most types only support certain orders (`bt` supports 1, 2 or 3)
. -snes_linesearch_norms                                            - Turn on/off the linesearch norms for the basic linesearch typem (`SNESLineSearchSetComputeNorms()`)
. -snes_linesearch_minlambda                                        - The minimum `lambda`
. -snes_linesearch_maxlambda                                        - The maximum `lambda`
. -snes_linesearch_rtol                                             - Relative tolerance for iterative line searches
. -snes_linesearch_atol                                             - Absolute tolerance for iterative line searches
. -snes_linesearch_ltol                                             - Change in `lambda` tolerance for iterative line searches
. -snes_linesearch_max_it                                           - The number of iterations for iterative line searches
. -snes_linesearch_monitor [:filename]                              - Print progress of line searches
. -snes_linesearch_monitor_solution_update [viewer:filename:format] - view each update tried by line search routine
. -snes_linesearch_damping                                          - The linesearch damping parameter
. -snes_linesearch_keeplambda                                       - Keep the previous `lambda` as the initial guess.
. -snes_linesearch_precheck_picard                                  - Use precheck that speeds up convergence of picard method
- -snes_linesearch_precheck_picard_angle                            - Angle used in Picard precheck method

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESGetLineSearch()`, `SNESLineSearchCreate()`, `SNESLineSearchSetOrder()`, `SNESLineSearchSetType()`, `SNESLineSearchSetTolerances()`, `SNESLineSearchSetDamping()`, `SNESLineSearchPreCheckPicard()`,
          `SNESLineSearchType`, `SNESLineSearchSetComputeNorms()`
@*/
PetscErrorCode SNESLineSearchSetFromOptions(SNESLineSearch linesearch)
{
  const char *deft = SNESLINESEARCHBASIC;
  char        type[256];
  PetscBool   flg, set;
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchRegisterAll());

  PetscObjectOptionsBegin((PetscObject)linesearch);
  if (((PetscObject)linesearch)->type_name) deft = ((PetscObject)linesearch)->type_name;
  PetscCall(PetscOptionsFList("-snes_linesearch_type", "Linesearch type", "SNESLineSearchSetType", SNESLineSearchList, deft, type, 256, &flg));
  if (flg) {
    PetscCall(SNESLineSearchSetType(linesearch, type));
  } else if (!((PetscObject)linesearch)->type_name) {
    PetscCall(SNESLineSearchSetType(linesearch, deft));
  }

  PetscCall(PetscOptionsCreateViewer(PetscObjectComm((PetscObject)linesearch), ((PetscObject)linesearch)->options, ((PetscObject)linesearch)->prefix, "-snes_linesearch_monitor", &viewer, NULL, &set));
  if (set) PetscCall(SNESLineSearchSetDefaultMonitor(linesearch, viewer));
  PetscCall(SNESLineSearchMonitorSetFromOptions(linesearch, "-snes_linesearch_monitor_solution_update", "View correction at each iteration", "SNESLineSearchMonitorSolutionUpdate", SNESLineSearchMonitorSolutionUpdate, NULL));

  /* tolerances */
  PetscCall(PetscOptionsReal("-snes_linesearch_minlambda", "Minimum lambda", "SNESLineSearchSetTolerances", linesearch->minlambda, &linesearch->minlambda, NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_maxlambda", "Maximum lambda", "SNESLineSearchSetTolerances", linesearch->maxlambda, &linesearch->maxlambda, NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_rtol", "Relative tolerance for iterative line search", "SNESLineSearchSetTolerances", linesearch->rtol, &linesearch->rtol, NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_atol", "Absolute tolerance for iterative line search", "SNESLineSearchSetTolerances", linesearch->atol, &linesearch->atol, NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_ltol", "Change in lambda tolerance for iterative line search", "SNESLineSearchSetTolerances", linesearch->ltol, &linesearch->ltol, NULL));
  PetscCall(PetscOptionsInt("-snes_linesearch_max_it", "Maximum iterations for iterative line searches", "SNESLineSearchSetTolerances", linesearch->max_it, &linesearch->max_it, NULL));

  /* deprecated options */
  PetscCall(PetscOptionsDeprecated("-snes_linesearch_maxstep", "-snes_linesearch_maxlambda", "3.24.0", NULL));

  /* damping parameters */
  PetscCall(PetscOptionsReal("-snes_linesearch_damping", "Line search damping (and depending on chosen line search initial lambda guess)", "SNESLineSearchSetDamping", linesearch->damping, &linesearch->damping, NULL));

  PetscCall(PetscOptionsBool("-snes_linesearch_keeplambda", "Use previous lambda as damping", "SNESLineSearchSetKeepLambda", linesearch->keeplambda, &linesearch->keeplambda, NULL));

  /* precheck */
  PetscCall(PetscOptionsBool("-snes_linesearch_precheck_picard", "Use a correction that sometimes improves convergence of Picard iteration", "SNESLineSearchPreCheckPicard", flg, &flg, &set));
  if (set) {
    if (flg) {
      linesearch->precheck_picard_angle = 10.; /* correction only active if angle is less than 10 degrees */

      PetscCall(PetscOptionsReal("-snes_linesearch_precheck_picard_angle", "Maximum angle at which to activate the correction", "none", linesearch->precheck_picard_angle, &linesearch->precheck_picard_angle, NULL));
      PetscCall(SNESLineSearchSetPreCheck(linesearch, SNESLineSearchPreCheckPicard, &linesearch->precheck_picard_angle));
    } else {
      PetscCall(SNESLineSearchSetPreCheck(linesearch, NULL, NULL));
    }
  }
  PetscCall(PetscOptionsInt("-snes_linesearch_order", "Order of approximation used in the line search", "SNESLineSearchSetOrder", linesearch->order, &linesearch->order, NULL));
  PetscCall(PetscOptionsBool("-snes_linesearch_norms", "Compute final norms in line search", "SNESLineSearchSetComputeNorms", linesearch->norms, &linesearch->norms, NULL));

  PetscTryTypeMethod(linesearch, setfromoptions, PetscOptionsObject);

  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)linesearch, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchView - Prints useful information about the line search

  Logically Collective

  Input Parameters:
+ linesearch - line search context
- viewer     - the `PetscViewer` to display the line search information to

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `PetscViewer`, `SNESLineSearchCreate()`
@*/
PetscErrorCode SNESLineSearchView(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)linesearch), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(linesearch, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)linesearch, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(linesearch, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  maxlambda=%e, minlambda=%e\n", (double)linesearch->maxlambda, (double)linesearch->minlambda));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  tolerances: relative=%e, absolute=%e, lambda=%e\n", (double)linesearch->rtol, (double)linesearch->atol, (double)linesearch->ltol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum iterations=%" PetscInt_FMT "\n", linesearch->max_it));
    if (linesearch->ops->precheck) {
      if (linesearch->ops->precheck == SNESLineSearchPreCheckPicard) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  using precheck step to speed up Picard convergence\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  using user-defined precheck step\n"));
      }
    }
    if (linesearch->ops->postcheck) PetscCall(PetscViewerASCIIPrintf(viewer, "  using user-defined postcheck step\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetType - Gets the `SNESLinesearchType` of a `SNESLineSearch`

  Logically Collective

  Input Parameter:
. linesearch - the line search context

  Output Parameter:
. type - The type of line search, or `NULL` if not set

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetFromOptions()`, `SNESLineSearchSetType()`
@*/
PetscErrorCode SNESLineSearchGetType(SNESLineSearch linesearch, SNESLineSearchType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)linesearch)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetType - Sets the `SNESLinesearchType` of a `SNESLineSearch` object to indicate the line search algorithm that should be used by a given `SNES` solver

  Logically Collective

  Input Parameters:
+ linesearch - the line search context
- type       - The type of line search to be used, see `SNESLineSearchType`

  Options Database Key:
. -snes_linesearch_type <type> - basic (or equivalently none), bt, secant, cp, nleqerr, bisection, shell

  Level: intermediate

  Note:
  The `SNESLineSearch` object is generally obtained with `SNESGetLineSearch()`

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetFromOptions()`, `SNESLineSearchGetType()`,
          `SNESGetLineSearch()`
@*/
PetscErrorCode SNESLineSearchSetType(SNESLineSearch linesearch, SNESLineSearchType type)
{
  PetscBool match;
  PetscErrorCode (*r)(SNESLineSearch);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)linesearch, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(SNESLineSearchList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested Line Search type %s", type);
  /* Destroy the previous private line search context */
  PetscTryTypeMethod(linesearch, destroy);
  linesearch->ops->destroy = NULL;
  /* Reinitialize function pointers in SNESLineSearchOps structure */
  linesearch->ops->apply          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->destroy        = NULL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)linesearch, type));
  PetscCall((*r)(linesearch));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetSNES - Sets the `SNES` for the linesearch for function evaluation.

  Input Parameters:
+ linesearch - the line search context
- snes       - The `SNES` instance

  Level: developer

  Note:
  This happens automatically when the line search is obtained/created with
  `SNESGetLineSearch()`.  This routine is therefore mainly called within `SNES`
  implementations.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetSNES()`, `SNESLineSearchSetVecs()`
@*/
PetscErrorCode SNESLineSearchSetSNES(SNESLineSearch linesearch, SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 2);
  linesearch->snes = snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetSNES - Gets the `SNES` instance associated with the line search.

  Not Collective

  Input Parameter:
. linesearch - the line search context

  Output Parameter:
. snes - The `SNES` instance

  Level: developer

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESType`, `SNESLineSearchSetVecs()`
@*/
PetscErrorCode SNESLineSearchGetSNES(SNESLineSearch linesearch, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscAssertPointer(snes, 2);
  *snes = linesearch->snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetLambda - Gets the last line search `lambda` used

  Not Collective

  Input Parameter:
. linesearch - the line search context

  Output Parameter:
. lambda - The last `lambda` (scaling of the solution update) computed during `SNESLineSearchApply()`

  Level: advanced

  Note:
  This is useful in methods where the solver is ill-scaled and
  requires some adaptive notion of the difference in scale between the
  solution and the function.  For instance, `SNESQN` may be scaled by the
  line search `lambda` using the argument -snes_qn_scaling ls.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetLambda()`, `SNESLineSearchGetDamping()`, `SNESLineSearchApply()`
@*/
PetscErrorCode SNESLineSearchGetLambda(SNESLineSearch linesearch, PetscReal *lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscAssertPointer(lambda, 2);
  *lambda = linesearch->lambda;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetLambda - Sets the line search `lambda` (scaling of the solution update)

  Input Parameters:
+ linesearch - line search context
- lambda     - The `lambda` to use

  Level: advanced

  Note:
  This routine is typically used within implementations of `SNESLineSearchApply()`
  to set the final `lambda`.  This routine (and `SNESLineSearchGetLambda()`) were
  added to facilitate Quasi-Newton methods that use the previous `lambda`
  as an inner scaling parameter.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetLambda()`
@*/
PetscErrorCode SNESLineSearchSetLambda(SNESLineSearch linesearch, PetscReal lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  linesearch->lambda = lambda;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetTolerances - Gets the tolerances for the line search.

  Not Collective

  Input Parameter:
. linesearch - the line search context

  Output Parameters:
+ minlambda - The minimum `lambda` allowed
. maxlambda - The maximum `lambda` allowed
. rtol      - The relative tolerance for iterative line searches
. atol      - The absolute tolerance for iterative line searches
. ltol      - The change in `lambda` tolerance for iterative line searches
- max_it    - The maximum number of iterations of the line search

  Level: intermediate

  Note:
  Different line searches may implement these parameters slightly differently as
  the type requires.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetTolerances()`
@*/
PetscErrorCode SNESLineSearchGetTolerances(SNESLineSearch linesearch, PetscReal *minlambda, PetscReal *maxlambda, PetscReal *rtol, PetscReal *atol, PetscReal *ltol, PetscInt *max_it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (minlambda) {
    PetscAssertPointer(minlambda, 2);
    *minlambda = linesearch->minlambda;
  }
  if (maxlambda) {
    PetscAssertPointer(maxlambda, 3);
    *maxlambda = linesearch->maxlambda;
  }
  if (rtol) {
    PetscAssertPointer(rtol, 4);
    *rtol = linesearch->rtol;
  }
  if (atol) {
    PetscAssertPointer(atol, 5);
    *atol = linesearch->atol;
  }
  if (ltol) {
    PetscAssertPointer(ltol, 6);
    *ltol = linesearch->ltol;
  }
  if (max_it) {
    PetscAssertPointer(max_it, 7);
    *max_it = linesearch->max_it;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetTolerances -  Sets the tolerances for the linesearch.

  Collective

  Input Parameters:
+ linesearch - the line search context
. minlambda  - The minimum `lambda` allowed
. maxlambda  - The maximum `lambda` allowed
. rtol       - The relative tolerance for iterative line searches
. atol       - The absolute tolerance for iterative line searches
. ltol       - The change in `lambda` tolerance for iterative line searches
- max_it     - The maximum number of iterations of the line search

  Options Database Keys:
+ -snes_linesearch_minlambda - The minimum `lambda` allowed
. -snes_linesearch_maxlambda - The maximum `lambda` allowed
. -snes_linesearch_rtol      - Relative tolerance for iterative line searches
. -snes_linesearch_atol      - Absolute tolerance for iterative line searches
. -snes_linesearch_ltol      - Change in `lambda` tolerance for iterative line searches
- -snes_linesearch_max_it    - The number of iterations for iterative line searches

  Level: intermediate

  Note:
  The user may choose to not set any of the tolerances using `PETSC_DEFAULT` in place of an argument.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetTolerances()`
@*/
PetscErrorCode SNESLineSearchSetTolerances(SNESLineSearch linesearch, PetscReal minlambda, PetscReal maxlambda, PetscReal rtol, PetscReal atol, PetscReal ltol, PetscInt max_it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscValidLogicalCollectiveReal(linesearch, minlambda, 2);
  PetscValidLogicalCollectiveReal(linesearch, maxlambda, 3);
  PetscValidLogicalCollectiveReal(linesearch, rtol, 4);
  PetscValidLogicalCollectiveReal(linesearch, atol, 5);
  PetscValidLogicalCollectiveReal(linesearch, ltol, 6);
  PetscValidLogicalCollectiveInt(linesearch, max_it, 7);

  if (minlambda != (PetscReal)PETSC_DEFAULT) {
    PetscCheck(minlambda >= 0.0, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_OUTOFRANGE, "Minimum lambda %14.12e must be non-negative", (double)minlambda);
    PetscCheck(minlambda < maxlambda, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_OUTOFRANGE, "Minimum lambda %14.12e must be smaller than maximum lambda %14.12e", (double)minlambda, (double)maxlambda);
    linesearch->minlambda = minlambda;
  }

  if (maxlambda != (PetscReal)PETSC_DEFAULT) {
    PetscCheck(maxlambda > 0.0, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_OUTOFRANGE, "Maximum lambda %14.12e must be positive", (double)maxlambda);
    linesearch->maxlambda = maxlambda;
  }

  if (rtol != (PetscReal)PETSC_DEFAULT) {
    PetscCheck(rtol >= 0.0 && rtol < 1.0, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_OUTOFRANGE, "Relative tolerance %14.12e must be non-negative and less than 1.0", (double)rtol);
    linesearch->rtol = rtol;
  }

  if (atol != (PetscReal)PETSC_DEFAULT) {
    PetscCheck(atol >= 0.0, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_OUTOFRANGE, "Absolute tolerance %14.12e must be non-negative", (double)atol);
    linesearch->atol = atol;
  }

  if (ltol != (PetscReal)PETSC_DEFAULT) {
    PetscCheck(ltol >= 0.0, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_OUTOFRANGE, "Lambda tolerance %14.12e must be non-negative", (double)ltol);
    linesearch->ltol = ltol;
  }

  if (max_it != PETSC_DEFAULT) {
    PetscCheck(max_it >= 0, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_ARG_OUTOFRANGE, "Maximum number of iterations %" PetscInt_FMT " must be non-negative", max_it);
    linesearch->max_it = max_it;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetDamping - Gets the line search damping parameter.

  Input Parameter:
. linesearch - the line search context

  Output Parameter:
. damping - The damping parameter

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESLineSearchGetStepTolerance()`, `SNESQN`
@*/
PetscErrorCode SNESLineSearchGetDamping(SNESLineSearch linesearch, PetscReal *damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscAssertPointer(damping, 2);
  *damping = linesearch->damping;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetDamping - Sets the line search damping parameter.

  Input Parameters:
+ linesearch - the line search context
- damping    - The damping parameter

  Options Database Key:
. -snes_linesearch_damping <damping> - the damping value

  Level: intermediate

  Note:
  The `SNESLINESEARCHNONE` line search merely takes the update step scaled by the damping parameter.
  The use of the damping parameter in the `SNESLINESEARCHSECANT` and `SNESLINESEARCHCP` line searches is much more subtle;
  it is used as a starting point for the secant method. Depending on the choice for `maxlambda`,
  the eventual `lambda` may be greater than the damping parameter however.
  For `SNESLINESEARCHBISECTION` and `SNESLINESEARCHBT` the damping is instead used as the initial guess,
  below which the line search will not go. Hence, it is the maximum possible value for `lambda`.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetDamping()`
@*/
PetscErrorCode SNESLineSearchSetDamping(SNESLineSearch linesearch, PetscReal damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  linesearch->damping = damping;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetOrder - Gets the line search approximation order.

  Input Parameter:
. linesearch - the line search context

  Output Parameter:
. order - The order

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetOrder()`
@*/
PetscErrorCode SNESLineSearchGetOrder(SNESLineSearch linesearch, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscAssertPointer(order, 2);
  *order = linesearch->order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetOrder - Sets the maximum order of the polynomial fit used in the line search

  Input Parameters:
+ linesearch - the line search context
- order      - The order

  Level: intermediate

  Values for `order`\:
+  1 or `SNES_LINESEARCH_ORDER_LINEAR` - linear order
.  2 or `SNES_LINESEARCH_ORDER_QUADRATIC` - quadratic order
-  3 or `SNES_LINESEARCH_ORDER_CUBIC` - cubic order

  Options Database Key:
. -snes_linesearch_order <order> - 1, 2, 3.  Most types only support certain orders (`SNESLINESEARCHBT` supports 2 or 3)

  Note:
  These orders are supported by `SNESLINESEARCHBT` and `SNESLINESEARCHCP`

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetOrder()`, `SNESLineSearchSetDamping()`
@*/
PetscErrorCode SNESLineSearchSetOrder(SNESLineSearch linesearch, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  linesearch->order = order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetNorms - Gets the norms for the current solution `X`, the current update `Y`, and the current function value `F`.

  Not Collective

  Input Parameter:
. linesearch - the line search context

  Output Parameters:
+ xnorm - The norm of the current solution
. fnorm - The norm of the current function, this is the `norm(function(X))` where `X` is the current solution.
- ynorm - The norm of the current update (after scaling by the linesearch computed `lambda`)

  Level: developer

  Notes:
  Some values may not be up-to-date at particular points in the code.

  This, in combination with `SNESLineSearchSetNorms()`, allow the line search and the `SNESSolve_XXX()` to share
  computed values.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetNorms()`, `SNESLineSearchGetVecs()`
@*/
PetscErrorCode SNESLineSearchGetNorms(SNESLineSearch linesearch, PetscReal *xnorm, PetscReal *fnorm, PetscReal *ynorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (xnorm) *xnorm = linesearch->xnorm;
  if (fnorm) *fnorm = linesearch->fnorm;
  if (ynorm) *ynorm = linesearch->ynorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetNorms - Sets the computed norms for the current solution `X`, the current update `Y`, and the current function value `F`.

  Collective

  Input Parameters:
+ linesearch - the line search context
. xnorm      - The norm of the current solution
. fnorm      - The norm of the current function, this is the `norm(function(X))` where `X` is the current solution
- ynorm      - The norm of the current update (after scaling by the linesearch computed `lambda`)

  Level: developer

  Note:
  This is called by the line search routines to store the values they have just computed

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetNorms()`, `SNESLineSearchSetVecs()`
@*/
PetscErrorCode SNESLineSearchSetNorms(SNESLineSearch linesearch, PetscReal xnorm, PetscReal fnorm, PetscReal ynorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  linesearch->xnorm = xnorm;
  linesearch->fnorm = fnorm;
  linesearch->ynorm = ynorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchComputeNorms - Explicitly computes the norms of the current solution `X`, the current update `Y`, and the current function value `F`.

  Input Parameter:
. linesearch - the line search context

  Options Database Key:
. -snes_linesearch_norms - turn norm computation on or off

  Level: intermediate

  Developer Note:
  The options database key is misnamed. It should be -snes_linesearch_compute_norms

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetNorms`, `SNESLineSearchSetNorms()`, `SNESLineSearchSetComputeNorms()`
@*/
PetscErrorCode SNESLineSearchComputeNorms(SNESLineSearch linesearch)
{
  SNES snes;

  PetscFunctionBegin;
  if (linesearch->norms) {
    if (linesearch->ops->vinorm) {
      PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
      PetscCall(VecNorm(linesearch->vec_sol, NORM_2, &linesearch->xnorm));
      PetscCall(VecNorm(linesearch->vec_update, NORM_2, &linesearch->ynorm));
      PetscCall((*linesearch->ops->vinorm)(snes, linesearch->vec_func, linesearch->vec_sol, &linesearch->fnorm));
    } else {
      PetscCall(VecNormBegin(linesearch->vec_func, NORM_2, &linesearch->fnorm));
      PetscCall(VecNormBegin(linesearch->vec_sol, NORM_2, &linesearch->xnorm));
      PetscCall(VecNormBegin(linesearch->vec_update, NORM_2, &linesearch->ynorm));
      PetscCall(VecNormEnd(linesearch->vec_func, NORM_2, &linesearch->fnorm));
      PetscCall(VecNormEnd(linesearch->vec_sol, NORM_2, &linesearch->xnorm));
      PetscCall(VecNormEnd(linesearch->vec_update, NORM_2, &linesearch->ynorm));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetComputeNorms - Turns on or off the computation of final norms in the line search.

  Input Parameters:
+ linesearch - the line search context
- flg        - indicates whether or not to compute norms

  Options Database Key:
. -snes_linesearch_norms <true> - Turns on/off computation of the norms for basic (none) `SNESLINESEARCHBASIC` line search

  Level: intermediate

  Note:
  This is most relevant to the `SNESLINESEARCHBASIC` (or equivalently `SNESLINESEARCHNONE`) line search type since most line searches have a stopping criteria involving the norm.

  Developer Note:
  The options database key is misnamed. It should be -snes_linesearch_compute_norms

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetNorms()`, `SNESLineSearchSetNorms()`, `SNESLineSearchComputeNorms()`, `SNESLINESEARCHBASIC`
@*/
PetscErrorCode SNESLineSearchSetComputeNorms(SNESLineSearch linesearch, PetscBool flg)
{
  PetscFunctionBegin;
  linesearch->norms = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetVecs - Gets the vectors from the `SNESLineSearch` context

  Not Collective but the vectors are parallel

  Input Parameter:
. linesearch - the line search context

  Output Parameters:
+ X - Solution vector
. F - Function vector
. Y - Search direction vector
. W - Solution work vector
- G - Function work vector

  Level: advanced

  Notes:
  At the beginning of a line search application, `X` should contain a
  solution and the vector `F` the function computed at `X`.  At the end of the
  line search application, `X` should contain the new solution, and `F` the
  function evaluated at the new solution.

  These vectors are owned by the `SNESLineSearch` and should not be destroyed by the caller

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetNorms()`, `SNESLineSearchSetVecs()`
@*/
PetscErrorCode SNESLineSearchGetVecs(SNESLineSearch linesearch, Vec *X, Vec *F, Vec *Y, Vec *W, Vec *G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (X) {
    PetscAssertPointer(X, 2);
    *X = linesearch->vec_sol;
  }
  if (F) {
    PetscAssertPointer(F, 3);
    *F = linesearch->vec_func;
  }
  if (Y) {
    PetscAssertPointer(Y, 4);
    *Y = linesearch->vec_update;
  }
  if (W) {
    PetscAssertPointer(W, 5);
    *W = linesearch->vec_sol_new;
  }
  if (G) {
    PetscAssertPointer(G, 6);
    *G = linesearch->vec_func_new;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetVecs - Sets the vectors on the `SNESLineSearch` context

  Logically Collective

  Input Parameters:
+ linesearch - the line search context
. X          - Solution vector
. F          - Function vector
. Y          - Search direction vector
. W          - Solution work vector
- G          - Function work vector

  Level: developer

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetNorms()`, `SNESLineSearchGetVecs()`
@*/
PetscErrorCode SNESLineSearchSetVecs(SNESLineSearch linesearch, Vec X, Vec F, Vec Y, Vec W, Vec G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (X) {
    PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
    linesearch->vec_sol = X;
  }
  if (F) {
    PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
    linesearch->vec_func = F;
  }
  if (Y) {
    PetscValidHeaderSpecific(Y, VEC_CLASSID, 4);
    linesearch->vec_update = Y;
  }
  if (W) {
    PetscValidHeaderSpecific(W, VEC_CLASSID, 5);
    linesearch->vec_sol_new = W;
  }
  if (G) {
    PetscValidHeaderSpecific(G, VEC_CLASSID, 6);
    linesearch->vec_func_new = G;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchAppendOptionsPrefix - Appends to the prefix used for searching for all
  `SNESLineSearch` options in the database.

  Logically Collective

  Input Parameters:
+ linesearch - the `SNESLineSearch` context
- prefix     - the prefix to prepend to all option names

  Level: advanced

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch()`, `SNESLineSearchSetFromOptions()`, `SNESGetOptionsPrefix()`
@*/
PetscErrorCode SNESLineSearchAppendOptionsPrefix(SNESLineSearch linesearch, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)linesearch, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetOptionsPrefix - Gets the prefix used for searching for all
  SNESLineSearch options in the database.

  Not Collective

  Input Parameter:
. linesearch - the `SNESLineSearch` context

  Output Parameter:
. prefix - pointer to the prefix string used

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESAppendOptionsPrefix()`
@*/
PetscErrorCode SNESLineSearchGetOptionsPrefix(SNESLineSearch linesearch, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)linesearch, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchSetWorkVecs - Sets work vectors for the line search.

  Input Parameters:
+ linesearch - the `SNESLineSearch` context
- nwork      - the number of work vectors

  Level: developer

  Developer Note:
  This is called from within the set up routines for each of the line search types `SNESLineSearchType`

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESSetWorkVecs()`
@*/
PetscErrorCode SNESLineSearchSetWorkVecs(SNESLineSearch linesearch, PetscInt nwork)
{
  PetscFunctionBegin;
  PetscCheck(linesearch->vec_sol, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "Cannot get linesearch work-vectors without setting a solution vec!");
  PetscCall(VecDuplicateVecs(linesearch->vec_sol, nwork, &linesearch->work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchGetReason - Gets the success/failure status of the last line search application

  Input Parameter:
. linesearch - the line search context

  Output Parameter:
. result - The success or failure status

  Level: developer

  Note:
  This is typically called after `SNESLineSearchApply()` in order to determine if the line search failed
  (and set into the `SNES` convergence accordingly).

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetReason()`, `SNESLineSearchReason`
@*/
PetscErrorCode SNESLineSearchGetReason(SNESLineSearch linesearch, SNESLineSearchReason *result)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscAssertPointer(result, 2);
  *result = linesearch->result;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchSetReason - Sets the success/failure status of the line search application

  Logically Collective; No Fortran Support

  Input Parameters:
+ linesearch - the line search context
- result     - The success or failure status

  Level: developer

  Note:
  This is typically called in a `SNESLineSearchType` implementation of `SNESLineSearchApply()` or a `SNESLINESEARCHSHELL` implementation to set
  the success or failure of the line search method.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchReason`, `SNESLineSearchGetSResult()`
@*/
PetscErrorCode SNESLineSearchSetReason(SNESLineSearch linesearch, SNESLineSearchReason result)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  linesearch->result = result;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma disable: -fdoc-param-list-func-parameter-documentation
/*@C
  SNESLineSearchSetVIFunctions - Sets VI-specific functions for line search computation.

  Logically Collective

  Input Parameters:
+ linesearch   - the linesearch object
. projectfunc  - function for projecting the function to the bounds, see `SNESLineSearchVIProjectFn` for calling sequence
. normfunc     - function for computing the norm of an active set, see `SNESLineSearchVINormFn` for calling sequence
- dirderivfunc - function for computing the directional derivative of an active set, see `SNESLineSearchVIDirDerivFn` for calling sequence

  Level: advanced

  Notes:
  The VI solvers require projection of the solution to the feasible set.  `projectfunc` should implement this.

  The VI solvers require special evaluation of the function norm such that the norm is only calculated
  on the inactive set.  This should be implemented by `normfunc`.

  The VI solvers further require special evaluation of the directional derivative (when assuming that there exists some $G(x)$
  for which the `SNESFunctionFn` $F(x) = grad G(x)$) such that it is only calculated on the inactive set.
  This should be implemented by `dirderivfunc`.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchGetVIFunctions()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchSetPreCheck()`,
          `SNESLineSearchVIProjectFn`, `SNESLineSearchVINormFn`, `SNESLineSearchVIDirDerivFn`
@*/
PetscErrorCode SNESLineSearchSetVIFunctions(SNESLineSearch linesearch, SNESLineSearchVIProjectFn *projectfunc, SNESLineSearchVINormFn *normfunc, SNESLineSearchVIDirDerivFn *dirderivfunc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (projectfunc) linesearch->ops->viproject = projectfunc;
  if (normfunc) linesearch->ops->vinorm = normfunc;
  if (dirderivfunc) linesearch->ops->vidirderiv = dirderivfunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchGetVIFunctions - Sets VI-specific functions for line search computation.

  Not Collective

  Input Parameter:
. linesearch - the line search context, obtain with `SNESGetLineSearch()`

  Output Parameters:
+ projectfunc  - function for projecting the function to the bounds, see `SNESLineSearchVIProjectFn` for calling sequence
. normfunc     - function for computing the norm of an active set, see `SNESLineSearchVINormFn ` for calling sequence
- dirderivfunc - function for computing the directional derivative of an active set, see `SNESLineSearchVIDirDerivFn` for calling sequence

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchSetVIFunctions()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchGetPreCheck()`,
          `SNESLineSearchVIProjectFn`, `SNESLineSearchVINormFn`
@*/
PetscErrorCode SNESLineSearchGetVIFunctions(SNESLineSearch linesearch, SNESLineSearchVIProjectFn **projectfunc, SNESLineSearchVINormFn **normfunc, SNESLineSearchVIDirDerivFn **dirderivfunc)
{
  PetscFunctionBegin;
  if (projectfunc) *projectfunc = linesearch->ops->viproject;
  if (normfunc) *normfunc = linesearch->ops->vinorm;
  if (dirderivfunc) *dirderivfunc = linesearch->ops->vidirderiv;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchRegister - register a line search type `SNESLineSearchType`

  Logically Collective, No Fortran Support

  Input Parameters:
+ sname    - name of the `SNESLineSearchType()`
- function - the creation function for that type

  Calling sequence of `function`:
. ls - the line search context

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchSetType()`
@*/
PetscErrorCode SNESLineSearchRegister(const char sname[], PetscErrorCode (*function)(SNESLineSearch ls))
{
  PetscFunctionBegin;
  PetscCall(SNESInitializePackage());
  PetscCall(PetscFunctionListAdd(&SNESLineSearchList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
