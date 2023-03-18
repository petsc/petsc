#include <petsctaolinesearch.h> /*I "petsctaolinesearch.h" I*/
#include <petsc/private/taolinesearchimpl.h>

PetscFunctionList TaoLineSearchList = NULL;

PetscClassId TAOLINESEARCH_CLASSID = 0;

PetscLogEvent TAOLINESEARCH_Apply;
PetscLogEvent TAOLINESEARCH_Eval;

/*@C
   TaoLineSearchViewFromOptions - View a `TaoLineSearch` object based on values in the options database

   Collective

   Input Parameters:
+  A - the `Tao` context
.  obj - Optional object
-  name - command line option

   Level: intermediate

   Note:
   See `PetscObjectViewFromOptions()` for available viewer options

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchView()`, `PetscObjectViewFromOptions()`, `TaoLineSearchCreate()`
@*/
PetscErrorCode TaoLineSearchViewFromOptions(TaoLineSearch A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, TAOLINESEARCH_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchView - Prints information about the `TaoLineSearch`

  Collective

  InputParameters:
+ ls - the `TaoLineSearch` context
- viewer - visualization context

  Options Database Key:
. -tao_ls_view - Calls `TaoLineSearchView()` at the end of each line search

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `PetscViewerASCIIOpen()`, `TaoLineSearchViewFromOptions()`
@*/
PetscErrorCode TaoLineSearchView(TaoLineSearch ls, PetscViewer viewer)
{
  PetscBool         isascii, isstring;
  TaoLineSearchType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)ls)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ls, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ls, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(ls, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "maximum function evaluations=%" PetscInt_FMT "\n", ls->max_funcs));
    PetscCall(PetscViewerASCIIPrintf(viewer, "tolerances: ftol=%g, rtol=%g, gtol=%g\n", (double)ls->ftol, (double)ls->rtol, (double)ls->gtol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function evaluations=%" PetscInt_FMT "\n", ls->nfeval));
    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of gradient evaluations=%" PetscInt_FMT "\n", ls->ngeval));
    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function/gradient evaluations=%" PetscInt_FMT "\n", ls->nfgeval));

    if (ls->bounded) PetscCall(PetscViewerASCIIPrintf(viewer, "using variable bounds\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Termination reason: %d\n", (int)ls->reason));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(TaoLineSearchGetType(ls, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " %-3.3s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchCreate - Creates a `TaoLineSearch` object.  Algorithms in `Tao` that use
  line-searches will automatically create one so this all is rarely needed

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newls - the new `TaoLineSearch` context

   Options Database Key:
.   -tao_ls_type - select which method `Tao` should use

   Level: developer

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchType`, `TaoLineSearchSetType()`, `TaoLineSearchApply()`, `TaoLineSearchDestroy()`
@*/
PetscErrorCode TaoLineSearchCreate(MPI_Comm comm, TaoLineSearch *newls)
{
  TaoLineSearch ls;

  PetscFunctionBegin;
  PetscValidPointer(newls, 2);
  PetscCall(TaoLineSearchInitializePackage());

  PetscCall(PetscHeaderCreate(ls, TAOLINESEARCH_CLASSID, "TaoLineSearch", "Linesearch", "Tao", comm, TaoLineSearchDestroy, TaoLineSearchView));
  ls->max_funcs = 30;
  ls->ftol      = 0.0001;
  ls->gtol      = 0.9;
#if defined(PETSC_USE_REAL_SINGLE)
  ls->rtol = 1.0e-5;
#else
  ls->rtol = 1.0e-10;
#endif
  ls->stepmin  = 1.0e-20;
  ls->stepmax  = 1.0e+20;
  ls->step     = 1.0;
  ls->initstep = 1.0;
  *newls       = ls;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchSetUp - Sets up the internal data structures for the later use
  of a `TaoLineSearch`

  Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Level: developer

  Note:
  The user will not need to explicitly call `TaoLineSearchSetUp()`, as it will
  automatically be called in `TaoLineSearchSolve()`.  However, if the user
  desires to call it explicitly, it should come after `TaoLineSearchCreate()`
  but before `TaoLineSearchApply()`.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchCreate()`, `TaoLineSearchApply()`
@*/
PetscErrorCode TaoLineSearchSetUp(TaoLineSearch ls)
{
  const char *default_type = TAOLINESEARCHMT;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  if (ls->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (!((PetscObject)ls)->type_name) PetscCall(TaoLineSearchSetType(ls, default_type));
  PetscTryTypeMethod(ls, setup);
  if (ls->usetaoroutines) {
    PetscCall(TaoIsObjectiveDefined(ls->tao, &flg));
    ls->hasobjective = flg;
    PetscCall(TaoIsGradientDefined(ls->tao, &flg));
    ls->hasgradient = flg;
    PetscCall(TaoIsObjectiveAndGradientDefined(ls->tao, &flg));
    ls->hasobjectiveandgradient = flg;
  } else {
    if (ls->ops->computeobjective) {
      ls->hasobjective = PETSC_TRUE;
    } else {
      ls->hasobjective = PETSC_FALSE;
    }
    if (ls->ops->computegradient) {
      ls->hasgradient = PETSC_TRUE;
    } else {
      ls->hasgradient = PETSC_FALSE;
    }
    if (ls->ops->computeobjectiveandgradient) {
      ls->hasobjectiveandgradient = PETSC_TRUE;
    } else {
      ls->hasobjectiveandgradient = PETSC_FALSE;
    }
  }
  ls->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchReset - Some line searches may carry state information
  from one `TaoLineSearchApply()` to the next.  This function resets this
  state information.

  Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Level: developer

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchCreate()`, `TaoLineSearchApply()`
@*/
PetscErrorCode TaoLineSearchReset(TaoLineSearch ls)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscTryTypeMethod(ls, reset);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchDestroy - Destroys the `TaoLineSearch` context that was created with
  `TaoLineSearchCreate()`

  Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Level: developer

.seealse: `TaoLineSearchCreate()`, `TaoLineSearchSolve()`
@*/
PetscErrorCode TaoLineSearchDestroy(TaoLineSearch *ls)
{
  PetscFunctionBegin;
  if (!*ls) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*ls, TAOLINESEARCH_CLASSID, 1);
  if (--((PetscObject)*ls)->refct > 0) {
    *ls = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecDestroy(&(*ls)->stepdirection));
  PetscCall(VecDestroy(&(*ls)->start_x));
  PetscCall(VecDestroy(&(*ls)->upper));
  PetscCall(VecDestroy(&(*ls)->lower));
  if ((*ls)->ops->destroy) PetscCall((*(*ls)->ops->destroy)(*ls));
  if ((*ls)->usemonitor) PetscCall(PetscViewerDestroy(&(*ls)->viewer));
  PetscCall(PetscHeaderDestroy(ls));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchApply - Performs a line-search in a given step direction.
  Criteria for acceptable step length depends on the line-search algorithm chosen

  Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
- s - search direction

  Output Parameters:
+ x - On input the current solution, on output `x` contains the new solution determined by the line search
. f - On input the objective function value at current solution, on output contains the objective function value at new solution
. g - On input the gradient evaluated at `x`, on output contains the gradient at new solution
. steplength - scalar multiplier of s used ( x = x0 + steplength * x)
- reason - `TaoLineSearchConvergedReason` reason why the line-search stopped

  Level: advanced

  Notes:
  The algorithm developer must set up the `TaoLineSearch` with calls to
  `TaoLineSearchSetObjectiveRoutine()` and `TaoLineSearchSetGradientRoutine()`,
  `TaoLineSearchSetObjectiveAndGradientRoutine()`, or `TaoLineSearchUseTaoRoutines()`.
  The latter is done automatically by default and thus requires no user input.

  You may or may not need to follow this with a call to
  `TaoAddLineSearchCounts()`, depending on whether you want these
  evaluations to count toward the total function/gradient evaluations.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearchConvergedReason`, `TaoLineSearch`, `TaoLineSearchCreate()`, `TaoLineSearchSetType()`,
          `TaoLineSearchSetInitialStepLength()`, `TaoAddLineSearchCounts()`
@*/
PetscErrorCode TaoLineSearchApply(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s, PetscReal *steplength, TaoLineSearchConvergedReason *reason)
{
  PetscInt low1, low2, low3, high1, high2, high3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidRealPointer(f, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(s, VEC_CLASSID, 5);
  PetscValidPointer(reason, 7);
  PetscCheckSameComm(ls, 1, x, 2);
  PetscCheckSameTypeAndComm(x, 2, g, 4);
  PetscCheckSameTypeAndComm(x, 2, s, 5);
  PetscCall(VecGetOwnershipRange(x, &low1, &high1));
  PetscCall(VecGetOwnershipRange(g, &low2, &high2));
  PetscCall(VecGetOwnershipRange(s, &low3, &high3));
  PetscCheck(low1 == low2 && low1 == low3 && high1 == high2 && high1 == high3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible vector local lengths");

  *reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscCall(PetscObjectReference((PetscObject)s));
  PetscCall(VecDestroy(&ls->stepdirection));
  ls->stepdirection = s;

  PetscCall(TaoLineSearchSetUp(ls));
  ls->nfeval  = 0;
  ls->ngeval  = 0;
  ls->nfgeval = 0;
  /* Check parameter values */
  if (ls->ftol < 0.0) {
    PetscCall(PetscInfo(ls, "Bad Line Search Parameter: ftol (%g) < 0\n", (double)ls->ftol));
    *reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  if (ls->rtol < 0.0) {
    PetscCall(PetscInfo(ls, "Bad Line Search Parameter: rtol (%g) < 0\n", (double)ls->rtol));
    *reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  if (ls->gtol < 0.0) {
    PetscCall(PetscInfo(ls, "Bad Line Search Parameter: gtol (%g) < 0\n", (double)ls->gtol));
    *reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  if (ls->stepmin < 0.0) {
    PetscCall(PetscInfo(ls, "Bad Line Search Parameter: stepmin (%g) < 0\n", (double)ls->stepmin));
    *reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  if (ls->stepmax < ls->stepmin) {
    PetscCall(PetscInfo(ls, "Bad Line Search Parameter: stepmin (%g) > stepmax (%g)\n", (double)ls->stepmin, (double)ls->stepmax));
    *reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  if (ls->max_funcs < 0) {
    PetscCall(PetscInfo(ls, "Bad Line Search Parameter: max_funcs (%" PetscInt_FMT ") < 0\n", ls->max_funcs));
    *reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "Initial Line Search Function Value is Inf or Nan (%g)\n", (double)*f));
    *reason = TAOLINESEARCH_FAILED_INFORNAN;
  }

  PetscCall(PetscObjectReference((PetscObject)x));
  PetscCall(VecDestroy(&ls->start_x));
  ls->start_x = x;

  PetscCall(PetscLogEventBegin(TAOLINESEARCH_Apply, ls, 0, 0, 0));
  PetscUseTypeMethod(ls, apply, x, f, g, s);
  PetscCall(PetscLogEventEnd(TAOLINESEARCH_Apply, ls, 0, 0, 0));
  *reason   = ls->reason;
  ls->new_f = *f;

  if (steplength) *steplength = ls->step;

  PetscCall(TaoLineSearchViewFromOptions(ls, NULL, "-tao_ls_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoLineSearchSetType - Sets the algorithm used in a line search

   Collective

   Input Parameters:
+  ls - the `TaoLineSearch` context
-  type - the `TaoLineSearchType` selection

  Options Database Key:
.  -tao_ls_type <type> - select which method Tao should use at runtime

  Level: beginner

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchType`, `TaoLineSearchCreate()`, `TaoLineSearchGetType()`,
          `TaoLineSearchApply()`
@*/
PetscErrorCode TaoLineSearchSetType(TaoLineSearch ls, TaoLineSearchType type)
{
  PetscErrorCode (*r)(TaoLineSearch);
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidCharPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)ls, type, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoLineSearchList, type, (void (**)(void)) & r));
  PetscCheck(r, PetscObjectComm((PetscObject)ls), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested TaoLineSearch type %s", type);
  PetscTryTypeMethod(ls, destroy);
  ls->max_funcs = 30;
  ls->ftol      = 0.0001;
  ls->gtol      = 0.9;
#if defined(PETSC_USE_REAL_SINGLE)
  ls->rtol = 1.0e-5;
#else
  ls->rtol = 1.0e-10;
#endif
  ls->stepmin = 1.0e-20;
  ls->stepmax = 1.0e+20;

  ls->nfeval              = 0;
  ls->ngeval              = 0;
  ls->nfgeval             = 0;
  ls->ops->setup          = NULL;
  ls->ops->apply          = NULL;
  ls->ops->view           = NULL;
  ls->ops->setfromoptions = NULL;
  ls->ops->destroy        = NULL;
  ls->setupcalled         = PETSC_FALSE;
  PetscCall((*r)(ls));
  PetscCall(PetscObjectChangeTypeName((PetscObject)ls, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchMonitor - Monitor the line search steps. This routine will otuput the
  iteration number, step length, and function value before calling the implementation
  specific monitor.

   Input Parameters:
+  ls - the `TaoLineSearch` context
.  its - the current iterate number (>=0)
.  f - the current objective function value
-  step - the step length

   Options Database Key:
.  -tao_ls_monitor - Use the default monitor, which prints statistics to standard output

   Level: developer

.seealso: `TaoLineSearch`
@*/
PetscErrorCode TaoLineSearchMonitor(TaoLineSearch ls, PetscInt its, PetscReal f, PetscReal step)
{
  PetscInt tabs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  if (ls->usemonitor) {
    PetscCall(PetscViewerASCIIGetTab(ls->viewer, &tabs));
    PetscCall(PetscViewerASCIISetTab(ls->viewer, ((PetscObject)ls)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(ls->viewer, "%3" PetscInt_FMT " LS", its));
    PetscCall(PetscViewerASCIIPrintf(ls->viewer, "  Function value: %g,", (double)f));
    PetscCall(PetscViewerASCIIPrintf(ls->viewer, "  Step length: %g\n", (double)step));
    if (ls->ops->monitor && its > 0) {
      PetscCall(PetscViewerASCIISetTab(ls->viewer, ((PetscObject)ls)->tablevel + 3));
      PetscUseTypeMethod(ls, monitor);
    }
    PetscCall(PetscViewerASCIISetTab(ls->viewer, tabs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchSetFromOptions - Sets various `TaoLineSearch` parameters from user
  options.

  Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Options Database Keys:
+ -tao_ls_type <type> - The algorithm that `TaoLineSearch` uses (more-thuente, gpcg, unit)
. -tao_ls_ftol <tol> - tolerance for sufficient decrease
. -tao_ls_gtol <tol> - tolerance for curvature condition
. -tao_ls_rtol <tol> - relative tolerance for acceptable step
. -tao_ls_stepinit <step> - initial steplength allowed
. -tao_ls_stepmin <step> - minimum steplength allowed
. -tao_ls_stepmax <step> - maximum steplength allowed
. -tao_ls_max_funcs <n> - maximum number of function evaluations allowed
- -tao_ls_view - display line-search results to standard output

  Level: beginner

.seealso: `TaoLineSearch`
@*/
PetscErrorCode TaoLineSearchSetFromOptions(TaoLineSearch ls)
{
  const char *default_type = TAOLINESEARCHMT;
  char        type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer monviewer;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)ls);
  if (((PetscObject)ls)->type_name) default_type = ((PetscObject)ls)->type_name;
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-tao_ls_type", "Tao Line Search type", "TaoLineSearchSetType", TaoLineSearchList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(TaoLineSearchSetType(ls, type));
  } else if (!((PetscObject)ls)->type_name) {
    PetscCall(TaoLineSearchSetType(ls, default_type));
  }

  PetscCall(PetscOptionsInt("-tao_ls_max_funcs", "max function evals in line search", "", ls->max_funcs, &ls->max_funcs, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_ftol", "tol for sufficient decrease", "", ls->ftol, &ls->ftol, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_gtol", "tol for curvature condition", "", ls->gtol, &ls->gtol, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_rtol", "relative tol for acceptable step", "", ls->rtol, &ls->rtol, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_stepmin", "lower bound for step", "", ls->stepmin, &ls->stepmin, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_stepmax", "upper bound for step", "", ls->stepmax, &ls->stepmax, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_stepinit", "initial step", "", ls->initstep, &ls->initstep, NULL));
  PetscCall(PetscOptionsString("-tao_ls_monitor", "enable the basic monitor", "TaoLineSearchSetMonitor", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)ls), monfilename, &monviewer));
    ls->viewer     = monviewer;
    ls->usemonitor = PETSC_TRUE;
  }
  PetscTryTypeMethod(ls, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchGetType - Gets the current line search algorithm

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameter:
. type - the line search algorithm in effect

  Level: developer

.seealso: `TaoLineSearch`
@*/
PetscErrorCode TaoLineSearchGetType(TaoLineSearch ls, TaoLineSearchType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)ls)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchGetNumberFunctionEvaluations - Gets the number of function and gradient evaluation
  routines used by the line search in last application (not cumulative).

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameters:
+ nfeval   - number of function evaluations
. ngeval   - number of gradient evaluations
- nfgeval  - number of function/gradient evaluations

  Level: intermediate

  Note:
  If the line search is using the `Tao` objective and gradient
  routines directly (see `TaoLineSearchUseTaoRoutines()`), then the `Tao`
  is already counting the number of evaluations.

.seealso: `TaoLineSearch`
@*/
PetscErrorCode TaoLineSearchGetNumberFunctionEvaluations(TaoLineSearch ls, PetscInt *nfeval, PetscInt *ngeval, PetscInt *nfgeval)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  *nfeval  = ls->nfeval;
  *ngeval  = ls->ngeval;
  *nfgeval = ls->nfgeval;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchIsUsingTaoRoutines - Checks whether the line search is using
  the standard `Tao` evaluation routines.

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameter:
. flg - `PETSC_TRUE` if the line search is using `Tao` evaluation routines,
        otherwise `PETSC_FALSE`

  Level: developer

.seealso: `TaoLineSearch`
@*/
PetscErrorCode TaoLineSearchIsUsingTaoRoutines(TaoLineSearch ls, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  *flg = ls->usetaoroutines;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchSetObjectiveRoutine - Sets the function evaluation routine for the line search

  Logically Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
. func - the objective function evaluation routine
- ctx - the (optional) user-defined context for private data

  Calling sequence of `func`:
$ PetscErrorCode func(TaoLinesearch ls, Vec x, PetscReal *f, void *ctx);
+ ls - the line search context
. x - input vector
. f - function value
- ctx (optional) user-defined context

  Level: advanced

  Notes:
  Use this routine only if you want the line search objective
  evaluation routine to be different from the `Tao`'s objective
  evaluation routine. If you use this routine you must also set
  the line search gradient and/or function/gradient routine.

  Some algorithms (lcl, gpcg) set their own objective routine for the
  line search, application programmers should be wary of overriding the
  default objective routine.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchCreate()`, `TaoLineSearchSetGradientRoutine()`, `TaoLineSearchSetObjectiveAndGradientRoutine()`, `TaoLineSearchUseTaoRoutines()`
@*/
PetscErrorCode TaoLineSearchSetObjectiveRoutine(TaoLineSearch ls, PetscErrorCode (*func)(TaoLineSearch ls, Vec x, PetscReal *, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);

  ls->ops->computeobjective = func;
  if (ctx) ls->userctx_func = ctx;
  ls->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchSetGradientRoutine - Sets the gradient evaluation routine for the line search

  Logically Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
. func - the gradient evaluation routine
- ctx - the (optional) user-defined context for private data

  Calling sequence of `func`:
$ PetscErrorCode func(TaoLinesearch ls, Vec x, Vec g, void *ctx);
+ ls - the linesearch object
. x - input vector
. g - gradient vector
- ctx (optional) user-defined context

  Level: beginner

  Note:
  Use this routine only if you want the line search gradient
  evaluation routine to be different from the `Tao`'s gradient
  evaluation routine. If you use this routine you must also set
  the line search function and/or function/gradient routine.

  Some algorithms (lcl, gpcg) set their own gradient routine for the
  line search, application programmers should be wary of overriding the
  default gradient routine.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchCreate()`, `TaoLineSearchSetObjectiveRoutine()`, `TaoLineSearchSetObjectiveAndGradientRoutine()`, `TaoLineSearchUseTaoRoutines()`
@*/
PetscErrorCode TaoLineSearchSetGradientRoutine(TaoLineSearch ls, PetscErrorCode (*func)(TaoLineSearch ls, Vec x, Vec g, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  ls->ops->computegradient = func;
  if (ctx) ls->userctx_grad = ctx;
  ls->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchSetObjectiveAndGradientRoutine - Sets the objective/gradient evaluation routine for the line search

  Logically Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
. func - the objective and gradient evaluation routine
- ctx - the (optional) user-defined context for private data

  Calling sequence of `func`:
$ PetscErrorCode func(TaoLinesearch ls, Vec x, PetscReal *f, Vec g, void *ctx);
+ ls - the linesearch object
. x - input vector
. f - function value
. g - gradient vector
- ctx (optional) user-defined context

  Level: beginner

  Note:
  Use this routine only if you want the line search objective and gradient
  evaluation routines to be different from the `Tao`'s objective
  and gradient evaluation routines.

  Some algorithms (lcl, gpcg) set their own objective routine for the
  line search, application programmers should be wary of overriding the
  default objective routine.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchCreate()`, `TaoLineSearchSetObjectiveRoutine()`, `TaoLineSearchSetGradientRoutine()`, `TaoLineSearchUseTaoRoutines()`
@*/
PetscErrorCode TaoLineSearchSetObjectiveAndGradientRoutine(TaoLineSearch ls, PetscErrorCode (*func)(TaoLineSearch ls, Vec x, PetscReal *, Vec g, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  ls->ops->computeobjectiveandgradient = func;
  if (ctx) ls->userctx_funcgrad = ctx;
  ls->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchSetObjectiveAndGTSRoutine - Sets the objective and
  (gradient'*stepdirection) evaluation routine for the line search.
  Sometimes it is more efficient to compute the inner product of the gradient
  and the step direction than it is to compute the gradient, and this is all
  the line search typically needs of the gradient.

  Logically Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
. func - the objective and gradient evaluation routine
- ctx - the (optional) user-defined context for private data

  Calling sequence of `func`:
$ PetscErrorCode func(TaoLinesearch ls, Vec x, PetscReal *f, PetscReal *gts, void *ctx);
+ ls - the linesearch context
. x - input vector
. s - step direction
. f - function value
. gts - inner product of gradient and step direction vectors
- ctx (optional) user-defined context

  Level: advanced

  Notes:
  The gradient will still need to be computed at the end of the line
  search, so you will still need to set a line search gradient evaluation
  routine

  Bounded line searches (those used in bounded optimization algorithms)
  don't use g's directly, but rather (g'x - g'x0)/steplength.  You can get the
  x0 and steplength with `TaoLineSearchGetStartingVector()` and `TaoLineSearchGetStepLength()`

  Some algorithms (lcl, gpcg) set their own objective routine for the
  line search, application programmers should be wary of overriding the
  default objective routine.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchCreate()`, `TaoLineSearchSetObjective()`, `TaoLineSearchSetGradient()`, `TaoLineSearchUseTaoRoutines()`
@*/
PetscErrorCode TaoLineSearchSetObjectiveAndGTSRoutine(TaoLineSearch ls, PetscErrorCode (*func)(TaoLineSearch ls, Vec x, Vec s, PetscReal *, PetscReal *, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  ls->ops->computeobjectiveandgts = func;
  if (ctx) ls->userctx_funcgts = ctx;
  ls->usegts         = PETSC_TRUE;
  ls->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoLineSearchUseTaoRoutines - Informs the `TaoLineSearch` to use the
  objective and gradient evaluation routines from the given `Tao` object. The default.

  Logically Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
- ts - the `Tao` context with defined objective/gradient evaluation routines

  Level: developer

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchCreate()`
@*/
PetscErrorCode TaoLineSearchUseTaoRoutines(TaoLineSearch ls, Tao ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TAO_CLASSID, 2);
  ls->tao            = ts;
  ls->usetaoroutines = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchComputeObjective - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
- x - input vector

  Output Parameter:
. f - Objective value at `x`

  Level: developer

  Note:
  `TaoLineSearchComputeObjective()` is typically used within line searches
  so most users would not generally call this routine themselves.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchComputeGradient()`, `TaoLineSearchComputeObjectiveAndGradient()`, `TaoLineSearchSetObjectiveRoutine()`
@*/
PetscErrorCode TaoLineSearchComputeObjective(TaoLineSearch ls, Vec x, PetscReal *f)
{
  Vec       gdummy;
  PetscReal gts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidRealPointer(f, 3);
  PetscCheckSameComm(ls, 1, x, 2);
  if (ls->usetaoroutines) {
    PetscCall(TaoComputeObjective(ls->tao, x, f));
  } else {
    PetscCheck(ls->ops->computeobjective || ls->ops->computeobjectiveandgradient || ls->ops->computeobjectiveandgts, PetscObjectComm((PetscObject)ls), PETSC_ERR_ARG_WRONGSTATE, "Line Search does not have objective function set");
    PetscCall(PetscLogEventBegin(TAOLINESEARCH_Eval, ls, 0, 0, 0));
    if (ls->ops->computeobjective) PetscCallBack("TaoLineSearch callback objective", (*ls->ops->computeobjective)(ls, x, f, ls->userctx_func));
    else if (ls->ops->computeobjectiveandgradient) {
      PetscCall(VecDuplicate(x, &gdummy));
      PetscCallBack("TaoLineSearch callback objective", (*ls->ops->computeobjectiveandgradient)(ls, x, f, gdummy, ls->userctx_funcgrad));
      PetscCall(VecDestroy(&gdummy));
    } else PetscCallBack("TaoLineSearch callback objective", (*ls->ops->computeobjectiveandgts)(ls, x, ls->stepdirection, f, &gts, ls->userctx_funcgts));
    PetscCall(PetscLogEventEnd(TAOLINESEARCH_Eval, ls, 0, 0, 0));
  }
  ls->nfeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
- x - input vector

  Output Parameters:
+ f - Objective value at `x`
- g - Gradient vector at `x`

  Level: developer

  Note:
  `TaoLineSearchComputeObjectiveAndGradient()` is typically used within line searches
  so most users would not generally call this routine themselves.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchComputeGradient()`, `TaoLineSearchComputeObjectiveAndGradient()`, `TaoLineSearchSetObjectiveRoutine()`
@*/
PetscErrorCode TaoLineSearchComputeObjectiveAndGradient(TaoLineSearch ls, Vec x, PetscReal *f, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidRealPointer(f, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscCheckSameComm(ls, 1, x, 2);
  PetscCheckSameComm(ls, 1, g, 4);
  if (ls->usetaoroutines) {
    PetscCall(TaoComputeObjectiveAndGradient(ls->tao, x, f, g));
  } else {
    PetscCall(PetscLogEventBegin(TAOLINESEARCH_Eval, ls, 0, 0, 0));
    if (ls->ops->computeobjectiveandgradient) PetscCallBack("TaoLineSearch callback objective/gradient", (*ls->ops->computeobjectiveandgradient)(ls, x, f, g, ls->userctx_funcgrad));
    else {
      PetscCallBack("TaoLineSearch callback objective", (*ls->ops->computeobjective)(ls, x, f, ls->userctx_func));
      PetscCallBack("TaoLineSearch callback gradient", (*ls->ops->computegradient)(ls, x, g, ls->userctx_grad));
    }
    PetscCall(PetscLogEventEnd(TAOLINESEARCH_Eval, ls, 0, 0, 0));
    PetscCall(PetscInfo(ls, "TaoLineSearch Function evaluation: %14.12e\n", (double)(*f)));
  }
  ls->nfgeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchComputeGradient - Computes the gradient of the objective function

  Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
- x - input vector

  Output Parameter:
. g - gradient vector

  Level: developer

  Note:
  `TaoComputeGradient()` is typically used within line searches
  so most users would not generally call this routine themselves.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchComputeObjective()`, `TaoLineSearchComputeObjectiveAndGradient()`, `TaoLineSearchSetGradient()`
@*/
PetscErrorCode TaoLineSearchComputeGradient(TaoLineSearch ls, Vec x, Vec g)
{
  PetscReal fdummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheckSameComm(ls, 1, x, 2);
  PetscCheckSameComm(ls, 1, g, 3);
  if (ls->usetaoroutines) {
    PetscCall(TaoComputeGradient(ls->tao, x, g));
  } else {
    PetscCall(PetscLogEventBegin(TAOLINESEARCH_Eval, ls, 0, 0, 0));
    if (ls->ops->computegradient) PetscCallBack("TaoLineSearch callback gradient", (*ls->ops->computegradient)(ls, x, g, ls->userctx_grad));
    else PetscCallBack("TaoLineSearch callback gradient", (*ls->ops->computeobjectiveandgradient)(ls, x, &fdummy, g, ls->userctx_funcgrad));
    PetscCall(PetscLogEventEnd(TAOLINESEARCH_Eval, ls, 0, 0, 0));
  }
  ls->ngeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchComputeObjectiveAndGTS - Computes the objective function value and inner product of gradient and
  step direction at a given point

  Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
- x - input vector

  Output Parameters:
+ f - Objective value at `x`
- gts - inner product of gradient and step direction at `x`

  Level: developer

  Note:
  `TaoLineSearchComputeObjectiveAndGTS()` is typically used within line searches
  so most users would not generally call this routine themselves.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchComputeGradient()`, `TaoLineSearchComputeObjectiveAndGradient()`, `TaoLineSearchSetObjectiveRoutine()`
@*/
PetscErrorCode TaoLineSearchComputeObjectiveAndGTS(TaoLineSearch ls, Vec x, PetscReal *f, PetscReal *gts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidRealPointer(f, 3);
  PetscValidRealPointer(gts, 4);
  PetscCheckSameComm(ls, 1, x, 2);
  PetscCall(PetscLogEventBegin(TAOLINESEARCH_Eval, ls, 0, 0, 0));
  PetscCallBack("TaoLineSearch callback objective/gts", (*ls->ops->computeobjectiveandgts)(ls, x, ls->stepdirection, f, gts, ls->userctx_funcgts));
  PetscCall(PetscLogEventEnd(TAOLINESEARCH_Eval, ls, 0, 0, 0));
  PetscCall(PetscInfo(ls, "TaoLineSearch Function evaluation: %14.12e\n", (double)(*f)));
  ls->nfeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchGetSolution - Returns the solution to the line search

  Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameters:
+ x - the new solution
. f - the objective function value at `x`
. g - the gradient at `x`
. steplength - the multiple of the step direction taken by the line search
- reason - the reason why the line search terminated

  Level: developer

.seealso: `TaoLineSearchGetStartingVector()`, `TaoLineSearchGetStepDirection()`
@*/
PetscErrorCode TaoLineSearchGetSolution(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, PetscReal *steplength, TaoLineSearchConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidRealPointer(f, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscValidIntPointer(reason, 6);
  if (ls->new_x) PetscCall(VecCopy(ls->new_x, x));
  *f = ls->new_f;
  if (ls->new_g) PetscCall(VecCopy(ls->new_g, g));
  if (steplength) *steplength = ls->step;
  *reason = ls->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchGetStartingVector - Gets a the initial point of the line
  search.

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameter:
. x - The initial point of the line search

  Level: advanced

.seealso: `TaoLineSearchGetSolution()`, `TaoLineSearchGetStepDirection()`
@*/
PetscErrorCode TaoLineSearchGetStartingVector(TaoLineSearch ls, Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  if (x) *x = ls->start_x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchGetStepDirection - Gets the step direction of the line
  search.

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameter:
. s - the step direction of the line search

  Level: advanced

.seealso: `TaoLineSearchGetSolution()`, `TaoLineSearchGetStartingVector()`
@*/
PetscErrorCode TaoLineSearchGetStepDirection(TaoLineSearch ls, Vec *s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  if (s) *s = ls->stepdirection;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchGetFullStepObjective - Returns the objective function value at the full step.  Useful for some minimization algorithms.

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameter:
. f - the objective value at the full step length

  Level: developer

.seealso: `TaoLineSearchGetSolution()`, `TaoLineSearchGetStartingVector()`, `TaoLineSearchGetStepDirection()`
@*/

PetscErrorCode TaoLineSearchGetFullStepObjective(TaoLineSearch ls, PetscReal *f_fullstep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  *f_fullstep = ls->f_fullstep;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchSetVariableBounds - Sets the upper and lower bounds for a bounded line search

  Logically Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
. xl  - vector of lower bounds
- xu  - vector of upper bounds

  Level: beginner

  Note:
  If the variable bounds are not set with this routine, then
  `PETSC_NINFINITY` and `PETSC_INFINITY` are assumed

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoSetVariableBounds()`, `TaoLineSearchCreate()`
@*/
PetscErrorCode TaoLineSearchSetVariableBounds(TaoLineSearch ls, Vec xl, Vec xu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  if (xl) PetscValidHeaderSpecific(xl, VEC_CLASSID, 2);
  if (xu) PetscValidHeaderSpecific(xu, VEC_CLASSID, 3);
  PetscCall(PetscObjectReference((PetscObject)xl));
  PetscCall(PetscObjectReference((PetscObject)xu));
  PetscCall(VecDestroy(&ls->lower));
  PetscCall(VecDestroy(&ls->upper));
  ls->lower   = xl;
  ls->upper   = xu;
  ls->bounded = (PetscBool)(xl || xu);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchSetInitialStepLength - Sets the initial step length of a line
  search.  If this value is not set then 1.0 is assumed.

  Logically Collective

  Input Parameters:
+ ls - the `TaoLineSearch` context
- s - the initial step size

  Level: intermediate

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchGetStepLength()`, `TaoLineSearchApply()`
@*/
PetscErrorCode TaoLineSearchSetInitialStepLength(TaoLineSearch ls, PetscReal s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ls, s, 2);
  ls->initstep = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoLineSearchGetStepLength - Get the current step length

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameter:
. s - the current step length

  Level: intermediate

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchSetInitialStepLength()`, `TaoLineSearchApply()`
@*/
PetscErrorCode TaoLineSearchGetStepLength(TaoLineSearch ls, PetscReal *s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  *s = ls->step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoLineSearchRegister - Adds a line-search algorithm to the registry

   Not Collective

   Input Parameters:
+  sname - name of a new user-defined solver
-  func - routine to Create method context

   Sample usage:
.vb
   TaoLineSearchRegister("my_linesearch",MyLinesearchCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     TaoLineSearchSetType(ls,"my_linesearch")
   or at runtime via the option
$     -tao_ls_type my_linesearch

   Level: developer

   Note:
   `TaoLineSearchRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`
@*/
PetscErrorCode TaoLineSearchRegister(const char sname[], PetscErrorCode (*func)(TaoLineSearch))
{
  PetscFunctionBegin;
  PetscCall(TaoLineSearchInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoLineSearchList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoLineSearchAppendOptionsPrefix - Appends to the prefix used for searching
   for all `TaoLineSearch` options in the database.

   Collective

   Input Parameters:
+  ls - the `TaoLineSearch` solver context
-  prefix - the prefix string to prepend to all line search requests

   Level: advanced

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   This is inherited from the `Tao` object so rarely needs to be set

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchSetOptionsPrefix()`, `TaoLineSearchGetOptionsPrefix()`
@*/
PetscErrorCode TaoLineSearchAppendOptionsPrefix(TaoLineSearch ls, const char p[])
{
  return PetscObjectAppendOptionsPrefix((PetscObject)ls, p);
}

/*@C
  TaoLineSearchGetOptionsPrefix - Gets the prefix used for searching for all
  `TaoLineSearch` options in the database

  Not Collective

  Input Parameter:
. ls - the `TaoLineSearch` context

  Output Parameter:
. prefix - pointer to the prefix string used is returned

  Level: advanced

  Fortran Note:
  The user should pass in a string 'prefix' of
  sufficient length to hold the prefix.

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchSetOptionsPrefix()`, `TaoLineSearchAppendOptionsPrefix()`
@*/
PetscErrorCode TaoLineSearchGetOptionsPrefix(TaoLineSearch ls, const char *p[])
{
  return PetscObjectGetOptionsPrefix((PetscObject)ls, p);
}

/*@C
   TaoLineSearchSetOptionsPrefix - Sets the prefix used for searching for all
   `TaoLineSearch` options in the database.

   Logically Collective

   Input Parameters:
+  ls - the `TaoLineSearch` context
-  prefix - the prefix string to prepend to all `ls` option requests

   Level: advanced

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   This is inherited from the `Tao` object so rarely needs to be set

   For example, to distinguish between the runtime options for two
   different line searches, one could call
.vb
      TaoLineSearchSetOptionsPrefix(ls1,"sys1_")
      TaoLineSearchSetOptionsPrefix(ls2,"sys2_")
.ve

   This would enable use of different options for each system, such as
.vb
      -sys1_tao_ls_type mt
      -sys2_tao_ls_type armijo
.ve

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchAppendOptionsPrefix()`, `TaoLineSearchGetOptionsPrefix()`
@*/
PetscErrorCode TaoLineSearchSetOptionsPrefix(TaoLineSearch ls, const char p[])
{
  return PetscObjectSetOptionsPrefix((PetscObject)ls, p);
}
