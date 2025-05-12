#include <petsc/private/regressorimpl.h>

PetscBool         PetscRegressorRegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscRegressorList              = NULL;

PetscClassId PETSCREGRESSOR_CLASSID;

/* Logging support */
PetscLogEvent PetscRegressor_SetUp, PetscRegressor_Fit, PetscRegressor_Predict;

/*@C
  PetscRegressorRegister - Adds a method to the `PetscRegressor` package.

  Not collective

  Input Parameters:
+ sname    - name of a new user-defined regressor
- function - routine to create method context

  Notes:
  `PetscRegressorRegister()` may be called multiple times to add several user-defined regressors.

  Example Usage:
.vb
   PetscRegressorRegister("my_regressor",MyRegressorCreate);
.ve

  Then, your regressor can be chosen with the procedural interface via
.vb
     PetscRegressorSetType(regressor,"my_regressor")
.ve
  or at runtime via the option
.vb
    -regressor_type my_regressor
.ve

  Level: advanced

.seealso: `PetscRegressorRegisterAll()`
@*/
PetscErrorCode PetscRegressorRegister(const char sname[], PetscErrorCode (*function)(PetscRegressor))
{
  PetscFunctionBegin;
  PetscCall(PetscRegressorInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscRegressorList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorCreate - Creates a `PetscRegressor` object.

  Collective

  Input Parameter:
. comm - the MPI communicator that will share the `PetscRegressor` object

  Output Parameter:
. newregressor - the new `PetscRegressor` object

  Level: beginner

.seealso: `PetscRegressorFit()`, `PetscRegressorPredict()`, `PetscRegressor`
@*/
PetscErrorCode PetscRegressorCreate(MPI_Comm comm, PetscRegressor *newregressor)
{
  PetscRegressor regressor;

  PetscFunctionBegin;
  PetscAssertPointer(newregressor, 2);
  *newregressor = NULL;
  PetscCall(PetscRegressorInitializePackage());

  PetscCall(PetscHeaderCreate(regressor, PETSCREGRESSOR_CLASSID, "PetscRegressor", "Regressor", "PetscRegressor", comm, PetscRegressorDestroy, PetscRegressorView));

  regressor->setupcalled = PETSC_FALSE;
  regressor->fitcalled   = PETSC_FALSE;
  regressor->data        = NULL;
  regressor->training    = NULL;
  regressor->target      = NULL;
  PetscObjectParameterSetDefault(regressor, regularizer_weight, 1.0); // Default to regularizer weight of 1.0, usually the default in SciKit-learn

  *newregressor = regressor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorView - Prints information about the `PetscRegressor` object

  Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context
- viewer    - a `PetscViewer` context

  Options Database Key:
. -regressor_view - Calls `PetscRegressorView()` at the end of `PetscRegressorFit()`

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

.seealso: [](ch_regressor), `PetscRegressor`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode PetscRegressorView(PetscRegressor regressor, PetscViewer viewer)
{
  PetscBool          isascii, isstring;
  PetscRegressorType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)regressor)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(regressor, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)regressor, viewer));

    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(regressor, view, viewer);
    if (regressor->tao) PetscCall(TaoView(regressor->tao, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(PetscRegressorGetType(regressor, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " PetscRegressorType: %-7.7s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorViewFromOptions - View a `PetscRegressor` object based on values in the options database

  Collective

  Input Parameters:
+ A    - the  `PetscRegressor` context
. obj  - Optional object that provides the prefix for the options database
- name - command line option

  Level: intermediate

.seealso: [](ch_regressor), `PetscRegressor`, `PetscRegressorView`, `PetscObjectViewFromOptions()`, `PetscRegressorCreate()`
@*/
PetscErrorCode PetscRegressorViewFromOptions(PetscRegressor A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSCREGRESSOR_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorSetFromOptions - Sets `PetscRegressor` options from the options database.

  Collective

  Input Parameter:
. regressor - the `PetscRegressor` context

  Options Database Keys:
. -regressor_type <type> - the particular type of regressor to be used; see `PetscRegressorType` for complete list

  Level: beginner

  Note:
  This routine must be called before `PetscRegressorSetUp()` (or `PetscRegressorFit()`, which calls
  the former) if the user is to be allowed to set the regressor type.

.seealso: `PetscRegressor`, `PetscRegressorCreate()`
@*/
PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor regressor)
{
  PetscBool          flg;
  PetscRegressorType default_type = PETSCREGRESSORLINEAR;
  char               type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (((PetscObject)regressor)->type_name) default_type = ((PetscObject)regressor)->type_name;
  PetscObjectOptionsBegin((PetscObject)regressor);
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-regressor_type", "PetscRegressor type", "PetscRegressorSetType", PetscRegressorList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(PetscRegressorSetType(regressor, type));
  } else if (!((PetscObject)regressor)->type_name) {
    PetscCall(PetscRegressorSetType(regressor, default_type));
  }
  PetscCall(PetscOptionsReal("-regressor_regularizer_weight", "Weight for the regularizer", "PetscRegressorSetRegularizerWeight", regressor->regularizer_weight, &regressor->regularizer_weight, &flg));
  if (flg) PetscCall(PetscRegressorSetRegularizerWeight(regressor, regressor->regularizer_weight));
  // The above is a little superfluous, because we have already set regressor->regularizer_weight above, but we also need to set the flag indicating that the user has set the weight!
  PetscTryTypeMethod(regressor, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorSetUp - Sets up the internal data structures for the later use of a regressor.

  Collective

  Input Parameter:
. regressor - the `PetscRegressor` context

  Notes:
  For basic use of the `PetscRegressor` solvers the user need not to explicitly call
  `PetscRegressorSetUp()`, since these actions will automatically occur during
  the call to `PetscRegressorFit()`.  However, if one wishes to control this
  phase separately, `PetscRegressorSetUp()` should be called after `PetscRegressorCreate()`,
  `PetscRegressorSetUp()`, and optional routines of the form `PetscRegressorSetXXX()`,
  but before `PetscRegressorFit()`.

  Level: advanced

.seealso: `PetscRegressorCreate()`, `PetscRegressorFit()`, `PetscRegressorDestroy()`
@*/
PetscErrorCode PetscRegressorSetUp(PetscRegressor regressor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (regressor->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(PetscRegressor_SetUp, regressor, 0, 0, 0));
  //TODO is there some mat vec etc that must be set, like TaoSolution?
  PetscTryTypeMethod(regressor, setup);
  regressor->setupcalled = PETSC_TRUE;
  PetscCall(PetscLogEventEnd(PetscRegressor_SetUp, regressor, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* NOTE: I've decided to make this take X and y, like the Scikit-learn Fit routines do.
 * Am I overlooking some reason that X should be set in a separate function call, a la KSPSetOperators()?. */
/*@
  PetscRegressorFit - Fit, or train, a regressor from a training dataset

  Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context
. X         - matrix of training data (of dimension [number of samples] x [number of features])
- y         - vector of target values from the training dataset

  Level: beginner

.seealso: `PetscRegressorCreate()`, `PetscRegressorSetUp()`, `PetscRegressorDestroy()`, `PetscRegressorPredict()`
@*/
PetscErrorCode PetscRegressorFit(PetscRegressor regressor, Mat X, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (X) PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  if (y) PetscValidHeaderSpecific(y, VEC_CLASSID, 3);

  if (X) {
    PetscCall(PetscObjectReference((PetscObject)X));
    PetscCall(MatDestroy(&regressor->training));
    regressor->training = X;
  }
  if (y) {
    PetscCall(PetscObjectReference((PetscObject)y));
    PetscCall(VecDestroy(&regressor->target));
    regressor->target = y;
  }
  PetscCall(PetscRegressorSetUp(regressor));

  PetscCall(PetscLogEventBegin(PetscRegressor_Fit, regressor, X, y, 0));
  PetscUseTypeMethod(regressor, fit);
  PetscCall(PetscLogEventEnd(PetscRegressor_Fit, regressor, X, y, 0));
  //TODO print convergence data
  PetscCall(PetscRegressorViewFromOptions(regressor, NULL, "-regressor_view"));
  regressor->fitcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorPredict - Compute predictions (that is, perform inference) using a fitted regression model.

  Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context (for which `PetscRegressorFit()` must have been called)
- X         - data matrix of unlabeled observations

  Output Parameter:
. y - vector of predicted labels

  Level: beginner

.seealso: `PetscRegressorFit()`, `PetscRegressorDestroy()`
@*/
PetscErrorCode PetscRegressorPredict(PetscRegressor regressor, Mat X, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (X) PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  if (y) PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscCheck(regressor->fitcalled == PETSC_TRUE, ((PetscObject)regressor)->comm, PETSC_ERR_ARG_WRONGSTATE, "PetscRegressorFit() must be called before PetscRegressorPredict()");
  PetscCall(PetscLogEventBegin(PetscRegressor_Predict, regressor, X, y, 0));
  PetscTryTypeMethod(regressor, predict, X, y);
  PetscCall(PetscLogEventEnd(PetscRegressor_Predict, regressor, X, y, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorReset - Resets a `PetscRegressor` context by removing any allocated `Vec` and `Mat`. Any options set in the object remain.

  Collective

  Input Parameter:
. regressor - context obtained from `PetscRegressorCreate()`

  Level: intermediate

.seealso: `PetscRegressorCreate()`, `PetscRegressorSetUp()`, `PetscRegressorFit()`, `PetscRegressorPredict()`, `PetscRegressorDestroy()`
@*/
PetscErrorCode PetscRegressorReset(PetscRegressor regressor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (regressor->ops->reset) PetscTryTypeMethod(regressor, reset);
  PetscCall(MatDestroy(&regressor->training));
  PetscCall(VecDestroy(&regressor->target));
  PetscCall(TaoDestroy(&regressor->tao));
  regressor->setupcalled = PETSC_FALSE;
  regressor->fitcalled   = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorDestroy - Destroys the regressor context that was created with `PetscRegressorCreate()`.

  Collective

  Input Parameter:
. regressor - the `PetscRegressor` context

  Level: beginner

.seealso: `PetscRegressorCreate()`, `PetscRegressorSetUp()`, `PetscRegressorReset()`, `PetscRegressor`
@*/
PetscErrorCode PetscRegressorDestroy(PetscRegressor *regressor)
{
  PetscFunctionBegin;
  if (!*regressor) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*regressor, PETSCREGRESSOR_CLASSID, 1);
  if (--((PetscObject)*regressor)->refct > 0) {
    *regressor = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscRegressorReset(*regressor));
  PetscTryTypeMethod(*regressor, destroy);

  PetscCall(PetscHeaderDestroy(regressor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorSetType - Sets the type for the regressor.

  Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context
- type      - a known regression method

  Options Database Key:
. -regressor_type <type> - Sets the type of regressor; use -help for a list of available types

  Level: intermediate

  Notes:
  See "include/petscregressor.h" for available methods (for instance)
.    `PETSCREGRESSORLINEAR` - Regression model that is linear in its coefficients; supports ordinary least squares as well as regularized variants

  Normally, it is best to use the `PetscRegressorSetFromOptions()` command and then
  set the `PetscRegressor` type from the options database rather than by using
  this routine, as this provides maximum flexibility.
  The `PetscRegressorSetType()` routine is provided for those situations where it
  is necessary to set the nonlinear solver independently of the command
  line or options database.

.seealso: `PetscRegressorType`
@*/
PetscErrorCode PetscRegressorSetType(PetscRegressor regressor, PetscRegressorType type)
{
  PetscErrorCode (*r)(PetscRegressor);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)regressor, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PetscRegressorList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PetscRegressor type %s", type);

  /* Destroy the existing solver information */
  PetscTryTypeMethod(regressor, destroy);
  PetscCall(TaoDestroy(&regressor->tao));
  regressor->ops->setup          = NULL;
  regressor->ops->setfromoptions = NULL;
  regressor->ops->settraining    = NULL;
  regressor->ops->fit            = NULL;
  regressor->ops->predict        = NULL;
  regressor->ops->destroy        = NULL;
  regressor->ops->reset          = NULL;
  regressor->ops->view           = NULL;

  /* Call the PetscRegressorCreate_XXX routine for this particular regressor */
  regressor->setupcalled = PETSC_FALSE;
  PetscCall((*r)(regressor));
  PetscCall(PetscObjectChangeTypeName((PetscObject)regressor, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorGetType - Gets the current `PetscRegressorType` being used in the `PetscRegressor` object

  Not Collective

  Input Parameter:
. regressor - the `PetscRegressor` solver context

  Output Parameter:
. type - the `PetscRegressorType`

  Level: intermediate

.seealso: [](ch_regressor), `PetscRegressor`, `PetscRegressorType`, `PetscRegressorSetType()`
@*/
PetscErrorCode PetscRegressorGetType(PetscRegressor regressor, PetscRegressorType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)regressor)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorSetRegularizerWeight - Sets the weight to be used for the regularizer for a `PetscRegressor` context

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context
- weight    - the regularizer weight

  Options Database Key:
. regressor_regularizer_weight <weight> - sets the regularizer's weight

  Level: beginner

.seealso: `PetscRegressorSetType`
@*/
PetscErrorCode PetscRegressorSetRegularizerWeight(PetscRegressor regressor, PetscReal weight)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscValidLogicalCollectiveReal(regressor, weight, 2);
  regressor->regularizer_weight = weight;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorGetTao - Returns the `Tao` context for a `PetscRegressor` object.

  Not Collective, but if the `PetscRegressor` is parallel, then the `Tao` object is parallel

  Input Parameter:
. regressor - the regressor context

  Output Parameter:
. tao - the `Tao` context

  Level: beginner

  Notes:
  The `Tao` object will be created if it does not yet exist.

  The user can directly manipulate the `Tao` context to set various
  options, etc.  Likewise, the user can then extract and manipulate the
  child contexts such as `KSP` or `TaoLineSearch`as well.

  Depending on the type of the regressor and the options that are set, the regressor may use not use a `Tao` object.

.seealso: `PetscRegressorLinearGetKSP()`
@*/
PetscErrorCode PetscRegressorGetTao(PetscRegressor regressor, Tao *tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(tao, 2);
  // Analogous to how SNESGetKSP() operates, this routine should create the Tao if it doesn't exist.
  if (!regressor->tao) {
    PetscCall(TaoCreate(PetscObjectComm((PetscObject)regressor), &regressor->tao));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)regressor->tao, (PetscObject)regressor, 1));
    PetscCall(PetscObjectSetOptions((PetscObject)regressor->tao, ((PetscObject)regressor)->options));
  }
  *tao = regressor->tao;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorSetOptionsPrefix - Sets the prefix used for searching for all
  PetscRegressor options in the database.

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context
- p         - the prefix string to prepend to all PetscRegressor option requests

  Level: advanced

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  For example, to distinguish between the runtime options for two
  different PetscRegressor solvers, one could call
.vb
      PetscRegressorSetOptionsPrefix(regressor1,"sys1_")
      PetscRegressorSetOptionsPrefix(regressor2,"sys2_")
.ve

  This would enable use of different options for each system, such as
.vb
      -sys1_regressor_method linear -sys1_regressor_regularizer_weight 1.2
      -sys2_regressor_method linear -sys2_regressor_regularizer_weight 1.1
.ve

.seealso: [](ch_regressor), `PetscRegressor`, `PetscRegressorSetFromOptions()`, `PetscRegressorAppendOptionsPrefix()`, `PetscRegressorGetOptionsPrefix()`
@*/
PetscErrorCode PetscRegressorSetOptionsPrefix(PetscRegressor regressor, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)regressor, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorAppendOptionsPrefix - Appends to the prefix used for searching for all PetscRegressor options in the database.

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` solver context
- p         - the prefix string to prepend to all `PetscRegressor` option requests

  Level: advanced

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is automatically the hyphen.

.seealso: [](ch_regressor), `PetscRegressor`, `PetscRegressorSetFromOptions()`, `PetscRegressorSetOptionsPrefix()`, `PetscRegressorGetOptionsPrefix()`
@*/
PetscErrorCode PetscRegressorAppendOptionsPrefix(PetscRegressor regressor, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)regressor, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorGetOptionsPrefix - Gets the prefix used for searching for all
  PetscRegressor options in the database

  Not Collective

  Input Parameter:
. regressor - the `PetscRegressor` context

  Output Parameter:
. p - pointer to the prefix string used is returned

  Fortran Notes:
  Pass in a string 'prefix' of sufficient length to hold the prefix.

  Level: advanced

.seealso: [](ch_regressor), `PetscRegressor`, `PetscRegressorSetFromOptions()`, `PetscRegressorSetOptionsPrefix()`, `PetscRegressorAppendOptionsPrefix()`
@*/
PetscErrorCode PetscRegressorGetOptionsPrefix(PetscRegressor regressor, const char *p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)regressor, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}
