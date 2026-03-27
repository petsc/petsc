#include <petsc/private/daimpl.h>
#include <petscblaslapack.h>

PetscClassId      PETSCDA_CLASSID          = 0;
PetscLogEvent     PetscDA_Analysis         = 0;
PetscBool         PetscDARegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscDAList              = NULL;

static PetscBool PetscDAPackageInitialized = PETSC_FALSE;

/*@C
  PetscDAInitializePackage - This function initializes everything in the `PetscDA`
  package. called on the first call to `PetscDACreate()` when using static or shared
  libraries.

  Logically Collective

  Level: developer

.seealso: `PetscDAFinalizePackage()`, `PetscInitialize()`
@*/
PetscErrorCode PetscDAInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDAPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);

  PetscDAPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("Data Assimilation", &PETSCDA_CLASSID));
  PetscCall(PetscDARegisterAll());
  PetscCall(PetscRegisterFinalize(PetscDAFinalizePackage));
  PetscCall(PetscLogEventRegister("PetscDAAnalysis", PETSCDA_CLASSID, &PetscDA_Analysis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDAFinalizePackage - This function finalizes everything in the `PetscDA` package. It
  is called from `PetscFinalize()`.

  Logically Collective

  Level: developer

.seealso: `PetscDAInitializePackage()`, `PetscInitialize()`
@*/
PetscErrorCode PetscDAFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscDAList));
  PetscDARegisterAllCalled  = PETSC_FALSE;
  PetscDAPackageInitialized = PETSC_FALSE;
  PetscDA_Analysis          = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDARegister - Registers a constructor for a `PetscDA` implementation with the
  dispatcher.

  Not Collective

  Input Parameters:
+ sname    - name associated with the implementation
- function - routine that creates the implementation and installs method table

  Level: developer

.seealso: [](ch_da), `PetscDARegisterAll()`, `PetscDASetType()`
@*/
PetscErrorCode PetscDARegister(const char sname[], PetscErrorCode (*function)(PetscDA))
{
  PetscFunctionBegin;
  PetscCall(PetscDAInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscDAList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDACreate_ETKF(PetscDA);
PETSC_INTERN PetscErrorCode PetscDACreate_LETKF(PetscDA);

/*@
  PetscDARegisterAll - Registers all data assimilation backends that were compiled in.

  Not Collective

  Level: developer

.seealso: [](ch_da), `PetscDARegister()`
@*/
PetscErrorCode PetscDARegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDARegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDARegisterAllCalled = PETSC_TRUE;
  PetscCall(PetscDARegister(PETSCDAETKF, PetscDACreate_ETKF));
  PetscCall(PetscDARegister(PETSCDALETKF, PetscDACreate_LETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PetscDASetOptionsPrefix - Sets the prefix used for searching for all
  PetscDA options in the database.

  Logically Collective

  Input Parameters:
+ das - the `PetscDA` context
- p   - the prefix string to prepend to all PetscDA option requests

  Level: advanced

.seealso: `PetscDA`, `PetscDASetFromOptions()`, `PetscDAAppendOptionsPrefix()`, `PetscDAGetOptionsPrefix()`
@*/
PetscErrorCode PetscDASetOptionsPrefix(PetscDA das, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAAppendOptionsPrefix - Appends to the prefix used for searching for all PetscDA options in the database.

  Logically Collective

  Input Parameters:
+ das - the `PetscDA` context
- p   - the prefix string to prepend to all `PetscDA` option requests

  Level: advanced

.seealso: `PetscDA`, `PetscDASetFromOptions()`, `PetscDASetOptionsPrefix()`, `PetscDAGetOptionsPrefix()`
@*/
PetscErrorCode PetscDAAppendOptionsPrefix(PetscDA das, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetOptionsPrefix - Gets the prefix used for searching for all
  PetscDA options in the database

  Not Collective

  Input Parameter:
. das - the `PetscDA` context

  Output Parameter:
. p - pointer to the prefix string used

  Level: advanced

.seealso: `PetscDA`, `PetscDASetFromOptions()`, `PetscDASetOptionsPrefix()`, `PetscDAAppendOptionsPrefix()`
@*/
PetscErrorCode PetscDAGetOptionsPrefix(PetscDA das, const char *p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDACreate - Creates a new `PetscDA` object for data assimilation.

  Collective

  Input Parameter:
. comm - MPI communicator used to create the object

  Output Parameter:
. da_out - newly created `PetscDA` object

  Level: beginner

.seealso: [](ch_da), `PetscDADestroy()`, `PetscDASetType()`, `PetscDASetUp()`
@*/
PetscErrorCode PetscDACreate(MPI_Comm comm, PetscDA *da_out)
{
  PetscDA da;

  PetscFunctionBegin;
  PetscAssertPointer(da_out, 2);
  PetscCall(PetscDAInitializePackage());
  PetscCall(PetscHeaderCreate(da, PETSCDA_CLASSID, "PetscDA", "Data Assimilation", "DA", comm, PetscDADestroy, PetscDAView));
  PetscCall(PetscMemzero(da->ops, sizeof(*da->ops)));
  da->state_size       = 0;
  da->local_state_size = PETSC_DECIDE;
  da->obs_size         = 0;
  da->local_obs_size   = PETSC_DECIDE;
  da->ndof             = 1;
  da->obs_error_var    = NULL;
  da->R                = NULL;
  da->data             = NULL;
  *da_out              = da;
  PetscCall(PetscDASetType(da, PETSCDAETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDADestroy - Destroys a `PetscDA` object and releases its resources.

  Collective

  Input Parameter:
. da - pointer to the `PetscDA` object to destroy

  Level: beginner

.seealso: [](ch_da), `PetscDACreate()`
@*/
PetscErrorCode PetscDADestroy(PetscDA *da)
{
  PetscFunctionBegin;
  if (!da || !*da) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*da, PETSCDA_CLASSID, 1);
  if (--((PetscObject)*da)->refct > 0) {
    *da = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscTryTypeMethod(*da, destroy);
  PetscCall(PetscHeaderDestroy(da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetType - Sets the data assimilation implementation used by a `PetscDA` object.

  Collective

  Input Parameters:
+ da   - the `PetscDA` context
- type - name of the implementation (for example `PETSCDAETKF`)

  Level: intermediate

.seealso: [](ch_da), `PetscDAGetType()`, `PetscDARegister()`
@*/
PetscErrorCode PetscDASetType(PetscDA da, PetscDAType type)
{
  PetscErrorCode (*r)(PetscDA);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)da, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscDARegisterAll());
  PetscCall(PetscFunctionListFind(PetscDAList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDA type: %s", type);

  PetscTryTypeMethod(da, destroy);
  da->ops->destroy = NULL;
  da->data         = NULL;

  PetscCall((*r)(da));
  PetscCall(PetscObjectChangeTypeName((PetscObject)da, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetType - Gets the name of the implementation currently associated with a `PetscDA`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. type - pointer that will receive the type name (may be `NULL`)

  Level: intermediate

.seealso: [](ch_da), `PetscDASetType()`
@*/
PetscErrorCode PetscDAGetType(PetscDA da, PetscDAType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (type) {
    PetscAssertPointer(type, 2);
    *type = ((PetscObject)da)->type_name;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetFromOptions - Configures a `PetscDA` object from the options database.

  Collective

  Input Parameter:
. da - the `PetscDA` context to set up

  Level: intermediate

.seealso: [](ch_da), `PetscDASetType()`, `PetscObjectOptionsBegin()`
@*/
PetscErrorCode PetscDASetFromOptions(PetscDA da)
{
  char      type_name[256];
  PetscBool type_set;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)da);
  PetscCall(PetscOptionsFList("-petscda_type", "Data assimilation method", "PetscDASetType", PetscDAList, ((PetscObject)da)->type_name, type_name, sizeof(type_name), &type_set));
  if (type_set) PetscCall(PetscDASetType(da, type_name));
  if (da->ops->setfromoptions) PetscCall((*da->ops->setfromoptions)(da, &PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetNDOF - Set the number of degrees of freedom per grid point

  Logically Collective

  Input Parameters:
+ da   - the `PetscDA` context
- ndof - number of degrees of freedom per grid point (e.g., 2 for shallow water with h and hu)

  Level: intermediate

  Note:
  This must be called before `PetscDASetUp()`. The default is 1 (scalar field).

  Developer Note:
  It is a limitation that each grid point needs the same number of degrees of freedom.

.seealso: `PetscDA`, `PetscDAGetNDOF()`, `PetscDASetUp()`, `PetscDASetSizes()`
@*/
PetscErrorCode PetscDASetNDOF(PetscDA da, PetscInt ndof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, ndof, 2);
  PetscCheck(ndof > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "ndof must be positive, got %" PetscInt_FMT, ndof);
  da->ndof = ndof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetNDOF - Get the number of degrees of freedom per grid point

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. ndof - number of degrees of freedom per grid point

  Level: intermediate

.seealso: `PetscDA`, `PetscDASetNDOF()`
@*/
PetscErrorCode PetscDAGetNDOF(PetscDA da, PetscInt *ndof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(ndof, 2);
  *ndof = da->ndof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetUp - Allocates internal data structures for a `PetscDA` based on the previously provided sizes.

  Collective

  Input Parameter:
. da - the `PetscDA` context to assemble

  Level: beginner

.seealso: [](ch_da), `PetscDASetSizes()`, `PetscDASetType()`
@*/
PetscErrorCode PetscDASetUp(PetscDA da)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscTryTypeMethod(da, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAView - Views a `PetscDA` and its implementation-specific data structure.

  Collective

  Input Parameters:
+ da     - the `PetscDA` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: beginner

.seealso: [](ch_da), `PetscDAViewFromOptions()`
@*/
PetscErrorCode PetscDAView(PetscDA da, PetscViewer viewer)
{
  PetscBool   iascii;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)da), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(da, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDA Object: %" PetscInt_FMT " MPI process%s\n", (PetscInt)size, size > 1 ? "es" : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  type: %s\n", ((PetscObject)da)->type_name ? ((PetscObject)da)->type_name : "not set"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  State size: %" PetscInt_FMT "\n", da->state_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Observation size: %" PetscInt_FMT "\n", da->obs_size));
  }

  if (da->ops->view) PetscCall((*da->ops->view)(da, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAViewFromOptions - Processes command-line options to determine if a `PetscDA` should be viewed.

  Collective

  Input Parameters:
+ da     - the `PetscDA` context
. obj    - optional object that provides the prefix for options
- option - option name to check (may be `NULL`)

  Level: beginner

.seealso: [](ch_da), `PetscDAView()`, `PetscObjectViewFromOptions()`
@*/
PetscErrorCode PetscDAViewFromOptions(PetscDA da, PetscObject obj, const char option[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)da, obj, option));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetObsErrorVariance - Sets the observation-error variances associated with a `PetscDA`.

  Collective

  Input Parameters:
+ da            - the `PetscDA` context
- obs_error_var - vector containing observation error variances (assumes R is a diagonal matrix)

  Notes:
  This function creates or updates both the observation error variance vector and the
  observation error covariance matrix `R`. The matrix `R` is constructed as a diagonal matrix
  with the variances on the diagonal.

  Level: beginner

.seealso: [](ch_da), `PetscDAGetObsErrorVariance()`
@*/
PetscErrorCode PetscDASetObsErrorVariance(PetscDA da, Vec obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(obs_error_var, VEC_CLASSID, 2);
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Call PetscDASetSizes() before PetscDASetObsErrorVariance()");

  /* Create or update observation error variance vector */
  if (!da->obs_error_var) PetscCall(VecDuplicate(obs_error_var, &da->obs_error_var));
  PetscCall(VecCopy(obs_error_var, da->obs_error_var));

  /* Create or update observation error covariance matrix R (p x p) as AIJ matrix
     This is currently initialized as a diagonal matrix, but can be used
     for non-diagonal covariance in the future */
  if (!da->R) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)da), &da->R));
    PetscCall(MatSetSizes(da->R, da->local_obs_size, da->local_obs_size, da->obs_size, da->obs_size));
    PetscCall(MatSetType(da->R, MATAIJ));
    PetscCall(MatSetFromOptions(da->R));
    PetscCall(MatSetUp(da->R));
  }

  /* Set R as diagonal matrix with variances on diagonal */
  PetscCall(MatZeroEntries(da->R));
  PetscCall(MatDiagonalSet(da->R, da->obs_error_var, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(da->R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(da->R, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetObsErrorVariance - Returns a borrowed reference to the observation-error variance vector.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. obs_error_var - pointer to the variance vector managed by the `PetscDA`

  Level: beginner

.seealso: [](ch_da), `PetscDASetObsErrorVariance()`
@*/
PetscErrorCode PetscDAGetObsErrorVariance(PetscDA da, Vec *obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(obs_error_var, 2);
  *obs_error_var = da->obs_error_var;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetSizes - Sets the state and observation sizes for a `PetscDA`

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
. state_size - number of state components
- obs_size   - number of observation components

  Level: beginner

  Developer Note:
  It is not clear this is a good API, shouldn't one provide template vectors for these?

.seealso: [](ch_da), `PetscDAGetSizes()`, `PetscDASetUp()`, `PetscDAEnsembleSetSize()`
@*/
PetscErrorCode PetscDASetSizes(PetscDA da, PetscInt state_size, PetscInt obs_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, state_size, 2);
  PetscValidLogicalCollectiveInt(da, obs_size, 3);

  da->state_size = state_size;
  da->obs_size   = obs_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetLocalSizes - Sets the local state and observation dimensions used by a `PetscDA`.

  Collective

  Input Parameters:
+ da               - the `PetscDA` context
. local_state_size - number of local state components (or `PETSC_DECIDE`)
- local_obs_size   - number of local observation components (or `PETSC_DECIDE`)

  Level: beginner

.seealso: [](ch_da), `PetscDASetSizes()`, `PetscDASetUp()`
@*/
PetscErrorCode PetscDASetLocalSizes(PetscDA da, PetscInt local_state_size, PetscInt local_obs_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  da->local_state_size = local_state_size;
  da->local_obs_size   = local_obs_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetSizes - Retrieves the state size and observation size from a `PetscDA`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameters:
+ state_size - number of state components (may be `NULL`)
- obs_size   - number of observation components (may be `NULL`)

  Level: beginner

.seealso: [](ch_da), `PetscDASetSizes()`
@*/
PetscErrorCode PetscDAGetSizes(PetscDA da, PetscInt *state_size, PetscInt *obs_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (state_size) PetscAssertPointer(state_size, 2);
  if (obs_size) PetscAssertPointer(obs_size, 3);
  if (state_size) *state_size = da->state_size;
  if (obs_size) *obs_size = da->obs_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}
