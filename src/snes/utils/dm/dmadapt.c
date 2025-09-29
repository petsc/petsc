#include <petscdmadaptor.h> /*I "petscdmadaptor.h" I*/
#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscds.h>
#include <petscblaslapack.h>
#include <petscsnes.h>
#include <petscdraw.h>

#include <petsc/private/dmadaptorimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/petscfeimpl.h>

PetscClassId DMADAPTOR_CLASSID;

PetscFunctionList DMAdaptorList              = NULL;
PetscBool         DMAdaptorRegisterAllCalled = PETSC_FALSE;

PetscFunctionList DMAdaptorMonitorList              = NULL;
PetscFunctionList DMAdaptorMonitorCreateList        = NULL;
PetscFunctionList DMAdaptorMonitorDestroyList       = NULL;
PetscBool         DMAdaptorMonitorRegisterAllCalled = PETSC_FALSE;

const char *const DMAdaptationCriteria[] = {"NONE", "REFINE", "LABEL", "METRIC", "DMAdaptationCriterion", "DM_ADAPTATION_", NULL};

/*@C
  DMAdaptorRegister - Adds a new adaptor component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine

  Example Usage:
.vb
  DMAdaptorRegister("my_adaptor", MyAdaptorCreate);
.ve

  Then, your adaptor type can be chosen with the procedural interface via
.vb
  DMAdaptorCreate(MPI_Comm, DMAdaptor *);
  DMAdaptorSetType(DMAdaptor, "my_adaptor");
.ve
  or at runtime via the option
.vb
  -adaptor_type my_adaptor
.ve

  Level: advanced

  Note:
  `DMAdaptorRegister()` may be called multiple times to add several user-defined adaptors

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMAdaptor`, `DMAdaptorRegisterAll()`, `DMAdaptorRegisterDestroy()`
@*/
PetscErrorCode DMAdaptorRegister(const char name[], PetscErrorCode (*create_func)(DMAdaptor))
{
  PetscFunctionBegin;
  PetscCall(DMInitializePackage());
  PetscCall(PetscFunctionListAdd(&DMAdaptorList, name, create_func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMAdaptorCreate_Gradient(DMAdaptor);
PETSC_EXTERN PetscErrorCode DMAdaptorCreate_Flux(DMAdaptor);

/*@C
  DMAdaptorRegisterAll - Registers all of the adaptor components in the `DM` package.

  Not Collective

  Level: advanced

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMAdaptorType`, `DMRegisterAll()`, `DMAdaptorRegisterDestroy()`
@*/
PetscErrorCode DMAdaptorRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMAdaptorRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  DMAdaptorRegisterAllCalled = PETSC_TRUE;

  PetscCall(DMAdaptorRegister(DMADAPTORGRADIENT, DMAdaptorCreate_Gradient));
  PetscCall(DMAdaptorRegister(DMADAPTORFLUX, DMAdaptorCreate_Flux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorRegisterDestroy - This function destroys the registered `DMAdaptorType`. It is called from `PetscFinalize()`.

  Not collective

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMAdaptorRegisterAll()`, `DMAdaptorType`, `PetscFinalize()`
@*/
PetscErrorCode DMAdaptorRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMAdaptorList));
  DMAdaptorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorMonitorMakeKey_Internal(const char name[], PetscViewerType vtype, PetscViewerFormat format, char key[])
{
  PetscFunctionBegin;
  PetscCall(PetscStrncpy(key, name, PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, vtype, PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, PetscViewerFormats[format], PETSC_MAX_PATH_LEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorRegister -  Registers a mesh adaptation monitor routine that may be accessed with `DMAdaptorMonitorSetFromOptions()`

  Not Collective

  Input Parameters:
+ name    - name of a new monitor routine
. vtype   - A `PetscViewerType` for the output
. format  - A `PetscViewerFormat` for the output
. monitor - Monitor routine
. create  - Creation routine, or `NULL`
- destroy - Destruction routine, or `NULL`

  Level: advanced

  Note:
  `DMAdaptorMonitorRegister()` may be called multiple times to add several user-defined monitors.

  Example Usage:
.vb
  DMAdaptorMonitorRegister("my_monitor", PETSCVIEWERASCII, PETSC_VIEWER_ASCII_INFO_DETAIL, MyMonitor, NULL, NULL);
.ve

  Then, your monitor can be chosen with the procedural interface via
.vb
  DMAdaptorMonitorSetFromOptions(ksp, "-adaptor_monitor_my_monitor", "my_monitor", NULL)
.ve
  or at runtime via the option `-adaptor_monitor_my_monitor`

.seealso: [](ch_snes), `DMAdaptor`, `DMAdaptorMonitorSet()`, `DMAdaptorMonitorRegisterAll()`, `DMAdaptorMonitorSetFromOptions()`
@*/
PetscErrorCode DMAdaptorMonitorRegister(const char name[], PetscViewerType vtype, PetscViewerFormat format, PetscErrorCode (*monitor)(DMAdaptor, PetscInt, DM, DM, PetscInt, PetscReal[], Vec, PetscViewerAndFormat *), PetscErrorCode (*create)(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **), PetscErrorCode (*destroy)(PetscViewerAndFormat **))
{
  char key[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(SNESInitializePackage());
  PetscCall(DMAdaptorMonitorMakeKey_Internal(name, vtype, format, key));
  PetscCall(PetscFunctionListAdd(&DMAdaptorMonitorList, key, monitor));
  if (create) PetscCall(PetscFunctionListAdd(&DMAdaptorMonitorCreateList, key, create));
  if (destroy) PetscCall(PetscFunctionListAdd(&DMAdaptorMonitorDestroyList, key, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorRegisterDestroy - This function destroys the registered monitors for `DMAdaptor`. It is called from `PetscFinalize()`.

  Not collective

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMAdaptorMonitorRegisterAll()`, `DMAdaptor`, `PetscFinalize()`
@*/
PetscErrorCode DMAdaptorMonitorRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMAdaptorMonitorList));
  PetscCall(PetscFunctionListDestroy(&DMAdaptorMonitorCreateList));
  PetscCall(PetscFunctionListDestroy(&DMAdaptorMonitorDestroyList));
  DMAdaptorMonitorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorCreate - Create a `DMAdaptor` object. Its purpose is to construct a adaptation `DMLabel` or metric `Vec` that can be used to modify the `DM`.

  Collective

  Input Parameter:
. comm - The communicator for the `DMAdaptor` object

  Output Parameter:
. adaptor - The `DMAdaptor` object

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMAdaptor`, `DMAdaptorDestroy()`, `DMAdaptorAdapt()`, `PetscConvEst`, `PetscConvEstCreate()`
@*/
PetscErrorCode DMAdaptorCreate(MPI_Comm comm, DMAdaptor *adaptor)
{
  VecTaggerBox refineBox, coarsenBox;

  PetscFunctionBegin;
  PetscAssertPointer(adaptor, 2);
  PetscCall(PetscSysInitializePackage());

  PetscCall(PetscHeaderCreate(*adaptor, DMADAPTOR_CLASSID, "DMAdaptor", "DM Adaptor", "DMAdaptor", comm, DMAdaptorDestroy, DMAdaptorView));
  (*adaptor)->adaptCriterion   = DM_ADAPTATION_NONE;
  (*adaptor)->numSeq           = 1;
  (*adaptor)->Nadapt           = -1;
  (*adaptor)->refinementFactor = 2.0;
  refineBox.min = refineBox.max = PETSC_MAX_REAL;
  PetscCall(VecTaggerCreate(PetscObjectComm((PetscObject)*adaptor), &(*adaptor)->refineTag));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*adaptor)->refineTag, "refine_"));
  PetscCall(VecTaggerSetType((*adaptor)->refineTag, VECTAGGERABSOLUTE));
  PetscCall(VecTaggerAbsoluteSetBox((*adaptor)->refineTag, &refineBox));
  coarsenBox.min = coarsenBox.max = PETSC_MAX_REAL;
  PetscCall(VecTaggerCreate(PetscObjectComm((PetscObject)*adaptor), &(*adaptor)->coarsenTag));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*adaptor)->coarsenTag, "coarsen_"));
  PetscCall(VecTaggerSetType((*adaptor)->coarsenTag, VECTAGGERABSOLUTE));
  PetscCall(VecTaggerAbsoluteSetBox((*adaptor)->coarsenTag, &coarsenBox));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorDestroy - Destroys a `DMAdaptor` object

  Collective

  Input Parameter:
. adaptor - The `DMAdaptor` object

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMAdaptor`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorDestroy(DMAdaptor *adaptor)
{
  PetscFunctionBegin;
  if (!*adaptor) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*adaptor, DMADAPTOR_CLASSID, 1);
  if (--((PetscObject)*adaptor)->refct > 0) {
    *adaptor = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecTaggerDestroy(&(*adaptor)->refineTag));
  PetscCall(VecTaggerDestroy(&(*adaptor)->coarsenTag));
  PetscCall(PetscFree2((*adaptor)->exactSol, (*adaptor)->exactCtx));
  PetscCall(DMAdaptorMonitorCancel(*adaptor));
  PetscCall(PetscHeaderDestroy(adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorSetType - Sets the particular implementation for a adaptor.

  Collective

  Input Parameters:
+ adaptor - The `DMAdaptor`
- method  - The name of the adaptor type

  Options Database Key:
. -adaptor_type <type> - Sets the adaptor type; see `DMAdaptorType`

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMAdaptor`, `DMAdaptorType`, `DMAdaptorGetType()`, `DMAdaptorCreate()`
@*/
PetscErrorCode DMAdaptorSetType(DMAdaptor adaptor, DMAdaptorType method)
{
  PetscErrorCode (*r)(DMAdaptor);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)adaptor, method, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMAdaptorRegisterAll());
  PetscCall(PetscFunctionListFind(DMAdaptorList, method, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)adaptor), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DMAdaptor type: %s", method);

  PetscTryTypeMethod(adaptor, destroy);
  PetscCall(PetscMemzero(adaptor->ops, sizeof(*adaptor->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)adaptor, method));
  PetscCall((*r)(adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorGetType - Gets the type name (as a string) from the adaptor.

  Not Collective

  Input Parameter:
. adaptor - The `DMAdaptor`

  Output Parameter:
. type - The `DMAdaptorType` name

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMAdaptor`, `DMAdaptorType`, `DMAdaptorSetType()`, `DMAdaptorCreate()`
@*/
PetscErrorCode DMAdaptorGetType(DMAdaptor adaptor, DMAdaptorType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(DMAdaptorRegisterAll());
  *type = ((PetscObject)adaptor)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerAndFormatCreate_Internal(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorSet - Sets an ADDITIONAL function to be called at every iteration to monitor
  the error etc.

  Logically Collective

  Input Parameters:
+ adaptor        - the `DMAdaptor`
. monitor        - pointer to function (if this is `NULL`, it turns off monitoring
. ctx            - [optional] context for private data for the monitor routine (use `NULL` if no context is needed)
- monitordestroy - [optional] routine that frees monitor context (may be `NULL`), see `PetscCtxDestroyFn` for its calling sequence

  Calling sequence of `monitor`:
+ adaptor - the `DMAdaptor`
. it      - iteration number
. odm     - the original `DM`
. adm     - the adapted `DM`
. Nf      - number of fields
. enorms  - (estimated) 2-norm of the error for each field
. error   - `Vec` of cellwise errors
- ctx     - optional monitoring context, as set by `DMAdaptorMonitorSet()`

  Options Database Keys:
+ -adaptor_monitor_size                - sets `DMAdaptorMonitorSize()`
. -adaptor_monitor_error               - sets `DMAdaptorMonitorError()`
. -adaptor_monitor_error draw          - sets `DMAdaptorMonitorErrorDraw()` and plots error
. -adaptor_monitor_error draw::draw_lg - sets `DMAdaptorMonitorErrorDrawLG()` and plots error
- -dm_adaptor_monitor_cancel           - Cancels all monitors that have been hardwired into a code by calls to `DMAdaptorMonitorSet()`, but does not cancel those set via the options database.

  Level: beginner

.seealso: [](ch_snes), `DMAdaptorMonitorError()`, `DMAdaptor`, `PetscCtxDestroyFn`
@*/
PetscErrorCode DMAdaptorMonitorSet(DMAdaptor adaptor, PetscErrorCode (*monitor)(DMAdaptor adaptor, PetscInt it, DM odm, DM adm, PetscInt Nf, PetscReal enorms[], Vec error, void *ctx), void *ctx, PetscCtxDestroyFn *monitordestroy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  for (PetscInt i = 0; i < adaptor->numbermonitors; i++) {
    PetscBool identical;

    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))(PetscVoidFn *)monitor, ctx, monitordestroy, (PetscErrorCode (*)(void))(PetscVoidFn *)adaptor->monitor[i], adaptor->monitorcontext[i], adaptor->monitordestroy[i], &identical));
    if (identical) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(adaptor->numbermonitors < MAXDMADAPTORMONITORS, PetscObjectComm((PetscObject)adaptor), PETSC_ERR_ARG_OUTOFRANGE, "Too many DMAdaptor monitors set");
  adaptor->monitor[adaptor->numbermonitors]          = monitor;
  adaptor->monitordestroy[adaptor->numbermonitors]   = monitordestroy;
  adaptor->monitorcontext[adaptor->numbermonitors++] = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorMonitorCancel - Clears all monitors for a `DMAdaptor` object.

  Logically Collective

  Input Parameter:
. adaptor - the `DMAdaptor`

  Options Database Key:
. -dm_adaptor_monitor_cancel - Cancels all monitors that have been hardwired into a code by calls to `DMAdaptorMonitorSet()`, but does not cancel those set via the options database.

  Level: intermediate

.seealso: [](ch_snes), `DMAdaptorMonitorError()`, `DMAdaptorMonitorSet()`, `DMAdaptor`
@*/
PetscErrorCode DMAdaptorMonitorCancel(DMAdaptor adaptor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  for (PetscInt i = 0; i < adaptor->numbermonitors; ++i) {
    if (adaptor->monitordestroy[i]) PetscCall((*adaptor->monitordestroy[i])(&adaptor->monitorcontext[i]));
  }
  adaptor->numbermonitors = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user in the options database

  Collective

  Input Parameters:
+ adaptor - `DMadaptor` object you wish to monitor
. opt     - the command line option for this monitor
. name    - the monitor type one is seeking
- ctx     - An optional user context for the monitor, or `NULL`

  Level: developer

.seealso: [](ch_snes), `DMAdaptorMonitorRegister()`, `DMAdaptorMonitorSet()`, `PetscOptionsGetViewer()`
@*/
PetscErrorCode DMAdaptorMonitorSetFromOptions(DMAdaptor adaptor, const char opt[], const char name[], void *ctx)
{
  PetscErrorCode (*mfunc)(DMAdaptor, PetscInt, DM, DM, PetscInt, PetscReal[], Vec, void *);
  PetscErrorCode (*cfunc)(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **);
  PetscErrorCode (*dfunc)(PetscViewerAndFormat **);
  PetscViewerAndFormat *vf;
  PetscViewer           viewer;
  PetscViewerFormat     format;
  PetscViewerType       vtype;
  char                  key[PETSC_MAX_PATH_LEN];
  PetscBool             flg;
  const char           *prefix = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)adaptor, &prefix));
  PetscCall(PetscOptionsCreateViewer(PetscObjectComm((PetscObject)adaptor), ((PetscObject)adaptor)->options, prefix, opt, &viewer, &format, &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscViewerGetType(viewer, &vtype));
  PetscCall(DMAdaptorMonitorMakeKey_Internal(name, vtype, format, key));
  PetscCall(PetscFunctionListFind(DMAdaptorMonitorList, key, &mfunc));
  PetscCall(PetscFunctionListFind(DMAdaptorMonitorCreateList, key, &cfunc));
  PetscCall(PetscFunctionListFind(DMAdaptorMonitorDestroyList, key, &dfunc));
  if (!cfunc) cfunc = PetscViewerAndFormatCreate_Internal;
  if (!dfunc) dfunc = PetscViewerAndFormatDestroy;

  PetscCall((*cfunc)(viewer, format, ctx, &vf));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMAdaptorMonitorSet(adaptor, mfunc, vf, (PetscCtxDestroyFn *)dfunc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorSetOptionsPrefix - Sets the prefix used for searching for all `DMAdaptor` options in the database.

  Logically Collective

  Input Parameters:
+ adaptor - the `DMAdaptor`
- prefix  - the prefix to prepend to all option names

  Level: advanced

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: [](ch_snes), `DMAdaptor`, `SNESSetOptionsPrefix()`, `DMAdaptorSetFromOptions()`
@*/
PetscErrorCode DMAdaptorSetOptionsPrefix(DMAdaptor adaptor, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)adaptor, prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)adaptor->refineTag, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)adaptor->refineTag, "refine_"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)adaptor->coarsenTag, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)adaptor->coarsenTag, "coarsen_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorSetFromOptions - Sets properties of a `DMAdaptor` object from values in the options database

  Collective

  Input Parameter:
. adaptor - The `DMAdaptor` object

  Options Database Keys:
+ -adaptor_monitor_size                - Monitor the mesh size
. -adaptor_monitor_error               - Monitor the solution error
. -adaptor_sequence_num <num>          - Number of adaptations to generate an optimal grid
. -adaptor_target_num <num>            - Set the target number of vertices N_adapt, -1 for automatic determination
. -adaptor_refinement_factor <r>       - Set r such that N_adapt = r^dim N_orig
- -adaptor_mixed_setup_function <func> - Set the function func that sets up the mixed problem

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMAdaptor`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorSetFromOptions(DMAdaptor adaptor)
{
  char                  typeName[PETSC_MAX_PATH_LEN];
  const char           *defName = DMADAPTORGRADIENT;
  char                  funcname[PETSC_MAX_PATH_LEN];
  DMAdaptationCriterion criterion = DM_ADAPTATION_NONE;
  PetscBool             flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)adaptor);
  PetscCall(PetscOptionsFList("-adaptor_type", "DMAdaptor", "DMAdaptorSetType", DMAdaptorList, defName, typeName, 1024, &flg));
  if (flg) PetscCall(DMAdaptorSetType(adaptor, typeName));
  else if (!((PetscObject)adaptor)->type_name) PetscCall(DMAdaptorSetType(adaptor, defName));
  PetscCall(PetscOptionsEnum("-adaptor_criterion", "Criterion used to drive adaptation", "", DMAdaptationCriteria, (PetscEnum)criterion, (PetscEnum *)&criterion, &flg));
  if (flg) PetscCall(DMAdaptorSetCriterion(adaptor, criterion));
  PetscCall(PetscOptionsInt("-adaptor_sequence_num", "Number of adaptations to generate an optimal grid", "DMAdaptorSetSequenceLength", adaptor->numSeq, &adaptor->numSeq, NULL));
  PetscCall(PetscOptionsInt("-adaptor_target_num", "Set the target number of vertices N_adapt, -1 for automatic determination", "DMAdaptor", adaptor->Nadapt, &adaptor->Nadapt, NULL));
  PetscCall(PetscOptionsReal("-adaptor_refinement_factor", "Set r such that N_adapt = r^dim N_orig", "DMAdaptor", adaptor->refinementFactor, &adaptor->refinementFactor, NULL));
  PetscCall(PetscOptionsString("-adaptor_mixed_setup_function", "Function to setup the mixed problem", "DMAdaptorSetMixedSetupFunction", funcname, funcname, sizeof(funcname), &flg));
  if (flg) {
    PetscErrorCode (*setupFunc)(DMAdaptor, DM);

    PetscCall(PetscDLSym(NULL, funcname, (void **)&setupFunc));
    PetscCheck(setupFunc, PetscObjectComm((PetscObject)adaptor), PETSC_ERR_ARG_WRONG, "Could not locate function %s", funcname);
    PetscCall(DMAdaptorSetMixedSetupFunction(adaptor, setupFunc));
  }
  PetscCall(DMAdaptorMonitorSetFromOptions(adaptor, "-adaptor_monitor_size", "size", adaptor));
  PetscCall(DMAdaptorMonitorSetFromOptions(adaptor, "-adaptor_monitor_error", "error", adaptor));
  PetscOptionsEnd();
  PetscCall(VecTaggerSetFromOptions(adaptor->refineTag));
  PetscCall(VecTaggerSetFromOptions(adaptor->coarsenTag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorView - Views a `DMAdaptor` object

  Collective

  Input Parameters:
+ adaptor - The `DMAdaptor` object
- viewer  - The `PetscViewer` object

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMAdaptor`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorView(DMAdaptor adaptor, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)adaptor, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "DM Adaptor\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  sequence length: %" PetscInt_FMT "\n", adaptor->numSeq));
  PetscCall(VecTaggerView(adaptor->refineTag, viewer));
  PetscCall(VecTaggerView(adaptor->coarsenTag, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorGetSolver - Gets the solver used to produce discrete solutions

  Not Collective

  Input Parameter:
. adaptor - The `DMAdaptor` object

  Output Parameter:
. snes - The solver

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMAdaptor`, `DMAdaptorSetSolver()`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorGetSolver(DMAdaptor adaptor, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscAssertPointer(snes, 2);
  *snes = adaptor->snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorSetSolver - Sets the solver used to produce discrete solutions

  Not Collective

  Input Parameters:
+ adaptor - The `DMAdaptor` object
- snes    - The solver, this MUST have an attached `DM`/`PetscDS`, so that the exact solution can be computed

  Level: intermediate

.seealso: [](ch_dmbase), `DMAdaptor`, `DMAdaptorGetSolver()`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorSetSolver(DMAdaptor adaptor, SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 2);
  adaptor->snes = snes;
  PetscCall(SNESGetDM(adaptor->snes, &adaptor->idm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorGetSequenceLength - Gets the number of sequential adaptations used by an adapter

  Not Collective

  Input Parameter:
. adaptor - The `DMAdaptor` object

  Output Parameter:
. num - The number of adaptations

  Level: intermediate

.seealso: [](ch_dmbase), `DMAdaptor`, `DMAdaptorSetSequenceLength()`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorGetSequenceLength(DMAdaptor adaptor, PetscInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscAssertPointer(num, 2);
  *num = adaptor->numSeq;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorSetSequenceLength - Sets the number of sequential adaptations

  Not Collective

  Input Parameters:
+ adaptor - The `DMAdaptor` object
- num     - The number of adaptations

  Level: intermediate

.seealso: [](ch_dmbase), `DMAdaptorGetSequenceLength()`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorSetSequenceLength(DMAdaptor adaptor, PetscInt num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  adaptor->numSeq = num;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorTransferSolution_Exact_Private(DMAdaptor adaptor, DM dm, Vec u, DM adm, Vec au, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(DMProjectFunction(adm, 0.0, adaptor->exactSol, adaptor->exactCtx, INSERT_ALL_VALUES, au));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorSetUp - After the solver is specified, creates data structures for controlling adaptivity

  Collective

  Input Parameter:
. adaptor - The `DMAdaptor` object

  Level: beginner

.seealso: [](ch_dmbase), `DMAdaptor`, `DMAdaptorCreate()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorSetUp(DMAdaptor adaptor)
{
  PetscDS  prob;
  PetscInt Nf, f;

  PetscFunctionBegin;
  PetscCall(DMGetDS(adaptor->idm, &prob));
  PetscCall(VecTaggerSetUp(adaptor->refineTag));
  PetscCall(VecTaggerSetUp(adaptor->coarsenTag));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscMalloc2(Nf, &adaptor->exactSol, Nf, &adaptor->exactCtx));
  for (f = 0; f < Nf; ++f) {
    PetscCall(PetscDSGetExactSolution(prob, f, &adaptor->exactSol[f], &adaptor->exactCtx[f]));
    /* TODO Have a flag that forces projection rather than using the exact solution */
    if (adaptor->exactSol[0]) PetscCall(DMAdaptorSetTransferFunction(adaptor, DMAdaptorTransferSolution_Exact_Private));
  }
  PetscTryTypeMethod(adaptor, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMAdaptorGetTransferFunction(DMAdaptor adaptor, PetscErrorCode (**tfunc)(DMAdaptor, DM, Vec, DM, Vec, void *))
{
  PetscFunctionBegin;
  *tfunc = adaptor->ops->transfersolution;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMAdaptorSetTransferFunction(DMAdaptor adaptor, PetscErrorCode (*tfunc)(DMAdaptor, DM, Vec, DM, Vec, void *))
{
  PetscFunctionBegin;
  adaptor->ops->transfersolution = tfunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorPreAdapt(DMAdaptor adaptor, Vec locX)
{
  DM           plex;
  PetscDS      prob;
  PetscObject  obj;
  PetscClassId id;
  PetscBool    isForest;

  PetscFunctionBegin;
  PetscCall(DMConvert(adaptor->idm, DMPLEX, &plex));
  PetscCall(DMGetDS(adaptor->idm, &prob));
  PetscCall(PetscDSGetDiscretization(prob, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));
  PetscCall(DMIsForest(adaptor->idm, &isForest));
  if (adaptor->adaptCriterion == DM_ADAPTATION_NONE) {
    if (isForest) adaptor->adaptCriterion = DM_ADAPTATION_LABEL;
#if defined(PETSC_HAVE_PRAGMATIC)
    else {
      adaptor->adaptCriterion = DM_ADAPTATION_METRIC;
    }
#elif defined(PETSC_HAVE_MMG)
    else {
      adaptor->adaptCriterion = DM_ADAPTATION_METRIC;
    }
#elif defined(PETSC_HAVE_PARMMG)
    else {
      adaptor->adaptCriterion = DM_ADAPTATION_METRIC;
    }
#else
    else {
      adaptor->adaptCriterion = DM_ADAPTATION_LABEL;
    }
#endif
  }
  if (id == PETSCFV_CLASSID) {
    adaptor->femType = PETSC_FALSE;
  } else {
    adaptor->femType = PETSC_TRUE;
  }
  if (adaptor->femType) {
    /* Compute local solution bc */
    PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, adaptor->faceGeom, adaptor->cellGeom, NULL));
  } else {
    PetscFV      fvm = (PetscFV)obj;
    PetscLimiter noneLimiter;
    Vec          grad;

    PetscCall(PetscFVGetComputeGradients(fvm, &adaptor->computeGradient));
    PetscCall(PetscFVSetComputeGradients(fvm, PETSC_TRUE));
    /* Use no limiting when reconstructing gradients for adaptivity */
    PetscCall(PetscFVGetLimiter(fvm, &adaptor->limiter));
    PetscCall(PetscObjectReference((PetscObject)adaptor->limiter));
    PetscCall(PetscLimiterCreate(PetscObjectComm((PetscObject)fvm), &noneLimiter));
    PetscCall(PetscLimiterSetType(noneLimiter, PETSCLIMITERNONE));
    PetscCall(PetscFVSetLimiter(fvm, noneLimiter));
    /* Get FVM data */
    PetscCall(DMPlexGetDataFVM(plex, fvm, &adaptor->cellGeom, &adaptor->faceGeom, &adaptor->gradDM));
    PetscCall(VecGetDM(adaptor->cellGeom, &adaptor->cellDM));
    PetscCall(VecGetArrayRead(adaptor->cellGeom, &adaptor->cellGeomArray));
    /* Compute local solution bc */
    PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, adaptor->faceGeom, adaptor->cellGeom, NULL));
    /* Compute gradients */
    PetscCall(DMCreateGlobalVector(adaptor->gradDM, &grad));
    PetscCall(DMPlexReconstructGradientsFVM(plex, locX, grad));
    PetscCall(DMGetLocalVector(adaptor->gradDM, &adaptor->cellGrad));
    PetscCall(DMGlobalToLocalBegin(adaptor->gradDM, grad, INSERT_VALUES, adaptor->cellGrad));
    PetscCall(DMGlobalToLocalEnd(adaptor->gradDM, grad, INSERT_VALUES, adaptor->cellGrad));
    PetscCall(VecDestroy(&grad));
    PetscCall(VecGetArrayRead(adaptor->cellGrad, &adaptor->cellGradArray));
  }
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorTransferSolution(DMAdaptor adaptor, DM dm, Vec x, DM adm, Vec ax)
{
  PetscReal time = 0.0;
  Mat       interp;
  void     *ctx;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &ctx));
  if (adaptor->ops->transfersolution) PetscUseTypeMethod(adaptor, transfersolution, dm, x, adm, ax, ctx);
  else {
    switch (adaptor->adaptCriterion) {
    case DM_ADAPTATION_LABEL:
      PetscCall(DMForestTransferVec(dm, x, adm, ax, PETSC_TRUE, time));
      break;
    case DM_ADAPTATION_REFINE:
    case DM_ADAPTATION_METRIC:
      PetscCall(DMCreateInterpolation(dm, adm, &interp, NULL));
      PetscCall(MatInterpolate(interp, x, ax));
      PetscCall(DMInterpolate(dm, interp, adm));
      PetscCall(MatDestroy(&interp));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)adaptor), PETSC_ERR_SUP, "No built-in projection for this adaptation criterion: %d", adaptor->adaptCriterion);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorPostAdapt(DMAdaptor adaptor)
{
  PetscDS      prob;
  PetscObject  obj;
  PetscClassId id;

  PetscFunctionBegin;
  PetscCall(DMGetDS(adaptor->idm, &prob));
  PetscCall(PetscDSGetDiscretization(prob, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));
  if (id == PETSCFV_CLASSID) {
    PetscFV fvm = (PetscFV)obj;

    PetscCall(PetscFVSetComputeGradients(fvm, adaptor->computeGradient));
    /* Restore original limiter */
    PetscCall(PetscFVSetLimiter(fvm, adaptor->limiter));

    PetscCall(VecRestoreArrayRead(adaptor->cellGeom, &adaptor->cellGeomArray));
    PetscCall(VecRestoreArrayRead(adaptor->cellGrad, &adaptor->cellGradArray));
    PetscCall(DMRestoreLocalVector(adaptor->gradDM, &adaptor->cellGrad));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMAdaptorComputeCellErrorIndicator_Gradient - Use the integrated gradient as an error indicator in the `DMAdaptor`

  Input Parameters:
+ adaptor  - The `DMAdaptor` object
. dim      - The topological dimension
. cell     - The cell
. field    - The field integrated over the cell
. gradient - The gradient integrated over the cell
. cg       - A `PetscFVCellGeom` struct
- ctx      - A user context

  Output Parameter:
. errInd   - The error indicator

  Developer Note:
  Some of the input arguments are absurdly specialized to special situations, it is not clear this is a good general API

.seealso: [](ch_dmbase), `DMAdaptor`
*/
static PetscErrorCode DMAdaptorComputeCellErrorIndicator_Gradient(DMAdaptor adaptor, PetscInt dim, PetscInt Nc, const PetscScalar *field, const PetscScalar *gradient, const PetscFVCellGeom *cg, PetscReal *errInd, void *ctx)
{
  PetscReal err = 0.;
  PetscInt  c, d;

  PetscFunctionBeginHot;
  for (c = 0; c < Nc; c++) {
    for (d = 0; d < dim; ++d) err += PetscSqr(PetscRealPart(gradient[c * dim + d]));
  }
  *errInd = cg->volume * err;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorComputeErrorIndicator_Gradient(DMAdaptor adaptor, Vec locX, Vec errVec)
{
  DM              dm, plex, edm, eplex;
  PetscDS         ds;
  PetscObject     obj;
  PetscClassId    id;
  void           *ctx;
  PetscQuadrature quad;
  PetscScalar    *earray;
  PetscReal       minMaxInd[2] = {PETSC_MAX_REAL, PETSC_MIN_REAL}, minMaxIndGlobal[2];
  PetscInt        dim, cdim, cStart, cEnd, Nf, Nc;

  PetscFunctionBegin;
  PetscCall(VecGetDM(locX, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(VecGetDM(errVec, &edm));
  PetscCall(DMConvert(edm, DMPLEX, &eplex));
  PetscCall(DMGetDimension(plex, &dim));
  PetscCall(DMGetCoordinateDim(plex, &cdim));
  PetscCall(DMGetApplicationContext(plex, &ctx));
  PetscCall(DMGetDS(plex, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetDiscretization(ds, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));

  PetscCall(VecGetArray(errVec, &earray));
  PetscCall(DMPlexGetSimplexOrBoxCells(plex, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscScalar *eval;
    PetscReal    errInd = 0.;

    if (id == PETSCFV_CLASSID) {
      PetscFV            fv = (PetscFV)obj;
      const PetscScalar *pointSols;
      const PetscScalar *pointSol;
      const PetscScalar *pointGrad;
      PetscFVCellGeom   *cg;

      PetscCall(PetscFVGetNumComponents(fv, &Nc));
      PetscCall(VecGetArrayRead(locX, &pointSols));
      PetscCall(DMPlexPointLocalRead(plex, cell, pointSols, (void *)&pointSol));
      PetscCall(DMPlexPointLocalRead(adaptor->gradDM, cell, adaptor->cellGradArray, (void *)&pointGrad));
      PetscCall(DMPlexPointLocalRead(adaptor->cellDM, cell, adaptor->cellGeomArray, &cg));
      PetscUseTypeMethod(adaptor, computecellerrorindicator, dim, Nc, pointSol, pointGrad, cg, &errInd, ctx);
      PetscCall(VecRestoreArrayRead(locX, &pointSols));
    } else {
      PetscFE          fe = (PetscFE)obj;
      PetscScalar     *x  = NULL, *field, *gradient, *interpolant, *interpolantGrad;
      PetscFVCellGeom  cg;
      PetscFEGeom      fegeom;
      const PetscReal *quadWeights;
      PetscReal       *coords;
      PetscInt         Nb, Nq, qNc;

      fegeom.dim      = dim;
      fegeom.dimEmbed = cdim;
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
      PetscCall(PetscFEGetQuadrature(fe, &quad));
      PetscCall(PetscFEGetDimension(fe, &Nb));
      PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights));
      PetscCall(PetscCalloc6(Nc, &field, cdim * Nc, &gradient, cdim * Nq, &coords, Nq, &fegeom.detJ, cdim * cdim * Nq, &fegeom.J, cdim * cdim * Nq, &fegeom.invJ));
      PetscCall(PetscMalloc2(Nc, &interpolant, cdim * Nc, &interpolantGrad));
      PetscCall(DMPlexComputeCellGeometryFEM(plex, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      PetscCall(DMPlexComputeCellGeometryFVM(plex, cell, &cg.volume, NULL, NULL));
      PetscCall(PetscArrayzero(gradient, cdim * Nc));
      PetscCall(DMPlexVecGetClosure(plex, NULL, locX, cell, NULL, &x));
      for (PetscInt f = 0; f < Nf; ++f) {
        PetscInt qc = 0;

        PetscCall(PetscDSGetDiscretization(ds, f, &obj));
        PetscCall(PetscArrayzero(interpolant, Nc));
        PetscCall(PetscArrayzero(interpolantGrad, cdim * Nc));
        for (PetscInt q = 0; q < Nq; ++q) {
          PetscCall(PetscFEInterpolateFieldAndGradient_Static((PetscFE)obj, 1, x, &fegeom, q, interpolant, interpolantGrad));
          for (PetscInt fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q * qNc + qc + fc];

            field[fc] += interpolant[fc] * wt * fegeom.detJ[q];
            for (PetscInt d = 0; d < cdim; ++d) gradient[fc * cdim + d] += interpolantGrad[fc * dim + d] * wt * fegeom.detJ[q];
          }
        }
        qc += Nc;
      }
      PetscCall(PetscFree2(interpolant, interpolantGrad));
      PetscCall(DMPlexVecRestoreClosure(plex, NULL, locX, cell, NULL, &x));
      for (PetscInt fc = 0; fc < Nc; ++fc) {
        field[fc] /= cg.volume;
        for (PetscInt d = 0; d < cdim; ++d) gradient[fc * cdim + d] /= cg.volume;
      }
      PetscUseTypeMethod(adaptor, computecellerrorindicator, dim, Nc, field, gradient, &cg, &errInd, ctx);
      PetscCall(PetscFree6(field, gradient, coords, fegeom.detJ, fegeom.J, fegeom.invJ));
    }
    PetscCall(DMPlexPointGlobalRef(eplex, cell, earray, (void *)&eval));
    eval[0]      = errInd;
    minMaxInd[0] = PetscMin(minMaxInd[0], errInd);
    minMaxInd[1] = PetscMax(minMaxInd[1], errInd);
  }
  PetscCall(VecRestoreArray(errVec, &earray));
  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&eplex));
  PetscCall(PetscGlobalMinMaxReal(PetscObjectComm((PetscObject)adaptor), minMaxInd, minMaxIndGlobal));
  PetscCall(PetscInfo(adaptor, "DMAdaptor: error indicator range (%g, %g)\n", (double)minMaxIndGlobal[0], (double)minMaxIndGlobal[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorComputeErrorIndicator_Flux(DMAdaptor adaptor, Vec lu, Vec errVec)
{
  DM          dm, mdm;
  SNES        msnes;
  Vec         mu, lmu;
  void       *ctx;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(VecGetDM(lu, &dm));

  // Set up and solve mixed problem
  PetscCall(DMClone(dm, &mdm));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)mdm), &msnes));
  PetscCall(SNESSetDM(msnes, mdm));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)adaptor, &prefix));
  PetscCall(SNESSetOptionsPrefix(msnes, prefix));
  PetscCall(SNESAppendOptionsPrefix(msnes, "mixed_"));

  PetscTryTypeMethod(adaptor, mixedsetup, mdm);
  PetscCall(DMGetApplicationContext(dm, &ctx));
  PetscCall(DMPlexSetSNESLocalFEM(mdm, PETSC_FALSE, ctx));
  PetscCall(SNESSetFromOptions(msnes));

  PetscCall(DMCreateGlobalVector(mdm, &mu));
  PetscCall(PetscObjectSetName((PetscObject)mu, "Mixed Solution"));
  PetscCall(VecSet(mu, 0.0));
  PetscCall(SNESSolve(msnes, NULL, mu));
  PetscCall(VecViewFromOptions(mu, (PetscObject)adaptor, "-adapt_mixed_sol_vec_view"));

  PetscCall(DMGetLocalVector(mdm, &lmu));
  PetscCall(DMGlobalToLocal(mdm, mu, INSERT_VALUES, lmu));
  PetscCall(DMPlexInsertBoundaryValues(mdm, PETSC_TRUE, lmu, 0.0, NULL, NULL, NULL));
  PetscCall(DMPlexComputeL2FluxDiffVecLocal(lu, 0, lmu, 0, errVec));
  PetscCall(DMRestoreLocalVector(mdm, &lmu));
  PetscCall(VecDestroy(&mu));
  PetscCall(SNESDestroy(&msnes));
  PetscCall(DMDestroy(&mdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorMonitor - runs the user provided monitor routines, if they exist

  Collective

  Input Parameters:
+ adaptor - the `DMAdaptor`
. it      - iteration number
. odm     - the original `DM`
. adm     - the adapted `DM`
. Nf      - the number of fields
. enorms  - the 2-norm error values for each field
- error   - `Vec` of cellwise errors

  Level: developer

  Note:
  This routine is called by the `DMAdaptor` implementations.
  It does not typically need to be called by the user.

.seealso: [](ch_snes), `DMAdaptorMonitorSet()`
@*/
PetscErrorCode DMAdaptorMonitor(DMAdaptor adaptor, PetscInt it, DM odm, DM adm, PetscInt Nf, PetscReal enorms[], Vec error)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < adaptor->numbermonitors; ++i) PetscCall((*adaptor->monitor[i])(adaptor, it, odm, adm, Nf, enorms, error, adaptor->monitorcontext[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorSize - Prints the mesh sizes at each iteration of an adaptation loop.

  Collective

  Input Parameters:
+ adaptor - the `DMAdaptor`
. n       - iteration number
. odm     - the original `DM`
. adm     - the adapted `DM`
. Nf      - number of fields
. enorms  - 2-norm error values for each field (may be estimated).
. error   - `Vec` of cellwise errors
- vf      - The viewer context

  Options Database Key:
. -adaptor_monitor_size - Activates `DMAdaptorMonitorSize()`

  Level: intermediate

  Note:
  This is not called directly by users, rather one calls `DMAdaptorMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the adaptation loop.

.seealso: [](ch_snes), `DMAdaptor`, `DMAdaptorMonitorSet()`, `DMAdaptorMonitorError()`, `DMAdaptorMonitorErrorDraw()`, `DMAdaptorMonitorErrorDrawLG()`
@*/
PetscErrorCode DMAdaptorMonitorSize(DMAdaptor adaptor, PetscInt n, DM odm, DM adm, PetscInt Nf, PetscReal enorms[], Vec error, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscInt          tablevel, cStart, cEnd, acStart, acEnd;
  const char       *prefix;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 8);
  PetscCall(PetscObjectGetTabLevel((PetscObject)adaptor, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)adaptor, &prefix));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)adaptor), &rank));
  PetscCall(DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(adm, 0, &acStart, &acEnd));

  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Sizes for %s adaptation.\n", prefix));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " DMAdaptor rank %d N_orig: %" PetscInt_FMT " N_adapt: %" PetscInt_FMT "\n", n, rank, cEnd - cStart, acEnd - acStart));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorError - Prints the error norm at each iteration of an adaptation loop.

  Collective

  Input Parameters:
+ adaptor - the `DMAdaptor`
. n       - iteration number
. odm     - the original `DM`
. adm     - the adapted `DM`
. Nf      - number of fields
. enorms  - 2-norm error values for each field (may be estimated).
. error   - `Vec` of cellwise errors
- vf      - The viewer context

  Options Database Key:
. -adaptor_monitor_error - Activates `DMAdaptorMonitorError()`

  Level: intermediate

  Note:
  This is not called directly by users, rather one calls `DMAdaptorMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the adaptation loop.

.seealso: [](ch_snes), `DMAdaptor`, `DMAdaptorMonitorSet()`, `DMAdaptorMonitorErrorDraw()`, `DMAdaptorMonitorErrorDrawLG()`
@*/
PetscErrorCode DMAdaptorMonitorError(DMAdaptor adaptor, PetscInt n, DM odm, DM adm, PetscInt Nf, PetscReal enorms[], Vec error, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscInt          tablevel, cStart, cEnd, acStart, acEnd;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 8);
  PetscCall(PetscObjectGetTabLevel((PetscObject)adaptor, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)adaptor, &prefix));

  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Error norms for %s adaptation.\n", prefix));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " DMAdaptor Error norm %s", n, Nf > 1 ? "[" : ""));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
  for (PetscInt f = 0; f < Nf; ++f) {
    if (f > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ", "));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%14.12e", (double)enorms[f]));
  }
  PetscCall(DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(adm, 0, &acStart, &acEnd));
  PetscCall(PetscViewerASCIIPrintf(viewer, " N: %" PetscInt_FMT " Nadapt: %" PetscInt_FMT "\n", cEnd - cStart, acEnd - acStart));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorErrorDraw - Plots the error at each iteration of an iterative solver.

  Collective

  Input Parameters:
+ adaptor - the `DMAdaptor`
. n       - iteration number
. odm     - the original `DM`
. adm     - the adapted `DM`
. Nf      - number of fields
. enorms  - 2-norm error values for each field (may be estimated).
. error   - `Vec` of cellwise errors
- vf      - The viewer context

  Options Database Key:
. -adaptor_monitor_error draw - Activates `DMAdaptorMonitorErrorDraw()`

  Level: intermediate

  Note:
  This is not called directly by users, rather one calls `DMAdaptorMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the adaptation loop.

.seealso: [](ch_snes), `PETSCVIEWERDRAW`, `DMAdaptor`, `DMAdaptorMonitorSet()`, `DMAdaptorMonitorErrorDrawLG()`
@*/
PetscErrorCode DMAdaptorMonitorErrorDraw(DMAdaptor adaptor, PetscInt n, DM odm, DM adm, PetscInt Nf, PetscReal enorms[], Vec error, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 8);
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscObjectSetName((PetscObject)error, "Error Estimator"));
  PetscCall(PetscObjectCompose((PetscObject)error, "__Vec_bc_zero__", (PetscObject)adaptor));
  PetscCall(VecView(error, viewer));
  PetscCall(PetscObjectCompose((PetscObject)error, "__Vec_bc_zero__", NULL));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorErrorDrawLGCreate - Creates the context for the error plotter `DMAdaptorMonitorErrorDrawLG()`

  Collective

  Input Parameters:
+ viewer - The `PetscViewer`
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf - The viewer context

  Level: intermediate

.seealso: [](ch_snes), `PETSCVIEWERDRAW`, `PetscViewerMonitorGLSetUp()`, `DMAdaptor`, `DMAdaptorMonitorSet()`, `DMAdaptorMonitorErrorDrawLG()`
@*/
PetscErrorCode DMAdaptorMonitorErrorDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  DMAdaptor adaptor = (DMAdaptor)ctx;
  char    **names;
  PetscInt  Nf;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(adaptor->idm, &Nf));
  PetscCall(PetscMalloc1(Nf + 1, &names));
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscObject disc;
    const char *fname;
    char        lname[PETSC_MAX_PATH_LEN];

    PetscCall(DMGetField(adaptor->idm, f, NULL, &disc));
    PetscCall(PetscObjectGetName(disc, &fname));
    PetscCall(PetscStrncpy(lname, fname, PETSC_MAX_PATH_LEN));
    PetscCall(PetscStrlcat(lname, " Error", PETSC_MAX_PATH_LEN));
    PetscCall(PetscStrallocpy(lname, &names[f]));
  }
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscCall(PetscViewerMonitorLGSetUp(viewer, NULL, NULL, "Log Error Norm", Nf, (const char **)names, PETSC_DECIDE, PETSC_DECIDE, 400, 300));
  for (PetscInt f = 0; f < Nf; ++f) PetscCall(PetscFree(names[f]));
  PetscCall(PetscFree(names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorErrorDrawLG - Plots the error norm at each iteration of an adaptive loop.

  Collective

  Input Parameters:
+ adaptor - the `DMAdaptor`
. n       - iteration number
. odm     - the original `DM`
. adm     - the adapted `DM`
. Nf      - number of fields
. enorms  - 2-norm error values for each field (may be estimated).
. error   - `Vec` of cellwise errors
- vf      - The viewer context, obtained via `DMAdaptorMonitorErrorDrawLGCreate()`

  Options Database Key:
. -adaptor_error draw::draw_lg - Activates `DMAdaptorMonitorErrorDrawLG()`

  Level: intermediate

  Notes:
  This is not called directly by users, rather one calls `DMAdaptorMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the adaptation loop.

  Call `DMAdaptorMonitorErrorDrawLGCreate()` to create the context needed for this monitor

.seealso: [](ch_snes), `PETSCVIEWERDRAW`, `DMAdaptor`, `DMAdaptorMonitorSet()`, `DMAdaptorMonitorErrorDraw()`, `DMAdaptorMonitorError()`,
          `DMAdaptorMonitorTrueResidualDrawLGCreate()`
@*/
PetscErrorCode DMAdaptorMonitorErrorDrawLG(DMAdaptor adaptor, PetscInt n, DM odm, DM adm, PetscInt Nf, PetscReal enorms[], Vec error, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscDrawLG       lg;
  PetscReal        *x, *e;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 8);
  PetscCall(PetscViewerDrawGetDrawLG(viewer, 0, &lg));
  PetscCall(PetscCalloc2(Nf, &x, Nf, &e));
  PetscCall(PetscViewerPushFormat(viewer, format));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  for (PetscInt f = 0; f < Nf; ++f) {
    x[f] = (PetscReal)n;
    e[f] = enorms[f] > 0.0 ? PetscLog10Real(enorms[f]) : -15.;
  }
  PetscCall(PetscDrawLGAddPoint(lg, x, e));
  PetscCall(PetscDrawLGDraw(lg));
  PetscCall(PetscDrawLGSave(lg));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscFree2(x, e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorMonitorRegisterAll - Registers all of the mesh adaptation monitors in the `SNES` package.

  Not Collective

  Level: advanced

.seealso: [](ch_snes), `SNES`, `DM`, `DMAdaptorMonitorRegister()`, `DMAdaptorRegister()`
@*/
PetscErrorCode DMAdaptorMonitorRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMAdaptorMonitorRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  DMAdaptorMonitorRegisterAllCalled = PETSC_TRUE;

  PetscCall(DMAdaptorMonitorRegister("size", PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, DMAdaptorMonitorSize, NULL, NULL));
  PetscCall(DMAdaptorMonitorRegister("error", PETSCVIEWERASCII, PETSC_VIEWER_DEFAULT, DMAdaptorMonitorError, NULL, NULL));
  PetscCall(DMAdaptorMonitorRegister("error", PETSCVIEWERDRAW, PETSC_VIEWER_DEFAULT, DMAdaptorMonitorErrorDraw, NULL, NULL));
  PetscCall(DMAdaptorMonitorRegister("error", PETSCVIEWERDRAW, PETSC_VIEWER_DRAW_LG, DMAdaptorMonitorErrorDrawLG, DMAdaptorMonitorErrorDrawLGCreate, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscInt Nc = uOff[1] - uOff[0];

  for (PetscInt i = 0; i < Nc; ++i) f[i] = u[i];
}

static void identityFunc(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  for (PetscInt i = 0; i < dim; ++i) {
    for (PetscInt j = 0; j < dim; ++j) f[i + dim * j] = u[i + dim * j];
  }
}

static PetscErrorCode DMAdaptorAdapt_Sequence_Private(DMAdaptor adaptor, Vec inx, PetscBool doSolve, DM *adm, Vec *ax)
{
  PetscDS   ds;
  PetscReal errorNorm = 0.;
  PetscInt  numAdapt  = adaptor->numSeq, adaptIter;
  PetscInt  dim, coordDim, Nf;
  void     *ctx;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(DMViewFromOptions(adaptor->idm, NULL, "-dm_adapt_pre_view"));
  PetscCall(VecViewFromOptions(inx, NULL, "-sol_adapt_pre_view"));
  PetscCall(PetscObjectGetComm((PetscObject)adaptor, &comm));
  PetscCall(DMGetDimension(adaptor->idm, &dim));
  PetscCall(DMGetCoordinateDim(adaptor->idm, &coordDim));
  PetscCall(DMGetApplicationContext(adaptor->idm, &ctx));
  PetscCall(DMGetDS(adaptor->idm, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCheck(Nf != 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot refine with no fields present!");

  /* Adapt until nothing changes */
  /* Adapt for a specified number of iterates */
  for (adaptIter = 0; adaptIter < numAdapt - 1; ++adaptIter) PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(comm)));
  for (adaptIter = 0; adaptIter < numAdapt; ++adaptIter) {
    PetscBool adapted = PETSC_FALSE;
    DM        dm      = adaptIter ? *adm : adaptor->idm, odm;
    Vec       x       = adaptIter ? *ax : inx, locX, ox;
    Vec       error   = NULL;

    PetscCall(DMGetLocalVector(dm, &locX));
    PetscCall(DMAdaptorPreAdapt(adaptor, locX));
    if (doSolve) {
      SNES snes;

      PetscCall(DMAdaptorGetSolver(adaptor, &snes));
      PetscCall(SNESSolve(snes, NULL, adaptIter ? *ax : x));
    }
    PetscCall(DMGlobalToLocalBegin(dm, adaptIter ? *ax : x, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(dm, adaptIter ? *ax : x, INSERT_VALUES, locX));
    PetscCall(VecViewFromOptions(adaptIter ? *ax : x, (PetscObject)adaptor, "-adapt_primal_sol_vec_view"));
    switch (adaptor->adaptCriterion) {
    case DM_ADAPTATION_REFINE:
      PetscCall(DMRefine(dm, comm, &odm));
      PetscCheck(odm, comm, PETSC_ERR_ARG_INCOMP, "DMRefine() did not perform any refinement, cannot continue grid sequencing");
      adapted = PETSC_TRUE;
      PetscCall(DMAdaptorMonitor(adaptor, adaptIter, dm, dm, 1, &errorNorm, NULL));
      break;
    case DM_ADAPTATION_LABEL: {
      /* Adapt DM
           Create local solution
           Reconstruct gradients (FVM) or solve adjoint equation (FEM)
           Produce cellwise error indicator */
      DM             edm, plex;
      PetscDS        ds;
      PetscFE        efe;
      DMLabel        adaptLabel;
      IS             refineIS, coarsenIS;
      DMPolytopeType ct;
      PetscScalar    errorVal;
      PetscInt       nRefine, nCoarsen, cStart;

      PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));

      // TODO Move this creation to PreAdapt
      PetscCall(DMClone(dm, &edm));
      PetscCall(DMConvert(edm, DMPLEX, &plex));
      PetscCall(DMPlexGetHeightStratum(plex, 0, &cStart, NULL));
      PetscCall(DMPlexGetCellType(plex, cStart, &ct));
      PetscCall(DMDestroy(&plex));
      PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, 0, PETSC_DEFAULT, &efe));
      PetscCall(PetscObjectSetName((PetscObject)efe, "Error"));
      PetscCall(DMSetField(edm, 0, NULL, (PetscObject)efe));
      PetscCall(PetscFEDestroy(&efe));
      PetscCall(DMCreateDS(edm));
      PetscCall(DMGetGlobalVector(edm, &error));
      PetscCall(PetscObjectSetName((PetscObject)error, "Error Estimator"));

      PetscUseTypeMethod(adaptor, computeerrorindicator, locX, error);
      PetscCall(VecViewFromOptions(error, (PetscObject)adaptor, "-adapt_error_vec_view"));
      PetscCall(DMGetDS(edm, &ds));
      PetscCall(PetscDSSetObjective(ds, 0, identity));
      PetscCall(DMPlexComputeIntegralFEM(edm, error, &errorVal, NULL));
      errorNorm = PetscRealPart(errorVal);

      // Compute IS from VecTagger
      PetscCall(VecTaggerComputeIS(adaptor->refineTag, error, &refineIS, NULL));
      PetscCall(VecTaggerComputeIS(adaptor->coarsenTag, error, &coarsenIS, NULL));
      PetscCall(ISViewFromOptions(refineIS, (PetscObject)adaptor->refineTag, "-is_view"));
      PetscCall(ISViewFromOptions(coarsenIS, (PetscObject)adaptor->coarsenTag, "-is_view"));
      PetscCall(ISGetSize(refineIS, &nRefine));
      PetscCall(ISGetSize(coarsenIS, &nCoarsen));
      PetscCall(PetscInfo(adaptor, "DMAdaptor: numRefine %" PetscInt_FMT ", numCoarsen %" PetscInt_FMT "\n", nRefine, nCoarsen));
      if (nRefine) PetscCall(DMLabelSetStratumIS(adaptLabel, DM_ADAPT_REFINE, refineIS));
      if (nCoarsen) PetscCall(DMLabelSetStratumIS(adaptLabel, DM_ADAPT_COARSEN, coarsenIS));
      PetscCall(ISDestroy(&coarsenIS));
      PetscCall(ISDestroy(&refineIS));
      // Adapt DM from label
      if (nRefine || nCoarsen) {
        char        oprefix[PETSC_MAX_PATH_LEN];
        const char *p;
        PetscBool   flg;

        PetscCall(PetscOptionsHasName(NULL, adaptor->hdr.prefix, "-adapt_vec_view", &flg));
        if (flg) {
          Vec ref;

          PetscCall(DMPlexCreateLabelField(dm, adaptLabel, &ref));
          PetscCall(VecViewFromOptions(ref, (PetscObject)adaptor, "-adapt_vec_view"));
          PetscCall(VecDestroy(&ref));
        }

        PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &p));
        PetscCall(PetscStrncpy(oprefix, p, PETSC_MAX_PATH_LEN));
        PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)dm, "adapt_"));
        PetscCall(DMAdaptLabel(dm, adaptLabel, &odm));
        PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, oprefix));
        PetscCall(PetscObjectSetOptionsPrefix((PetscObject)odm, oprefix));
        PetscCall(DMAdaptorMonitor(adaptor, adaptIter, dm, odm, 1, &errorNorm, error));
        adapted = PETSC_TRUE;
      } else {
        PetscCall(DMAdaptorMonitor(adaptor, adaptIter, dm, dm, 1, &errorNorm, error));
      }
      PetscCall(DMLabelDestroy(&adaptLabel));
      PetscCall(DMRestoreGlobalVector(edm, &error));
      PetscCall(DMDestroy(&edm));
    } break;
    case DM_ADAPTATION_METRIC: {
      DM        dmGrad, dmHess, dmMetric, dmDet;
      Vec       xGrad, xHess, metric, determinant;
      PetscReal N;
      DMLabel   bdLabel = NULL, rgLabel = NULL;
      PetscBool higherOrder = PETSC_FALSE;
      PetscInt  Nd          = coordDim * coordDim, f, vStart, vEnd;
      void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);

      PetscCall(PetscMalloc(1, &funcs));
      funcs[0] = identityFunc;

      /*     Setup finite element spaces */
      PetscCall(DMClone(dm, &dmGrad));
      PetscCall(DMClone(dm, &dmHess));
      PetscCheck(Nf <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Adaptation with multiple fields not yet considered"); // TODO
      for (f = 0; f < Nf; ++f) {
        PetscFE         fe, feGrad, feHess;
        PetscDualSpace  Q;
        PetscSpace      space;
        DM              K;
        PetscQuadrature q;
        PetscInt        Nc, qorder, p;
        const char     *prefix;

        PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *)&fe));
        PetscCall(PetscFEGetNumComponents(fe, &Nc));
        PetscCheck(Nc <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Adaptation with multiple components not yet considered"); // TODO
        PetscCall(PetscFEGetBasisSpace(fe, &space));
        PetscCall(PetscSpaceGetDegree(space, NULL, &p));
        if (p > 1) higherOrder = PETSC_TRUE;
        PetscCall(PetscFEGetDualSpace(fe, &Q));
        PetscCall(PetscDualSpaceGetDM(Q, &K));
        PetscCall(DMPlexGetDepthStratum(K, 0, &vStart, &vEnd));
        PetscCall(PetscFEGetQuadrature(fe, &q));
        PetscCall(PetscQuadratureGetOrder(q, &qorder));
        PetscCall(PetscObjectGetOptionsPrefix((PetscObject)fe, &prefix));
        PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dmGrad), dim, Nc * coordDim, PETSC_TRUE, prefix, qorder, &feGrad));
        PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dmHess), dim, Nc * Nd, PETSC_TRUE, prefix, qorder, &feHess));
        PetscCall(DMSetField(dmGrad, f, NULL, (PetscObject)feGrad));
        PetscCall(DMSetField(dmHess, f, NULL, (PetscObject)feHess));
        PetscCall(DMCreateDS(dmGrad));
        PetscCall(DMCreateDS(dmHess));
        PetscCall(PetscFEDestroy(&feGrad));
        PetscCall(PetscFEDestroy(&feHess));
      }
      /*     Compute vertexwise gradients from cellwise gradients */
      PetscCall(DMCreateLocalVector(dmGrad, &xGrad));
      PetscCall(VecViewFromOptions(locX, NULL, "-sol_adapt_loc_pre_view"));
      PetscCall(DMPlexComputeGradientClementInterpolant(dm, locX, xGrad));
      PetscCall(VecViewFromOptions(xGrad, NULL, "-adapt_gradient_view"));
      /*     Compute vertexwise Hessians from cellwise Hessians */
      PetscCall(DMCreateLocalVector(dmHess, &xHess));
      PetscCall(DMPlexComputeGradientClementInterpolant(dmGrad, xGrad, xHess));
      PetscCall(VecViewFromOptions(xHess, NULL, "-adapt_hessian_view"));
      PetscCall(VecDestroy(&xGrad));
      PetscCall(DMDestroy(&dmGrad));
      /*     Compute L-p normalized metric */
      PetscCall(DMClone(dm, &dmMetric));
      N = adaptor->Nadapt >= 0 ? adaptor->Nadapt : PetscPowRealInt(adaptor->refinementFactor, dim) * ((PetscReal)(vEnd - vStart));
      // TODO This was where the old monitor was, figure out how to show metric and target N
      PetscCall(DMPlexMetricSetTargetComplexity(dmMetric, N));
      if (higherOrder) {
        /*   Project Hessian into P1 space, if required */
        PetscCall(DMPlexMetricCreate(dmMetric, 0, &metric));
        PetscCall(DMProjectFieldLocal(dmMetric, 0.0, xHess, funcs, INSERT_ALL_VALUES, metric));
        PetscCall(VecDestroy(&xHess));
        xHess = metric;
      }
      PetscCall(PetscFree(funcs));
      PetscCall(DMPlexMetricCreate(dmMetric, 0, &metric));
      PetscCall(DMPlexMetricDeterminantCreate(dmMetric, 0, &determinant, &dmDet));
      PetscCall(DMPlexMetricNormalize(dmMetric, xHess, PETSC_TRUE, PETSC_TRUE, metric, determinant));
      PetscCall(VecDestroy(&determinant));
      PetscCall(DMDestroy(&dmDet));
      PetscCall(VecDestroy(&xHess));
      PetscCall(DMDestroy(&dmHess));
      /*     Adapt DM from metric */
      PetscCall(DMGetLabel(dm, "marker", &bdLabel));
      PetscCall(DMAdaptMetric(dm, metric, bdLabel, rgLabel, &odm));
      PetscCall(DMAdaptorMonitor(adaptor, adaptIter, dm, odm, 1, &errorNorm, NULL));
      adapted = PETSC_TRUE;
      /*     Cleanup */
      PetscCall(VecDestroy(&metric));
      PetscCall(DMDestroy(&dmMetric));
    } break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid adaptation type: %d", adaptor->adaptCriterion);
    }
    PetscCall(DMAdaptorPostAdapt(adaptor));
    PetscCall(DMRestoreLocalVector(dm, &locX));
    /* If DM was adapted, replace objects and recreate solution */
    if (adapted) {
      const char *name;

      PetscCall(PetscObjectGetName((PetscObject)dm, &name));
      PetscCall(PetscObjectSetName((PetscObject)odm, name));
      /* Reconfigure solver */
      PetscCall(SNESReset(adaptor->snes));
      PetscCall(SNESSetDM(adaptor->snes, odm));
      PetscCall(DMAdaptorSetSolver(adaptor, adaptor->snes));
      PetscCall(DMPlexSetSNESLocalFEM(odm, PETSC_FALSE, ctx));
      PetscCall(SNESSetFromOptions(adaptor->snes));
      /* Transfer system */
      PetscCall(DMCopyDisc(dm, odm));
      /* Transfer solution */
      PetscCall(DMCreateGlobalVector(odm, &ox));
      PetscCall(PetscObjectGetName((PetscObject)x, &name));
      PetscCall(PetscObjectSetName((PetscObject)ox, name));
      PetscCall(DMAdaptorTransferSolution(adaptor, dm, x, odm, ox));
      /* Cleanup adaptivity info */
      if (adaptIter > 0) PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(comm)));
      PetscCall(DMForestSetAdaptivityForest(dm, NULL)); /* clear internal references to the previous dm */
      PetscCall(DMDestroy(&dm));
      PetscCall(VecDestroy(&x));
      *adm = odm;
      *ax  = ox;
    } else {
      *adm      = dm;
      *ax       = x;
      adaptIter = numAdapt;
    }
    if (adaptIter < numAdapt - 1) {
      PetscCall(DMViewFromOptions(odm, NULL, "-dm_adapt_iter_view"));
      PetscCall(VecViewFromOptions(ox, NULL, "-sol_adapt_iter_view"));
    }
  }
  PetscCall(DMViewFromOptions(*adm, NULL, "-dm_adapt_view"));
  PetscCall(VecViewFromOptions(*ax, NULL, "-sol_adapt_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorAdapt - Creates a new `DM` that is adapted to the problem

  Not Collective

  Input Parameters:
+ adaptor  - The `DMAdaptor` object
. x        - The global approximate solution
- strategy - The adaptation strategy, see `DMAdaptationStrategy`

  Output Parameters:
+ adm - The adapted `DM`
- ax  - The adapted solution

  Options Database Keys:
+ -snes_adapt <strategy> - initial, sequential, multigrid
. -adapt_gradient_view   - View the Clement interpolant of the solution gradient
. -adapt_hessian_view    - View the Clement interpolant of the solution Hessian
- -adapt_metric_view     - View the metric tensor for adaptive mesh refinement

  Level: intermediate

.seealso: [](ch_dmbase), `DMAdaptor`, `DMAdaptationStrategy`, `DMAdaptorSetSolver()`, `DMAdaptorCreate()`
@*/
PetscErrorCode DMAdaptorAdapt(DMAdaptor adaptor, Vec x, DMAdaptationStrategy strategy, DM *adm, Vec *ax)
{
  PetscFunctionBegin;
  switch (strategy) {
  case DM_ADAPTATION_INITIAL:
    PetscCall(DMAdaptorAdapt_Sequence_Private(adaptor, x, PETSC_FALSE, adm, ax));
    break;
  case DM_ADAPTATION_SEQUENTIAL:
    PetscCall(DMAdaptorAdapt_Sequence_Private(adaptor, x, PETSC_TRUE, adm, ax));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)adaptor), PETSC_ERR_ARG_WRONG, "Unrecognized adaptation strategy %d", strategy);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorGetMixedSetupFunction - Get the function setting up the mixed problem, if it exists

  Not Collective

  Input Parameter:
. adaptor - the `DMAdaptor`

  Output Parameter:
. setupFunc - the function setting up the mixed problem, or `NULL`

  Level: advanced

.seealso: `DMAdaptor`, `DMAdaptorSetMixedSetupFunction()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorGetMixedSetupFunction(DMAdaptor adaptor, PetscErrorCode (**setupFunc)(DMAdaptor, DM))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscAssertPointer(setupFunc, 2);
  *setupFunc = adaptor->ops->mixedsetup;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAdaptorSetMixedSetupFunction - Set the function setting up the mixed problem

  Not Collective

  Input Parameters:
+ adaptor   - the `DMAdaptor`
- setupFunc - the function setting up the mixed problem

  Level: advanced

.seealso: `DMAdaptor`, `DMAdaptorGetMixedSetupFunction()`, `DMAdaptorAdapt()`
@*/
PetscErrorCode DMAdaptorSetMixedSetupFunction(DMAdaptor adaptor, PetscErrorCode (*setupFunc)(DMAdaptor, DM))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscValidFunction(setupFunc, 2);
  adaptor->ops->mixedsetup = setupFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorGetCriterion - Get the adaptation criterion

  Not Collective

  Input Parameter:
. adaptor - the `DMAdaptor`

  Output Parameter:
. criterion - the criterion for adaptation

  Level: advanced

.seealso: `DMAdaptor`, `DMAdaptorSetCriterion()`, `DMAdaptationCriterion`
@*/
PetscErrorCode DMAdaptorGetCriterion(DMAdaptor adaptor, DMAdaptationCriterion *criterion)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  PetscAssertPointer(criterion, 2);
  *criterion = adaptor->adaptCriterion;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMAdaptorSetCriterion - Set the adaptation criterion

  Not Collective

  Input Parameters:
+ adaptor   - the `DMAdaptor`
- criterion - the adaptation criterion

  Level: advanced

.seealso: `DMAdaptor`, `DMAdaptorGetCriterion()`, `DMAdaptationCriterion`
@*/
PetscErrorCode DMAdaptorSetCriterion(DMAdaptor adaptor, DMAdaptationCriterion criterion)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  adaptor->adaptCriterion = criterion;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorInitialize_Gradient(DMAdaptor adaptor)
{
  PetscFunctionBegin;
  adaptor->ops->computeerrorindicator     = DMAdaptorComputeErrorIndicator_Gradient;
  adaptor->ops->computecellerrorindicator = DMAdaptorComputeCellErrorIndicator_Gradient;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMAdaptorCreate_Gradient(DMAdaptor adaptor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  adaptor->data = NULL;

  PetscCall(DMAdaptorInitialize_Gradient(adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorInitialize_Flux(DMAdaptor adaptor)
{
  PetscFunctionBegin;
  adaptor->ops->computeerrorindicator = DMAdaptorComputeErrorIndicator_Flux;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMAdaptorCreate_Flux(DMAdaptor adaptor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMADAPTOR_CLASSID, 1);
  adaptor->data = NULL;

  PetscCall(DMAdaptorInitialize_Flux(adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}
