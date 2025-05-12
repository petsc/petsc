#include <petsc/private/petscimpl.h>
#include <petsc/private/bmimpl.h> /*I  "petscbm.h"   I*/
#include <petscviewer.h>

PetscClassId             BM_CLASSID;
static PetscBool         PetscBenchPackageInitialized = PETSC_FALSE;
static PetscFunctionList PetscBenchList               = NULL;

// PetscClangLinter pragma disable: -fdoc-internal-linkage
/*@C
  PetscBenchFinalizePackage - This function destroys everything in the `PetscBench` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: `PetscFinalize()`, `PetscBenchInitializePackage()`, `PetscBenchCreate()`, `PetscBench`, `PetscBenchType`
@*/
static PetscErrorCode PetscBenchFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscBenchList));
  PetscBenchPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBenchInitializePackage - This function initializes everything in the `PetscBench` package.

  Level: developer

.seealso: `PetscInitialize()`, `PetscBenchCreate()`, `PetscBench`, `PetscBenchType`
@*/
PetscErrorCode PetscBenchInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscBenchPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscBenchPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("PetscBench", &BM_CLASSID));
  PetscCall(PetscRegisterFinalize(PetscBenchFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBenchRegister -  Adds a benchmark test, `PetscBenchType`, to the `PetscBench` package

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of a new benchmark
- function - routine to create benchmark

  Level: advanced

  Note:
  `PetscBenchRegister()` may be called multiple times

.seealso: `PetscBenchInitializePackage()`, `PetscBenchCreate()`, `PetscBench`, `PetscBenchType`, `PetscBenchSetType()`, `PetscBenchGetType()`
@*/
PetscErrorCode PetscBenchRegister(const char sname[], PetscErrorCode (*function)(PetscBench))
{
  PetscFunctionBegin;
  PetscCall(PetscBenchInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscBenchList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchReset - removes all the intermediate data structures in a `PetscBench`

  Collective

  Input Parameter:
. bm - the `PetscBench`

  Level: advanced

.seealso: `PetscBench`, `PetscBenchView()`, `PetscBenchSetFromOptions()`, `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`
@*/
PetscErrorCode PetscBenchReset(PetscBench bm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscCall(PetscLogHandlerDestroy(&bm->lhdlr)); // Temporarily here until PetscLogHandlerReset() exists
  PetscTryTypeMethod(bm, reset);
  bm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchDestroy - Destroys a `PetscBench`

  Collective

  Input Parameter:
. bm - the `PetscBench`

  Level: advanced

.seealso: `PetscBench`, `PetscBenchView()`, `PetscBenchSetFromOptions()`, `PetscBenchCreate()`
@*/
PetscErrorCode PetscBenchDestroy(PetscBench *bm)
{
  PetscFunctionBegin;
  PetscAssertPointer(bm, 1);
  if (!*bm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*bm, BM_CLASSID, 1);
  if (--((PetscObject)*bm)->refct > 0) {
    *bm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscBenchReset(*bm));
  PetscTryTypeMethod(*bm, destroy);
  PetscCall(PetscHeaderDestroy(bm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchSetUp - sets up the `PetscBench`

  Collective

  Input Parameter:
. bm - the `PetscBench`

  Level: advanced

.seealso: `PetscBench`, `PetscBenchView()`, `PetscBenchSetFromOptions()`, `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetType()`,
          `PetscBenchRun()`, `PetscBenchSetSize()`, `PetscBenchGetSize()`
@*/
PetscErrorCode PetscBenchSetUp(PetscBench bm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  if (bm->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogHandlerCreate(PETSC_COMM_WORLD, &bm->lhdlr)); // Temporarily here until PetscLogHandlerReset() exists
  PetscCall(PetscLogHandlerSetType(bm->lhdlr, PETSCLOGHANDLERDEFAULT));
  PetscTryTypeMethod(bm, setup);
  bm->setupcalled = PETSC_TRUE;
  PetscTryTypeMethod(bm, run);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchRun - runs the `PetscBench`

  Collective

  Input Parameter:
. bm - the `PetscBench`

  Level: advanced

.seealso: `PetscBench`, `PetscBenchView()`, `PetscBenchSetFromOptions()`, `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetSize()`, `PetscBenchGetSize()`
@*/
PetscErrorCode PetscBenchRun(PetscBench bm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  if (!bm->setupcalled) PetscCall(PetscBenchSetUp(bm));
  PetscCall(PetscLogHandlerStart(bm->lhdlr));
  PetscTryTypeMethod(bm, run);
  PetscCall(PetscLogHandlerStop(bm->lhdlr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchSetFromOptions - Sets options to a `PetscBench` using the options database

  Collective

  Input Parameter:
. bm - the `PetscBench`

  Level: advanced

.seealso: `PetscBench`, `PetscBenchView()`, `PetscBenchRun()`, `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetSize()`, `PetscBenchGetSize()`
@*/
PetscErrorCode PetscBenchSetFromOptions(PetscBench bm)
{
  char      type[256];
  PetscBool flg;
  PetscInt  m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)bm);
  PetscCall(PetscOptionsFList("-bm_type", "PetscBench", "PetscBenchSetType", PetscBenchList, ((PetscObject)bm)->type_name, type, sizeof(type), &flg));
  if (flg) PetscCall(PetscBenchSetType(bm, type));
  PetscCheck(((PetscObject)bm)->type_name, PetscObjectComm((PetscObject)bm), PETSC_ERR_ARG_WRONGSTATE, "No PetscBenchType provided for PetscBench");
  PetscCall(PetscOptionsInt("-bm_size", "Size of benchmark", "PetscBenchSetSize", bm->size, &m, &flg));
  if (flg) PetscCall(PetscBenchSetSize(bm, m));
  PetscTryTypeMethod(bm, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchView - Views a PETSc benchmark `PetscBench`

  Collective

  Input Parameters:
+ bm     - the `PetscBench`
- viewer - location to view `bm`

  Level: advanced

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchRun()`, `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetSize()`, `PetscBenchGetSize()`, `PetscBenchViewFromOptions()`
@*/
PetscErrorCode PetscBenchView(PetscBench bm, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscTryTypeMethod(bm, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchViewFromOptions - Processes command line options to determine if/how a `PetscBench` is to be viewed.

  Collective

  Input Parameters:
+ bm         - the object
. bobj       - optional other object that provides prefix (if `NULL` then the prefix in `bm` is used)
- optionname - option to activate viewing

  Level: advanced

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchRun()`, `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetSize()`, `PetscBenchGetSize()`
@*/
PetscErrorCode PetscBenchViewFromOptions(PetscBench bm, PetscObject bobj, const char optionname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)bm, bobj, optionname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchCreate - Create a PETSc benchmark `PetscBench` object

  Collective

  Input Parameter:
. comm - communicator to share the `PetscBench`

  Output Parameter:
. bm - the `PetscBench`

  Level: advanced

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchRun()`, `PetscBenchViewFromOptions()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetSize()`, `PetscBenchGetSize()`
@*/
PetscErrorCode PetscBenchCreate(MPI_Comm comm, PetscBench *bm)
{
  PetscFunctionBegin;
  PetscAssertPointer(bm, 2);
  PetscCall(PetscBenchInitializePackage());

  PetscCall(PetscHeaderCreate(*bm, BM_CLASSID, "BM", "PetscBench", "BM", comm, PetscBenchDestroy, PetscBenchView));
  (*bm)->size = PETSC_DECIDE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchSetOptionsPrefix - Sets the prefix used for searching for all `PetscBench` items in the options database.

  Logically Collective

  Input Parameters:
+ bm  - the `PetscBench`
- pre - the prefix to prepend all `PetscBench` option names

  Level: advanced

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchRun()`, `PetscBenchViewFromOptions()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetSize()`, `PetscBenchGetSize()`
@*/
PetscErrorCode PetscBenchSetOptionsPrefix(PetscBench bm, const char pre[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)bm, pre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchSetSize - Sets the size of the `PetscBench` benchmark to run

  Logically Collective

  Input Parameters:
+ bm - the `PetscBench`
- n  - the size

  Level: advanced

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchRun()`, `PetscBenchViewFromOptions()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetOptionsPrefix()`, `PetscBenchGetSize()`
@*/
PetscErrorCode PetscBenchSetSize(PetscBench bm, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  if (bm->size > 0 && bm->size != n && bm->setupcalled) {
    PetscCall(PetscBenchReset(bm));
    bm->setupcalled = PETSC_FALSE;
  }
  PetscCheck(n > 0, PetscObjectComm((PetscObject)bm), PETSC_ERR_ARG_OUTOFRANGE, "Illegal value of n. Must be > 0");
  bm->size = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchGetSize - Gets the size of the `PetscBench` benchmark to run

  Logically Collective

  Input Parameter:
. bm - the `PetscBench`

  Output Parameter:
. n - the size

  Level: advanced

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchRun()`, `PetscBenchViewFromOptions()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchSetType()`,
          `PetscBenchSetOptionsPrefix()`, `PetscBenchSetSize()`
@*/
PetscErrorCode PetscBenchGetSize(PetscBench bm, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscAssertPointer(n, 2);
  *n = bm->size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchSetType - set the type of `PetscBench` benchmark to run

  Collective

  Input Parameters:
+ bm   - the `PetscBench`
- type - a known method

  Options Database Key:
. -bm_type <type> - Sets `PetscBench` type

  Level: advanced

  Developer Note:
  `PetscBenchRegister()` is used to add new benchmark types

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchRun()`, `PetscBenchViewFromOptions()`, `PetscBenchDestroy()`, `PetscBenchSetUp()`, `PetscBenchGetSize()`,
          `PetscBenchSetOptionsPrefix()`, `PetscBenchSetSize()`, `PetscBenchGetType()`, `PetscBenchCreate()`
@*/
PetscErrorCode PetscBenchSetType(PetscBench bm, PetscBenchType type)
{
  PetscBool match;
  PetscErrorCode (*r)(PetscBench);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)bm, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PetscBenchList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)bm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PetscBench type %s", type);
  /* Destroy the previous private BM context */
  PetscTryTypeMethod(bm, destroy);
  bm->ops->destroy = NULL;
  bm->data         = NULL;

  PetscCall(PetscFunctionListDestroy(&((PetscObject)bm)->qlist));
  /* Reinitialize function pointers in PetscBenchOps structure */
  PetscCall(PetscMemzero(bm->ops, sizeof(struct _PetscBenchOps)));

  PetscCall(PetscObjectChangeTypeName((PetscObject)bm, type));
  PetscCall((*r)(bm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBenchGetType - Gets the `PetscBenchType` (as a string) from the `PetscBench`
  context.

  Not Collective

  Input Parameter:
. bm - the `PetscBench`

  Output Parameter:
. type - name of benchmark method

  Level: intermediate

.seealso: `PetscBench`, `PetscBenchType`, `PetscBenchSetType()`, `PetscBenchCreate()`
@*/
PetscErrorCode PetscBenchGetType(PetscBench bm, PetscBenchType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)bm)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}
