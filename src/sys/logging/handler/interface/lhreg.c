#include <petsc/private/petscimpl.h>
#include <petsc/private/loghandlerimpl.h>

PetscClassId PETSCLOGHANDLER_CLASSID = 0;

PetscFunctionList PetscLogHandlerList               = NULL;
PetscBool         PetscLogHandlerRegisterAllCalled  = PETSC_FALSE;
PetscBool         PetscLogHandlerPackageInitialized = PETSC_FALSE;

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandler);
PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Nested(PetscLogHandler);
PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Trace(PetscLogHandler);
#if PetscDefined(HAVE_MPE)
PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_MPE(PetscLogHandler);
#endif
#if PetscDefined(HAVE_TAU_PERFSTUBS)
PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Perfstubs(PetscLogHandler);
#endif
PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Legacy(PetscLogHandler);
#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_NVTX(PetscLogHandler);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_ROCTX(PetscLogHandler);
#endif

static PetscErrorCode PetscLogHandlerRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscLogHandlerRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscLogHandlerRegisterAllCalled = PETSC_TRUE;
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERDEFAULT, PetscLogHandlerCreate_Default));
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERNESTED, PetscLogHandlerCreate_Nested));
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERTRACE, PetscLogHandlerCreate_Trace));
#if PetscDefined(HAVE_MPE)
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERMPE, PetscLogHandlerCreate_MPE));
#endif
#if PetscDefined(HAVE_TAU_PERFSTUBS)
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERPERFSTUBS, PetscLogHandlerCreate_Perfstubs));
#endif
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERLEGACY, PetscLogHandlerCreate_Legacy));
#if PetscDefined(HAVE_CUDA)
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERNVTX, PetscLogHandlerCreate_NVTX));
#endif
#if PetscDefined(HAVE_ROCTX)
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLERROCTX, PetscLogHandlerCreate_ROCTX));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerRegister - Register a new `PetscLogHandler`

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - The name of a new user-defined creation routine
- function - The creation routine

  Example Usage:
.vb
    PetscLogHandlerRegister("my_profiler", MyPetscLogHandlerCreate);
.ve

  Then, your `PetscLogHandler` type can be chosen with the procedural interface via
.vb
    PetscLogHandlerCreate(MPI_Comm, PetscLogHandler *);
    PetscLogHandlerSetType(PetscFE, "my_fe");
.ve

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`, `PetscLogHandlerSetType()`, `PetscLogHandlerGetType()`
@*/
PetscErrorCode PetscLogHandlerRegister(const char sname[], PetscErrorCode (*function)(PetscLogHandler))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&PetscLogHandlerList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerSetType - Set the type of a `PetscLogHandler`

  Input Parameters:
+ handler - the `PetscLogHandler`
- name    - The kind of log handler

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`, `PetscLogHandlerRegister()`, `PetscLogHandlerGetType()`
@*/
PetscErrorCode PetscLogHandlerSetType(PetscLogHandler handler, PetscLogHandlerType name)
{
  PetscErrorCode (*r)(PetscLogHandler);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)handler, name, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogHandlerRegisterAll());
  PetscCall(PetscFunctionListFind(PetscLogHandlerList, name, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)handler), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscLogHandler type: %s", name);

  PetscTryTypeMethod(handler, destroy);
  handler->ops->destroy = NULL;

  PetscCall((*r)(handler));
  PetscCall(PetscObjectChangeTypeName((PetscObject)handler, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerGetType - Gets the `PetscLoagHandlerType` (as a string) from the `PetscLogHandler` object.

  Not collective

  Input Parameter:
. handler - the `PetscLogHandler`

  Output Parameter:
. name - The `PetscLogHandlerType` name

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`, `PetscLogHandlerRegister()`, `PetscLogHandlerSetType()`
@*/
PetscErrorCode PetscLogHandlerGetType(PetscLogHandler handler, PetscLogHandlerType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCall(PetscLogHandlerRegisterAll());
  PetscCall(PetscObjectGetType((PetscObject)handler, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscLogHandlerList));
  PetscLogHandlerRegisterAllCalled  = PETSC_FALSE;
  PetscLogHandlerPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogHandlerPackageInitialize(void)
{
  PetscFunctionBegin;
  if (PetscLogHandlerPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscLogHandlerPackageInitialized = PETSC_TRUE;

  PetscCall(PetscClassIdRegister("PETSc Log Handler", &PETSCLOGHANDLER_CLASSID));
  PetscCall(PetscLogHandlerRegisterAll());
  PetscCall(PetscRegisterFinalize(PetscLogHandlerFinalizePackage));
  {
    const PetscClassId classids[] = {PETSCLOGHANDLER_CLASSID};

    PetscCall(PetscInfoProcessClass("loghandler", PETSC_STATIC_ARRAY_LENGTH(classids), classids));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
