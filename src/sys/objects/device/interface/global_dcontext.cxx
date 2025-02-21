#include "petscdevice_interface_internal.hpp" /*I <petscdevice.h> I*/
#include <petscdevice_cupm.h>

static auto               rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
static auto               rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
static PetscDeviceContext globalContext  = nullptr;

/* when PetscDevice initializes PetscDeviceContext eagerly the type of device created should
 * match whatever device is eagerly initialized */
PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType type)
{
  PetscFunctionBegin;
  PetscValidDeviceType(type, 1);
  rootDeviceType = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSetRootStreamType_Internal(PetscStreamType type)
{
  PetscFunctionBegin;
  PetscValidStreamType(type, 1);
  rootStreamType = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscSetDefaultCUPMStreamFromDeviceContext(PetscDeviceContext dctx, PetscDeviceType dtype)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_CUDA)
  if (dtype == PETSC_DEVICE_CUDA) {
    void *handle;

    PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &handle));
    PetscDefaultCudaStream = *static_cast<cudaStream_t *>(handle);
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (dtype == PETSC_DEVICE_HIP) {
    void *handle;

    PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &handle));
    PetscDefaultHipStream = *static_cast<hipStream_t *>(handle);
  }
#endif
#if !PetscDefined(HAVE_CUDA) && !PetscDefined(HAVE_HIP)
  (void)dctx, (void)dtype;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDeviceContextSetupGlobalContext_Private() noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!globalContext)) {
    PetscObject pobj;
    const auto  dtype     = rootDeviceType;
    const auto  finalizer = [] {
      PetscDeviceType dtype;

      PetscFunctionBegin;
      PetscCall(PetscDeviceContextGetDeviceType(globalContext, &dtype));
      PetscCall(PetscInfo(globalContext, "Destroying global PetscDeviceContext with device type %s\n", PetscDeviceTypes[dtype]));
      PetscCall(PetscDeviceContextDestroy(&globalContext));
      rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
      rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    /* this exists purely as a valid device check. */
    PetscCall(PetscDeviceInitializePackage());
    PetscCall(PetscRegisterFinalize(std::move(finalizer)));
    PetscCall(PetscDeviceContextCreate(&globalContext));
    PetscCall(PetscInfo(globalContext, "Initializing global PetscDeviceContext with device type %s\n", PetscDeviceTypes[dtype]));
    pobj = PetscObjectCast(globalContext);
    PetscCall(PetscObjectSetName(pobj, "global root"));
    PetscCall(PetscObjectSetOptionsPrefix(pobj, "root_"));
    PetscCall(PetscDeviceContextSetStreamType(globalContext, rootStreamType));
    PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(globalContext, dtype));
    PetscCall(PetscDeviceContextSetUp(globalContext));
    PetscCall(PetscSetDefaultCUPMStreamFromDeviceContext(globalContext, dtype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextGetCurrentContext - Get the current active `PetscDeviceContext`

  Not Collective

  Output Parameter:
. dctx - The `PetscDeviceContext`

  Notes:
  The user generally should not destroy contexts retrieved with this routine unless they
  themselves have created them. There exists no protection against destroying the root
  context.

  Developer Notes:
  Unless the user has set their own, this routine creates the "root" context the first time it
  is called, registering its destructor to `PetscFinalize()`.

  Level: beginner

.seealso: `PetscDeviceContextSetCurrentContext()`, `PetscDeviceContextFork()`,
          `PetscDeviceContextJoin()`, `PetscDeviceContextCreate()`
@*/
PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscAssertPointer(dctx, 1);
  PetscCall(PetscDeviceContextSetupGlobalContext_Private());
  /* while the static analyzer can find global variables, it will throw a warning about not
   * being able to connect this back to the function arguments */
  PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidDeviceContext(globalContext, -1));
  *dctx = globalContext;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextSetCurrentContext - Set the current active `PetscDeviceContext`

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Notes:
  This routine can be used to set the defacto "root" `PetscDeviceContext` to a user-defined
  implementation by calling this routine immediately after `PetscInitialize()` and ensuring that
  `PetscDevice` is not greedily initialized. In this case the user is responsible for destroying
  their `PetscDeviceContext` before `PetscFinalize()` returns.

  The old context is not stored in any way by this routine; if one is overriding a context that
  they themselves do not control, one should take care to temporarily store it by calling
  `PetscDeviceContextGetCurrentContext()` before calling this routine.

  Level: beginner

.seealso: `PetscDeviceContextGetCurrentContext()`, `PetscDeviceContextFork()`,
          `PetscDeviceContextJoin()`, `PetscDeviceContextCreate()`
@*/
PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext dctx)
{
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssert(dctx->setup, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt64_FMT " must be set up before being set as global context", PetscObjectCast(dctx)->id);
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  PetscCall(PetscDeviceSetDefaultDeviceType(dtype));
  globalContext = dctx;
  PetscCall(PetscInfo(dctx, "Set global PetscDeviceContext id %" PetscInt64_FMT "\n", PetscObjectCast(dctx)->id));
  PetscCall(PetscSetDefaultCUPMStreamFromDeviceContext(globalContext, dtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}
