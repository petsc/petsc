#include "petscdevice_interface_internal.hpp" /*I <petscdevice.h> I*/
#include <petscdevice_cupm.h>

/* rootContext holds the default context created by PETSc, for cleanup during PetscFinalize().
   globalContext selects the current context, which may instead be supplied by the application.
   PetscDeviceContextSetCurrentContext() changes this selection without transferring ownership,
   so rootContext must be kept separately for PETSc to clean up the context it created. */
static auto               rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
static auto               rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
static PetscDeviceContext globalContext  = nullptr;
static PetscDeviceContext rootContext    = nullptr;

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
      PetscCall(PetscDeviceContextGetDeviceType(rootContext, &dtype));
      PetscCall(PetscInfo(rootContext, "Destroying global PetscDeviceContext with device type %s\n", PetscDeviceTypes[dtype]));
      PetscCall(PetscDeviceContextDestroy(&rootContext));
      globalContext  = nullptr;
      rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
      rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    /* this exists purely as a valid device check. */
    PetscCall(PetscDeviceInitializePackage());
    PetscCall(PetscRegisterFinalize(std::move(finalizer)));
    PetscCall(PetscDeviceContextCreate(&globalContext));
    rootContext = globalContext;
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

/*@
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

/*@
  PetscDeviceContextSetCurrentContext - Set the current active `PetscDeviceContext`

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Level: beginner

  Notes:
  This routine selects the context used by subsequent operations of PETSc's built-in CUDA and HIP
  implementations. Kokkos-based types keep the execution space created when PETSc initialized
  Kokkos, and external packages may use their own streams. Selecting another context does not
  replace PETSc's root context.

  The supplied context must be set up, or be `PetscDeviceContextDefault`, which is resolved before
  being installed as current. To restore a previous context, pass the handle
  saved from `PetscDeviceContextGetCurrentContext()`.

  This routine does not take ownership of a user-created context; keep it alive while it is current
  and destroy it before `PetscFinalize()`. PETSc retains ownership of its root and cached default contexts.

.seealso: `PetscDeviceContextGetCurrentContext()`, `PetscDeviceContextFork()`,
          `PetscDeviceContextJoin()`, `PetscDeviceContextCreate()`
@*/
PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext dctx)
{
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextResolveDefault_Private(&dctx));
  PetscValidDeviceContext(dctx, 1);
  PetscAssert(dctx->setup, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt64_FMT " must be set up before being set as global context", PetscObjectCast(dctx)->id);
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  PetscCall(PetscDeviceSetDefaultDeviceType(dtype));
  globalContext = dctx;
  PetscCall(PetscInfo(dctx, "Set global PetscDeviceContext id %" PetscInt64_FMT "\n", PetscObjectCast(dctx)->id));
  PetscCall(PetscSetDefaultCUPMStreamFromDeviceContext(globalContext, dtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}
