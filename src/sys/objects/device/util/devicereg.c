#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/

PetscLogEvent CUBLAS_HANDLE_CREATE;
PetscLogEvent CUSOLVER_HANDLE_CREATE;
PetscLogEvent HIPSOLVER_HANDLE_CREATE;
PetscLogEvent HIPBLAS_HANDLE_CREATE;

PetscLogEvent DCONTEXT_Create;
PetscLogEvent DCONTEXT_Destroy;
PetscLogEvent DCONTEXT_ChangeStream;
PetscLogEvent DCONTEXT_SetDevice;
PetscLogEvent DCONTEXT_SetUp;
PetscLogEvent DCONTEXT_Duplicate;
PetscLogEvent DCONTEXT_QueryIdle;
PetscLogEvent DCONTEXT_WaitForCtx;
PetscLogEvent DCONTEXT_Fork;
PetscLogEvent DCONTEXT_Join;
PetscLogEvent DCONTEXT_Sync;
PetscLogEvent DCONTEXT_Mark;

// DO NOT MOVE THESE (literally, they must be exactly here)!
//
// pgcc has a _very_ strange bug, where if both of these are defined at the top of this file,
// then building src/sys/objects/device/test/ex2.c results in "undefined reference to
// PETSC_DEVICE_CONTEXT_CLASSID". If you initialize PETSC_DEVICE_CONTEXT_CLASSID it goes
// away. If you move the definition down, it goes away.
PetscClassId PETSC_DEVICE_CLASSID;
PetscClassId PETSC_DEVICE_CONTEXT_CLASSID;

// clang-format off
const char *const PetscStreamTypes[] = {
  "global_blocking",
  "default_blocking",
  "global_nonblocking",
  "max",
  "PetscStreamType",
  "PETSC_STREAM_",
  PETSC_NULLPTR
};

const char *const PetscDeviceContextJoinModes[] = {
  "destroy",
  "sync",
  "no_sync",
  "PetscDeviceContextJoinMode",
  "PETSC_DEVICE_CONTEXT_JOIN_",
  PETSC_NULLPTR
};

const char *const PetscDeviceTypes[] = {
  "host",
  "cuda",
  "hip",
  "sycl",
  "max",
  "PetscDeviceType",
  "PETSC_DEVICE_",
  PETSC_NULLPTR
};

const char *const PetscDeviceInitTypes[] = {
  "none",
  "lazy",
  "eager",
  "PetscDeviceInitType",
  "PETSC_DEVICE_INIT_",
  PETSC_NULLPTR
};

#ifdef __cplusplus
#include <petsc/private/cpp/type_traits.hpp>

static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_NONE) == 0, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_LAZY) == 1, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_EAGER) == 2, "");

static_assert(
  PETSC_STATIC_ARRAY_LENGTH(PetscDeviceInitTypes) == 6,
  "Must change CUPMDevice<T>::initialize number of enum values in -device_enable_cupm to match!"
);
#endif

const char *const PetscDeviceAttributes[] = {
  "shared_mem_per_block",
  "max",
  "PetscDeviceAttribute",
  "PETSC_DEVICE_ATTR_",
  PETSC_NULLPTR
};
// clang-format on

static PetscBool registered = PETSC_FALSE;

static PetscErrorCode PetscDeviceRegisterEvent_Private(const char name[], PetscClassId id, PetscLogEvent *event)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventRegister(name, id, event));
  PetscCall(PetscLogEventSetCollective(*event, PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceFinalizePackage - This function cleans up all components of the `PetscDevice`
  package. It is called from `PetscFinalize()`.

  Developer Note:
  This function is automatically registered to be called during `PetscFinalize()` by
  `PetscDeviceInitializePackage()` so there should be no need to call it yourself.

  Level: developer

.seealso: `PetscFinalize()`, `PetscDeviceInitializePackage()`
@*/
PetscErrorCode PetscDeviceFinalizePackage(void)
{
  PetscFunctionBegin;
  registered = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceInitializePackage - This function initializes everything in the `PetscDevice`
  package. It is called on the first call to `PetscDeviceContextCreate()` or
  `PetscDeviceCreate()` when using shared or static libraries.

  Level: developer

.seealso: `PetscInitialize()`, `PetscDeviceFinalizePackage()`, `PetscDeviceContextCreate()`,
`PetscDeviceCreate()`
@*/
PetscErrorCode PetscDeviceInitializePackage(void)
{
  PetscFunctionBegin;
  PetscCheck(PetscDeviceConfiguredFor_Internal(PETSC_DEVICE_DEFAULT()), PETSC_COMM_SELF, PETSC_ERR_SUP, "PETSc is not configured with device support (PETSC_DEVICE_DEFAULT = '%s')", PetscDeviceTypes[PETSC_DEVICE_DEFAULT()]);
  if (PetscLikely(registered)) PetscFunctionReturn(0);
  registered = PETSC_TRUE;
  PetscCall(PetscRegisterFinalize(PetscDeviceFinalizePackage));
  // class registration
  PetscCall(PetscClassIdRegister("PetscDevice", &PETSC_DEVICE_CLASSID));
  PetscCall(PetscClassIdRegister("PetscDeviceContext", &PETSC_DEVICE_CONTEXT_CLASSID));
  // events
  if (PetscDefined(HAVE_CUDA)) {
    PetscCall(PetscDeviceRegisterEvent_Private("cuBLAS Init", PETSC_DEVICE_CONTEXT_CLASSID, &CUBLAS_HANDLE_CREATE));
    PetscCall(PetscDeviceRegisterEvent_Private("cuSolver Init", PETSC_DEVICE_CONTEXT_CLASSID, &CUSOLVER_HANDLE_CREATE));
  }
  if (PetscDefined(HAVE_HIP)) {
    PetscCall(PetscDeviceRegisterEvent_Private("hipBLAS Init", PETSC_DEVICE_CONTEXT_CLASSID, &HIPBLAS_HANDLE_CREATE));
    PetscCall(PetscDeviceRegisterEvent_Private("hipSolver Init", PETSC_DEVICE_CONTEXT_CLASSID, &HIPSOLVER_HANDLE_CREATE));
  }
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxCreate", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Create));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxDestroy", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Destroy));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxChangeStream", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_ChangeStream));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxSetUp", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_SetUp));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxSetDevice", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_SetDevice));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxDuplicate", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Duplicate));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxQueryIdle", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_QueryIdle));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxWaitForCtx", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_WaitForCtx));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxFork", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Fork));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxJoin", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Join));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxSync", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Sync));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxMark", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Mark));
  {
    const PetscClassId classids[] = {PETSC_DEVICE_CONTEXT_CLASSID, PETSC_DEVICE_CLASSID};

    PetscCall(PetscInfoProcessClass("device", PETSC_STATIC_ARRAY_LENGTH(classids), classids));
  }
  PetscFunctionReturn(0);
}
