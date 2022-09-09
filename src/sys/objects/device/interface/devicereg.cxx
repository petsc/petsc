#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

PetscClassId PETSC_DEVICE_CLASSID, PETSC_DEVICE_CONTEXT_CLASSID;

PetscLogEvent CUBLAS_HANDLE_CREATE, CUSOLVER_HANDLE_CREATE;
PetscLogEvent HIPSOLVER_HANDLE_CREATE, HIPBLAS_HANDLE_CREATE;

static PetscBool PetscDevicePackageInitialized = PETSC_FALSE;

static PetscErrorCode PetscDeviceRegisterEvent_Private(const char name[], PetscClassId id, PetscLogEvent *event) {
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
PetscErrorCode PetscDeviceFinalizePackage(void) {
  PetscFunctionBegin;
  PetscDevicePackageInitialized = PETSC_FALSE;
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
PetscErrorCode PetscDeviceInitializePackage(void) {
  PetscFunctionBegin;
  if (PetscLikely(PetscDevicePackageInitialized)) PetscFunctionReturn(0);
  PetscCheck(PetscDeviceConfiguredFor_Internal(PETSC_DEVICE_DEFAULT), PETSC_COMM_SELF, PETSC_ERR_SUP, "PETSc is not configured with device support (PETSC_DEVICE_DEFAULT = '%s')", PetscDeviceTypes[PETSC_DEVICE_DEFAULT]);
  PetscDevicePackageInitialized = PETSC_TRUE;
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
  PetscFunctionReturn(0);
}
