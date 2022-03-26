#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

static PetscBool PetscDevicePackageInitialized = PETSC_FALSE;

/*@C
  PetscDeviceFinalizePackage - This function cleans up all components of the PetscDevice package.
  It is called from PetscFinalize().

  Developer Notes:
  This function is automatically registered to be called during PetscFinalize() by
  PetscDeviceInitializePackage() so there should be no need to call it yourself.

  Level: developer

.seealso: PetscFinalize(), PetscDeviceInitializePackage()
@*/
PetscErrorCode PetscDeviceFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDevicePackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceInitializePackage - This function initializes everything in the PetscDevice
  package. It is called on the first call to PetscDeviceContextCreate() or PetscDeviceCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscDeviceFinalizePackage(), PetscDeviceContextCreate(), PetscDeviceCreate()
@*/
PetscErrorCode PetscDeviceInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscLikely(PetscDevicePackageInitialized)) PetscFunctionReturn(0);
  PetscCheck(PetscDeviceConfiguredFor_Internal(PETSC_DEVICE_DEFAULT),PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with device support (PETSC_DEVICE_DEFAULT = '%s')",PetscDeviceTypes[PETSC_DEVICE_DEFAULT]);
  PetscDevicePackageInitialized = PETSC_TRUE;
  PetscCall(PetscRegisterFinalize(PetscDeviceFinalizePackage));
  PetscFunctionReturn(0);
}
