#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PetscStreamTypes[] = {"global_blocking","default_blocking","global_nonblocking","max","PetscStreamType","PETSC_STREAM_",PETSC_NULLPTR};

const char *const PetscDeviceContextJoinModes[] = {"destroy","sync","no_sync","PetscDeviceContextJoinMode","PETSC_DEVICE_CONTEXT_JOIN_",PETSC_NULLPTR};

static PetscBool PetscDevicePackageInitialized = PETSC_FALSE;

/*@C
  PetscDeviceFinalizePackage - This function cleans up all components of the PetscDevice package.
  It is called from PetscFinalize().

  Developer Notes:
  This function is automatically registered to be called during PetscFinalize() by PetscDeviceInitializePackage() so
  there should be no need to call it yourself.

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
  PetscDeviceInitializePackage - This function initializes everything in the PetscDevice package. It is called from
  PetscDLLibraryRegister_petscsys() when using dynamic libraries, and on the first call to PetscDeviceContextCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscDeviceFinalizePackage(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PETSC_DEVICE_DEFAULT == PETSC_DEVICE_INVALID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"No suitable PetscDeviceKind found, must configure PETSc with a device backend enabled");
  if (PetscDevicePackageInitialized) PetscFunctionReturn(0);
  PetscDevicePackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscDeviceFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
