#include "cupmdevice.hpp" /* I "petscdevice.h" */

using namespace Petsc;

#if PetscDefined(HAVE_CUDA)
static CUPMDevice<CUPMDeviceKind::CUDA> cudaDevice(PetscDeviceContextCreate_CUDA);
#endif
#if PetscDefined(HAVE_HIP)
static CUPMDevice<CUPMDeviceKind::HIP>  hipDevice(PetscDeviceContextCreate_HIP);
#endif

const char *const PetscDeviceKinds[] = {"invalid","cuda","hip","default","max","PetscDeviceKind","PETSC_DEVICE_",PETSC_NULLPTR};

/*@C
  PetscDeviceCreate - Get a new handle for a particular device kind

  Not Collective, Possibly Synchronous

  Input Parameter:
. kind - The kind of PetscDevice

  Output Parameter:
. device - The PetscDevice

  Notes:
  If this is the first time that a PetscDevice is created, this routine may initialize
  the corresponding backend. If this is the case, this will most likely cause some sort of
  device synchronization.

  Level: beginner

.seealso: PetscDeviceConfigure(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceCreate(PetscDeviceKind kind, PetscDevice *device)
{
  static PetscInt PetscDeviceCounter = 0;
  PetscDevice     dev;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidDeviceKind(kind,1);
  PetscValidPointer(device,2);
  ierr = PetscNew(&dev);CHKERRQ(ierr);
  dev->id   = PetscDeviceCounter++;
  dev->kind = kind;
  switch (kind) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    ierr = cudaDevice.getDevice(dev);CHKERRQ(ierr);
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    ierr = hipDevice.getDevice(dev);CHKERRQ(ierr);
    break;
#endif
  default:
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[kind],kind);
    break;
  }
  *device = dev;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceConfigure - Configure a particular PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice to Configure

  Developer Notes:
  Currently a no-op

  Level: developer

.seealso: PetscDeviceCreate(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceConfigure(PetscDevice device)
{
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  PetscValidDevice(device,1);
  switch (device->kind) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    ierr = cudaDevice.configureDevice(device);CHKERRQ(ierr);
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    ierr = hipDevice.configureDevice(device);CHKERRQ(ierr);
    break;
#endif
  default:
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[device->kind],device->kind);
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDestroy - Free a PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice

  Level: beginner

.seealso: PetscDeviceCreate(), PetscDeviceConfigure()
@*/
PetscErrorCode PetscDeviceDestroy(PetscDevice *device)
{
  PetscFunctionBegin;
  if (!*device) PetscFunctionReturn(0);
  if (!--(*device)->refcnt) {
    PetscErrorCode ierr;

    if (PetscUnlikelyDebug((*device)->refcnt < 0)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDevice %D reference count %D < 0",(*device)->id,(*device)->refcnt);
    ierr = PetscFree(*device);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscDevice defaultDevices[PETSC_DEVICE_MAX];

static PetscErrorCode InitializeDeviceHelper_Private(PetscDeviceKind kind, bool supported = false)
{
  const int      kindIdx = static_cast<int>(kind);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (supported) {
    /* on the off chance that someone fumbles calling this with INVALID or MAX */
    PetscValidDeviceKind(kind,1);
    ierr = PetscInfo1(NULL,"PetscDeviceKind %s supported, initializing\n",PetscDeviceKinds[kindIdx]);CHKERRQ(ierr);
    ierr = PetscDeviceCreate(kind,defaultDevices+kindIdx);CHKERRQ(ierr);
    ierr = PetscDeviceConfigure(defaultDevices[kindIdx]);CHKERRQ(ierr);
    /* the default devices are all automatically "referenced" at least once, otherwise the
       reference counting is off for them. We could alternatively increase the reference
       count when they are retrieved but that is a lot more brittle; whats to stop someone
       from doing thhe following?

       for (int i = 0; i < 10000; ++i) auto device = PetscDeviceDefault_Internal();
    */
    defaultDevices[kindIdx] = PetscDeviceReference(defaultDevices[kindIdx]);
  } else {
    ierr = PetscInfo1(NULL,"PetscDeviceKind %s not supported\n",PetscDeviceKinds[kindIdx]);CHKERRQ(ierr);
    defaultDevices[kindIdx] = PETSC_NULLPTR;
  }
  PetscFunctionReturn(0);
}

/* called from PetscFinalize() do not call yourself! */
static PetscErrorCode PetscDeviceFinalizeDefaultDevices_Private(void)
{
  const int      maxIdx = static_cast<int>(PETSC_DEVICE_MAX);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (int i = 0; i < maxIdx; ++i) {ierr = PetscDeviceDestroy(defaultDevices+i);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* called from PetscInitialize() do not call yourself! */
PetscErrorCode PetscDeviceInitializeDefaultDevices_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscRegisterFinalize(PetscDeviceFinalizeDefaultDevices_Private);CHKERRQ(ierr);
  ierr = InitializeDeviceHelper_Private(PETSC_DEVICE_INVALID);CHKERRQ(ierr);
  ierr = InitializeDeviceHelper_Private(PETSC_DEVICE_CUDA,PetscDefined(HAVE_CUDA));CHKERRQ(ierr);
  ierr = InitializeDeviceHelper_Private(PETSC_DEVICE_HIP,PetscDefined(HAVE_HIP));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Get the default PetscDevice for a particular kind, usually one should use
   PetscDeviceDefault_Internal() since that will return the automatically selected
   default kind. */
PetscDevice PetscDeviceDefaultKind_Internal(PetscDeviceKind kind)
{
  return defaultDevices[static_cast<int>(kind)];
}
