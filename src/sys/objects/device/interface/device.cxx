#include "cupmdevice.hpp" /* I "petscdevice.h" */
#include <petsc/private/petscadvancedmacros.h>

using namespace Petsc::Device;

/*
  note to anyone adding more classes, the name must be ALL_CAPS_SHORT_NAME + Device exactly to
  be picked up by the switch-case macros below
*/
#if PetscDefined(HAVE_CUDA)
static CUPM::Device<CUPM::DeviceType::CUDA> CUDADevice(PetscDeviceContextCreate_CUDA);
#endif
#if PetscDefined(HAVE_HIP)
static CUPM::Device<CUPM::DeviceType::HIP>  HIPDevice(PetscDeviceContextCreate_HIP);
#endif
#if PetscDefined(HAVE_SYCL)
#include "sycldevice.hpp"
static SYCL::Device                         SYCLDevice(PetscDeviceContextCreate_SYCL);
#endif

static_assert(Petsc::util::integral_value(PETSC_DEVICE_INVALID) == 0,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_CUDA)    == 1,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_HIP)     == 2,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_SYCL)    == 3,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_MAX)     == 4,"");
const char *const PetscDeviceTypes[] = {
  "invalid",
  "cuda",
  "hip",
  "sycl",
  "max",
  "PetscDeviceType",
  "PETSC_DEVICE_",
  PETSC_NULLPTR
};

static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_NONE)  == 0,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_LAZY)  == 1,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_EAGER) == 2,"");
const char *const PetscDeviceInitTypes[] = {
  "none",
  "lazy",
  "eager",
  "PetscDeviceInitType",
  "PETSC_DEVICE_INIT_",
  PETSC_NULLPTR
};
static_assert(
  sizeof(PetscDeviceInitTypes)/sizeof(*PetscDeviceInitTypes) == 6,
  "Must change CUPMDevice<T>::initialize number of enum values in -device_enable_cupm to match!"
);

#define PETSC_DEVICE_CASE(IMPLS,func,...)                                     \
  case PetscConcat_(PETSC_DEVICE_,IMPLS): {                                   \
    auto ierr_ = PetscConcat_(IMPLS,Device).func(__VA_ARGS__);CHKERRQ(ierr_); \
  } break

/*
  Suppose you have:

  CUDADevice.myFunction(arg1,arg2)

  that you would like to conditionally define and call in a switch-case:

  switch(PetscDeviceType) {
  #if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA: {
    auto ierr = CUDADevice.myFunction(arg1,arg2);CHKERRQ(ierr);
  } break;
  #endif
  }

  then calling this macro:

  PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,myFunction,arg1,arg2)

  will expand to the following case statement:

  case PETSC_DEVICE_CUDA: {
    auto ierr = CUDADevice.myFunction(arg1,arg2);CHKERRQ(ierr);
  } break

  if PetscDefined(HAVE_CUDA) evaluates to 1, and expand to nothing otherwise
*/
#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED(IMPLS,func,...)                                     \
  PetscIfPetscDefined(PetscConcat_(HAVE_,IMPLS),PETSC_DEVICE_CASE,PetscExpandToNothing)(IMPLS,func,__VA_ARGS__)

/*@C
  PetscDeviceCreate - Get a new handle for a particular device type

  Not Collective, Possibly Synchronous

  Input Parameter:
. type  - The type of PetscDevice
. devid - The numeric ID# of the device (pass PETSC_DECIDE to assign automatically)

  Output Parameter:
. device - The PetscDevice

  Notes:
  This routine may initialize PetscDevice. If this is the case, this will most likely cause
  some sort of device synchronization.

  devid is what you might pass to cudaSetDevice() for example.

  Level: beginner

.seealso: PetscDevice, PetscDeviceInitType, PetscDeviceInitialize(),
PetscDeviceInitialized(), PetscDeviceConfigure(), PetscDeviceView(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceCreate(PetscDeviceType type, PetscInt devid, PetscDevice *device)
{
  static PetscInt PetscDeviceCounter = 0;
  PetscDevice     dev;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidDeviceType(type,1);
  PetscValidPointer(device,3);
  ierr = PetscDeviceInitializePackage();CHKERRQ(ierr);
  ierr = PetscNew(&dev);CHKERRQ(ierr);
  dev->id     = PetscDeviceCounter++;
  dev->type   = type;
  dev->refcnt = 1;
  /*
    if you are adding a device, you also need to add it's initialization in
    PetscDeviceInitializeTypeFromOptions_Private() below
  */
  switch (type) {
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,getDevice,dev,devid);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HIP,getDevice,dev,devid);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(SYCL,getDevice,dev,devid);
  default:
    /* in case the above macros expand to nothing this silences any unused variable warnings */
    (void)(devid);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PETSc was seemingly configured for PetscDeviceType %s but we've fallen through all cases in a switch",PetscDeviceTypes[type]);
  }
  *device = dev;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDestroy - Free a PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice

  Level: beginner

.seealso: PetscDevice, PetscDeviceCreate(), PetscDeviceConfigure(), PetscDeviceView()
@*/
PetscErrorCode PetscDeviceDestroy(PetscDevice *device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*device) PetscFunctionReturn(0);
  PetscValidDevice(*device,1);
  ierr = PetscDeviceDereference_Internal(*device);CHKERRQ(ierr);
  if ((*device)->refcnt) {
    *device = PETSC_NULLPTR;
    PetscFunctionReturn(0);
  }
  ierr = PetscFree((*device)->data);CHKERRQ(ierr);
  ierr = PetscFree(*device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceConfigure - Configure a particular PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice to configure

  Notes:
  The user should not assume that this is a cheap operation

  Level: beginner

.seealso: PetscDevice, PetscDeviceCreate(), PetscDeviceView(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceConfigure(PetscDevice device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDevice(device,1);
  if (PetscDefined(USE_DEBUG)) {
    /*
      if no available configuration is available, this cascades all the way down to default
      and error
    */
    switch (device->type) {
    case PETSC_DEVICE_CUDA: if (PetscDefined(HAVE_CUDA)) break;
    case PETSC_DEVICE_HIP:  if (PetscDefined(HAVE_HIP))  break;
    case PETSC_DEVICE_SYCL: if (PetscDefined(HAVE_SYCL)) break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PETSc was seemingly configured for PetscDeviceType %s but we've fallen through all cases in a switch",PetscDeviceTypes[device->type]);
    }
  }
  ierr = (*device->ops->configure)(device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceView - View a PetscDevice

  Collective on viewer, Asynchronous

  Input Parameter:
+ device - The PetscDevice to view
- viewer - The PetscViewer to view the device with (NULL for PETSC_VIEWER_STDOUT_WORLD)

  Level: beginner

.seealso: PetscDevice, PetscDeviceCreate(), PetscDeviceConfigure(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceView(PetscDevice device, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDevice(device,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = (*device->ops->view)(device,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static std::array<bool,PETSC_DEVICE_MAX>        initializedDevice = {};
static std::array<PetscDevice,PETSC_DEVICE_MAX> defaultDevices    = {};
static_assert(initializedDevice.size() == defaultDevices.size(),"");

/*@C
  PetscDeviceInitialize - Initialize PetscDevice

  Not Collective, Possibly Synchronous

  Input Parameter:
. type - The PetscDeviceType to initialize

  Notes:
  Eagerly initializes the corresponding PetscDeviceType if needed.

  Level: beginner

.seealso: PetscDevice, PetscDeviceInitType, PetscDeviceInitialized(), PetscDeviceCreate(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceInitialize(PetscDeviceType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceType(type,1);
  ierr = PetscDeviceInitializeDefaultDevice_Internal(type,PETSC_DECIDE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceInitialized - Determines whether PetscDevice is initialized for a particular
  PetscDeviceType

  Not Collective, Asynchronous

  Input Parameter:
. type - The PetscDeviceType to check

  Output Parameter:
. [return value] - PETSC_TRUE if type is initialized, PETSC_FALSE otherwise

  Notes:
  If one has not configured PETSc for a particular PetscDeviceType then this routine will
  return PETSC_FALSE for that PetscDeviceType.

  Level: beginner

.seealso: PetscDevice, PetscDeviceInitType, PetscDeviceInitialize(), PetscDeviceCreate(), PetscDeviceDestroy()
@*/
PetscBool PetscDeviceInitialized(PetscDeviceType type)
{
  return static_cast<PetscBool>(PetscDeviceConfiguredFor_Internal(type) && initializedDevice[type]);
}

/*
  Actual intialization function; any functions claiming to initialize PetscDevice or
  PetscDeviceContext will have to run through this one
*/
PetscErrorCode PetscDeviceInitializeDefaultDevice_Internal(PetscDeviceType type, PetscInt defaultDeviceId)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceType(type,1);
  if (PetscLikely(PetscDeviceInitialized(type))) PetscFunctionReturn(0);
  if (PetscUnlikelyDebug(defaultDevices[type])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEM,"Trying to overwrite existing default device of type %s",PetscDeviceTypes[type]);
  ierr = PetscDeviceCreate(type,defaultDeviceId,&defaultDevices[type]);CHKERRQ(ierr);
  ierr = PetscDeviceConfigure(defaultDevices[type]);CHKERRQ(ierr);
  initializedDevice[type] = true;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceInitializeTypeFromOptions_Private(MPI_Comm comm, PetscDeviceType type, PetscInt defaultDeviceId, PetscBool defaultView, PetscDeviceInitType *defaultInitType)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PetscDeviceConfiguredFor_Internal(type)) {
    ierr = PetscInfo1(PETSC_NULLPTR,"PetscDeviceType %s not supported\n",PetscDeviceTypes[type]);CHKERRQ(ierr);
    defaultDevices[type] = PETSC_NULLPTR;
    PetscFunctionReturn(0);
  }
  ierr = PetscInfo1(PETSC_NULLPTR,"PetscDeviceType %s supported, initializing\n",PetscDeviceTypes[type]);CHKERRQ(ierr);
  /* ugly switch needed to pick the right global variable... could maybe do this as a union? */
  switch (type) {
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,initialize,comm,&defaultDeviceId,defaultInitType);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HIP,initialize,comm,&defaultDeviceId,defaultInitType);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(SYCL,initialize,comm,&defaultDeviceId,defaultInitType);
  default:
    SETERRQ1(comm,PETSC_ERR_PLIB,"PETSc was seemingly configured for PetscDeviceType %s but we've fallen through all cases in a switch",PetscDeviceTypes[type]);
  }
  /*
    defaultInitType and defaultDeviceId now represent what the individual TYPES have decided to
    initialize as
  */
  if (*defaultInitType == PETSC_DEVICE_INIT_EAGER) {
    ierr = PetscInfo1(PETSC_NULLPTR,"Eagerly initializing %s PetscDevice\n",PetscDeviceTypes[type]);CHKERRQ(ierr);
    ierr = PetscDeviceInitializeDefaultDevice_Internal(type,defaultDeviceId);CHKERRQ(ierr);
    if (defaultView) {
      PetscViewer vwr;

      ierr = PetscViewerASCIIGetStdout(comm,&vwr);CHKERRQ(ierr);
      ierr = PetscDeviceView(defaultDevices[type],vwr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* called from PetscFinalize() do not call yourself! */
static PetscErrorCode PetscDeviceFinalize_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    const auto PetscDeviceCheckAllDestroyedAfterFinalize = [](){
      PetscFunctionBegin;
      for (auto&& device : defaultDevices) {
        if (PetscUnlikely(device)) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_COR,"Device of type '%s' had reference count %" PetscInt_FMT " and was not fully destroyed during PetscFinalize()",PetscDeviceTypes[device->type],device->refcnt);
      }
      PetscFunctionReturn(0);
    };
    /* you might be thinking, why on earth are you registered yet another finalizer in a
     * function already called during PetscRegisterFinalizeAll()? If this seems stupid it's
     * because it is.
     *
     * The crux of the problem is that the initializer (and therefore the ~finalizer~) of
     * PetscDeviceContext is guaranteed to run after PetscDevice's. So if the global context
     * had a default PetscDevice attached, that PetscDevice will have a reference count >0 and
     * hence won't be destroyed yet. So we need to repeat the check that all devices have been
     * destroyed again ~after~ the global context is destroyed. In summary:
     *
     * 1. This finalizer runs and destroys all devices, except it may not because the global
     *    context may still hold a reference!
     * 2. The global context finalizer runs and does the final reference count decrement
     *    required, which actually destroys the held device.
     * 3. Our newly added finalizer runs and checks that all is well.
     */
    ierr = PetscRegisterFinalize(PetscDeviceCheckAllDestroyedAfterFinalize);CHKERRQ(ierr);
  }
  for (auto &&device : defaultDevices) {ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);}
  CHKERRCXX(initializedDevice.fill(false));
  PetscFunctionReturn(0);
}

/*
  Begins the init proceeedings for the entire PetscDevice stack. there are 3 stages of
  initialization types:

  1. defaultInitType - how does PetscDevice as a whole expect to initialize?
  2. subTypeDefaultInitType - how does each PetscDevice implementation expect to initialize?
     e.g. you may want to blanket disable PetscDevice init (and disable say Kokkos init), but
     have all CUDA devices still initialize.

  All told the following happens:

  0. defaultInitType -> LAZY
  1. Check for log_view/log_summary, if yes defaultInitType -> EAGER
  2. PetscDevice initializes each sub type with deviceDefaultInitType.
  2.1 Each enabled PetscDevice sub-type then does the above disable or view check in addition
      to checking for specific device init. if view or specific device init
      subTypeDefaultInitType -> EAGER. disabled once again overrides all.
*/
PetscErrorCode PetscDeviceInitializeFromOptions_Internal(MPI_Comm comm)
{
  PetscBool           flg,defaultView = PETSC_FALSE,initializeDeviceContextEagerly = PETSC_FALSE;
  PetscInt            defaultDevice   = PETSC_DECIDE;
  PetscDeviceType     deviceContextInitDevice = PETSC_DEVICE_DEFAULT;
  PetscDeviceInitType defaultInitType;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    int result;

    ierr = MPI_Comm_compare(comm,PETSC_COMM_WORLD,&result);CHKERRMPI(ierr);
    /* in order to accurately assign ranks to gpus we need to get the MPI_Comm_rank of the
     * global space */
    if (PetscUnlikely(result != MPI_IDENT)) {
      char name[MPI_MAX_OBJECT_NAME] = {};
      int  len; /* unused */

      ierr = MPI_Comm_get_name(comm,name,&len);CHKERRMPI(ierr);
      SETERRQ1(comm,PETSC_ERR_MPI,"Default devices being initialized on MPI_Comm '%s' not PETSC_COMM_WORLD",name);
    }
  }
  comm = PETSC_COMM_WORLD; /* from this point on we assume we're on PETSC_COMM_WORLD */
  ierr = PetscRegisterFinalize(PetscDeviceFinalize_Private);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULLPTR,PETSC_NULLPTR,"-log_view",&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscOptionsHasName(PETSC_NULLPTR,PETSC_NULLPTR,"-log_summary",&flg);CHKERRQ(ierr);
  }
  {
    PetscInt initIdx = flg ? PETSC_DEVICE_INIT_EAGER : PETSC_DEVICE_INIT_LAZY;

    ierr = PetscOptionsBegin(comm,PETSC_NULLPTR,"PetscDevice Options","Sys");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-device_enable","How (or whether) to initialize PetscDevices","PetscDeviceInitializeFromOptions_Internal()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[initIdx],&initIdx,PETSC_NULLPTR);CHKERRQ(ierr);
    ierr = PetscOptionsRangeInt("-device_select","Which device to use. Pass " PetscStringize(PETSC_DECIDE) " to have PETSc decide or (given they exist) [0-NUM_DEVICE) for a specific device","PetscDeviceCreate()",defaultDevice,&defaultDevice,PETSC_NULLPTR,PETSC_DECIDE,std::numeric_limits<int>::max());CHKERRQ(ierr);
    ierr = PetscOptionsBool("-device_view","Display device information and assignments (forces eager initialization)",PETSC_NULLPTR,defaultView,&defaultView,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (initIdx == PETSC_DEVICE_INIT_NONE) {
      /* disabled all device initialization if devices are globally disabled */
      if (PetscUnlikely(defaultDevice != PETSC_DECIDE)) SETERRQ(comm,PETSC_ERR_USER_INPUT,"You have disabled devices but also specified a particular device to use, these options are mutually exlusive");
      defaultView = PETSC_FALSE;
    } else {
      defaultView = static_cast<decltype(defaultView)>(defaultView && flg);
      if (defaultView) initIdx = PETSC_DEVICE_INIT_EAGER;
    }
    defaultInitType = static_cast<decltype(defaultInitType)>(initIdx);
  }
  static_assert((PETSC_DEVICE_INVALID == 0) && (PETSC_DEVICE_MAX < std::numeric_limits<int>::max()),"");
  for (int i = 1; i < PETSC_DEVICE_MAX; ++i) {
    const auto deviceType = static_cast<PetscDeviceType>(i);
    auto initType         = defaultInitType;

    ierr = PetscDeviceInitializeTypeFromOptions_Private(comm,deviceType,defaultDevice,defaultView,&initType);CHKERRQ(ierr);
    if (PetscDeviceConfiguredFor_Internal(deviceType) && (initType == PETSC_DEVICE_INIT_EAGER)) {
      initializeDeviceContextEagerly = PETSC_TRUE;
      deviceContextInitDevice        = deviceType;
    }
  }
  if (initializeDeviceContextEagerly) {
    PetscDeviceContext dctx;

    /*
      somewhat inefficient here as the device context is potentially fully set up twice (once
      when retrieved then the second time if setfromoptions makes changes)
    */
    ierr = PetscInfo1(PETSC_NULLPTR,"Eagerly initializing PetscDeviceContext with %s device\n",PetscDeviceTypes[deviceContextInitDevice]);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetRootDeviceType_Internal(deviceContextInitDevice);CHKERRQ(ierr);
    ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetFromOptions(comm,"root_",dctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetUp(dctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Get the default PetscDevice for a particular type and constructs them if lazily initialized. */
PetscErrorCode PetscDeviceGetDefaultForType_Internal(PetscDeviceType type, PetscDevice *device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(device,2);
  ierr = PetscDeviceInitialize(type);CHKERRQ(ierr);
  *device = defaultDevices[type];
  PetscFunctionReturn(0);
}
