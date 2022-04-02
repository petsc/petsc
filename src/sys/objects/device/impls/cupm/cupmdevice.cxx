#include "../../interface/cupmdevice.hpp"
#include <algorithm>
#include <csetjmp> // for cuda mpi awareness
#include <csignal> // SIGSEGV
#include <iterator>
#include <type_traits>

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#else
#define PetscLogInitialize() 0
#endif

namespace Petsc
{

namespace Device
{

namespace CUPM
{

// internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <DeviceType T>
class Device<T>::DeviceInternal
{
  const int        id_;
  bool             devInitialized_ = false;
  cupmDeviceProp_t dprop_; // cudaDeviceProp appears to be an actual struct, i.e. you can't
                           // initialize it with nullptr or NULL (i've tried)

  PETSC_CXX_COMPAT_DECL(bool CUPMAwareMPI_());

public:
  // default constructor
  explicit constexpr DeviceInternal(int dev) noexcept : id_(dev) { }

  // gather all relevant information for a particular device, a cupmDeviceProp_t is
  // usually sufficient here
  PETSC_NODISCARD PetscErrorCode initialize() noexcept;
  PETSC_NODISCARD PetscErrorCode configure() noexcept;
  PETSC_NODISCARD PetscErrorCode view(PetscViewer) const noexcept;
  PETSC_NODISCARD PetscErrorCode finalize() noexcept;

  PETSC_NODISCARD auto id()          const -> decltype(id_)             { return id_;             }
  PETSC_NODISCARD auto initialized() const -> decltype(devInitialized_) { return devInitialized_; }
  PETSC_NODISCARD auto prop()        const -> const decltype(dprop_)&   { return dprop_;          }

  // factory
  PETSC_CXX_COMPAT_DECL(std::unique_ptr<DeviceInternal> makeDevice(int i))
  {
    return std::unique_ptr<DeviceInternal>(new DeviceInternal(i));
  }
};

// the goal here is simply to get the cupm backend to create its context, not to do any type of
// modification of it, or create objects (since these may be affected by subsequent
// configuration changes)
template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::initialize() noexcept
{
  PetscFunctionBegin;
  if (devInitialized_) PetscFunctionReturn(0);
  devInitialized_ = true;
  // need to do this BEFORE device has been set, although if the user
  // has already done this then we just ignore it
  if (cupmSetDeviceFlags(cupmDeviceMapHost) == cupmErrorSetOnActiveProcess) {
    // reset the error if it was cupmErrorSetOnActiveProcess
    const auto PETSC_UNUSED unused = cupmGetLastError();
  } else {PetscCallCUPM(cupmGetLastError());}
  // cuda 5.0+ will create a context when cupmSetDevice is called
  if (cupmSetDevice(id_) != cupmErrorDeviceAlreadyInUse) PetscCallCUPM(cupmGetLastError());
  // forces cuda < 5.0 to initialize a context
  PetscCallCUPM(cupmFree(nullptr));
  // where is this variable defined and when is it set? who knows! but it is defined and set
  // at this point. either way, each device must make this check since I guess MPI might not be
  // aware of all of them?
  if (use_gpu_aware_mpi) {
    // For OpenMPI, we could do a compile time check with
    // "defined(PETSC_HAVE_OMPI_MAJOR_VERSION) && defined(MPIX_CUDA_AWARE_SUPPORT) &&
    // MPIX_CUDA_AWARE_SUPPORT" to see if it is CUDA-aware. However, recent versions of IBM
    // Spectrum MPI (e.g., 10.3.1) on Summit meet above conditions, but one has to use jsrun
    // --smpiargs=-gpu to really enable GPU-aware MPI. So we do the check at runtime with a
    // code that works only with GPU-aware MPI.
    if (PetscUnlikely(!CUPMAwareMPI_())) {
      (*PetscErrorPrintf)("PETSc is configured with GPU support, but your MPI is not GPU-aware. For better performance, please use a GPU-aware MPI.\n");
      (*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n");
      (*PetscErrorPrintf)("If you do care, for IBM Spectrum MPI on OLCF Summit, you may need jsrun --smpiargs=-gpu.\n");
      (*PetscErrorPrintf)("For OpenMPI, you need to configure it --with-cuda (https://www.open-mpi.org/faq/?category=buildcuda)\n");
      (*PetscErrorPrintf)("For MVAPICH2-GDR, you need to set MV2_USE_CUDA=1 (http://mvapich.cse.ohio-state.edu/userguide/gdr/)\n");
      (*PetscErrorPrintf)("For Cray-MPICH, you need to set MPICH_RDMA_ENABLED_CUDA=1 (https://www.olcf.ornl.gov/tutorials/gpudirect-mpich-enabled-cuda/)\n");
      PETSCABORT(PETSC_COMM_SELF,PETSC_ERR_LIB);
    }
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::configure() noexcept
{
  PetscFunctionBegin;
  PetscAssert(devInitialized_,PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being configured before it was initialized",id_);
  // why on EARTH nvidia insists on making otherwise informational states into
  // fully-fledged error codes is beyond me. Why couldn't a pointer to bool argument have
  // sufficed?!?!?!
  if (cupmSetDevice(id_) != cupmErrorDeviceAlreadyInUse) PetscCallCUPM(cupmGetLastError());
  // need to update the device properties
  PetscCallCUPM(cupmGetDeviceProperties(&dprop_,id_));
  PetscCall(PetscInfo(nullptr,"Configured device %d\n",id_));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::view(PetscViewer viewer) const noexcept
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscAssert(devInitialized_,PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being viewed before it was initialized or configured",id_);
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(viewer),PETSCVIEWERASCII,&iascii));
  if (iascii) {
    MPI_Comm    comm;
    PetscMPIInt rank;
    PetscViewer sviewer;

    PetscCall(PetscObjectGetComm(PetscObjectCast(viewer),&comm));
    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"[%d] device %d: %s\n",rank,id_,dprop_.name));
    PetscCall(PetscViewerASCIIPushTab(sviewer));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Compute capability: %d.%d\n",dprop_.major,dprop_.minor));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Multiprocessor Count: %d\n",dprop_.multiProcessorCount));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Maximum Grid Dimensions: %d x %d x %d\n",dprop_.maxGridSize[0],dprop_.maxGridSize[1],dprop_.maxGridSize[2]));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Maximum Block Dimensions: %d x %d x %d\n",dprop_.maxThreadsDim[0],dprop_.maxThreadsDim[1],dprop_.maxThreadsDim[2]));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Maximum Threads Per Block: %d\n",dprop_.maxThreadsPerBlock));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Warp Size: %d\n",dprop_.warpSize));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Total Global Memory (bytes): %zu\n",dprop_.totalGlobalMem));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Total Constant Memory (bytes): %zu\n",dprop_.totalConstMem));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Shared Memory Per Block (bytes): %zu\n",dprop_.sharedMemPerBlock));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Multiprocessor Clock Rate (KHz): %d\n",dprop_.clockRate));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Memory Clock Rate (KHz): %d\n",dprop_.memoryClockRate));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Memory Bus Width (bits): %d\n",dprop_.memoryBusWidth));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Peak Memory Bandwidth (GB/s): %f\n",2.0*dprop_.memoryClockRate*(dprop_.memoryBusWidth/8)/1.0e6));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Can map host memory: %s\n",dprop_.canMapHostMemory ? "PETSC_TRUE" : "PETSC_FALSE"));
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Can execute multiple kernels concurrently: %s\n",dprop_.concurrentKernels ? "PETSC_TRUE" : "PETSC_FALSE"));
    PetscCall(PetscViewerASCIIPopTab(sviewer));
    PetscCall(PetscViewerFlush(sviewer));
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    PetscCall(PetscViewerFlush(viewer));
  }
  PetscFunctionReturn(0);
}

static std::jmp_buf cupmMPIAwareJumpBuffer;
static bool         cupmMPIAwareJumpBufferSet;

// godspeed to anyone that attempts to call this function
void SilenceVariableIsNotNeededAndWillNotBeEmittedWarning_ThisFunctionShouldNeverBeCalled()
{
  PETSCABORT(MPI_COMM_NULL,INT_MAX);
  if (cupmMPIAwareJumpBufferSet) (void)cupmMPIAwareJumpBuffer;
}

#define CHKCUPMAWARE(...) do {                                  \
    cupmError_t cerr_ = __VA_ARGS__;                            \
    if (PetscUnlikely(cerr_ != cupmSuccess)) return false;      \
  } while (0)

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(bool Device<T>::DeviceInternal::CUPMAwareMPI_())
{
  constexpr int  bufSize = 2;
  constexpr int  hbuf[bufSize] = {1,0};
  int            *dbuf = nullptr;
  constexpr auto bytes = bufSize*sizeof(*dbuf);
  auto           awareness = false;
  const auto     cupmSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
    if ((signal == SIGSEGV) && cupmMPIAwareJumpBufferSet) std::longjmp(cupmMPIAwareJumpBuffer,1);
    return PetscSignalHandlerDefault(signal,ptr);
  };

  PetscFunctionBegin;
  CHKCUPMAWARE(cupmMalloc(reinterpret_cast<void**>(&dbuf),bytes));
  CHKCUPMAWARE(cupmMemcpy(dbuf,hbuf,bytes,cupmMemcpyHostToDevice));
  PetscCallAbort(PETSC_COMM_SELF,PetscPushSignalHandler(cupmSignalHandler,nullptr));
  cupmMPIAwareJumpBufferSet = true;
  if (setjmp(cupmMPIAwareJumpBuffer)) {
    // if a segv was triggered in the MPI_Allreduce below, it is very likely due to MPI not
    // being GPU-aware
    awareness = false;
    // control flow up until this point:
    // 1. CUPMDevice<T>::CUPMDeviceInternal::MPICUPMAware__()
    // 2. MPI_Allreduce
    // 3. SIGSEGV
    // 4. PetscSignalHandler_Private
    // 5. cupmSignalHandler (lambda function)
    // 6. here
    // PetscSignalHandler_Private starts with PetscFunctionBegin and is pushed onto the stack
    // so we must undo this. This would be most naturally done in cupmSignalHandler, however
    // the C/C++ standard dictates:
    //
    // After invoking longjmp(), non-volatile-qualified local objects should not be accessed if
    // their values could have changed since the invocation of setjmp(). Their value in this
    // case is considered indeterminate, and accessing them is undefined behavior.
    //
    // so for safety (since we don't know what PetscStackPop may try to read/declare) we do it
    // outside of the longjmp control flow
    PetscStackPop;
  } else if (!MPI_Allreduce(dbuf,dbuf+1,1,MPI_INT,MPI_SUM,PETSC_COMM_SELF)) awareness = true;
  cupmMPIAwareJumpBufferSet = false;
  PetscCallAbort(PETSC_COMM_SELF,PetscPopSignalHandler());
  CHKCUPMAWARE(cupmFree(dbuf));
  PetscFunctionReturn(awareness);
}

#undef CHKCUPMAWARE

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::finalize() noexcept
{
  PetscFunctionBegin;
  devInitialized_ = false;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::finalize_() noexcept
{
  PetscFunctionBegin;
  if (!initialized_) PetscFunctionReturn(0);
  for (auto&& device : devices_) {
    if (device) {
      PetscCall(device->finalize());
      device.reset();
    }
  }
  defaultDevice_ = PETSC_CUPM_DEVICE_NONE;  // disabled by default
  initialized_   = false;
  PetscFunctionReturn(0);
}

// these functions should be named identically to the option they produce where "CUPMTYPE" and
// "cupmtype" are the uppercase and lowercase string versions of the cupm backend respectively
template <DeviceType T>
PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* PetscDevice_CUPMTYPE_Options())
{
  switch (T) {
  case DeviceType::CUDA: return "PetscDevice CUDA Options";
  case DeviceType::HIP:  return "PetscDevice HIP Options";
  }
  PetscUnreachable();
  return "PETSC_ERROR_PLIB";
}

template <DeviceType T>
PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* device_enable_cupmtype())
{
  switch (T) {
  case DeviceType::CUDA: return "-device_enable_cuda";
  case DeviceType::HIP:  return "-device_enable_hip";
  }
  PetscUnreachable();
  return "PETSC_ERROR_PLIB";
}

template <DeviceType T>
PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* device_select_cupmtype())
{
  switch (T) {
  case DeviceType::CUDA: return "-device_select_cuda";
  case DeviceType::HIP:  return "-device_select_hip";
  }
  PetscUnreachable();
  return "PETSC_ERROR_PLIB";
}

template <DeviceType T>
PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* device_view_cupmtype())
{
  switch (T) {
  case DeviceType::CUDA: return "-device_view_cuda";
  case DeviceType::HIP:  return "-device_view_hip";
  }
  PetscUnreachable();
  return "PETSC_ERROR_PLIB";
}

template <DeviceType T>
PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* CUPM_VISIBLE_DEVICES())
{
  switch (T) {
  case DeviceType::CUDA: return "CUDA_VISIBLE_DEVICES";
  case DeviceType::HIP:  return "HIP_VISIBLE_DEVICES";
  }
  PetscUnreachable();
  return "PETSC_ERROR_PLIB";
}

template <DeviceType T>
PetscErrorCode Device<T>::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscDeviceInitType *defaultInitType) noexcept
{
  PetscInt    initTypeCUPM = *defaultInitType,id = *defaultDeviceId;
  PetscBool   view         = PETSC_FALSE,flg;
  int         ndev;
  cupmError_t cerr;

  PetscFunctionBegin;
  if (initialized_) PetscFunctionReturn(0);
  initialized_ = true;
  PetscCall(PetscRegisterFinalize(finalize_));

  {
    // the functions to populate the command line strings are named after the string they return
    PetscOptionsBegin(comm,nullptr,PetscDevice_CUPMTYPE_Options<T>(),"Sys");
    PetscCall(PetscOptionsEList(device_enable_cupmtype<T>(),"How (or whether) to initialize a device","CUPMDevice<CUPMDeviceType>::initialize()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[initTypeCUPM],&initTypeCUPM,nullptr));
    PetscCall(PetscOptionsRangeInt(device_select_cupmtype<T>(),"Which device to use. Pass " PetscStringize(PETSC_DECIDE) " to have PETSc decide or (given they exist) [0-NUM_DEVICE) for a specific device","PetscDeviceCreate",id,&id,nullptr,PETSC_DECIDE,std::numeric_limits<decltype(defaultDevice_)>::max()));
    PetscCall(PetscOptionsBool(device_view_cupmtype<T>(),"Display device information and assignments (forces eager initialization)",nullptr,view,&view,&flg));
    PetscOptionsEnd();
  }

  cerr = cupmGetDeviceCount(&ndev);
  switch (cerr) {
  case cupmErrorNoDevice: {
    PetscBool found;
    PetscBool ignoreCupmError = PETSC_FALSE;
    char      buf[16];

    PetscCall(PetscOptionsGetenv(comm,CUPM_VISIBLE_DEVICES<T>(),buf,sizeof(buf),&found));
    if (found) {
      size_t len;

      PetscCall(PetscStrlen(buf,&len));
      if (!len || buf[0] == '-') ignoreCupmError = PETSC_TRUE;
    }
    id = PETSC_CUPM_DEVICE_NONE; // there are no devices anyway
    if (ignoreCupmError) {
      initTypeCUPM = PETSC_DEVICE_INIT_NONE;
      auto PETSC_UNUSED ignored = cupmGetLastError();
      break;
    }
    // if we don't outright ignore the error we then drop and check if the user tried to
    // eagerly initialize the device
  }
  case cupmErrorStubLibrary:
    if (PetscUnlikely((initTypeCUPM == PETSC_DEVICE_INIT_EAGER) || (view && flg))) {
      const auto name    = cupmGetErrorName(cerr);
      const auto desc    = cupmGetErrorString(cerr);
      const auto backend = cupmName();
      SETERRQ(comm,PETSC_ERR_USER_INPUT,"Cannot eagerly initialize %s, as doing so results in %s error %d (%s) : %s",backend,backend,static_cast<PetscErrorCode>(cerr),name,desc);
    }
    initTypeCUPM = PETSC_DEVICE_INIT_NONE;
    {auto PETSC_UNUSED ignored = cupmGetLastError();}
    break;
  default:
    PetscCallCUPM(cerr);
    break;
  }

  if (initTypeCUPM == PETSC_DEVICE_INIT_NONE) {
    if ((id > 0) || (id == PETSC_DECIDE)) id = PETSC_CUPM_DEVICE_NONE;
  } else {
    PetscCall(PetscDeviceCheckDeviceCount_Internal(ndev));
    if (id == PETSC_DECIDE) {
      if (ndev) {
        PetscMPIInt rank;

        PetscCallMPI(MPI_Comm_rank(comm,&rank));
        id   = rank % ndev;
      } else id = 0;
    }
    view = static_cast<decltype(view)>(view && flg);
    if (view) initTypeCUPM = PETSC_DEVICE_INIT_EAGER;
  }

  static_assert(std::is_same<PetscMPIInt,decltype(defaultDevice_)>::value,"");
  // id is PetscInt, _defaultDevice is int
  PetscCall(PetscMPIIntCast(id,&defaultDevice_));
  if (initTypeCUPM == PETSC_DEVICE_INIT_EAGER) {
    devices_[defaultDevice_] = DeviceInternal::makeDevice(defaultDevice_);
    PetscCall(devices_[defaultDevice_]->initialize());
    PetscCall(devices_[defaultDevice_]->configure());
    if (view) {
      PetscViewer vwr;

      PetscCall(PetscLogInitialize());
      PetscCall(PetscViewerASCIIGetStdout(comm,&vwr));
      PetscCall(devices_[defaultDevice_]->view(vwr));
    }
  }

  // record the results of the initialization
  *defaultInitType = static_cast<PetscDeviceInitType>(initTypeCUPM);
  *defaultDeviceId = id;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::getDevice(PetscDevice device, PetscInt id) const noexcept
{
  const auto cerr = static_cast<cupmError_t>(-defaultDevice_);

  PetscFunctionBegin;
  PetscCheck(defaultDevice_ != PETSC_CUPM_DEVICE_NONE,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Trying to retrieve a %s PetscDevice when it has been disabled",cupmName());
  PetscCheck(defaultDevice_ >= 0,PETSC_COMM_SELF,PETSC_ERR_GPU,"Cannot lazily initialize PetscDevice: %s error %d (%s) : %s",cupmName(),static_cast<PetscErrorCode>(cerr),cupmGetErrorName(cerr),cupmGetErrorString(cerr));
  if (id == PETSC_DECIDE) id = defaultDevice_;
  PetscAssert(static_cast<decltype(devices_.size())>(id) < devices_.size(),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only supports %zu number of devices but trying to get device with id %" PetscInt_FMT,devices_.size(),id);
  if (devices_[id]) {
    PetscAssert(id == devices_[id]->id(),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Entry %" PetscInt_FMT " contains device with mismatching id %d",id,devices_[id]->id());
  } else devices_[id] = DeviceInternal::makeDevice(id);
  PetscCall(devices_[id]->initialize());
  device->deviceId           = devices_[id]->id(); // technically id = _devices[id]->_id here
  device->ops->createcontext = create_;
  device->ops->configure     = this->configureDevice;
  device->ops->view          = this->viewDevice;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::configureDevice(PetscDevice device) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[device->deviceId]->configure());
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscFunctionBegin;
  // now this __shouldn't__ reconfigure the device, but there is a petscinfo call to indicate
  // it is being reconfigured
  PetscCall(devices_[device->deviceId]->configure());
  PetscCall(devices_[device->deviceId]->view(viewer));
  PetscFunctionReturn(0);
}

// explicitly instantiate the classes
#if PetscDefined(HAVE_CUDA)
template class Device<DeviceType::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template class Device<DeviceType::HIP>;
#endif

} // namespace CUPM

} // namespace Device

} // namespace Petsc
