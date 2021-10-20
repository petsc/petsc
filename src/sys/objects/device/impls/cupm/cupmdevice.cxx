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

// internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <CUPMDeviceType T>
class CUPMDevice<T>::CUPMDeviceInternal
{
  const int        _id;
  bool             _devInitialized = false;
  cupmDeviceProp_t _dprop; // cudaDeviceProp appears to be an actual struct, i.e. you can't
                           // initialize it with nullptr or NULL (i've tried)

  PETSC_NODISCARD static bool __CUPMAwareMPI() noexcept;

public:
  // default constructor
  explicit constexpr CUPMDeviceInternal(int dev) noexcept : _id(dev) { }

  // gather all relevant information for a particular device, a cupmDeviceProp_t is
  // usually sufficient here
  PETSC_NODISCARD PetscErrorCode initialize() noexcept;
  PETSC_NODISCARD PetscErrorCode configure() noexcept;
  PETSC_NODISCARD PetscErrorCode view(PetscViewer) const noexcept;
  PETSC_NODISCARD PetscErrorCode finalize() noexcept;

  PETSC_NODISCARD auto id() const -> decltype(_id) { return _id; }
  PETSC_NODISCARD auto initialized() const -> decltype(_devInitialized) { return _devInitialized; }
  PETSC_NODISCARD auto prop() const -> const decltype(_dprop)& { return _dprop; }

  // factory
  static constexpr std::unique_ptr<CUPMDeviceInternal> makeDevice(int i) noexcept
  {
    return std::unique_ptr<CUPMDeviceInternal>(new CUPMDeviceInternal(i));
  }
};

// the goal here is simply to get the cupm backend to create its context, not to do any type of
// modification of it, or create objects (since these may be affected by subsequent
// configuration changes)
template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::CUPMDeviceInternal::initialize() noexcept
{
  cupmError_t cerr;

  PetscFunctionBegin;
  if (_devInitialized) PetscFunctionReturn(0);
  _devInitialized = true;
  // need to do this BEFORE device has been set, although if the user
  // has already done this then we just ignore it
  if (cupmSetDeviceFlags(cupmDeviceMapHost) == cupmErrorSetOnActiveProcess) {
    // reset the error if it was cupmErrorSetOnActiveProcess
    const auto PETSC_UNUSED unused = cupmGetLastError();
  } else {CHKERRCUPM(cupmGetLastError());}
  // cuda 5.0+ will create a context when cupmSetDevice is called
  if (cupmSetDevice(_id) != cupmErrorDeviceAlreadyInUse) CHKERRCUPM(cupmGetLastError());
  // forces cuda < 5.0 to initialize a context
  cerr = cupmFree(nullptr);CHKERRCUPM(cerr);
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
    if (PetscUnlikely(!__CUPMAwareMPI())) {
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

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::CUPMDeviceInternal::configure() noexcept
{
  cupmError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!_devInitialized)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being configured before it was initialized",_id);
  // why on EARTH nvidia insists on making otherwise informational states into
  // fully-fledged error codes is beyond me. Why couldn't a pointer to bool argument have
  // sufficed?!?!?!
  if (cupmSetDevice(_id) != cupmErrorDeviceAlreadyInUse) CHKERRCUPM(cupmGetLastError());
  // need to update the device properties
  cerr = cupmGetDeviceProperties(&_dprop,_id);CHKERRCUPM(cerr);
  ierr = PetscInfo1(nullptr,"Configured device %d\n",_id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::CUPMDeviceInternal::view(PetscViewer viewer) const noexcept
{
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!_devInitialized)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being viewed before it was initialized or configured",_id);
  ierr = PetscObjectTypeCompare(reinterpret_cast<PetscObject>(viewer),PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectGetComm(reinterpret_cast<PetscObject>(viewer),&comm);CHKERRQ(ierr);
  if (PetscUnlikely(!iascii)) SETERRQ(comm,PETSC_ERR_SUP,"Only PetscViewer of type PETSCVIEWERASCII is supported");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] device %d: %s\n",rank,_id,_dprop.name);CHKERRQ(ierr);
  // flush the assignment information
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Compute capability: %d.%d\n",_dprop.major,_dprop.minor);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Multiprocessor Count: %d\n",_dprop.multiProcessorCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Maximum Grid Dimensions: %d x %d x %d\n",_dprop.maxGridSize[0],_dprop.maxGridSize[1],_dprop.maxGridSize[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Maximum Block Dimensions: %d x %d x %d\n",_dprop.maxThreadsDim[0],_dprop.maxThreadsDim[1],_dprop.maxThreadsDim[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Maximum Threads Per Block: %d\n",_dprop.maxThreadsPerBlock);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Warp Size: %d\n",_dprop.warpSize);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Total Global Memory (bytes): %zu\n",_dprop.totalGlobalMem);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Total Constant Memory (bytes): %zu\n",_dprop.totalConstMem);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Shared Memory Per Block (bytes): %zu\n",_dprop.sharedMemPerBlock);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Multiprocessor Clock Rate (KHz): %d\n",_dprop.clockRate);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Memory Clock Rate (KHz): %d\n",_dprop.memoryClockRate);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Memory Bus Width (bits): %d\n",_dprop.memoryBusWidth);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Peak Memory Bandwidth (GB/s): %f\n",2.0*_dprop.memoryClockRate*(_dprop.memoryBusWidth/8)/1.0e6);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Can map host memory: %s\n",_dprop.canMapHostMemory ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Can execute multiple kernels concurrently: %s\n",_dprop.concurrentKernels ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
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

#define CHKCUPMAWARE(expr) if (PetscUnlikely((expr) != cupmSuccess)) return false;

template <CUPMDeviceType T>
bool CUPMDevice<T>::CUPMDeviceInternal::__CUPMAwareMPI() noexcept
{
  constexpr int  bufSize = 2;
  constexpr int  hbuf[bufSize] = {1,0};
  int            *dbuf = nullptr;
  bool           awareness = false;
  cupmError_t    cerr;
  PetscErrorCode ierr;
  const auto     cupmSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
    if ((signal == SIGSEGV) && cupmMPIAwareJumpBufferSet) std::longjmp(cupmMPIAwareJumpBuffer,1);
    return PetscSignalHandlerDefault(signal,ptr);
  };

  PetscFunctionBegin;
  cerr = cupmMalloc(reinterpret_cast<void**>(&dbuf),sizeof(*dbuf)*bufSize);CHKCUPMAWARE(cerr);
  cerr = cupmMemcpy(dbuf,hbuf,sizeof(*dbuf)*bufSize,cupmMemcpyHostToDevice);CHKCUPMAWARE(cerr);
  ierr = PetscPushSignalHandler(cupmSignalHandler,nullptr);CHKERRABORT(PETSC_COMM_SELF,ierr);
  cupmMPIAwareJumpBufferSet = true;
  if (setjmp(cupmMPIAwareJumpBuffer)) {
    // if a segv was triggered in the MPI_Allreduce below, it is very likely due to MPI not
    // being GPU-aware
    awareness = false;
    // control flow up until this point:
    // 1. CUPMDevice<T>::CUPMDeviceInternal::__MPICUPMAware()
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
  ierr = PetscPopSignalHandler();CHKERRABORT(PETSC_COMM_SELF,ierr);
  cerr = cupmFree(dbuf);CHKCUPMAWARE(cerr);
  PetscFunctionReturn(awareness);
}

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::CUPMDeviceInternal::finalize() noexcept
{
  PetscFunctionBegin;
  _devInitialized = false;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::__finalize() noexcept
{
  PetscFunctionBegin;
  if (!_initialized) PetscFunctionReturn(0);
  for (auto&& device : _devices) {
    if (device) {
      const auto ierr = device->finalize();CHKERRQ(ierr);
      device.reset();
    }
  }
  _defaultDevice = PETSC_CUPM_DEVICE_NONE;  // disabled by default
  _initialized   = false;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
static constexpr const std::array<const char*const,4> cupmOptions() noexcept;

#define CAT_(x,y) x ## y
#define CAT(x,y)  CAT_(x,y)

// PetscDefined(HAVE_TYPE) = 0 -> expands to nothing
#define CUPM_DECLARE_OPTIONS_IF_PETSC_DEFINED_0(TYPE,type)
// PetscDefined(HAVE_TYPE) = 1 -> expands to the function
#define CUPM_DECLARE_OPTIONS_IF_PETSC_DEFINED_1(TYPE,type)              \
  template <>                                                           \
  constexpr const std::array<const char*const,4>                        \
  cupmOptions<CUPMDeviceType::TYPE>() noexcept                          \
  {                                                                     \
    return {                                                            \
      "PetscDevice " PetscStringize(TYPE) " Options",                   \
      "-device_enable_" PetscStringize(type),                           \
      "-device_select_" PetscStringize(type),                           \
      "-device_view_" PetscStringize(type)                              \
    };                                                                  \
  }

// expands to either the cupmOptions function or nothing, we have to do this with macros
// because for all the lovely compile time features c++ provides, the one thing it can't do is
// compile time string concatenation.
#define CUPM_DECLARE_OPTIONS_IF_PETSC_DEFINED(TYPE,type)                \
  CAT(CUPM_DECLARE_OPTIONS_IF_PETSC_DEFINED_,PetscDefined(CAT(HAVE_,TYPE)))(TYPE,type)

CUPM_DECLARE_OPTIONS_IF_PETSC_DEFINED(CUDA,cuda);
CUPM_DECLARE_OPTIONS_IF_PETSC_DEFINED(HIP,hip);

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscDeviceInitType *defaultInitType) noexcept
{
  PetscInt       initTypeCUPM = *defaultInitType,id = *defaultDeviceId;
  PetscBool      view = PETSC_FALSE,flg;
  cupmError_t    cerr;
  int            ndev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (_initialized) PetscFunctionReturn(0);
  _initialized = true;
  ierr = PetscRegisterFinalize(__finalize);CHKERRQ(ierr);

  {
    constexpr const auto options = cupmOptions<T>();

    ierr = PetscOptionsBegin(comm,nullptr,std::get<0>(options),"Sys");CHKERRQ(ierr);
    ierr = PetscOptionsEList(std::get<1>(options),"How (or whether) to initialize a device","CUPMDevice<CUPMDeviceType>::initialize()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[initTypeCUPM],&initTypeCUPM,nullptr);CHKERRQ(ierr);
    ierr = PetscOptionsRangeInt(std::get<2>(options),"Which device to use. Pass " PetscStringize(PETSC_DECIDE) " to have PETSc decide or (given they exist) [0-NUM_DEVICE) for a specific device","PetscDeviceCreate",id,&id,nullptr,PETSC_DECIDE,std::numeric_limits<int>::max());CHKERRQ(ierr);
    ierr = PetscOptionsBool(std::get<3>(options),"Display device information and assignments (forces eager initialization)",nullptr,view,&view,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }

  // post-process the options and lay the groundwork for initialization if needs be
  cerr = cupmGetDeviceCount(&ndev);
  if (PetscUnlikely(cerr == cupmErrorStubLibrary)) {
    if (PetscUnlikely((initTypeCUPM == PETSC_DEVICE_INIT_EAGER) || (view && flg))) {
      const auto name    = cupmGetErrorName(cerr);
      const auto desc    = cupmGetErrorString(cerr);
      const auto backend = cupmName();
      SETERRQ5(comm,PETSC_ERR_USER_INPUT,"Cannot eagerly initialize %s, as doing so results in %s error %d (%s) : %s",backend,backend,static_cast<PetscErrorCode>(cerr),name,desc);
    }
    cerr = cupmGetLastError(); // reset error
    initTypeCUPM = PETSC_DEVICE_INIT_NONE;
  } else {CHKERRCUPM(cerr);}

  if (initTypeCUPM == PETSC_DEVICE_INIT_NONE) id = PETSC_CUPM_DEVICE_NONE;
  else {
    ierr = PetscDeviceCheckDeviceCount_Internal(ndev);CHKERRQ(ierr);
    if (id == PETSC_DECIDE) {
      if (ndev) {
        PetscMPIInt rank;

        ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
        id   = rank % ndev;
      } else id = 0;
    }
    view = static_cast<decltype(view)>(view && flg);
    if (view) initTypeCUPM = PETSC_DEVICE_INIT_EAGER;
  }

  // id is PetscInt, _defaultDevice is int
  static_assert(std::is_same<PetscMPIInt,decltype(_defaultDevice)>::value,"");
  ierr = PetscMPIIntCast(id,&_defaultDevice);CHKERRQ(ierr);
  if (initTypeCUPM == PETSC_DEVICE_INIT_EAGER) {
    _devices[_defaultDevice] = CUPMDeviceInternal::makeDevice(_defaultDevice);
    ierr = _devices[_defaultDevice]->initialize();CHKERRQ(ierr);
    ierr = _devices[_defaultDevice]->configure();CHKERRQ(ierr);
    if (view) {
      PetscViewer vwr;

      ierr = PetscLogInitialize();CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&vwr);CHKERRQ(ierr);
      ierr = _devices[_defaultDevice]->view(vwr);CHKERRQ(ierr);
    }
  }

  // record the results of the initialization
  *defaultInitType = static_cast<PetscDeviceInitType>(initTypeCUPM);
  *defaultDeviceId = id;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::getDevice(PetscDevice device, PetscInt id) const noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(_defaultDevice == PETSC_CUPM_DEVICE_NONE)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Trying to retrieve a %s PetscDevice when it has been disabled",cupmName());
  if (id == PETSC_DECIDE) id = _defaultDevice;
  if (PetscUnlikelyDebug(static_cast<std::size_t>(id) >= _devices.size())) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only supports %zu number of devices but trying to get device with id %" PetscInt_FMT,_devices.size(),id);
  if (_devices[id]) {
    if (PetscUnlikelyDebug(id != _devices[id]->id())) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Entry %" PetscInt_FMT " contains device with mismatching id %" PetscInt_FMT,id,_devices[id]->id());
  } else _devices[id] = CUPMDeviceInternal::makeDevice(id);
  ierr = _devices[id]->initialize();CHKERRQ(ierr);
  device->deviceId           = _devices[id]->id(); // technically id = _devices[id]->_id here
  device->ops->createcontext = _create;
  device->ops->configure     = this->configureDevice;
  device->ops->view          = this->viewDevice;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::configureDevice(PetscDevice device) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = _devices[device->deviceId]->configure();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PetscErrorCode CUPMDevice<T>::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // now this __shouldn't__ reconfigure the device, but there is a petscinfo call to indicate
  // it is being reconfigured
  ierr = _devices[device->deviceId]->configure();CHKERRQ(ierr);
  ierr = _devices[device->deviceId]->view(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// explicitly instantiate the classes
#if PetscDefined(HAVE_CUDA)
template class CUPMDevice<CUPMDeviceType::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template class CUPMDevice<CUPMDeviceType::HIP>;
#endif

} // namespace Petsc
