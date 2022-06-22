#include "../../interface/sycldevice.hpp"
#include <csetjmp> // for MPI sycl device awareness
#include <csignal> // SIGSEGV
#include <vector>
#include <CL/sycl.hpp>

#if PetscDefined(USE_LOG)
  PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#else
  #define PetscLogInitialize() 0
#endif

namespace Petsc
{

namespace Device
{

namespace SYCL
{

// definition for static
std::array<Device::DeviceInternal*,PETSC_DEVICE_MAX_DEVICES> Device::devices_array_ = {};
Device::DeviceInternal**                                     Device::devices_       = &Device::devices_array_[1];
int                                                          Device::defaultDevice_ = PETSC_SYCL_DEVICE_NONE;
bool                                                         Device::initialized_   = false;

static std::jmp_buf MPISyclAwareJumpBuffer;
static bool         MPISyclAwareJumpBufferSet;

// internal "impls" class for SyclDevice. Each instance represents a single sycl device
class PETSC_NODISCARD Device::DeviceInternal
{
  const int            id_; // -1 for the host device; 0 and up for gpu devices
  bool                 devInitialized_;
  const sycl::device   syclDevice_;

public:
  // default constructor
  DeviceInternal(int id) noexcept : id_(id),devInitialized_(false),syclDevice_(chooseSYCLDevice_(id)){}
  int  id() const {return id_;}
  bool initialized() const {return devInitialized_;}

  PETSC_NODISCARD PetscErrorCode initialize() noexcept
  {
    PetscFunctionBegin;
    if (devInitialized_) PetscFunctionReturn(0);
    if (syclDevice_.is_gpu() && use_gpu_aware_mpi) {
      if (!isMPISyclAware_()) {
        (*PetscErrorPrintf)("PETSc is configured with sycl support, but your MPI is not aware of sycl GPU devices. For better performance, please use a sycl GPU-aware MPI.\n");
        (*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n");
        PETSCABORT(PETSC_COMM_SELF,PETSC_ERR_LIB);
      }
    }
    devInitialized_ = true;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode view(PetscViewer viewer) const noexcept
  {
    MPI_Comm       comm;
    PetscMPIInt    rank;
    PetscBool      iascii;

    PetscFunctionBegin;
    PetscCheck(devInitialized_,PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being viewed before it was initialized or configured",id_);
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(viewer),PETSCVIEWERASCII,&iascii));
    PetscCall(PetscObjectGetComm(reinterpret_cast<PetscObject>(viewer),&comm));
    if (iascii) {
      PetscViewer sviewer;

      PetscCallMPI(MPI_Comm_rank(comm,&rank));
      PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      PetscCall(PetscViewerASCIIPrintf(sviewer,"[%d] device: %s\n",rank,syclDevice_.get_info<sycl::info::device::name>().c_str()));
      PetscCall(PetscViewerASCIIPushTab(sviewer));
      PetscCall(PetscViewerASCIIPrintf(sviewer,"-> Device vendor: %s\n",syclDevice_.get_info<sycl::info::device::vendor>().c_str()));
      PetscCall(PetscViewerASCIIPopTab(sviewer));
      PetscCall(PetscViewerFlush(sviewer));
      PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      PetscCall(PetscViewerFlush(viewer));
    }
    PetscFunctionReturn(0);
  }

private:
  static sycl::device chooseSYCLDevice_(int id)
  {
    if (id == PETSC_SYCL_DEVICE_HOST) {
      return sycl::device(sycl::host_selector());
    } else {
      return sycl::device::get_devices(sycl::info::device_type::gpu)[id];
    }
  }

  // Is the underlying MPI aware of sycl (GPU) devices?
  bool isMPISyclAware_() noexcept
  {
    const int      bufSize = 2;
    const int      hbuf[bufSize] = {1,0};
    int            *dbuf = nullptr;
    bool           awareness = false;
    const auto     SyclSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
      if ((signal == SIGSEGV) && MPISyclAwareJumpBufferSet) std::longjmp(MPISyclAwareJumpBuffer,1);
      return PetscSignalHandlerDefault(signal,ptr);
    };

    PetscFunctionBegin;
    auto Q = sycl::queue(syclDevice_);
    dbuf   = sycl::malloc_device<int>(bufSize,Q);
    Q.memcpy(dbuf,hbuf,sizeof(int)*bufSize).wait();
    PetscCallAbort(PETSC_COMM_SELF,PetscPushSignalHandler(SyclSignalHandler,nullptr));
    MPISyclAwareJumpBufferSet = true;
    if (setjmp(MPISyclAwareJumpBuffer)) {
      // if a segv was triggered in the MPI_Allreduce below, it is very likely due to MPI not being GPU-aware
      awareness = false;
      PetscStackPop;
    } else if (!MPI_Allreduce(dbuf,dbuf+1,1,MPI_INT,MPI_SUM,PETSC_COMM_SELF)) awareness = true;
    MPISyclAwareJumpBufferSet = false;
    PetscCallAbort(PETSC_COMM_SELF,PetscPopSignalHandler());
    sycl::free(dbuf,Q);
    PetscFunctionReturn(awareness);
  }
};

PetscErrorCode Device::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscDeviceInitType *defaultInitType) noexcept
{
  PetscInt       initType = *defaultInitType,id = *defaultDeviceId;
  PetscBool      view = PETSC_FALSE,flg;
  PetscInt       ngpus;

  PetscFunctionBegin;
  if (initialized_) PetscFunctionReturn(0);
  initialized_ = true;
  PetscCall(PetscRegisterFinalize(finalize_));

  PetscOptionsBegin(comm,nullptr,"PetscDevice SYCL Options","Sys");
  PetscCall(PetscOptionsEList("-device_enable_sycl","How (or whether) to initialize a device","SyclDevice::initialize()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[initType],&initType,nullptr));
  PetscCall(PetscOptionsRangeInt("-device_select_sycl","Which sycl device to use? Pass -2 for host, PETSC_DECIDE (-1) to let PETSc decide, 0 and up for GPUs","PetscDeviceCreate",id,&id,nullptr,-2,std::numeric_limits<decltype(ngpus)>::max()));
  PetscCall(PetscOptionsBool("-device_view_sycl","Display device information and assignments (forces eager initialization)",nullptr,view,&view,&flg));
  PetscOptionsEnd();

  // post-process the options and lay the groundwork for initialization if needs be
  std::vector<sycl::device> gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  ngpus = static_cast<PetscInt>(gpu_devices.size());
  PetscCheck(ngpus || id < 0,comm,PETSC_ERR_USER_INPUT,"You specified a sycl gpu device with -device_select_sycl %d but there is no GPU", (int)id);
  PetscCheck(ngpus <= 0 || id < ngpus,comm,PETSC_ERR_USER_INPUT,"You specified a sycl gpu device with -device_select_sycl %d but there are only %d GPU", (int)id, (int)ngpus);

  if (initType == PETSC_DEVICE_INIT_NONE) id = PETSC_SYCL_DEVICE_NONE; /* user wants to disable all sycl devices */
  else {
    PetscCall(PetscDeviceCheckDeviceCount_Internal(ngpus));
    if (id == PETSC_DECIDE) { /* petsc will choose a GPU device if any, otherwise a CPU device */
      if (ngpus) {
        PetscMPIInt rank;
        PetscCallMPI(MPI_Comm_rank(comm,&rank));
        id   = rank % ngpus;
      } else id = PETSC_SYCL_DEVICE_HOST;
    }
    view = static_cast<decltype(view)>(view && flg);
    if (view) initType = PETSC_DEVICE_INIT_EAGER;
  }

  if (id == -2) id = PETSC_SYCL_DEVICE_HOST; // user passed in '-device_select_sycl -2'. We transform it into canonical form

  defaultDevice_ = static_cast<decltype(defaultDevice_)>(id);
  PetscCheck(initType != PETSC_DEVICE_INIT_EAGER || id != PETSC_SYCL_DEVICE_NONE,comm,PETSC_ERR_USER_INPUT,"Cannot eagerly initialize sycl devices as you disabled them by -device_enable_sycl none");

  if (initType == PETSC_DEVICE_INIT_EAGER) {
    devices_[defaultDevice_] = new DeviceInternal(defaultDevice_);
    PetscCall(devices_[defaultDevice_]->initialize());
    if (view) {
      PetscViewer viewer;
      PetscCall(PetscLogInitialize());
      PetscCall(PetscViewerASCIIGetStdout(comm,&viewer));
      PetscCall(devices_[defaultDevice_]->view(viewer));
    }
  }

  // record the results of the initialization
  *defaultInitType = static_cast<PetscDeviceInitType>(initType);
  *defaultDeviceId = id;
  PetscFunctionReturn(0);
}

PetscErrorCode Device::finalize_() noexcept
{
  PetscFunctionBegin;
  if (!initialized_) PetscFunctionReturn(0);
  for (auto&& devPtr : devices_array_) delete devPtr;
  defaultDevice_ = PETSC_SYCL_DEVICE_NONE;  // disabled by default
  initialized_   = false;
  PetscFunctionReturn(0);
}

PetscErrorCode Device::getDevice(PetscDevice device, PetscInt id) const noexcept
{
  PetscFunctionBegin;
  PetscCheck(defaultDevice_ != PETSC_SYCL_DEVICE_NONE,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Trying to retrieve a SYCL PetscDevice when it has been disabled");
  if (id == PETSC_DECIDE) id = defaultDevice_;
  PetscCheck(!(id < PETSC_SYCL_DEVICE_HOST) && !(id-PETSC_SYCL_DEVICE_HOST >= PETSC_DEVICE_MAX_DEVICES),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only supports %zu number of devices but trying to get device with id %" PetscInt_FMT,devices_array_.size(),id);
  if (devices_[id]) {
    PetscCheck(id == devices_[id]->id(),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Entry %" PetscInt_FMT " contains device with mismatching id %" PetscInt_FMT,id,devices_[id]->id());
  } else devices_[id] = new DeviceInternal(id);
  PetscCall(devices_[id]->initialize());
  device->deviceId           = devices_[id]->id(); // technically id = devices_[id]->id_ here
  device->ops->createcontext = create_;
  device->ops->configure     = this->configureDevice;
  device->ops->view          = this->viewDevice;
  PetscFunctionReturn(0);
}

PetscErrorCode Device::configureDevice(PetscDevice device) noexcept
{
  PetscFunctionBegin;
  // Nothing for now
  PetscFunctionReturn(0);
}

PetscErrorCode Device::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[device->deviceId]->view(viewer));
  PetscFunctionReturn(0);
}

} // namespace SYCL

} // namespace Device

} // namespace Petsc
