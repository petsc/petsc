#include "sycldevice.hpp"
#include <limits>  // for std::numeric_limits
#include <csetjmp> // for MPI sycl device awareness
#include <csignal> // SIGSEGV
#include <vector>
#include <sycl/sycl.hpp>

namespace Petsc
{

namespace device
{

namespace sycl
{

// definition for static
std::array<Device::DeviceInternal *, PETSC_DEVICE_MAX_DEVICES> Device::devices_array_ = {};
Device::DeviceInternal                                       **Device::devices_       = &Device::devices_array_[1];
int                                                            Device::defaultDevice_ = PETSC_SYCL_DEVICE_NONE;
bool                                                           Device::initialized_   = false;

static std::jmp_buf MPISyclAwareJumpBuffer;
static bool         MPISyclAwareJumpBufferSet;

// internal "impls" class for SyclDevice. Each instance represents a single sycl device
class PETSC_NODISCARD Device::DeviceInternal {
  const int            id_; // -1 for the host device; 0 and up for gpu devices
  bool                 devInitialized_;
  const ::sycl::device syclDevice_;

public:
  // default constructor
  DeviceInternal(int id) noexcept : id_(id), devInitialized_(false), syclDevice_(chooseSYCLDevice_(id)) { }
  int  id() const { return id_; }
  bool initialized() const { return devInitialized_; }

  PetscErrorCode initialize() noexcept
  {
    PetscFunctionBegin;
    if (initialized()) PetscFunctionReturn(PETSC_SUCCESS);
    if (syclDevice_.is_gpu() && use_gpu_aware_mpi) {
      if (!isMPISyclAware_()) {
        PetscCall((*PetscErrorPrintf)("PETSc is configured with sycl support, but your MPI is not aware of sycl GPU devices. For better performance, please use a sycl GPU-aware MPI.\n"));
        PetscCall((*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n"));
        PETSCABORT(PETSC_COMM_SELF, PETSC_ERR_LIB);
      }
    }
    devInitialized_ = true;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode view(PetscViewer viewer) const noexcept
  {
    MPI_Comm    comm;
    PetscMPIInt rank;
    PetscBool   iascii;

    PetscFunctionBegin;
    PetscCheck(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d being viewed before it was initialized or configured", id());
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(viewer), PETSCVIEWERASCII, &iascii));
    PetscCall(PetscObjectGetComm(reinterpret_cast<PetscObject>(viewer), &comm));
    if (iascii) {
      PetscViewer sviewer;

      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
      PetscCall(PetscViewerASCIIPrintf(sviewer, "[%d] device: %s\n", rank, syclDevice_.get_info<::sycl::info::device::name>().c_str()));
      PetscCall(PetscViewerASCIIPushTab(sviewer));
      PetscCall(PetscViewerASCIIPrintf(sviewer, "-> Device vendor: %s\n", syclDevice_.get_info<::sycl::info::device::vendor>().c_str()));
      PetscCall(PetscViewerASCIIPopTab(sviewer));
      PetscCall(PetscViewerFlush(sviewer));
      PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode getattribute(PetscDeviceAttribute attr, void *value) const noexcept
  {
    PetscFunctionBegin;
    PetscCheck(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d not initialized", id());
    switch (attr) {
    case PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK:
      *static_cast<std::size_t *>(value) = syclDevice_.get_info<::sycl::info::device::local_mem_size>();
    case PETSC_DEVICE_ATTR_MAX:
      break;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  static ::sycl::device chooseSYCLDevice_(int id)
  {
    if (id == PETSC_SYCL_DEVICE_HOST) {
      return ::sycl::device(::sycl::host_selector());
    } else {
      return ::sycl::device::get_devices(::sycl::info::device_type::gpu)[id];
    }
  }

  // Is the underlying MPI aware of sycl (GPU) devices?
  bool isMPISyclAware_() noexcept
  {
    const int  bufSize           = 2;
    const int  hbuf[bufSize]     = {1, 0};
    int       *dbuf              = nullptr;
    bool       awareness         = false;
    const auto SyclSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
      if ((signal == SIGSEGV) && MPISyclAwareJumpBufferSet) std::longjmp(MPISyclAwareJumpBuffer, 1);
      return PetscSignalHandlerDefault(signal, ptr);
    };

    PetscFunctionBegin;
    auto Q = ::sycl::queue(syclDevice_);
    dbuf   = ::sycl::malloc_device<int>(bufSize, Q);
    Q.memcpy(dbuf, hbuf, sizeof(int) * bufSize).wait();
    PetscCallAbort(PETSC_COMM_SELF, PetscPushSignalHandler(SyclSignalHandler, nullptr));
    MPISyclAwareJumpBufferSet = true;
    if (setjmp(MPISyclAwareJumpBuffer)) {
      // if a segv was triggered in the MPI_Allreduce below, it is very likely due to MPI not being GPU-aware
      awareness = false;
      PetscStackPop;
    } else if (!MPI_Allreduce(dbuf, dbuf + 1, 1, MPI_INT, MPI_SUM, PETSC_COMM_SELF)) awareness = true;
    MPISyclAwareJumpBufferSet = false;
    PetscCallAbort(PETSC_COMM_SELF, PetscPopSignalHandler());
    ::sycl::free(dbuf, Q);
    PetscFunctionReturn(awareness);
  }
};

PetscErrorCode Device::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscBool *defaultView, PetscDeviceInitType *defaultInitType) noexcept
{
  auto     id       = *defaultDeviceId;
  auto     initType = *defaultInitType;
  auto     view = *defaultView, flg = PETSC_FALSE;
  PetscInt ngpus;

  PetscFunctionBegin;
  if (initialized_) PetscFunctionReturn(PETSC_SUCCESS);
  initialized_ = true;
  PetscCall(PetscRegisterFinalize(finalize_));
  PetscOptionsBegin(comm, nullptr, "PetscDevice sycl Options", "Sys");
  PetscCall(base_type::PetscOptionDeviceInitialize(PetscOptionsObject, &initType, nullptr));
  PetscCall(base_type::PetscOptionDeviceSelect(PetscOptionsObject, "Which sycl device to use? Pass -2 for host, PETSC_DECIDE (" PetscStringize(PETSC_DECIDE) ") to let PETSc decide, 0 and up for GPUs", "PetscDeviceCreate()", id, &id, nullptr, -2, std::numeric_limits<decltype(ngpus)>::max()));
  static_assert(PETSC_DECIDE == -1, "Expect PETSC_DECIDE to be -1");
  PetscCall(base_type::PetscOptionDeviceView(PetscOptionsObject, &view, &flg));
  PetscOptionsEnd();

  // post-process the options and lay the groundwork for initialization if needs be
  std::vector<::sycl::device> gpu_devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);
  ngpus                                   = static_cast<PetscInt>(gpu_devices.size());
  PetscCheck(ngpus || id < 0, comm, PETSC_ERR_USER_INPUT, "You specified a sycl gpu device with -device_select_sycl %d but there is no GPU", (int)id);
  PetscCheck(ngpus <= 0 || id < ngpus, comm, PETSC_ERR_USER_INPUT, "You specified a sycl gpu device with -device_select_sycl %d but there are only %d GPU", (int)id, (int)ngpus);

  if (initType == PETSC_DEVICE_INIT_NONE) id = PETSC_SYCL_DEVICE_NONE; /* user wants to disable all sycl devices */
  else {
    PetscCall(PetscDeviceCheckDeviceCount_Internal(ngpus));
    if (id == PETSC_DECIDE) { /* petsc will choose a GPU device if any, otherwise a CPU device */
      if (ngpus) {
        PetscMPIInt rank;
        PetscCallMPI(MPI_Comm_rank(comm, &rank));
        id = rank % ngpus;
      } else id = PETSC_SYCL_DEVICE_HOST;
    }
    if (view) initType = PETSC_DEVICE_INIT_EAGER;
  }

  if (id == -2) id = PETSC_SYCL_DEVICE_HOST; // user passed in '-device_select_sycl -2'. We transform it into canonical form

  defaultDevice_ = static_cast<decltype(defaultDevice_)>(id);
  PetscCheck(initType != PETSC_DEVICE_INIT_EAGER || id != PETSC_SYCL_DEVICE_NONE, comm, PETSC_ERR_USER_INPUT, "Cannot eagerly initialize sycl devices as you disabled them by -device_enable_sycl none");
  // record the results of the initialization
  *defaultDeviceId = id;
  *defaultView     = view;
  *defaultInitType = initType;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Device::finalize_() noexcept
{
  PetscFunctionBegin;
  if (!initialized_) PetscFunctionReturn(PETSC_SUCCESS);
  for (auto &&devPtr : devices_array_) delete devPtr;
  defaultDevice_ = PETSC_SYCL_DEVICE_NONE; // disabled by default
  initialized_   = false;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Device::init_device_id_(PetscInt *inid) const noexcept
{
  const auto id = *inid == PETSC_DECIDE ? defaultDevice_ : (int)*inid;

  PetscFunctionBegin;
  PetscCheck(defaultDevice_ != PETSC_SYCL_DEVICE_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Trying to retrieve a SYCL PetscDevice when it has been disabled");
  PetscCheck(!(id < PETSC_SYCL_DEVICE_HOST) && !(id - PETSC_SYCL_DEVICE_HOST >= PETSC_DEVICE_MAX_DEVICES), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Only supports %zu number of devices but trying to get device with id %d", devices_array_.size(), id);
  if (!devices_[id]) devices_[id] = new DeviceInternal(id);
  PetscCheck(id == devices_[id]->id(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Entry %d contains device with mismatching id %d", id, devices_[id]->id());
  PetscCall(devices_[id]->initialize());
  *inid = id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Device::view_device_(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[device->deviceId]->view(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Device::get_attribute_(PetscInt id, PetscDeviceAttribute attr, void *value) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[id]->getattribute(attr, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace sycl

} // namespace device

} // namespace Petsc
