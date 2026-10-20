#include <petsc/private/cpp/memory.hpp> // make_unique

#include "cupmdevice.hpp"

#include <algorithm>
#include <csetjmp> // for cuda mpi awareness
#include <csignal> // SIGSEGV
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace Petsc
{

namespace device
{

namespace cupm
{

#if PetscDefined(HAVE_HWLOC)
  #if PETSC_PKG_HWLOC_VERSION_GE(2, 0, 0)
struct HwlocBitmapCleaner {
  void operator()(hwloc_bitmap_t bitmap) const noexcept { hwloc_bitmap_free(bitmap); }
};
using HwlocBitmap = std::unique_ptr<hwloc_bitmap_s, HwlocBitmapCleaner>;

struct HwlocTopologyCleaner {
  void operator()(hwloc_topology_t topo) const noexcept
  {
    if (topo != nullptr) hwloc_topology_destroy(topo);
  }
};
using HwlocTopology = std::unique_ptr<hwloc_topology, HwlocTopologyCleaner>;
  #endif
#endif

// internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <DeviceType T>
class Device<T>::DeviceInternal {
  const int        id_;
  bool             devInitialized_ = false;
  cupmDeviceProp_t dprop_{}; // cudaDeviceProp appears to be an actual struct, i.e. you can't
                             // initialize it with nullptr or NULL (i've tried)

  static PetscErrorCode CUPMAwareMPI_(bool *) noexcept;

public:
  // default constructor
  explicit constexpr DeviceInternal(int dev) noexcept : id_(dev) { }

  // gather all relevant information for a particular device, a cupmDeviceProp_t is
  // usually sufficient here
  PetscErrorCode initialize() noexcept;
  PetscErrorCode configure() noexcept;
  PetscErrorCode view(PetscViewer) const noexcept;
  PetscErrorCode getattribute(PetscDeviceAttribute, void *) const noexcept;
  PetscErrorCode shutdown() noexcept;

  PETSC_NODISCARD auto id() const -> decltype(id_) { return id_; }
  PETSC_NODISCARD auto initialized() const -> decltype(devInitialized_) { return devInitialized_; }
  PETSC_NODISCARD auto prop() const -> const decltype(dprop_) & { return dprop_; }
};

// the goal here is simply to get the cupm backend to create its context, not to do any type of
// modification of it, or create objects (since these may be affected by subsequent
// configuration changes)
template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::initialize() noexcept
{
  PetscFunctionBegin;
  if (initialized()) PetscFunctionReturn(PETSC_SUCCESS);
  devInitialized_ = true;
  // need to do this BEFORE device has been set, although if the user
  // has already done this then we just ignore it
  if (cupmSetDeviceFlags(cupmDeviceMapHost) == cupmErrorSetOnActiveProcess) {
    // reset the error if it was cupmErrorSetOnActiveProcess
    const auto PETSC_UNUSED unused = cupmGetLastError();
  } else PetscCallCUPM(cupmGetLastError());
  // cuda 5.0+ will create a context when cupmSetDevice is called
  if (cupmSetDevice(id()) != cupmErrorDeviceAlreadyInUse) PetscCallCUPM(cupmGetLastError());
  // and in case it doesn't, explicitly call init here
  PetscCallCUPM(cupmInit(0));
#if PetscDefined(HAVE_NVML)
  // nvmlInit() deprecated in NVML 5.319
  PetscCallNVML(nvmlInit_v2());
#endif
  // where is this variable defined and when is it set? who knows! but it is defined and set
  // at this point. either way, each device must make this check since I guess MPI might not be
  // aware of all of them?
  if (use_gpu_aware_mpi) {
    bool aware;

    // Even the MPI implementation is configured with GPU-aware, it might still need extra settings to enable it.
    // So we do the check at runtime with a code that works only with GPU-aware MPI.
    PetscCall(CUPMAwareMPI_(&aware));
    if (PetscUnlikely(!aware)) {
      PetscCall((*PetscErrorPrintf)("PETSc is configured with GPU support, but your MPI is not GPU-aware. For better performance, please use a GPU-aware MPI.\n"));
      PetscCall((*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n"));
      PetscCall((*PetscErrorPrintf)("For Open MPI, you need to configure it with CUDA, ROCm or GPU-aware UCX (https://docs.open-mpi.org/en/main/tuning-apps/accelerators/index.html)\n"));
      PetscCall((*PetscErrorPrintf)("If you already configured it with GPU-aware UCX, you may need 'mpiexec -n <np> --mca pml ucx' or export 'OMPI_MCA_pml=\"ucx\"' to use it.\n"));
      PetscCall((*PetscErrorPrintf)("For MVAPICH2-GDR, you need to set MV2_USE_CUDA=1 (http://mvapich.cse.ohio-state.edu/userguide/gdr/)\n"));
      PetscCall((*PetscErrorPrintf)("For Cray-MPICH, export MPICH_GPU_SUPPORT_ENABLED=1 (see its 'man mpi'); for MPICH, export MPIR_CVAR_ENABLE_GPU=1\n"));
      PETSCABORT(PETSC_COMM_SELF, PETSC_ERR_LIB);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::configure() noexcept
{
  PetscFunctionBegin;
  PetscAssert(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d being configured before it was initialized", id());
  // why on EARTH nvidia insists on making otherwise informational states into
  // fully-fledged error codes is beyond me. Why couldn't a pointer to bool argument have
  // sufficed?!?!?!
  if (cupmSetDevice(id_) != cupmErrorDeviceAlreadyInUse) PetscCallCUPM(cupmGetLastError());
  // need to update the device properties
  PetscCallCUPM(cupmGetDeviceProperties(&dprop_, id_));
  PetscDeviceCUPMRuntimeArch = dprop_.major * 10 + dprop_.minor;
  PetscCall(PetscInfo(nullptr, "Configured device %d\n", id_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::view(PetscViewer viewer) const noexcept
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscAssert(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d being viewed before it was initialized or configured", id());
  // we don't print device-specific info in CI-mode
  if (PetscUnlikely(PetscCIEnabled)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(viewer), PETSCVIEWERASCII, &isascii));
  if (isascii) {
    MPI_Comm    comm;
    PetscMPIInt rank;
    PetscViewer sviewer;

    int clock, memclock;
    PetscCallCUPM(cupmDeviceGetAttribute(&clock, cupmDevAttrClockRate, id_));
    PetscCallCUPM(cupmDeviceGetAttribute(&memclock, cupmDevAttrMemoryClockRate, id_));

    PetscCall(PetscObjectGetComm(PetscObjectCast(viewer), &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "[%d] name: %s\n", rank, dprop_.name));
    PetscCall(PetscViewerASCIIPushTab(sviewer));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Compute capability: %d.%d\n", dprop_.major, dprop_.minor));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Multiprocessor Count: %d\n", dprop_.multiProcessorCount));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Maximum Grid Dimensions: %d x %d x %d\n", dprop_.maxGridSize[0], dprop_.maxGridSize[1], dprop_.maxGridSize[2]));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Maximum Block Dimensions: %d x %d x %d\n", dprop_.maxThreadsDim[0], dprop_.maxThreadsDim[1], dprop_.maxThreadsDim[2]));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Maximum Threads Per Block: %d\n", dprop_.maxThreadsPerBlock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Warp Size: %d\n", dprop_.warpSize));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Total Global Memory (bytes): %zu\n", dprop_.totalGlobalMem));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Total Constant Memory (bytes): %zu\n", dprop_.totalConstMem));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Shared Memory Per Block (bytes): %zu\n", dprop_.sharedMemPerBlock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Multiprocessor Clock Rate (kHz): %d\n", clock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Memory Clock Rate (kHz): %d\n", memclock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Memory Bus Width (bits): %d\n", dprop_.memoryBusWidth));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Peak Memory Bandwidth (GB/s): %f\n", 2.0 * memclock * (dprop_.memoryBusWidth / 8) / 1.0e6));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Can map host memory: %s\n", dprop_.canMapHostMemory ? "PETSC_TRUE" : "PETSC_FALSE"));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Can execute multiple kernels concurrently: %s\n", dprop_.concurrentKernels ? "PETSC_TRUE" : "PETSC_FALSE"));
    PetscCall(PetscViewerASCIIPopTab(sviewer));
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::getattribute(PetscDeviceAttribute attr, void *value) const noexcept
{
  PetscFunctionBegin;
  PetscAssert(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d was not initialized", id());
  switch (attr) {
  case PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK:
    *static_cast<std::size_t *>(value) = prop().sharedMemPerBlock;
  case PETSC_DEVICE_ATTR_MAX:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::shutdown() noexcept
{
  PetscFunctionBegin;
  if (!initialized()) PetscFunctionReturn(PETSC_SUCCESS);
#if PetscDefined(HAVE_NVML)
  PetscCallNVML(nvmlShutdown());
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static std::jmp_buf cupmMPIAwareJumpBuffer;
static bool         cupmMPIAwareJumpBufferSet;

// godspeed to anyone that attempts to call this function
void SilenceVariableIsNotNeededAndWillNotBeEmittedWarning_ThisFunctionShouldNeverBeCalled()
{
  PETSCABORT(MPI_COMM_NULL, (PetscErrorCode)INT_MAX);
  if (cupmMPIAwareJumpBufferSet) (void)cupmMPIAwareJumpBuffer;
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::CUPMAwareMPI_(bool *awareness) noexcept
{
  constexpr int hbuf[]            = {1, 0};
  int          *dbuf              = nullptr;
  const auto    cupmSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
    if ((signal == SIGSEGV) && cupmMPIAwareJumpBufferSet) std::longjmp(cupmMPIAwareJumpBuffer, 1);
    return PetscSignalHandlerDefault(signal, ptr);
  };

  PetscFunctionBegin;
  *awareness = false;
  PetscCallCUPM(cupmMalloc(reinterpret_cast<void **>(&dbuf), sizeof(hbuf)));
  PetscCallCUPM(cupmMemcpy(dbuf, hbuf, sizeof(hbuf), cupmMemcpyHostToDevice));
  PetscCallCUPM(cupmDeviceSynchronize());
  PetscCall(PetscPushSignalHandler(cupmSignalHandler, nullptr));
  cupmMPIAwareJumpBufferSet = true;
  if (!setjmp(cupmMPIAwareJumpBuffer) && !MPI_Allreduce(dbuf, dbuf + 1, 1, MPI_INT, MPI_SUM, PETSC_COMM_SELF)) *awareness = true;
  cupmMPIAwareJumpBufferSet = false;
  PetscCall(PetscPopSignalHandler());
  PetscCallCUPM(cupmFree(dbuf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::finalize_() noexcept
{
  PetscFunctionBegin;
  deferredError_ = cupmSuccess;
  if (defaultDeviceSet_) defaultDevice_ = PETSC_CUPM_DEVICE_NONE; // disabled by default
  defaultDeviceSet_ = PETSC_FALSE;
  if (PetscUnlikely(!initialized_)) PetscFunctionReturn(PETSC_SUCCESS);
  for (auto &&device : devices_) {
    if (device) PetscCall(device->shutdown());
    device.reset();
  }
  initialized_ = false;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PETSC_NODISCARD static PETSC_CONSTEXPR_14 const char *CUPM_VISIBLE_DEVICES() noexcept
{
  switch (T) {
  case DeviceType::CUDA:
    return "CUDA_VISIBLE_DEVICES";
  case DeviceType::HIP:
    return "ROCR_VISIBLE_DEVICES";
  }
  PetscUnreachable();
  return "PETSC_ERROR_PLIB";
}

template <DeviceType T>
PetscErrorCode Device<T>::select_device_petsc_decide_(MPI_Comm comm, PetscInt ndev, int *initId) noexcept
{
  PetscFunctionBegin;
  if (ndev) {
    // TORCHELASTIC_RUN_ID is used as a proxy to determine if the current process was launched with torchrun
    char *pytorch_exists = (char *)getenv("TORCHELASTIC_RUN_ID");
    char *pytorch_rank   = (char *)getenv("LOCAL_RANK");

    if (pytorch_exists && pytorch_rank) {
      char *endptr;

      *initId = (int)strtol(pytorch_rank, &endptr, 10);
      PetscCheck(*initId < ndev, PETSC_COMM_SELF, PETSC_ERR_LIB, "PyTorch environmental variable LOCAL_RANK %s > number devices %" PetscInt_FMT, pytorch_rank, ndev);
    } else {
      PetscMPIInt rank;

      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      *initId = rank % ndev;
    }
  } else *initId = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     The default device ID is
       MPI     -- rank % number_local_devices
       PyTorch -- getenv("LOCAL_RANK")
*/
#if PetscDefined(HAVE_HWLOC)
  #if PETSC_PKG_HWLOC_VERSION_GE(2, 0, 0)
template <DeviceType T>
PetscErrorCode Device<T>::get_device_placement_in_cpuset_(PetscInt dev_count, hwloc_cpuset_t superset_cpuset, hwloc_obj_t process_cpu_obj, hwloc_topology_t topology, PetscInt *relative_device_index) noexcept
{
  // hwloc_bitmap_weight() returns the number of non-zero entries in a cpuset.
  PetscInt cores_in_anc_obj = hwloc_bitmap_weight(superset_cpuset);
  PetscInt ctr              = 0;

  PetscFunctionBegin;
  // Enumerate cpuset in topological order
  for (auto this_cpu = hwloc_get_next_obj_inside_cpuset_by_type(topology, superset_cpuset, HWLOC_OBJ_PU, nullptr); this_cpu != nullptr; this_cpu = hwloc_get_next_obj_inside_cpuset_by_type(topology, superset_cpuset, HWLOC_OBJ_PU, this_cpu)) {
    // If the first CPU core in this thread's cpuset is found, set relative_device_index and return.
    if (this_cpu->logical_index == process_cpu_obj->logical_index) {
      *relative_device_index = dev_count * ctr / cores_in_anc_obj;
      break;
    }
    ctr++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::select_device_topology_aware_(PetscInt ndev, int *initId, PetscBool *success) noexcept
{
  PetscFunctionBegin;
  if (ndev == 0) *initId = PETSC_DECIDE;
  else if (ndev == 1) *initId = 0;
  else {
    PetscInt                 selected_device = -1;
    PetscInt                 max_depth       = -1;
    PetscInt                 max_count       = 1;
    PetscInt                 this_max_depth;
    PetscBool                same_devices;
    std::vector<hwloc_obj_t> hwloc_devs(ndev);
    std::vector<hwloc_obj_t> common_ancestors(ndev);
    std::vector<PetscInt>    device_depths(ndev);
    std::vector<std::string> device_addrs(ndev, std::string(32, '\0'));
    hwloc_obj_t              first_cpu = nullptr;
    hwloc_topology_t         raw_topo  = nullptr;
    HwlocBitmap              cpuset_mine(hwloc_bitmap_alloc());
    HwlocBitmap              sibling_cpuset(hwloc_bitmap_alloc());
    HwlocTopology            topology;

    /* Ensure *initId is set to a sensible fallback value if any hwloc
       calls fail. */
    *initId = PETSC_DECIDE;

    // Get PCI Bus addresses for each CUPM device
    for (PetscInt idev = 0; idev < ndev; idev++) {
      auto cerr = cupmDeviceGetPCIBusId(&device_addrs[idev][0], 32, idev);

      if (cerr != cupmSuccess) {
        /* Do not set deferredError_ here to allow fallback to select_device_petsc_decide_()
           which doesn't rely on any cupm calls. */
        auto PETSC_UNUSED ignored = cupmGetLastError();

        *success = PETSC_FALSE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }

    if (hwloc_topology_init(&raw_topo) == -1) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    topology.reset(raw_topo);

    /* Causes binding functions to call OS-specific system calls - required for determining
       current CPU binding */
    if (hwloc_topology_set_flags(topology.get(), HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM) == -1) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* By default IO devices (i.e. GPUs, storage, etc.) are filtered out. GPUs are considered
       important. This filter makes sure those are included in the detected topology. */
    if (hwloc_topology_set_io_types_filter(topology.get(), HWLOC_TYPE_FILTER_KEEP_IMPORTANT) == -1) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (hwloc_topology_load(topology.get()) == -1) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Get the current thread's CPU binding mask
    if (cpuset_mine.get() == nullptr) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (hwloc_get_cpubind(topology.get(), cpuset_mine.get(), HWLOC_CPUBIND_THREAD) == -1) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    // An unbound process reports every PU, which would send all ranks to the same GPU
    if (hwloc_bitmap_isincluded(hwloc_topology_get_topology_cpuset(topology.get()), cpuset_mine.get())) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* The cpuset returned from hwloc_get_cpubind() is in OS-order, which is not necessarily
       the same as the topological order. Create a PU object from the first CPU detected
       in this cpuset. A PU object is the lowest object in any hwloc topology. It is not
       allowed to have any child objects. */
    first_cpu = hwloc_get_pu_obj_by_os_index(topology.get(), hwloc_bitmap_first(cpuset_mine.get()));
    if (first_cpu == nullptr) {
      *success = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    for (PetscInt idev = 0; idev < ndev; idev++) {
      /* Use the PCI Bus address of each GPU to find it among the hwloc
         topology object */
      hwloc_devs[idev] = hwloc_get_pcidev_by_busidstring(topology.get(), device_addrs[idev].c_str());
      if (hwloc_devs[idev] == nullptr) {
        *success = PETSC_FALSE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      /* hwloc does not consider IO devices to have ancestor or child objects, therefore
         a call to hwloc_get_non_io_ancestor_obj() is required for each device to find the
         nearest non-IO device that has the same locality as the GPU. hwloc_get_common_ancestor_obj()
         object returns the lowest-level object ('Group', CPU socket, Machine, etc) that contains
         both the GPU and the first CPU core in the current cpuset. */
      common_ancestors[idev] = hwloc_get_common_ancestor_obj(topology.get(), first_cpu, hwloc_get_non_io_ancestor_obj(topology.get(), hwloc_devs[idev]));
      /* Every object at depth n+1 contains a subset of the PU and IO objects at depth n. Therefore
         the smallest object containing both device idev and CPU core will have the largest
         'depth' value. */
      if (common_ancestors[idev]->depth > max_depth) {
        max_depth       = common_ancestors[idev]->depth;
        max_count       = 1;
        selected_device = idev;
        // Prepare for the case where multiple devices are reported at the same depth level.
      } else if (common_ancestors[idev]->depth == max_depth) max_count++;
    }
    if (max_count == 1) *initId = selected_device;
    else {
      /* Fallback method to handle the case where multiple devices are reported at the same depth level.
         Evenly distribute processes among closest devices sequentially. */
      std::vector<PetscInt> devices_at_max_depth(max_count);
      PetscInt              ctr                 = 0;
      PetscInt              relative_device_idx = 0;
      auto                  global_cpuset       = hwloc_topology_get_allowed_cpuset(topology.get());

      for (PetscInt idev = 0; idev < ndev; idev++) {
        if (common_ancestors[idev]->depth == max_depth) {
          devices_at_max_depth[ctr] = idev;
          ctr++;
        }
      }
      // Construct a cpuset of CPUs with the same set of closest devices as this process.
      if (sibling_cpuset.get() == nullptr) {
        *success = PETSC_FALSE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      // Repeat the common ancestor depth calculation for every CPU core detected in the current cgroup
      for (auto this_cpu = hwloc_get_next_obj_inside_cpuset_by_type(topology.get(), global_cpuset, HWLOC_OBJ_PU, nullptr); this_cpu != nullptr; this_cpu = hwloc_get_next_obj_inside_cpuset_by_type(topology.get(), global_cpuset, HWLOC_OBJ_PU, this_cpu)) {
        same_devices = PETSC_TRUE;
        for (PetscInt jdev = 0; jdev < ndev; jdev++) device_depths[jdev] = hwloc_get_common_ancestor_obj(topology.get(), this_cpu, hwloc_get_non_io_ancestor_obj(topology.get(), hwloc_devs[jdev]))->depth;
        this_max_depth = *std::max_element(device_depths.begin(), device_depths.end());
        // Identify a CPU as a sibling when it has the same view of device depths that the current CPU has.
        for (PetscInt jdev = 0; jdev < ndev && (same_devices == PETSC_TRUE); jdev++) same_devices = ((device_depths[jdev] == this_max_depth) == (common_ancestors[jdev]->depth == max_depth));
        if (same_devices == PETSC_TRUE && hwloc_bitmap_set(sibling_cpuset.get(), this_cpu->os_index) == -1) {
          *success = PETSC_FALSE;
          PetscFunctionReturn(PETSC_SUCCESS);
        }
      }
      PetscCall(get_device_placement_in_cpuset_(max_count, sibling_cpuset.get(), first_cpu, topology.get(), &relative_device_idx));
      *initId = devices_at_max_depth[relative_device_idx];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #endif // PETSC_PKG_HWLOC_VERSION_GE(2, 0, 0)
#endif   // PETSC_HAVE_HWLOC

template <DeviceType T>
PetscErrorCode Device<T>::select_device_(MPI_Comm comm, int *dev_id) noexcept
{
  int         ndev = 0;
  cupmError_t cerr = cupmSuccess;

  PetscFunctionBegin;
  if ((cerr = cupmGetDeviceCount(&ndev))) {
    auto PETSC_UNUSED ignored = cupmGetLastError();
    deferredError_            = cerr;
    *dev_id                   = PETSC_CUPM_DEVICE_NONE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscDeviceCheckDeviceCount_Internal(ndev));
#if PetscDefined(HAVE_HWLOC)
  #if PETSC_PKG_HWLOC_VERSION_GE(2, 0, 0)
  if (*dev_id == PETSC_DEVICE_TOPOLOGY_AWARE) {
    PetscBool hwloc_success = PETSC_TRUE;

    PetscCall(select_device_topology_aware_(ndev, dev_id, &hwloc_success));
    /* If select_device_topology_aware_() fails, it will set *dev_id to
       PETSC_DECIDE in order to fall through to the default algorithm */
    if (hwloc_success == PETSC_FALSE) PetscCall(PetscInfo(nullptr, "Topology aware GPU device allocation failed. Falling back to default algorithm\n"));
  }
  #endif
#endif
  if (*dev_id == PETSC_DECIDE) PetscCall(select_device_petsc_decide_(comm, ndev, dev_id));
  PetscCall(PetscInfo(nullptr, "GPU device id selected: %" PetscInt32_FMT "\n", *dev_id));
  static_assert(std::is_same<PetscMPIInt, decltype(defaultDevice_)>::value, "");
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscBool *defaultView, PetscDeviceInitType *defaultInitType) noexcept
{
  auto initId   = std::make_pair(*defaultDeviceId, PETSC_FALSE);
  auto initView = std::make_pair(*defaultView, PETSC_FALSE);
  auto initType = std::make_pair(*defaultInitType, PETSC_FALSE);
  int  dev_id;

  PetscFunctionBegin;
  if (initialized_) PetscFunctionReturn(PETSC_SUCCESS);
  // Somehow back here after something has already gone wrong, bail immediately
  if (deferredError_ != cupmSuccess) PetscFunctionReturn(PETSC_SUCCESS);
  static_assert(std::is_same<PetscMPIInt, decltype(dev_id)>::value, "");
  PetscCall(PetscRegisterFinalize(finalize_));
  PetscCall(base_type::PetscOptionDeviceAll(comm, initType, initId, initView));

  PetscCall(PetscMPIIntCast(initId.first, &dev_id));
  if (initType.first == PETSC_DEVICE_INIT_NONE) {
    // initType overrides initView
    dev_id         = PETSC_CUPM_DEVICE_NONE;
    initView.first = PETSC_FALSE;
  } else {
    PetscCall(select_device_(comm, &dev_id));
    if (initView.first) initType.first = PETSC_DEVICE_INIT_EAGER;
  }
  initId.first = dev_id;

  // Something went wrong in device selection
  if (deferredError_ != cupmSuccess) {
    PetscCheck((initType.first != PETSC_DEVICE_INIT_EAGER) && !initView.first, comm, PETSC_ERR_USER_INPUT, "Cannot eagerly initialize %s, as doing so results in %s error %d (%s) : %s", cupmName(), cupmName(), static_cast<PetscErrorCode>(deferredError_), cupmGetErrorName(deferredError_), cupmGetErrorString(deferredError_));
    initType.first = PETSC_DEVICE_INIT_NONE;
    initId.first   = PETSC_CUPM_DEVICE_NONE;
    initView.first = PETSC_FALSE;
  }

  // record the results of the initialization
  defaultDevice_    = dev_id;
  defaultDeviceSet_ = PETSC_TRUE;
  *defaultDeviceId  = initId.first;
  *defaultView      = initView.first;
  *defaultInitType  = initType.first;
  deviceComm_       = comm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::init_device_id_(PetscInt *inid) const noexcept
{
  auto id = *inid == PETSC_DECIDE ? defaultDevice_ : (int)*inid;

  PetscFunctionBegin;
  /* In this callpath, a negative inid may be passed from PetscDeviceCreate().
     If this happens, -1 (PETSC_DECIDE) is intercepted above, but any other
     negative value will be passed to select_device_(). Positive values
     (i.e. the user has selected a specific device) always override
     negative values. */

  PetscCheck(deferredError_ == cupmSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "Cannot lazily initialize PetscDevice: %s error %d (%s) : %s", cupmName(), static_cast<PetscErrorCode>(deferredError_), cupmGetErrorName(deferredError_), cupmGetErrorString(deferredError_));
  PetscCheck(defaultDevice_ != PETSC_CUPM_DEVICE_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Trying to retrieve a %s PetscDevice when it has been disabled", cupmName());

  if (id < 0) {
    PetscCall(select_device_(deviceComm_, &id));
    PetscCheck(deferredError_ == cupmSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "Cannot lazily initialize PetscDevice: %s error %d (%s) : %s", cupmName(), static_cast<PetscErrorCode>(deferredError_), cupmGetErrorName(deferredError_), cupmGetErrorString(deferredError_));
    PetscCheck(id >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported %s device selection strategy %" PetscInt_FMT "; pass " PetscStringize(PETSC_DECIDE) " to have PETSc decide", cupmName(), *inid);
  }
  PetscAssert(static_cast<decltype(devices_.size())>(id) < devices_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Only supports %zu number of devices but trying to get device with id %d", devices_.size(), id);
  if (!devices_[id]) devices_[id] = util::make_unique<DeviceInternal>(id);
  PetscAssert(id == devices_[id]->id(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Entry %d contains device with mismatching id %d", id, devices_[id]->id());
  PetscCall(devices_[id]->initialize());
  *inid        = id;
  initialized_ = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::configure_device_(PetscDevice device) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[device->deviceId]->configure());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::view_device_(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscFunctionBegin;
  // now this __shouldn't__ reconfigure the device, but there is a petscinfo call to indicate
  // it is being reconfigured
  PetscCall(devices_[device->deviceId]->configure());
  PetscCall(devices_[device->deviceId]->view(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::get_attribute_(PetscInt id, PetscDeviceAttribute attr, void *value) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[id]->getattribute(attr, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// explicitly instantiate the classes
#if PetscDefined(HAVE_CUDA)
template class Device<DeviceType::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template class Device<DeviceType::HIP>;
#endif

} // namespace cupm

} // namespace device

} // namespace Petsc
