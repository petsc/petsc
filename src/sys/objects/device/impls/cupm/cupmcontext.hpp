#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmsolverinterface.hpp>
#include <petsc/private/logimpl.h>

#include <petsc/private/cpp/array.hpp>

#include "../segmentedmempool.hpp"
#include "cupmallocator.hpp"
#include "cupmstream.hpp"
#include "cupmevent.hpp"

#if defined(__cplusplus)

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

template <DeviceType T>
class DeviceContext : SolverInterface<T> {
public:
  PETSC_CUPMSOLVER_INHERIT_INTERFACE_TYPEDEFS_USING(T);

private:
  template <typename H, std::size_t>
  struct HandleTag {
    using type = H;
  };

  using stream_tag = HandleTag<cupmStream_t, 0>;
  using blas_tag   = HandleTag<cupmBlasHandle_t, 1>;
  using solver_tag = HandleTag<cupmSolverHandle_t, 2>;

  using stream_type = CUPMStream<T>;
  using event_type  = CUPMEvent<T>;

public:
  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS : memory::PoolAllocated<PetscDeviceContext_IMPLS> {
    stream_type stream{};
    cupmEvent_t event{};
    cupmEvent_t begin{}; // timer-only
    cupmEvent_t end{};   // timer-only
  #if PetscDefined(USE_DEBUG)
    PetscBool timerInUse{};
  #endif
    cupmBlasHandle_t   blas{};
    cupmSolverHandle_t solver{};

    constexpr PetscDeviceContext_IMPLS() noexcept = default;

    PETSC_NODISCARD cupmStream_t get(stream_tag) const noexcept { return this->stream.get_stream(); }

    PETSC_NODISCARD cupmBlasHandle_t get(blas_tag) const noexcept { return this->blas; }

    PETSC_NODISCARD cupmSolverHandle_t get(solver_tag) const noexcept { return this->solver; }
  };

private:
  static bool initialized_;

  static std::array<cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES>   blashandles_;
  static std::array<cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> solverhandles_;

  PETSC_NODISCARD static constexpr PetscDeviceContext_IMPLS *impls_cast_(PetscDeviceContext ptr) noexcept { return static_cast<PetscDeviceContext_IMPLS *>(ptr->data); }

  PETSC_NODISCARD static constexpr CUPMEvent<T> *event_cast_(PetscEvent event) noexcept { return static_cast<CUPMEvent<T> *>(event->data); }

  PETSC_NODISCARD static PetscLogEvent CUPMBLAS_HANDLE_CREATE() noexcept { return T == DeviceType::CUDA ? CUBLAS_HANDLE_CREATE : HIPBLAS_HANDLE_CREATE; }

  PETSC_NODISCARD static PetscLogEvent CUPMSOLVER_HANDLE_CREATE() noexcept { return T == DeviceType::CUDA ? CUSOLVER_HANDLE_CREATE : HIPSOLVER_HANDLE_CREATE; }

  // this exists purely to satisfy the compiler so the tag-based dispatch works for the other
  // handles
  static PetscErrorCode initialize_handle_(stream_tag, PetscDeviceContext) noexcept { return PETSC_SUCCESS; }

  static PetscErrorCode initialize_handle_(blas_tag, PetscDeviceContext dctx) noexcept
  {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = blashandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    if (!handle) {
      PetscLogEvent event;

      PetscCall(PetscLogPauseCurrentEvent_Internal(&event));
      PetscCall(PetscLogEventBegin(CUPMBLAS_HANDLE_CREATE(), 0, 0, 0, 0));
      for (auto i = 0; i < 3; ++i) {
        const auto cberr = cupmBlasCreate(handle.ptr_to());
        if (PetscLikely(cberr == CUPMBLAS_STATUS_SUCCESS)) break;
        if (PetscUnlikely(cberr != CUPMBLAS_STATUS_ALLOC_FAILED) && (cberr != CUPMBLAS_STATUS_NOT_INITIALIZED)) PetscCallCUPMBLAS(cberr);
        if (i != 2) {
          PetscCall(PetscSleep(3));
          continue;
        }
        PetscCheck(cberr == CUPMBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, "Unable to initialize %s", cupmBlasName());
      }
      PetscCall(PetscLogEventEnd(CUPMBLAS_HANDLE_CREATE(), 0, 0, 0, 0));
      PetscCall(PetscLogEventResume_Internal(event));
    }
    PetscCallCUPMBLAS(cupmBlasSetStream(handle, dci->stream.get_stream()));
    dci->blas = handle;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode initialize_handle_(solver_tag, PetscDeviceContext dctx) noexcept
  {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = solverhandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    if (!handle) {
      PetscLogEvent event;

      PetscCall(PetscLogPauseCurrentEvent_Internal(&event));
      PetscCall(PetscLogEventBegin(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
      for (auto i = 0; i < 3; ++i) {
        const auto cerr = cupmSolverCreate(&handle);
        if (PetscLikely(cerr == CUPMSOLVER_STATUS_SUCCESS)) break;
        if ((cerr != CUPMSOLVER_STATUS_NOT_INITIALIZED) && (cerr != CUPMSOLVER_STATUS_ALLOC_FAILED)) PetscCallCUPMSOLVER(cerr);
        if (i < 2) {
          PetscCall(PetscSleep(3));
          continue;
        }
        PetscCheck(cerr == CUPMSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, "Unable to initialize %s", cupmSolverName());
      }
      PetscCall(PetscLogEventEnd(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
      PetscCall(PetscLogEventResume_Internal(event));
    }
    PetscCallCUPMSOLVER(cupmSolverSetStream(handle, dci->stream.get_stream()));
    dci->solver = handle;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode check_current_device_(PetscDeviceContext dctxl, PetscDeviceContext dctxr) noexcept
  {
    const auto devidl = dctxl->device->deviceId, devidr = dctxr->device->deviceId;

    PetscFunctionBegin;
    PetscCheck(devidl == devidr, PETSC_COMM_SELF, PETSC_ERR_GPU, "Device contexts must be on the same device; dctx A (id %" PetscInt64_FMT " device id %" PetscInt_FMT ") dctx B (id %" PetscInt64_FMT " device id %" PetscInt_FMT ")",
               PetscObjectCast(dctxl)->id, devidl, PetscObjectCast(dctxr)->id, devidr);
    PetscCall(PetscDeviceCheckDeviceCount_Internal(devidl));
    PetscCall(PetscDeviceCheckDeviceCount_Internal(devidr));
    PetscCallCUPM(cupmSetDevice(static_cast<int>(devidl)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode check_current_device_(PetscDeviceContext dctx) noexcept { return check_current_device_(dctx, dctx); }

  static PetscErrorCode finalize_() noexcept
  {
    PetscFunctionBegin;
    for (auto &&handle : blashandles_) {
      if (handle) {
        PetscCallCUPMBLAS(cupmBlasDestroy(handle));
        handle = nullptr;
      }
    }

    for (auto &&handle : solverhandles_) {
      if (handle) {
        PetscCallCUPMSOLVER(cupmSolverDestroy(handle));
        handle = nullptr;
      }
    }
    initialized_ = false;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename Allocator, typename PoolType = ::Petsc::memory::SegmentedMemoryPool<typename Allocator::value_type, stream_type, Allocator, 256 * sizeof(PetscScalar)>>
  PETSC_NODISCARD static PoolType &default_pool_() noexcept
  {
    static PoolType pool;
    return pool;
  }

  static PetscErrorCode check_memtype_(PetscMemType mtype, const char mess[]) noexcept
  {
    PetscFunctionBegin;
    PetscCheck(PetscMemTypeHost(mtype) || (mtype == PETSC_MEMTYPE_DEVICE) || (mtype == PETSC_MEMTYPE_CUPM()), PETSC_COMM_SELF, PETSC_ERR_SUP, "%s device context can only handle %s (pinned) host or device memory", cupmName(), mess);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

public:
  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  static PetscErrorCode destroy(PetscDeviceContext) noexcept;
  static PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType) noexcept;
  static PetscErrorCode setUp(PetscDeviceContext) noexcept;
  static PetscErrorCode query(PetscDeviceContext, PetscBool *) noexcept;
  static PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext) noexcept;
  static PetscErrorCode synchronize(PetscDeviceContext) noexcept;
  template <typename Handle_t>
  static PetscErrorCode getHandle(PetscDeviceContext, void *) noexcept;
  static PetscErrorCode beginTimer(PetscDeviceContext) noexcept;
  static PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *) noexcept;
  static PetscErrorCode memAlloc(PetscDeviceContext, PetscBool, PetscMemType, std::size_t, std::size_t, void **) noexcept;
  static PetscErrorCode memFree(PetscDeviceContext, PetscMemType, void **) noexcept;
  static PetscErrorCode memCopy(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, std::size_t, PetscDeviceCopyMode) noexcept;
  static PetscErrorCode memSet(PetscDeviceContext, PetscMemType, void *, PetscInt, std::size_t) noexcept;
  static PetscErrorCode createEvent(PetscDeviceContext, PetscEvent) noexcept;
  static PetscErrorCode recordEvent(PetscDeviceContext, PetscEvent) noexcept;
  static PetscErrorCode waitForEvent(PetscDeviceContext, PetscEvent) noexcept;

  // not a PetscDeviceContext method, this registers the class
  static PetscErrorCode initialize(PetscDevice) noexcept;

  // clang-format off
  static constexpr _DeviceContextOps ops = {
    PetscDesignatedInitializer(destroy, destroy),
    PetscDesignatedInitializer(changestreamtype, changeStreamType),
    PetscDesignatedInitializer(setup, setUp),
    PetscDesignatedInitializer(query, query),
    PetscDesignatedInitializer(waitforcontext, waitForContext),
    PetscDesignatedInitializer(synchronize, synchronize),
    PetscDesignatedInitializer(getblashandle, getHandle<blas_tag>),
    PetscDesignatedInitializer(getsolverhandle, getHandle<solver_tag>),
    PetscDesignatedInitializer(getstreamhandle, getHandle<stream_tag>),
    PetscDesignatedInitializer(begintimer, beginTimer),
    PetscDesignatedInitializer(endtimer, endTimer),
    PetscDesignatedInitializer(memalloc, memAlloc),
    PetscDesignatedInitializer(memfree, memFree),
    PetscDesignatedInitializer(memcopy, memCopy),
    PetscDesignatedInitializer(memset, memSet),
    PetscDesignatedInitializer(createevent, createEvent),
    PetscDesignatedInitializer(recordevent, recordEvent),
    PetscDesignatedInitializer(waitforevent, waitForEvent)
  };
  // clang-format on
};

// not a PetscDeviceContext method, this initializes the CLASS
template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::initialize(PetscDevice device) noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!initialized_)) {
    uint64_t      threshold = UINT64_MAX;
    cupmMemPool_t mempool;

    initialized_ = true;
    PetscCallCUPM(cupmDeviceGetMemPool(&mempool, static_cast<int>(device->deviceId)));
    PetscCallCUPM(cupmMemPoolSetAttribute(mempool, cupmMemPoolAttrReleaseThreshold, &threshold));
    blashandles_.fill(nullptr);
    solverhandles_.fill(nullptr);
    PetscCall(PetscRegisterFinalize(finalize_));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::destroy(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (const auto dci = impls_cast_(dctx)) {
    PetscCall(dci->stream.destroy());
    if (dci->event) PetscCall(cupm_fast_event_pool<T>().deallocate(&dci->event));
    if (dci->begin) PetscCallCUPM(cupmEventDestroy(dci->begin));
    if (dci->end) PetscCallCUPM(cupmEventDestroy(dci->end));
    delete dci;
    dctx->data = nullptr;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype) noexcept
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(dci->stream.destroy());
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::setUp(PetscDeviceContext dctx) noexcept
{
  const auto dci   = impls_cast_(dctx);
  auto      &event = dci->event;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(dci->stream.change_type(dctx->streamType));
  if (!event) PetscCall(cupm_fast_event_pool<T>().allocate(&event));
  #if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::query(PetscDeviceContext dctx, PetscBool *idle) noexcept
{
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  switch (auto cerr = cupmStreamQuery(impls_cast_(dctx)->stream.get_stream())) {
  case cupmSuccess:
    *idle = PETSC_TRUE;
    break;
  case cupmErrorNotReady:
    *idle = PETSC_FALSE;
    // reset the error
    cerr = cupmGetLastError();
    static_cast<void>(cerr);
    break;
  default:
    PetscCallCUPM(cerr);
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) noexcept
{
  const auto dcib  = impls_cast_(dctxb);
  const auto event = dcib->event;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctxa, dctxb));
  PetscCallCUPM(cupmEventRecord(event, dcib->stream.get_stream()));
  PetscCallCUPM(cupmStreamWaitEvent(impls_cast_(dctxa)->stream.get_stream(), event, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::synchronize(PetscDeviceContext dctx) noexcept
{
  auto idle = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(query(dctx, &idle));
  if (!idle) PetscCallCUPM(cupmStreamSynchronize(impls_cast_(dctx)->stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
template <typename handle_t>
inline PetscErrorCode DeviceContext<T>::getHandle(PetscDeviceContext dctx, void *handle) noexcept
{
  PetscFunctionBegin;
  PetscCall(initialize_handle_(handle_t{}, dctx));
  *static_cast<typename handle_t::type *>(handle) = impls_cast_(dctx)->get(handle_t{});
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::beginTimer(PetscDeviceContext dctx) noexcept
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  #if PetscDefined(USE_DEBUG)
  PetscCheck(!dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
  #endif
  if (!dci->begin) {
    PetscAssert(!dci->end, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Don't have a 'begin' event, but somehow have an end event");
    PetscCallCUPM(cupmEventCreate(&dci->begin));
    PetscCallCUPM(cupmEventCreate(&dci->end));
  }
  PetscCallCUPM(cupmEventRecord(dci->begin, dci->stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed) noexcept
{
  float      gtime;
  const auto dci = impls_cast_(dctx);
  const auto end = dci->end;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  #if PetscDefined(USE_DEBUG)
  PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
  #endif
  PetscCallCUPM(cupmEventRecord(end, dci->stream.get_stream()));
  PetscCallCUPM(cupmEventSynchronize(end));
  PetscCallCUPM(cupmEventElapsedTime(&gtime, dci->begin, end));
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memAlloc(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, std::size_t n, std::size_t alignment, void **dest) noexcept
{
  const auto &stream = impls_cast_(dctx)->stream;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "allocating"));
  if (PetscMemTypeHost(mtype)) {
    PetscCall(default_pool_<HostAllocator<T>>().allocate(n, reinterpret_cast<char **>(dest), &stream, alignment));
  } else {
    PetscCall(default_pool_<DeviceAllocator<T>>().allocate(n, reinterpret_cast<char **>(dest), &stream, alignment));
  }
  if (clear) PetscCallCUPM(cupmMemsetAsync(*dest, 0, n, stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memFree(PetscDeviceContext dctx, PetscMemType mtype, void **ptr) noexcept
{
  const auto &stream = impls_cast_(dctx)->stream;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "freeing"));
  if (!*ptr) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscMemTypeHost(mtype)) {
    PetscCall(default_pool_<HostAllocator<T>>().deallocate(reinterpret_cast<char **>(ptr), &stream));
    // if ptr exists still exists the pool didn't own it
    if (*ptr) {
      auto registered = PETSC_FALSE, managed = PETSC_FALSE;

      PetscCall(PetscCUPMGetMemType(*ptr, nullptr, &registered, &managed));
      if (registered) {
        PetscCallCUPM(cupmFreeHost(*ptr));
      } else if (managed) {
        PetscCallCUPM(cupmFreeAsync(*ptr, stream.get_stream()));
      }
    }
  } else {
    PetscCall(default_pool_<DeviceAllocator<T>>().deallocate(reinterpret_cast<char **>(ptr), &stream));
    // if ptr still exists the pool didn't own it
    if (*ptr) PetscCallCUPM(cupmFreeAsync(*ptr, stream.get_stream()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memCopy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode) noexcept
{
  const auto stream = impls_cast_(dctx)->stream.get_stream();

  PetscFunctionBegin;
  // can't use PetscCUPMMemcpyAsync here since we don't know sizeof(*src)...
  if (mode == PETSC_DEVICE_COPY_HTOH) {
    const auto cerr = cupmStreamQuery(stream);

    // yes this is faster
    if (cerr == cupmSuccess) {
      PetscCall(PetscMemcpy(dest, src, n));
      PetscFunctionReturn(PETSC_SUCCESS);
    } else if (cerr == cupmErrorNotReady) {
      auto PETSC_UNUSED unused = cupmGetLastError();

      static_cast<void>(unused);
    } else {
      PetscCallCUPM(cerr);
    }
  }
  PetscCallCUPM(cupmMemcpyAsync(dest, src, n, PetscDeviceCopyModeToCUPMMemcpyKind(mode), stream));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memSet(PetscDeviceContext dctx, PetscMemType mtype, void *ptr, PetscInt v, std::size_t n) noexcept
{
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "zeroing"));
  PetscCallCUPM(cupmMemsetAsync(ptr, static_cast<int>(v), n, impls_cast_(dctx)->stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::createEvent(PetscDeviceContext, PetscEvent event) noexcept
{
  PetscFunctionBegin;
  PetscCallCXX(event->data = new event_type());
  event->destroy = [](PetscEvent event) {
    PetscFunctionBegin;
    delete event_cast_(event);
    event->data = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  };
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::recordEvent(PetscDeviceContext dctx, PetscEvent event) noexcept
{
  PetscFunctionBegin;
  PetscCall(impls_cast_(dctx)->stream.record_event(*event_cast_(event)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::waitForEvent(PetscDeviceContext dctx, PetscEvent event) noexcept
{
  PetscFunctionBegin;
  PetscCall(impls_cast_(dctx)->stream.wait_for_event(*event_cast_(event)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// initialize the static member variables
template <DeviceType T>
bool DeviceContext<T>::initialized_ = false;

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::blashandles_ = {};

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::solverhandles_ = {};

template <DeviceType T>
constexpr _DeviceContextOps DeviceContext<T>::ops;

} // namespace impl

// shorten this one up a bit (and instantiate the templates)
using CUPMContextCuda = impl::DeviceContext<DeviceType::CUDA>;
using CUPMContextHip  = impl::DeviceContext<DeviceType::HIP>;

  // shorthand for what is an EXTREMELY long name
  #define PetscDeviceContext_(IMPLS) ::Petsc::device::cupm::impl::DeviceContext<::Petsc::device::cupm::DeviceType::IMPLS>::PetscDeviceContext_IMPLS

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCDEVICECONTEXTCUDA_HPP
