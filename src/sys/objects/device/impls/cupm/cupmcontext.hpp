#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>
#include <petsc/private/logimpl.h>

#include <array>

namespace Petsc {

namespace Device {

namespace CUPM {

namespace Impl {

// Forward declare
template <DeviceType T>
class PETSC_VISIBILITY_INTERNAL DeviceContext;

template <DeviceType T>
class DeviceContext : Impl::BlasInterface<T> {
public:
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t, T);

private:
  // for tag-based dispatch of handle retrieval
  template <typename H, std::size_t>
  struct HandleTag {
    using type = H;
  };
  using stream_tag = HandleTag<cupmStream_t, 0>;
  using blas_tag   = HandleTag<cupmBlasHandle_t, 1>;
  using solver_tag = HandleTag<cupmSolverHandle_t, 2>;

public:
  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS {
    cupmStream_t stream;
    cupmEvent_t  event;
    cupmEvent_t  begin; // timer-only
    cupmEvent_t  end;   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse;
#endif
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;

    PETSC_NODISCARD auto get(stream_tag) const -> decltype(this->stream) {
      return this->stream;
    }
    PETSC_NODISCARD auto get(blas_tag) const -> decltype(this->blas) {
      return this->blas;
    }
    PETSC_NODISCARD auto get(solver_tag) const -> decltype(this->solver) {
      return this->solver;
    }
  };

private:
  static bool                                                     initialized_;
  static std::array<cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES>   blashandles_;
  static std::array<cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> solverhandles_;

  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceContext_IMPLS *impls_cast_(PetscDeviceContext ptr)) {
    return static_cast<PetscDeviceContext_IMPLS *>(ptr->data);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscLogEvent CUPMBLAS_HANDLE_CREATE()) {
    return T == DeviceType::CUDA ? CUBLAS_HANDLE_CREATE : HIPBLAS_HANDLE_CREATE;
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscLogEvent CUPMSOLVER_HANDLE_CREATE()) {
    return T == DeviceType::CUDA ? CUSOLVER_HANDLE_CREATE : HIPSOLVER_HANDLE_CREATE;
  }

  // this exists purely to satisfy the compiler so the tag-based dispatch works for the other
  // handles
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(stream_tag, PetscDeviceContext)) {
    return 0;
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_handle_(cupmBlasHandle_t &handle)) {
    PetscLogEvent event;

    PetscFunctionBegin;
    if (PetscLikely(handle)) PetscFunctionReturn(0);
    PetscCall(PetscLogPauseCurrentEvent_Internal(&event));
    PetscCall(PetscLogEventBegin(CUPMBLAS_HANDLE_CREATE(), 0, 0, 0, 0));
    for (auto i = 0; i < 3; ++i) {
      auto cberr = cupmBlasCreate(&handle);
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
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(blas_tag, PetscDeviceContext dctx)) {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = blashandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    PetscCall(create_handle_(handle));
    PetscCallCUPMBLAS(cupmBlasSetStream(handle, dci->stream));
    dci->blas = handle;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_handle_(cupmSolverHandle_t &handle)) {
    PetscLogEvent event;

    PetscFunctionBegin;
    PetscCall(PetscLogPauseCurrentEvent_Internal(&event));
    PetscCall(PetscLogEventBegin(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
    PetscCall(cupmBlasInterface_t::InitializeHandle(handle));
    PetscCall(PetscLogEventEnd(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
    PetscCall(PetscLogEventResume_Internal(event));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(solver_tag, PetscDeviceContext dctx)) {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = solverhandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    PetscCall(create_handle_(handle));
    PetscCall(cupmBlasInterface_t::SetHandleStream(handle, dci->stream));
    dci->solver = handle;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode finalize_()) {
    PetscFunctionBegin;
    for (auto &&handle : blashandles_) {
      if (handle) {
        PetscCallCUPMBLAS(cupmBlasDestroy(handle));
        handle = nullptr;
      }
    }
    for (auto &&handle : solverhandles_) {
      if (handle) {
        PetscCall(cupmBlasInterface_t::DestroyHandle(handle));
        handle = nullptr;
      }
    }
    initialized_ = false;
    PetscFunctionReturn(0);
  }

public:
  const struct _DeviceContextOps ops = {
    destroy, changeStreamType, setUp, query, waitForContext, synchronize, getHandle<blas_tag>, getHandle<solver_tag>, getHandle<stream_tag>, beginTimer, endTimer,
  };

  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setUp(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(PetscDeviceContext, PetscBool *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext));
  template <typename Handle_t>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getHandle(PetscDeviceContext, void *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *));

  // not a PetscDeviceContext method, this registers the class
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize());
};

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::initialize()) {
  PetscFunctionBegin;
  if (PetscUnlikely(!initialized_)) {
    initialized_ = true;
    PetscCall(PetscRegisterFinalize(finalize_));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::destroy(PetscDeviceContext dctx)) {
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) PetscCallCUPM(cupmStreamDestroy(dci->stream));
  if (dci->event) PetscCallCUPM(cupmEventDestroy(dci->event));
  if (dci->begin) PetscCallCUPM(cupmEventDestroy(dci->begin));
  if (dci->end) PetscCallCUPM(cupmEventDestroy(dci->end));
  PetscCall(PetscFree(dctx->data));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype)) {
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (auto &stream = dci->stream) {
    PetscCallCUPM(cupmStreamDestroy(stream));
    stream = nullptr;
  }
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::setUp(PetscDeviceContext dctx)) {
  const auto dci    = impls_cast_(dctx);
  auto      &stream = dci->stream;

  PetscFunctionBegin;
  if (stream) {
    PetscCallCUPM(cupmStreamDestroy(stream));
    stream = nullptr;
  }
  switch (const auto stype = dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    // don't create a stream for global blocking
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING: PetscCallCUPM(cupmStreamCreate(&stream)); break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING: PetscCallCUPM(cupmStreamCreateWithFlags(&stream, cupmStreamNonBlocking)); break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Invalid PetscStreamType %s", PetscStreamTypes[util::integral_value(stype)]); break;
  }
  if (!dci->event) PetscCallCUPM(cupmEventCreate(&dci->event));
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::query(PetscDeviceContext dctx, PetscBool *idle)) {
  cupmError_t cerr;

  PetscFunctionBegin;
  cerr = cupmStreamQuery(impls_cast_(dctx)->stream);
  if (cerr == cupmSuccess) *idle = PETSC_TRUE;
  else {
    // somethings gone wrong
    if (PetscUnlikely(cerr != cupmErrorNotReady)) PetscCallCUPM(cerr);
    *idle = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb)) {
  auto dcib = impls_cast_(dctxb);

  PetscFunctionBegin;
  PetscCallCUPM(cupmEventRecord(dcib->event, dcib->stream));
  PetscCallCUPM(cupmStreamWaitEvent(impls_cast_(dctxa)->stream, dcib->event, 0));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::synchronize(PetscDeviceContext dctx)) {
  auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  // in case anything was queued on the event
  PetscCallCUPM(cupmStreamWaitEvent(dci->stream, dci->event, 0));
  PetscCallCUPM(cupmStreamSynchronize(dci->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename handle_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getHandle(PetscDeviceContext dctx, void *handle)) {
  PetscFunctionBegin;
  PetscCall(initialize_handle_(handle_t{}, dctx));
  *static_cast<typename handle_t::type *>(handle) = impls_cast_(dctx)->get(handle_t{});
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::beginTimer(PetscDeviceContext dctx)) {
  auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  PetscCheck(!dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  if (!dci->begin) {
    PetscCallCUPM(cupmEventCreate(&dci->begin));
    PetscCallCUPM(cupmEventCreate(&dci->end));
  }
  PetscCallCUPM(cupmEventRecord(dci->begin, dci->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed)) {
  float gtime;
  auto  dci = impls_cast_(dctx);

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscCallCUPM(cupmEventRecord(dci->end, dci->stream));
  PetscCallCUPM(cupmEventSynchronize(dci->end));
  PetscCallCUPM(cupmEventElapsedTime(&gtime, dci->begin, dci->end));
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(0);
}

// initialize the static member variables
template <DeviceType T>
bool DeviceContext<T>::initialized_ = false;

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::blashandles_ = {};

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::solverhandles_ = {};

} // namespace Impl

// shorten this one up a bit (and instantiate the templates)
using CUPMContextCuda = Impl::DeviceContext<DeviceType::CUDA>;
using CUPMContextHip  = Impl::DeviceContext<DeviceType::HIP>;

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(IMPLS) Petsc::Device::CUPM::Impl::DeviceContext<Petsc::Device::CUPM::DeviceType::IMPLS>::PetscDeviceContext_IMPLS

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif // PETSCDEVICECONTEXTCUDA_HPP
