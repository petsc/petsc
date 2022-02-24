#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#include <array>

namespace Petsc
{

namespace Device
{

namespace CUPM
{

namespace Impl
{

namespace detail
{

// for tag-based dispatch of handle retrieval
template <typename T> struct HandleTag { using type = T; };

} // namespace detail

// Forward declare
template <DeviceType T> class PETSC_VISIBILITY_INTERNAL DeviceContext;

template <DeviceType T>
class DeviceContext : Impl::BlasInterface<T>
{
public:
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,T);

private:
  template <typename H> using HandleTag = typename detail::HandleTag<H>;
  using stream_tag = HandleTag<cupmStream_t>;
  using blas_tag   = HandleTag<cupmBlasHandle_t>;
  using solver_tag = HandleTag<cupmSolverHandle_t>;

public:
  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS
  {
    cupmStream_t       stream;
    cupmEvent_t        event;
    cupmEvent_t        begin; // timer-only
    cupmEvent_t        end;   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool          timerInUse;
#endif
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;

    PETSC_NODISCARD auto get(stream_tag) const -> decltype(this->stream) { return this->stream; }
    PETSC_NODISCARD auto get(blas_tag)   const -> decltype(this->blas)   { return this->blas;   }
    PETSC_NODISCARD auto get(solver_tag) const -> decltype(this->solver) { return this->solver; }
  };

private:
  static bool initialized_;
  static std::array<cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   blashandles_;
  static std::array<cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> solverhandles_;

  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceContext_IMPLS* impls_cast_(PetscDeviceContext ptr))
  {
    return static_cast<PetscDeviceContext_IMPLS*>(ptr->data);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(cupmBlasHandle_t &handle))
  {
    PetscFunctionBegin;
    if (handle) PetscFunctionReturn(0);
    for (auto i = 0; i < 3; ++i) {
      auto cberr = cupmBlasCreate(&handle);
      if (PetscLikely(cberr == CUPMBLAS_STATUS_SUCCESS)) break;
      if (PetscUnlikely(cberr != CUPMBLAS_STATUS_ALLOC_FAILED) && (cberr != CUPMBLAS_STATUS_NOT_INITIALIZED)) CHKERRCUPMBLAS(cberr);
      if (i != 2) {
        CHKERRQ(PetscSleep(3));
        continue;
      }
      PetscCheck(cberr == CUPMBLAS_STATUS_SUCCESS,PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize %s",cupmBlasName());
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_handle_stream_(cupmBlasHandle_t &handle, cupmStream_t &stream))
  {
    cupmStream_t    cupmStream;

    PetscFunctionBegin;
    CHKERRCUPMBLAS(cupmBlasGetStream(handle,&cupmStream));
    if (cupmStream != stream) CHKERRCUPMBLAS(cupmBlasSetStream(handle,stream));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode finalize_())
  {
    PetscFunctionBegin;
    for (auto&& handle : blashandles_) {
      if (handle) {
        CHKERRCUPMBLAS(cupmBlasDestroy(handle));
        handle     = nullptr;
      }
    }
    for (auto&& handle : solverhandles_) {
      if (handle) {
        CHKERRQ(cupmBlasInterface_t::DestroyHandle(handle));
        handle    = nullptr;
      }
    }
    initialized_ = false;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_(PetscInt id, PetscDeviceContext_IMPLS *dci))
  {

    PetscFunctionBegin;
    CHKERRQ(PetscDeviceCheckDeviceCount_Internal(id));
    if (!initialized_) {
      initialized_ = true;
      CHKERRQ(PetscRegisterFinalize(finalize_));
    }
    // use the blashandle as a canary
    if (!blashandles_[id]) {
      CHKERRQ(initialize_handle_(blashandles_[id]));
      CHKERRQ(cupmBlasInterface_t::InitializeHandle(solverhandles_[id]));
    }
    CHKERRQ(set_handle_stream_(blashandles_[id],dci->stream));
    CHKERRQ(cupmBlasInterface_t::SetHandleStream(solverhandles_[id],dci->stream));
    dci->blas   = blashandles_[id];
    dci->solver = solverhandles_[id];
    PetscFunctionReturn(0);
  }

public:
  const struct _DeviceContextOps ops = {
    destroy,
    changeStreamType,
    setUp,
    query,
    waitForContext,
    synchronize,
    getHandle<blas_tag>,
    getHandle<solver_tag>,
    getHandle<stream_tag>,
    beginTimer,
    endTimer,
  };

  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode changeStreamType(PetscDeviceContext,PetscStreamType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setUp(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(PetscDeviceContext,PetscBool*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext));
  template <typename Handle_t>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getHandle(PetscDeviceContext,void*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext,PetscLogDouble*));
};

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::destroy(PetscDeviceContext dctx))
{
  auto           dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) CHKERRCUPM(cupmStreamDestroy(dci->stream));
  if (dci->event)  CHKERRCUPM(cupmEventDestroy(dci->event));
  if (dci->begin)  CHKERRCUPM(cupmEventDestroy(dci->begin));
  if (dci->end)    CHKERRCUPM(cupmEventDestroy(dci->end));
  CHKERRQ(PetscFree(dctx->data));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype))
{
  auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) {
    CHKERRCUPM(cupmStreamDestroy(dci->stream));
    dci->stream = nullptr;
  }
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::setUp(PetscDeviceContext dctx))
{
  auto           dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) {
    CHKERRCUPM(cupmStreamDestroy(dci->stream));
    dci->stream = nullptr;
  }
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    // don't create a stream for global blocking
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    CHKERRCUPM(cupmStreamCreate(&dci->stream));
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    CHKERRCUPM(cupmStreamCreateWithFlags(&dci->stream,cupmStreamNonBlocking));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %s",PetscStreamTypes[util::integral_value(dctx->streamType)]);
    break;
  }
  if (!dci->event) CHKERRCUPM(cupmEventCreate(&dci->event));
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  CHKERRQ(initialize_(dctx->device->deviceId,dci));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::query(PetscDeviceContext dctx, PetscBool *idle))
{
  cupmError_t cerr;

  PetscFunctionBegin;
  cerr = cupmStreamQuery(impls_cast_(dctx)->stream);
  if (cerr == cupmSuccess) *idle = PETSC_TRUE;
  else {
    // somethings gone wrong
    if (PetscUnlikely(cerr != cupmErrorNotReady)) CHKERRCUPM(cerr);
    *idle = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb))
{
  auto        dcib = impls_cast_(dctxb);

  PetscFunctionBegin;
  CHKERRCUPM(cupmEventRecord(dcib->event,dcib->stream));
  CHKERRCUPM(cupmStreamWaitEvent(impls_cast_(dctxa)->stream,dcib->event,0));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::synchronize(PetscDeviceContext dctx))
{
  auto        dci = impls_cast_(dctx);

  PetscFunctionBegin;
  // in case anything was queued on the event
  CHKERRCUPM(cupmStreamWaitEvent(dci->stream,dci->event,0));
  CHKERRCUPM(cupmStreamSynchronize(dci->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename handle_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getHandle(PetscDeviceContext dctx, void *handle))
{
  PetscFunctionBegin;
  *static_cast<typename handle_t::type*>(handle) = impls_cast_(dctx)->get(handle_t());
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::beginTimer(PetscDeviceContext dctx))
{
  auto        dci = impls_cast_(dctx);

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  PetscCheck(!dci->timerInUse,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  if (!dci->begin) {
    CHKERRCUPM(cupmEventCreate(&dci->begin));
    CHKERRCUPM(cupmEventCreate(&dci->end));
  }
  CHKERRCUPM(cupmEventRecord(dci->begin,dci->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed))
{
  float       gtime;
  auto        dci = impls_cast_(dctx);

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  PetscCheck(dci->timerInUse,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  CHKERRCUPM(cupmEventRecord(dci->end,dci->stream));
  CHKERRCUPM(cupmEventSynchronize(dci->end));
  CHKERRCUPM(cupmEventElapsedTime(&gtime,dci->begin,dci->end));
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(0);
}

// initialize the static member variables
template <DeviceType T> bool DeviceContext<T>::initialized_ = false;

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   DeviceContext<T>::blashandles_ = {};

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::solverhandles_ = {};

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
