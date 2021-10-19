#if !defined(PETSCDEVICECONTEXTCUPM_HPP)
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#if !defined(PETSC_HAVE_CXX_DIALECT_CXX11)
#error PetscDeviceContext backends for CUDA and HIP requires C++11
#endif

#include <array>

namespace Petsc
{

namespace detail
{

// for tag-based dispatch of handle retrieval
template <typename HT> struct HandleTag { };

} // namespace detail

// Forward declare
template <CUPMDeviceType T> class CUPMContext;

template <CUPMDeviceType T>
class CUPMContext : CUPMInterface<T>
{
  template <typename H>
  using HandleTag = typename detail::HandleTag<H>;

public:
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,T)

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

    PETSC_NODISCARD cupmBlasHandle_t   handle(HandleTag<cupmBlasHandle_t>)   { return blas;   }
    PETSC_NODISCARD cupmSolverHandle_t handle(HandleTag<cupmSolverHandle_t>) { return solver; }
  };

private:
  static bool _initialized;
  static std::array<cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   _blashandles;
  static std::array<cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> _solverhandles;

  PETSC_NODISCARD static constexpr PetscDeviceContext_IMPLS* __impls_cast(PetscDeviceContext ptr) noexcept
  {
    return static_cast<PetscDeviceContext_IMPLS*>(ptr->data);
  }

  PETSC_NODISCARD static PetscErrorCode __finalize() noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    for (auto&& handle : _blashandles) {
      if (handle) {ierr = cupmInterface_t::DestroyHandle(handle);CHKERRQ(ierr);}
    }
    for (auto&& handle : _solverhandles) {
      if (handle) {ierr = cupmInterface_t::DestroyHandle(handle);CHKERRQ(ierr);}
    }
    _initialized = false;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __initialize(PetscInt id, PetscDeviceContext_IMPLS *dci) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceCheckDeviceCount_Internal(id);CHKERRQ(ierr);
    if (!_initialized) {
      _initialized = true;
      ierr = PetscRegisterFinalize(__finalize);CHKERRQ(ierr);
    }
    // use the blashandle as a canary
    if (!_blashandles[id]) {
      ierr = cupmInterface_t::InitializeHandle(_blashandles[id]);CHKERRQ(ierr);
      ierr = cupmInterface_t::InitializeHandle(_solverhandles[id]);CHKERRQ(ierr);
    }
    ierr = cupmInterface_t::SetHandleStream(_blashandles[id],dci->stream);CHKERRQ(ierr);
    ierr = cupmInterface_t::SetHandleStream(_solverhandles[id],dci->stream);CHKERRQ(ierr);
    dci->blas   = _blashandles[id];
    dci->solver = _solverhandles[id];
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
    getHandle<cupmBlasHandle_t>,
    getHandle<cupmSolverHandle_t>,
    beginTimer,
    endTimer
  };

  // default constructor
  constexpr CUPMContext() noexcept = default;

  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  PETSC_NODISCARD static PetscErrorCode destroy(PetscDeviceContext) noexcept;
  PETSC_NODISCARD static PetscErrorCode changeStreamType(PetscDeviceContext,PetscStreamType) noexcept;
  PETSC_NODISCARD static PetscErrorCode setUp(PetscDeviceContext) noexcept;
  PETSC_NODISCARD static PetscErrorCode query(PetscDeviceContext,PetscBool*) noexcept;
  PETSC_NODISCARD static PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext) noexcept;
  PETSC_NODISCARD static PetscErrorCode synchronize(PetscDeviceContext) noexcept;
  template <typename Handle_t>
  PETSC_NODISCARD static PetscErrorCode getHandle(PetscDeviceContext,void*) noexcept;
  PETSC_NODISCARD static PetscErrorCode beginTimer(PetscDeviceContext) noexcept;
  PETSC_NODISCARD static PetscErrorCode endTimer(PetscDeviceContext,PetscLogDouble*) noexcept;
};

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::destroy(PetscDeviceContext dctx) noexcept
{
  cupmError_t    cerr;
  PetscErrorCode ierr;
  auto           dci = __impls_cast(dctx);

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  if (dci->event)  {
    cerr = cupmEventDestroy(dci->event);CHKERRCUPM(cerr);
    cerr = cupmEventDestroy(dci->begin);CHKERRCUPM(cerr);
    cerr = cupmEventDestroy(dci->end);CHKERRCUPM(cerr);
  }
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype) noexcept
{
  auto dci = __impls_cast(dctx);

  PetscFunctionBegin;
  if (dci->stream) {
    cupmError_t cerr;

    cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);
    dci->stream = nullptr;
  }
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::setUp(PetscDeviceContext dctx) noexcept
{
  PetscErrorCode ierr;
  cupmError_t    cerr;
  auto           dci = __impls_cast(dctx);

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    // don't create a stream for global blocking
    dci->stream = nullptr;
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    cerr = cupmStreamCreate(&dci->stream);CHKERRCUPM(cerr);
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    cerr = cupmStreamCreateWithFlags(&dci->stream,cupmStreamNonBlocking);CHKERRCUPM(cerr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %s",PetscStreamTypes[static_cast<int>(dctx->streamType)]);
    break;
  }
  if (!dci->event) {
    cerr = cupmEventCreate(&dci->event);CHKERRCUPM(cerr);
    cerr = cupmEventCreate(&dci->begin);CHKERRCUPM(cerr);
    cerr = cupmEventCreate(&dci->end);CHKERRCUPM(cerr);
  }
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  ierr = __initialize(dctx->device->deviceId,dci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::query(PetscDeviceContext dctx, PetscBool *idle) noexcept
{
  cupmError_t cerr;

  PetscFunctionBegin;
  cerr = cupmStreamQuery(__impls_cast(dctx)->stream);
  if (cerr == cupmSuccess) *idle = PETSC_TRUE;
  else {
    // somethings gone wrong
    if (PetscUnlikely(cerr != cupmErrorNotReady)) CHKERRCUPM(cerr);
    *idle = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) noexcept
{
  cupmError_t cerr;
  auto        dcia = __impls_cast(dctxa),dcib = __impls_cast(dctxb);

  PetscFunctionBegin;
  cerr = cupmEventRecord(dcib->event,dcib->stream);CHKERRCUPM(cerr);
  cerr = cupmStreamWaitEvent(dcia->stream,dcib->event,0);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::synchronize(PetscDeviceContext dctx) noexcept
{
  cupmError_t cerr;
  auto        dci = __impls_cast(dctx);

  PetscFunctionBegin;
  // in case anything was queued on the event
  cerr = cupmStreamWaitEvent(dci->stream,dci->event,0);CHKERRCUPM(cerr);
  cerr = cupmStreamSynchronize(dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <typename Handle_T>
inline PetscErrorCode CUPMContext<T>::getHandle(PetscDeviceContext dctx, void *handle) noexcept
{
  PetscFunctionBegin;
  *static_cast<Handle_T*>(handle) = __impls_cast(dctx)->handle(HandleTag<Handle_T>());
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::beginTimer(PetscDeviceContext dctx) noexcept
{
  auto        dci = __impls_cast(dctx);
  cupmError_t cerr;

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  if (PetscUnlikely(dci->timerInUse)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  cerr = cupmEventRecord(dci->begin,dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode CUPMContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed) noexcept
{
  cupmError_t cerr;
  float       gtime;
  auto        dci = __impls_cast(dctx);

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  if (PetscUnlikely(!dci->timerInUse)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  cerr = cupmEventRecord(dci->end,dci->stream);CHKERRCUPM(cerr);
  cerr = cupmEventSynchronize(dci->end);CHKERRCUPM(cerr);
  cerr = cupmEventElapsedTime(&gtime,dci->begin,dci->end);CHKERRCUPM(cerr);
  *elapsed = static_cast<PetscLogDouble>(gtime);
  PetscFunctionReturn(0);
}

// initialize the static member variables
template <CUPMDeviceType T> bool CUPMContext<T>::_initialized = false;

template <CUPMDeviceType T>
std::array<typename CUPMContext<T>::cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   CUPMContext<T>::_blashandles = {};

template <CUPMDeviceType T>
std::array<typename CUPMContext<T>::cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> CUPMContext<T>::_solverhandles = {};

// shorten this one up a bit (and instantiate the templates)
using CUPMContextCuda = CUPMContext<CUPMDeviceType::CUDA>;
using CUPMContextHip  = CUPMContext<CUPMDeviceType::HIP>;

} // namespace Petsc

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(IMPLS) Petsc::CUPMContext<Petsc::CUPMDeviceType::IMPLS>::PetscDeviceContext_IMPLS

// shorthand for casting dctx->data to the appropriate object to access the handles
#define PDC_IMPLS_STATIC_CAST(IMPLS,obj) static_cast<PetscDeviceContext_(IMPLS) *>((obj)->data)

#endif // PETSCDEVICECONTEXTCUDA_HPP
