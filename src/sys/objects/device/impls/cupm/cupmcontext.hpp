#if !defined(PETSCDEVICECONTEXTCUPM_HPP)
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include <petsc/private/cupminterface.hpp>

#if !defined(PETSC_HAVE_CXX_DIALECT_CXX11)
#error PetscDeviceContext backends for CUDA and HIP requires C++11
#endif

namespace Petsc {

// Forward declare
template <CUPMDeviceKind T> class CUPMContext;

template <CUPMDeviceKind T>
class CUPMContext : CUPMInterface<T>
{
public:
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,T);

  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS
  {
    cupmStream_t       stream;
    cupmEvent_t        event;
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;
  };

private:
  static cupmBlasHandle_t   _blashandle;
  static cupmSolverHandle_t _solverhandle;

  PETSC_NODISCARD static PetscErrorCode __finalizeBLASHandle() noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = cupmInterface_t::DestroyHandle(_blashandle);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __finalizeSOLVERHandle() noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = cupmInterface_t::DestroyHandle(_solverhandle);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __setupHandles(PetscDeviceContext_IMPLS *dci) noexcept
  {
    PetscErrorCode  ierr;

    PetscFunctionBegin;
    if (!_blashandle) {
      ierr = cupmInterface_t::InitializeHandle(_blashandle);CHKERRQ(ierr);
      ierr = PetscRegisterFinalize(__finalizeBLASHandle);CHKERRQ(ierr);
    }
    if (!_solverhandle) {
      ierr = cupmInterface_t::InitializeHandle(_solverhandle);CHKERRQ(ierr);
      ierr = PetscRegisterFinalize(__finalizeSOLVERHandle);CHKERRQ(ierr);
    }
    ierr = cupmInterface_t::SetHandleStream(_blashandle,dci->stream);CHKERRQ(ierr);
    ierr = cupmInterface_t::SetHandleStream(_solverhandle,dci->stream);CHKERRQ(ierr);
    dci->blas   = _blashandle;
    dci->solver = _solverhandle;
    PetscFunctionReturn(0);
  }

public:
  const struct _DeviceContextOps ops {destroy,changeStreamType,setUp,query,waitForContext,synchronize};

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
};

#define IMPLS_RCAST_(obj_) static_cast<PetscDeviceContext_IMPLS*>((obj_)->data)

template <CUPMDeviceKind T>
inline PetscErrorCode CUPMContext<T>::destroy(PetscDeviceContext dctx) noexcept
{
  PetscDeviceContext_IMPLS *dci = IMPLS_RCAST_(dctx);
  cupmError_t              cerr;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  if (dci->event)  {cerr = cupmEventDestroy(dci->event);CHKERRCUPM(cerr);}
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
inline PetscErrorCode CUPMContext<T>::changeStreamType(PetscDeviceContext dctx, PetscStreamType stype) noexcept
{
  PetscDeviceContext_IMPLS *dci = IMPLS_RCAST_(dctx);

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

template <CUPMDeviceKind T>
inline PetscErrorCode CUPMContext<T>::setUp(PetscDeviceContext dctx) noexcept
{
  PetscDeviceContext_IMPLS *dci = IMPLS_RCAST_(dctx);
  PetscErrorCode           ierr;
  cupmError_t              cerr;

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
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %d",dctx->streamType);
    break;
  }
  if (!dci->event) {cerr = cupmEventCreate(&dci->event);CHKERRCUPM(cerr);}
  ierr = __setupHandles(dci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
inline PetscErrorCode CUPMContext<T>::query(PetscDeviceContext dctx, PetscBool *idle) noexcept
{
  cupmError_t cerr;

  PetscFunctionBegin;
  cerr = cupmStreamQuery(IMPLS_RCAST_(dctx)->stream);
  if (cerr == cupmSuccess)
    *idle = PETSC_TRUE;
  else if (cerr == cupmErrorNotReady) {
    *idle = PETSC_FALSE;
  } else {
    // somethings gone wrong
    CHKERRCUPM(cerr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
inline PetscErrorCode CUPMContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) noexcept
{
  PetscDeviceContext_IMPLS *dcia = IMPLS_RCAST_(dctxa);
  PetscDeviceContext_IMPLS *dcib = IMPLS_RCAST_(dctxb);
  cupmError_t               cerr;

  PetscFunctionBegin;
  cerr = cupmEventRecord(dcib->event,dcib->stream);CHKERRCUPM(cerr);
  cerr = cupmStreamWaitEvent(dcia->stream,dcib->event,0);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
inline PetscErrorCode CUPMContext<T>::synchronize(PetscDeviceContext dctx) noexcept
{
  PetscDeviceContext_IMPLS *dci = IMPLS_RCAST_(dctx);
  cupmError_t               cerr;

  PetscFunctionBegin;
  // in case anything was queued on the event
  cerr = cupmStreamWaitEvent(dci->stream,dci->event,0);CHKERRCUPM(cerr);
  cerr = cupmStreamSynchronize(dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

// initialize the static member variables
template <CUPMDeviceKind T>
typename CUPMContext<T>::cupmBlasHandle_t   CUPMContext<T>::_blashandle   = nullptr;

template <CUPMDeviceKind T>
typename CUPMContext<T>::cupmSolverHandle_t CUPMContext<T>::_solverhandle = nullptr;

// shorten this one up a bit
using CUPMContextCuda = CUPMContext<CUPMDeviceKind::CUDA>;
using CUPMContextHip  = CUPMContext<CUPMDeviceKind::HIP>;

// make sure these doesn't leak out
#undef CHKERRCUPM
#undef IMPLS_RCAST_

} // namespace Petsc

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(impls_) Petsc::CUPMContext<Petsc::CUPMDeviceKind::impls_>::PetscDeviceContext_IMPLS

// shorthand for casting dctx->data to the appropriate object to access the handles
#define PDC_IMPLS_RCAST(impls_,obj_) reinterpret_cast<PetscDeviceContext_(impls_) *>((obj_)->data)

#endif /* PETSCDEVICECONTEXTCUDA_HPP */
