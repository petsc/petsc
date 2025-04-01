#include "sycldevice.hpp"
#include <sycl/sycl.hpp>
#include <Kokkos_Core.hpp>

namespace Petsc
{

namespace device
{

namespace sycl
{

namespace impl
{

class DeviceContext {
public:
  struct PetscDeviceContext_SYCL {
    ::sycl::event event;
    ::sycl::event begin;   // timer-only
    ::sycl::event end;     // timer-only
    Kokkos::Timer timer{}; // use cpu time since sycl events are return value of queue submission and we have no infrastructure to store them
    double        timeBegin{};
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse{};
#endif
    ::sycl::queue queue;
  };

private:
  static bool initialized_;

  static PetscErrorCode finalize_() noexcept
  {
    PetscFunctionBegin;
    initialized_ = false;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode initialize_(PetscInt id, PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscDeviceCheckDeviceCount_Internal(id));
    if (!initialized_) {
      initialized_ = true;
      PetscCall(PetscRegisterFinalize(finalize_));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

public:
  const struct _DeviceContextOps ops = {destroy, changeStreamType, setUp, query, waitForContext, synchronize, getBlasHandle, getSolverHandle, getStreamHandle, beginTimer, endTimer, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  // default constructor
  DeviceContext() noexcept = default;

  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  static PetscErrorCode destroy(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    delete static_cast<PetscDeviceContext_SYCL *>(dctx->data);
    dctx->data = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode setUp(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    static_cast<PetscDeviceContext_SYCL *>(dctx->data)->timerInUse = PETSC_FALSE;
#endif
    // petsc/sycl currently only uses Kokkos's default execution space (and its queue),
    // so in some sense, we have only one PETSc device context.
    PetscCall(PetscKokkosInitializeCheck());
    static_cast<PetscDeviceContext_SYCL *>(dctx->data)->queue = Kokkos::DefaultExecutionSpace().sycl_queue();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode query(PetscDeviceContext dctx, PetscBool *idle) noexcept
  {
    PetscFunctionBegin;
    // available in future, https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_queue_empty.asciidoc
    // *idle = static_cast<PetscDeviceContext_SYCL*>(dctx->data)->queue.empty() ? PETSC_TRUE : PETSC_FALSE;
    *idle = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode synchronize(PetscDeviceContext dctx) noexcept
  {
    PetscBool  idle = PETSC_TRUE;
    const auto dci  = static_cast<PetscDeviceContext_SYCL *>(dctx->data);

    PetscFunctionBegin;
    PetscCall(query(dctx, &idle));
    if (!idle) PetscCallCXX(dci->queue.wait());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode getStreamHandle(PetscDeviceContext dctx, void **handle) noexcept
  {
    PetscFunctionBegin;
    *reinterpret_cast<::sycl::queue **>(handle) = &(static_cast<PetscDeviceContext_SYCL *>(dctx->data)->queue);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode beginTimer(PetscDeviceContext dctx) noexcept
  {
    const auto dci = static_cast<PetscDeviceContext_SYCL *>(dctx->data);

    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    PetscCheck(!dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeEnd()?");
    dci->timerInUse = PETSC_TRUE;
#endif
    PetscCallCXX(dci->timeBegin = dci->timer.seconds());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed) noexcept
  {
    const auto dci = static_cast<PetscDeviceContext_SYCL *>(dctx->data);

    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
    dci->timerInUse = PETSC_FALSE;
#endif
    PetscCallCXX(dci->queue.wait());
    PetscCallCXX(*elapsed = dci->timer.seconds() - dci->timeBegin);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  static PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  static PetscErrorCode getBlasHandle(PetscDeviceContext, void *) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  static PetscErrorCode getSolverHandle(PetscDeviceContext, void *) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
};

} // namespace impl

} // namespace sycl

} // namespace device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext dctx)
{
  using namespace Petsc::device::sycl::impl;

  static const DeviceContext syclctx;

  PetscFunctionBegin;
  PetscCallCXX(dctx->data = new DeviceContext::PetscDeviceContext_SYCL());
  dctx->ops[0] = syclctx.ops;
  PetscFunctionReturn(PETSC_SUCCESS);
}
