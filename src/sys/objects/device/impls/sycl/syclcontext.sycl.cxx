#include "sycldevice.hpp"
#include <sycl/sycl.hpp>
#include <chrono>

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
    ::sycl::event begin; // timer-only
    ::sycl::event end;   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse{};
#endif
    ::sycl::queue queue;

    std::chrono::time_point<std::chrono::steady_clock> timeBegin{};
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
    PetscDevice dev;
    PetscInt    id;

    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    static_cast<PetscDeviceContext_SYCL *>(dctx->data)->timerInUse = PETSC_FALSE;
#endif
    PetscCall(PetscDeviceContextGetDevice(dctx, &dev));
    PetscCall(PetscDeviceGetDeviceId(dev, &id));
    const ::sycl::device &syclDevice = (id == PETSC_SYCL_DEVICE_HOST) ? ::sycl::device(::sycl::cpu_selector_v) : ::sycl::device::get_devices(::sycl::info::device_type::gpu)[id];

    static_cast<PetscDeviceContext_SYCL *>(dctx->data)->queue = ::sycl::queue(syclDevice, ::sycl::property::queue::in_order());
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
    // It is not a good approach to time SYCL kernels because the timer starts at the kernel launch time at host,
    // not at the start of execution time on device. SYCL provides this style of kernel timing:
    /*
      sycl::queue q(sycl::default_selector_v, sycl::property::queue::enable_profiling{});
      sycl::event e = q.submit([&](sycl::handler &h) {
        ...
      });
      e.wait();
      auto start_time = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto end_time = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      long long kernel_duration_ns = end_time - start_time;
    */
    // It requires 1) enable profiling at the queue's creation time, and 2) store the event returned by kernel launch.
    // But neither we have control of the input queue, nor does PetscDeviceContext support 2), so we just use a
    // host side timer.
    PetscCallCXX(dci->timeBegin = std::chrono::steady_clock::now());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed) noexcept
  {
    const auto                    dci = static_cast<PetscDeviceContext_SYCL *>(dctx->data);
    std::chrono::duration<double> duration;

    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
    dci->timerInUse = PETSC_FALSE;
#endif
    PetscCallCXX(dci->queue.wait());
    PetscCallCXX(duration = std::chrono::steady_clock::now() - dci->timeBegin);
    PetscCallCXX(*elapsed = duration.count());
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
