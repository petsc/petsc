#include "sycldevice.hpp"
#include <CL/sycl.hpp>

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
    ::sycl::event event{};
    ::sycl::event begin{}; // timer-only
    ::sycl::event end{};   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse{};
#endif
  };

private:
  PETSC_NODISCARD static PetscDeviceContext_SYCL *impls_cast_(PetscDeviceContext dctx) noexcept { return static_cast<PetscDeviceContext_SYCL *>(dctx->data); }

public:
  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  static PetscErrorCode destroy(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    delete impls_cast_(dctx);
    dctx->data = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  // clang-format off
  static constexpr _DeviceContextOps ops = {
    PetscDesignatedInitializer(destroy, nullptr),
    PetscDesignatedInitializer(changestreamtype, nullptr),
    PetscDesignatedInitializer(setup, nullptr),
    PetscDesignatedInitializer(query, nullptr),
    PetscDesignatedInitializer(waitforcontext, nullptr),
    PetscDesignatedInitializer(synchronize, nullptr),
    PetscDesignatedInitializer(getblashandle, nullptr),
    PetscDesignatedInitializer(getsolverhandle, nullptr),
    PetscDesignatedInitializer(getstreamhandle, nullptr),
    PetscDesignatedInitializer(begintimer, nullptr),
    PetscDesignatedInitializer(endtimer, nullptr),
    PetscDesignatedInitializer(memalloc, nullptr),
    PetscDesignatedInitializer(memfree, nullptr),
    PetscDesignatedInitializer(memcopy, nullptr),
    PetscDesignatedInitializer(memset, nullptr),
    PetscDesignatedInitializer(createevent, nullptr),
    PetscDesignatedInitializer(recordevent, nullptr),
    PetscDesignatedInitializer(waitforevent, nullptr)
  };
  // clang-format on
};

constexpr _DeviceContextOps DeviceContext::ops;

} // namespace impl

} // namespace sycl

} // namespace device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext dctx)
{
  using namespace Petsc::device::sycl::impl;

  static constexpr DeviceContext syclctx;

  PetscFunctionBegin;
  PetscCallCXX(dctx->data = new DeviceContext::PetscDeviceContext_SYCL{});
  *(dctx->ops) = syclctx.ops;
  PetscFunctionReturn(PETSC_SUCCESS);
}
