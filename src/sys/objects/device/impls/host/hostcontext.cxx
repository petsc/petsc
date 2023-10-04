#include <petsc/private/deviceimpl.h>

#include <petsc/private/cpp/utility.hpp> // PetscObjectCast()

namespace Petsc
{

namespace device
{

namespace host
{

namespace impl
{

class DeviceContext {
public:
  static PetscErrorCode destroy(PetscDeviceContext) noexcept { return PETSC_SUCCESS; }
  static PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType) noexcept { return PETSC_SUCCESS; }
  static PetscErrorCode setUp(PetscDeviceContext) noexcept { return PETSC_SUCCESS; }
  static PetscErrorCode query(PetscDeviceContext, PetscBool *idle) noexcept
  {
    PetscFunctionBegin;
    *idle = PETSC_TRUE; // the host is always idle
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  static PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext) noexcept { return PETSC_SUCCESS; }
  static PetscErrorCode synchronize(PetscDeviceContext) noexcept { return PETSC_SUCCESS; }

  // clang-format off
  const _DeviceContextOps ops = {
    PetscDesignatedInitializer(destroy, destroy),
    PetscDesignatedInitializer(changestreamtype, changeStreamType),
    PetscDesignatedInitializer(setup, setUp),
    PetscDesignatedInitializer(query, query),
    PetscDesignatedInitializer(waitforcontext, waitForContext),
    PetscDesignatedInitializer(synchronize, synchronize),
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

} // namespace impl

} // namespace host

} // namespace device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_HOST(PetscDeviceContext dctx)
{
  static constexpr auto hostctx = ::Petsc::device::host::impl::DeviceContext{};

  PetscFunctionBegin;
  PetscAssert(!dctx->data, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt64_FMT " is of type host, but still has data member %p", PetscObjectCast(dctx)->id, dctx->data);
  *dctx->ops = hostctx.ops;
  PetscFunctionReturn(PETSC_SUCCESS);
}
