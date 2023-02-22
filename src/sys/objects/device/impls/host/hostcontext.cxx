#include <petsc/private/deviceimpl.h>

#include <petsc/private/cpp/macros.hpp>
#include <petsc/private/cpp/utility.hpp>

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
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext)) { return PETSC_SUCCESS; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType)) { return PETSC_SUCCESS; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setUp(PetscDeviceContext)) { return PETSC_SUCCESS; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(PetscDeviceContext, PetscBool *idle))
  {
    PetscFunctionBegin;
    *idle = PETSC_TRUE; // the host is always idle
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext)) { return PETSC_SUCCESS; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext)) { return PETSC_SUCCESS; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getBlasHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getSolverHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getStreamHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }

  const _DeviceContextOps ops = {destroy, changeStreamType, setUp, query, waitForContext, synchronize, getBlasHandle, getSolverHandle, getStreamHandle, beginTimer, endTimer, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
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
  PetscCall(PetscArraycpy(dctx->ops, &hostctx.ops, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
