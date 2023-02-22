#include "hostdevice.hpp"

namespace Petsc
{

namespace device
{

namespace host
{

PetscErrorCode Device::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscBool *defaultView, PetscDeviceInitType *defaultInitType) noexcept
{
  PetscFunctionBegin;
  // the host is always id 0
  *defaultDeviceId = 0;
  // the host is always "lazily" initialized
  *defaultInitType = PETSC_DEVICE_INIT_LAZY;

  PetscOptionsBegin(comm, nullptr, "PetscDevice host Options", "Sys");
  PetscCall(base_type::PetscOptionDeviceView(PetscOptionsObject, defaultView, nullptr));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Device::get_attribute_(PetscInt, PetscDeviceAttribute attr, void *value) noexcept
{
  PetscFunctionBegin;
  switch (attr) {
  case PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK:
    *static_cast<std::size_t *>(value) = 64000;
    break;
  default:
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace host

} // namespace device

} // namespace Petsc
