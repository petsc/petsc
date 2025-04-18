#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <petscpkg_version.h>

#include <petscdevice_hip.h>
#include <petsc/private/sfimpl.h>
#include "../sfcupm.hpp"
#include "../sfcupm_impl.hpp"

namespace Petsc
{

namespace sf
{

namespace cupm
{

namespace impl
{

template struct SfInterface<device::cupm::DeviceType::HIP>;

} // namespace impl

} // namespace cupm

} // namespace sf

} // namespace Petsc

using PetscSFHIP = ::Petsc::sf::cupm::impl::SfInterface<::Petsc::device::cupm::DeviceType::HIP>;

PetscErrorCode PetscSFMalloc_HIP(PetscMemType mtype, size_t size, void **ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscSFHIP::Malloc(mtype, size, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSFFree_HIP(PetscMemType mtype, void *ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscSFHIP::Free(mtype, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSFLinkSetUp_HIP(PetscSF sf, PetscSFLink link, MPI_Datatype unit)
{
  PetscFunctionBegin;
  PetscCall(PetscSFHIP::LinkSetUp(sf, link, unit));
  PetscFunctionReturn(PETSC_SUCCESS);
}
