#include <petscdevice_cuda.h>
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

template struct SfInterface<device::cupm::DeviceType::CUDA>;

} // namespace impl

} // namespace cupm

} // namespace sf

} // namespace Petsc

using PetscSFCuda = ::Petsc::sf::cupm::impl::SfInterface<::Petsc::device::cupm::DeviceType::CUDA>;

PetscErrorCode PetscSFMalloc_CUDA(PetscMemType mtype, size_t size, void **ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscSFCuda::Malloc(mtype, size, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSFFree_CUDA(PetscMemType mtype, void *ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscSFCuda::Free(mtype, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSFLinkSetUp_CUDA(PetscSF sf, PetscSFLink link, MPI_Datatype unit)
{
  PetscFunctionBegin;
  PetscCall(PetscSFCuda::LinkSetUp(sf, link, unit));
  PetscFunctionReturn(PETSC_SUCCESS);
}
