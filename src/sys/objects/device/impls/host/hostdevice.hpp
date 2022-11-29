#ifndef HOSTDEVICE_HPP
#define HOSTDEVICE_HPP

#if defined(__cplusplus)
  #include "../impldevicebase.hpp" /*I "petscdevice.h" I*/

namespace Petsc
{

namespace device
{

namespace host
{

class Device : public ::Petsc::device::impl::DeviceBase<Device> {
public:
  PETSC_DEVICE_IMPL_BASE_CLASS_HEADER(base_type, Device);

  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm, PetscInt *, PetscBool *, PetscDeviceInitType *) noexcept;

private:
  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceType PETSC_DEVICE_IMPL_()) { return PETSC_DEVICE_HOST; }

  PETSC_NODISCARD static PetscErrorCode get_attribute_(PetscInt, PetscDeviceAttribute, void *) noexcept;
};

} // namespace host

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // HOSTDEVICE_HPP
