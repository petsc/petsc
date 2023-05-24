#ifndef PETSCSYCLDEVICE_HPP
#define PETSCSYCLDEVICE_HPP

#if defined(__cplusplus)
  #include "../impldevicebase.hpp" /* I "petscdevice.h" */

  #include <array>

namespace Petsc
{

namespace device
{

namespace sycl
{

  #define PETSC_SYCL_DEVICE_HOST -1 // Note -1 is also used by PETSC_DECIDE, so user needs to pass -2 to explicitly select the host
  #define PETSC_SYCL_DEVICE_NONE -3

class Device : public ::Petsc::device::impl::DeviceBase<Device> {
public:
  PETSC_DEVICE_IMPL_BASE_CLASS_HEADER(base_type, Device);

  ~Device() { auto PETSC_UNUSED _ = finalize_(); }

  static PetscErrorCode initialize(MPI_Comm, PetscInt *, PetscBool *, PetscDeviceInitType *) noexcept;

private:
  // opaque class representing a single device instance
  class DeviceInternal;

  // currently stores sycl host and gpu devices
  static std::array<DeviceInternal *, PETSC_DEVICE_MAX_DEVICES> devices_array_;
  static DeviceInternal                                       **devices_; // alias to devices_array_, but shifted to support devices_[-1] for sycl host device

  // this rank's default device. If equals to PETSC_SYCL_DEVICE_NONE, then all sycl devices are disabled
  static int defaultDevice_;

  // have we tried looking for devices
  static bool initialized_;

  // clean-up
  static PetscErrorCode finalize_() noexcept;

  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceType PETSC_DEVICE_IMPL_()) { return PETSC_DEVICE_SYCL; }
  PetscErrorCode        init_device_id_(PetscInt *) const noexcept;
  static PetscErrorCode view_device_(PetscDevice, PetscViewer) noexcept;
  static PetscErrorCode get_attribute_(PetscInt, PetscDeviceAttribute, void *) noexcept;
};

} // namespace sycl

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif /* PETSCSYCLDEVICE_HPP */
