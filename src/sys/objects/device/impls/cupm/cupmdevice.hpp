#ifndef PETSCCUPMDEVICE_HPP
#define PETSCCUPMDEVICE_HPP

#if defined(__cplusplus)
  #include <petsc/private/cupminterface.hpp>
  #include <petsc/private/cpp/memory.hpp>
  #include <petsc/private/cpp/array.hpp>

  #include "../impldevicebase.hpp" /* I "petscdevice.h" */

namespace Petsc
{

namespace device
{

namespace cupm
{

  #if defined(PETSC_CUPM_DEVICE_NONE)
    #error redefinition of PETSC_CUPM_DEVICE_NONE
  #endif

  #define PETSC_CUPM_DEVICE_NONE -3

template <DeviceType T>
class Device : public ::Petsc::device::impl::DeviceBase<Device<T>>, impl::Interface<T> {
public:
  PETSC_DEVICE_IMPL_BASE_CLASS_HEADER(base_type, Device<T>);
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(cupmInterface_t, T);

  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm, PetscInt *, PetscBool *, PetscDeviceInitType *) noexcept;

private:
  // opaque class representing a single device
  class DeviceInternal;

  // all known devices
  using devices_type = std::array<std::unique_ptr<DeviceInternal>, PETSC_DEVICE_MAX_DEVICES>;
  static devices_type devices_;

  // this ranks default device, if < 0  then devices are specifically disabled
  static int defaultDevice_;

  // have we tried looking for devices
  static bool initialized_;

  // clean-up
  PETSC_NODISCARD static PetscErrorCode finalize_() noexcept;

  PETSC_NODISCARD static constexpr PetscDeviceType PETSC_DEVICE_IMPL_() noexcept { return PETSC_DEVICE_CUPM(); }

  PETSC_NODISCARD PetscErrorCode        init_device_id_(PetscInt *) const noexcept;
  PETSC_NODISCARD static PetscErrorCode configure_device_(PetscDevice) noexcept;
  PETSC_NODISCARD static PetscErrorCode view_device_(PetscDevice, PetscViewer) noexcept;
  PETSC_NODISCARD static PetscErrorCode get_attribute_(PetscInt, PetscDeviceAttribute, void *) noexcept;
};

// define static variables
template <DeviceType T>
typename Device<T>::devices_type Device<T>::devices_ = {};

template <DeviceType T>
int Device<T>::defaultDevice_ = PETSC_CUPM_DEVICE_NONE;

template <DeviceType T>
bool Device<T>::initialized_ = false;

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif /* PETSCCUPMDEVICE_HPP */
