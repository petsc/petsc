#ifndef PETSCCUPMDEVICE_HPP
#define PETSCCUPMDEVICE_HPP

#include <petsc/private/deviceimpl.h> /* I "petscdevice.h" */
#include <petsc/private/cupminterface.hpp>
#include <petsc/private/viewerimpl.h>
#include <array>
#include <memory>
#include <limits>

namespace Petsc
{

namespace Device
{

namespace CUPM
{

#if defined(PETSC_CUPM_DEVICE_NONE)
#  error redefinition of PETSC_CUPM_DEVICE_NONE
#endif

#define PETSC_CUPM_DEVICE_NONE -3

template <DeviceType T>
class Device : Impl::Interface<T>
{
public:
  using createContextFunction_t = PetscErrorCode (*)(PetscDeviceContext);
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(cupmInterface_t,T);

  // default constructor
  explicit Device(createContextFunction_t func) noexcept : create_(func) { }

  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm,PetscInt*,PetscDeviceInitType*) noexcept;

  PETSC_NODISCARD PetscErrorCode getDevice(PetscDevice,PetscInt) const noexcept;

  PETSC_NODISCARD static PetscErrorCode configureDevice(PetscDevice) noexcept;

  PETSC_NODISCARD static PetscErrorCode viewDevice(PetscDevice,PetscViewer) noexcept;

private:
  // opaque class representing a single device
  class DeviceInternal;

  // all known devices
  static std::array<std::unique_ptr<DeviceInternal>,PETSC_DEVICE_MAX_DEVICES> devices_;

  // this ranks default device, if < 0  then devices are specifically disabled
  static int defaultDevice_;

  // function to create a PetscDeviceContext (the (*create) function pointer usually set
  // via XXXSetType() for other PETSc objects)
  const createContextFunction_t create_;

  // have we tried looking for devices
  static bool initialized_;

  // clean-up
  PETSC_NODISCARD static PetscErrorCode finalize_() noexcept;
};

// define static variables
template <DeviceType T> bool Device<T>::initialized_ = false;

template <DeviceType T>
std::array<std::unique_ptr<typename Device<T>::DeviceInternal>,PETSC_DEVICE_MAX_DEVICES>
Device<T>::devices_ = { };

template <DeviceType T> int Device<T>::defaultDevice_ = PETSC_CUPM_DEVICE_NONE;

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif /* PETSCCUPMDEVICE_HPP */
