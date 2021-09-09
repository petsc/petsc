#ifndef PETSCCUPMDEVICE_HPP
#define PETSCCUPMDEVICE_HPP

#include <petsc/private/deviceimpl.h> /* I "petscdevice.h" */
#include <petsc/private/cupminterface.hpp>
#include <vector>
#include <memory>

namespace Petsc {

template <CUPMDeviceKind T>
class CUPMDevice : CUPMInterface<T>
{
public:
  typedef PetscErrorCode (*createContextFunc_t)(PetscDeviceContext);
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,T)

  // default constructor
  explicit CUPMDevice(createContextFunc_t func) : _create{func} {}

  // copy constructor
  CUPMDevice(const CUPMDevice &other) noexcept = default;

  // move constructor
  CUPMDevice(CUPMDevice &&other) noexcept = default;

  // destructor
  ~CUPMDevice() noexcept = default;

  // copy assignment operator
  CUPMDevice& operator=(const CUPMDevice &other) = default;

  // move assignment operator
  CUPMDevice& operator=(CUPMDevice &&other) noexcept = default;

  PETSC_NODISCARD PetscErrorCode getDevice(PetscDevice&) noexcept;

  PETSC_NODISCARD PetscErrorCode configureDevice(PetscDevice&) noexcept;

private:
  // opaque class representing a single device
  class PetscDeviceInternal;

  // all known devices
  static std::vector<std::unique_ptr<PetscDeviceInternal>> _devices;

  // function to create a PetscDeviceContext (the (*create) function pointer usually set
  // via XXXSetType() for other PETSc objects)
  createContextFunc_t _create;

  // have we tried looking for devices
  static PetscBool _initialized;

  // look for devices
  PETSC_NODISCARD static PetscErrorCode __initialize() noexcept;
};

// define static variables
template <CUPMDeviceKind T_>
PetscBool CUPMDevice<T_>::_initialized = PETSC_FALSE;

template <CUPMDeviceKind T_>
std::vector<std::unique_ptr<typename CUPMDevice<T_>::PetscDeviceInternal>> CUPMDevice<T_>::_devices;

} // namespace Petsc

#endif /* PETSCCUPMDEVICE_HPP */
