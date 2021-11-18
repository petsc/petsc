#ifndef PETSCCUPMDEVICE_HPP
#define PETSCCUPMDEVICE_HPP

#include <petsc/private/deviceimpl.h> /* I "petscdevice.h" */
#include <petsc/private/cupminterface.hpp>
#include <petscviewer.h>
#include <array>
#include <memory>
#include <limits>

namespace Petsc
{

#if defined(PETSC_CUPM_DEVICE_NONE)
#  error "redefinition of PETSC_CUPM_DEVICE_NONE"
#endif

#define PETSC_CUPM_DEVICE_NONE -3

template <CUPMDeviceType T>
class CUPMDevice : CUPMInterface<T>
{
public:
  using createContextFunction_t = PetscErrorCode (*)(PetscDeviceContext);
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,T)

  // default constructor
  explicit CUPMDevice(createContextFunction_t func) noexcept : _create(func) { }

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

  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm,PetscInt*,PetscDeviceInitType*) noexcept;

  PETSC_NODISCARD PetscErrorCode getDevice(PetscDevice,PetscInt) const noexcept;

  PETSC_NODISCARD static PetscErrorCode configureDevice(PetscDevice) noexcept;

  PETSC_NODISCARD static PetscErrorCode viewDevice(PetscDevice,PetscViewer) noexcept;

private:
  // opaque class representing a single device
  class CUPMDeviceInternal;

  // all known devices
  static std::array<std::unique_ptr<CUPMDeviceInternal>,PETSC_DEVICE_MAX_DEVICES> _devices;

  // this ranks default device, if < 0  then devices are specifically disabled
  static int _defaultDevice;

  // function to create a PetscDeviceContext (the (*create) function pointer usually set
  // via XXXSetType() for other PETSc objects)
  const createContextFunction_t _create;

  // have we tried looking for devices
  static bool _initialized;

  // clean-up
  PETSC_NODISCARD static PetscErrorCode __finalize() noexcept;
};

// define static variables
template <CUPMDeviceType T> bool CUPMDevice<T>::_initialized = false;

template <CUPMDeviceType T>
std::array<std::unique_ptr<typename CUPMDevice<T>::CUPMDeviceInternal>,PETSC_DEVICE_MAX_DEVICES>
CUPMDevice<T>::_devices = { };

template <CUPMDeviceType T> int CUPMDevice<T>::_defaultDevice = PETSC_CUPM_DEVICE_NONE;

} // namespace Petsc

#endif /* PETSCCUPMDEVICE_HPP */
