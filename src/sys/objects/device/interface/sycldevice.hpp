#ifndef PETSCSYCLDEVICE_HPP
#define PETSCSYCLDEVICE_HPP

#include <petsc/private/deviceimpl.h> /* I "petscdevice.h" */
#include <petscviewer.h>
#include <array>
#include <limits>

namespace Petsc
{
#define PETSC_SYCL_DEVICE_HOST -1  // Note -1 is also used by PETSC_DECIDE, so user needs to pass -2 to explicitly select the host
#define PETSC_SYCL_DEVICE_NONE -3

class SyclDevice
{
public:
  using createContextFunction_t = PetscErrorCode (*)(PetscDeviceContext);

  explicit SyclDevice(createContextFunction_t func) noexcept : create_(func) { }
  ~SyclDevice() {static_cast<void>(finalize_());}

  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm,PetscInt*,PetscDeviceInitType*) noexcept;
  PETSC_NODISCARD PetscErrorCode getDevice(PetscDevice,PetscInt) const noexcept;
  PETSC_NODISCARD static PetscErrorCode configureDevice(PetscDevice) noexcept;
  PETSC_NODISCARD static PetscErrorCode viewDevice(PetscDevice,PetscViewer) noexcept;

private:
  // opaque class representing a single device instance
  class SyclDeviceInternal;

  const createContextFunction_t create_;

  // currently stores sycl host and gpu devices
  static std::array<SyclDeviceInternal*,PETSC_DEVICE_MAX_DEVICES> devices_array_;
  static SyclDeviceInternal **devices_; // alias to devices_array_, but shifted to support devices_[-1] for sycl host device

  // this rank's default device. If equals to PETSC_SYCL_DEVICE_NONE, then all sycl devices are disabled
  static int defaultDevice_;

  // have we tried looking for devices
  static bool initialized_;

  // clean-up
  PETSC_NODISCARD static PetscErrorCode finalize_() noexcept;
};


} // namespace Petsc
#endif /* PETSCSYCLDEVICE_HPP */
