#include "../../interface/cupmdevice.hpp"

namespace Petsc {

// internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <CUPMDeviceKind T>
class CUPMDevice<T>::PetscDeviceInternal
{
private:
  const int        _id;
  cupmDeviceProp_t _dprop;

public:
  // default constructor
  explicit PetscDeviceInternal(int dev) noexcept : _id{dev} {}

  // gather all relevant information for a particular device, a cupmDeviceProp_t is
  // usually sufficient here
  PETSC_NODISCARD PetscErrorCode initialize() noexcept
  {
    cupmError_t cerr;

    PetscFunctionBegin;
    cerr = cupmGetDeviceProperties(&this->_dprop,this->_id);CHKERRCUPM(cerr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD int id() const noexcept { return this->_id;}
  PETSC_NODISCARD const cupmDeviceProp_t& deviceProp() const noexcept { return this->_dprop;}
};

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::__initialize() noexcept
{
  int         ndev;
  cupmError_t cerr;

  PetscFunctionBegin;
  if (_initialized) PetscFunctionReturn(0);
  cerr = cupmGetDeviceCount(&ndev);CHKERRCUPM(cerr);
  CHKERRCXX(_devices.reserve(ndev));
  for (int i = 0; i < ndev; ++i) {
    PetscErrorCode ierr;

    CHKERRCXX(_devices.emplace_back(std::unique_ptr<PetscDeviceInternal>{new PetscDeviceInternal{i}}));
    ierr = _devices[i]->initialize();CHKERRQ(ierr);
  }
  CHKERRCXX(_devices.shrink_to_fit());
  _initialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::getDevice(PetscDevice &device) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = this->__initialize();CHKERRQ(ierr);
  // default device is always the first device for now?
  device->deviceId = this->_devices[0]->id();
  device->ops->createcontext = this->_create;
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::configureDevice(PetscDevice &device) noexcept
{
  cupmError_t cerr;

  PetscFunctionBegin;
  cerr = cupmSetDevice(device->deviceId);
  // why on EARTH nvidia insists on making otherwise informational states into
  // fully-fledged error codes is beyond me. Why couldn't a pointer to bool argument have
  // sufficed?!?!?!
  if (cerr != cupmErrorDeviceAlreadyInUse) CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

// explicitly instantiate the classes
#if PetscDefined(HAVE_CUDA)
template class CUPMDevice<CUPMDeviceKind::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template class CUPMDevice<CUPMDeviceKind::HIP>;
#endif

} // namespace Petsc
