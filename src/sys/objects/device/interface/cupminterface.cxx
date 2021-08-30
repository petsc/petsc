#include <petsc/private/cupminterface.hpp> /* I "petscdevice.h" */

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc {

#if PetscDefined(HAVE_CUDA)
const decltype(cudaSuccess)                 CUPMInterface<CUPMDeviceKind::CUDA>::cupmSuccess;
const decltype(cudaErrorNotReady)           CUPMInterface<CUPMDeviceKind::CUDA>::cupmErrorNotReady;
const decltype(cudaStreamNonBlocking)       CUPMInterface<CUPMDeviceKind::CUDA>::cupmStreamNonBlocking;
const decltype(cudaErrorDeviceAlreadyInUse) CUPMInterface<CUPMDeviceKind::CUDA>::cupmErrorDeviceAlreadyInUse;
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
const decltype(hipSuccess)            CUPMInterface<CUPMDeviceKind::HIP>::cupmSuccess;
const decltype(hipErrorNotReady)      CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorNotReady;
const decltype(hipStreamNonBlocking)  CUPMInterface<CUPMDeviceKind::HIP>::cupmStreamNonBlocking;
const decltype(hipSuccess)            CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorDeviceAlreadyInUse;
#endif // PetscDefined(HAVE_HIP)

} // namespace Petsc
