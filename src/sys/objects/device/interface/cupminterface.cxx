#include <petsc/private/cupminterface.hpp> /* I "petscdevice.h" */

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc {

#if PetscDefined(HAVE_CUDA)
constexpr decltype(cudaSuccess)                 CUPMInterface<CUPMDeviceKind::CUDA>::cupmSuccess;
constexpr decltype(cudaErrorNotReady)           CUPMInterface<CUPMDeviceKind::CUDA>::cupmErrorNotReady;
constexpr decltype(cudaStreamNonBlocking)       CUPMInterface<CUPMDeviceKind::CUDA>::cupmStreamNonBlocking;
constexpr decltype(cudaErrorDeviceAlreadyInUse) CUPMInterface<CUPMDeviceKind::CUDA>::cupmErrorDeviceAlreadyInUse;
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
constexpr decltype(hipSuccess)            CUPMInterface<CUPMDeviceKind::HIP>::cupmSuccess;
constexpr decltype(hipErrorNotReady)      CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorNotReady;
constexpr decltype(hipStreamNonBlocking)  CUPMInterface<CUPMDeviceKind::HIP>::cupmStreamNonBlocking;
constexpr decltype(hipSuccess)            CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorDeviceAlreadyInUse;
#endif // PetscDefined(HAVE_HIP)

} // namespace Petsc
