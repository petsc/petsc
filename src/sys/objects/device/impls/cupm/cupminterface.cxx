#include <petsc/private/cupminterface.hpp>

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc
{

// do all of this with macros to enforce that both CUDA and HIP implementations both have
// things defined. If you for example implement something on the HIP side but forget to
// implement it on the CUDA side you'll get an error.

// need these for the indirection when building the if_0 and if_1 variants of the macro
#define CAT_(x,y) x ## y
#define CAT(x,y) CAT_(x,y)

#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT_0(PREFIX,original,mapped)
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT_1(PREFIX,original,mapped) \
  const decltype(original) CUPMInterface<CUPMDeviceType::PREFIX>::mapped;

#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT(HAVE,PREFIX,orginal,mapped) \
  CAT(PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT_,HAVE)(PREFIX,orginal,mapped)

#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT_(PREFIX,orginal,mapped) \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT(PetscDefined(HAVE_ ## PREFIX),PREFIX,orginal,mapped)

// in case either one or the other don't agree on a name, you can specify all three here
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(cuoriginal,hiporiginal,mapped) \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT_(CUDA,CAT(cuda,cuoriginal),CAT(cupm,mapped)) \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT_(HIP,CAT(hip,hiporiginal),CAT(cupm,mapped))

// if both cuda and hip agree on the same name
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE(stem)                 \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(stem,stem,stem)

// error codes
PETSC_CUPM_DEFINE_STATIC_VARIABLE(Success)
PETSC_CUPM_DEFINE_STATIC_VARIABLE(ErrorNotReady)
PETSC_CUPM_DEFINE_STATIC_VARIABLE(ErrorSetOnActiveProcess)
PETSC_CUPM_DEFINE_STATIC_VARIABLE(ErrorNoDevice)

// hip not conforming, see declaration in cupminterface.hpp
PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(ErrorDeviceAlreadyInUse,ErrorContextAlreadyInUse,ErrorDeviceAlreadyInUse)

// hip not conforming, and cuda faffs around with versions see declaration in cupminterface.hpp
#if PetscDefined(HAVE_CUDA)
#  if PETSC_PKG_CUDA_VERSION_GE(11,1,0)
#    define PetscCudaErrorStubLibrary ErrorStubLibrary
#  endif
#endif

#ifndef PetscCudaErrorStubLibrary
#define PetscCudaErrorStubLibrary ErrorInsufficientDriver
#endif

PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(PetscCudaErrorStubLibrary,ErrorInsufficientDriver,ErrorStubLibrary)

// enums
PETSC_CUPM_DEFINE_STATIC_VARIABLE(StreamNonBlocking)
PETSC_CUPM_DEFINE_STATIC_VARIABLE(DeviceMapHost)
PETSC_CUPM_DEFINE_STATIC_VARIABLE(MemcpyHostToDevice)

} // namespace Petsc
