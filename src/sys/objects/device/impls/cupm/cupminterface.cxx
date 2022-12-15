#include <petsc/private/cupminterface.hpp>
#include <petsc/private/petscadvancedmacros.h>

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

#define PETSC_CUPM_STATIC_VARIABLE_DEFN(theirs, DEVICE, ours) const decltype(theirs) InterfaceImpl<DeviceType::DEVICE>::ours

#define PETSC_CUPM_STATIC_VARIABLE_DEFN_CLASS(type_name, DEVICE, ours) const typename InterfaceImpl<DeviceType::DEVICE>::type_name InterfaceImpl<DeviceType::DEVICE>::ours

#define PETSC_CUPM_STATIC_VARIABLE_DEFN_EXACT(type_name, DEVICE, ours) const type_name InterfaceImpl<DeviceType::DEVICE>::ours

// in case either one or the other don't agree on a name, you can specify all three here:
//
// PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(cudaSuccess, hipAllGood, cupmSuccess) ->
// const decltype(cudaSuccess) Interface<DeviceType::CUDA>::cupmSuccess;
// const decltype(hipAllGood)  Interface<DeviceType::HIP>::cupmSuccess;
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(cuoriginal, hiporiginal, ours) \
  PetscIfPetscDefined(HAVE_CUDA, PETSC_CUPM_STATIC_VARIABLE_DEFN, PetscExpandToNothing)(cuoriginal, CUDA, ours); \
  PetscIfPetscDefined(HAVE_HIP, PETSC_CUPM_STATIC_VARIABLE_DEFN, PetscExpandToNothing)(hiporiginal, HIP, ours)

// define the static variable in terms of the class typename
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(type_name, ours) \
  PetscIfPetscDefined(HAVE_CUDA, PETSC_CUPM_STATIC_VARIABLE_DEFN_CLASS, PetscExpandToNothing)(type_name, CUDA, ours); \
  PetscIfPetscDefined(HAVE_HIP, PETSC_CUPM_STATIC_VARIABLE_DEFN_CLASS, PetscExpandToNothing)(type_name, HIP, ours)

#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_EXACT_TYPENAME(type_name, ours) \
  PetscIfPetscDefined(HAVE_CUDA, PETSC_CUPM_STATIC_VARIABLE_DEFN_EXACT, PetscExpandToNothing)(type_name, CUDA, ours); \
  PetscIfPetscDefined(HAVE_HIP, PETSC_CUPM_STATIC_VARIABLE_DEFN_EXACT, PetscExpandToNothing)(type_name, HIP, ours)

// if both cuda and hip agree on the same naming scheme i.e. cudaSuccess and hipSuccess:
//
// PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(Success) ->
// const decltype(cudaSuccess) Interface<DeviceType::CUDA>::cupmSuccess;
// const decltype(hipSuccess)  Interface<DeviceType::HIP>::cupmSuccess;
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(suffix) PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(PetscConcat(cuda, suffix), PetscConcat(hip, suffix), PetscConcat(cupm, suffix))

// error codes
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmError_t, cupmSuccess);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmError_t, cupmErrorNotReady);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmError_t, cupmErrorSetOnActiveProcess);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmError_t, cupmErrorNoDevice);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmError_t, cupmErrorDeviceAlreadyInUse);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmError_t, cupmErrorStubLibrary);

// enums
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(StreamDefault);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(StreamNonBlocking);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(DeviceMapHost);

PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmMemcpyKind_t, cupmMemcpyHostToDevice);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmMemcpyKind_t, cupmMemcpyDeviceToHost);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmMemcpyKind_t, cupmMemcpyDeviceToDevice);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmMemcpyKind_t, cupmMemcpyHostToHost);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmMemcpyKind_t, cupmMemcpyDefault);

PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemoryTypeHost);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemoryTypeDevice);
// A vile, vile, hack. Would use PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME() with
// cupmMemoryType_t however MSVC chokes on this with:
//
// \src\sys\objects\device\impls\cupm\cupminterface.cxx(69): error C2371: 'cupmMemoryTypeHost':
// redefinition; different basic types
// \include\petsc/private/cupminterface.hpp(314): note: see declaration of 'cupmMemoryTypeHost'
// \src\sys\objects\device\impls\cupm\cupminterface.cxx(70): error C2371:
// 'cupmMemoryTypeDevice': redefinition; different basic types
// \include\petsc/private/cupminterface.hpp(315): note: see declaration of 'cupmMemoryTypeDevice'
// \src\sys\objects\device\impls\cupm\cupminterface.cxx(71): error C2371:
// 'cupmMemoryTypeManaged': redefinition; different basic types
// \include\petsc/private/cupminterface.hpp(316): note: see declaration of
// 'cupmMemoryTypeManaged'
//
// the only way to get it to compile is to use
// PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME() but since cupmMemoryTypeManaged is
// secretly hipMemoryTypeUnified that doesn't work unless we fudge it with preprocessor
#define hipMemoryTypeManaged hipMemoryTypeUnified
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemoryTypeManaged);

PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(EventDisableTiming);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_EXACT_TYPENAME(int, cupmHostAllocDefault);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_EXACT_TYPENAME(int, cupmHostAllocWriteCombined);

PETSC_CUPM_DEFINE_STATIC_VARIABLE_VIA_CLASS_TYPENAME(cupmMemPoolAttr, cupmMemPoolAttrReleaseThreshold);

#if PetscDefined(HAVE_CUDA)
template struct Interface<DeviceType::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template struct Interface<DeviceType::HIP>;
#endif

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
