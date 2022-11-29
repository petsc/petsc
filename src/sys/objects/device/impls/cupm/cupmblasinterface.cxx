#include <petsc/private/cupmblasinterface.hpp>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

#define PETSC_CUPMBLAS_STATIC_VARIABLE_DEFN(THEIRS, DEVICE, OURS) const decltype(THEIRS) BlasInterfaceImpl<DeviceType::DEVICE>::OURS;

// in case either one or the other don't agree on a name, you can specify all three here:
//
// PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(CUBLAS_STATUS_SUCCESS, rocblas_status_success,
// CUPMBLAS_STATUS_SUCCESS) ->
// const decltype(CUBLAS_STATUS_SUCCESS)  BlasInterface<DeviceType::CUDA>::CUPMBLAS_STATUS_SUCCESS;
// const decltype(rocblas_status_success) BlasInterface<DeviceType::HIP>::CUPMBLAS_STATUS_SUCCESS;
#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(CUORIGINAL, HIPORIGINAL, OURS) \
  PetscIfPetscDefined(HAVE_CUDA, PETSC_CUPMBLAS_STATIC_VARIABLE_DEFN, PetscExpandToNothing)(CUORIGINAL, CUDA, OURS) PetscIfPetscDefined(HAVE_HIP, PETSC_CUPMBLAS_STATIC_VARIABLE_DEFN, PetscExpandToNothing)(HIPORIGINAL, HIP, OURS)

// if both cuda and hip agree on the same naming scheme i.e. CUBLAS_STATUS_SUCCESS and
// HIPBLAS_STATUS_SUCCESS:
//
// PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_PREFIX(STATUS_SUCCESS) ->
// const decltype(CUBLAS_STATUS_SUCCESS)  BlasInterface<DeviceType::CUDA>::CUPMBLAS_STATUS_SUCCESS;
// const decltype(HIPBLAS_STATUS_SUCCESS) BlasInterface<DeviceType::HIP>::CUPMBLAS_STATUS_SUCCESS;
#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(SUFFIX) PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(PetscConcat(CUBLAS_, SUFFIX), PetscConcat(HIPBLAS_, SUFFIX), PetscConcat(CUPMBLAS_, SUFFIX))

PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(STATUS_SUCCESS)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(STATUS_NOT_INITIALIZED)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(STATUS_ALLOC_FAILED)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(POINTER_MODE_HOST)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(POINTER_MODE_DEVICE)

#if PetscDefined(HAVE_CUDA)
template struct BlasInterface<DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
template struct BlasInterface<DeviceType::HIP>;
#endif

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
