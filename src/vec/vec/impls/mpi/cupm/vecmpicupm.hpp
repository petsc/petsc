#pragma once

#include <petsc/private/veccupmimpl.h> /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/seq/cupm/vecseqcupm.hpp>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>

namespace Petsc
{

namespace vec
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType T>
class VecMPI_CUPM : public Vec_CUPMBase<T, VecMPI_CUPM<T>> {
public:
  PETSC_VEC_CUPM_BASE_CLASS_HEADER(base_type, T, VecMPI_CUPM<T>);
  using VecSeq_T = VecSeq_CUPM<T>;

private:
  PETSC_NODISCARD static Vec_MPI          *VecIMPLCast_(Vec) noexcept;
  PETSC_NODISCARD static constexpr VecType VECIMPLCUPM_() noexcept;
  PETSC_NODISCARD static constexpr VecType VECIMPL_() noexcept;

  static PetscErrorCode VecDestroy_IMPL_(Vec) noexcept;
  static PetscErrorCode VecResetArray_IMPL_(Vec) noexcept;
  static PetscErrorCode VecPlaceArray_IMPL_(Vec, const PetscScalar *) noexcept;
  static PetscErrorCode VecCreate_IMPL_Private_(Vec, PetscBool *, PetscInt, PetscScalar *) noexcept;

  static PetscErrorCode CreateMPICUPM_(Vec, PetscDeviceContext, PetscBool /*allocate_missing*/ = PETSC_TRUE, PetscInt /*nghost*/ = 0, PetscScalar * /*host_array*/ = nullptr, PetscScalar * /*device_array*/ = nullptr) noexcept;

public:
  // callable directly via a bespoke function
  static PetscErrorCode CreateMPICUPM(MPI_Comm, PetscInt, PetscInt, PetscInt, Vec *, PetscBool) noexcept;
  static PetscErrorCode CreateMPICUPMWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], Vec *) noexcept;

  static PetscErrorCode Duplicate(Vec, Vec *) noexcept;
  static PetscErrorCode BindToCPU(Vec, PetscBool) noexcept;
  static PetscErrorCode Norm(Vec, NormType, PetscReal *) noexcept;
  static PetscErrorCode Dot(Vec, Vec, PetscScalar *) noexcept;
  static PetscErrorCode TDot(Vec, Vec, PetscScalar *) noexcept;
  static PetscErrorCode MDot(Vec, PetscInt, const Vec[], PetscScalar *) noexcept;
  static PetscErrorCode DotNorm2(Vec, Vec, PetscScalar *, PetscScalar *) noexcept;
  static PetscErrorCode Max(Vec, PetscInt *, PetscReal *) noexcept;
  static PetscErrorCode Min(Vec, PetscInt *, PetscReal *) noexcept;
  static PetscErrorCode SetPreallocationCOO(Vec, PetscCount, const PetscInt[]) noexcept;
  static PetscErrorCode SetValuesCOO(Vec, const PetscScalar[], InsertMode) noexcept;
  static PetscErrorCode ErrorWnorm(Vec, Vec, Vec, NormType, PetscReal, Vec, PetscReal, Vec, PetscReal, PetscReal *, PetscInt *, PetscReal *, PetscInt *, PetscReal *, PetscInt *) noexcept;
};

} // namespace impl

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCreateMPICUPMAsync(MPI_Comm comm, PetscInt n, PetscInt N, Vec *v) noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(v, 4);
  PetscCall(impl::VecMPI_CUPM<T>::CreateMPICUPM(comm, 0, n, N, v, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCreateMPICUPMWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  if (n && cpuarray) PetscAssertPointer(cpuarray, 5);
  PetscAssertPointer(v, 7);
  PetscCall(impl::VecMPI_CUPM<T>::CreateMPICUPMWithArrays(comm, bs, n, N, cpuarray, gpuarray, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCreateMPICUPMWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecCreateMPICUPMWithArrays<T>(comm, bs, n, N, nullptr, gpuarray, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace cupm

} // namespace vec

} // namespace Petsc

#if PetscDefined(HAVE_CUDA)
extern template class PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL ::Petsc::vec::cupm::impl::VecMPI_CUPM<::Petsc::device::cupm::DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
extern template class PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL ::Petsc::vec::cupm::impl::VecMPI_CUPM<::Petsc::device::cupm::DeviceType::HIP>;
#endif
