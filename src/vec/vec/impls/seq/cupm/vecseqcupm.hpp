#pragma once

#include <petsc/private/veccupmimpl.h>
#include <petsc/private/cpp/utility.hpp> // util::index_sequence

#include <../src/sys/objects/device/impls/cupm/kernels.hpp> // grid_stride_1D()
#include <../src/vec/vec/impls/dvecimpl.h>                  // Vec_Seq

namespace Petsc
{

namespace vec
{

namespace cupm
{

namespace impl
{

// ==========================================================================================
// VecSeq_CUPM
// ==========================================================================================

template <device::cupm::DeviceType T>
class VecSeq_CUPM : Vec_CUPMBase<T, VecSeq_CUPM<T>> {
public:
  PETSC_VEC_CUPM_BASE_CLASS_HEADER(base_type, T, VecSeq_CUPM<T>);

private:
  PETSC_NODISCARD static Vec_Seq          *VecIMPLCast_(Vec) noexcept;
  PETSC_NODISCARD static constexpr VecType VECIMPLCUPM_() noexcept;
  PETSC_NODISCARD static constexpr VecType VECIMPL_() noexcept;

  static PetscErrorCode VecDestroy_IMPL_(Vec) noexcept;
  static PetscErrorCode VecResetArray_IMPL_(Vec) noexcept;
  static PetscErrorCode VecPlaceArray_IMPL_(Vec, const PetscScalar *) noexcept;
  static PetscErrorCode VecCreate_IMPL_Private_(Vec, PetscBool *, PetscInt, PetscScalar *) noexcept;

  static PetscErrorCode MaybeIncrementEmptyLocalVec(Vec) noexcept;

  // common core for min and max
  template <typename TupleFuncT, typename UnaryFuncT>
  static PetscErrorCode MinMax_(TupleFuncT &&, UnaryFuncT &&, Vec, PetscInt *, PetscReal *) noexcept;
  // common core for pointwise binary and pointwise unary thrust functions
  template <typename BinaryFuncT>
  static PetscErrorCode PointwiseBinary_(BinaryFuncT &&, Vec, Vec, Vec, PetscDeviceContext = nullptr) noexcept;
  template <typename BinaryFuncT>
  static PetscErrorCode PointwiseBinaryDispatch_(PetscErrorCode (*)(Vec, Vec, Vec), BinaryFuncT &&, Vec, Vec, Vec, PetscDeviceContext = nullptr) noexcept;
  template <typename UnaryFuncT>
  static PetscErrorCode PointwiseUnary_(UnaryFuncT &&, Vec, Vec, PetscDeviceContext = nullptr) noexcept;
  // mdot dispatchers
  static PetscErrorCode MDot_(/* use complex = */ std::true_type, Vec, PetscInt, const Vec[], PetscScalar *, PetscDeviceContext) noexcept;
  static PetscErrorCode MDot_(/* use complex = */ std::false_type, Vec, PetscInt, const Vec[], PetscScalar *, PetscDeviceContext) noexcept;
  template <std::size_t... Idx>
  static PetscErrorCode MDot_kernel_dispatch_(PetscDeviceContext, cupmStream_t, const PetscScalar *, const Vec[], PetscInt, PetscScalar *, util::index_sequence<Idx...>) noexcept;
  template <int>
  static PetscErrorCode MDot_kernel_dispatch_(PetscDeviceContext, cupmStream_t, const PetscScalar *, const Vec[], PetscInt, PetscScalar *, PetscInt &) noexcept;
  template <std::size_t... Idx>
  static PetscErrorCode MAXPY_kernel_dispatch_(PetscDeviceContext, cupmStream_t, PetscScalar *, const PetscScalar *, const Vec *, PetscInt, util::index_sequence<Idx...>) noexcept;
  template <int>
  static PetscErrorCode MAXPY_kernel_dispatch_(PetscDeviceContext, cupmStream_t, PetscScalar *, const PetscScalar *, const Vec *, PetscInt, PetscInt &) noexcept;
  // common core for the various create routines
  static PetscErrorCode CreateSeqCUPM_(Vec, PetscDeviceContext, PetscScalar * /*host_ptr*/ = nullptr, PetscScalar * /*device_ptr*/ = nullptr) noexcept;

public:
  // callable directly via a bespoke function
  static PetscErrorCode CreateSeqCUPM(MPI_Comm, PetscInt, PetscInt, Vec *, PetscBool) noexcept;
  static PetscErrorCode CreateSeqCUPMWithBothArrays(MPI_Comm, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], Vec *) noexcept;

  static PetscErrorCode InitializeAsyncFunctions(Vec) noexcept;
  static PetscErrorCode ClearAsyncFunctions(Vec) noexcept;

  // callable indirectly via function pointers
  static PetscErrorCode Duplicate(Vec, Vec *) noexcept;
  static PetscErrorCode AYPX(Vec, PetscScalar, Vec) noexcept;
  static PetscErrorCode AYPXAsync(Vec, PetscScalar, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode AXPY(Vec, PetscScalar, Vec) noexcept;
  static PetscErrorCode AXPYAsync(Vec, PetscScalar, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode PointwiseDivide(Vec, Vec, Vec) noexcept;
  static PetscErrorCode PointwiseDivideAsync(Vec, Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode PointwiseMult(Vec, Vec, Vec) noexcept;
  static PetscErrorCode PointwiseMultAsync(Vec, Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode PointwiseMax(Vec, Vec, Vec) noexcept;
  static PetscErrorCode PointwiseMaxAsync(Vec, Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode PointwiseMaxAbs(Vec, Vec, Vec) noexcept;
  static PetscErrorCode PointwiseMaxAbsAsync(Vec, Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode PointwiseMin(Vec, Vec, Vec) noexcept;
  static PetscErrorCode PointwiseMinAsync(Vec, Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode Reciprocal(Vec) noexcept;
  static PetscErrorCode ReciprocalAsync(Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode Abs(Vec) noexcept;
  static PetscErrorCode AbsAsync(Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode SqrtAbs(Vec) noexcept;
  static PetscErrorCode SqrtAbsAsync(Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode Exp(Vec) noexcept;
  static PetscErrorCode ExpAsync(Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode Log(Vec) noexcept;
  static PetscErrorCode LogAsync(Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode WAXPY(Vec, PetscScalar, Vec, Vec) noexcept;
  static PetscErrorCode WAXPYAsync(Vec, PetscScalar, Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode MAXPY(Vec, PetscInt, const PetscScalar[], Vec *) noexcept;
  static PetscErrorCode MAXPYAsync(Vec, PetscInt, const PetscScalar[], Vec *, PetscDeviceContext) noexcept;
  static PetscErrorCode Dot(Vec, Vec, PetscScalar *) noexcept;
  static PetscErrorCode MDot(Vec, PetscInt, const Vec[], PetscScalar *) noexcept;
  static PetscErrorCode Set(Vec, PetscScalar) noexcept;
  static PetscErrorCode SetAsync(Vec, PetscScalar, PetscDeviceContext) noexcept;
  static PetscErrorCode Scale(Vec, PetscScalar) noexcept;
  static PetscErrorCode ScaleAsync(Vec, PetscScalar, PetscDeviceContext) noexcept;
  static PetscErrorCode TDot(Vec, Vec, PetscScalar *) noexcept;
  static PetscErrorCode Copy(Vec, Vec) noexcept;
  static PetscErrorCode CopyAsync(Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode Swap(Vec, Vec) noexcept;
  static PetscErrorCode SwapAsync(Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode AXPBY(Vec, PetscScalar, PetscScalar, Vec) noexcept;
  static PetscErrorCode AXPBYAsync(Vec, PetscScalar, PetscScalar, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode AXPBYPCZ(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec) noexcept;
  static PetscErrorCode AXPBYPCZAsync(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec, PetscDeviceContext) noexcept;
  static PetscErrorCode Norm(Vec, NormType, PetscReal *) noexcept;
  static PetscErrorCode ErrorWnorm(Vec, Vec, Vec, NormType, PetscReal, Vec, PetscReal, Vec, PetscReal, PetscReal *, PetscInt *, PetscReal *, PetscInt *, PetscReal *, PetscInt *) noexcept;
  static PetscErrorCode DotNorm2(Vec, Vec, PetscScalar *, PetscScalar *) noexcept;
  static PetscErrorCode Conjugate(Vec) noexcept;
  static PetscErrorCode ConjugateAsync(Vec, PetscDeviceContext) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetLocalVector(Vec, Vec) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreLocalVector(Vec, Vec) noexcept;
  static PetscErrorCode Max(Vec, PetscInt *, PetscReal *) noexcept;
  static PetscErrorCode Min(Vec, PetscInt *, PetscReal *) noexcept;
  static PetscErrorCode Sum(Vec, PetscScalar *) noexcept;
  static PetscErrorCode Shift(Vec, PetscScalar) noexcept;
  static PetscErrorCode ShiftAsync(Vec, PetscScalar, PetscDeviceContext) noexcept;
  static PetscErrorCode SetRandom(Vec, PetscRandom) noexcept;
  static PetscErrorCode BindToCPU(Vec, PetscBool) noexcept;
  static PetscErrorCode SetPreallocationCOO(Vec, PetscCount, const PetscInt[]) noexcept;
  static PetscErrorCode SetValuesCOO(Vec, const PetscScalar[], InsertMode) noexcept;
};

namespace kernels
{

template <typename F>
PETSC_DEVICE_INLINE_DECL void add_coo_values_impl(const PetscScalar *PETSC_RESTRICT vv, PetscCount n, const PetscCount *PETSC_RESTRICT jmap, const PetscCount *PETSC_RESTRICT perm, InsertMode imode, PetscScalar *PETSC_RESTRICT xv, F &&xvindex)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(n, [=](PetscCount i) {
    const auto  end = jmap[i + 1];
    const auto  idx = xvindex(i);
    PetscScalar sum = 0.0;

    for (auto k = jmap[i]; k < end; ++k) sum += vv[perm[k]];

    if (imode == INSERT_VALUES) {
      xv[idx] = sum;
    } else {
      xv[idx] += sum;
    }
  });
  return;
}

namespace
{
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wunused-function")
PETSC_KERNEL_DECL void add_coo_values(const PetscScalar *PETSC_RESTRICT v, PetscCount n, const PetscCount *PETSC_RESTRICT jmap1, const PetscCount *PETSC_RESTRICT perm1, InsertMode imode, PetscScalar *PETSC_RESTRICT xv)
{
  add_coo_values_impl(v, n, jmap1, perm1, imode, xv, [](PetscCount i) { return i; });
  return;
}
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()
} // namespace

} // namespace kernels

} // namespace impl

// ==========================================================================================
// VecSeq_CUPM - Implementations
// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCreateSeqCUPMAsync(MPI_Comm comm, PetscInt n, Vec *v) noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(v, 4);
  PetscCall(impl::VecSeq_CUPM<T>::CreateSeqCUPM(comm, 0, n, v, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCreateSeqCUPMWithArraysAsync(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v) noexcept
{
  PetscFunctionBegin;
  if (n && cpuarray) PetscAssertPointer(cpuarray, 4);
  PetscAssertPointer(v, 6);
  PetscCall(impl::VecSeq_CUPM<T>::CreateSeqCUPMWithBothArrays(comm, bs, n, cpuarray, gpuarray, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <PetscMemoryAccessMode mode, device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMGetArrayAsync_Private(Vec v, PetscScalar **a, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscAssertPointer(a, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(impl::VecSeq_CUPM<T>::template GetArray<PETSC_MEMTYPE_DEVICE, mode>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <PetscMemoryAccessMode mode, device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMRestoreArrayAsync_Private(Vec v, PetscScalar **a, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(impl::VecSeq_CUPM<T>::template RestoreArray<PETSC_MEMTYPE_DEVICE, mode>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMGetArrayAsync(Vec v, PetscScalar **a, PetscDeviceContext dctx = nullptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_READ_WRITE, T>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMRestoreArrayAsync(Vec v, PetscScalar **a, PetscDeviceContext dctx = nullptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_READ_WRITE, T>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMGetArrayReadAsync(Vec v, const PetscScalar **a, PetscDeviceContext dctx = nullptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_READ, T>(v, const_cast<PetscScalar **>(a), dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMRestoreArrayReadAsync(Vec v, const PetscScalar **a, PetscDeviceContext dctx = nullptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_READ, T>(v, const_cast<PetscScalar **>(a), dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMGetArrayWriteAsync(Vec v, PetscScalar **a, PetscDeviceContext dctx = nullptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_WRITE, T>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMRestoreArrayWriteAsync(Vec v, PetscScalar **a, PetscDeviceContext dctx = nullptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_WRITE, T>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMPlaceArrayAsync(Vec vin, const PetscScalar a[]) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin, VEC_CLASSID, 1);
  PetscCall(impl::VecSeq_CUPM<T>::template PlaceArray<PETSC_MEMTYPE_DEVICE>(vin, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMReplaceArrayAsync(Vec vin, const PetscScalar a[]) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin, VEC_CLASSID, 1);
  PetscCall(impl::VecSeq_CUPM<T>::template ReplaceArray<PETSC_MEMTYPE_DEVICE>(vin, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecCUPMResetArrayAsync(Vec vin) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin, VEC_CLASSID, 1);
  PetscCall(impl::VecSeq_CUPM<T>::template ResetArray<PETSC_MEMTYPE_DEVICE>(vin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace cupm

} // namespace vec

} // namespace Petsc

#if PetscDefined(HAVE_CUDA)
extern template class PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL ::Petsc::vec::cupm::impl::VecSeq_CUPM<::Petsc::device::cupm::DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
extern template class PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL ::Petsc::vec::cupm::impl::VecSeq_CUPM<::Petsc::device::cupm::DeviceType::HIP>;
#endif
