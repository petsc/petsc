#ifndef PETSCVECMPICUPM_HPP
#define PETSCVECMPICUPM_HPP

#if defined(__cplusplus)
  #include <petsc/private/veccupmimpl.h> /*I <petscvec.h> I*/
  #include <../src/vec/vec/impls/seq/cupm/vecseqcupm.hpp>
  #include <../src/vec/vec/impls/mpi/pvecimpl.h>
  #include <petsc/private/sfimpl.h> // for vec->localupdate (_p_VecScatter) in duplicate()

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
  PETSC_NODISCARD static PetscErrorCode    VecDestroy_IMPL_(Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode    VecResetArray_IMPL_(Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode    VecPlaceArray_IMPL_(Vec, const PetscScalar *) noexcept;
  PETSC_NODISCARD static PetscErrorCode    VecCreate_IMPL_Private_(Vec, PetscBool *, PetscInt, PetscScalar *) noexcept;

  PETSC_NODISCARD static PetscErrorCode creatempicupm_(Vec, PetscDeviceContext, PetscBool /*allocate_missing*/ = PETSC_TRUE, PetscInt /*nghost*/ = 0, PetscScalar * /*host_array*/ = nullptr, PetscScalar * /*device_array*/ = nullptr) noexcept;

public:
  // callable directly via a bespoke function
  PETSC_NODISCARD static PetscErrorCode creatempicupm(MPI_Comm, PetscInt, PetscInt, PetscInt, Vec *, PetscBool) noexcept;
  PETSC_NODISCARD static PetscErrorCode creatempicupmwitharrays(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], Vec *) noexcept;

  PETSC_NODISCARD static PetscErrorCode duplicate(Vec, Vec *) noexcept;
  PETSC_NODISCARD static PetscErrorCode bindtocpu(Vec, PetscBool) noexcept;
  PETSC_NODISCARD static PetscErrorCode norm(Vec, NormType, PetscReal *) noexcept;
  PETSC_NODISCARD static PetscErrorCode dot(Vec, Vec, PetscScalar *) noexcept;
  PETSC_NODISCARD static PetscErrorCode tdot(Vec, Vec, PetscScalar *) noexcept;
  PETSC_NODISCARD static PetscErrorCode mdot(Vec, PetscInt, const Vec[], PetscScalar *) noexcept;
  PETSC_NODISCARD static PetscErrorCode dotnorm2(Vec, Vec, PetscScalar *, PetscScalar *) noexcept;
  PETSC_NODISCARD static PetscErrorCode max(Vec, PetscInt *, PetscReal *) noexcept;
  PETSC_NODISCARD static PetscErrorCode min(Vec, PetscInt *, PetscReal *) noexcept;
  PETSC_NODISCARD static PetscErrorCode setpreallocationcoo(Vec, PetscCount, const PetscInt[]) noexcept;
  PETSC_NODISCARD static PetscErrorCode setvaluescoo(Vec, const PetscScalar[], InsertMode) noexcept;
};

template <device::cupm::DeviceType T>
inline Vec_MPI *VecMPI_CUPM<T>::VecIMPLCast_(Vec v) noexcept
{
  return static_cast<Vec_MPI *>(v->data);
}

template <device::cupm::DeviceType T>
inline constexpr VecType VecMPI_CUPM<T>::VECIMPLCUPM_() noexcept
{
  return VECMPICUPM();
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::VecDestroy_IMPL_(Vec v) noexcept
{
  return VecDestroy_MPI(v);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::VecResetArray_IMPL_(Vec v) noexcept
{
  return VecResetArray_MPI(v);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::VecPlaceArray_IMPL_(Vec v, const PetscScalar *a) noexcept
{
  return VecPlaceArray_MPI(v, a);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::VecCreate_IMPL_Private_(Vec v, PetscBool *alloc_missing, PetscInt nghost, PetscScalar *) noexcept
{
  PetscFunctionBegin;
  if (alloc_missing) *alloc_missing = PETSC_TRUE;
  // note host_array is always ignored, we never create it as part of the construction sequence
  // for VecMPI since we always want to either allocate it ourselves with pinned memory or set
  // it in Initialize_CUPMBase()
  PetscCall(VecCreate_MPI_Private(v, PETSC_FALSE, nghost, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::creatempicupm_(Vec v, PetscDeviceContext dctx, PetscBool allocate_missing, PetscInt nghost, PetscScalar *host_array, PetscScalar *device_array) noexcept
{
  PetscFunctionBegin;
  PetscCall(base_type::VecCreate_IMPL_Private(v, nullptr, nghost));
  PetscCall(Initialize_CUPMBase(v, allocate_missing, host_array, device_array, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ================================================================================== //
//                                                                                    //
//                                  public methods                                    //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                             constructors/destructors                               //

// VecCreateMPICUPM()
template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::creatempicupm(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, Vec *v, PetscBool call_set_type) noexcept
{
  PetscFunctionBegin;
  PetscCall(Create_CUPMBase(comm, bs, n, N, v, call_set_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// VecCreateMPICUPMWithArray[s]()
template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::creatempicupmwitharrays(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar host_array[], const PetscScalar device_array[], Vec *v) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  // do NOT call VecSetType(), otherwise ops->create() -> create() ->
  // creatempicupm_() is called!
  PetscCall(creatempicupm(comm, bs, n, N, v, PETSC_FALSE));
  PetscCall(creatempicupm_(*v, dctx, PETSC_FALSE, 0, PetscRemoveConstCast(host_array), PetscRemoveConstCast(device_array)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->duplicate
template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::duplicate(Vec v, Vec *y) noexcept
{
  const auto         vimpl  = VecIMPLCast(v);
  const auto         nghost = vimpl->nghost;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  // does not call VecSetType(), we set up the data structures ourselves
  PetscCall(Duplicate_CUPMBase(v, y, dctx, [=](Vec z) { return creatempicupm_(z, dctx, PETSC_FALSE, nghost); }));

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (const auto locrep = vimpl->localrep) {
    const auto   yimpl   = VecIMPLCast(*y);
    auto        &ylocrep = yimpl->localrep;
    PetscScalar *array;

    PetscCall(VecGetArray(*y, &array));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, std::abs(v->map->bs), v->map->n + nghost, array, &ylocrep));
    PetscCall(VecRestoreArray(*y, &array));
    PetscCall(PetscArraycpy(ylocrep->ops, locrep->ops, 1));
    if (auto &scatter = (yimpl->localupdate = vimpl->localupdate)) PetscCall(PetscObjectReference(PetscObjectCast(scatter)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->bintocpu
template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::bindtocpu(Vec v, PetscBool usehost) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(BindToCPU_CUPMBase(v, usehost, dctx));

  VecSetOp_CUPM(dot, VecDot_MPI, dot);
  VecSetOp_CUPM(mdot, VecMDot_MPI, mdot);
  VecSetOp_CUPM(norm, VecNorm_MPI, norm);
  VecSetOp_CUPM(tdot, VecTDot_MPI, tdot);
  VecSetOp_CUPM(resetarray, VecResetArray_MPI, base_type::template resetarray<PETSC_MEMTYPE_HOST>);
  VecSetOp_CUPM(placearray, VecPlaceArray_MPI, base_type::template placearray<PETSC_MEMTYPE_HOST>);
  VecSetOp_CUPM(max, VecMax_MPI, max);
  VecSetOp_CUPM(min, VecMin_MPI, min);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ================================================================================== //
//                                   compute methods                                  //

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::norm(Vec v, NormType type, PetscReal *z) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecNorm_MPI_Default(v, type, z, VecSeq_T::norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::dot(Vec x, Vec y, PetscScalar *z) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(x, y, z, VecSeq_T::dot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::tdot(Vec x, Vec y, PetscScalar *z) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(x, y, z, VecSeq_T::tdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::mdot(Vec x, PetscInt nv, const Vec y[], PetscScalar *z) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(x, nv, y, z, VecSeq_T::mdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::dotnorm2(Vec x, Vec y, PetscScalar *dp, PetscScalar *nm) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecDotNorm2_MPI_Default(x, y, dp, nm, VecSeq_T::dotnorm2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::max(Vec x, PetscInt *idx, PetscReal *z) noexcept
{
  const MPI_Op ops[] = {MPIU_MAXLOC, MPIU_MAX};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Default(x, idx, z, VecSeq_T::max, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::min(Vec x, PetscInt *idx, PetscReal *z) noexcept
{
  const MPI_Op ops[] = {MPIU_MINLOC, MPIU_MIN};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Default(x, idx, z, VecSeq_T::min, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::setpreallocationcoo(Vec x, PetscCount ncoo, const PetscInt coo_i[]) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecSetPreallocationCOO_MPI(x, ncoo, coo_i));
  // both of these must exist for this to work
  PetscCall(VecCUPMAllocateCheck_(x));
  {
    const auto vcu  = VecCUPMCast(x);
    const auto vmpi = VecIMPLCast(x);

    // clang-format off
    PetscCall(
      SetPreallocationCOO_CUPMBase(
        x, ncoo, coo_i,
        util::make_array(
          make_coo_pair(vcu->imap2_d, vmpi->imap2, vmpi->nnz2),
          make_coo_pair(vcu->jmap2_d, vmpi->jmap2, vmpi->nnz2 + 1),
          make_coo_pair(vcu->perm2_d, vmpi->perm2, vmpi->recvlen),
          make_coo_pair(vcu->Cperm_d, vmpi->Cperm, vmpi->sendlen)
        ),
        util::make_array(
          make_coo_pair(vcu->sendbuf_d, vmpi->sendbuf, vmpi->sendlen),
          make_coo_pair(vcu->recvbuf_d, vmpi->recvbuf, vmpi->recvlen)
        )
      )
    );
    // clang-format on
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace kernels
{

PETSC_KERNEL_DECL static void pack_coo_values(const PetscScalar *PETSC_RESTRICT vv, PetscCount nnz, const PetscCount *PETSC_RESTRICT perm, PetscScalar *PETSC_RESTRICT buf)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(nnz, [=](PetscCount i) { buf[i] = vv[perm[i]]; });
  return;
}

PETSC_KERNEL_DECL static void add_remote_coo_values(const PetscScalar *PETSC_RESTRICT vv, PetscCount nnz2, const PetscCount *PETSC_RESTRICT imap2, const PetscCount *PETSC_RESTRICT jmap2, const PetscCount *PETSC_RESTRICT perm2, PetscScalar *PETSC_RESTRICT xv)
{
  add_coo_values_impl(vv, nnz2, jmap2, perm2, ADD_VALUES, xv, [=](PetscCount i) { return imap2[i]; });
  return;
}

} // namespace kernels

template <device::cupm::DeviceType T>
inline PetscErrorCode VecMPI_CUPM<T>::setvaluescoo(Vec x, const PetscScalar v[], InsertMode imode) noexcept
{
  PetscDeviceContext dctx;
  PetscMemType       v_memtype;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscGetMemType(v, &v_memtype));
  {
    const auto vmpi      = VecIMPLCast(x);
    const auto vcu       = VecCUPMCast(x);
    const auto sf        = vmpi->coo_sf;
    const auto sendbuf_d = vcu->sendbuf_d;
    const auto recvbuf_d = vcu->recvbuf_d;
    const auto xv        = imode == INSERT_VALUES ? DeviceArrayWrite(dctx, x).data() : DeviceArrayReadWrite(dctx, x).data();
    auto       vv        = const_cast<PetscScalar *>(v);

    if (PetscMemTypeHost(v_memtype)) {
      const auto size = vmpi->coo_n;

      /* If user gave v[] in host, we might need to copy it to device if any */
      PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_CUPM(), size, &vv));
      PetscCall(PetscCUPMMemcpyAsync(vv, v, size, cupmMemcpyHostToDevice, stream));
    }

    /* Pack entries to be sent to remote */
    if (const auto sendlen = vmpi->sendlen) {
      PetscCall(PetscCUPMLaunchKernel1D(sendlen, 0, stream, kernels::pack_coo_values, vv, sendlen, vcu->Cperm_d, sendbuf_d));
      // need to sync up here since we are about to send this to petscsf
      // REVIEW ME: no we dont, sf just needs to learn to use PetscDeviceContext
      PetscCallCUPM(cupmStreamSynchronize(stream));
    }

    PetscCall(PetscSFReduceWithMemTypeBegin(sf, MPIU_SCALAR, PETSC_MEMTYPE_CUPM(), sendbuf_d, PETSC_MEMTYPE_CUPM(), recvbuf_d, MPI_REPLACE));

    if (const auto n = x->map->n) PetscCall(PetscCUPMLaunchKernel1D(n, 0, stream, kernels::add_coo_values, vv, n, vcu->jmap1_d, vcu->perm1_d, imode, xv));

    PetscCall(PetscSFReduceEnd(sf, MPIU_SCALAR, sendbuf_d, recvbuf_d, MPI_REPLACE));

    /* Add received remote entries */
    if (const auto nnz2 = vmpi->nnz2) PetscCall(PetscCUPMLaunchKernel1D(nnz2, 0, stream, kernels::add_remote_coo_values, recvbuf_d, nnz2, vcu->imap2_d, vcu->jmap2_d, vcu->perm2_d, xv));

    if (PetscMemTypeHost(v_memtype)) PetscCall(PetscDeviceFree(dctx, vv));
    PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace vec

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCVECMPICUPM_HPP
