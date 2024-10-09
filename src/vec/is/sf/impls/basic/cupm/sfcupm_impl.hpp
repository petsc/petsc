#pragma once

#include "sfcupm.hpp"
#include <../src/sys/objects/device/impls/cupm/kernels.hpp>
#include <petsc/private/cupmatomics.hpp>

namespace Petsc
{

namespace sf
{

namespace cupm
{

namespace kernels
{

/* Map a thread id to an index in root/leaf space through a series of 3D subdomains. See PetscSFPackOpt. */
PETSC_NODISCARD static PETSC_DEVICE_INLINE_DECL PetscInt MapTidToIndex(const PetscInt *opt, PetscInt tid) noexcept
{
  PetscInt        i, j, k, m, n, r;
  const PetscInt *offset, *start, *dx, *dy, *X, *Y;

  n      = opt[0];
  offset = opt + 1;
  start  = opt + n + 2;
  dx     = opt + 2 * n + 2;
  dy     = opt + 3 * n + 2;
  X      = opt + 5 * n + 2;
  Y      = opt + 6 * n + 2;
  for (r = 0; r < n; r++) {
    if (tid < offset[r + 1]) break;
  }
  m = (tid - offset[r]);
  k = m / (dx[r] * dy[r]);
  j = (m - k * dx[r] * dy[r]) / dx[r];
  i = m - k * dx[r] * dy[r] - j * dx[r];

  return start[r] + k * X[r] * Y[r] + j * X[r] + i;
}

/*====================================================================================*/
/*  Templated CUPM kernels for pack/unpack. The Op can be regular or atomic           */
/*====================================================================================*/

/* Suppose user calls PetscSFReduce(sf,unit,...) and <unit> is an MPI data type made of 16 PetscReals, then
   <Type> is PetscReal, which is the primitive type we operate on.
   <bs>   is 16, which says <unit> contains 16 primitive types.
   <BS>   is 8, which is the maximal SIMD width we will try to vectorize operations on <unit>.
   <EQ>   is 0, which is (bs == BS ? 1 : 0)

  If instead, <unit> has 8 PetscReals, then bs=8, BS=8, EQ=1, rendering MBS below to a compile time constant.
  For the common case in VecScatter, bs=1, BS=1, EQ=1, MBS=1, the inner for-loops below will be totally unrolled.
*/
template <class Type, PetscInt BS, PetscInt EQ>
PETSC_KERNEL_DECL static void d_Pack(PetscInt bs, PetscInt count, PetscInt start, const PetscInt *opt, const PetscInt *idx, const Type *data, Type *buf)
{
  const PetscInt M   = (EQ) ? 1 : bs / BS; /* If EQ, then M=1 enables compiler's const-propagation */
  const PetscInt MBS = M * BS;             /* MBS=bs. We turn MBS into a compile-time const when EQ=1. */

  ::Petsc::device::cupm::kernels::util::grid_stride_1D(count, [&](PetscInt tid) {
    PetscInt t = (opt ? MapTidToIndex(opt, tid) : (idx ? idx[tid] : start + tid)) * MBS;
    PetscInt s = tid * MBS;
    for (PetscInt i = 0; i < MBS; i++) buf[s + i] = data[t + i];
  });
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
PETSC_KERNEL_DECL static void d_UnpackAndOp(PetscInt bs, PetscInt count, PetscInt start, const PetscInt *opt, const PetscInt *idx, Type *data, const Type *buf)
{
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  ::Petsc::device::cupm::kernels::util::grid_stride_1D(count, [&](PetscInt tid) {
    PetscInt t = (opt ? MapTidToIndex(opt, tid) : (idx ? idx[tid] : start + tid)) * MBS;
    PetscInt s = tid * MBS;
    for (PetscInt i = 0; i < MBS; i++) op(data[t + i], buf[s + i]);
  });
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
PETSC_KERNEL_DECL static void d_FetchAndOp(PetscInt bs, PetscInt count, PetscInt rootstart, const PetscInt *rootopt, const PetscInt *rootidx, Type *rootdata, Type *leafbuf)
{
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  ::Petsc::device::cupm::kernels::util::grid_stride_1D(count, [&](PetscInt tid) {
    PetscInt r = (rootopt ? MapTidToIndex(rootopt, tid) : (rootidx ? rootidx[tid] : rootstart + tid)) * MBS;
    PetscInt l = tid * MBS;
    for (PetscInt i = 0; i < MBS; i++) leafbuf[l + i] = op(rootdata[r + i], leafbuf[l + i]);
  });
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
PETSC_KERNEL_DECL static void d_ScatterAndOp(PetscInt bs, PetscInt count, PetscInt srcx, PetscInt srcy, PetscInt srcX, PetscInt srcY, PetscInt srcStart, const PetscInt *srcIdx, const Type *src, PetscInt dstx, PetscInt dsty, PetscInt dstX, PetscInt dstY, PetscInt dstStart, const PetscInt *dstIdx, Type *dst)
{
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  ::Petsc::device::cupm::kernels::util::grid_stride_1D(count, [&](PetscInt tid) {
    PetscInt s, t;

    if (!srcIdx) { /* src is either contiguous or 3D */
      PetscInt k = tid / (srcx * srcy);
      PetscInt j = (tid - k * srcx * srcy) / srcx;
      PetscInt i = tid - k * srcx * srcy - j * srcx;

      s = srcStart + k * srcX * srcY + j * srcX + i;
    } else {
      s = srcIdx[tid];
    }

    if (!dstIdx) { /* dst is either contiguous or 3D */
      PetscInt k = tid / (dstx * dsty);
      PetscInt j = (tid - k * dstx * dsty) / dstx;
      PetscInt i = tid - k * dstx * dsty - j * dstx;

      t = dstStart + k * dstX * dstY + j * dstX + i;
    } else {
      t = dstIdx[tid];
    }

    s *= MBS;
    t *= MBS;
    for (PetscInt i = 0; i < MBS; i++) op(dst[t + i], src[s + i]);
  });
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
PETSC_KERNEL_DECL static void d_FetchAndOpLocal(PetscInt bs, PetscInt count, PetscInt rootstart, const PetscInt *rootopt, const PetscInt *rootidx, Type *rootdata, PetscInt leafstart, const PetscInt *leafopt, const PetscInt *leafidx, const Type *leafdata, Type *leafupdate)
{
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  ::Petsc::device::cupm::kernels::util::grid_stride_1D(count, [&](PetscInt tid) {
    PetscInt r = (rootopt ? MapTidToIndex(rootopt, tid) : (rootidx ? rootidx[tid] : rootstart + tid)) * MBS;
    PetscInt l = (leafopt ? MapTidToIndex(leafopt, tid) : (leafidx ? leafidx[tid] : leafstart + tid)) * MBS;
    for (PetscInt i = 0; i < MBS; i++) leafupdate[l + i] = op(rootdata[r + i], leafdata[l + i]);
  });
}

/*====================================================================================*/
/*                             Regular operations on device                           */
/*====================================================================================*/
template <typename Type>
struct Insert {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = y;
    return old;
  }
};
template <typename Type>
struct Add {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x += y;
    return old;
  }
};
template <typename Type>
struct Mult {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x *= y;
    return old;
  }
};
template <typename Type>
struct Min {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = PetscMin(x, y);
    return old;
  }
};
template <typename Type>
struct Max {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = PetscMax(x, y);
    return old;
  }
};
template <typename Type>
struct LAND {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x && y;
    return old;
  }
};
template <typename Type>
struct LOR {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x || y;
    return old;
  }
};
template <typename Type>
struct LXOR {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = !x != !y;
    return old;
  }
};
template <typename Type>
struct BAND {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x & y;
    return old;
  }
};
template <typename Type>
struct BOR {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x | y;
    return old;
  }
};
template <typename Type>
struct BXOR {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x ^ y;
    return old;
  }
};
template <typename Type>
struct Minloc {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    if (y.a < x.a) x = y;
    else if (y.a == x.a) x.b = min(x.b, y.b);
    return old;
  }
};
template <typename Type>
struct Maxloc {
  PETSC_DEVICE_DECL Type operator()(Type &x, Type y) const
  {
    Type old = x;
    if (y.a > x.a) x = y;
    else if (y.a == x.a) x.b = min(x.b, y.b); /* See MPI MAXLOC */
    return old;
  }
};

} // namespace kernels

namespace impl
{

/*====================================================================================*/
/*  Wrapper functions of cupm kernels. Function pointers are stored in 'link'         */
/*====================================================================================*/
template <device::cupm::DeviceType T>
template <typename Type, PetscInt BS, PetscInt EQ>
inline PetscErrorCode SfInterface<T>::Pack(PetscSFLink link, PetscInt count, PetscInt start, PetscSFPackOpt opt, const PetscInt *idx, const void *data, void *buf) noexcept
{
  const PetscInt *iarray = opt ? opt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscDefined(USING_NVCC) && !opt && !idx) { /* It is a 'CUDA data to nvshmem buf' memory copy */
    PetscCallCUPM(cupmMemcpyAsync(buf, (char *)data + start * link->unitbytes, count * link->unitbytes, cupmMemcpyDeviceToDevice, link->stream));
  } else {
    PetscCall(PetscCUPMLaunchKernel1D(count, 0, link->stream, kernels::d_Pack<Type, BS, EQ>, link->bs, count, start, iarray, idx, (const Type *)data, (Type *)buf));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <typename Type, class Op, PetscInt BS, PetscInt EQ>
inline PetscErrorCode SfInterface<T>::UnpackAndOp(PetscSFLink link, PetscInt count, PetscInt start, PetscSFPackOpt opt, const PetscInt *idx, void *data, const void *buf) noexcept
{
  const PetscInt *iarray = opt ? opt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscDefined(USING_NVCC) && std::is_same<Op, kernels::Insert<Type>>::value && !opt && !idx) { /* It is a 'nvshmem buf to CUDA data' memory copy */
    PetscCallCUPM(cupmMemcpyAsync((char *)data + start * link->unitbytes, buf, count * link->unitbytes, cupmMemcpyDeviceToDevice, link->stream));
  } else {
    PetscCall(PetscCUPMLaunchKernel1D(count, 0, link->stream, kernels::d_UnpackAndOp<Type, Op, BS, EQ>, link->bs, count, start, iarray, idx, (Type *)data, (const Type *)buf));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <typename Type, class Op, PetscInt BS, PetscInt EQ>
inline PetscErrorCode SfInterface<T>::FetchAndOp(PetscSFLink link, PetscInt count, PetscInt start, PetscSFPackOpt opt, const PetscInt *idx, void *data, void *buf) noexcept
{
  const PetscInt *iarray = opt ? opt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMLaunchKernel1D(count, 0, link->stream, kernels::d_FetchAndOp<Type, Op, BS, EQ>, link->bs, count, start, iarray, idx, (Type *)data, (const Type *)buf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <typename Type, class Op, PetscInt BS, PetscInt EQ>
inline PetscErrorCode SfInterface<T>::ScatterAndOp(PetscSFLink link, PetscInt count, PetscInt srcStart, PetscSFPackOpt srcOpt, const PetscInt *srcIdx, const void *src, PetscInt dstStart, PetscSFPackOpt dstOpt, const PetscInt *dstIdx, void *dst) noexcept
{
  PetscInt nthreads = 256;
  PetscInt nblocks  = (count + nthreads - 1) / nthreads;
  PetscInt srcx = 0, srcy = 0, srcX = 0, srcY = 0, dstx = 0, dsty = 0, dstX = 0, dstY = 0;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(PETSC_SUCCESS);
  nblocks = PetscMin(nblocks, link->maxResidentThreadsPerGPU / nthreads);

  /* The 3D shape of source subdomain may be different than that of the destination, which makes it difficult to use 3D grid and block */
  if (srcOpt) {
    srcx     = srcOpt->dx[0];
    srcy     = srcOpt->dy[0];
    srcX     = srcOpt->X[0];
    srcY     = srcOpt->Y[0];
    srcStart = srcOpt->start[0];
    srcIdx   = NULL;
  } else if (!srcIdx) {
    srcx = srcX = count;
    srcy = srcY = 1;
  }

  if (dstOpt) {
    dstx     = dstOpt->dx[0];
    dsty     = dstOpt->dy[0];
    dstX     = dstOpt->X[0];
    dstY     = dstOpt->Y[0];
    dstStart = dstOpt->start[0];
    dstIdx   = NULL;
  } else if (!dstIdx) {
    dstx = dstX = count;
    dsty = dstY = 1;
  }

  PetscCall(PetscCUPMLaunchKernel1D(count, 0, link->stream, kernels::d_ScatterAndOp<Type, Op, BS, EQ>, link->bs, count, srcx, srcy, srcX, srcY, srcStart, srcIdx, (const Type *)src, dstx, dsty, dstX, dstY, dstStart, dstIdx, (Type *)dst));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
/* Specialization for Insert since we may use cupmMemcpyAsync */
template <typename Type, PetscInt BS, PetscInt EQ>
inline PetscErrorCode SfInterface<T>::ScatterAndInsert(PetscSFLink link, PetscInt count, PetscInt srcStart, PetscSFPackOpt srcOpt, const PetscInt *srcIdx, const void *src, PetscInt dstStart, PetscSFPackOpt dstOpt, const PetscInt *dstIdx, void *dst) noexcept
{
  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(PETSC_SUCCESS);
  /*src and dst are contiguous */
  if ((!srcOpt && !srcIdx) && (!dstOpt && !dstIdx) && src != dst) {
    PetscCallCUPM(cupmMemcpyAsync((Type *)dst + dstStart * link->bs, (const Type *)src + srcStart * link->bs, count * link->unitbytes, cupmMemcpyDeviceToDevice, link->stream));
  } else {
    PetscCall(ScatterAndOp<Type, kernels::Insert<Type>, BS, EQ>(link, count, srcStart, srcOpt, srcIdx, src, dstStart, dstOpt, dstIdx, dst));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <typename Type, class Op, PetscInt BS, PetscInt EQ>
inline PetscErrorCode SfInterface<T>::FetchAndOpLocal(PetscSFLink link, PetscInt count, PetscInt rootstart, PetscSFPackOpt rootopt, const PetscInt *rootidx, void *rootdata, PetscInt leafstart, PetscSFPackOpt leafopt, const PetscInt *leafidx, const void *leafdata, void *leafupdate) noexcept
{
  const PetscInt *rarray = rootopt ? rootopt->array : NULL;
  const PetscInt *larray = leafopt ? leafopt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMLaunchKernel1D(count, 0, link->stream, kernels::d_FetchAndOpLocal<Type, Op, BS, EQ>, link->bs, count, rootstart, rarray, rootidx, (Type *)rootdata, leafstart, larray, leafidx, (const Type *)leafdata, (Type *)leafupdate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*====================================================================================*/
/*  Init various types and instantiate pack/unpack function pointers                  */
/*====================================================================================*/
template <device::cupm::DeviceType T>
template <typename Type, PetscInt BS, PetscInt EQ>
inline void SfInterface<T>::PackInit_RealType(PetscSFLink link) noexcept
{
  /* Pack/unpack for remote communication */
  link->d_Pack            = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert = UnpackAndOp<Type, kernels::Insert<Type>, BS, EQ>;
  link->d_UnpackAndAdd    = UnpackAndOp<Type, kernels::Add<Type>, BS, EQ>;
  link->d_UnpackAndMult   = UnpackAndOp<Type, kernels::Mult<Type>, BS, EQ>;
  link->d_UnpackAndMin    = UnpackAndOp<Type, kernels::Min<Type>, BS, EQ>;
  link->d_UnpackAndMax    = UnpackAndOp<Type, kernels::Max<Type>, BS, EQ>;
  link->d_FetchAndAdd     = FetchAndOp<Type, kernels::Add<Type>, BS, EQ>;

  /* Scatter for local communication */
  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>; /* Has special optimizations */
  link->d_ScatterAndAdd    = ScatterAndOp<Type, kernels::Add<Type>, BS, EQ>;
  link->d_ScatterAndMult   = ScatterAndOp<Type, kernels::Mult<Type>, BS, EQ>;
  link->d_ScatterAndMin    = ScatterAndOp<Type, kernels::Min<Type>, BS, EQ>;
  link->d_ScatterAndMax    = ScatterAndOp<Type, kernels::Max<Type>, BS, EQ>;
  link->d_FetchAndAddLocal = FetchAndOpLocal<Type, kernels::Add<Type>, BS, EQ>;

  /* Atomic versions when there are data-race possibilities */
  link->da_UnpackAndInsert = UnpackAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_UnpackAndAdd    = UnpackAndOp<Type, AtomicAdd<Type>, BS, EQ>;
  link->da_UnpackAndMult   = UnpackAndOp<Type, AtomicMult<Type>, BS, EQ>;
  link->da_UnpackAndMin    = UnpackAndOp<Type, AtomicMin<Type>, BS, EQ>;
  link->da_UnpackAndMax    = UnpackAndOp<Type, AtomicMax<Type>, BS, EQ>;
  link->da_FetchAndAdd     = FetchAndOp<Type, AtomicAdd<Type>, BS, EQ>;

  link->da_ScatterAndInsert = ScatterAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_ScatterAndAdd    = ScatterAndOp<Type, AtomicAdd<Type>, BS, EQ>;
  link->da_ScatterAndMult   = ScatterAndOp<Type, AtomicMult<Type>, BS, EQ>;
  link->da_ScatterAndMin    = ScatterAndOp<Type, AtomicMin<Type>, BS, EQ>;
  link->da_ScatterAndMax    = ScatterAndOp<Type, AtomicMax<Type>, BS, EQ>;
  link->da_FetchAndAddLocal = FetchAndOpLocal<Type, AtomicAdd<Type>, BS, EQ>;
}

/* Have this templated class to specialize for char integers */
template <device::cupm::DeviceType T>
template <typename Type, PetscInt BS, PetscInt EQ, PetscInt size /*sizeof(Type)*/>
struct SfInterface<T>::PackInit_IntegerType_Atomic {
  static inline void Init(PetscSFLink link) noexcept
  {
    link->da_UnpackAndInsert = UnpackAndOp<Type, AtomicInsert<Type>, BS, EQ>;
    link->da_UnpackAndAdd    = UnpackAndOp<Type, AtomicAdd<Type>, BS, EQ>;
    link->da_UnpackAndMult   = UnpackAndOp<Type, AtomicMult<Type>, BS, EQ>;
    link->da_UnpackAndMin    = UnpackAndOp<Type, AtomicMin<Type>, BS, EQ>;
    link->da_UnpackAndMax    = UnpackAndOp<Type, AtomicMax<Type>, BS, EQ>;
    link->da_UnpackAndLAND   = UnpackAndOp<Type, AtomicLAND<Type>, BS, EQ>;
    link->da_UnpackAndLOR    = UnpackAndOp<Type, AtomicLOR<Type>, BS, EQ>;
    link->da_UnpackAndLXOR   = UnpackAndOp<Type, AtomicLXOR<Type>, BS, EQ>;
    link->da_UnpackAndBAND   = UnpackAndOp<Type, AtomicBAND<Type>, BS, EQ>;
    link->da_UnpackAndBOR    = UnpackAndOp<Type, AtomicBOR<Type>, BS, EQ>;
    link->da_UnpackAndBXOR   = UnpackAndOp<Type, AtomicBXOR<Type>, BS, EQ>;
    link->da_FetchAndAdd     = FetchAndOp<Type, AtomicAdd<Type>, BS, EQ>;

    link->da_ScatterAndInsert = ScatterAndOp<Type, AtomicInsert<Type>, BS, EQ>;
    link->da_ScatterAndAdd    = ScatterAndOp<Type, AtomicAdd<Type>, BS, EQ>;
    link->da_ScatterAndMult   = ScatterAndOp<Type, AtomicMult<Type>, BS, EQ>;
    link->da_ScatterAndMin    = ScatterAndOp<Type, AtomicMin<Type>, BS, EQ>;
    link->da_ScatterAndMax    = ScatterAndOp<Type, AtomicMax<Type>, BS, EQ>;
    link->da_ScatterAndLAND   = ScatterAndOp<Type, AtomicLAND<Type>, BS, EQ>;
    link->da_ScatterAndLOR    = ScatterAndOp<Type, AtomicLOR<Type>, BS, EQ>;
    link->da_ScatterAndLXOR   = ScatterAndOp<Type, AtomicLXOR<Type>, BS, EQ>;
    link->da_ScatterAndBAND   = ScatterAndOp<Type, AtomicBAND<Type>, BS, EQ>;
    link->da_ScatterAndBOR    = ScatterAndOp<Type, AtomicBOR<Type>, BS, EQ>;
    link->da_ScatterAndBXOR   = ScatterAndOp<Type, AtomicBXOR<Type>, BS, EQ>;
    link->da_FetchAndAddLocal = FetchAndOpLocal<Type, AtomicAdd<Type>, BS, EQ>;
  }
};

/* CUDA does not support atomics on chars. It is TBD in PETSc. */
template <device::cupm::DeviceType T>
template <typename Type, PetscInt BS, PetscInt EQ>
struct SfInterface<T>::PackInit_IntegerType_Atomic<Type, BS, EQ, 1> {
  static inline void Init(PetscSFLink) { /* Nothing to leave function pointers NULL */ }
};

template <device::cupm::DeviceType T>
template <typename Type, PetscInt BS, PetscInt EQ>
inline void SfInterface<T>::PackInit_IntegerType(PetscSFLink link) noexcept
{
  link->d_Pack            = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert = UnpackAndOp<Type, kernels::Insert<Type>, BS, EQ>;
  link->d_UnpackAndAdd    = UnpackAndOp<Type, kernels::Add<Type>, BS, EQ>;
  link->d_UnpackAndMult   = UnpackAndOp<Type, kernels::Mult<Type>, BS, EQ>;
  link->d_UnpackAndMin    = UnpackAndOp<Type, kernels::Min<Type>, BS, EQ>;
  link->d_UnpackAndMax    = UnpackAndOp<Type, kernels::Max<Type>, BS, EQ>;
  link->d_UnpackAndLAND   = UnpackAndOp<Type, kernels::LAND<Type>, BS, EQ>;
  link->d_UnpackAndLOR    = UnpackAndOp<Type, kernels::LOR<Type>, BS, EQ>;
  link->d_UnpackAndLXOR   = UnpackAndOp<Type, kernels::LXOR<Type>, BS, EQ>;
  link->d_UnpackAndBAND   = UnpackAndOp<Type, kernels::BAND<Type>, BS, EQ>;
  link->d_UnpackAndBOR    = UnpackAndOp<Type, kernels::BOR<Type>, BS, EQ>;
  link->d_UnpackAndBXOR   = UnpackAndOp<Type, kernels::BXOR<Type>, BS, EQ>;
  link->d_FetchAndAdd     = FetchAndOp<Type, kernels::Add<Type>, BS, EQ>;

  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>;
  link->d_ScatterAndAdd    = ScatterAndOp<Type, kernels::Add<Type>, BS, EQ>;
  link->d_ScatterAndMult   = ScatterAndOp<Type, kernels::Mult<Type>, BS, EQ>;
  link->d_ScatterAndMin    = ScatterAndOp<Type, kernels::Min<Type>, BS, EQ>;
  link->d_ScatterAndMax    = ScatterAndOp<Type, kernels::Max<Type>, BS, EQ>;
  link->d_ScatterAndLAND   = ScatterAndOp<Type, kernels::LAND<Type>, BS, EQ>;
  link->d_ScatterAndLOR    = ScatterAndOp<Type, kernels::LOR<Type>, BS, EQ>;
  link->d_ScatterAndLXOR   = ScatterAndOp<Type, kernels::LXOR<Type>, BS, EQ>;
  link->d_ScatterAndBAND   = ScatterAndOp<Type, kernels::BAND<Type>, BS, EQ>;
  link->d_ScatterAndBOR    = ScatterAndOp<Type, kernels::BOR<Type>, BS, EQ>;
  link->d_ScatterAndBXOR   = ScatterAndOp<Type, kernels::BXOR<Type>, BS, EQ>;
  link->d_FetchAndAddLocal = FetchAndOpLocal<Type, kernels::Add<Type>, BS, EQ>;
  PackInit_IntegerType_Atomic<Type, BS, EQ, sizeof(Type)>::Init(link);
}

#if defined(PETSC_HAVE_COMPLEX)
template <device::cupm::DeviceType T>
template <typename Type, PetscInt BS, PetscInt EQ>
inline void SfInterface<T>::PackInit_ComplexType(PetscSFLink link) noexcept
{
  link->d_Pack            = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert = UnpackAndOp<Type, kernels::Insert<Type>, BS, EQ>;
  link->d_UnpackAndAdd    = UnpackAndOp<Type, kernels::Add<Type>, BS, EQ>;
  link->d_UnpackAndMult   = UnpackAndOp<Type, kernels::Mult<Type>, BS, EQ>;
  link->d_FetchAndAdd     = FetchAndOp<Type, kernels::Add<Type>, BS, EQ>;

  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>;
  link->d_ScatterAndAdd    = ScatterAndOp<Type, kernels::Add<Type>, BS, EQ>;
  link->d_ScatterAndMult   = ScatterAndOp<Type, kernels::Mult<Type>, BS, EQ>;
  link->d_FetchAndAddLocal = FetchAndOpLocal<Type, kernels::Add<Type>, BS, EQ>;

  link->da_UnpackAndInsert = UnpackAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_UnpackAndAdd    = UnpackAndOp<Type, AtomicAdd<Type>, BS, EQ>;
  link->da_UnpackAndMult   = NULL; /* Not implemented yet */
  link->da_FetchAndAdd     = NULL; /* Return value of atomicAdd on complex is not atomic */

  link->da_ScatterAndInsert = ScatterAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_ScatterAndAdd    = ScatterAndOp<Type, AtomicAdd<Type>, BS, EQ>;
}
#endif

typedef signed char   SignedChar;
typedef unsigned char UnsignedChar;
typedef struct {
  int a;
  int b;
} PairInt;
typedef struct {
  PetscInt a;
  PetscInt b;
} PairPetscInt;

template <device::cupm::DeviceType T>
template <typename Type>
inline void SfInterface<T>::PackInit_PairType(PetscSFLink link) noexcept
{
  link->d_Pack            = Pack<Type, 1, 1>;
  link->d_UnpackAndInsert = UnpackAndOp<Type, kernels::Insert<Type>, 1, 1>;
  link->d_UnpackAndMaxloc = UnpackAndOp<Type, kernels::Maxloc<Type>, 1, 1>;
  link->d_UnpackAndMinloc = UnpackAndOp<Type, kernels::Minloc<Type>, 1, 1>;

  link->d_ScatterAndInsert = ScatterAndOp<Type, kernels::Insert<Type>, 1, 1>;
  link->d_ScatterAndMaxloc = ScatterAndOp<Type, kernels::Maxloc<Type>, 1, 1>;
  link->d_ScatterAndMinloc = ScatterAndOp<Type, kernels::Minloc<Type>, 1, 1>;
  /* Atomics for pair types are not implemented yet */
}

template <device::cupm::DeviceType T>
template <typename Type, PetscInt BS, PetscInt EQ>
inline void SfInterface<T>::PackInit_DumbType(PetscSFLink link) noexcept
{
  link->d_Pack             = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert  = UnpackAndOp<Type, kernels::Insert<Type>, BS, EQ>;
  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>;
  /* Atomics for dumb types are not implemented yet */
}

/* Some device-specific utilities */
template <device::cupm::DeviceType T>
inline PetscErrorCode SfInterface<T>::LinkSyncDevice(PetscSFLink) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmDeviceSynchronize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode SfInterface<T>::LinkSyncStream(PetscSFLink link) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmStreamSynchronize(link->stream));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode SfInterface<T>::LinkMemcpy(PetscSFLink link, PetscMemType dstmtype, void *dst, PetscMemType srcmtype, const void *src, size_t n) noexcept
{
  PetscFunctionBegin;
  cupmMemcpyKind_t kinds[2][2] = {
    {cupmMemcpyHostToHost,   cupmMemcpyHostToDevice  },
    {cupmMemcpyDeviceToHost, cupmMemcpyDeviceToDevice}
  };

  if (n) {
    if (PetscMemTypeHost(dstmtype) && PetscMemTypeHost(srcmtype)) { /* Separate HostToHost so that pure-cpu code won't call cupm runtime */
      PetscCall(PetscMemcpy(dst, src, n));
    } else {
      int stype = PetscMemTypeDevice(srcmtype) ? 1 : 0;
      int dtype = PetscMemTypeDevice(dstmtype) ? 1 : 0;
      PetscCallCUPM(cupmMemcpyAsync(dst, src, n, kinds[stype][dtype], link->stream));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode SfInterface<T>::Malloc(PetscMemType mtype, size_t size, void **ptr) noexcept
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscCall(PetscMalloc(size, ptr));
  else if (PetscMemTypeDevice(mtype)) {
    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
    PetscCallCUPM(cupmMalloc(ptr, size));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Wrong PetscMemType %d", (int)mtype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode SfInterface<T>::Free(PetscMemType mtype, void *ptr) noexcept
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscCall(PetscFree(ptr));
  else if (PetscMemTypeDevice(mtype)) PetscCallCUPM(cupmFree(ptr));
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Wrong PetscMemType %d", (int)mtype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Destructor when the link uses MPI for communication on CUPM device */
template <device::cupm::DeviceType T>
inline PetscErrorCode SfInterface<T>::LinkDestroy_MPI(PetscSF, PetscSFLink link) noexcept
{
  PetscFunctionBegin;
  for (int i = PETSCSF_LOCAL; i <= PETSCSF_REMOTE; i++) {
    PetscCallCUPM(cupmFree(link->rootbuf_alloc[i][PETSC_MEMTYPE_DEVICE]));
    PetscCallCUPM(cupmFree(link->leafbuf_alloc[i][PETSC_MEMTYPE_DEVICE]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*====================================================================================*/
/*                Main driver to init MPI datatype on device                          */
/*====================================================================================*/

/* Some fields of link are initialized by PetscSFPackSetUp_Host. This routine only does what needed on device */
template <device::cupm::DeviceType T>
inline PetscErrorCode SfInterface<T>::LinkSetUp(PetscSF sf, PetscSFLink link, MPI_Datatype unit) noexcept
{
  PetscInt  nSignedChar = 0, nUnsignedChar = 0, nInt = 0, nPetscInt = 0, nPetscReal = 0;
  PetscBool is2Int, is2PetscInt;
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt nPetscComplex = 0;
#endif

  PetscFunctionBegin;
  if (link->deviceinited) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPI_SIGNED_CHAR, &nSignedChar));
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPI_UNSIGNED_CHAR, &nUnsignedChar));
  /* MPI_CHAR is treated below as a dumb type that does not support reduction according to MPI standard */
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPI_INT, &nInt));
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPIU_INT, &nPetscInt));
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPIU_REAL, &nPetscReal));
#if defined(PETSC_HAVE_COMPLEX)
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPIU_COMPLEX, &nPetscComplex));
#endif
  PetscCall(MPIPetsc_Type_compare(unit, MPI_2INT, &is2Int));
  PetscCall(MPIPetsc_Type_compare(unit, MPIU_2INT, &is2PetscInt));

  if (is2Int) {
    PackInit_PairType<PairInt>(link);
  } else if (is2PetscInt) { /* TODO: when is2PetscInt and nPetscInt=2, we don't know which path to take. The two paths support different ops. */
    PackInit_PairType<PairPetscInt>(link);
  } else if (nPetscReal) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nPetscReal == 8) PackInit_RealType<PetscReal, 8, 1>(link);
    else if (nPetscReal % 8 == 0) PackInit_RealType<PetscReal, 8, 0>(link);
    else if (nPetscReal == 4) PackInit_RealType<PetscReal, 4, 1>(link);
    else if (nPetscReal % 4 == 0) PackInit_RealType<PetscReal, 4, 0>(link);
    else if (nPetscReal == 2) PackInit_RealType<PetscReal, 2, 1>(link);
    else if (nPetscReal % 2 == 0) PackInit_RealType<PetscReal, 2, 0>(link);
    else if (nPetscReal == 1) PackInit_RealType<PetscReal, 1, 1>(link);
    else if (nPetscReal % 1 == 0)
#endif
      PackInit_RealType<PetscReal, 1, 0>(link);
  } else if (nPetscInt && sizeof(PetscInt) == sizeof(llint)) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nPetscInt == 8) PackInit_IntegerType<llint, 8, 1>(link);
    else if (nPetscInt % 8 == 0) PackInit_IntegerType<llint, 8, 0>(link);
    else if (nPetscInt == 4) PackInit_IntegerType<llint, 4, 1>(link);
    else if (nPetscInt % 4 == 0) PackInit_IntegerType<llint, 4, 0>(link);
    else if (nPetscInt == 2) PackInit_IntegerType<llint, 2, 1>(link);
    else if (nPetscInt % 2 == 0) PackInit_IntegerType<llint, 2, 0>(link);
    else if (nPetscInt == 1) PackInit_IntegerType<llint, 1, 1>(link);
    else if (nPetscInt % 1 == 0)
#endif
      PackInit_IntegerType<llint, 1, 0>(link);
  } else if (nInt) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nInt == 8) PackInit_IntegerType<int, 8, 1>(link);
    else if (nInt % 8 == 0) PackInit_IntegerType<int, 8, 0>(link);
    else if (nInt == 4) PackInit_IntegerType<int, 4, 1>(link);
    else if (nInt % 4 == 0) PackInit_IntegerType<int, 4, 0>(link);
    else if (nInt == 2) PackInit_IntegerType<int, 2, 1>(link);
    else if (nInt % 2 == 0) PackInit_IntegerType<int, 2, 0>(link);
    else if (nInt == 1) PackInit_IntegerType<int, 1, 1>(link);
    else if (nInt % 1 == 0)
#endif
      PackInit_IntegerType<int, 1, 0>(link);
  } else if (nSignedChar) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nSignedChar == 8) PackInit_IntegerType<SignedChar, 8, 1>(link);
    else if (nSignedChar % 8 == 0) PackInit_IntegerType<SignedChar, 8, 0>(link);
    else if (nSignedChar == 4) PackInit_IntegerType<SignedChar, 4, 1>(link);
    else if (nSignedChar % 4 == 0) PackInit_IntegerType<SignedChar, 4, 0>(link);
    else if (nSignedChar == 2) PackInit_IntegerType<SignedChar, 2, 1>(link);
    else if (nSignedChar % 2 == 0) PackInit_IntegerType<SignedChar, 2, 0>(link);
    else if (nSignedChar == 1) PackInit_IntegerType<SignedChar, 1, 1>(link);
    else if (nSignedChar % 1 == 0)
#endif
      PackInit_IntegerType<SignedChar, 1, 0>(link);
  } else if (nUnsignedChar) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nUnsignedChar == 8) PackInit_IntegerType<UnsignedChar, 8, 1>(link);
    else if (nUnsignedChar % 8 == 0) PackInit_IntegerType<UnsignedChar, 8, 0>(link);
    else if (nUnsignedChar == 4) PackInit_IntegerType<UnsignedChar, 4, 1>(link);
    else if (nUnsignedChar % 4 == 0) PackInit_IntegerType<UnsignedChar, 4, 0>(link);
    else if (nUnsignedChar == 2) PackInit_IntegerType<UnsignedChar, 2, 1>(link);
    else if (nUnsignedChar % 2 == 0) PackInit_IntegerType<UnsignedChar, 2, 0>(link);
    else if (nUnsignedChar == 1) PackInit_IntegerType<UnsignedChar, 1, 1>(link);
    else if (nUnsignedChar % 1 == 0)
#endif
      PackInit_IntegerType<UnsignedChar, 1, 0>(link);
#if defined(PETSC_HAVE_COMPLEX)
  } else if (nPetscComplex) {
  #if !defined(PETSC_HAVE_DEVICE)
    if (nPetscComplex == 8) PackInit_ComplexType<PetscComplex, 8, 1>(link);
    else if (nPetscComplex % 8 == 0) PackInit_ComplexType<PetscComplex, 8, 0>(link);
    else if (nPetscComplex == 4) PackInit_ComplexType<PetscComplex, 4, 1>(link);
    else if (nPetscComplex % 4 == 0) PackInit_ComplexType<PetscComplex, 4, 0>(link);
    else if (nPetscComplex == 2) PackInit_ComplexType<PetscComplex, 2, 1>(link);
    else if (nPetscComplex % 2 == 0) PackInit_ComplexType<PetscComplex, 2, 0>(link);
    else if (nPetscComplex == 1) PackInit_ComplexType<PetscComplex, 1, 1>(link);
    else if (nPetscComplex % 1 == 0)
  #endif
      PackInit_ComplexType<PetscComplex, 1, 0>(link);
#endif
  } else {
    MPI_Aint lb, nbyte;

    PetscCallMPI(MPI_Type_get_extent(unit, &lb, &nbyte));
    PetscCheck(lb == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "Datatype with nonzero lower bound %ld", (long)lb);
    if (nbyte % sizeof(int)) { /* If the type size is not multiple of int */
#if !defined(PETSC_HAVE_DEVICE)
      if (nbyte == 4) PackInit_DumbType<char, 4, 1>(link);
      else if (nbyte % 4 == 0) PackInit_DumbType<char, 4, 0>(link);
      else if (nbyte == 2) PackInit_DumbType<char, 2, 1>(link);
      else if (nbyte % 2 == 0) PackInit_DumbType<char, 2, 0>(link);
      else if (nbyte == 1) PackInit_DumbType<char, 1, 1>(link);
      else if (nbyte % 1 == 0)
#endif
        PackInit_DumbType<char, 1, 0>(link);
    } else {
      PetscCall(PetscIntCast(nbyte / sizeof(int), &nInt));
#if !defined(PETSC_HAVE_DEVICE)
      if (nInt == 8) PackInit_DumbType<int, 8, 1>(link);
      else if (nInt % 8 == 0) PackInit_DumbType<int, 8, 0>(link);
      else if (nInt == 4) PackInit_DumbType<int, 4, 1>(link);
      else if (nInt % 4 == 0) PackInit_DumbType<int, 4, 0>(link);
      else if (nInt == 2) PackInit_DumbType<int, 2, 1>(link);
      else if (nInt % 2 == 0) PackInit_DumbType<int, 2, 0>(link);
      else if (nInt == 1) PackInit_DumbType<int, 1, 1>(link);
      else if (nInt % 1 == 0)
#endif
        PackInit_DumbType<int, 1, 0>(link);
    }
  }

  if (!sf->maxResidentThreadsPerGPU) { /* Not initialized */
    int              device;
    cupmDeviceProp_t props;

    PetscCallCUPM(cupmGetDevice(&device));
    PetscCallCUPM(cupmGetDeviceProperties(&props, device));
    sf->maxResidentThreadsPerGPU = props.maxThreadsPerMultiProcessor * props.multiProcessorCount;
  }
  link->maxResidentThreadsPerGPU = sf->maxResidentThreadsPerGPU;

  {
    cupmStream_t      *stream;
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUPM()));
    PetscCall(PetscDeviceContextGetStreamHandle(dctx, (void **)&stream));
    link->stream = *stream;
  }
  link->Destroy      = LinkDestroy_MPI;
  link->SyncDevice   = LinkSyncDevice;
  link->SyncStream   = LinkSyncStream;
  link->Memcpy       = LinkMemcpy;
  link->deviceinited = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace sf

} // namespace Petsc
