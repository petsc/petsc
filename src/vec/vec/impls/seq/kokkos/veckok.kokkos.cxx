/*
   Implements the sequential Kokkos vectors.
*/
#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>

#include <petsc/private/sfimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscmath.h>
#include <petscviewer.h>
#include <KokkosBlas.hpp>
#include <Kokkos_Functional.hpp>

#include <../src/vec/vec/impls/dvecimpl.h> /* for VecCreate_Seq_Private */
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>

template <class MemorySpace>
static PetscErrorCode VecGetKokkosView_Private(Vec v, PetscScalarKokkosViewType<MemorySpace> *kv, PetscBool overwrite)
{
  Vec_Kokkos *veckok   = static_cast<Vec_Kokkos *>(v->spptr);
  using ExecutionSpace = typename PetscScalarKokkosViewType<MemorySpace>::traits::device_type;

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  if (!overwrite) { /* If overwrite=true, no need to sync the space, since caller will overwrite the data */
    PetscCallCXX(veckok->v_dual.sync<ExecutionSpace>());
  }
  PetscCallCXX(*kv = veckok->v_dual.view<ExecutionSpace>());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class MemorySpace>
static PetscErrorCode VecRestoreKokkosView_Private(Vec v, PetscScalarKokkosViewType<MemorySpace> *kv, PetscBool overwrite)
{
  Vec_Kokkos *veckok   = static_cast<Vec_Kokkos *>(v->spptr);
  using ExecutionSpace = typename PetscScalarKokkosViewType<MemorySpace>::traits::device_type;

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  if (overwrite) PetscCallCXX(veckok->v_dual.clear_sync_state()); /* If overwrite=true, clear the old sync state since user forced an overwrite */
  PetscCallCXX(veckok->v_dual.modify<ExecutionSpace>());
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class MemorySpace>
PetscErrorCode VecGetKokkosView(Vec v, ConstPetscScalarKokkosViewType<MemorySpace> *kv)
{
  Vec_Kokkos *veckok   = static_cast<Vec_Kokkos *>(v->spptr);
  using ExecutionSpace = typename PetscScalarKokkosViewType<MemorySpace>::traits::device_type;

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  PetscCallCXX(veckok->v_dual.sync<ExecutionSpace>());
  PetscCallCXX(*kv = veckok->v_dual.view<ExecutionSpace>());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Function template explicit instantiation */
template PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView(Vec, ConstPetscScalarKokkosView *);
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView(Vec v, PetscScalarKokkosView *kv)
{
  return VecGetKokkosView_Private(v, kv, PETSC_FALSE);
}
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosView(Vec v, PetscScalarKokkosView *kv)
{
  return VecRestoreKokkosView_Private(v, kv, PETSC_FALSE);
}
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosViewWrite(Vec v, PetscScalarKokkosView *kv)
{
  return VecGetKokkosView_Private(v, kv, PETSC_TRUE);
}
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosViewWrite(Vec v, PetscScalarKokkosView *kv)
{
  return VecRestoreKokkosView_Private(v, kv, PETSC_TRUE);
}

#if !defined(KOKKOS_ENABLE_UNIFIED_MEMORY) /* Get host views if the default memory space is not host space */
template PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView(Vec, ConstPetscScalarKokkosViewHost *);
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView(Vec v, PetscScalarKokkosViewHost *kv)
{
  return VecGetKokkosView_Private(v, kv, PETSC_FALSE);
}
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosView(Vec v, PetscScalarKokkosViewHost *kv)
{
  return VecRestoreKokkosView_Private(v, kv, PETSC_FALSE);
}
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosViewWrite(Vec v, PetscScalarKokkosViewHost *kv)
{
  return VecGetKokkosView_Private(v, kv, PETSC_TRUE);
}
template <>
PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosViewWrite(Vec v, PetscScalarKokkosViewHost *kv)
{
  return VecRestoreKokkosView_Private(v, kv, PETSC_TRUE);
}
#endif

PetscErrorCode VecSetRandom_SeqKokkos(Vec xin, PetscRandom r)
{
  const PetscInt n = xin->map->n;
  PetscScalar   *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayWrite(xin, &xx)); /* TODO: generate randoms directly on device */
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscRandomGetValue(r, &xx[i]));
  PetscCall(VecRestoreArrayWrite(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* x = |x| */
PetscErrorCode VecAbs_SeqKokkos(Vec xin)
{
  PetscScalarKokkosView xv;
  auto                  exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCallCXX(KokkosBlas::abs(exec, xv, xv));
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* x = 1/x */
PetscErrorCode VecReciprocal_SeqKokkos(Vec xin)
{
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCallCXX(Kokkos::parallel_for(
    "VecReciprocal", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n), KOKKOS_LAMBDA(const PetscInt &i) {
      if (xv(i) != (PetscScalar)0.0) xv(i) = (PetscScalar)1.0 / xv(i);
    }));
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMin_SeqKokkos(Vec xin, PetscInt *p, PetscReal *val)
{
  ConstPetscScalarKokkosView                           xv;
  Kokkos::MinFirstLoc<PetscReal, PetscInt>::value_type result;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCallCXX(Kokkos::parallel_reduce(
    "VecMin", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n),
    KOKKOS_LAMBDA(const PetscInt &i, Kokkos::MinFirstLoc<PetscReal, PetscInt>::value_type &lupdate) {
      if (PetscRealPart(xv(i)) < lupdate.val) {
        lupdate.val = PetscRealPart(xv(i));
        lupdate.loc = i;
      }
    },
    Kokkos::MinFirstLoc<PetscReal, PetscInt>(result)));
  *val = result.val;
  if (p) *p = result.loc;
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMax_SeqKokkos(Vec xin, PetscInt *p, PetscReal *val)
{
  ConstPetscScalarKokkosView                           xv;
  Kokkos::MaxFirstLoc<PetscReal, PetscInt>::value_type result;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCallCXX(Kokkos::parallel_reduce(
    "VecMax", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n),
    KOKKOS_LAMBDA(const PetscInt &i, Kokkos::MaxFirstLoc<PetscReal, PetscInt>::value_type &lupdate) {
      if (PetscRealPart(xv(i)) > lupdate.val) {
        lupdate.val = PetscRealPart(xv(i));
        lupdate.loc = i;
      }
    },
    Kokkos::MaxFirstLoc<PetscReal, PetscInt>(result)));
  *val = result.val;
  if (p) *p = result.loc;
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSum_SeqKokkos(Vec xin, PetscScalar *sum)
{
  ConstPetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCallCXX(*sum = KokkosBlas::sum(PetscGetKokkosExecutionSpace(), xv));
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecShift_SeqKokkos(Vec xin, PetscScalar shift)
{
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCallCXX(Kokkos::parallel_for("VecShift", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n), KOKKOS_LAMBDA(const PetscInt &i) { xv(i) += shift; }); PetscCall(VecRestoreKokkosView(xin, &xv)));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(xin->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = alpha x + y */
PetscErrorCode VecAXPY_SeqKokkos(Vec yin, PetscScalar alpha, Vec xin)
{
  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(PETSC_SUCCESS);
  if (yin == xin) {
    PetscCall(VecScale_SeqKokkos(yin, alpha + 1));
  } else {
    PetscBool xiskok, yiskok;

    PetscCall(PetscObjectTypeCompareAny((PetscObject)xin, &xiskok, VECSEQKOKKOS, VECMPIKOKKOS, ""));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)yin, &yiskok, VECSEQKOKKOS, VECMPIKOKKOS, ""));
    if (xiskok && yiskok) {
      PetscScalarKokkosView      yv;
      ConstPetscScalarKokkosView xv;

      PetscCall(PetscLogGpuTimeBegin());
      PetscCall(VecGetKokkosView(xin, &xv));
      PetscCall(VecGetKokkosView(yin, &yv));
      PetscCallCXX(KokkosBlas::axpy(PetscGetKokkosExecutionSpace(), alpha, xv, yv));
      PetscCall(VecRestoreKokkosView(xin, &xv));
      PetscCall(VecRestoreKokkosView(yin, &yv));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(PetscLogGpuFlops(2.0 * yin->map->n));
    } else {
      PetscCall(VecAXPY_Seq(yin, alpha, xin));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = x + beta y */
PetscErrorCode VecAYPX_SeqKokkos(Vec yin, PetscScalar beta, Vec xin)
{
  PetscFunctionBegin;
  /* One needs to define KOKKOSBLAS_OPTIMIZATION_LEVEL_AXPBY > 2 to have optimizations for cases alpha/beta = 0,+/-1 */
  PetscCall(VecAXPBY_SeqKokkos(yin, 1.0, beta, xin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = y^T x */
PetscErrorCode VecTDot_SeqKokkos(Vec xin, Vec yin, PetscScalar *z)
{
  ConstPetscScalarKokkosView xv, yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCall(VecGetKokkosView(yin, &yv));
  // Kokkos always overwrites z, so no need to init it
  PetscCallCXX(Kokkos::parallel_reduce("VecTDot", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n), KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &update) { update += yv(i) * xv(i); }, *z));
  PetscCall(VecRestoreKokkosView(yin, &yv));
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  if (xin->map->n > 0) PetscCall(PetscLogGpuFlops(2.0 * xin->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct TransposeDotTag { };
struct ConjugateDotTag { };

template <PetscInt ValueCount>
struct MDotFunctor {
  static_assert(ValueCount >= 1 && ValueCount <= 8, "ValueCount must be in [1, 8]");
  /* Note the C++ notation for an array typedef */
  // noted, thanks
  typedef PetscScalar                           value_type[];
  typedef ConstPetscScalarKokkosView::size_type size_type;

  /* Tell Kokkos the result array's number of entries. This must be a public value in the functor */
  static constexpr size_type value_count = ValueCount;
  ConstPetscScalarKokkosView xv, yv[8];

  MDotFunctor(ConstPetscScalarKokkosView &xv, ConstPetscScalarKokkosView &yv0, ConstPetscScalarKokkosView &yv1, ConstPetscScalarKokkosView &yv2, ConstPetscScalarKokkosView &yv3, ConstPetscScalarKokkosView &yv4, ConstPetscScalarKokkosView &yv5, ConstPetscScalarKokkosView &yv6, ConstPetscScalarKokkosView &yv7) :
    xv(xv)
  {
    yv[0] = yv0;
    yv[1] = yv1;
    yv[2] = yv2;
    yv[3] = yv3;
    yv[4] = yv4;
    yv[5] = yv5;
    yv[6] = yv6;
    yv[7] = yv7;
  }

  KOKKOS_INLINE_FUNCTION void operator()(TransposeDotTag, const size_type i, value_type sum) const
  {
    PetscScalar xval = xv(i);
    for (size_type j = 0; j < value_count; ++j) sum[j] += yv[j](i) * xval;
  }

  KOKKOS_INLINE_FUNCTION void operator()(ConjugateDotTag, const size_type i, value_type sum) const
  {
    PetscScalar xval = xv(i);
    for (size_type j = 0; j < value_count; ++j) sum[j] += PetscConj(yv[j](i)) * xval;
  }

  // Per https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html#requirements
  // "when specifying a tag in the policy, the functor's potential init/join/final member functions must also be tagged"
  // So we have this kind of duplicated code.
  KOKKOS_INLINE_FUNCTION void join(TransposeDotTag, value_type dst, const value_type src) const { join(dst, src); }
  KOKKOS_INLINE_FUNCTION void join(ConjugateDotTag, value_type dst, const value_type src) const { join(dst, src); }

  KOKKOS_INLINE_FUNCTION void init(TransposeDotTag, value_type sum) const { init(sum); }
  KOKKOS_INLINE_FUNCTION void init(ConjugateDotTag, value_type sum) const { init(sum); }

  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const
  {
    for (size_type j = 0; j < value_count; j++) dst[j] += src[j];
  }

  KOKKOS_INLINE_FUNCTION void init(value_type sum) const
  {
    for (size_type j = 0; j < value_count; j++) sum[j] = 0.0;
  }
};

template <class WorkTag>
PetscErrorCode VecMultiDot_Private(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)
{
  PetscInt                   i, j, cur = 0, ngroup = nv / 8, rem = nv % 8, N = xin->map->n;
  ConstPetscScalarKokkosView xv, yv[8];
  PetscScalarKokkosViewHost  zv(z, nv);
  auto                       exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCall(VecGetKokkosView(xin, &xv));
  for (i = 0; i < ngroup; i++) { /* 8 y's per group */
    for (j = 0; j < 8; j++) PetscCall(VecGetKokkosView(yin[cur + j], &yv[j]));
    MDotFunctor<8> mdot(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]); /* Hope Kokkos make it asynchronous */
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCXX(Kokkos::parallel_reduce(Kokkos::RangePolicy<WorkTag>(exec, 0, N), mdot, Kokkos::subview(zv, Kokkos::pair<PetscInt, PetscInt>(cur, cur + 8))));
    PetscCall(PetscLogGpuTimeEnd());
    for (j = 0; j < 8; j++) PetscCall(VecRestoreKokkosView(yin[cur + j], &yv[j]));
    cur += 8;
  }

  if (rem) { /* The remaining */
    for (j = 0; j < rem; j++) PetscCall(VecGetKokkosView(yin[cur + j], &yv[j]));
    Kokkos::RangePolicy<WorkTag> policy(exec, 0, N);
    auto                         results = Kokkos::subview(zv, Kokkos::pair<PetscInt, PetscInt>(cur, cur + rem));
    // clang-format off
    PetscCall(PetscLogGpuTimeBegin());
    switch (rem) {
    case 1: PetscCallCXX(Kokkos::parallel_reduce(policy, MDotFunctor<1>(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]), results)); break;
    case 2: PetscCallCXX(Kokkos::parallel_reduce(policy, MDotFunctor<2>(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]), results)); break;
    case 3: PetscCallCXX(Kokkos::parallel_reduce(policy, MDotFunctor<3>(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]), results)); break;
    case 4: PetscCallCXX(Kokkos::parallel_reduce(policy, MDotFunctor<4>(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]), results)); break;
    case 5: PetscCallCXX(Kokkos::parallel_reduce(policy, MDotFunctor<5>(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]), results)); break;
    case 6: PetscCallCXX(Kokkos::parallel_reduce(policy, MDotFunctor<6>(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]), results)); break;
    case 7: PetscCallCXX(Kokkos::parallel_reduce(policy, MDotFunctor<7>(xv, yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]), results)); break;
    }
    PetscCall(PetscLogGpuTimeEnd());
    // clang-format on
    for (j = 0; j < rem; j++) PetscCall(VecRestoreKokkosView(yin[cur + j], &yv[j]));
  }
  PetscCall(VecRestoreKokkosView(xin, &xv));
  exec.fence(); /* If reduce is async, then we need this fence to make sure z is ready for use on host */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMultiDot_Verbose(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)
{
  PetscInt                   ngroup = nv / 8, rem = nv % 8, N = xin->map->n;
  ConstPetscScalarKokkosView xv, y0, y1, y2, y3, y4, y5, y6, y7;
  PetscScalar               *zp = z;
  const Vec                 *yp = yin;
  Kokkos::RangePolicy<>      policy(PetscGetKokkosExecutionSpace(), 0, N);

  // clang-format off
  PetscFunctionBegin;
  PetscCall(VecGetKokkosView(xin, &xv));
  for (PetscInt k = 0; k < ngroup; k++) { // 8 y's per group
    PetscCall(VecGetKokkosView(yp[0], &y0));
    PetscCall(VecGetKokkosView(yp[1], &y1));
    PetscCall(VecGetKokkosView(yp[2], &y2));
    PetscCall(VecGetKokkosView(yp[3], &y3));
    PetscCall(VecGetKokkosView(yp[4], &y4));
    PetscCall(VecGetKokkosView(yp[5], &y5));
    PetscCall(VecGetKokkosView(yp[6], &y6));
    PetscCall(VecGetKokkosView(yp[7], &y7));
    PetscCall(PetscLogGpuTimeBegin()); // only for GPU kernel execution
    Kokkos::parallel_reduce(
      "VecMDot8", policy,
      KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0, PetscScalar &lsum1, PetscScalar &lsum2, PetscScalar &lsum3, PetscScalar &lsum4, PetscScalar &lsum5, PetscScalar &lsum6, PetscScalar &lsum7) {
        lsum0 += xv(i) * PetscConj(y0(i)); lsum1 += xv(i) * PetscConj(y1(i)); lsum2 += xv(i) * PetscConj(y2(i)); lsum3 += xv(i) * PetscConj(y3(i));
        lsum4 += xv(i) * PetscConj(y4(i)); lsum5 += xv(i) * PetscConj(y5(i)); lsum6 += xv(i) * PetscConj(y6(i)); lsum7 += xv(i) * PetscConj(y7(i));
      }, zp[0], zp[1], zp[2], zp[3], zp[4], zp[5], zp[6], zp[7]);
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuToCpu(8 * sizeof(PetscScalar))); // for copying to z[] on host
    PetscCall(VecRestoreKokkosView(yp[0], &y0));
    PetscCall(VecRestoreKokkosView(yp[1], &y1));
    PetscCall(VecRestoreKokkosView(yp[2], &y2));
    PetscCall(VecRestoreKokkosView(yp[3], &y3));
    PetscCall(VecRestoreKokkosView(yp[4], &y4));
    PetscCall(VecRestoreKokkosView(yp[5], &y5));
    PetscCall(VecRestoreKokkosView(yp[6], &y6));
    PetscCall(VecRestoreKokkosView(yp[7], &y7));
    yp += 8;
    zp += 8;
  }

  if (rem) { /* The remaining */
    if (rem > 0) PetscCall(VecGetKokkosView(yp[0], &y0));
    if (rem > 1) PetscCall(VecGetKokkosView(yp[1], &y1));
    if (rem > 2) PetscCall(VecGetKokkosView(yp[2], &y2));
    if (rem > 3) PetscCall(VecGetKokkosView(yp[3], &y3));
    if (rem > 4) PetscCall(VecGetKokkosView(yp[4], &y4));
    if (rem > 5) PetscCall(VecGetKokkosView(yp[5], &y5));
    if (rem > 6) PetscCall(VecGetKokkosView(yp[6], &y6));
    PetscCall(PetscLogGpuTimeBegin());
    switch (rem) {
    case 7:
      Kokkos::parallel_reduce(
        "VecMDot7", policy,
        KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0, PetscScalar &lsum1, PetscScalar &lsum2, PetscScalar &lsum3, PetscScalar &lsum4, PetscScalar &lsum5, PetscScalar &lsum6) {
        lsum0 += xv(i) * PetscConj(y0(i)); lsum1 += xv(i) * PetscConj(y1(i)); lsum2 += xv(i) * PetscConj(y2(i)); lsum3 += xv(i) * PetscConj(y3(i));
        lsum4 += xv(i) * PetscConj(y4(i)); lsum5 += xv(i) * PetscConj(y5(i)); lsum6 += xv(i) * PetscConj(y6(i));
      }, zp[0], zp[1], zp[2], zp[3], zp[4], zp[5], zp[6]);
      break;
    case 6:
      Kokkos::parallel_reduce(
        "VecMDot6", policy,
        KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0, PetscScalar &lsum1, PetscScalar &lsum2, PetscScalar &lsum3, PetscScalar &lsum4, PetscScalar &lsum5) {
        lsum0 += xv(i) * PetscConj(y0(i)); lsum1 += xv(i) * PetscConj(y1(i)); lsum2 += xv(i) * PetscConj(y2(i)); lsum3 += xv(i) * PetscConj(y3(i));
        lsum4 += xv(i) * PetscConj(y4(i)); lsum5 += xv(i) * PetscConj(y5(i));
      }, zp[0], zp[1], zp[2], zp[3], zp[4], zp[5]);
      break;
    case 5:
      Kokkos::parallel_reduce(
        "VecMDot5", policy,
        KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0, PetscScalar &lsum1, PetscScalar &lsum2, PetscScalar &lsum3, PetscScalar &lsum4) {
        lsum0 += xv(i) * PetscConj(y0(i)); lsum1 += xv(i) * PetscConj(y1(i)); lsum2 += xv(i) * PetscConj(y2(i)); lsum3 += xv(i) * PetscConj(y3(i));
        lsum4 += xv(i) * PetscConj(y4(i));
      }, zp[0], zp[1], zp[2], zp[3], zp[4]);
      break;
    case 4:
      Kokkos::parallel_reduce(
        "VecMDot4", policy,
        KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0, PetscScalar &lsum1, PetscScalar &lsum2, PetscScalar &lsum3) {
        lsum0 += xv(i) * PetscConj(y0(i)); lsum1 += xv(i) * PetscConj(y1(i)); lsum2 += xv(i) * PetscConj(y2(i)); lsum3 += xv(i) * PetscConj(y3(i));
      }, zp[0], zp[1], zp[2], zp[3]);
      break;
    case 3:
      Kokkos::parallel_reduce(
        "VecMDot3", policy,
        KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0, PetscScalar &lsum1, PetscScalar &lsum2) {
        lsum0 += xv(i) * PetscConj(y0(i)); lsum1 += xv(i) * PetscConj(y1(i)); lsum2 += xv(i) * PetscConj(y2(i));
      }, zp[0], zp[1], zp[2]);
      break;
    case 2:
      Kokkos::parallel_reduce(
        "VecMDot2", policy,
        KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0, PetscScalar &lsum1) {
        lsum0 += xv(i) * PetscConj(y0(i)); lsum1 += xv(i) * PetscConj(y1(i));
      }, zp[0], zp[1]);
      break;
    case 1:
      Kokkos::parallel_reduce(
        "VecMDot1", policy,
        KOKKOS_LAMBDA(const PetscInt &i, PetscScalar &lsum0) {
        lsum0 += xv(i) * PetscConj(y0(i));
      }, zp[0]);
      break;
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuToCpu(rem * sizeof(PetscScalar))); // for copying to z[] on host
    if (rem > 0) PetscCall(VecRestoreKokkosView(yp[0], &y0));
    if (rem > 1) PetscCall(VecRestoreKokkosView(yp[1], &y1));
    if (rem > 2) PetscCall(VecRestoreKokkosView(yp[2], &y2));
    if (rem > 3) PetscCall(VecRestoreKokkosView(yp[3], &y3));
    if (rem > 4) PetscCall(VecRestoreKokkosView(yp[4], &y4));
    if (rem > 5) PetscCall(VecRestoreKokkosView(yp[5], &y5));
    if (rem > 6) PetscCall(VecRestoreKokkosView(yp[6], &y6));
  }
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
  // clang-format on
}

/* z[i] = (x,y_i) = y_i^H x */
PetscErrorCode VecMDot_SeqKokkos(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)
{
  PetscFunctionBegin;
  // With no good reason, VecMultiDot_Private() performs much worse than VecMultiDot_Verbose() with HIP,
  // but they are on par with CUDA. Kokkos team is investigating this problem.
#if 0
  PetscCall(VecMultiDot_Private<ConjugateDotTag>(xin, nv, yin, z));
#else
  PetscCall(VecMultiDot_Verbose(xin, nv, yin, z));
#endif
  PetscCall(PetscLogGpuFlops(PetscMax(nv * (2.0 * xin->map->n - 1), 0.0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z[i] = (x,y_i) = y_i^T x */
PetscErrorCode VecMTDot_SeqKokkos(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMultiDot_Private<TransposeDotTag>(xin, nv, yin, z));
  PetscCall(PetscLogGpuFlops(PetscMax(nv * (2.0 * xin->map->n - 1), 0.0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// z[i] = (x,y_i) = y_i^H x OR y_i^T x
static PetscErrorCode VecMultiDot_SeqKokkos_GEMV(PetscBool conjugate, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z_h)
{
  PetscInt                   i, j, nfail;
  ConstPetscScalarKokkosView xv, yfirst, ynext;
  const PetscScalar         *yarray;
  PetscBool                  stop  = PETSC_FALSE;
  PetscScalar               *z_d   = nullptr;
  const char                *trans = conjugate ? "C" : "T";
  PetscInt64                 lda   = 0;
  PetscInt                   m, n = xin->map->n;

  PetscFunctionBegin;
  PetscCall(VecGetKokkosView(xin, &xv));
#if defined(KOKKOS_ENABLE_UNIFIED_MEMORY)
  z_d = z_h;
#endif
  i = nfail = 0;
  while (i < nv) {
    // search a sequence of vectors with a fixed stride
    stop = PETSC_FALSE;
    PetscCall(VecGetKokkosView(yin[i], &yfirst));
    yarray = yfirst.data();
    for (j = i + 1; j < nv; j++) {
      PetscCall(VecGetKokkosView(yin[j], &ynext));
      if (j == i + 1) {
        lda = ynext.data() - yarray;                       // arbitrary ptrdiff could be very large
        if (lda < 0 || lda - n > 64) stop = PETSC_TRUE;    // avoid using arbitrary lda; 64 bytes are a big enough alignment in VecDuplicateVecs
      } else if (lda * (j - i) != ynext.data() - yarray) { // not in the same stride? if so, stop searching
        stop = PETSC_TRUE;
      }
      PetscCall(VecRestoreKokkosView(yin[j], &ynext));
      if (stop) break;
    }
    PetscCall(VecRestoreKokkosView(yin[i], &yfirst));

    // found m vectors yin[i..j) with a stride lda at address yarray
    m = j - i;
    if (m > 1) {
      if (!z_d) {
        if (nv > PetscScalarPoolSize) { // rare case
          PetscScalarPoolSize = nv;
          PetscCallCXX(PetscScalarPool = static_cast<PetscScalar *>(Kokkos::kokkos_realloc(PetscScalarPool, PetscScalarPoolSize)));
        }
        z_d = PetscScalarPool;
      }
      const auto &A  = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft>(yarray, lda, m);
      const auto &Y  = Kokkos::subview(A, std::pair<PetscInt, PetscInt>(0, n), Kokkos::ALL);
      auto        zv = PetscScalarKokkosDualView(PetscScalarKokkosView(z_d + i, m), PetscScalarKokkosViewHost(z_h + i, m));
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), trans, 1.0, Y, xv, 0.0, zv.view_device()));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCallCXX(zv.modify_device());
      PetscCall(KokkosDualViewSyncHost(zv, PetscGetKokkosExecutionSpace()));
      PetscCall(PetscLogGpuFlops(PetscMax(m * (2.0 * n - 1), 0.0)));
    } else {
      // we only allow falling back on VecDot once, to avoid doing VecMultiDot via individual VecDots
      if (nfail++ == 0) {
        if (conjugate) PetscCall(VecDot_SeqKokkos(xin, yin[i], z_h + i));
        else PetscCall(VecTDot_SeqKokkos(xin, yin[i], z_h + i));
      } else break; // break the while loop
    }
    i = j;
  }
  PetscCall(VecRestoreKokkosView(xin, &xv));
  if (i < nv) { // finish the remaining if any
    if (conjugate) PetscCall(VecMDot_SeqKokkos(xin, nv - i, yin + i, z_h + i));
    else PetscCall(VecMTDot_SeqKokkos(xin, nv - i, yin + i, z_h + i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMDot_SeqKokkos_GEMV(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMultiDot_SeqKokkos_GEMV(PETSC_TRUE, xin, nv, yin, z)); // conjugate
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMTDot_SeqKokkos_GEMV(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMultiDot_SeqKokkos_GEMV(PETSC_FALSE, xin, nv, yin, z)); // transpose
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* x[:] = alpha */
PetscErrorCode VecSet_SeqKokkos(Vec xin, PetscScalar alpha)
{
  PetscScalarKokkosView xv;
  auto                  exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosViewWrite(xin, &xv));
  PetscCallCXX(KokkosBlas::fill(exec, xv, alpha));
  PetscCall(VecRestoreKokkosViewWrite(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* x = alpha x */
PetscErrorCode VecScale_SeqKokkos(Vec xin, PetscScalar alpha)
{
  auto exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecSet_SeqKokkos(xin, alpha));
  } else if (alpha != (PetscScalar)1.0) {
    PetscScalarKokkosView xv;

    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin, &xv));
    PetscCallCXX(KokkosBlas::scal(exec, xv, alpha, xv));
    PetscCall(VecRestoreKokkosView(xin, &xv));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(xin->map->n));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = y^H x */
PetscErrorCode VecDot_SeqKokkos(Vec xin, Vec yin, PetscScalar *z)
{
  ConstPetscScalarKokkosView xv, yv;
  auto                       exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCall(VecGetKokkosView(yin, &yv));
  PetscCallCXX(*z = KokkosBlas::dot(exec, yv, xv)); /* KokkosBlas::dot(a,b) takes conjugate of a */
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(VecRestoreKokkosView(yin, &yv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(PetscMax(2.0 * xin->map->n - 1, 0.0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = x, where x is VECKOKKOS, but y may be not */
PetscErrorCode VecCopy_SeqKokkos(Vec xin, Vec yin)
{
  auto exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  if (xin != yin) {
    Vec_Kokkos *xkok = static_cast<Vec_Kokkos *>(xin->spptr);
    if (yin->offloadmask == PETSC_OFFLOAD_KOKKOS) {
      /* y is also a VecKokkos */
      Vec_Kokkos *ykok = static_cast<Vec_Kokkos *>(yin->spptr);
      /* Kokkos rule: if x's host has newer data, it will copy to y's host view; otherwise to y's device view
        In case x's host is newer, y's device is newer, it will error (though should not, I think). So we just
        clear y's sync state.
       */
      ykok->v_dual.clear_sync_state();
      PetscCallCXX(Kokkos::deep_copy(exec, ykok->v_dual, xkok->v_dual)); // either cpu2cpu or gpu2cpu, so don't log it
    } else {
      PetscScalar *yarray;
      PetscCall(VecGetArrayWrite(yin, &yarray));
      PetscScalarKokkosViewHost yv(yarray, yin->map->n);
      if (xkok->v_dual.need_sync_host()) {                                     // x's device has newer data
        PetscCallCXX(Kokkos::deep_copy(exec, yv, xkok->v_dual.view_device())); // gpu2cpu
        PetscCallCXX(exec.fence());                                            // finish the deep copy
        PetscCall(PetscLogGpuToCpu(xkok->v_dual.extent(0) * sizeof(PetscScalar)));
      } else {
        PetscCallCXX(exec.fence());                                          // make sure xkok->v_dual.view_host() in ready for use on host;  Kokkos might also call it inside deep_copy(). We do it here for safety.
        PetscCallCXX(Kokkos::deep_copy(exec, yv, xkok->v_dual.view_host())); // Host view to host view deep copy, done on host
      }
      PetscCall(VecRestoreArrayWrite(yin, &yarray));
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y[i] <--> x[i] */
PetscErrorCode VecSwap_SeqKokkos(Vec xin, Vec yin)
{
  PetscFunctionBegin;
  if (xin != yin) {
    PetscScalarKokkosView xv, yv;

    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin, &xv));
    PetscCall(VecGetKokkosView(yin, &yv));
    PetscCallCXX(Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n), KOKKOS_LAMBDA(const PetscInt &i) {
        PetscScalar tmp = xv(i);
        xv(i)           = yv(i);
        yv(i)           = tmp;
      }));
    PetscCall(VecRestoreKokkosView(xin, &xv));
    PetscCall(VecRestoreKokkosView(yin, &yv));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*  w = alpha x + y */
PetscErrorCode VecWAXPY_SeqKokkos(Vec win, PetscScalar alpha, Vec xin, Vec yin)
{
  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecCopy_SeqKokkos(yin, win));
  } else {
    ConstPetscScalarKokkosView xv, yv;
    PetscScalarKokkosView      wv;

    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosViewWrite(win, &wv));
    PetscCall(VecGetKokkosView(xin, &xv));
    PetscCall(VecGetKokkosView(yin, &yv));
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, win->map->n), KOKKOS_LAMBDA(const PetscInt &i) { wv(i) = alpha * xv(i) + yv(i); }));
    PetscCall(VecRestoreKokkosView(xin, &xv));
    PetscCall(VecRestoreKokkosView(yin, &yv));
    PetscCall(VecRestoreKokkosViewWrite(win, &wv));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(2.0 * win->map->n));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <PetscInt ValueCount>
struct MAXPYFunctor {
  static_assert(ValueCount >= 1 && ValueCount <= 8, "ValueCount must be in [1, 8]");
  typedef ConstPetscScalarKokkosView::size_type size_type;

  PetscScalarKokkosView      yv;
  PetscScalar                a[8];
  ConstPetscScalarKokkosView xv[8];

  MAXPYFunctor(PetscScalarKokkosView yv, PetscScalar a0, PetscScalar a1, PetscScalar a2, PetscScalar a3, PetscScalar a4, PetscScalar a5, PetscScalar a6, PetscScalar a7, ConstPetscScalarKokkosView xv0, ConstPetscScalarKokkosView xv1, ConstPetscScalarKokkosView xv2, ConstPetscScalarKokkosView xv3, ConstPetscScalarKokkosView xv4, ConstPetscScalarKokkosView xv5, ConstPetscScalarKokkosView xv6, ConstPetscScalarKokkosView xv7) :
    yv(yv)
  {
    a[0]  = a0;
    a[1]  = a1;
    a[2]  = a2;
    a[3]  = a3;
    a[4]  = a4;
    a[5]  = a5;
    a[6]  = a6;
    a[7]  = a7;
    xv[0] = xv0;
    xv[1] = xv1;
    xv[2] = xv2;
    xv[3] = xv3;
    xv[4] = xv4;
    xv[5] = xv5;
    xv[6] = xv6;
    xv[7] = xv7;
  }

  KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
  {
    for (PetscInt j = 0; j < ValueCount; ++j) yv(i) += a[j] * xv[j](i);
  }
};

/*  y = y + sum alpha[i] x[i] */
PetscErrorCode VecMAXPY_SeqKokkos(Vec yin, PetscInt nv, const PetscScalar *alpha, Vec *xin)
{
  PetscInt                   i, j, cur = 0, ngroup = nv / 8, rem = nv % 8, N = yin->map->n;
  PetscScalarKokkosView      yv;
  PetscScalar                a[8];
  ConstPetscScalarKokkosView xv[8];
  Kokkos::RangePolicy<>      policy(PetscGetKokkosExecutionSpace(), 0, N);

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(yin, &yv));
  for (i = 0; i < ngroup; i++) { /* 8 x's per group */
    for (j = 0; j < 8; j++) {    /* Fill the parameters */
      a[j] = alpha[cur + j];
      PetscCall(VecGetKokkosView(xin[cur + j], &xv[j]));
    }
    MAXPYFunctor<8> maxpy(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]);
    PetscCallCXX(Kokkos::parallel_for(policy, maxpy));
    for (j = 0; j < 8; j++) PetscCall(VecRestoreKokkosView(xin[cur + j], &xv[j]));
    cur += 8;
  }

  if (rem) { /* The remaining */
    for (j = 0; j < rem; j++) {
      a[j] = alpha[cur + j];
      PetscCall(VecGetKokkosView(xin[cur + j], &xv[j]));
    }
    // clang-format off
    switch (rem) {
    case 1: PetscCallCXX(Kokkos::parallel_for(policy, MAXPYFunctor<1>(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]))); break;
    case 2: PetscCallCXX(Kokkos::parallel_for(policy, MAXPYFunctor<2>(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]))); break;
    case 3: PetscCallCXX(Kokkos::parallel_for(policy, MAXPYFunctor<3>(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]))); break;
    case 4: PetscCallCXX(Kokkos::parallel_for(policy, MAXPYFunctor<4>(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]))); break;
    case 5: PetscCallCXX(Kokkos::parallel_for(policy, MAXPYFunctor<5>(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]))); break;
    case 6: PetscCallCXX(Kokkos::parallel_for(policy, MAXPYFunctor<6>(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]))); break;
    case 7: PetscCallCXX(Kokkos::parallel_for(policy, MAXPYFunctor<7>(yv, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], xv[6], xv[7]))); break;
    }
    // clang-format on
    for (j = 0; j < rem; j++) PetscCall(VecRestoreKokkosView(xin[cur + j], &xv[j]));
  }
  PetscCall(VecRestoreKokkosView(yin, &yv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(nv * 2.0 * yin->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*  y = y + sum alpha[i] x[i] */
PetscErrorCode VecMAXPY_SeqKokkos_GEMV(Vec yin, PetscInt nv, const PetscScalar *a_h, Vec *xin)
{
  const PetscInt             n = yin->map->n;
  PetscInt                   i, j, nfail;
  PetscScalarKokkosView      yv;
  ConstPetscScalarKokkosView xfirst, xnext;
  PetscBool                  stop = PETSC_FALSE;
  PetscInt                   lda  = 0, m;
  const PetscScalar         *xarray;
  PetscScalar               *a_d = nullptr;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(yin, &yv));
#if defined(KOKKOS_ENABLE_UNIFIED_MEMORY)
  a_d = const_cast<PetscScalar *>(a_h);
#endif
  i = nfail = 0;
  while (i < nv) {
    stop = PETSC_FALSE;
    PetscCall(VecGetKokkosView(xin[i], &xfirst));
    xarray = xfirst.data();
    for (j = i + 1; j < nv; j++) {
      PetscCall(VecGetKokkosView(xin[j], &xnext));
      if (j == i + 1) {
        lda = xnext.data() - xfirst.data();
        if (lda < 0 || lda - n > 64) stop = PETSC_TRUE;    // avoid using arbitrary lda; 64 bytes are a big enough alignment in VecDuplicateVecs
      } else if (lda * (j - i) != xnext.data() - xarray) { // not in the same stride? if so, stop here
        stop = PETSC_TRUE;
      }
      PetscCall(VecRestoreKokkosView(xin[j], &xnext));
      if (stop) break;
    }
    PetscCall(VecRestoreKokkosView(xin[i], &xfirst));

    m = j - i;
    if (m > 1) {
      if (!a_d) {
        if (nv > PetscScalarPoolSize) { // rare case
          PetscScalarPoolSize = nv;
          PetscCallCXX(PetscScalarPool = static_cast<PetscScalar *>(Kokkos::kokkos_realloc(PetscScalarPool, PetscScalarPoolSize)));
        }
        a_d = PetscScalarPool;
      }
      const auto &B  = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft>(xarray, lda, m);
      const auto &A  = Kokkos::subview(B, std::pair<PetscInt, PetscInt>(0, n), Kokkos::ALL);
      auto        av = PetscScalarKokkosDualView(PetscScalarKokkosView(a_d + i, m), PetscScalarKokkosViewHost(const_cast<PetscScalar *>(a_h) + i, m));
      av.modify_host();
      PetscCall(KokkosDualViewSyncDevice(av, PetscGetKokkosExecutionSpace()));
      PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "N", 1.0, A, av.view_device(), 1.0, yv));
      PetscCall(PetscLogGpuFlops(m * 2.0 * n));
    } else {
      // we only allow falling back on VecAXPY once
      if (nfail++ == 0) PetscCall(VecAXPY_SeqKokkos(yin, a_h[i], xin[i]));
      else break; // break the while loop
    }
    i = j;
  }
  // finish the remaining if any
  PetscCall(VecRestoreKokkosView(yin, &yv));
  if (i < nv) PetscCall(VecMAXPY_SeqKokkos(yin, nv - i, a_h + i, xin + i));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = alpha x + beta y */
PetscErrorCode VecAXPBY_SeqKokkos(Vec yin, PetscScalar alpha, PetscScalar beta, Vec xin)
{
  PetscBool xiskok, yiskok;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)xin, &xiskok, VECSEQKOKKOS, VECMPIKOKKOS, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)yin, &yiskok, VECSEQKOKKOS, VECMPIKOKKOS, ""));
  if (xiskok && yiskok) {
    ConstPetscScalarKokkosView xv;
    PetscScalarKokkosView      yv;

    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin, &xv));
    PetscCall(VecGetKokkosView(yin, &yv));
    PetscCallCXX(KokkosBlas::axpby(PetscGetKokkosExecutionSpace(), alpha, xv, beta, yv));
    PetscCall(VecRestoreKokkosView(xin, &xv));
    PetscCall(VecRestoreKokkosView(yin, &yv));
    PetscCall(PetscLogGpuTimeEnd());
    if (alpha == (PetscScalar)0.0 || beta == (PetscScalar)0.0) {
      PetscCall(PetscLogGpuFlops(xin->map->n));
    } else if (beta == (PetscScalar)1.0 || alpha == (PetscScalar)1.0) {
      PetscCall(PetscLogGpuFlops(2.0 * xin->map->n));
    } else {
      PetscCall(PetscLogGpuFlops(3.0 * xin->map->n));
    }
  } else {
    PetscCall(VecAXPBY_Seq(yin, alpha, beta, xin));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = alpha x + beta y + gamma z */
PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin)
{
  ConstPetscScalarKokkosView xv, yv;
  PetscScalarKokkosView      zv;
  Kokkos::RangePolicy<>      policy(PetscGetKokkosExecutionSpace(), 0, zin->map->n);

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(zin, &zv));
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCall(VecGetKokkosView(yin, &yv));
  if (gamma == (PetscScalar)0.0) { // a common case
    if (alpha == -beta) {
      PetscCallCXX(Kokkos::parallel_for( // a common case
        policy, KOKKOS_LAMBDA(const PetscInt &i) { zv(i) = alpha * (xv(i) - yv(i)); }));
    } else {
      PetscCallCXX(Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const PetscInt &i) { zv(i) = alpha * xv(i) + beta * yv(i); }));
    }
  } else {
    PetscCallCXX(KokkosBlas::update(PetscGetKokkosExecutionSpace(), alpha, xv, beta, yv, gamma, zv));
  }
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(VecRestoreKokkosView(yin, &yv));
  PetscCall(VecRestoreKokkosView(zin, &zv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(zin->map->n * 5.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* w = x*y. Any subset of the x, y, and w may be the same vector.

  w is of type VecKokkos, but x, y may be not.
*/
PetscErrorCode VecPointwiseMult_SeqKokkos(Vec win, Vec xin, Vec yin)
{
  PetscInt n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetLocalSize(win, &n));
  if (xin->offloadmask != PETSC_OFFLOAD_KOKKOS || yin->offloadmask != PETSC_OFFLOAD_KOKKOS) {
    PetscScalarKokkosViewHost wv;
    const PetscScalar        *xp, *yp;
    PetscCall(VecGetArrayRead(xin, &xp));
    PetscCall(VecGetArrayRead(yin, &yp));
    PetscCall(VecGetKokkosViewWrite(win, &wv));

    ConstPetscScalarKokkosViewHost xv(xp, n), yv(yp, n);
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n), KOKKOS_LAMBDA(const PetscInt &i) { wv(i) = xv(i) * yv(i); }));

    PetscCall(VecRestoreArrayRead(xin, &xp));
    PetscCall(VecRestoreArrayRead(yin, &yp));
    PetscCall(VecRestoreKokkosViewWrite(win, &wv));
  } else {
    ConstPetscScalarKokkosView xv, yv;
    PetscScalarKokkosView      wv;

    PetscCall(VecGetKokkosViewWrite(win, &wv));
    PetscCall(VecGetKokkosView(xin, &xv));
    PetscCall(VecGetKokkosView(yin, &yv));
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt &i) { wv(i) = xv(i) * yv(i); }));
    PetscCall(VecRestoreKokkosView(yin, &yv));
    PetscCall(VecRestoreKokkosView(xin, &xv));
    PetscCall(VecRestoreKokkosViewWrite(win, &wv));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* w = x/y */
PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec win, Vec xin, Vec yin)
{
  PetscInt n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetLocalSize(win, &n));
  if (xin->offloadmask != PETSC_OFFLOAD_KOKKOS || yin->offloadmask != PETSC_OFFLOAD_KOKKOS) {
    PetscScalarKokkosViewHost wv;
    const PetscScalar        *xp, *yp;
    PetscCall(VecGetArrayRead(xin, &xp));
    PetscCall(VecGetArrayRead(yin, &yp));
    PetscCall(VecGetKokkosViewWrite(win, &wv));

    ConstPetscScalarKokkosViewHost xv(xp, n), yv(yp, n);
    PetscCallCXX(Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n), KOKKOS_LAMBDA(const PetscInt &i) {
        if (yv(i) != 0.0) wv(i) = xv(i) / yv(i);
        else wv(i) = 0.0;
      }));

    PetscCall(VecRestoreArrayRead(xin, &xp));
    PetscCall(VecRestoreArrayRead(yin, &yp));
    PetscCall(VecRestoreKokkosViewWrite(win, &wv));
  } else {
    ConstPetscScalarKokkosView xv, yv;
    PetscScalarKokkosView      wv;

    PetscCall(VecGetKokkosViewWrite(win, &wv));
    PetscCall(VecGetKokkosView(xin, &xv));
    PetscCall(VecGetKokkosView(yin, &yv));
    PetscCallCXX(Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt &i) {
        if (yv(i) != 0.0) wv(i) = xv(i) / yv(i);
        else wv(i) = 0.0;
      }));
    PetscCall(VecRestoreKokkosView(yin, &yv));
    PetscCall(VecRestoreKokkosView(xin, &xv));
    PetscCall(VecRestoreKokkosViewWrite(win, &wv));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(win->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNorm_SeqKokkos(Vec xin, NormType type, PetscReal *z)
{
  const PetscInt             n = xin->map->n;
  ConstPetscScalarKokkosView xv;
  auto                       exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  if (type == NORM_1_AND_2) {
    PetscCall(VecNorm_SeqKokkos(xin, NORM_1, z));
    PetscCall(VecNorm_SeqKokkos(xin, NORM_2, z + 1));
  } else {
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin, &xv));
    if (type == NORM_2 || type == NORM_FROBENIUS) {
      PetscCallCXX(*z = KokkosBlas::nrm2(exec, xv));
      PetscCall(PetscLogGpuFlops(PetscMax(2.0 * n - 1, 0.0)));
    } else if (type == NORM_1) {
      PetscCallCXX(*z = KokkosBlas::nrm1(exec, xv));
      PetscCall(PetscLogGpuFlops(PetscMax(n - 1.0, 0.0)));
    } else if (type == NORM_INFINITY) {
      PetscCallCXX(*z = KokkosBlas::nrminf(exec, xv));
    }
    PetscCall(VecRestoreKokkosView(xin, &xv));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecErrorWeightedNorms_SeqKokkos(Vec U, Vec Y, Vec E, NormType wnormtype, PetscReal atol, Vec vatol, PetscReal rtol, Vec vrtol, PetscReal ignore_max, PetscReal *norm, PetscInt *norm_loc, PetscReal *norma, PetscInt *norma_loc, PetscReal *normr, PetscInt *normr_loc)
{
  ConstPetscScalarKokkosView u, y, erra, atola, rtola;
  PetscBool                  has_E = PETSC_FALSE, has_atol = PETSC_FALSE, has_rtol = PETSC_FALSE;
  PetscInt                   n, n_loc = 0, na_loc = 0, nr_loc = 0;
  PetscReal                  nrm = 0, nrma = 0, nrmr = 0;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(U, &n));
  PetscCall(VecGetKokkosView(U, &u));
  PetscCall(VecGetKokkosView(Y, &y));
  if (E) {
    PetscCall(VecGetKokkosView(E, &erra));
    has_E = PETSC_TRUE;
  }
  if (vatol) {
    PetscCall(VecGetKokkosView(vatol, &atola));
    has_atol = PETSC_TRUE;
  }
  if (vrtol) {
    PetscCall(VecGetKokkosView(vrtol, &rtola));
    has_rtol = PETSC_TRUE;
  }

  if (wnormtype == NORM_INFINITY) {
    PetscCallCXX(Kokkos::parallel_reduce(
      "VecErrorWeightedNorms_INFINITY", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n),
      KOKKOS_LAMBDA(const PetscInt &i, PetscReal &l_nrm, PetscReal &l_nrma, PetscReal &l_nrmr, PetscInt &l_n_loc, PetscInt &l_na_loc, PetscInt &l_nr_loc) {
        PetscReal err, tol, tola, tolr, l_atol, l_rtol;
        if (PetscAbsScalar(y(i)) >= ignore_max && PetscAbsScalar(u(i)) >= ignore_max) {
          l_atol = has_atol ? PetscRealPart(atola(i)) : atol;
          l_rtol = has_rtol ? PetscRealPart(rtola(i)) : rtol;
          err    = has_E ? PetscAbsScalar(erra(i)) : PetscAbsScalar(y(i) - u(i));
          tola   = l_atol;
          tolr   = l_rtol * PetscMax(PetscAbsScalar(u(i)), PetscAbsScalar(y(i)));
          tol    = tola + tolr;
          if (tola > 0.) {
            l_nrma = PetscMax(l_nrma, err / tola);
            l_na_loc++;
          }
          if (tolr > 0.) {
            l_nrmr = PetscMax(l_nrmr, err / tolr);
            l_nr_loc++;
          }
          if (tol > 0.) {
            l_nrm = PetscMax(l_nrm, err / tol);
            l_n_loc++;
          }
        }
      },
      Kokkos::Max<PetscReal>(nrm), Kokkos::Max<PetscReal>(nrma), Kokkos::Max<PetscReal>(nrmr), n_loc, na_loc, nr_loc));
  } else {
    PetscCallCXX(Kokkos::parallel_reduce(
      "VecErrorWeightedNorms_NORM_2", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n),
      KOKKOS_LAMBDA(const PetscInt &i, PetscReal &l_nrm, PetscReal &l_nrma, PetscReal &l_nrmr, PetscInt &l_n_loc, PetscInt &l_na_loc, PetscInt &l_nr_loc) {
        PetscReal err, tol, tola, tolr, l_atol, l_rtol;
        if (PetscAbsScalar(y(i)) >= ignore_max && PetscAbsScalar(u(i)) >= ignore_max) {
          l_atol = has_atol ? PetscRealPart(atola(i)) : atol;
          l_rtol = has_rtol ? PetscRealPart(rtola(i)) : rtol;
          err    = has_E ? PetscAbsScalar(erra(i)) : PetscAbsScalar(y(i) - u(i));
          tola   = l_atol;
          tolr   = l_rtol * PetscMax(PetscAbsScalar(u(i)), PetscAbsScalar(y(i)));
          tol    = tola + tolr;
          if (tola > 0.) {
            l_nrma += PetscSqr(err / tola);
            l_na_loc++;
          }
          if (tolr > 0.) {
            l_nrmr += PetscSqr(err / tolr);
            l_nr_loc++;
          }
          if (tol > 0.) {
            l_nrm += PetscSqr(err / tol);
            l_n_loc++;
          }
        }
      },
      nrm, nrma, nrmr, n_loc, na_loc, nr_loc));
  }

  if (wnormtype == NORM_2) {
    *norm  = PetscSqrtReal(nrm);
    *norma = PetscSqrtReal(nrma);
    *normr = PetscSqrtReal(nrmr);
  } else {
    *norm  = nrm;
    *norma = nrma;
    *normr = nrmr;
  }
  *norm_loc  = n_loc;
  *norma_loc = na_loc;
  *normr_loc = nr_loc;

  if (E) PetscCall(VecRestoreKokkosView(E, &erra));
  if (vatol) PetscCall(VecRestoreKokkosView(vatol, &atola));
  if (vrtol) PetscCall(VecRestoreKokkosView(vrtol, &rtola));
  PetscCall(VecRestoreKokkosView(U, &u));
  PetscCall(VecRestoreKokkosView(Y, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A functor for DotNorm2 so that we can compute dp and nm in one kernel */
struct DotNorm2 {
  typedef PetscScalar                           value_type[];
  typedef ConstPetscScalarKokkosView::size_type size_type;

  size_type                  value_count;
  ConstPetscScalarKokkosView xv_, yv_; /* first and second vectors in VecDotNorm2. The order matters. */

  DotNorm2(ConstPetscScalarKokkosView &xv, ConstPetscScalarKokkosView &yv) : value_count(2), xv_(xv), yv_(yv) { }

  KOKKOS_INLINE_FUNCTION void operator()(const size_type i, value_type result) const
  {
    result[0] += PetscConj(yv_(i)) * xv_(i);
    result[1] += PetscConj(yv_(i)) * yv_(i);
  }

  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const
  {
    dst[0] += src[0];
    dst[1] += src[1];
  }

  KOKKOS_INLINE_FUNCTION void init(value_type result) const
  {
    result[0] = 0.0;
    result[1] = 0.0;
  }
};

/* dp = y^H x, nm = y^H y */
PetscErrorCode VecDotNorm2_SeqKokkos(Vec xin, Vec yin, PetscScalar *dp, PetscScalar *nm)
{
  ConstPetscScalarKokkosView xv, yv;
  PetscScalar                result[2];

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCall(VecGetKokkosView(yin, &yv));
  DotNorm2 dn(xv, yv);
  PetscCallCXX(Kokkos::parallel_reduce(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n), dn, result));
  *dp = result[0];
  *nm = result[1];
  PetscCall(VecRestoreKokkosView(yin, &yv));
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(4.0 * xin->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecConjugate_SeqKokkos(Vec xin)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin, &xv));
  PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, xin->map->n), KOKKOS_LAMBDA(const PetscInt &i) { xv(i) = PetscConj(xv(i)); }));
  PetscCall(VecRestoreKokkosView(xin, &xv));
  PetscCall(PetscLogGpuTimeEnd());
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Temporarily replace the array in vin with a[]. Return to the original array with a call to VecResetArray() */
PetscErrorCode VecPlaceArray_SeqKokkos(Vec vin, const PetscScalar *a)
{
  Vec_Seq    *vecseq = (Vec_Seq *)vin->data;
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(vin->spptr);

  PetscFunctionBegin;
  PetscCall(VecPlaceArray_Seq(vin, a));
  PetscCall(veckok->UpdateArray<HostMirrorMemorySpace>(vecseq->array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecResetArray_SeqKokkos(Vec vin)
{
  Vec_Seq    *vecseq = (Vec_Seq *)vin->data;
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(vin->spptr);

  PetscFunctionBegin;
  /* User wants to unhook the provided host array. Sync it so that user can get the latest */
  PetscCall(KokkosDualViewSyncHost(veckok->v_dual, PetscGetKokkosExecutionSpace()));
  PetscCall(VecResetArray_Seq(vin)); /* Swap back the old host array, assuming its has the latest value */
  PetscCall(veckok->UpdateArray<HostMirrorMemorySpace>(vecseq->array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecKokkosPlaceArray - Allows one to replace the device array in a VecKokkos vector with a
  device array provided by the user. This is useful to avoid copying an array into a vector.

  Logically Collective; No Fortran Support

  Input Parameters:
+ v - the VecKokkos vector
- a - the device array

  Level: developer

  Notes:

  You can return to the original array with a call to `VecKokkosResetArray()`. `vec` does not take
  ownership of `array` in any way.

  The user manages the device array so PETSc doesn't care how it was allocated.

  The user must free `array` themselves but be careful not to
  do so before the vector has either been destroyed, had its original array restored with
  `VecKokkosResetArray()` or permanently replaced with `VecReplaceArray()`.

.seealso: [](ch_vectors), `Vec`, `VecGetArray()`, `VecKokkosResetArray()`, `VecReplaceArray()`, `VecResetArray()`
@*/
PetscErrorCode VecKokkosPlaceArray(Vec v, PetscScalar *a)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  // Sync the old device view before replacing it; so that when it is put back, it has the saved value.
  PetscCall(KokkosDualViewSyncDevice(veckok->v_dual, PetscGetKokkosExecutionSpace()));
  PetscCallCXX(veckok->unplaced_d = veckok->v_dual.view_device());
  // We assume a[] contains the latest data and discard the vector's old sync state
  PetscCall(veckok->UpdateArray<DefaultMemorySpace>(a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecKokkosResetArray - Resets a vector to use its default memory. Call this
  after the use of `VecKokkosPlaceArray()`.

  Not Collective

  Input Parameter:
. v - the vector

  Level: developer

  Notes:

  After the call, the original array placed in with `VecKokkosPlaceArray()` will contain the latest value of the vector.
  Note that device kernels are asynchronous. Users are responsible to sync the device if they wish to have immediate access
  to the data in the array. Also, after the call, `v` will contain whatever data before `VecKokkosPlaceArray()`.

.seealso: [](ch_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecReplaceArray()`, `VecKokkosPlaceArray()`
@*/
PetscErrorCode VecKokkosResetArray(Vec v)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  // User wants to unhook the provided device array. Sync it so that user can get the latest
  PetscCall(KokkosDualViewSyncDevice(veckok->v_dual, PetscGetKokkosExecutionSpace()));
  // Put the unplaced device array back, and set an appropriate modify flag
  PetscCall(veckok->UpdateArray<DefaultMemorySpace>(veckok->unplaced_d.data()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Replace the array in vin with a[] that must be allocated by PetscMalloc. a[] is owned by vin afterwards. */
PetscErrorCode VecReplaceArray_SeqKokkos(Vec vin, const PetscScalar *a)
{
  Vec_Seq    *vecseq = (Vec_Seq *)vin->data;
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(vin->spptr);

  PetscFunctionBegin;
  /* Make sure the users array has the latest values */
  if (vecseq->array != vecseq->array_allocated) PetscCall(KokkosDualViewSyncHost(veckok->v_dual, PetscGetKokkosExecutionSpace()));
  PetscCall(VecReplaceArray_Seq(vin, a));
  PetscCall(veckok->UpdateArray<HostMirrorMemorySpace>(vecseq->array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Maps the local portion of vector v into vector w */
PetscErrorCode VecGetLocalVector_SeqKokkos(Vec v, Vec w)
{
  Vec_Seq    *vecseq = static_cast<Vec_Seq *>(w->data);
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(w->spptr);

  PetscFunctionBegin;
  PetscCheckTypeNames(w, VECSEQKOKKOS, VECMPIKOKKOS);
  /* Destroy w->data, w->spptr */
  if (vecseq) {
    PetscCall(PetscFree(vecseq->array_allocated));
    PetscCall(PetscFree(w->data));
  }
  delete veckok;

  /* Replace with v's */
  w->data  = v->data;
  w->spptr = v->spptr;
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec v, Vec w)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(w, VECSEQKOKKOS, VECMPIKOKKOS);
  v->data  = w->data;
  v->spptr = w->spptr;
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  /* TODO: need to think if setting w->data/spptr to NULL is safe */
  w->data  = NULL;
  w->spptr = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetArray_SeqKokkos(Vec v, PetscScalar **a)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  PetscCall(KokkosDualViewSyncHost(veckok->v_dual, PetscGetKokkosExecutionSpace()));
  *a = *((PetscScalar **)v->data);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecRestoreArray_SeqKokkos(Vec v, PetscScalar **a)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  PetscCallCXX(veckok->v_dual.modify_host());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Get array on host to overwrite, so no need to sync host. In VecRestoreArrayWrite() we will mark host is modified. */
PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec v, PetscScalar **a)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  PetscCallCXX(veckok->v_dual.clear_sync_state());
  *a = veckok->v_dual.view_host().data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetArrayAndMemType_SeqKokkos(Vec v, PetscScalar **a, PetscMemType *mtype)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  /* Always return up-to-date in the default memory space */
  PetscCall(KokkosDualViewSyncDevice(veckok->v_dual, PetscGetKokkosExecutionSpace()));
  *a = veckok->v_dual.view_device().data();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS; // Could be PETSC_MEMTYPE_HOST when Kokkos was not configured with cuda etc.
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecRestoreArrayAndMemType_SeqKokkos(Vec v, PetscScalar **a)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  if (PetscMemTypeHost(PETSC_MEMTYPE_KOKKOS)) {
    PetscCallCXX(veckok->v_dual.modify_host());
  } else {
    PetscCallCXX(veckok->v_dual.modify_device());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetArrayWriteAndMemType_SeqKokkos(Vec v, PetscScalar **a, PetscMemType *mtype)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);

  PetscFunctionBegin;
  // Since the data will be overwritten, we clear the sync state to suppress potential memory copying in sync'ing
  PetscCallCXX(veckok->v_dual.clear_sync_state()); // So that in restore, we can safely modify_device()
  PetscCall(KokkosDualViewSyncDevice(veckok->v_dual, PetscGetKokkosExecutionSpace()));
  *a = veckok->v_dual.view_device().data();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS; // Could be PETSC_MEMTYPE_HOST when Kokkos was not configured with cuda etc.
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Copy xin's sync state to y */
static PetscErrorCode VecCopySyncState_Kokkos_Private(Vec xin, Vec yout)
{
  Vec_Kokkos *xkok = static_cast<Vec_Kokkos *>(xin->spptr);
  Vec_Kokkos *ykok = static_cast<Vec_Kokkos *>(yout->spptr);

  PetscFunctionBegin;
  PetscCallCXX(ykok->v_dual.clear_sync_state());
  if (xkok->v_dual.need_sync_host()) {
    PetscCallCXX(ykok->v_dual.modify_device());
  } else if (xkok->v_dual.need_sync_device()) {
    PetscCallCXX(ykok->v_dual.modify_host());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecCreateSeqKokkosWithArrays_Private(MPI_Comm, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], Vec *);

/* Internal routine shared by VecGetSubVector_{SeqKokkos,MPIKokkos} */
PetscErrorCode VecGetSubVector_Kokkos_Private(Vec x, PetscBool xIsMPI, IS is, Vec *y)
{
  PetscBool contig;
  PetscInt  n, N, start, bs;
  MPI_Comm  comm;
  Vec       z;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetSize(is, &N));
  PetscCall(VecGetSubVectorContiguityAndBS_Private(x, is, &contig, &start, &bs));

  if (contig) { /* We can do a no-copy (in-place) implementation with y sharing x's arrays */
    Vec_Kokkos        *xkok    = static_cast<Vec_Kokkos *>(x->spptr);
    const PetscScalar *array_h = xkok->v_dual.view_host().data() + start;
    const PetscScalar *array_d = xkok->v_dual.view_device().data() + start;

    /* These calls assume the input arrays are synced */
    if (xIsMPI) PetscCall(VecCreateMPIKokkosWithArrays_Private(comm, bs, n, N, array_h, array_d, &z)); /* x could be MPI even when x's comm size = 1 */
    else PetscCall(VecCreateSeqKokkosWithArrays_Private(comm, bs, n, array_h, array_d, &z));

    PetscCall(VecCopySyncState_Kokkos_Private(x, z)); /* Copy x's sync state to z */

    /* This is relevant only in debug mode */
    PetscInt state = 0;
    PetscCall(VecLockGet(x, &state));
    if (state) { /* x is either in read or read/write mode, therefore z, overlapped with x, can only be in read mode */
      PetscCall(VecLockReadPush(z));
    }

    z->ops->placearray   = NULL; /* z's arrays can't be replaced, because z does not own them */
    z->ops->replacearray = NULL;

  } else { /* Have to create a VecScatter and a stand-alone vector */
    PetscCall(VecGetSubVectorThroughVecScatter_Private(x, is, bs, &z));
  }
  *y = z;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecGetSubVector_SeqKokkos(Vec x, IS is, Vec *y)
{
  PetscFunctionBegin;
  PetscCall(VecGetSubVector_Kokkos_Private(x, PETSC_FALSE, is, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Restore subvector y to x */
PetscErrorCode VecRestoreSubVector_SeqKokkos(Vec x, IS is, Vec *y)
{
  VecScatter                    vscat;
  PETSC_UNUSED PetscObjectState dummystate = 0;
  PetscBool                     unchanged;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataGetInt((PetscObject)*y, VecGetSubVectorSavedStateId, dummystate, unchanged));
  if (unchanged) PetscFunctionReturn(PETSC_SUCCESS); /* If y's state has not changed since VecGetSubVector(), we only need to destroy it */

  PetscCall(PetscObjectQuery((PetscObject)*y, "VecGetSubVector_Scatter", (PetscObject *)&vscat));
  if (vscat) {
    PetscCall(VecScatterBegin(vscat, *y, x, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(vscat, *y, x, INSERT_VALUES, SCATTER_REVERSE));
  } else { /* y and x's (host and device) arrays overlap */
    Vec_Kokkos *xkok = static_cast<Vec_Kokkos *>(x->spptr);
    Vec_Kokkos *ykok = static_cast<Vec_Kokkos *>((*y)->spptr);
    PetscInt    state;

    PetscCall(VecLockGet(x, &state));
    PetscCheck(!state, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_WRONGSTATE, "Vec x is locked for read-only or read/write access");

    /* The tricky part: one has to carefully sync the arrays */
    auto exec = PetscGetKokkosExecutionSpace();
    if (xkok->v_dual.need_sync_device()) { /* x's host has newer data */
      /* Move y's latest values to host (since y is just a subset of x) */
      PetscCall(KokkosDualViewSyncHost(ykok->v_dual, exec));
    } else if (xkok->v_dual.need_sync_host()) {                /* x's device has newer data */
      PetscCall(KokkosDualViewSyncDevice(ykok->v_dual, exec)); /* Move y's latest data to device */
    } else {                                                   /* x's host and device data is already sync'ed; Copy y's sync state to x */
      PetscCall(VecCopySyncState_Kokkos_Private(*y, x));
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)x)); /* Since x is updated */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetPreallocationCOO_SeqKokkos(Vec x, PetscCount ncoo, const PetscInt coo_i[])
{
  Vec_Seq    *vecseq = static_cast<Vec_Seq *>(x->data);
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(x->spptr);
  PetscInt    m;

  PetscFunctionBegin;
  PetscCall(VecSetPreallocationCOO_Seq(x, ncoo, coo_i));
  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(veckok->SetUpCOO(vecseq, m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetValuesCOO_SeqKokkos(Vec x, const PetscScalar v[], InsertMode imode)
{
  Vec_Seq                    *vecseq = static_cast<Vec_Seq *>(x->data);
  Vec_Kokkos                 *veckok = static_cast<Vec_Kokkos *>(x->spptr);
  const PetscCountKokkosView &jmap1  = veckok->jmap1_d;
  const PetscCountKokkosView &perm1  = veckok->perm1_d;
  PetscScalarKokkosView       xv; /* View for vector x */
  ConstPetscScalarKokkosView  vv; /* View for array v[] */
  PetscInt                    m;
  PetscMemType                memtype;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(PetscGetMemType(v, &memtype));
  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
    PetscCallCXX(vv = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), ConstPetscScalarKokkosViewHost(v, vecseq->coo_n)));
  } else {
    PetscCallCXX(vv = ConstPetscScalarKokkosView(v, vecseq->coo_n)); /* Directly use v[]'s memory */
  }

  if (imode == INSERT_VALUES) PetscCall(VecGetKokkosViewWrite(x, &xv)); /* write vector */
  else PetscCall(VecGetKokkosView(x, &xv));                             /* read & write vector */

  PetscCallCXX(Kokkos::parallel_for(
    Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, m), KOKKOS_LAMBDA(const PetscInt &i) {
      PetscScalar sum = 0.0;
      for (PetscCount k = jmap1(i); k < jmap1(i + 1); k++) sum += vv(perm1(k));
      xv(i) = (imode == INSERT_VALUES ? 0.0 : xv(i)) + sum;
    }));

  if (imode == INSERT_VALUES) PetscCall(VecRestoreKokkosViewWrite(x, &xv));
  else PetscCall(VecRestoreKokkosView(x, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecCreate_SeqKokkos_Common(Vec); // forward declaration

/* Duplicate layout etc but not the values in the input vector */
static PetscErrorCode VecDuplicate_SeqKokkos(Vec win, Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecDuplicate_Seq(win, v)); /* It also dups ops of win */
  PetscCall(VecCreate_SeqKokkos_Common(*v));
  (*v)->ops[0] = win->ops[0]; // recover the ops[]. We need to always follow ops in win
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecDestroy_SeqKokkos(Vec v)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos *>(v->spptr);
  Vec_Seq    *vecseq = static_cast<Vec_Seq *>(v->data);

  PetscFunctionBegin;
  delete veckok;
  v->spptr = NULL;
  if (vecseq) PetscCall(VecDestroy_Seq(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Shared by all VecCreate/Duplicate routines for VecSeqKokkos
static PetscErrorCode VecCreate_SeqKokkos_Common(Vec v)
{
  PetscFunctionBegin;
  v->ops->abs             = VecAbs_SeqKokkos;
  v->ops->reciprocal      = VecReciprocal_SeqKokkos;
  v->ops->pointwisemult   = VecPointwiseMult_SeqKokkos;
  v->ops->min             = VecMin_SeqKokkos;
  v->ops->max             = VecMax_SeqKokkos;
  v->ops->sum             = VecSum_SeqKokkos;
  v->ops->shift           = VecShift_SeqKokkos;
  v->ops->norm            = VecNorm_SeqKokkos;
  v->ops->scale           = VecScale_SeqKokkos;
  v->ops->copy            = VecCopy_SeqKokkos;
  v->ops->set             = VecSet_SeqKokkos;
  v->ops->swap            = VecSwap_SeqKokkos;
  v->ops->axpy            = VecAXPY_SeqKokkos;
  v->ops->axpby           = VecAXPBY_SeqKokkos;
  v->ops->axpbypcz        = VecAXPBYPCZ_SeqKokkos;
  v->ops->pointwisedivide = VecPointwiseDivide_SeqKokkos;
  v->ops->setrandom       = VecSetRandom_SeqKokkos;

  v->ops->dot   = VecDot_SeqKokkos;
  v->ops->tdot  = VecTDot_SeqKokkos;
  v->ops->mdot  = VecMDot_SeqKokkos;
  v->ops->mtdot = VecMTDot_SeqKokkos;

  v->ops->dot_local   = VecDot_SeqKokkos;
  v->ops->tdot_local  = VecTDot_SeqKokkos;
  v->ops->mdot_local  = VecMDot_SeqKokkos;
  v->ops->mtdot_local = VecMTDot_SeqKokkos;

  v->ops->norm_local             = VecNorm_SeqKokkos;
  v->ops->maxpy                  = VecMAXPY_SeqKokkos;
  v->ops->aypx                   = VecAYPX_SeqKokkos;
  v->ops->waxpy                  = VecWAXPY_SeqKokkos;
  v->ops->dotnorm2               = VecDotNorm2_SeqKokkos;
  v->ops->errorwnorm             = VecErrorWeightedNorms_SeqKokkos;
  v->ops->placearray             = VecPlaceArray_SeqKokkos;
  v->ops->replacearray           = VecReplaceArray_SeqKokkos;
  v->ops->resetarray             = VecResetArray_SeqKokkos;
  v->ops->destroy                = VecDestroy_SeqKokkos;
  v->ops->duplicate              = VecDuplicate_SeqKokkos;
  v->ops->conjugate              = VecConjugate_SeqKokkos;
  v->ops->getlocalvector         = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvector     = VecRestoreLocalVector_SeqKokkos;
  v->ops->getlocalvectorread     = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvectorread = VecRestoreLocalVector_SeqKokkos;
  v->ops->getarraywrite          = VecGetArrayWrite_SeqKokkos;
  v->ops->getarray               = VecGetArray_SeqKokkos;
  v->ops->restorearray           = VecRestoreArray_SeqKokkos;

  v->ops->getarrayandmemtype      = VecGetArrayAndMemType_SeqKokkos;
  v->ops->restorearrayandmemtype  = VecRestoreArrayAndMemType_SeqKokkos;
  v->ops->getarraywriteandmemtype = VecGetArrayWriteAndMemType_SeqKokkos;
  v->ops->getsubvector            = VecGetSubVector_SeqKokkos;
  v->ops->restoresubvector        = VecRestoreSubVector_SeqKokkos;

  v->ops->setpreallocationcoo = VecSetPreallocationCOO_SeqKokkos;
  v->ops->setvaluescoo        = VecSetValuesCOO_SeqKokkos;

  v->offloadmask = PETSC_OFFLOAD_KOKKOS; // Mark this is a VECKOKKOS; We use this flag for cheap VECKOKKOS test.
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecCreateSeqKokkosWithArray - Creates a Kokkos sequential array-style vector,
  where the user provides the array space to store the vector values. The array
  provided must be a device array.

  Collective

  Input Parameters:
+ comm   - the communicator, should be PETSC_COMM_SELF
. bs     - the block size
. n      - the vector length
- darray - device memory where the vector elements are to be stored.

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
  same type as an existing vector.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy().
  The user should not free the array until the vector is destroyed.

  Level: intermediate

.seealso: `VecCreateMPICUDAWithArray()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `VecCreateGhost()`, `VecCreateSeq()`, `VecCreateSeqWithArray()`,
          `VecCreateMPIWithArray()`
@*/
PetscErrorCode VecCreateSeqKokkosWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar darray[], Vec *v)
{
  PetscMPIInt  size;
  Vec          w;
  Vec_Kokkos  *veckok = NULL;
  PetscScalar *harray;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQKOKKOS on more than one process");

  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(VecCreate(comm, &w));
  PetscCall(VecSetSizes(w, n, n));
  PetscCall(VecSetBlockSize(w, bs));
  if (!darray) { /* Allocate memory ourself if user provided NULL */
    PetscCall(VecSetType(w, VECSEQKOKKOS));
  } else {
    /* Build a VECSEQ, get its harray, and then build Vec_Kokkos along with darray */
    if (std::is_same<DefaultMemorySpace, HostMirrorMemorySpace>::value) {
      harray = const_cast<PetscScalar *>(darray);
      PetscCall(VecCreate_Seq_Private(w, harray)); /* Build a sequential vector with harray */
    } else {
      PetscCall(VecSetType(w, VECSEQ));
      harray = static_cast<Vec_Seq *>(w->data)->array;
    }
    PetscCall(PetscObjectChangeTypeName((PetscObject)w, VECSEQKOKKOS)); /* Change it to Kokkos */
    PetscCall(VecCreate_SeqKokkos_Common(w));
    PetscCallCXX(veckok = new Vec_Kokkos{n, harray, const_cast<PetscScalar *>(darray)});
    PetscCallCXX(veckok->v_dual.modify_device()); /* Mark the device is modified */
    w->spptr = static_cast<void *>(veckok);
  }
  *v = w;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecConvert_Seq_SeqKokkos_inplace(Vec v)
{
  Vec_Seq *vecseq;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscLayoutSetUp(v->map));
  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECSEQKOKKOS));
  PetscCall(VecCreate_SeqKokkos_Common(v));
  PetscCheck(!v->spptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "v->spptr not NULL");
  vecseq = static_cast<Vec_Seq *>(v->data);
  PetscCallCXX(v->spptr = new Vec_Kokkos(v->map->n, vecseq->array, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create a VECSEQKOKKOS with layout and arrays
static PetscErrorCode VecCreateSeqKokkosWithLayoutAndArrays_Private(PetscLayout map, const PetscScalar harray[], const PetscScalar darray[], Vec *v)
{
  Vec w;

  PetscFunctionBegin;
  if (map->n > 0) PetscCheck(darray, map->comm, PETSC_ERR_ARG_WRONG, "darray cannot be NULL");
#if defined(KOKKOS_ENABLE_UNIFIED_MEMORY)
  PetscCheck(harray == darray, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "harray and darray must be the same");
#endif
  PetscCall(VecCreateSeqWithLayoutAndArray_Private(map, harray, &w));
  PetscCall(PetscObjectChangeTypeName((PetscObject)w, VECSEQKOKKOS)); // Change it to VECKOKKOS
  PetscCall(VecCreate_SeqKokkos_Common(w));
  PetscCallCXX(w->spptr = new Vec_Kokkos(map->n, const_cast<PetscScalar *>(harray), const_cast<PetscScalar *>(darray)));
  *v = w;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   VecCreateSeqKokkosWithArrays_Private - Creates a Kokkos sequential array-style vector
   with user-provided arrays on host and device.

   Collective

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
.  harray - host memory where the vector elements are to be stored.
-  darray - device memory where the vector elements are to be stored.

   Output Parameter:
.  v - the vector

   Notes:
   Unlike VecCreate{Seq,MPI}CUDAWithArrays(), this routine is private since we do not expect users to use it directly.

   If there is no device, then harray and darray must be the same.
   If n is not zero, then harray and darray must be allocated.
   After the call, the created vector is supposed to be in a synchronized state, i.e.,
   we suppose harray and darray have the same data.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   Caller should not free the array until the vector is destroyed.
*/
static PetscErrorCode VecCreateSeqKokkosWithArrays_Private(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar harray[], const PetscScalar darray[], Vec *v)
{
  PetscMPIInt size;
  PetscLayout map;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQKOKKOS on more than one process");
  PetscCall(PetscLayoutCreateFromSizes(comm, n, n, bs, &map));
  PetscCall(VecCreateSeqKokkosWithLayoutAndArrays_Private(map, harray, darray, v));
  PetscCall(PetscLayoutDestroy(&map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO: ftn-auto generates veckok.kokkosf.c */
/*@C
  VecCreateSeqKokkos - Creates a standard, sequential array-style vector.

  Collective

  Input Parameters:
+ comm - the communicator, should be `PETSC_COMM_SELF`
- n    - the vector length

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
  same type as an existing vector.

  Level: intermediate

.seealso: `VecCreateMPI()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`
 @*/
PetscErrorCode VecCreateSeqKokkos(MPI_Comm comm, PetscInt n, Vec *v)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(VecCreate(comm, v));
  PetscCall(VecSetSizes(*v, n, n));
  PetscCall(VecSetType(*v, VECSEQKOKKOS)); /* Calls VecCreate_SeqKokkos */
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Duplicate a VECSEQKOKKOS
static PetscErrorCode VecDuplicateVecs_SeqKokkos_GEMV(Vec w, PetscInt m, Vec *V[])
{
  PetscInt64                lda; // use 64-bit as we will do "m * lda"
  PetscScalar              *array_h, *array_d;
  PetscLayout               map;
  PetscScalarKokkosDualView w_dual;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck()); // as we'll call kokkos_malloc()
  PetscCall(PetscMalloc1(m, V));
  PetscCall(VecGetLayout(w, &map));
  VecGetLocalSizeAligned(w, 64, &lda); // get in lda the 64-bytes aligned local size
  // See comments in VecCreate_SeqKokkos() on why we use DualView to allocate the memory
  PetscCallCXX(w_dual = PetscScalarKokkosDualView("VecDuplicateVecs", m * lda)); // Kokkos init's w_dual to zero

  // create the m vectors with raw arrays from v_dual
  array_h = w_dual.view_host().data();
  array_d = w_dual.view_device().data();
  for (PetscInt i = 0; i < m; i++) {
    Vec v;
    PetscCall(VecCreateSeqKokkosWithLayoutAndArrays_Private(map, &array_h[i * lda], &array_d[i * lda], &v));
    PetscCall(PetscObjectListDuplicate(((PetscObject)w)->olist, &((PetscObject)v)->olist));
    PetscCall(PetscFunctionListDuplicate(((PetscObject)w)->qlist, &((PetscObject)v)->qlist));
    v->ops[0] = w->ops[0];
    (*V)[i]   = v;
  }

  // let the first vector own the long DualView, so when it is destroyed it will free the v_dual
  if (m) {
    Vec v = (*V)[0];

    static_cast<Vec_Kokkos *>(v->spptr)->w_dual = w_dual; // stash the memory
    // disable replacearray of the first vector, as freeing its memory also frees others in the group.
    // But replacearray of others is ok, as they don't own their array.
    if (m > 1) v->ops->replacearray = VecReplaceArray_Default_GEMV_Error;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   VECSEQKOKKOS - VECSEQKOKKOS = "seqkokkos" - The basic sequential vector, modified to use Kokkos

   Options Database Keys:
. -vec_type seqkokkos - sets the vector type to VECSEQKOKKOS during a call to VecSetFromOptions()

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateMPIWithArray()`, `VECMPI`, `VecType`, `VecCreateMPI()`
M*/
PetscErrorCode VecCreate_SeqKokkos(Vec v)
{
  PetscBool                 mdot_use_gemv  = PETSC_TRUE;
  PetscBool                 maxpy_use_gemv = PETSC_FALSE; // default is false as we saw bad performance with vendors' GEMV with tall skinny matrices.
  PetscScalarKokkosDualView v_dual;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscLayoutSetUp(v->map));

  // Use DualView to allocate both the host array and the device array.
  // DualView first allocates the device array and then mirrors it to host.
  // With unified memory (e.g., on AMD MI300A APU), the two arrays are the same, with the host array
  // sharing the device array allocated by hipMalloc(), but not the other way around if we call
  // VecCreate_Seq() first and let the device array share the host array allocated by malloc().
  // hipMalloc() has an advantage over malloc() as it gives great binding and page size settings automatically, see
  // https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/pro-tips
  PetscCallCXX(v_dual = PetscScalarKokkosDualView("v_dual", v->map->n)); // Kokkos init's v_dual to zero

  PetscCall(VecCreate_Seq_Private(v, v_dual.view_host().data()));
  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECSEQKOKKOS));
  PetscCall(VecCreate_SeqKokkos_Common(v));
  PetscCheck(!v->spptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "v->spptr not NULL");
  PetscCallCXX(v->spptr = new Vec_Kokkos(v_dual));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-vec_mdot_use_gemv", &mdot_use_gemv, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-vec_maxpy_use_gemv", &maxpy_use_gemv, NULL));

  // allocate multiple vectors together
  if (mdot_use_gemv || maxpy_use_gemv) v->ops[0].duplicatevecs = VecDuplicateVecs_SeqKokkos_GEMV;

  if (mdot_use_gemv) {
    v->ops[0].mdot        = VecMDot_SeqKokkos_GEMV;
    v->ops[0].mtdot       = VecMTDot_SeqKokkos_GEMV;
    v->ops[0].mdot_local  = VecMDot_SeqKokkos_GEMV;
    v->ops[0].mtdot_local = VecMTDot_SeqKokkos_GEMV;
  }
  if (maxpy_use_gemv) v->ops[0].maxpy = VecMAXPY_SeqKokkos_GEMV;
  PetscFunctionReturn(PETSC_SUCCESS);
}
