/*
   Implements the sequential Kokkos vectors.
*/
#include <petscvec_kokkos.hpp>

#include <petsc/private/sfimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscmath.h>
#include <petscviewer.h>
#include <KokkosBlas.hpp>

#include <petscerror.h>
#include <../src/vec/vec/impls/dvecimpl.h> /* for VecCreate_Seq_Private */
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>

#if defined(PETSC_USE_DEBUG)
#define VecErrorIfNotKokkos(v)                                                                 \
  do {                                                                                         \
    PetscBool isKokkos = PETSC_FALSE;                                                          \
    PetscCall(PetscObjectTypeCompareAny((PetscObject)(v),&isKokkos,VECSEQKOKKOS,VECMPIKOKKOS,VECKOKKOS,"")); \
    PetscCheck(isKokkos,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Calling VECKOKKOS methods on a non-VECKOKKOS object"); \
  } while (0)
#else
#define VecErrorIfNotKokkos(v) do {(void)(v);} while (0)
#endif

template<class MemorySpace>
PetscErrorCode VecGetKokkosView_Private(Vec v,PetscScalarKokkosViewType<MemorySpace>* kv,PetscBool overwrite)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  if (!overwrite) veckok->v_dual.sync<MemorySpace>(); /* If overwrite=true, no need to sync the space, since caller will overwrite the data */
  *kv = veckok->v_dual.view<MemorySpace>();
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode VecRestoreKokkosView_Private(Vec v,PetscScalarKokkosViewType<MemorySpace>* kv,PetscBool overwrite)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  if (overwrite) veckok->v_dual.clear_sync_state(); /* If overwrite=true, clear the old sync state since user forced an overwrite */
  veckok->v_dual.modify<MemorySpace>();
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode VecGetKokkosView(Vec v,ConstPetscScalarKokkosViewType<MemorySpace>* kv)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);
  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  veckok->v_dual.sync<MemorySpace>(); /* Sync the space for caller to read */
  *kv = veckok->v_dual.view<MemorySpace>();
  PetscFunctionReturn(0);
}

/* Function template explicit instantiation */
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView         (Vec,ConstPetscScalarKokkosView*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView         (Vec v,PetscScalarKokkosView* kv) {return VecGetKokkosView_Private(v,kv,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosView     (Vec v,PetscScalarKokkosView* kv) {return VecRestoreKokkosView_Private(v,kv,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosViewWrite    (Vec v,PetscScalarKokkosView* kv) {return VecGetKokkosView_Private(v,kv,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosViewWrite(Vec v,PetscScalarKokkosView* kv) {return VecRestoreKokkosView_Private(v,kv,PETSC_TRUE);}

#if !defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HOST) /* Get host views if the default memory space is not host space */
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView         (Vec,ConstPetscScalarKokkosViewHost*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosView         (Vec v,PetscScalarKokkosViewHost* kv) {return VecGetKokkosView_Private(v,kv,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosView     (Vec v,PetscScalarKokkosViewHost* kv) {return VecRestoreKokkosView_Private(v,kv,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecGetKokkosViewWrite    (Vec v,PetscScalarKokkosViewHost* kv) {return VecGetKokkosView_Private(v,kv,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode VecRestoreKokkosViewWrite(Vec v,PetscScalarKokkosViewHost* kv) {return VecRestoreKokkosView_Private(v,kv,PETSC_TRUE);}
#endif

PetscErrorCode VecSetRandom_SeqKokkos(Vec xin,PetscRandom r)
{
  const PetscInt n = xin->map->n;
  PetscScalar    *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayWrite(xin,&xx)); /* TODO: generate randoms directly on device */
  for (PetscInt i=0; i<n; i++) PetscCall(PetscRandomGetValue(r,&xx[i]));
  PetscCall(VecRestoreArrayWrite(xin,&xx));
  PetscFunctionReturn(0);
}

/* x = |x| */
PetscErrorCode VecAbs_SeqKokkos(Vec xin)
{
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  KokkosBlas::abs(xv,xv);
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* x = 1/x */
PetscErrorCode VecReciprocal_SeqKokkos(Vec xin)
{
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  Kokkos::parallel_for(xv.extent(0),KOKKOS_LAMBDA(const int64_t i) {if (xv(i) != (PetscScalar)0.0) xv(i) = (PetscScalar)1.0/xv(i);});
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_SeqKokkos(Vec xin,PetscInt *p,PetscReal *val)
{
  typedef Kokkos::MinLoc<PetscReal,PetscInt>::value_type MinLocValue_t;
  ConstPetscScalarKokkosView xv;
  MinLocValue_t              minloc;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  Kokkos::parallel_reduce("VecMin",xin->map->n,KOKKOS_LAMBDA(PetscInt i,MinLocValue_t& lminloc) {
    if (PetscRealPart(xv(i)) < lminloc.val) {
      lminloc.val = PetscRealPart(xv(i));
      lminloc.loc = i;
    }
  },Kokkos::MinLoc<PetscReal,PetscInt>(minloc)); /* Kokkos will set minloc properly even if xin is zero-lengthed */
  if (p) *p = minloc.loc;
  *val = minloc.val;
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_SeqKokkos(Vec xin,PetscInt *p,PetscReal *val)
{
  typedef Kokkos::MaxLoc<PetscReal,PetscInt>::value_type MaxLocValue_t;
  ConstPetscScalarKokkosView xv;
  MaxLocValue_t              maxloc;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  Kokkos::parallel_reduce("VecMax",xin->map->n,KOKKOS_LAMBDA(PetscInt i,MaxLocValue_t& lmaxloc) {
    if (PetscRealPart(xv(i)) > lmaxloc.val) {
      lmaxloc.val = PetscRealPart(xv(i));
      lmaxloc.loc = i;
    }
  },Kokkos::MaxLoc<PetscReal,PetscInt>(maxloc));
  if (p) *p = maxloc.loc;
  *val = maxloc.val;
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

PetscErrorCode VecSum_SeqKokkos(Vec xin,PetscScalar* sum)
{
  ConstPetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  *sum = KokkosBlas::sum(xv);
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

PetscErrorCode VecShift_SeqKokkos(Vec xin,PetscScalar shift)
{
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  Kokkos::parallel_for("VecShift",xin->map->n,KOKKOS_LAMBDA(PetscInt i) {xv(i) += shift;});
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* y = alpha x + y */
PetscErrorCode VecAXPY_SeqKokkos(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscBool                  xiskok,yiskok;
  PetscScalarKokkosView      yv;
  ConstPetscScalarKokkosView xv;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(0);
  if (yin == xin) {
    PetscCall(VecScale_SeqKokkos(yin,alpha+1));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscObjectTypeCompareAny((PetscObject)xin,&xiskok,VECSEQKOKKOS,VECMPIKOKKOS,""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)yin,&yiskok,VECSEQKOKKOS,VECMPIKOKKOS,""));
  if (xiskok && yiskok) {
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin,&xv));
    PetscCall(VecGetKokkosView(yin,&yv));
    KokkosBlas::axpy(alpha,xv,yv);
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(VecRestoreKokkosView(yin,&yv));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(2.0*yin->map->n));
  } else {
    PetscCall(VecAXPY_Seq(yin,alpha,xin));
  }
  PetscFunctionReturn(0);
}

/* y = x + beta y */
PetscErrorCode VecAYPX_SeqKokkos(Vec yin,PetscScalar beta,Vec xin)
{
  PetscFunctionBegin;
  /* One needs to define KOKKOSBLAS_OPTIMIZATION_LEVEL_AXPBY > 2 to have optimizations for cases alpha/beta = 0,+/-1 */
  PetscCall(VecAXPBY_SeqKokkos(yin,1.0,beta,xin));
  PetscFunctionReturn(0);
}

/* z = y^T x */
PetscErrorCode VecTDot_SeqKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  ConstPetscScalarKokkosView xv,yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  PetscCall(VecGetKokkosView(yin,&yv));
  Kokkos::parallel_reduce("VecTDot",xin->map->n,KOKKOS_LAMBDA(int64_t i, PetscScalar& update) {
    update += yv(i)*xv(i);
  },*z); /* Kokkos always overwrites z, so no need to init it */
  PetscCall(VecRestoreKokkosView(yin,&yv));
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  if (xin->map->n > 0) PetscCall(PetscLogGpuFlops(2.0*xin->map->n));
  PetscFunctionReturn(0);
}

struct TransposeDotTag {};
struct ConjugateDotTag {};

struct MDotFunctor {
  /* Note the C++ notation for an array typedef */
  typedef PetscScalar value_type[];
  typedef ConstPetscScalarKokkosView::size_type size_type;

  /* Tell Kokkos the result array's number of entries. This must be a public value in the functor */
  const size_type              value_count;
  ConstPetscScalarKokkosView   xv,yv[8];

  MDotFunctor(ConstPetscScalarKokkosView& xv,
              const PetscInt ny, /* Number of valid entries in yv[8]. 1 <= ny <= 8 */
              ConstPetscScalarKokkosView& yv0, ConstPetscScalarKokkosView& yv1,
              ConstPetscScalarKokkosView& yv2, ConstPetscScalarKokkosView& yv3,
              ConstPetscScalarKokkosView& yv4, ConstPetscScalarKokkosView& yv5,
              ConstPetscScalarKokkosView& yv6, ConstPetscScalarKokkosView& yv7)
    : value_count(ny),xv(xv)
  {
    yv[0] = yv0; yv[1] = yv1;
    yv[2] = yv2; yv[3] = yv3;
    yv[4] = yv4; yv[5] = yv5;
    yv[6] = yv6; yv[7] = yv7;
  }

  KOKKOS_INLINE_FUNCTION void operator() (TransposeDotTag,const size_type i,value_type sum) const
  {
    PetscScalar xval = xv(i);
    for (size_type j = 0; j<value_count; ++j) sum[j] += yv[j](i)*xval;
  }

  KOKKOS_INLINE_FUNCTION void operator() (ConjugateDotTag,const size_type i,value_type sum) const
  {
    PetscScalar xval = xv(i);
    for (size_type j = 0; j<value_count; ++j) sum[j] += PetscConj(yv[j](i))*xval;
  }

  KOKKOS_INLINE_FUNCTION void join (volatile value_type dst,const volatile value_type src) const
  {
    for (size_type j = 0; j<value_count; j++) dst[j] += src[j];
  }

  KOKKOS_INLINE_FUNCTION void init (value_type sum) const
  {
    for (size_type j = 0; j<value_count; j++) sum[j] = 0.0;
  }
};

template<class WorkTag>
PetscErrorCode VecMultiDot_Private(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscInt                        i,j,cur=0,ngroup=nv/8,rem=nv%8,N=xin->map->n;
  ConstPetscScalarKokkosView      xv,yv[8];
  PetscScalarKokkosViewHost       zv(z,nv);

  PetscFunctionBegin;
  PetscCall(VecGetKokkosView(xin,&xv));
  for (i=0; i<ngroup; i++) { /* 8 y's per group */
    for (j=0; j<8; j++) PetscCall(VecGetKokkosView(yin[cur+j],&yv[j]));
    MDotFunctor mdot(xv,8,yv[0],yv[1],yv[2],yv[3],yv[4],yv[5],yv[6],yv[7]); /* Hope Kokkos make it asynchronous */
    Kokkos::parallel_reduce(Kokkos::RangePolicy<WorkTag>(0,N),mdot,Kokkos::subview(zv,Kokkos::pair<PetscInt,PetscInt>(cur,cur+8)));
    for (j=0; j<8; j++) PetscCall(VecRestoreKokkosView(yin[cur+j],&yv[j]));
    cur += 8;
  }

  if (rem) { /* The remaining */
    for (j=0; j<rem; j++) PetscCall(VecGetKokkosView(yin[cur+j],&yv[j]));
    MDotFunctor mdot(xv,rem,yv[0],yv[1],yv[2],yv[3],yv[4],yv[5],yv[6],yv[7]);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<WorkTag>(0,N),mdot,Kokkos::subview(zv,Kokkos::pair<PetscInt,PetscInt>(cur,cur+rem)));
    for (j=0; j<rem; j++) PetscCall(VecRestoreKokkosView(yin[cur+j],&yv[j]));
  }
  PetscCall(VecRestoreKokkosView(xin,&xv));
  Kokkos::fence(); /* If reduce is async, then we need this fence to make sure z is ready for use on host */
  PetscFunctionReturn(0);
}

/* z[i] = (x,y_i) = y_i^H x */
PetscErrorCode VecMDot_SeqKokkos(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMultiDot_Private<ConjugateDotTag>(xin,nv,yin,z));
  PetscFunctionReturn(0);
}

/* z[i] = (x,y_i) = y_i^T x */
PetscErrorCode VecMTDot_SeqKokkos(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecMultiDot_Private<TransposeDotTag>(xin,nv,yin,z));
  PetscFunctionReturn(0);
}

/* x[:] = alpha */
PetscErrorCode VecSet_SeqKokkos(Vec xin,PetscScalar alpha)
{
  PetscScalarKokkosView     xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosViewWrite(xin,&xv));
  KokkosBlas::fill(xv,alpha);
  PetscCall(VecRestoreKokkosViewWrite(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* x = alpha x */
PetscErrorCode VecScale_SeqKokkos(Vec xin,PetscScalar alpha)
{
  PetscScalarKokkosView     xv;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecSet_SeqKokkos(xin,alpha));
  } else if (alpha != (PetscScalar)1.0) {
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin,&xv));
    KokkosBlas::scal(xv,alpha,xv);
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(xin->map->n));
  }
  PetscFunctionReturn(0);
}

/* z = y^H x */
PetscErrorCode VecDot_SeqKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  ConstPetscScalarKokkosView   xv,yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  PetscCall(VecGetKokkosView(yin,&yv));
  *z = KokkosBlas::dot(yv,xv); /* KokkosBlas::dot(a,b) takes conjugate of a */
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(VecRestoreKokkosView(yin,&yv));
  PetscCall(PetscLogGpuTimeEnd());
  if (xin->map->n > 0) PetscCall(PetscLogGpuFlops(2.0*xin->map->n-1));
  PetscFunctionReturn(0);
}

/* y = x, where x is VECKOKKOS, but y may be not */
PetscErrorCode VecCopy_SeqKokkos(Vec xin,Vec yin)
{
  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  if (xin != yin) {
    Vec_Kokkos *xkok = static_cast<Vec_Kokkos*>(xin->spptr);
    if (yin->offloadmask == PETSC_OFFLOAD_KOKKOS) {
      /* y is also a VecKokkos */
      Vec_Kokkos *ykok = static_cast<Vec_Kokkos*>(yin->spptr);
      /* Kokkos rule: if x's host has newer data, it will copy to y's host view; otherwise to y's device view
        In case x's host is newer, y's device is newer, it will error (though should not, I think). So we just
        clear y's sync state.
       */
      ykok->v_dual.clear_sync_state();
      Kokkos::deep_copy(ykok->v_dual,xkok->v_dual);
    } else {
      PetscScalar *yarray;
      PetscCall(VecGetArrayWrite(yin,&yarray));
      PetscScalarKokkosViewHost yv(yarray,yin->map->n);
      if (xkok->v_dual.need_sync_host()) { /* x's device has newer data */
        Kokkos::deep_copy(yv,xkok->v_dual.view_device());
      } else {
        Kokkos::deep_copy(yv,xkok->v_dual.view_host());
      }
      PetscCall(VecRestoreArrayWrite(yin,&yarray));
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* y[i] <--> x[i] */
PetscErrorCode VecSwap_SeqKokkos(Vec xin,Vec yin)
{
  PetscScalarKokkosView           xv,yv;

  PetscFunctionBegin;
  if (xin != yin) {
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin,&xv));
    PetscCall(VecGetKokkosView(yin,&yv));
    Kokkos::parallel_for(xin->map->n,KOKKOS_LAMBDA(const int64_t i) {
      PetscScalar tmp = xv(i);
      xv(i) = yv(i);
      yv(i) = tmp;
    });
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(VecRestoreKokkosView(yin,&yv));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscFunctionReturn(0);
}

/*  w = alpha x + y */
PetscErrorCode VecWAXPY_SeqKokkos(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  ConstPetscScalarKokkosView      xv,yv;
  PetscScalarKokkosView           wv;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecCopy_SeqKokkos(yin,win));
  } else {
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosViewWrite(win,&wv));
    PetscCall(VecGetKokkosView(xin,&xv));
    PetscCall(VecGetKokkosView(yin,&yv));
    Kokkos::parallel_for(win->map->n,KOKKOS_LAMBDA(const int64_t i) {wv(i) = alpha*xv(i) + yv(i);});
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(VecRestoreKokkosView(yin,&yv));
    PetscCall(VecRestoreKokkosViewWrite(win,&wv));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(2*win->map->n));
  }
  PetscFunctionReturn(0);
}

struct MAXPYFunctor {
  typedef ConstPetscScalarKokkosView::size_type size_type;

  PetscScalarKokkosView         yv;
  PetscInt                      nx;   /* Significent entries in a[8] and xv[8] */
  PetscScalar                   a[8];
  ConstPetscScalarKokkosView    xv[8];

  MAXPYFunctor(PetscScalarKokkosView yv,
               PetscInt nx,
               PetscScalar a0,PetscScalar a1,PetscScalar a2,PetscScalar a3,
               PetscScalar a4,PetscScalar a5,PetscScalar a6,PetscScalar a7,
               ConstPetscScalarKokkosView xv0, ConstPetscScalarKokkosView xv1,
               ConstPetscScalarKokkosView xv2, ConstPetscScalarKokkosView xv3,
               ConstPetscScalarKokkosView xv4, ConstPetscScalarKokkosView xv5,
               ConstPetscScalarKokkosView xv6, ConstPetscScalarKokkosView xv7)
    : yv(yv),nx(nx)
  {
    a[0]  = a0;  a[1]  = a1;
    a[2]  = a2;  a[3]  = a3;
    a[4]  = a4;  a[5]  = a5;
    a[6]  = a6;  a[7]  = a7;
    xv[0] = xv0; xv[1] = xv1;
    xv[2] = xv2; xv[3] = xv3;
    xv[4] = xv4; xv[5] = xv5;
    xv[6] = xv6; xv[7] = xv7;
  }

  KOKKOS_INLINE_FUNCTION void operator() (const size_type i) const
  {
    for (PetscInt j = 0; j<nx; ++j) yv(i) += a[j]*xv[j](i);
  }
};

/*  y = y + sum alpha[i] x[i] */
PetscErrorCode VecMAXPY_SeqKokkos(Vec yin, PetscInt nv,const PetscScalar *alpha,Vec *xin)
{
  PetscInt                        i,j,cur=0,ngroup=nv/8,rem=nv%8;
  PetscScalarKokkosView           yv;
  PetscScalar                     a[8];
  ConstPetscScalarKokkosView      xv[8];

  PetscFunctionBegin;
  PetscCall(VecGetKokkosView(yin,&yv));
  for (i=0; i<ngroup; i++) { /* 8 x's per group */
    for (j=0; j<8; j++) { /* Fill the parameters */
      a[j] = alpha[cur+j];
      PetscCall(VecGetKokkosView(xin[cur+j],&xv[j]));
    }
    MAXPYFunctor maxpy(yv,8,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],xv[0],xv[1],xv[2],xv[3],xv[4],xv[5],xv[6],xv[7]);
    Kokkos::parallel_for(yin->map->n,maxpy);
    for (j=0; j<8; j++) PetscCall(VecRestoreKokkosView(xin[cur+j],&xv[j]));
    cur += 8;
  }

  if (rem) { /* The remaining */
    for (j=0; j<rem; j++) {
      a[j] = alpha[cur+j];
      PetscCall(VecGetKokkosView(xin[cur+j],&xv[j]));
    }
    MAXPYFunctor maxpy(yv,rem,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],xv[0],xv[1],xv[2],xv[3],xv[4],xv[5],xv[6],xv[7]);
    Kokkos::parallel_for(yin->map->n,maxpy);
    for (j=0; j<rem; j++) PetscCall(VecRestoreKokkosView(xin[cur+j],&xv[j]));
  }
  PetscCall(VecRestoreKokkosView(yin,&yv));
  PetscCall(PetscLogGpuFlops(nv*2.0*yin->map->n));
  PetscFunctionReturn(0);
}

/* y = alpha x + beta y */
PetscErrorCode VecAXPBY_SeqKokkos(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  ConstPetscScalarKokkosView   xv;
  PetscScalarKokkosView        yv;
  PetscBool                    xiskok,yiskok;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)xin,&xiskok,VECSEQKOKKOS,VECMPIKOKKOS,""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)yin,&yiskok,VECSEQKOKKOS,VECMPIKOKKOS,""));
  if (xiskok && yiskok) {
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin,&xv));
    PetscCall(VecGetKokkosView(yin,&yv));
    KokkosBlas::axpby(alpha,xv,beta,yv);
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(VecRestoreKokkosView(yin,&yv));
    PetscCall(PetscLogGpuTimeEnd());
    if (alpha == (PetscScalar)0.0 || beta == (PetscScalar)0.0) {
      PetscCall(PetscLogGpuFlops(xin->map->n));
    } else if (beta == (PetscScalar)1.0 || alpha == (PetscScalar)1.0) {
      PetscCall(PetscLogGpuFlops(2.0*xin->map->n));
    } else {
      PetscCall(PetscLogGpuFlops(3.0*xin->map->n));
    }
  } else {
    PetscCall(VecAXPBY_Seq(yin,alpha,beta,xin));
  }
  PetscFunctionReturn(0);
}

/* z = alpha x + beta y + gamma z */
PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  ConstPetscScalarKokkosView    xv,yv;
  PetscScalarKokkosView         zv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(zin,&zv));
  PetscCall(VecGetKokkosView(xin,&xv));
  PetscCall(VecGetKokkosView(yin,&yv));
  KokkosBlas::update(alpha,xv,beta,yv,gamma,zv);
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(VecRestoreKokkosView(yin,&yv));
  PetscCall(VecRestoreKokkosView(zin,&zv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(zin->map->n*5));
  PetscFunctionReturn(0);
}

/* w = x*y. Any subset of the x, y, and w may be the same vector.

  w is of type VecKokkos, but x, y may be not.
*/
PetscErrorCode VecPointwiseMult_SeqKokkos(Vec win,Vec xin,Vec yin)
{
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetLocalSize(win,&n));
  if (xin->offloadmask != PETSC_OFFLOAD_KOKKOS || yin->offloadmask != PETSC_OFFLOAD_KOKKOS) {
    PetscScalarKokkosViewHost  wv;
    const PetscScalar          *xp,*yp;
    PetscCall(VecGetArrayRead(xin,&xp));
    PetscCall(VecGetArrayRead(yin,&yp));
    PetscCall(VecGetKokkosViewWrite(win,&wv));

    ConstPetscScalarKokkosViewHost xv(xp,n),yv(yp,n);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,n),KOKKOS_LAMBDA(const PetscInt i) {wv(i) = xv(i)*yv(i);});

    PetscCall(VecRestoreArrayRead(xin,&xp));
    PetscCall(VecRestoreArrayRead(yin,&yp));
    PetscCall(VecRestoreKokkosViewWrite(win,&wv));
  } else {
    ConstPetscScalarKokkosView      xv,yv;
    PetscScalarKokkosView           wv;

    PetscCall(VecGetKokkosViewWrite(win,&wv));
    PetscCall(VecGetKokkosView(xin,&xv));
    PetscCall(VecGetKokkosView(yin,&yv));
    Kokkos::parallel_for(n,KOKKOS_LAMBDA(const PetscInt i) {wv(i) = xv(i)*yv(i);});
    PetscCall(VecRestoreKokkosView(yin,&yv));
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(VecRestoreKokkosViewWrite(win,&wv));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

/* w = x/y */
PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec win,Vec xin,Vec yin)
{
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetLocalSize(win,&n));
  if (xin->offloadmask != PETSC_OFFLOAD_KOKKOS || yin->offloadmask != PETSC_OFFLOAD_KOKKOS) {
    PetscScalarKokkosViewHost  wv;
    const PetscScalar          *xp,*yp;
    PetscCall(VecGetArrayRead(xin,&xp));
    PetscCall(VecGetArrayRead(yin,&yp));
    PetscCall(VecGetKokkosViewWrite(win,&wv));

    ConstPetscScalarKokkosViewHost xv(xp,n),yv(yp,n);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,n),KOKKOS_LAMBDA(const PetscInt i) {
      if (yv(i) != 0.0) wv(i) = xv(i)/yv(i);
      else wv(i) = 0.0;
    });

    PetscCall(VecRestoreArrayRead(xin,&xp));
    PetscCall(VecRestoreArrayRead(yin,&yp));
    PetscCall(VecRestoreKokkosViewWrite(win,&wv));
  } else {
    ConstPetscScalarKokkosView      xv,yv;
    PetscScalarKokkosView           wv;

    PetscCall(VecGetKokkosViewWrite(win,&wv));
    PetscCall(VecGetKokkosView(xin,&xv));
    PetscCall(VecGetKokkosView(yin,&yv));
    Kokkos::parallel_for(n,KOKKOS_LAMBDA(const PetscInt i) {
      if (yv(i) != 0.0) wv(i) = xv(i)/yv(i);
      else wv(i) = 0.0;
    });
    PetscCall(VecRestoreKokkosView(yin,&yv));
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(VecRestoreKokkosViewWrite(win,&wv));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(win->map->n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_SeqKokkos(Vec xin,NormType type,PetscReal *z)
{
  const PetscInt                n = xin->map->n;
  ConstPetscScalarKokkosView    xv;

  PetscFunctionBegin;
  if (type == NORM_1_AND_2) {
    PetscCall(VecNorm_SeqKokkos(xin,NORM_1,z));
    PetscCall(VecNorm_SeqKokkos(xin,NORM_2,z+1));
  } else {
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(VecGetKokkosView(xin,&xv));
    if (type == NORM_2 || type == NORM_FROBENIUS) {
      *z   = KokkosBlas::nrm2(xv);
      PetscCall(PetscLogGpuFlops(PetscMax(2.0*n-1,0.0)));
    } else if (type == NORM_1) {
      *z   = KokkosBlas::nrm1(xv);
      PetscCall(PetscLogGpuFlops(PetscMax(n-1.0,0.0)));
    } else if (type == NORM_INFINITY) {
      *z = KokkosBlas::nrminf(xv);
    }
    PetscCall(VecRestoreKokkosView(xin,&xv));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscFunctionReturn(0);
}

/* A functor for DotNorm2 so that we can compute dp and nm in one kernel */
struct DotNorm2 {
  typedef PetscScalar                            value_type[];
  typedef ConstPetscScalarKokkosView::size_type  size_type;

  size_type                    value_count;
  ConstPetscScalarKokkosView   xv_, yv_; /* first and second vectors in VecDotNorm2. The order matters. */

  DotNorm2(ConstPetscScalarKokkosView& xv,ConstPetscScalarKokkosView& yv) :
    value_count(2), xv_(xv), yv_(yv) {}

  KOKKOS_INLINE_FUNCTION void operator() (const size_type i, value_type result) const
  {
    result[0] += PetscConj(yv_(i))*xv_(i);
    result[1] += PetscConj(yv_(i))*yv_(i);
  }

  KOKKOS_INLINE_FUNCTION void join (volatile value_type dst, const volatile value_type src) const
  {
    dst[0] += src[0];
    dst[1] += src[1];
  }

  KOKKOS_INLINE_FUNCTION void init (value_type result) const
  {
    result[0] = 0.0;
    result[1] = 0.0;
  }
};

/* dp = y^H x, nm = y^H y */
PetscErrorCode VecDotNorm2_SeqKokkos(Vec xin, Vec yin, PetscScalar *dp, PetscScalar *nm)
{
  ConstPetscScalarKokkosView      xv,yv;
  PetscScalar                     result[2];

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  PetscCall(VecGetKokkosView(yin,&yv));
  DotNorm2 dn(xv,yv);
  Kokkos::parallel_reduce(xin->map->n,dn,result);
  *dp  = result[0];
  *nm  = result[1];
  PetscCall(VecRestoreKokkosView(yin,&yv));
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(4.0*xin->map->n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecConjugate_SeqKokkos(Vec xin)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalarKokkosView     xv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(xin,&xv));
  Kokkos::parallel_for(xin->map->n,KOKKOS_LAMBDA(int64_t i) {xv(i) = PetscConj(xv(i));});
  PetscCall(VecRestoreKokkosView(xin,&xv));
  PetscCall(PetscLogGpuTimeEnd());
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

/* Temporarily replace the array in vin with a[]. Return to the original array with a call to VecResetArray() */
PetscErrorCode VecPlaceArray_SeqKokkos(Vec vin,const PetscScalar *a)
{
  Vec_Seq        *vecseq = (Vec_Seq*)vin->data;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(vin->spptr);

  PetscFunctionBegin;
  PetscCall(VecPlaceArray_Seq(vin,a));
  veckok->UpdateArray<Kokkos::HostSpace>(vecseq->array);
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqKokkos(Vec vin)
{
  Vec_Seq        *vecseq = (Vec_Seq*)vin->data;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(vin->spptr);

  PetscFunctionBegin;
  veckok->v_dual.sync_host(); /* User wants to unhook the provided host array. Sync it so that user can get the latest */
  PetscCall(VecResetArray_Seq(vin)); /* Swap back the old host array, assuming its has the latest value */
  veckok->UpdateArray<Kokkos::HostSpace>(vecseq->array);
  PetscFunctionReturn(0);
}

/* Replace the array in vin with a[] that must be allocated by PetscMalloc. a[] is owned by vin afterwords. */
PetscErrorCode VecReplaceArray_SeqKokkos(Vec vin,const PetscScalar *a)
{
  Vec_Seq        *vecseq = (Vec_Seq*)vin->data;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(vin->spptr);

  PetscFunctionBegin;
  /* Make sure the users array has the latest values */
  if (vecseq->array != vecseq->array_allocated) veckok->v_dual.sync_host();
  PetscCall(VecReplaceArray_Seq(vin,a));
  veckok->UpdateArray<Kokkos::HostSpace>(vecseq->array);
  PetscFunctionReturn(0);
}

/* Maps the local portion of vector v into vector w */
PetscErrorCode VecGetLocalVector_SeqKokkos(Vec v,Vec w)
{
  Vec_Seq          *vecseq = static_cast<Vec_Seq*>(w->data);
  Vec_Kokkos       *veckok = static_cast<Vec_Kokkos*>(w->spptr);

  PetscFunctionBegin;
  PetscCheckTypeName(w,VECSEQKOKKOS);
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
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec v,Vec w)
{
  PetscFunctionBegin;
  PetscCheckTypeName(w,VECSEQKOKKOS);
  v->data  = w->data;
  v->spptr = w->spptr;
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  /* TODO: need to think if setting w->data/spptr to NULL is safe */
  w->data  = NULL;
  w->spptr = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArray_SeqKokkos(Vec v,PetscScalar **a)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  veckok->v_dual.sync_host();
  *a = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArray_SeqKokkos(Vec v,PetscScalar **a)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  veckok->v_dual.modify_host();
  PetscFunctionReturn(0);
}

/* Get array on host to overwrite, so no need to sync host. In VecRestoreArrayWrite() we will mark host is modified. */
PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec v,PetscScalar **a)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  veckok->v_dual.clear_sync_state();
  *a = veckok->v_dual.view_host().data();
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayAndMemType_SeqKokkos(Vec v,PetscScalar** a,PetscMemType *mtype)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) {
    *a = veckok->v_dual.view_host().data();
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  } else {
    /* When there is device, we always return up-to-date device data */
    veckok->v_dual.sync_device();
    *a = veckok->v_dual.view_device().data();
    if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayAndMemType_SeqKokkos(Vec v,PetscScalar** a)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) {
    veckok->v_dual.modify_host();
  } else {
    veckok->v_dual.modify_device();
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWriteAndMemType_SeqKokkos(Vec v,PetscScalar** a,PetscMemType *mtype)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) {
    *a = veckok->v_dual.view_host().data();
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  } else {
    /* When there is device, we always return device data (but no need to sync the device) */
    veckok->v_dual.clear_sync_state(); /* So that in restore, we can safely modify_device() */
    *a = veckok->v_dual.view_device().data();
    if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  }
  PetscFunctionReturn(0);
}

/* Copy xin's sync state to y */
static PetscErrorCode VecCopySyncState_Kokkos_Private(Vec xin,Vec yout)
{
  Vec_Kokkos   *xkok = static_cast<Vec_Kokkos*>(xin->spptr);
  Vec_Kokkos   *ykok = static_cast<Vec_Kokkos*>(yout->spptr);

  PetscFunctionBegin;
  ykok->v_dual.clear_sync_state();
  if (xkok->v_dual.need_sync_host()) {
    ykok->v_dual.modify_device();
  } else if (xkok->v_dual.need_sync_device()) {
    ykok->v_dual.modify_host();
  }
  PetscFunctionReturn(0);
}

/* Interal routine shared by VecGetSubVector_{SeqKokkos,MPIKokkos} */
PetscErrorCode VecGetSubVector_Kokkos_Private(Vec x,PetscBool xIsMPI,IS is,Vec *y)
{
  PetscBool      contig;
  PetscInt       n,N,start,bs;
  MPI_Comm       comm;
  Vec            z;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)x,&comm));
  PetscCall(ISGetLocalSize(is,&n));
  PetscCall(ISGetSize(is,&N));
  PetscCall(VecGetSubVectorContiguityAndBS_Private(x,is,&contig,&start,&bs));

  if (contig) { /* We can do a no-copy (in-place) implementation with y sharing x's arrays */
    Vec_Kokkos        *xkok = static_cast<Vec_Kokkos*>(x->spptr);
    const PetscScalar *array_h = xkok->v_dual.view_host().data() + start;
    const PetscScalar *array_d = xkok->v_dual.view_device().data() + start;

    /* These calls assume the input arrays are synced */
    if (xIsMPI) PetscCall(VecCreateMPIKokkosWithArrays_Private(comm,bs,n,N,array_h,array_d,&z)); /* x could be MPI even when x's comm size = 1 */
    else PetscCall(VecCreateSeqKokkosWithArrays_Private(comm,bs,n,array_h,array_d,&z));

    PetscCall(VecCopySyncState_Kokkos_Private(x,z)); /* Copy x's sync state to z */

    /* This is relevant only in debug mode */
    PetscInt state = 0;
    PetscCall(VecLockGet(x,&state));
    if (state) { /* x is either in read or read/write mode, therefore z, overlapped with x, can only be in read mode */
      PetscCall(VecLockReadPush(z));
    }

    z->ops->placearray   = NULL; /* z's arrays can't be replaced, because z does not own them */
    z->ops->replacearray = NULL;

  } else { /* Have to create a VecScatter and a stand-alone vector */
    PetscCall(VecGetSubVectorThroughVecScatter_Private(x,is,bs,&z));
  }
  *y = z;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetSubVector_SeqKokkos(Vec x,IS is,Vec *y)
{
  PetscFunctionBegin;
  PetscCall(VecGetSubVector_Kokkos_Private(x,PETSC_FALSE,is,y));
  PetscFunctionReturn(0);
}

/* Restore subvector y to x */
PetscErrorCode VecRestoreSubVector_SeqKokkos(Vec x,IS is,Vec *y)
{
  VecScatter                    vscat;
  PETSC_UNUSED PetscObjectState dummystate = 0;
  PetscBool                     unchanged;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataGetInt((PetscObject)*y,VecGetSubVectorSavedStateId,dummystate,unchanged));
  if (unchanged) PetscFunctionReturn(0); /* If y's state has not changed since VecGetSubVector(), we only need to destroy it */

  PetscCall(PetscObjectQuery((PetscObject)*y,"VecGetSubVector_Scatter",(PetscObject*)&vscat));
  if (vscat) {
    PetscCall(VecScatterBegin(vscat,*y,x,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(vscat,*y,x,INSERT_VALUES,SCATTER_REVERSE));
  } else { /* y and x's (host and device) arrays overlap */
    Vec_Kokkos *xkok = static_cast<Vec_Kokkos*>(x->spptr);
    Vec_Kokkos *ykok = static_cast<Vec_Kokkos*>((*y)->spptr);
    PetscInt   state;

    PetscCall(VecLockGet(x,&state));
    PetscCheck(!state,PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_WRONGSTATE,"Vec x is locked for read-only or read/write access");

    /* The tricky part: one has to carefully sync the arrays */
    if (xkok->v_dual.need_sync_device()) { /* x's host has newer data */
      ykok->v_dual.sync_host(); /* Move y's latest values to host (since y is just a subset of x) */
    } else if (xkok->v_dual.need_sync_host()) { /* x's device has newer data */
      ykok->v_dual.sync_device(); /* Move y's latest data to device */
    } else { /* x's host and device data is already sync'ed; Copy y's sync state to x */
      PetscCall(VecCopySyncState_Kokkos_Private(*y,x));
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)x)); /* Since x is updated */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetPreallocationCOO_SeqKokkos(Vec x, PetscCount ncoo, const PetscInt coo_i[])
{
  Vec_Seq      *vecseq = static_cast<Vec_Seq*>(x->data);
  Vec_Kokkos   *veckok = static_cast<Vec_Kokkos*>(x->spptr);
  PetscInt     m;

  PetscFunctionBegin;
  PetscCall(VecSetPreallocationCOO_Seq(x,ncoo,coo_i));
  PetscCall(VecGetLocalSize(x,&m));
  PetscCallCXX(veckok->SetUpCOO(vecseq,m));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetValuesCOO_SeqKokkos(Vec x,const PetscScalar v[],InsertMode imode)
{
  Vec_Seq                     *vecseq = static_cast<Vec_Seq*>(x->data);
  Vec_Kokkos                  *veckok = static_cast<Vec_Kokkos*>(x->spptr);
  const PetscCountKokkosView& jmap1 = veckok->jmap1_d;
  const PetscCountKokkosView& perm1 = veckok->perm1_d;
  PetscScalarKokkosView       xv; /* View for vector x */
  ConstPetscScalarKokkosView  vv; /* View for array v[] */
  PetscInt                    m;
  PetscMemType                memtype;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x,&m));
  PetscCall(PetscGetMemType(v,&memtype));
  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
    vv = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),ConstPetscScalarKokkosViewHost(v,vecseq->coo_n));
  } else {
    vv = ConstPetscScalarKokkosView(v,vecseq->coo_n); /* Directly use v[]'s memory */
  }

  if (imode == INSERT_VALUES) PetscCall(VecGetKokkosViewWrite(x,&xv)); /* write vector */
  else PetscCall(VecGetKokkosView(x,&xv)); /* read & write vector */

  Kokkos::parallel_for(m,KOKKOS_LAMBDA(const PetscCount i) {
    PetscScalar sum = 0.0;
    for (PetscCount k=jmap1(i); k<jmap1(i+1); k++) sum += vv(perm1(k));
    xv(i) = (imode == INSERT_VALUES? 0.0 : xv(i)) + sum;
  });

  if (imode == INSERT_VALUES) PetscCall(VecRestoreKokkosViewWrite(x,&xv));
  else PetscCall(VecRestoreKokkosView(x,&xv));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetOps_SeqKokkos(Vec v)
{
  PetscFunctionBegin;
  v->ops->abs                    = VecAbs_SeqKokkos;
  v->ops->reciprocal             = VecReciprocal_SeqKokkos;
  v->ops->pointwisemult          = VecPointwiseMult_SeqKokkos;
  v->ops->min                    = VecMin_SeqKokkos;
  v->ops->max                    = VecMax_SeqKokkos;
  v->ops->sum                    = VecSum_SeqKokkos;
  v->ops->shift                  = VecShift_SeqKokkos;
  v->ops->norm                   = VecNorm_SeqKokkos;
  v->ops->scale                  = VecScale_SeqKokkos;
  v->ops->copy                   = VecCopy_SeqKokkos;
  v->ops->set                    = VecSet_SeqKokkos;
  v->ops->swap                   = VecSwap_SeqKokkos;
  v->ops->axpy                   = VecAXPY_SeqKokkos;
  v->ops->axpby                  = VecAXPBY_SeqKokkos;
  v->ops->axpbypcz               = VecAXPBYPCZ_SeqKokkos;
  v->ops->pointwisedivide        = VecPointwiseDivide_SeqKokkos;
  v->ops->setrandom              = VecSetRandom_SeqKokkos;

  v->ops->dot                    = VecDot_SeqKokkos;
  v->ops->tdot                   = VecTDot_SeqKokkos;
  v->ops->mdot                   = VecMDot_SeqKokkos;
  v->ops->mtdot                  = VecMTDot_SeqKokkos;

  v->ops->dot_local              = VecDot_SeqKokkos;
  v->ops->tdot_local             = VecTDot_SeqKokkos;
  v->ops->mdot_local             = VecMDot_SeqKokkos;
  v->ops->mtdot_local            = VecMTDot_SeqKokkos;

  v->ops->norm_local             = VecNorm_SeqKokkos;
  v->ops->maxpy                  = VecMAXPY_SeqKokkos;
  v->ops->aypx                   = VecAYPX_SeqKokkos;
  v->ops->waxpy                  = VecWAXPY_SeqKokkos;
  v->ops->dotnorm2               = VecDotNorm2_SeqKokkos;
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

  v->ops->getarrayandmemtype     = VecGetArrayAndMemType_SeqKokkos;
  v->ops->restorearrayandmemtype = VecRestoreArrayAndMemType_SeqKokkos;
  v->ops->getarraywriteandmemtype= VecGetArrayWriteAndMemType_SeqKokkos;
  v->ops->getsubvector           = VecGetSubVector_SeqKokkos;
  v->ops->restoresubvector       = VecRestoreSubVector_SeqKokkos;

  v->ops->setpreallocationcoo    = VecSetPreallocationCOO_SeqKokkos;
  v->ops->setvaluescoo           = VecSetValuesCOO_SeqKokkos;
  PetscFunctionReturn(0);
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
  Vec_Seq        *vecseq;
  Vec_Kokkos     *veckok;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscLayoutSetUp(v->map));
  PetscCall(VecCreate_Seq(v));  /* Build a sequential vector, allocate array */
  PetscCall(PetscObjectChangeTypeName((PetscObject)v,VECSEQKOKKOS));
  PetscCall(VecSetOps_SeqKokkos(v));

  PetscCheck(!v->spptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"v->spptr not NULL");
  vecseq   = static_cast<Vec_Seq*>(v->data);
  veckok   = new Vec_Kokkos(v->map->n,vecseq->array,NULL); /* Let host claim it has the latest data (zero) */
  v->spptr = static_cast<void*>(veckok);
  v->offloadmask = PETSC_OFFLOAD_KOKKOS;
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqKokkosWithArray - Creates a Kokkos sequential array-style vector,
   where the user provides the array space to store the vector values. The array
   provided must be a device array.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  array - device memory where the vector elements are to be stored.

   Output Parameter:
.  v - the vector

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
PetscErrorCode  VecCreateSeqKokkosWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar darray[],Vec *v)
{
  PetscMPIInt    size;
  Vec            w;
  Vec_Kokkos     *veckok = NULL;
  PetscScalar    *harray;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCheck(size <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQKOKKOS on more than one process");

  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(VecCreate(comm,&w));
  PetscCall(VecSetSizes(w,n,n));
  PetscCall(VecSetBlockSize(w,bs));
  if (!darray) { /* Allocate memory ourself if user provided NULL */
    PetscCall(VecSetType(w,VECSEQKOKKOS));
  } else {
    /* Build a VECSEQ, get its harray, and then build Vec_Kokkos along with darray */
    if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) {
      harray = const_cast<PetscScalar*>(darray);
      PetscCall(VecCreate_Seq_Private(w,harray)); /* Build a sequential vector with harray */
    } else {
      PetscCall(VecSetType(w,VECSEQ));
      harray = static_cast<Vec_Seq*>(w->data)->array;
    }
    PetscCall(PetscObjectChangeTypeName((PetscObject)w,VECSEQKOKKOS)); /* Change it to Kokkos */
    PetscCall(VecSetOps_SeqKokkos(w));
    veckok = new Vec_Kokkos(n,harray,const_cast<PetscScalar*>(darray));
    veckok->v_dual.modify_device(); /* Mark the device is modified */
    w->offloadmask = PETSC_OFFLOAD_KOKKOS;
    w->spptr = static_cast<void*>(veckok);
  }
  *v       = w;
  PetscFunctionReturn(0);
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
PetscErrorCode  VecCreateSeqKokkosWithArrays_Private(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar harray[],const PetscScalar darray[],Vec *v)
{
  PetscMPIInt size;
  Vec         w;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCheck(size <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQKOKKOS on more than one process");
  if (n) {
    PetscValidScalarPointer(harray,4);
    PetscCheck(darray,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"darray cannot be NULL");
  }
  if (std::is_same<DefaultMemorySpace,Kokkos::HostSpace>::value) PetscCheck(harray == darray,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"harray and darray must be the same");

  PetscCall(VecCreateSeqWithArray(comm,bs,n,harray,&w));
  PetscCall(PetscObjectChangeTypeName((PetscObject)w,VECSEQKOKKOS)); /* Change it to Kokkos */
  PetscCall(VecSetOps_SeqKokkos(w));
  PetscCallCXX(w->spptr = new Vec_Kokkos(n,const_cast<PetscScalar*>(harray),const_cast<PetscScalar*>(darray)));
  w->offloadmask = PETSC_OFFLOAD_KOKKOS;
  *v = w;
  PetscFunctionReturn(0);
}

/* TODO: ftn-auto generates veckok.kokkosf.c */
/*@C
 VecCreateSeqKokkos - Creates a standard, sequential array-style vector.

 Collective

 Input Parameter:
 +  comm - the communicator, should be PETSC_COMM_SELF
 -  n - the vector length

 Output Parameter:
 .  v - the vector

 Notes:
 Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
 same type as an existing vector.

 Level: intermediate

 .seealso: `VecCreateMPI()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`
 @*/
PetscErrorCode VecCreateSeqKokkos(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(VecCreate(comm,v));
  PetscCall(VecSetSizes(*v,n,n));
  PetscCall(VecSetType(*v,VECSEQKOKKOS)); /* Calls VecCreate_SeqKokkos */
  PetscFunctionReturn(0);
}

/* Duplicate layout etc but not the values in the input vector */
PetscErrorCode VecDuplicate_SeqKokkos(Vec win,Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecDuplicate_Seq(win,v)); /* It also dups ops of win */
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqKokkos(Vec v)
{
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);
  Vec_Seq        *vecseq = static_cast<Vec_Seq*>(v->data);

  PetscFunctionBegin;
  delete veckok;
  v->spptr = NULL;
  if (vecseq) PetscCall(VecDestroy_Seq(v));
  PetscFunctionReturn(0);
}
