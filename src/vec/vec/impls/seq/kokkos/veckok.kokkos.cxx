/*
   Implements the sequential Kokkos vectors.
*/
#include "Kokkos_Core_fwd.hpp"
#include "Kokkos_Macros.hpp"
#include "Kokkos_Parallel.hpp"
#include "Kokkos_Parallel_Reduce.hpp"
#include <petsc/private/sfimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscmath.h>
#include <petscviewer.h>
#include <KokkosBlas.hpp>

#include <petscconf.h>
#include <petscvec.hpp>
#include <petscerror.h>
#include <../src/vec/vec/impls/dvecimpl.h> /* for VecCreate_Seq_Private */
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>

#if defined(PETSC_USE_DEBUG)
  #define VecErrorIfNotKokkos(v) \
    do {                     \
      PetscErrorCode   ierr; \
      PetscBool        isKokkos = PETSC_FALSE; \
      ierr = PetscObjectTypeCompareAny((PetscObject)(v),&isKokkos,VECSEQKOKKOS,VECMPIKOKKOS,VECKOKKOS,"");CHKERRQ(ierr); \
      if (!isKokkos) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Calling VECKOKKOS methods on a non-VECKOKKOS object"); \
    } while (0)
#else
  #define VecErrorIfNotKokkos(v) do {(void)(v);} while (0)
#endif

PetscErrorCode VecKokkosSyncHost(Vec v)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  veckok->dual_v.sync_host();
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosModifyHost(Vec v)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  veckok->dual_v.modify_host();
  PetscFunctionReturn(0);
}

/* TODO: Is VecGetKokkosDeviceView() a better name, if we always use -vec_type kokkos */
PetscErrorCode VecKokkosGetDeviceView(Vec v,PetscScalarViewDevice_t* d_view)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  veckok->dual_v.sync_device(); /* User might read and write the device view */
  *d_view = veckok->dual_v.view_device();
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosRestoreDeviceView(Vec v,PetscScalarViewDevice_t* d_view)
{
  PetscErrorCode ierr;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  veckok->dual_v.modify_device();
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosGetDeviceViewRead(Vec v,ConstPetscScalarViewDevice_t* d_view)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  veckok->dual_v.sync_device();
  *d_view = veckok->dual_v.view_device();
  PetscFunctionReturn(0);
}

/* VecKokkosRestoreDeviceViewRead() is defined as a no-op in header */

/* This one will overwrite the device, so no need to sync device.
  But let's say the host is modified. In VecKokkosRestoreDeviceViewWrite(), we have to
  clear_sync_state() to only mark the device as modified. Otherwise we will have a wrong
  state saying both host and device are modified. If xin=yin, we have to support code sequence:

  VecKokkosGetDeviceViewWrite(xin,&xv);
  VecKokkosGetDeviceViewRead(yin,&yv);
  ...
  VecKokkosRestoreDeviceViewWrite(xin,&xv);
  VecKokkosRestoreDeviceViewRead(yin,&yv);
 */
PetscErrorCode VecKokkosGetDeviceViewWrite(Vec v,PetscScalarViewDevice_t* d_view)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  *d_view = veckok->dual_v.view_device();
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosRestoreDeviceViewWrite(Vec v,PetscScalarViewDevice_t* dv)
{
  PetscErrorCode ierr;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  veckok->dual_v.clear_sync_state();
  veckok->dual_v.modify_device();
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosGetArrayInPlace(Vec v,PetscScalar** array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecKokkosGetArrayInPlace_Internal(v,array,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosGetArrayInPlace_Internal(Vec v,PetscScalar** array,PetscMemType *mtype)
{
  Vec_Kokkos  *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  if (veckok->dual_v.need_sync_device()) {
   /* Host has newer data than device */
    *array = veckok->dual_v.view_host().data();
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  } else {
    /* Device has newer or same data as host. We prefer returning devcie data*/
    *array = veckok->dual_v.view_device().data();
    if (mtype) *mtype = PETSC_MEMTYPE_DEVICE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosRestoreArrayInPlace(Vec v,PetscScalar** array)
{
  PetscErrorCode ierr;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  if (veckok->dual_v.need_sync_device()) { /* Host has newer data than device */
    veckok->dual_v.modify_host();
  } else {
    veckok->dual_v.modify_device();
  }
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosGetArrayReadInPlace(Vec v,const PetscScalar** array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecKokkosGetArrayReadInPlace_Internal(v,array,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecKokkosGetArrayReadInPlace_Internal(Vec v,const PetscScalar** array,PetscMemType *mtype)
{
  Vec_Kokkos  *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  VecErrorIfNotKokkos(v);
  if (veckok->dual_v.need_sync_device()) { /* Host has newer data than device */
    *array = veckok->dual_v.view_host().data();
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  } else {
    *array = veckok->dual_v.view_device().data();
    if (mtype) *mtype = PETSC_MEMTYPE_DEVICE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqKokkos(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayWrite(xin,&xx);CHKERRQ(ierr); /* TODO: generate randoms directly on device */
  for (i=0; i<n; i++) { ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr); }
  ierr = VecRestoreArrayWrite(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* x = |x| */
PetscErrorCode VecAbs_SeqKokkos(Vec xin)
{
  PetscErrorCode             ierr;
  PetscScalarViewDevice_t    xv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(xin,&xv);CHKERRQ(ierr);
  KokkosBlas::abs(xv,xv);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* x = 1/x */
PetscErrorCode VecReciprocal_SeqKokkos(Vec xin)
{
  PetscErrorCode             ierr;
  PetscScalarViewDevice_t    xv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(xin,&xv);CHKERRQ(ierr);
  /* KokkosBlas::reciprocal(xv,xv); */
  Kokkos::parallel_for(xv.extent(0),KOKKOS_LAMBDA(const int64_t i) {if (xv(i) != (PetscScalar)0.0) xv(i) = (PetscScalar)1.0/xv(i);});
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_SeqKokkos(Vec xin,PetscInt *p,PetscReal *val)
{
  typedef Kokkos::MinLoc<PetscReal,PetscInt>::value_type MinLocValue_t;

  PetscErrorCode                  ierr;
  ConstPetscScalarViewDevice_t    xv;
  MinLocValue_t                   minloc;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  Kokkos::parallel_reduce("VecMin",xin->map->n,KOKKOS_LAMBDA(PetscInt i,MinLocValue_t& lminloc) {
    if (xv(i) < lminloc.val) {
      lminloc.val = xv(i);
      lminloc.loc = i;
    }
  },Kokkos::MinLoc<PetscReal,PetscInt>(minloc)); /* Kokkos will set minloc properly even if xin is zero-lengthed */
  *p   = minloc.loc;
  *val = minloc.val;
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_SeqKokkos(Vec xin,PetscInt *p,PetscReal *val)
{
  typedef Kokkos::MaxLoc<PetscReal,PetscInt>::value_type MaxLocValue_t;

  PetscErrorCode                  ierr;
  ConstPetscScalarViewDevice_t    xv;
  MaxLocValue_t                   maxloc;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  Kokkos::parallel_reduce("VecMax",xin->map->n,KOKKOS_LAMBDA(PetscInt i,MaxLocValue_t& lmaxloc) {
    if (xv(i) > lmaxloc.val) {
      lmaxloc.val = xv(i);
      lmaxloc.loc = i;
    }
  },Kokkos::MaxLoc<PetscReal,PetscInt>(maxloc));
  *p   = maxloc.loc;
  *val = maxloc.val;
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecShift_SeqKokkos(Vec xin,PetscScalar shift)
{
  PetscErrorCode                  ierr;
  PetscScalarViewDevice_t         xv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(xin,&xv);CHKERRQ(ierr);
  Kokkos::parallel_for("VecShift",xin->map->n,KOKKOS_LAMBDA(PetscInt i) {xv(i) += shift;});
  ierr = VecKokkosRestoreDeviceView(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = alpha x + y */
PetscErrorCode VecAXPY_SeqKokkos(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode               ierr;
  PetscBool                    xiskok,yiskok;
  PetscScalarViewDevice_t      yv;
  ConstPetscScalarViewDevice_t xv;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(0);
  if (yin == xin) {
    ierr = VecScale_SeqKokkos(yin,alpha+1);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&xiskok,VECSEQKOKKOS,VECMPIKOKKOS,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiskok,VECSEQKOKKOS,VECMPIKOKKOS,"");CHKERRQ(ierr);
  if (xiskok && yiskok) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceView(yin,&yv);CHKERRQ(ierr);
    KokkosBlas::axpy(alpha,xv,yv);
    ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosRestoreDeviceView(yin,&yv);CHKERRQ(ierr);
    ierr = WaitForKokkos();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* y = x + beta y */
PetscErrorCode VecAYPX_SeqKokkos(Vec yin,PetscScalar beta,Vec xin)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* One needs to define KOKKOSBLAS_OPTIMIZATION_LEVEL_AXPBY > 2 to have optimizations for cases alpha/beta = 0,+/-1 */
  ierr = VecAXPBY_SeqKokkos(yin,1.0,beta,xin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = y^H x */
PetscErrorCode VecDot_SeqKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode                  ierr;
  ConstPetscScalarViewDevice_t    xv,yv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  Kokkos::parallel_reduce("VecDot",xin->map->n,KOKKOS_LAMBDA(int64_t i, PetscScalar& update) {
#if defined(PETSC_USE_COMPLEX)
    update += Kokkos::conj(yv(i))*xv(i);
#else
    update += yv(i)*xv(i);
#endif
  },*z); /* Kokkos always overwrites z, so no need to init it */
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n > 0) {ierr = PetscLogGpuFlops(2.0*xin->map->n);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

struct MDotFunctor {
  /* Note the C++ notation for an array typedef */
  typedef PetscScalar value_type[];
  typedef ConstPetscScalarViewDevice_t::size_type size_type;

  /* Tell Kokkos the result array's number of entries. This must be a public value in the functor */
  const size_type              value_count;
  ConstPetscScalarViewDevice_t xv,yv[8];

  MDotFunctor(ConstPetscScalarViewDevice_t& xv,
              const PetscInt ny, /* Number of valid entries in yv[8]. 1 <= ny <= 8 */
              ConstPetscScalarViewDevice_t& yv0, ConstPetscScalarViewDevice_t& yv1,
              ConstPetscScalarViewDevice_t& yv2, ConstPetscScalarViewDevice_t& yv3,
              ConstPetscScalarViewDevice_t& yv4, ConstPetscScalarViewDevice_t& yv5,
              ConstPetscScalarViewDevice_t& yv6, ConstPetscScalarViewDevice_t& yv7)
    : value_count(ny),xv(xv)
  {
    yv[0] = yv0; yv[1] = yv1;
    yv[2] = yv2; yv[3] = yv3;
    yv[4] = yv4; yv[5] = yv5;
    yv[6] = yv6; yv[7] = yv7;
  }

  KOKKOS_INLINE_FUNCTION void operator() (const size_type i,value_type sum) const {
    PetscScalar xval = xv(i);
    for (size_type j = 0; j<value_count; ++j) {
     #if defined(PETSC_USE_COMPLEX)
      sum[j] += Kokkos::conj(yv[j](i))*xval;
     #else
      sum[j] += yv[j](i)*xval;
     #endif
    }
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

/* z[i] = (x,y_i) = y_i^H x */
PetscErrorCode VecMDot_SeqKokkos(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode                  ierr;
  PetscInt                        i,j,cur=0,ngroup=nv/8,rem=nv%8;
  ConstPetscScalarViewDevice_t    xv,yv[8];
  PetscScalarViewHost_t           zv(z,nv);

  PetscFunctionBegin;
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  for (i=0; i<ngroup; i++) { /* 8 y's per group */
    for (j=0; j<8; j++) {ierr = VecKokkosGetDeviceViewRead(yin[cur+j],&yv[j]);CHKERRQ(ierr);}
    MDotFunctor mdot(xv,8,yv[0],yv[1],yv[2],yv[3],yv[4],yv[5],yv[6],yv[7]); /* Hope Kokkos make it asynchronous */
    Kokkos::parallel_reduce(xin->map->n,mdot,Kokkos::subview(zv,Kokkos::pair<PetscInt,PetscInt>(cur,cur+8)));
    for (j=0; j<8; j++) {ierr = VecKokkosRestoreDeviceViewRead(yin[cur+j],&yv[j]);CHKERRQ(ierr);}
    cur += 8;
  }

  if (rem) { /* The remaining */
    for (j=0; j<rem; j++) {ierr = VecKokkosGetDeviceViewRead(yin[cur+j],&yv[j]);CHKERRQ(ierr);}
    MDotFunctor mdot(xv,rem,yv[0],yv[1],yv[2],yv[3],yv[4],yv[5],yv[6],yv[7]);
    Kokkos::parallel_reduce(xin->map->n,mdot,Kokkos::subview(zv,Kokkos::pair<PetscInt,PetscInt>(cur,cur+rem)));
    for (j=0; j<rem; j++) {ierr = VecKokkosRestoreDeviceViewRead(yin[cur+j],&yv[j]);CHKERRQ(ierr);}
  }
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  Kokkos::fence(); /* If reduce is async, then we need this fence to make sure z is ready for use on host */
  PetscFunctionReturn(0);
}

/* x[:] = alpha */
PetscErrorCode VecSet_SeqKokkos(Vec xin,PetscScalar alpha)
{
  PetscErrorCode            ierr;
  PetscScalarViewDevice_t   xv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewWrite(xin,&xv);CHKERRQ(ierr);
  KokkosBlas::fill(xv,alpha);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewWrite(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* x = alpha x */
PetscErrorCode VecScale_SeqKokkos(Vec xin,PetscScalar alpha)
{
  PetscErrorCode            ierr;
  PetscScalarViewDevice_t   xv;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_SeqKokkos(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceView(xin,&xv);CHKERRQ(ierr);
    KokkosBlas::scal(xv,alpha,xv);
    ierr = VecKokkosRestoreDeviceView(xin,&xv);CHKERRQ(ierr);
    ierr = WaitForKokkos();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* z = x^T y */
PetscErrorCode VecTDot_SeqKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode               ierr;
  ConstPetscScalarViewDevice_t xv,yv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  *z = KokkosBlas::dot(xv,yv);
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n > 0) {ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* y = x, where x is VECKOKKOS, but y may be not */
PetscErrorCode VecCopy_SeqKokkos(Vec xin,Vec yin)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (xin != yin) {
    Vec_Kokkos *xkok = static_cast<Vec_Kokkos*>(xin->spptr);
    if (yin->offloadmask == PETSC_OFFLOAD_VECKOKKOS) {
      /* y is also a VecKokkos */
      Vec_Kokkos *ykok = static_cast<Vec_Kokkos*>(yin->spptr);
      /* Kokkos rule: if x's host has newer data, it will copy to y's host view; otherwise to y's device view
        In case x's host ins newer, y's device is newer, it will error (though should not, I think). So we just
        clear y's sync state.
       */
      ykok->dual_v.clear_sync_state();
      Kokkos::deep_copy(ykok->dual_v,xkok->dual_v);
    } else {
      PetscScalar *yarray;
      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      PetscScalarViewHost_t yv(yarray,yin->map->n);
      if (xkok->dual_v.need_sync_host()) { /* x's device has newer data */
        Kokkos::deep_copy(yv,xkok->dual_v.view_device());
      } else {
        Kokkos::deep_copy(yv,xkok->dual_v.view_host());
      }
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    }
  }
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y[i] <--> x[i] */
PetscErrorCode VecSwap_SeqKokkos(Vec xin,Vec yin)
{
  PetscErrorCode                  ierr;
  PetscScalarViewDevice_t         xv,yv;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceView(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceView(yin,&yv);CHKERRQ(ierr);
    Kokkos::parallel_for(xin->map->n,KOKKOS_LAMBDA(const int64_t i) {
      PetscScalar tmp = xv(i);
      xv(i) = yv(i);
      yv(i) = tmp;
    });
    ierr = WaitForKokkos();CHKERRQ(ierr);
    ierr = VecKokkosRestoreDeviceView(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosRestoreDeviceView(yin,&yv);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*  w = alpha x + y */
PetscErrorCode VecWAXPY_SeqKokkos(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  PetscErrorCode                  ierr;
  ConstPetscScalarViewDevice_t    xv,yv;
  PetscScalarViewDevice_t         wv;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecCopy_SeqKokkos(yin,win);CHKERRQ(ierr);
  } else {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceViewWrite(win,&wv);CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceViewRead(yin,&yv);CHKERRQ(ierr);
    Kokkos::parallel_for(win->map->n,KOKKOS_LAMBDA(const int64_t i) {wv(i) = alpha*xv(i) + yv(i);});
    ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosRestoreDeviceViewRead(yin,&yv);CHKERRQ(ierr);
    ierr = VecKokkosRestoreDeviceViewWrite(win,&wv);CHKERRQ(ierr);
    ierr = WaitForKokkos();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*win->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct MAXPYFunctor {
  typedef ConstPetscScalarViewDevice_t::size_type size_type;

  PetscScalarViewDevice_t       yv;
  PetscInt                      nx;   /* Significent entries in a[8] and xv[8] */
  PetscScalar                   a[8];
  ConstPetscScalarViewDevice_t  xv[8];

  MAXPYFunctor(PetscScalarViewDevice_t yv,
               PetscInt nx,
               PetscScalar a0,PetscScalar a1,PetscScalar a2,PetscScalar a3,
               PetscScalar a4,PetscScalar a5,PetscScalar a6,PetscScalar a7,
               ConstPetscScalarViewDevice_t xv0, ConstPetscScalarViewDevice_t xv1,
               ConstPetscScalarViewDevice_t xv2, ConstPetscScalarViewDevice_t xv3,
               ConstPetscScalarViewDevice_t xv4, ConstPetscScalarViewDevice_t xv5,
               ConstPetscScalarViewDevice_t xv6, ConstPetscScalarViewDevice_t xv7)
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

  KOKKOS_INLINE_FUNCTION void operator() (const size_type i) const {
    for (PetscInt j = 0; j<nx; ++j) yv(i) += a[j]*xv[j](i);
  }
};

/*  y = y + sum alpha[i] x[i] */
PetscErrorCode VecMAXPY_SeqKokkos(Vec yin, PetscInt nv,const PetscScalar *alpha,Vec *xin)
{
  PetscErrorCode                  ierr;
  PetscInt                        i,j,cur=0,ngroup=nv/8,rem=nv%8;
  PetscScalarViewDevice_t         yv;
  PetscScalar                     a[8];
  ConstPetscScalarViewDevice_t    xv[8];

  PetscFunctionBegin;
  ierr = VecKokkosGetDeviceView(yin,&yv);CHKERRQ(ierr);
  for (i=0; i<ngroup; i++) { /* 8 x's per group */
    for (j=0; j<8; j++) { /* Fill the parameters */
      a[j] = alpha[cur+j];
      ierr = VecKokkosGetDeviceViewRead(xin[cur+j],&xv[j]);CHKERRQ(ierr);
    }
    MAXPYFunctor maxpy(yv,8,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],xv[0],xv[1],xv[2],xv[3],xv[4],xv[5],xv[6],xv[7]);
    Kokkos::parallel_for(yin->map->n,maxpy);
    for (j=0; j<8; j++) {ierr = VecKokkosRestoreDeviceViewRead(xin[cur+j],&xv[j]);CHKERRQ(ierr);}
    cur += 8;
  }

  if (rem) { /* The remaining */
    for (j=0; j<rem; j++) {
      a[j] = alpha[cur+j];
      ierr = VecKokkosGetDeviceViewRead(xin[cur+j],&xv[j]);CHKERRQ(ierr);
    }
    MAXPYFunctor maxpy(yv,rem,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],xv[0],xv[1],xv[2],xv[3],xv[4],xv[5],xv[6],xv[7]);
    Kokkos::parallel_for(yin->map->n,maxpy);
    for (j=0; j<rem; j++) {ierr = VecKokkosRestoreDeviceViewRead(xin[cur+j],&xv[j]);CHKERRQ(ierr);}
  }
  ierr = VecKokkosRestoreDeviceView(yin,&yv);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(nv*2.0*yin->map->n);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = alpha x + beta y */
PetscErrorCode VecAXPBY_SeqKokkos(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode               ierr;
  ConstPetscScalarViewDevice_t xv;
  PetscScalarViewDevice_t      yv;
  PetscBool                    xiskok,yiskok;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&xiskok,VECSEQKOKKOS,VECMPIKOKKOS,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiskok,VECSEQKOKKOS,VECMPIKOKKOS,"");CHKERRQ(ierr);
  if (xiskok && yiskok) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceView(yin,&yv);CHKERRQ(ierr);
    KokkosBlas::axpby(alpha,xv,beta,yv);
    ierr = WaitForKokkos();CHKERRQ(ierr);
    ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    ierr = VecKokkosRestoreDeviceView(yin,&yv);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    if (alpha == (PetscScalar)0.0 || beta == (PetscScalar)0.0) {
      ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
    } else if (beta == (PetscScalar)1.0 || alpha == (PetscScalar)1.0) {
      ierr = PetscLogGpuFlops(2.0*xin->map->n);CHKERRQ(ierr);
    } else {
      ierr = PetscLogGpuFlops(3.0*xin->map->n);CHKERRQ(ierr);
    }
  } else {
    ierr = VecAXPBY_Seq(yin,alpha,beta,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* z = alpha x + beta y + gamma z */
PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  ConstPetscScalarViewDevice_t    xv,yv;
  PetscScalarViewDevice_t         zv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zin,&zv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  KokkosBlas::update(alpha,xv,beta,yv,gamma,zv);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zin,&zv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(zin->map->n*5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* w = x*y. Any subset of the x, y, and w may be the same vector. */
PetscErrorCode VecPointwiseMult_SeqKokkos(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode                  ierr;
  ConstPetscScalarViewDevice_t    xv,yv;
  PetscScalarViewDevice_t         wv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewWrite(win,&wv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  Kokkos::parallel_for(win->map->n,KOKKOS_LAMBDA(const int64_t i) {wv(i) = xv(i)*yv(i);});
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewWrite(win,&wv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(win->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* w = x/y */
PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode                  ierr;
  ConstPetscScalarViewDevice_t    xv,yv;
  PetscScalarViewDevice_t         wv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewWrite(win,&wv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  Kokkos::parallel_for(win->map->n,KOKKOS_LAMBDA(const int64_t i) {
    if (yv(i) != 0.0) wv(i) = xv(i)/yv(i);
    else wv(i) = 0.0;
  });
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewWrite(win,&wv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(win->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_SeqKokkos(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode                ierr;
  const PetscInt                n = xin->map->n;
  ConstPetscScalarViewDevice_t  xv;

  PetscFunctionBegin;
  if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqKokkos(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqKokkos(xin,NORM_2,z+1);CHKERRQ(ierr);
  } else {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    if (type == NORM_2 || type == NORM_FROBENIUS) {
      *z   = KokkosBlas::nrm2(xv);
      ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
    } else if (type == NORM_1) {
      *z   = KokkosBlas::nrm1(xv);
      ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) {
      *z = KokkosBlas::nrminf(xv);
    }
    ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* A functor for DotNorm2 so that we can compute dp and nm in one kernel */
struct DotNorm2 {
  typedef PetscScalar                                    value_type[];
  typedef ConstPetscScalarViewDevice_t::size_type  size_type;

  size_type                    value_count;
  ConstPetscScalarViewDevice_t xv_, yv_;

  DotNorm2(ConstPetscScalarViewDevice_t& xv,ConstPetscScalarViewDevice_t& yv) :
    value_count(2), xv_(xv), yv_(yv) {}

  KOKKOS_INLINE_FUNCTION void operator() (const size_type i, value_type result) const {
    #if defined(PETSC_USE_COMPLEX)
      result[0] += Kokkos::conj(yv_(i))*xv_(i);
      result[1] += Kokkos::conj(yv_(i))*xv_(i);
    #else
      result[0] += yv_(i)*xv_(i);
      result[1] += yv_(i)*yv_(i);
    #endif
  }

  KOKKOS_INLINE_FUNCTION void join (volatile value_type dst, const volatile value_type src) const {
    dst[0] += src[0];
    dst[1] += src[1];
  }

  KOKKOS_INLINE_FUNCTION void init (value_type result) const {
    result[0] = 0.0;
    result[1] = 0.0;
  }
};

/* dp = y^H x, nm = y^H y */
PetscErrorCode VecDotNorm2_SeqKokkos(Vec xin, Vec yin, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode                  ierr;
  ConstPetscScalarViewDevice_t    xv,yv;
  PetscScalar                     result[2];

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  DotNorm2 dn(xv,yv);
  Kokkos::parallel_reduce(xin->map->n,dn,result);
  ierr = VecKokkosRestoreDeviceViewRead(yin,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(xin,&xv);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(4.0*xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecConjugate_SeqKokkos(Vec xin)
{
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode            ierr;
  PetscScalarViewDevice_t   xv;

  PetscFunctionBegin;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(xin,&xv);CHKERRQ(ierr);
  Kokkos::parallel_for(xin->map->n,KOKKOS_LAMBDA(int64_t i) {xv(i) = Kokkos::conj(xv(i));});
  ierr = VecKokkosRestoreDeviceView(xin,&xv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(err);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode VecKokkosUpdateAfterChangingHostArray_Private(Vec v)
{
  Vec_Kokkos *veckok = static_cast<Vec_Kokkos*>(v->spptr);
  Vec_Seq    *vecseq = static_cast<Vec_Seq*>(v->data);

  PetscFunctionBegin;
  /* Rebuild h_v and dual_v with the new host array*/
  veckok->h_v    = PetscScalarViewHost_t(vecseq->array,v->map->n);
  veckok->dual_v = PetscScalarKokkosDualView_t(veckok->d_v,veckok->h_v);
  veckok->dual_v.modify_host();
  PetscFunctionReturn(0);
}

/* Temporarily replace the array in vin with a[]. Return to the original array with a call to VecResetArray() */
PetscErrorCode VecPlaceArray_SeqKokkos(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  ierr = VecKokkosUpdateAfterChangingHostArray_Private(vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqKokkos(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecKokkosSyncHost(vin);CHKERRQ(ierr); /* User wants to unhook the provided host array. Sync it so that user can get the latest */
  ierr = VecResetArray_Seq(vin);CHKERRQ(ierr);
  ierr = VecKokkosUpdateAfterChangingHostArray_Private(vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Replace the array in vin with a[] that must be allocated by PetscMalloc. a[] is owned by vin afterwords. */
PetscErrorCode VecReplaceArray_SeqKokkos(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;
  Vec_Seq        *vecseq = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  /* Make sure the users array has the latest values */
  if (vecseq->array != vecseq->array_allocated) {ierr = VecKokkosSyncHost(vin);CHKERRQ(ierr);}
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  ierr = VecKokkosUpdateAfterChangingHostArray_Private(vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Maps the local portion of vector v into vector w */
PetscErrorCode VecGetLocalVector_SeqKokkos(Vec v,Vec w)
{
  PetscErrorCode   ierr;
  Vec_Seq          *vecseq = static_cast<Vec_Seq*>(w->data);
  Vec_Kokkos       *veckok = static_cast<Vec_Kokkos*>(w->spptr);

  PetscFunctionBegin;
  PetscCheckTypeName(w,VECSEQKOKKOS);
  /* Destroy w->data, w->spptr */
  if (vecseq) {
    ierr = PetscFree(vecseq->array_allocated);CHKERRQ(ierr);
    ierr = PetscFree(w->data);CHKERRQ(ierr);
  }
  delete veckok;

  /* Replace with v's */
  w->data  = v->data;
  w->spptr = v->spptr;
  ierr     = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec v,Vec w)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeName(w,VECSEQKOKKOS);
  v->data  = w->data;
  v->spptr = w->spptr;
  ierr     = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  /* TODO: need to think if setting w->data/spptr to NULL is safe */
  w->data  = NULL;
  w->spptr = NULL;
  PetscFunctionReturn(0);
}

/* Get array on host to overwrite, so no need to sync host. In VecRestoreArrayWrite() we will mark host is modified. */
PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec v,PetscScalar **vv)
{
  Vec_Kokkos       *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  veckok->dual_v.clear_sync_state();
  *vv = veckok->dual_v.view_host().data();
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
  v->ops->shift                  = VecShift_SeqKokkos;
  v->ops->dot                    = VecDot_SeqKokkos;
  v->ops->norm                   = VecNorm_SeqKokkos;
  v->ops->tdot                   = VecTDot_SeqKokkos;
  v->ops->scale                  = VecScale_SeqKokkos;
  v->ops->copy                   = VecCopy_SeqKokkos;
  v->ops->set                    = VecSet_SeqKokkos;
  v->ops->swap                   = VecSwap_SeqKokkos;
  v->ops->axpy                   = VecAXPY_SeqKokkos;
  v->ops->axpby                  = VecAXPBY_SeqKokkos;
  v->ops->axpbypcz               = VecAXPBYPCZ_SeqKokkos;
  v->ops->pointwisedivide        = VecPointwiseDivide_SeqKokkos;
  v->ops->setrandom              = VecSetRandom_SeqKokkos;
  v->ops->dot_local              = VecDot_SeqKokkos;
  v->ops->tdot_local             = VecTDot_SeqKokkos;
  v->ops->norm_local             = VecNorm_SeqKokkos;
  v->ops->mdot_local             = VecMDot_SeqKokkos;
  v->ops->maxpy                  = VecMAXPY_SeqKokkos;
  v->ops->mdot                   = VecMDot_SeqKokkos;
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
  PetscFunctionReturn(0);
}

/* Assuming the Vec_Seq struct of the vector is ready, build the Vec_Kokkos struct for it.
   Also assume the vector is to be init'ed, so it is in a host-device sync'ed state after the call.
 */
static PetscErrorCode BuildVecKokkosFromVecSeq_Private(Vec v)
{
  Vec_Seq        *vecseq = static_cast<Vec_Seq*>(v->data);
  Vec_Kokkos     *veckok = NULL;
  PetscScalar    *darray;

  PetscFunctionBegin;
  if (std::is_same<DeviceMemorySpace,HostMemorySpace>::value) {
    darray = vecseq->array;
  } else {
    darray = static_cast<PetscScalar*>(Kokkos::kokkos_malloc<DeviceMemorySpace>(sizeof(PetscScalar)*v->map->n));
  }
  if (v->spptr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"v->spptr not NULL");
  veckok = new Vec_Kokkos(v->map->n,vecseq->array,darray,darray);
  Kokkos::deep_copy(veckok->dual_v.view_device(),0.0);
  v->spptr = static_cast<void*>(veckok);
  v->offloadmask = PETSC_OFFLOAD_VECKOKKOS;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_SeqKokkos(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = VecCreate_Seq(v);CHKERRQ(ierr);  /* Build a sequential vector, allocate array */
  ierr = VecSet_Seq(v,0.0);CHKERRQ(ierr); /* Zero the host array */
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECSEQKOKKOS);CHKERRQ(ierr);
  ierr = VecSetOps_SeqKokkos(v);CHKERRQ(ierr);
  ierr = BuildVecKokkosFromVecSeq_Private(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqKokkosWithArray - Creates a Kokkos sequential array-style vector,
   where the user provides the array space to store the vector values. The array
   provided must be a device array.

   Collective

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  array - device memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecCUDAPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateMPICUDAWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
          VecCreateGhost(), VecCreateSeq(), VecCUDAPlaceArray(), VecCreateSeqWithArray(),
          VecCreateMPIWithArray()
@*/
PetscErrorCode  VecCreateSeqKokkosWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar darray[],Vec *v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Vec            w;
  Vec_Kokkos     *veckok = NULL;
  PetscScalar    *harray;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQKOKKOS on more than one process");

  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = VecCreate(comm,&w);CHKERRQ(ierr);
  ierr = VecSetSizes(w,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(w,bs);CHKERRQ(ierr);

  /* Given a device array, build the Vec_Seq struct */
  if (std::is_same<DeviceMemorySpace,HostMemorySpace>::value) {harray = const_cast<PetscScalar*>(darray);}
  else {ierr = PetscMalloc1(w->map->n,&harray);CHKERRQ(ierr);}
  ierr   = VecCreate_Seq_Private(w,harray);CHKERRQ(ierr); /* Build a sequential vector with harray */

  ierr   = PetscObjectChangeTypeName((PetscObject)w,VECSEQKOKKOS);CHKERRQ(ierr); /* Change it to Kokkos */
  ierr   = VecSetOps_SeqKokkos(w);CHKERRQ(ierr);
  veckok = new Vec_Kokkos(n,harray,const_cast<PetscScalar*>(darray),NULL);
  veckok->dual_v.modify_device(); /* Mark the device is modified */
  w->offloadmask = PETSC_OFFLOAD_VECKOKKOS;
  w->spptr = static_cast<void*>(veckok);
  *v       = w;
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

 .seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
 @*/
PetscErrorCode VecCreateSeqKokkos(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQKOKKOS);CHKERRQ(ierr); /* Calls VecCreate_SeqKokkos */
  PetscFunctionReturn(0);
}

/* Duplicate layout etc but not the values in the input vector */
PetscErrorCode VecDuplicate_SeqKokkos(Vec win,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate_Seq(win,v);CHKERRQ(ierr); /* It also dups ops of win */
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqKokkos(Vec v)
{
  PetscErrorCode ierr;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);
  Vec_Seq        *vecseq = static_cast<Vec_Seq*>(v->data);

  PetscFunctionBegin;
  delete veckok;
  v->spptr = NULL;
  if (vecseq) {ierr = VecDestroy_Seq(v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
