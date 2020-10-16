
/*
   This file contains routines for Parallel vector operations.
 */

#include "petsc/private/petscimpl.h"
#include <petscveckokkos.hpp>
#include <petsc/private/vecimpl.h> /* for struct Vec */
#include <../src/vec/vec/impls/mpi/pvecimpl.h> /* for VecCreate/Destroy_MPI */
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>

PetscErrorCode VecDestroy_MPIKokkos(Vec v)
{
  PetscErrorCode ierr;
  Vec_Kokkos     *veckok = static_cast<Vec_Kokkos*>(v->spptr);

  PetscFunctionBegin;
  delete veckok;
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIKokkos(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqKokkos(xin,NORM_2,&work);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqKokkos(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqKokkos(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqKokkos(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqKokkos(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

/* z = y^H x */
PetscErrorCode VecDot_MPIKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqKokkos(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIKokkos(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);}
  ierr = VecMDot_SeqKokkos(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  if (nv > 128) {ierr = PetscFree(work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* z = y^T x */
PetscErrorCode VecTDot_MPIKokkos(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqKokkos(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

extern MPI_Op MPIU_MAXINDEX_OP, MPIU_MININDEX_OP;

PetscErrorCode VecMax_MPIKokkos(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  ierr = VecMax_SeqKokkos(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global max */
  if (!idx) { /* User does not need idx */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;
    rstart   = xin->map->rstart;
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr     = MPIU_Allreduce(work2,z2,2,MPIU_REAL,MPIU_MAXINDEX_OP,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    *z       = z2[0];
    *idx     = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPIKokkos(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  ierr = VecMin_SeqKokkos(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global Min */
  if (!idx) {
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;

    ierr = VecGetOwnershipRange(xin,&rstart,NULL);CHKERRQ(ierr);
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPIU_Allreduce(work2,z2,2,MPIU_REAL,MPIU_MININDEX_OP,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_MPIKokkos(Vec win,Vec *vv)
{
  PetscErrorCode ierr;
  Vec            v;
  PetscScalar    *darray;
  Vec_MPI        *vecmpi;
  Vec_Kokkos     *veckok;

  PetscFunctionBegin;
  /* Reuse VecDuplicate_MPI, which contains a lot of stuff */
  ierr = VecDuplicate_MPI(win,&v);CHKERRQ(ierr); /* after the call, v is a VECMPI, with data zero'ed */
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECMPIKOKKOS);CHKERRQ(ierr);
  ierr = PetscMemcpy(v->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* Build the Vec_Kokkos struct */
  vecmpi = static_cast<Vec_MPI*>(v->data);
  if (std::is_same<DeviceMemorySpace,HostMemorySpace>::value) {
    darray = vecmpi->array;
  } else {
    darray = static_cast<PetscScalar*>(Kokkos::kokkos_malloc<DeviceMemorySpace>(sizeof(PetscScalar)*(v->map->n+vecmpi->nghost)));
  }
  veckok   = new Vec_Kokkos(v->map->n,vecmpi->array,darray,darray);
  Kokkos::deep_copy(veckok->dual_v.view_device(),0.0);
  v->spptr       = veckok;
  v->offloadmask = PETSC_OFFLOAD_VECKOKKOS;
  *vv = v;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPIKokkos(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqKokkos(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRQ(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetOps_MPIKokkos(Vec v)
{
  PetscFunctionBegin;
  v->ops->abs                    = VecAbs_SeqKokkos;
  v->ops->reciprocal             = VecReciprocal_SeqKokkos;
  v->ops->pointwisemult          = VecPointwiseMult_SeqKokkos;
  v->ops->setrandom              = VecSetRandom_SeqKokkos;
  v->ops->dotnorm2               = VecDotNorm2_MPIKokkos;
  v->ops->waxpy                  = VecWAXPY_SeqKokkos;
  v->ops->dot                    = VecDot_MPIKokkos;
  v->ops->mdot                   = VecMDot_MPIKokkos;
  v->ops->tdot                   = VecTDot_MPIKokkos;
  v->ops->norm                   = VecNorm_MPIKokkos;
  v->ops->min                    = VecMin_MPIKokkos;
  v->ops->max                    = VecMax_MPIKokkos;
  v->ops->shift                  = VecShift_SeqKokkos;
  v->ops->scale                  = VecScale_SeqKokkos;
  v->ops->copy                   = VecCopy_SeqKokkos;
  v->ops->set                    = VecSet_SeqKokkos;
  v->ops->swap                   = VecSwap_SeqKokkos;
  v->ops->axpy                   = VecAXPY_SeqKokkos;
  v->ops->axpby                  = VecAXPBY_SeqKokkos;
  v->ops->maxpy                  = VecMAXPY_SeqKokkos;
  v->ops->aypx                   = VecAYPX_SeqKokkos;
  v->ops->axpbypcz               = VecAXPBYPCZ_SeqKokkos;
  v->ops->pointwisedivide        = VecPointwiseDivide_SeqKokkos;
  v->ops->placearray             = VecPlaceArray_SeqKokkos;
  v->ops->replacearray           = VecReplaceArray_SeqKokkos;
  v->ops->resetarray             = VecResetArray_SeqKokkos;
  v->ops->dot_local              = VecDot_SeqKokkos;
  v->ops->tdot_local             = VecTDot_SeqKokkos;
  v->ops->norm_local             = VecNorm_SeqKokkos;
  v->ops->mdot_local             = VecMDot_SeqKokkos;
  v->ops->duplicate              = VecDuplicate_MPIKokkos;
  v->ops->destroy                = VecDestroy_MPIKokkos;
  v->ops->getlocalvector         = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvector     = VecRestoreLocalVector_SeqKokkos;
  v->ops->getlocalvectorread     = VecGetLocalVector_SeqKokkos;
  v->ops->restorelocalvectorread = VecRestoreLocalVector_SeqKokkos;
  v->ops->getarraywrite          = VecGetArrayWrite_SeqKokkos;
  v->ops->getarray               = VecGetArray_SeqKokkos;
  v->ops->restorearray           = VecRestoreArray_SeqKokkos;
  v->ops->getarrayandmemtype        = VecGetArrayAndMemType_SeqKokkos;
  v->ops->restorearrayandmemtype    = VecRestoreArrayAndMemType_SeqKokkos;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPIKokkos(Vec v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vecmpi;
  Vec_Kokkos     *veckok;
  PetscScalar    *darray;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = VecCreate_MPI(v);CHKERRQ(ierr);  /* Build a sequential vector, allocate array */
  ierr = VecSet_Seq(v,0.0);CHKERRQ(ierr); /* Zero the host array */
  vecmpi = static_cast<Vec_MPI*>(v->data);

  if (std::is_same<DeviceMemorySpace,HostMemorySpace>::value) {
    darray = vecmpi->array;
  } else {
    darray = static_cast<PetscScalar*>(Kokkos::kokkos_malloc<DeviceMemorySpace>(sizeof(PetscScalar)*v->map->n));
  }
  ierr   = PetscObjectChangeTypeName((PetscObject)v,VECMPIKOKKOS);CHKERRQ(ierr);
  ierr   = VecSetOps_MPIKokkos(v);CHKERRQ(ierr);
  veckok = new Vec_Kokkos(v->map->n,vecmpi->array,darray,darray);
  Kokkos::deep_copy(veckok->dual_v.view_device(),0.0);
  v->spptr = static_cast<void*>(veckok);
  v->offloadmask = PETSC_OFFLOAD_VECKOKKOS;
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIKokkosWithArray - Creates a parallel, array-style vector,
   where the user provides the GPU array space to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  array - the user provided GPU array to store the vector values

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecKokkosPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqKokkosWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PetscErrorCode  VecCreateMPIKokkosWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar darray[],Vec *v)
{
  PetscErrorCode ierr;
  Vec            w;
  Vec_Kokkos     *veckok;
  Vec_MPI        *vecmpi;
  PetscScalar    *harray;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,&w);CHKERRQ(ierr);
  ierr = VecSetSizes(w,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(w,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(w->map);CHKERRQ(ierr);
  if (std::is_same<DeviceMemorySpace,HostMemorySpace>::value) {harray = const_cast<PetscScalar*>(darray);}
  else {harray = (PetscScalar*)Kokkos::kokkos_malloc<HostMemorySpace>(sizeof(PetscScalar)*w->map->n);}

  ierr   = VecCreate_MPI_Private(w,PETSC_FALSE,0,harray);CHKERRQ(ierr); /* Build a sequential vector with provided data */
  vecmpi = static_cast<Vec_MPI*>(w->data);
  if (std::is_same<DeviceMemorySpace,HostMemorySpace>::value) vecmpi->array_allocated = harray;

  ierr   = PetscObjectChangeTypeName((PetscObject)w,VECMPIKOKKOS);CHKERRQ(ierr);
  ierr   = VecSetOps_MPIKokkos(w);CHKERRQ(ierr);
  veckok = new Vec_Kokkos(n,harray,const_cast<PetscScalar*>(darray),NULL);
  veckok->dual_v.modify_device(); /* Mark the device is modified */
  w->spptr = static_cast<void*>(veckok);
  w->offloadmask = PETSC_OFFLOAD_VECKOKKOS;
  *v = w;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_Kokkos(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRQ(ierr);
  if (size == 1) {ierr = VecSetType(v,VECSEQKOKKOS);CHKERRQ(ierr);}
  else {ierr = VecSetType(v,VECMPIKOKKOS);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}