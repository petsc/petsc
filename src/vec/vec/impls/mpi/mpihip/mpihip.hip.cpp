
/*
   This file contains routines for Parallel vector operations.
 */
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <petsc/private/hipvecimpl.h>

/*MC
   VECHIP - VECHIP = "hip" - A VECSEQHIP on a single-process communicator, and VECMPIHIP otherwise.

   Options Database Keys:
. -vec_type hip - sets the vector type to VECHIP during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQHIP, VECMPIHIP, VECSTANDARD, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

PETSC_INTERN
PetscErrorCode VecDestroy_MPIHIP(Vec v)
{
  Vec_MPI        *vecmpi = (Vec_MPI*)v->data;
  Vec_HIP        *vechip;
  PetscErrorCode ierr;
  hipError_t    err;

  PetscFunctionBegin;
  if (v->spptr) {
    vechip = (Vec_HIP*)v->spptr;
    if (vechip->GPUarray_allocated) {
      err = hipFree(vechip->GPUarray_allocated);CHKERRHIP(err);
      vechip->GPUarray_allocated = NULL;
    }
    if (vechip->stream) {
      err = hipStreamDestroy(vechip->stream);CHKERRHIP(err);
    }
    if (v->pinned_memory) {
      ierr = PetscMallocSetHIPHost();CHKERRQ(ierr);
      ierr = PetscFree(vecmpi->array_allocated);CHKERRQ(ierr);
      ierr = PetscMallocResetHIPHost();CHKERRQ(ierr);
      v->pinned_memory = PETSC_FALSE;
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecNorm_MPIHIP(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqHIP(xin,NORM_2,&work);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqHIP(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqHIP(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqHIP(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqHIP(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecDot_MPIHIP(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqHIP(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecTDot_MPIHIP(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqHIP(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecMDot_MPIHIP(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqHIP(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   VECMPIHIP - VECMPIHIP = "mpihip" - The basic parallel vector, modified to use HIP

   Options Database Keys:
. -vec_type mpihip - sets the vector type to VECMPIHIP during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/


PETSC_INTERN
PetscErrorCode VecDuplicate_MPIHIP(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPIHIP_Private(*v,PETSC_TRUE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPI*)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)*v,(PetscObject)vw->localrep);CHKERRQ(ierr);
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash   = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  /* change type_name appropriately */
  ierr = VecHIPAllocateCheck(*v);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)(*v),VECMPIHIP);CHKERRQ(ierr);

  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecDotNorm2_MPIHIP(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqHIP(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRQ(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecCreate_MPIHIP(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHIPInitializeCheck();CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(vv->map);CHKERRQ(ierr);
  ierr = VecHIPAllocateCheck(vv);CHKERRQ(ierr);
  ierr = VecCreate_MPIHIP_Private(vv,PETSC_FALSE,0,((Vec_HIP*)vv->spptr)->GPUarray_allocated);CHKERRQ(ierr);
  ierr = VecHIPAllocateCheckHost(vv);CHKERRQ(ierr);
  ierr = VecSet(vv,0.0);CHKERRQ(ierr);
  ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
  vv->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecCreate_HIP(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQHIP);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPIHIP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIHIPWithArray - Creates a parallel, array-style vector,
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

   If the user-provided array is NULL, then VecHIPPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqHIPWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PETSC_EXTERN
PetscErrorCode  VecCreateMPIHIPWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscHIPInitializeCheck();CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  ierr = VecCreate_MPIHIP_Private(*vv,PETSC_FALSE,0,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIHIPWithArrays - Creates a parallel, array-style vector,
   where the user provides the GPU array space to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  cpuarray - the user provided CPU array to store the vector values
-  gpuarray - the user provided GPU array to store the vector values

   Output Parameter:
.  vv - the vector

   Notes:
   If both cpuarray and gpuarray are provided, the caller must ensure that
   the provided arrays have identical values.

   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   PETSc does NOT free the provided arrays when the vector is destroyed via
   VecDestroy(). The user should not free the array until the vector is
   destroyed.

   Level: intermediate

.seealso: VecCreateSeqHIPWithArrays(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecHIPPlaceArray(), VecPlaceArray(),
          VecHIPAllocateCheckHost()
@*/
PETSC_EXTERN
PetscErrorCode  VecCreateMPIHIPWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPIHIPWithArray(comm,bs,n,N,gpuarray,vv);CHKERRQ(ierr);

  if (cpuarray && gpuarray) {
    Vec_MPI *s         = (Vec_MPI*)((*vv)->data);
    s->array           = (PetscScalar*)cpuarray;
    (*vv)->offloadmask = PETSC_OFFLOAD_BOTH;
  } else if (cpuarray) {
    Vec_MPI *s         = (Vec_MPI*)((*vv)->data);
    s->array           = (PetscScalar*)cpuarray;
    (*vv)->offloadmask =  PETSC_OFFLOAD_CPU;
  } else if (gpuarray) {
    (*vv)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*vv)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecBindToCPU_MPIHIP(Vec V,PetscBool pin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->boundtocpu = pin;
  if (pin) {
    ierr = VecHIPCopyFromGPU(V);CHKERRQ(ierr);
    V->offloadmask = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    V->ops->dotnorm2               = NULL;
    V->ops->waxpy                  = VecWAXPY_Seq;
    V->ops->dot                    = VecDot_MPI;
    V->ops->mdot                   = VecMDot_MPI;
    V->ops->tdot                   = VecTDot_MPI;
    V->ops->norm                   = VecNorm_MPI;
    V->ops->scale                  = VecScale_Seq;
    V->ops->copy                   = VecCopy_Seq;
    V->ops->set                    = VecSet_Seq;
    V->ops->swap                   = VecSwap_Seq;
    V->ops->axpy                   = VecAXPY_Seq;
    V->ops->axpby                  = VecAXPBY_Seq;
    V->ops->maxpy                  = VecMAXPY_Seq;
    V->ops->aypx                   = VecAYPX_Seq;
    V->ops->axpbypcz               = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult          = VecPointwiseMult_Seq;
    V->ops->setrandom              = VecSetRandom_Seq;
    V->ops->placearray             = VecPlaceArray_Seq;
    V->ops->replacearray           = VecReplaceArray_SeqHIP;
    V->ops->resetarray             = VecResetArray_Seq;
    V->ops->dot_local              = VecDot_Seq;
    V->ops->tdot_local             = VecTDot_Seq;
    V->ops->norm_local             = VecNorm_Seq;
    V->ops->mdot_local             = VecMDot_Seq;
    V->ops->pointwisedivide        = VecPointwiseDivide_Seq;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;
  } else {
    V->ops->dotnorm2               = VecDotNorm2_MPIHIP;
    V->ops->waxpy                  = VecWAXPY_SeqHIP;
    V->ops->duplicate              = VecDuplicate_MPIHIP;
    V->ops->dot                    = VecDot_MPIHIP;
    V->ops->mdot                   = VecMDot_MPIHIP;
    V->ops->tdot                   = VecTDot_MPIHIP;
    V->ops->norm                   = VecNorm_MPIHIP;
    V->ops->scale                  = VecScale_SeqHIP;
    V->ops->copy                   = VecCopy_SeqHIP;
    V->ops->set                    = VecSet_SeqHIP;
    V->ops->swap                   = VecSwap_SeqHIP;
    V->ops->axpy                   = VecAXPY_SeqHIP;
    V->ops->axpby                  = VecAXPBY_SeqHIP;
    V->ops->maxpy                  = VecMAXPY_SeqHIP;
    V->ops->aypx                   = VecAYPX_SeqHIP;
    V->ops->axpbypcz               = VecAXPBYPCZ_SeqHIP;
    V->ops->pointwisemult          = VecPointwiseMult_SeqHIP;
    V->ops->setrandom              = VecSetRandom_SeqHIP;
    V->ops->placearray             = VecPlaceArray_SeqHIP;
    V->ops->replacearray           = VecReplaceArray_SeqHIP;
    V->ops->resetarray             = VecResetArray_SeqHIP;
    V->ops->dot_local              = VecDot_SeqHIP;
    V->ops->tdot_local             = VecTDot_SeqHIP;
    V->ops->norm_local             = VecNorm_SeqHIP;
    V->ops->mdot_local             = VecMDot_SeqHIP;
    V->ops->destroy                = VecDestroy_MPIHIP;
    V->ops->pointwisedivide        = VecPointwiseDivide_SeqHIP;
    V->ops->getlocalvector         = VecGetLocalVector_SeqHIP;
    V->ops->restorelocalvector     = VecRestoreLocalVector_SeqHIP;
    V->ops->getlocalvectorread     = VecGetLocalVector_SeqHIP;
    V->ops->restorelocalvectorread = VecRestoreLocalVector_SeqHIP;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqHIP;
    V->ops->getarray               = VecGetArray_SeqHIP;
    V->ops->restorearray           = VecRestoreArray_SeqHIP;
    V->ops->getarrayandmemtype        = VecGetArrayAndMemType_SeqHIP;
    V->ops->restorearrayandmemtype    = VecRestoreArrayAndMemType_SeqHIP;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN
PetscErrorCode VecCreate_MPIHIP_Private(Vec vv,PetscBool alloc,PetscInt nghost,const PetscScalar array[])
{
  PetscErrorCode ierr;
  Vec_HIP       *vechip;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_FALSE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPIHIP);CHKERRQ(ierr);

  ierr = VecBindToCPU_MPIHIP(vv,PETSC_FALSE);CHKERRQ(ierr);
  vv->ops->bindtocpu = VecBindToCPU_MPIHIP;

  /* Later, functions check for the Vec_HIP structure existence, so do not create it without array */
  if (alloc && !array) {
    ierr = VecHIPAllocateCheck(vv);CHKERRQ(ierr);
    ierr = VecHIPAllocateCheckHost(vv);CHKERRQ(ierr);
    ierr = VecSet(vv,0.0);CHKERRQ(ierr);
    ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
    vv->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  if (array) {
    if (!vv->spptr) {
      PetscReal pinned_memory_min;
      PetscBool flag;
      /* Cannot use PetscNew() here because spptr is void* */
      ierr = PetscMalloc(sizeof(Vec_HIP),&vv->spptr);CHKERRQ(ierr);
      vechip = (Vec_HIP*)vv->spptr;
      vechip->stream = 0; /* using default stream */
      vechip->GPUarray_allocated = 0;
      vv->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
      vv->minimum_bytes_pinned_memory = 0;

      /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
         Note: This same code duplicated in VecCreate_SeqHIP_Private() and VecHIPAllocateCheck(). Is there a good way to avoid this? */
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)vv),((PetscObject)vv)->prefix,"VECHIP Options","Vec");CHKERRQ(ierr);
      pinned_memory_min = vv->minimum_bytes_pinned_memory;
      ierr = PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&flag);CHKERRQ(ierr);
      if (flag) vv->minimum_bytes_pinned_memory = pinned_memory_min;
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    vechip = (Vec_HIP*)vv->spptr;
    vechip->GPUarray = (PetscScalar*)array;
  }
  PetscFunctionReturn(0);
}
