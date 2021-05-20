
/*
   This file contains routines for Parallel vector operations.
 */
#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <petsc/private/cudavecimpl.h>

/*MC
   VECCUDA - VECCUDA = "cuda" - A VECSEQCUDA on a single-process communicator, and VECMPICUDA otherwise.

   Options Database Keys:
. -vec_type cuda - sets the vector type to VECCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQCUDA, VECMPICUDA, VECSTANDARD, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecDestroy_MPICUDA(Vec v)
{
  Vec_MPI        *vecmpi = (Vec_MPI*)v->data;
  Vec_CUDA       *veccuda;
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  if (v->spptr) {
    veccuda = (Vec_CUDA*)v->spptr;
    if (veccuda->GPUarray_allocated) {
      err = cudaFree(((Vec_CUDA*)v->spptr)->GPUarray_allocated);CHKERRCUDA(err);
      veccuda->GPUarray_allocated = NULL;
    }
    if (veccuda->stream) {
      err = cudaStreamDestroy(((Vec_CUDA*)v->spptr)->stream);CHKERRCUDA(err);
    }
    if (v->pinned_memory) {
      ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
      ierr = PetscFree(vecmpi->array_allocated);CHKERRQ(ierr);
      ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
      v->pinned_memory = PETSC_FALSE;
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPICUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqCUDA(xin,NORM_2,&work);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqCUDA(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqCUDA(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqCUDA(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDA(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqCUDA(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqCUDA(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPICUDA(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqCUDA(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   VECMPICUDA - VECMPICUDA = "mpicuda" - The basic parallel vector, modified to use CUDA

   Options Database Keys:
. -vec_type mpicuda - sets the vector type to VECMPICUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/


PetscErrorCode VecDuplicate_MPICUDA(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPICUDA_Private(*v,PETSC_TRUE,w->nghost,0);CHKERRQ(ierr);
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
  ierr = VecCUDAAllocateCheck(*v);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)(*v),VECMPICUDA);CHKERRQ(ierr);

  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPICUDA(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqCUDA(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRMPI(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPICUDA(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(vv->map);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(vv);CHKERRQ(ierr);
  ierr = VecCreate_MPICUDA_Private(vv,PETSC_FALSE,0,((Vec_CUDA*)vv->spptr)->GPUarray_allocated);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheckHost(vv);CHKERRQ(ierr);
  ierr = VecSet(vv,0.0);CHKERRQ(ierr);
  ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
  vv->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_CUDA(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQCUDA);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPICUDA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPICUDAWithArray - Creates a parallel, array-style vector,
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

   If the user-provided array is NULL, then VecCUDAPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqCUDAWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PetscErrorCode  VecCreateMPICUDAWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  ierr = VecCreate_MPICUDA_Private(*vv,PETSC_FALSE,0,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPICUDAWithArrays - Creates a parallel, array-style vector,
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

.seealso: VecCreateSeqCUDAWithArrays(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecCUDAPlaceArray(), VecPlaceArray(),
          VecCUDAAllocateCheckHost()
@*/
PetscErrorCode  VecCreateMPICUDAWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPICUDAWithArray(comm,bs,n,N,gpuarray,vv);CHKERRQ(ierr);

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

PetscErrorCode VecMax_MPICUDA(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  ierr = VecMax_SeqCUDA(xin,idx,&work);CHKERRQ(ierr);
  if (!idx) {
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;
    rstart   = xin->map->rstart;
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr     = MPIU_Allreduce(work2,z2,2,MPIU_REAL,MPIU_MAXINDEX_OP,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z       = z2[0];
    *idx     = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPICUDA(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  ierr = VecMin_SeqCUDA(xin,idx,&work);CHKERRQ(ierr);
  if (!idx) {
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;

    ierr = VecGetOwnershipRange(xin,&rstart,NULL);CHKERRQ(ierr);
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPIU_Allreduce(work2,z2,2,MPIU_REAL,MPIU_MININDEX_OP,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_MPICUDA(Vec V,PetscBool pin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->boundtocpu = pin;
  if (pin) {
    ierr = VecCUDACopyFromGPU(V);CHKERRQ(ierr);
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
    V->ops->replacearray           = VecReplaceArray_SeqCUDA;
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
    V->ops->max                    = VecMax_MPI;
    V->ops->min                    = VecMin_MPI;
    V->ops->reciprocal             = VecReciprocal_Default;
    /* default random number generator */
    ierr = PetscFree(V->defaultrandtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCRANDER48,&V->defaultrandtype);CHKERRQ(ierr);
  } else {
    V->ops->dotnorm2               = VecDotNorm2_MPICUDA;
    V->ops->waxpy                  = VecWAXPY_SeqCUDA;
    V->ops->duplicate              = VecDuplicate_MPICUDA;
    V->ops->dot                    = VecDot_MPICUDA;
    V->ops->mdot                   = VecMDot_MPICUDA;
    V->ops->tdot                   = VecTDot_MPICUDA;
    V->ops->norm                   = VecNorm_MPICUDA;
    V->ops->scale                  = VecScale_SeqCUDA;
    V->ops->copy                   = VecCopy_SeqCUDA;
    V->ops->set                    = VecSet_SeqCUDA;
    V->ops->swap                   = VecSwap_SeqCUDA;
    V->ops->axpy                   = VecAXPY_SeqCUDA;
    V->ops->axpby                  = VecAXPBY_SeqCUDA;
    V->ops->maxpy                  = VecMAXPY_SeqCUDA;
    V->ops->aypx                   = VecAYPX_SeqCUDA;
    V->ops->axpbypcz               = VecAXPBYPCZ_SeqCUDA;
    V->ops->pointwisemult          = VecPointwiseMult_SeqCUDA;
    V->ops->setrandom              = VecSetRandom_SeqCUDA;
    V->ops->placearray             = VecPlaceArray_SeqCUDA;
    V->ops->replacearray           = VecReplaceArray_SeqCUDA;
    V->ops->resetarray             = VecResetArray_SeqCUDA;
    V->ops->dot_local              = VecDot_SeqCUDA;
    V->ops->tdot_local             = VecTDot_SeqCUDA;
    V->ops->norm_local             = VecNorm_SeqCUDA;
    V->ops->mdot_local             = VecMDot_SeqCUDA;
    V->ops->destroy                = VecDestroy_MPICUDA;
    V->ops->pointwisedivide        = VecPointwiseDivide_SeqCUDA;
    V->ops->getlocalvector         = VecGetLocalVector_SeqCUDA;
    V->ops->restorelocalvector     = VecRestoreLocalVector_SeqCUDA;
    V->ops->getlocalvectorread     = VecGetLocalVector_SeqCUDA;
    V->ops->restorelocalvectorread = VecRestoreLocalVector_SeqCUDA;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqCUDA;
    V->ops->getarray               = VecGetArray_SeqCUDA;
    V->ops->restorearray           = VecRestoreArray_SeqCUDA;
    V->ops->getarrayandmemtype     = VecGetArrayAndMemType_SeqCUDA;
    V->ops->restorearrayandmemtype = VecRestoreArrayAndMemType_SeqCUDA;
    V->ops->max                    = VecMax_MPICUDA;
    V->ops->min                    = VecMin_MPICUDA;
    V->ops->reciprocal             = VecReciprocal_SeqCUDA;
    /* default random number generator */
    ierr = PetscFree(V->defaultrandtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCCURAND,&V->defaultrandtype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPICUDA_Private(Vec vv,PetscBool alloc,PetscInt nghost,const PetscScalar array[])
{
  PetscErrorCode ierr;
  Vec_CUDA       *veccuda;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_FALSE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPICUDA);CHKERRQ(ierr);

  ierr = VecBindToCPU_MPICUDA(vv,PETSC_FALSE);CHKERRQ(ierr);
  vv->ops->bindtocpu = VecBindToCPU_MPICUDA;

  /* Later, functions check for the Vec_CUDA structure existence, so do not create it without array */
  if (alloc && !array) {
    ierr = VecCUDAAllocateCheck(vv);CHKERRQ(ierr);
    ierr = VecCUDAAllocateCheckHost(vv);CHKERRQ(ierr);
    ierr = VecSet(vv,0.0);CHKERRQ(ierr);
    ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
    vv->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  if (array) {
    if (!vv->spptr) {
      PetscReal pinned_memory_min;
      PetscBool flag;
      /* Cannot use PetscNew() here because spptr is void* */
      ierr = PetscMalloc(sizeof(Vec_CUDA),&vv->spptr);CHKERRQ(ierr);
      veccuda = (Vec_CUDA*)vv->spptr;
      veccuda->stream = 0; /* using default stream */
      veccuda->GPUarray_allocated = 0;
      vv->minimum_bytes_pinned_memory = 0;

      /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
         Note: This same code duplicated in VecCreate_SeqCUDA_Private() and VecCUDAAllocateCheck(). Is there a good way to avoid this? */
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)vv),((PetscObject)vv)->prefix,"VECCUDA Options","Vec");CHKERRQ(ierr);
      pinned_memory_min = vv->minimum_bytes_pinned_memory;
      ierr = PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&flag);CHKERRQ(ierr);
      if (flag) vv->minimum_bytes_pinned_memory = pinned_memory_min;
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    veccuda = (Vec_CUDA*)vv->spptr;
    veccuda->GPUarray = (PetscScalar*)array;
    vv->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}
