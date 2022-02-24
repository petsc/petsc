
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

  PetscFunctionBegin;
  if (v->spptr) {
    veccuda = (Vec_CUDA*)v->spptr;
    if (veccuda->GPUarray_allocated) {
      CHKERRCUDA(cudaFree(((Vec_CUDA*)v->spptr)->GPUarray_allocated));
      veccuda->GPUarray_allocated = NULL;
    }
    if (veccuda->stream) {
      CHKERRCUDA(cudaStreamDestroy(((Vec_CUDA*)v->spptr)->stream));
    }
    if (v->pinned_memory) {
      CHKERRQ(PetscMallocSetCUDAHost());
      CHKERRQ(PetscFree(vecmpi->array_allocated));
      CHKERRQ(PetscMallocResetCUDAHost());
      v->pinned_memory = PETSC_FALSE;
    }
    CHKERRQ(PetscFree(v->spptr));
  }
  CHKERRQ(VecDestroy_MPI(v));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPICUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    CHKERRQ(VecNorm_SeqCUDA(xin,NORM_2,&work));
    work *= work;
    CHKERRMPI(MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    CHKERRQ(VecNorm_SeqCUDA(xin,NORM_1,&work));
    /* Find the global max */
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    CHKERRQ(VecNorm_SeqCUDA(xin,NORM_INFINITY,&work));
    /* Find the global max */
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    CHKERRQ(VecNorm_SeqCUDA(xin,NORM_1,temp));
    CHKERRQ(VecNorm_SeqCUDA(xin,NORM_2,temp+1));
    temp[1] = temp[1]*temp[1];
    CHKERRMPI(MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  CHKERRQ(VecDot_SeqCUDA(xin,yin,&work));
  CHKERRMPI(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  CHKERRQ(VecTDot_SeqCUDA(xin,yin,&work));
  CHKERRMPI(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPICUDA(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) {
    CHKERRQ(PetscMalloc1(nv,&work));
  }
  CHKERRQ(VecMDot_SeqCUDA(xin,nv,y,work));
  CHKERRMPI(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) {
    CHKERRQ(PetscFree(work));
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
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)win),v));
  CHKERRQ(PetscLayoutReference(win->map,&(*v)->map));

  CHKERRQ(VecCreate_MPICUDA_Private(*v,PETSC_TRUE,w->nghost,0));
  vw   = (Vec_MPI*)(*v)->data;
  CHKERRQ(PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps)));

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    CHKERRQ(VecGetArray(*v,&array));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,win->map->n+w->nghost,array,&vw->localrep));
    CHKERRQ(PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps)));
    CHKERRQ(VecRestoreArray(*v,&array));
    CHKERRQ(PetscLogObjectParent((PetscObject)*v,(PetscObject)vw->localrep));
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      CHKERRQ(PetscObjectReference((PetscObject)vw->localupdate));
    }
  }

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash   = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  /* change type_name appropriately */
  CHKERRQ(VecCUDAAllocateCheck(*v));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)(*v),VECMPICUDA));

  CHKERRQ(PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist));
  CHKERRQ(PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist));
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPICUDA(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  CHKERRQ(VecDotNorm2_SeqCUDA(s,t,work,work+1));
  CHKERRMPI(MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s)));
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPICUDA(Vec vv)
{
  PetscFunctionBegin;
  CHKERRQ(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  CHKERRQ(PetscLayoutSetUp(vv->map));
  CHKERRQ(VecCUDAAllocateCheck(vv));
  CHKERRQ(VecCreate_MPICUDA_Private(vv,PETSC_FALSE,0,((Vec_CUDA*)vv->spptr)->GPUarray_allocated));
  CHKERRQ(VecCUDAAllocateCheckHost(vv));
  CHKERRQ(VecSet(vv,0.0));
  CHKERRQ(VecSet_Seq(vv,0.0));
  vv->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_CUDA(Vec v)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v),&size));
  if (size == 1) {
    CHKERRQ(VecSetType(v,VECSEQCUDA));
  } else {
    CHKERRQ(VecSetType(v,VECMPICUDA));
  }
  PetscFunctionReturn(0);
}

/*@
 VecCreateMPICUDA - Creates a standard, parallel array-style vector for CUDA devices.

 Collective

 Input Parameters:
 +  comm - the MPI communicator to use
 .  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
 -  N - global vector length (or PETSC_DETERMINE to have calculated if n is given)

    Output Parameter:
 .  v - the vector

    Notes:
    Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
    same type as an existing vector.

    Level: intermediate

 .seealso: VecCreateMPICUDAWithArray(), VecCreateMPICUDAWithArrays(), VecCreateSeqCUDA(), VecCreateSeq(),
           VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
           VecCreateMPIWithArray(), VecCreateGhostWithArray(), VecMPISetGhost()

 @*/
 PetscErrorCode VecCreateMPICUDA(MPI_Comm comm,PetscInt n,PetscInt N,Vec *v)
 {

   PetscFunctionBegin;
   CHKERRQ(VecCreate(comm,v));
   CHKERRQ(VecSetSizes(*v,n,N));
   CHKERRQ(VecSetType(*v,VECMPICUDA));
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

.seealso: VecCreateMPICUDA(), VecCreateSeqCUDAWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PetscErrorCode  VecCreateMPICUDAWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscFunctionBegin;
  PetscCheckFalse(n == PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  CHKERRQ(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  CHKERRQ(VecCreate(comm,vv));
  CHKERRQ(VecSetSizes(*vv,n,N));
  CHKERRQ(VecSetBlockSize(*vv,bs));
  CHKERRQ(VecCreate_MPICUDA_Private(*vv,PETSC_FALSE,0,array));
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
  PetscFunctionBegin;
  CHKERRQ(VecCreateMPICUDAWithArray(comm,bs,n,N,gpuarray,vv));

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
  PetscReal      work;

  PetscFunctionBegin;
  CHKERRQ(VecMax_SeqCUDA(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  if (!idx) {
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    CHKERRMPI(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPICUDA(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscReal      work;

  PetscFunctionBegin;
  CHKERRQ(VecMin_SeqCUDA(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  if (!idx) {
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    CHKERRMPI(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_MPICUDA(Vec V,PetscBool bind)
{
  PetscFunctionBegin;
  V->boundtocpu = bind;
  if (bind) {
    CHKERRQ(VecCUDACopyFromGPU(V));
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
    V->ops->getarrayandmemtype     = NULL;
    V->ops->restorearrayandmemtype = NULL;
    V->ops->getarraywriteandmemtype= NULL;
    V->ops->max                    = VecMax_MPI;
    V->ops->min                    = VecMin_MPI;
    V->ops->reciprocal             = VecReciprocal_Default;
    V->ops->sum                    = NULL;
    V->ops->shift                  = NULL;
    /* default random number generator */
    CHKERRQ(PetscFree(V->defaultrandtype));
    CHKERRQ(PetscStrallocpy(PETSCRANDER48,&V->defaultrandtype));
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
    V->ops->getlocalvectorread     = VecGetLocalVectorRead_SeqCUDA;
    V->ops->restorelocalvectorread = VecRestoreLocalVectorRead_SeqCUDA;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqCUDA;
    V->ops->getarray               = VecGetArray_SeqCUDA;
    V->ops->restorearray           = VecRestoreArray_SeqCUDA;
    V->ops->getarrayandmemtype     = VecGetArrayAndMemType_SeqCUDA;
    V->ops->restorearrayandmemtype = VecRestoreArrayAndMemType_SeqCUDA;
    V->ops->getarraywriteandmemtype= VecGetArrayWriteAndMemType_SeqCUDA;
    V->ops->max                    = VecMax_MPICUDA;
    V->ops->min                    = VecMin_MPICUDA;
    V->ops->reciprocal             = VecReciprocal_SeqCUDA;
    V->ops->sum                    = VecSum_SeqCUDA;
    V->ops->shift                  = VecShift_SeqCUDA;
    /* default random number generator */
    CHKERRQ(PetscFree(V->defaultrandtype));
    CHKERRQ(PetscStrallocpy(PETSCCURAND,&V->defaultrandtype));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPICUDA_Private(Vec vv,PetscBool alloc,PetscInt nghost,const PetscScalar array[])
{
  PetscErrorCode ierr;
  Vec_CUDA       *veccuda;

  PetscFunctionBegin;
  CHKERRQ(VecCreate_MPI_Private(vv,PETSC_FALSE,0,0));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)vv,VECMPICUDA));

  CHKERRQ(VecBindToCPU_MPICUDA(vv,PETSC_FALSE));
  vv->ops->bindtocpu = VecBindToCPU_MPICUDA;

  /* Later, functions check for the Vec_CUDA structure existence, so do not create it without array */
  if (alloc && !array) {
    CHKERRQ(VecCUDAAllocateCheck(vv));
    CHKERRQ(VecCUDAAllocateCheckHost(vv));
    CHKERRQ(VecSet(vv,0.0));
    CHKERRQ(VecSet_Seq(vv,0.0));
    vv->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  if (array) {
    if (!vv->spptr) {
      PetscReal pinned_memory_min;
      PetscBool flag;
      /* Cannot use PetscNew() here because spptr is void* */
      CHKERRQ(PetscCalloc(sizeof(Vec_CUDA),&vv->spptr));
      veccuda = (Vec_CUDA*)vv->spptr;
      vv->minimum_bytes_pinned_memory = 0;

      /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
         Note: This same code duplicated in VecCreate_SeqCUDA_Private() and VecCUDAAllocateCheck(). Is there a good way to avoid this? */
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)vv),((PetscObject)vv)->prefix,"VECCUDA Options","Vec");CHKERRQ(ierr);
      pinned_memory_min = vv->minimum_bytes_pinned_memory;
      CHKERRQ(PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&flag));
      if (flag) vv->minimum_bytes_pinned_memory = pinned_memory_min;
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    veccuda = (Vec_CUDA*)vv->spptr;
    veccuda->GPUarray = (PetscScalar*)array;
    vv->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}
