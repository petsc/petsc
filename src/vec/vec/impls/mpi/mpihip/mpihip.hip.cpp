
/*
   This file contains routines for Parallel vector operations.
 */
#define PETSC_SKIP_SPINLOCK

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

PetscErrorCode VecDestroy_MPIHIP(Vec v)
{
  Vec_MPI        *vecmpi = (Vec_MPI*)v->data;
  Vec_HIP        *vechip;

  PetscFunctionBegin;
  if (v->spptr) {
    vechip = (Vec_HIP*)v->spptr;
    if (vechip->GPUarray_allocated) {
      PetscCallHIP(hipFree(vechip->GPUarray_allocated));
      vechip->GPUarray_allocated = NULL;
    }
    if (vechip->stream) {
      PetscCallHIP(hipStreamDestroy(vechip->stream));
    }
    if (v->pinned_memory) {
      PetscCall(PetscMallocSetHIPHost());
      PetscCall(PetscFree(vecmpi->array_allocated));
      PetscCall(PetscMallocResetHIPHost());
      v->pinned_memory = PETSC_FALSE;
    }
    PetscCall(PetscFree(v->spptr));
  }
  PetscCall(VecDestroy_MPI(v));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIHIP(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCall(VecNorm_SeqHIP(xin,NORM_2,&work));
    work *= work;
    PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    PetscCall(VecNorm_SeqHIP(xin,NORM_1,&work));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    PetscCall(VecNorm_SeqHIP(xin,NORM_INFINITY,&work));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    PetscCall(VecNorm_SeqHIP(xin,NORM_1,temp));
    PetscCall(VecNorm_SeqHIP(xin,NORM_2,temp+1));
    temp[1] = temp[1]*temp[1];
    PetscCall(MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPIHIP(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecDot_SeqHIP(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPIHIP(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecTDot_SeqHIP(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIHIP(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) {
    PetscCall(PetscMalloc1(nv,&work));
  }
  PetscCall(VecMDot_SeqHIP(xin,nv,y,work));
  PetscCall(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) {
    PetscCall(PetscFree(work));
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

PetscErrorCode VecDuplicate_MPIHIP(Vec win,Vec *v)
{
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  PetscCall(VecCreate(PetscObjectComm((PetscObject)win),v));
  PetscCall(PetscLayoutReference(win->map,&(*v)->map));

  PetscCall(VecCreate_MPIHIP_Private(*v,PETSC_TRUE,w->nghost,0));
  vw   = (Vec_MPI*)(*v)->data;
  PetscCall(PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps)));

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    PetscCall(VecGetArray(*v,&array));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,win->map->n+w->nghost,array,&vw->localrep));
    PetscCall(PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps)));
    PetscCall(VecRestoreArray(*v,&array));
    PetscCall(PetscLogObjectParent((PetscObject)*v,(PetscObject)vw->localrep));
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      PetscCall(PetscObjectReference((PetscObject)vw->localupdate));
    }
  }

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash   = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  /* change type_name appropriately */
  PetscCall(VecHIPAllocateCheck(*v));
  PetscCall(PetscObjectChangeTypeName((PetscObject)(*v),VECMPIHIP));

  PetscCall(PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist));
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPIHIP(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  PetscCall(VecDotNorm2_SeqHIP(s,t,work,work+1));
  PetscCall(MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s)));
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPIHIP(Vec vv)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
  PetscCall(PetscLayoutSetUp(vv->map));
  PetscCall(VecHIPAllocateCheck(vv));
  PetscCall(VecCreate_MPIHIP_Private(vv,PETSC_FALSE,0,((Vec_HIP*)vv->spptr)->GPUarray_allocated));
  PetscCall(VecHIPAllocateCheckHost(vv));
  PetscCall(VecSet(vv,0.0));
  PetscCall(VecSet_Seq(vv,0.0));
  vv->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_HIP(Vec v)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v),&size));
  if (size == 1) {
    PetscCall(VecSetType(v,VECSEQHIP));
  } else {
    PetscCall(VecSetType(v,VECMPIHIP));
  }
  PetscFunctionReturn(0);
}

/*@
 VecCreateMPIHIP - Creates a standard, parallel array-style vector for HIP devices.

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

 .seealso: VecCreateMPIHIPWithArray(), VecCreateMPIHIPWithArrays(), VecCreateSeqHIP(), VecCreateSeq(),
           VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
           VecCreateMPIWithArray(), VecCreateGhostWithArray(), VecMPISetGhost()

 @*/
 PetscErrorCode VecCreateMPIHIP(MPI_Comm comm,PetscInt n,PetscInt N,Vec *v)
 {
   PetscFunctionBegin;
   PetscCall(VecCreate(comm,v));
   PetscCall(VecSetSizes(*v,n,N));
   PetscCall(VecSetType(*v,VECMPIHIP));
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
PetscErrorCode  VecCreateMPIHIPWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscFunctionBegin;
  PetscCheck(n != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
  PetscCall(VecCreate(comm,vv));
  PetscCall(VecSetSizes(*vv,n,N));
  PetscCall(VecSetBlockSize(*vv,bs));
  PetscCall(VecCreate_MPIHIP_Private(*vv,PETSC_FALSE,0,array));
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
PetscErrorCode  VecCreateMPIHIPWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *vv)
{
  PetscFunctionBegin;
  PetscCall(VecCreateMPIHIPWithArray(comm,bs,n,N,gpuarray,vv));

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

PetscErrorCode VecMax_MPIHIP(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscReal      work;

  PetscFunctionBegin;
  PetscCall(VecMax_SeqHIP(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  if (!idx) {
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPIHIP(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscReal      work;

  PetscFunctionBegin;
  PetscCall(VecMin_SeqHIP(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  if (!idx) {
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_MPIHIP(Vec V,PetscBool bind)
{
  PetscFunctionBegin;
  V->boundtocpu = bind;
  if (bind) {
    PetscCall(VecHIPCopyFromGPU(V));
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
    V->ops->getarrayandmemtype     = NULL;
    V->ops->restorearrayandmemtype = NULL;
    V->ops->getarraywriteandmemtype= NULL;
    V->ops->max                    = VecMax_MPI;
    V->ops->min                    = VecMin_MPI;
    V->ops->reciprocal             = VecReciprocal_Default;
    V->ops->sum                    = NULL;
    V->ops->shift                  = NULL;
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
    V->ops->getlocalvectorread     = VecGetLocalVectorRead_SeqHIP;
    V->ops->restorelocalvectorread = VecRestoreLocalVectorRead_SeqHIP;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqHIP;
    V->ops->getarray               = VecGetArray_SeqHIP;
    V->ops->restorearray           = VecRestoreArray_SeqHIP;
    V->ops->getarrayandmemtype     = VecGetArrayAndMemType_SeqHIP;
    V->ops->restorearrayandmemtype = VecRestoreArrayAndMemType_SeqHIP;
    V->ops->getarraywriteandmemtype= VecGetArrayWriteAndMemType_SeqHIP;
    V->ops->max                    = VecMax_MPIHIP;
    V->ops->min                    = VecMin_MPIHIP;
    V->ops->reciprocal             = VecReciprocal_SeqHIP;
    V->ops->sum                    = VecSum_SeqHIP;
    V->ops->shift                  = VecShift_SeqHIP;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPIHIP_Private(Vec vv,PetscBool alloc,PetscInt nghost,const PetscScalar array[])
{
  Vec_HIP *vechip;

  PetscFunctionBegin;
  PetscCall(VecCreate_MPI_Private(vv,PETSC_FALSE,0,0));
  PetscCall(PetscObjectChangeTypeName((PetscObject)vv,VECMPIHIP));

  PetscCall(VecBindToCPU_MPIHIP(vv,PETSC_FALSE));
  vv->ops->bindtocpu = VecBindToCPU_MPIHIP;

  /* Later, functions check for the Vec_HIP structure existence, so do not create it without array */
  if (alloc && !array) {
    PetscCall(VecHIPAllocateCheck(vv));
    PetscCall(VecHIPAllocateCheckHost(vv));
    PetscCall(VecSet(vv,0.0));
    PetscCall(VecSet_Seq(vv,0.0));
    vv->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  if (array) {
    if (!vv->spptr) {
      PetscReal      pinned_memory_min;
      PetscBool      flag;
      PetscErrorCode ierr;

      /* Cannot use PetscNew() here because spptr is void* */
      PetscCall(PetscCalloc(sizeof(Vec_HIP),&vv->spptr));
      vechip = (Vec_HIP*)vv->spptr;
      vv->minimum_bytes_pinned_memory = 0;

      /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
         Note: This same code duplicated in VecCreate_SeqHIP_Private() and VecHIPAllocateCheck(). Is there a good way to avoid this? */
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)vv),((PetscObject)vv)->prefix,"VECHIP Options","Vec");PetscCall(ierr);
      pinned_memory_min = vv->minimum_bytes_pinned_memory;
      PetscCall(PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&flag));
      if (flag) vv->minimum_bytes_pinned_memory = pinned_memory_min;
      ierr = PetscOptionsEnd();PetscCall(ierr);
    }
    vechip = (Vec_HIP*)vv->spptr;
    vechip->GPUarray = (PetscScalar*)array;
    vv->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}
