
/*
   This file contains routines for Parallel vector operations.
 */
#include <petscconf.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

/*MC
   VECVIENNACL - VECVIENNACL = "viennacl" - A VECSEQVIENNACL on a single-process communicator, and VECMPIVIENNACL otherwise.

   Options Database Keys:
. -vec_type viennacl - sets the vector type to VECVIENNACL during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQVIENNACL, VECMPIVIENNACL, VECSTANDARD, VecType, VecCreateMPI(), VecCreateMPI()
M*/

PetscErrorCode VecDestroy_MPIViennaCL(Vec v)
{
  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_ViennaCL*)v->spptr)->GPUarray_allocated;
      delete (Vec_ViennaCL*) v->spptr;
    }
  } catch(std::exception const & ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }
  PetscCall(VecDestroy_MPI(v));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIViennaCL(Vec xin,NormType type,PetscReal *z)
{
  PetscReal sum,work = 0.0;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCall(VecNorm_SeqViennaCL(xin,NORM_2,&work));
    work *= work;
    PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    PetscCall(VecNorm_SeqViennaCL(xin,NORM_1,&work));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    PetscCall(VecNorm_SeqViennaCL(xin,NORM_INFINITY,&work));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    PetscCall(VecNorm_SeqViennaCL(xin,NORM_1,temp));
    PetscCall(VecNorm_SeqViennaCL(xin,NORM_2,temp+1));
    temp[1] = temp[1]*temp[1];
    PetscCall(MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPIViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecDot_SeqViennaCL(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPIViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;

  PetscFunctionBegin;
  PetscCall(VecTDot_SeqViennaCL(xin,yin,&work));
  PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIViennaCL(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) {
    PetscCall(PetscMalloc1(nv,&work));
  }
  PetscCall(VecMDot_SeqViennaCL(xin,nv,y,work));
  PetscCall(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) {
    PetscCall(PetscFree(work));
  }
  PetscFunctionReturn(0);
}

/*MC
   VECMPIVIENNACL - VECMPIVIENNACL = "mpiviennacl" - The basic parallel vector, modified to use ViennaCL

   Options Database Keys:
. -vec_type mpiviennacl - sets the vector type to VECMPIVIENNACL during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMPI()
M*/

PetscErrorCode VecDuplicate_MPIViennaCL(Vec win,Vec *v)
{
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  PetscCall(VecCreate(PetscObjectComm((PetscObject)win),v));
  PetscCall(PetscLayoutReference(win->map,&(*v)->map));

  PetscCall(VecCreate_MPI_Private(*v,PETSC_FALSE,w->nghost,0));
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
  PetscCall(PetscObjectChangeTypeName((PetscObject)(*v),VECMPIVIENNACL));

  PetscCall(PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist));
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPIViennaCL(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  PetscCall(VecDotNorm2_SeqViennaCL(s,t,work,work+1));
  PetscCall(MPIU_Allreduce((void*)&work,(void*)&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s)));
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_MPIViennaCL(Vec vv, PetscBool bind)
{
  PetscFunctionBegin;
  vv->boundtocpu = bind;

  if (bind) {
    PetscCall(VecViennaCLCopyFromGPU(vv));
    vv->offloadmask = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    vv->ops->dotnorm2               = NULL;
    vv->ops->waxpy                  = VecWAXPY_Seq;
    vv->ops->dot                    = VecDot_MPI;
    vv->ops->mdot                   = VecMDot_MPI;
    vv->ops->tdot                   = VecTDot_MPI;
    vv->ops->norm                   = VecNorm_MPI;
    vv->ops->scale                  = VecScale_Seq;
    vv->ops->copy                   = VecCopy_Seq;
    vv->ops->set                    = VecSet_Seq;
    vv->ops->swap                   = VecSwap_Seq;
    vv->ops->axpy                   = VecAXPY_Seq;
    vv->ops->axpby                  = VecAXPBY_Seq;
    vv->ops->maxpy                  = VecMAXPY_Seq;
    vv->ops->aypx                   = VecAYPX_Seq;
    vv->ops->axpbypcz               = VecAXPBYPCZ_Seq;
    vv->ops->pointwisemult          = VecPointwiseMult_Seq;
    vv->ops->setrandom              = VecSetRandom_Seq;
    vv->ops->placearray             = VecPlaceArray_Seq;
    vv->ops->replacearray           = VecReplaceArray_Seq;
    vv->ops->resetarray             = VecResetArray_Seq;
    vv->ops->dot_local              = VecDot_Seq;
    vv->ops->tdot_local             = VecTDot_Seq;
    vv->ops->norm_local             = VecNorm_Seq;
    vv->ops->mdot_local             = VecMDot_Seq;
    vv->ops->pointwisedivide        = VecPointwiseDivide_Seq;
    vv->ops->getlocalvector         = NULL;
    vv->ops->restorelocalvector     = NULL;
    vv->ops->getlocalvectorread     = NULL;
    vv->ops->restorelocalvectorread = NULL;
    vv->ops->getarraywrite          = NULL;
  } else {
    vv->ops->dotnorm2        = VecDotNorm2_MPIViennaCL;
    vv->ops->waxpy           = VecWAXPY_SeqViennaCL;
    vv->ops->duplicate       = VecDuplicate_MPIViennaCL;
    vv->ops->dot             = VecDot_MPIViennaCL;
    vv->ops->mdot            = VecMDot_MPIViennaCL;
    vv->ops->tdot            = VecTDot_MPIViennaCL;
    vv->ops->norm            = VecNorm_MPIViennaCL;
    vv->ops->scale           = VecScale_SeqViennaCL;
    vv->ops->copy            = VecCopy_SeqViennaCL;
    vv->ops->set             = VecSet_SeqViennaCL;
    vv->ops->swap            = VecSwap_SeqViennaCL;
    vv->ops->axpy            = VecAXPY_SeqViennaCL;
    vv->ops->axpby           = VecAXPBY_SeqViennaCL;
    vv->ops->maxpy           = VecMAXPY_SeqViennaCL;
    vv->ops->aypx            = VecAYPX_SeqViennaCL;
    vv->ops->axpbypcz        = VecAXPBYPCZ_SeqViennaCL;
    vv->ops->pointwisemult   = VecPointwiseMult_SeqViennaCL;
    vv->ops->setrandom       = VecSetRandom_SeqViennaCL;
    vv->ops->dot_local       = VecDot_SeqViennaCL;
    vv->ops->tdot_local      = VecTDot_SeqViennaCL;
    vv->ops->norm_local      = VecNorm_SeqViennaCL;
    vv->ops->mdot_local      = VecMDot_SeqViennaCL;
    vv->ops->destroy         = VecDestroy_MPIViennaCL;
    vv->ops->pointwisedivide = VecPointwiseDivide_SeqViennaCL;
    vv->ops->placearray      = VecPlaceArray_SeqViennaCL;
    vv->ops->replacearray    = VecReplaceArray_SeqViennaCL;
    vv->ops->resetarray      = VecResetArray_SeqViennaCL;
    vv->ops->getarraywrite   = VecGetArrayWrite_SeqViennaCL;
    vv->ops->getarray        = VecGetArray_SeqViennaCL;
    vv->ops->restorearray    = VecRestoreArray_SeqViennaCL;
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_MPIViennaCL(Vec vv)
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(vv->map));
  PetscCall(VecViennaCLAllocateCheck(vv));
  PetscCall(VecCreate_MPIViennaCL_Private(vv,PETSC_FALSE,0,((Vec_ViennaCL*)(vv->spptr))->GPUarray));
  PetscCall(VecViennaCLAllocateCheckHost(vv));
  PetscCall(VecSet(vv,0.0));
  PetscCall(VecSet_Seq(vv,0.0));
  vv->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_ViennaCL(Vec v)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v),&size));
  if (size == 1) {
    PetscCall(VecSetType(v,VECSEQVIENNACL));
  } else {
    PetscCall(VecSetType(v,VECMPIVIENNACL));
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIViennaCLWithArray - Creates a parallel, array-style vector,
   where the user provides the viennacl vector to store the vector values.

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

   If the user-provided array is NULL, then VecViennaCLPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqViennaCLWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecCreateMPI(), VecCreateGhostWithArray(), VecViennaCLPlaceArray()

@*/
PetscErrorCode  VecCreateMPIViennaCLWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const ViennaCLVector *array,Vec *vv)
{
  PetscFunctionBegin;
  PetscCheck(n != PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  PetscCall(PetscSplitOwnership(comm,&n,&N));
  PetscCall(VecCreate(comm,vv));
  PetscCall(VecSetSizes(*vv,n,N));
  PetscCall(VecSetBlockSize(*vv,bs));
  PetscCall(VecCreate_MPIViennaCL_Private(*vv,PETSC_FALSE,0,array));
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIViennaCLWithArrays - Creates a parallel, array-style vector,
   where the user provides the ViennaCL vector to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  cpuarray - the user provided CPU array to store the vector values
-  viennaclvec - ViennaCL vector where the Vec entries are to be stored on the device.

   Output Parameter:
.  vv - the vector

   Notes:
   If both cpuarray and viennaclvec are provided, the caller must ensure that
   the provided arrays have identical values.

   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   PETSc does NOT free the provided arrays when the vector is destroyed via
   VecDestroy(). The user should not free the array until the vector is
   destroyed.

   Level: intermediate

.seealso: VecCreateSeqViennaCLWithArrays(), VecCreateMPIWithArray()
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecViennaCLPlaceArray(),
          VecPlaceArray(), VecCreateMPICUDAWithArrays(),
          VecViennaCLAllocateCheckHost()
@*/
PetscErrorCode  VecCreateMPIViennaCLWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar cpuarray[],const ViennaCLVector *viennaclvec,Vec *vv)
{
  PetscFunctionBegin;
  PetscCall(VecCreateMPIViennaCLWithArray(comm,bs,n,N,viennaclvec,vv));
  if (cpuarray && viennaclvec) {
    Vec_MPI *s         = (Vec_MPI*)((*vv)->data);
    s->array           = (PetscScalar*)cpuarray;
    (*vv)->offloadmask = PETSC_OFFLOAD_BOTH;
  } else if (cpuarray) {
    Vec_MPI *s         = (Vec_MPI*)((*vv)->data);
    s->array           = (PetscScalar*)cpuarray;
    (*vv)->offloadmask =  PETSC_OFFLOAD_CPU;
  } else if (viennaclvec) {
    (*vv)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*vv)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPIViennaCL_Private(Vec vv,PetscBool alloc,PetscInt nghost,const ViennaCLVector *array)
{
  Vec_ViennaCL   *vecviennacl;

  PetscFunctionBegin;
  PetscCall(VecCreate_MPI_Private(vv,PETSC_FALSE,0,0));
  PetscCall(PetscObjectChangeTypeName((PetscObject)vv,VECMPIVIENNACL));

  PetscCall(VecBindToCPU_MPIViennaCL(vv,PETSC_FALSE));
  vv->ops->bindtocpu = VecBindToCPU_MPIViennaCL;

  if (alloc && !array) {
    PetscCall(VecViennaCLAllocateCheck(vv));
    PetscCall(VecViennaCLAllocateCheckHost(vv));
    PetscCall(VecSet(vv,0.0));
    PetscCall(VecSet_Seq(vv,0.0));
    vv->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  if (array) {
    if (!vv->spptr)
      vv->spptr = new Vec_ViennaCL;
    vecviennacl = (Vec_ViennaCL*)vv->spptr;
    vecviennacl->GPUarray_allocated = 0;
    vecviennacl->GPUarray = (ViennaCLVector*)array;
    vv->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}
