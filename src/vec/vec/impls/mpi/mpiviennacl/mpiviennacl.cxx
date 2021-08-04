
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_ViennaCL*)v->spptr)->GPUarray_allocated;
      delete (Vec_ViennaCL*) v->spptr;
    }
  } catch(std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIViennaCL(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqViennaCL(xin,NORM_2,&work);CHKERRQ(ierr);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqViennaCL(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqViennaCL(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqViennaCL(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqViennaCL(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPIViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqViennaCL(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPIViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqViennaCL(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIViennaCL(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqViennaCL(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPI_Private(*v,PETSC_FALSE,w->nghost,0);CHKERRQ(ierr);
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
  ierr = PetscObjectChangeTypeName((PetscObject)(*v),VECMPIVIENNACL);CHKERRQ(ierr);

  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPIViennaCL(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqViennaCL(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce((void*)&work,(void*)&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRMPI(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_MPIViennaCL(Vec vv, PetscBool pin)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  vv->boundtocpu = pin;

  if (pin) {
    ierr = VecViennaCLCopyFromGPU(vv);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(vv->map);CHKERRQ(ierr);
  ierr = VecViennaCLAllocateCheck(vv);CHKERRQ(ierr);
  ierr = VecCreate_MPIViennaCL_Private(vv,PETSC_FALSE,0,((Vec_ViennaCL*)(vv->spptr))->GPUarray);CHKERRQ(ierr);
  ierr = VecViennaCLAllocateCheckHost(vv);CHKERRQ(ierr);
  ierr = VecSet(vv,0.0);CHKERRQ(ierr);
  ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
  vv->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_ViennaCL(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQVIENNACL);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPIVIENNACL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  ierr = VecCreate_MPIViennaCL_Private(*vv,PETSC_FALSE,0,array);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPIViennaCLWithArray(comm,bs,n,N,viennaclvec,vv);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  Vec_ViennaCL   *vecviennacl;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_FALSE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPIVIENNACL);CHKERRQ(ierr);

  ierr = VecBindToCPU_MPIViennaCL(vv,PETSC_FALSE);CHKERRQ(ierr);
  vv->ops->bindtocpu = VecBindToCPU_MPIViennaCL;

  if (alloc && !array) {
    ierr = VecViennaCLAllocateCheck(vv);CHKERRQ(ierr);
    ierr = VecViennaCLAllocateCheckHost(vv);CHKERRQ(ierr);
    ierr = VecSet(vv,0.0);CHKERRQ(ierr);
    ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
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

