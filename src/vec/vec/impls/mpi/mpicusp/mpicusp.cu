
/*
   This file contains routines for Parallel vector operations.
 */
#include <petscconf.h>
PETSC_CUDA_EXTERN_C_BEGIN
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
PETSC_CUDA_EXTERN_C_END
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_MPICUSP"
PetscErrorCode VecDestroy_MPICUSP(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try{
    if (v->spptr) {
      delete ((Vec_CUSP*)v->spptr)->GPUarray;
      delete (Vec_CUSP *)v->spptr;
    }
  } catch(char* ex){
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNorm_MPICUSP"
PetscErrorCode VecNorm_MPICUSP(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecNorm_SeqCUSP(xin,NORM_2,&work);
    work *= work;
    ierr = MPI_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqCUSP(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqCUSP(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqCUSP(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUSP(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPI_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDot_MPICUSP"
PetscErrorCode VecDot_MPICUSP(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqCUSP(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecTDot_MPICUSP"
PetscErrorCode VecTDot_MPICUSP(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqCUSP(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMDot_MPICUSP"
PetscErrorCode VecMDot_MPICUSP(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqCUSP(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   VECMPICUSP - VECMPICUSP = "mpicusp" - The basic parallel vector, modified to use CUSP

   Options Database Keys:
. -vec_type mpicusp - sets the vector type to VECMPICUSP during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMpiWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMpi()
M*/


#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPICUSP"
PetscErrorCode VecDuplicate_MPICUSP(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI *)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPI_Private(*v,PETSC_FALSE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(*v,vw->localrep);CHKERRQ(ierr);
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }    

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  /* change type_name appropriately */
  ierr = PetscObjectChangeTypeName((PetscObject)(*v),VECMPICUSP);CHKERRQ(ierr);

  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  (*v)->map->bs    = win->map->bs;
  (*v)->bstash.bs = win->bstash.bs;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDotNorm2_MPICUSP"
PetscErrorCode VecDotNorm2_MPICUSP(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode  ierr;
  PetscScalar     work[2],sum[2];

  PetscFunctionBegin;
  ierr    = VecDotNorm2_SeqCUSP(s,t,work,work+1);CHKERRQ(ierr);
  ierr    = MPI_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,((PetscObject)s)->comm);CHKERRQ(ierr);
  *dp     = sum[0];
  *nm     = sum[1];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPICUSP"
PetscErrorCode  VecCreate_MPICUSP(Vec vv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_FALSE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPICUSP);CHKERRQ(ierr);
  vv->ops->dotnorm2        = VecDotNorm2_MPICUSP;
  vv->ops->waxpy           = VecWAXPY_SeqCUSP;
  vv->ops->duplicate       = VecDuplicate_MPICUSP;
  vv->ops->dot             = VecDot_MPICUSP;
  vv->ops->mdot            = VecMDot_MPICUSP;
  vv->ops->tdot            = VecTDot_MPICUSP;
  vv->ops->norm            = VecNorm_MPICUSP;
  vv->ops->scale           = VecScale_SeqCUSP;
  vv->ops->copy            = VecCopy_SeqCUSP;
  vv->ops->set             = VecSet_SeqCUSP;
  vv->ops->swap            = VecSwap_SeqCUSP;
  vv->ops->axpy            = VecAXPY_SeqCUSP;
  vv->ops->axpby           = VecAXPBY_SeqCUSP;
  vv->ops->maxpy           = VecMAXPY_SeqCUSP;
  vv->ops->aypx            = VecAYPX_SeqCUSP;
  vv->ops->axpbypcz        = VecAXPBYPCZ_SeqCUSP;
  vv->ops->pointwisemult   = VecPointwiseMult_SeqCUSP;
  vv->ops->setrandom       = VecSetRandom_SeqCUSP;
  vv->ops->replacearray    = VecReplaceArray_SeqCUSP;
  vv->ops->dot_local       = VecDot_SeqCUSP;
  vv->ops->tdot_local      = VecTDot_SeqCUSP;
  vv->ops->norm_local      = VecNorm_SeqCUSP;
  vv->ops->mdot_local      = VecMDot_SeqCUSP;
  vv->ops->destroy         = VecDestroy_MPICUSP;
  vv->ops->pointwisedivide = VecPointwiseDivide_SeqCUSP;
  /* place array?
     reset array?
     get values?
  */
  ierr = VecCUSPAllocateCheck(vv);CHKERRCUSP(ierr);
  vv->valid_GPU_array      = PETSC_CUSP_GPU;
  ierr = VecSet(vv,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_CUSP"
PetscErrorCode  VecCreate_CUSP(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)v)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQCUSP);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPICUSP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END





