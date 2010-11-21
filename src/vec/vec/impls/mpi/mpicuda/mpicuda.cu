#define PETSCVEC_DLL
/*
   This file contains routines for Parallel vector operations.
 */
#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/
PETSC_CUDA_EXTERN_C_END
#include "../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_MPICUDA"
PetscErrorCode VecDestroy_MPICUDA(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try{
    if (v->spptr) {
      delete ((Vec_CUDA*)v->spptr)->GPUarray;
      delete (Vec_CUDA *)v->spptr;
    }
  } catch(char* ex){
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNorm_MPICUDA"
PetscErrorCode VecNorm_MPICUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecNorm_SeqCUDA(xin,NORM_2,&work);
    work *= work;
    ierr = MPI_Allreduce(&work,&sum,1,MPIU_REAL,MPI_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z = sqrt(sum);
    ierr = PetscLogFlops(2.0*xin->map->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqCUDA(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPI_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqCUDA(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPI_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqCUDA(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDA(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPI_Allreduce(temp,z,2,MPIU_REAL,MPI_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    z[1] = sqrt(z[1]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDot_MPICUDA"
PetscErrorCode VecDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqCUDA(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecTDot_MPICUDA"
PetscErrorCode VecTDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqCUDA(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMDot_MPICUDA"
PetscErrorCode VecMDot_MPICUDA(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqCUDA(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
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

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMpiWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMpi()
M*/


#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPICUDA"
PetscErrorCode VecDuplicate_MPICUDA(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI *)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,v);CHKERRQ(ierr);

  /* use the map that exists aleady in win */
  ierr = PetscLayoutDestroy((*v)->map);CHKERRQ(ierr);
  (*v)->map = win->map;
  win->map->refcnt++;

  ierr = VecCreate_MPI_Private(*v,PETSC_FALSE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArrayPrivate(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArrayPrivate(*v,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(*v,vw->localrep);CHKERRQ(ierr);
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }    

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;
  
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  if (win->mapping) {
    ierr = PetscObjectReference((PetscObject)win->mapping);CHKERRQ(ierr);
    (*v)->mapping = win->mapping;
  }
  if (win->bmapping) {
    ierr = PetscObjectReference((PetscObject)win->bmapping);CHKERRQ(ierr);
    (*v)->bmapping = win->bmapping;
  }
  (*v)->map->bs    = win->map->bs;
  (*v)->bstash.bs = win->bstash.bs;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDotNorm2_MPICUDA"
PetscErrorCode VecDotNorm2_MPICUDA(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode  ierr;
  PetscScalar     work[2],sum[2];

  PetscFunctionBegin;
  ierr    = VecDotNorm2_SeqCUDA(s,t,work,work+1);CHKERRQ(ierr);
  ierr    = MPI_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,((PetscObject)s)->comm);CHKERRQ(ierr);
  *dp     = sum[0];
  *nm     = sum[1];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPICUDA"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_MPICUDA(Vec vv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_FALSE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPICUDA);CHKERRQ(ierr);
  vv->valid_GPU_array      = PETSC_CUDA_UNALLOCATED;
  vv->ops->dotnorm2        = VecDotNorm2_MPICUDA;
  vv->ops->waxpy           = VecWAXPY_SeqCUDA;
  vv->ops->duplicate       = VecDuplicate_MPICUDA;
  vv->ops->dot             = VecDot_MPICUDA;
  vv->ops->mdot            = VecMDot_MPICUDA;
  vv->ops->tdot            = VecTDot_MPICUDA;
  vv->ops->norm            = VecNorm_MPICUDA;
  vv->ops->scale           = VecScale_SeqCUDA;
  vv->ops->copy            = VecCopy_SeqCUDA;
  vv->ops->set             = VecSet_SeqCUDA;
  vv->ops->swap            = VecSwap_SeqCUDA;
  vv->ops->axpy            = VecAXPY_SeqCUDA;
  vv->ops->axpby           = VecAXPBY_SeqCUDA;
  vv->ops->maxpy           = VecMAXPY_SeqCUDA;
  vv->ops->aypx            = VecAYPX_SeqCUDA;
  vv->ops->axpbypcz        = VecAXPBYPCZ_SeqCUDA;
  vv->ops->pointwisemult   = VecPointwiseMult_SeqCUDA;
  vv->ops->setrandom       = VecSetRandom_SeqCUDA;
  vv->ops->replacearray    = VecReplaceArray_SeqCUDA;
  vv->ops->dot_local       = VecDot_SeqCUDA;
  vv->ops->tdot_local      = VecTDot_SeqCUDA;
  vv->ops->norm_local      = VecNorm_SeqCUDA;
  vv->ops->mdot_local      = VecMDot_SeqCUDA;
  vv->ops->destroy         = VecDestroy_MPICUDA;
  vv->ops->pointwisedivide = VecPointwiseDivide_SeqCUDA;
  /* place array?
     reset array?
     get values?
  */
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_CUDA"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_CUDA(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)v)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQCUDA);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPICUDA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END





