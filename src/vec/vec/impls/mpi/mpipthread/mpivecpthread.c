
#include <../src/vec/vec/impls/mpi/mpipthread/mpivecpthreadimpl.h>   /*I  "petscvec.h"   I*/
#include <petscblaslapack.h>

extern PetscMPIInt  PetscMaxThreads;

extern PetscInt     vecs_created;
extern Kernel_Data  *kerneldatap;
extern Kernel_Data  **pdata;

#undef __FUNCT__  
#define __FUNCT__ "VecDot_MPIPThread"
PetscErrorCode VecDot_MPIPThread(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqPThread(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMDot_MPIPThread"
PetscErrorCode VecMDot_MPIPThread(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqPThread(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
#undef __FUNCT__  
#define __FUNCT__ "VecNorm_MPIPThread"
PetscErrorCode VecNorm_MPIPThread(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscScalar    *xx = *(PetscScalar**)xin->data;
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(n);

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    work  = BLASnrm2_(&bn,xx,&one);
    work *= work;
    ierr = MPI_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z = PetscSqrtReal(sum);
    ierr = PetscLogFlops(2.0*xin->map->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqPThread(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqPThread(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqPThread(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqPThread(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPI_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

extern MPI_Op VecMax_Local_Op;
extern MPI_Op VecMin_Local_Op;

#undef __FUNCT__  
#define __FUNCT__ "VecMax_MPIPThread"
PetscErrorCode VecMax_MPIPThread(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  ierr = VecMax_SeqPThread(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global max */
  if (!idx) {
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;
    rstart = xin->map->rstart;
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPI_Allreduce(work2,z2,2,MPIU_REAL,VecMax_Local_Op,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMin_MPIPThread"
PetscErrorCode VecMin_MPIPThread(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  ierr = VecMin_SeqPThread(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global Min */
  if (!idx) {
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,((PetscObject)xin)->comm);CHKERRQ(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;

    ierr = VecGetOwnershipRange(xin,&rstart,PETSC_NULL);CHKERRQ(ierr);
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPI_Allreduce(work2,z2,2,MPIU_REAL,VecMin_Local_Op,((PetscObject)xin)->comm);CHKERRQ(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPIPThread_Private(Vec,PetscBool,PetscInt,const PetscScalar []);

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPIPThread"
PetscErrorCode VecDuplicate_MPIPThread(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPIPthread *vw,*w = (Vec_MPIPthread *)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPIPThread_Private(*v,PETSC_TRUE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPIPthread *)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
  if(!w->arrindex) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must set the number of threads for the first vector before duplicating it");
  ierr = VecSeqPThreadSetNThreads(*v,w->nthreads);CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
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
  
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  (*v)->map->bs    = win->map->bs;
  (*v)->bstash.bs = win->bstash.bs;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_MPIPThread"
PetscErrorCode VecDestroy_MPIPThread(Vec v)
{
  Vec_MPIPthread *x = (Vec_MPIPthread*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->N);
#endif
  if (!x) PetscFunctionReturn(0);
  ierr = PetscFree(x->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree2(x->arrindex,x->nelem);CHKERRQ(ierr);

  /* Destroy local representation of vector if it exists */
  if (x->localrep) {
    ierr = VecDestroy(&x->localrep);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&x->localupdate);CHKERRQ(ierr);
  }

  vecs_created--;
  /* Free the kernel data structure on the destruction of the last vector */
  if(vecs_created == 0) {
    ierr = PetscFree(kerneldatap);CHKERRQ(ierr);
    ierr = PetscFree(pdata);CHKERRQ(ierr);
  }

  /* Destroy the stashes: note the order - so that the tags are freed properly */
  ierr = VecStashDestroy_Private(&v->bstash);CHKERRQ(ierr);
  ierr = VecStashDestroy_Private(&v->stash);CHKERRQ(ierr);
  ierr = PetscFree(v->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


extern PetscErrorCode VecSetOption_MPI(Vec,VecOption,PetscBool);
extern PetscErrorCode VecResetArray_MPI(Vec);

static struct _VecOps DvOps = { VecDuplicate_MPIPThread, /* 1 */
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default,
            VecDot_MPIPThread,
            VecMDot_MPIPThread,
            VecNorm_MPIPThread,
            VecTDot_MPI,
            VecMTDot_MPI,
            VecScale_SeqPThread,
            VecCopy_SeqPThread, /* 10 */
            VecSet_SeqPThread,
            VecSwap_SeqPThread,
            VecAXPY_SeqPThread,
            VecAXPBY_Seq,
            VecMAXPY_SeqPThread,
            VecAYPX_SeqPThread,
            VecWAXPY_SeqPThread,
            VecAXPBYPCZ_Seq,
            VecPointwiseMult_SeqPThread,
            VecPointwiseDivide_SeqPThread,
            VecSetValues_MPI, /* 20 */
            VecAssemblyBegin_MPI,
            VecAssemblyEnd_MPI,
            0,
            VecGetSize_MPI,
            VecGetSize_Seq,
            0,
            VecMax_MPIPThread,
            VecMin_MPIPThread,
            VecSetRandom_SeqPThread,
            VecSetOption_MPI,
            VecSetValuesBlocked_MPI,
            VecDestroy_MPIPThread,
            VecView_MPI,
            VecPlaceArray_MPI,
            VecReplaceArray_Seq,
            VecDot_SeqPThread,
            VecTDot_Seq,
            VecNorm_SeqPThread,
            VecMDot_SeqPThread,
            VecMTDot_Seq,
	    VecLoad_Default,			
            VecReciprocal_Default,
            VecConjugate_Seq,
            0,
            0,
            VecResetArray_MPI,
            VecSetFromOptions_SeqPThread,
            VecMaxPointwiseDivide_Seq,
            VecPointwiseMax_Seq,
            VecPointwiseMaxAbs_Seq,
            VecPointwiseMin_Seq,
  	    VecGetValues_MPI,
    	    0,
    	    0,
    	    0,
    	    0,
    	    0,
    	    0,
   	    VecStrideGather_Default,
   	    VecStrideScatter_Default
};

#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPIPThread_Private"
PetscErrorCode VecCreate_MPIPThread_Private(Vec v,PetscBool  alloc,PetscInt nghost,const PetscScalar array[])
{
  Vec_MPIPthread *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr           = PetscNewLog(v,Vec_MPIPthread,&s);CHKERRQ(ierr);
  v->data        = (void*)s;
  ierr           = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  s->nghost      = nghost;
  v->petscnative = PETSC_TRUE;

  if (v->map->bs == -1) v->map->bs = 1;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  s->array           = (PetscScalar *)array;
  s->array_allocated = 0;
  s->nthreads        = 0;
  s->arrindex        = 0;
  if (alloc && !array) {
    PetscInt n         = v->map->n+nghost;
    ierr               = PetscMalloc(n*sizeof(PetscScalar),&s->array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory(v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr               = PetscMemzero(s->array,v->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array_allocated = s->array;
  }

  if(!vecs_created) {
    ierr = PetscMalloc(PetscMaxThreads*sizeof(Kernel_Data),&kerneldatap);CHKERRQ(ierr);
    ierr = PetscMalloc(PetscMaxThreads*sizeof(Kernel_Data*),&pdata);CHKERRQ(ierr);
  }
  vecs_created++;

  /* By default parallel vectors do not have local representation */
  s->localrep    = 0;
  s->localupdate = 0;

  v->stash.insertmode  = NOT_SET_VALUES;
  /* create the stashes. The block-size for bstash is set later when 
     VecSetValuesBlocked is called.
  */
  ierr = VecStashCreate_Private(((PetscObject)v)->comm,1,&v->stash);CHKERRQ(ierr);
  ierr = VecStashCreate_Private(((PetscObject)v)->comm,v->map->bs,&v->bstash);CHKERRQ(ierr); 
                                                        
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEnginePut_C","VecMatlabEnginePut_Default",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEngineGet_C","VecMatlabEngineGet_Default",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECMPIPTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   VECMPIPTHREAD - VECMPIPTHREAD = "mpipthread" - The basic parallel vector

   Options Database Keys:
. -vec_type mpipthread - sets the vector type to VECMPIPTHREAD during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMpiWithArray(), VECMPI, VecType, VecCreateMPI()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPIPThread"
PetscErrorCode  VecCreate_MPIPThread(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate_MPIPThread_Private(vv,PETSC_TRUE,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_PThread"
PetscErrorCode VecCreate_PThread(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)v)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecCreate_SeqPThread(v);CHKERRQ(ierr);
  } else {
    ierr = VecCreate_MPIPThread(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
