/*
 Implementation of the sequential cusp vectors.

 This file contains the code that can be compiled with a C
 compiler.  The companion file veccusp2.cu contains the code that
 must be compiled with nvcc or a C++ compiler.
 */

#define PETSC_SKIP_COMPLEX
#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>


/*
    Allocates space for the vector array on the Host if it does not exist.
    Does NOT change the PetscCUSPFlag for the vector
    Does NOT zero the CUSP array
 */
PetscErrorCode VecCUSPAllocateCheckHost(Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec_Seq        *s = (Vec_Seq*)v->data;
  PetscInt       n = v->map->n;

  PetscFunctionBegin;
  if (!s) {
    ierr = PetscNewLog((PetscObject)v,&s);CHKERRQ(ierr);
    v->data = s;
  }
  if (!s->array) {
    ierr               = PetscMalloc1(n,&array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
    if (v->valid_GPU_array == PETSC_CUSP_UNALLOCATED) {
      v->valid_GPU_array = PETSC_CUSP_CPU;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUSP_Private(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecCUSPAllocateCheckHost(xin);
  ierr = VecCUSPAllocateCheckHost(yin);
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscMemcpy(ya,xa,xin->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqCUSP_Private(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUSP_Private(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  if (vs) {
    if (vs->array_allocated) ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
    ierr = PetscFree(vs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqCUSP_Private(Vec vin)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUSPAllocateCheck_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUSPCopyToGPU_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyToGPU(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    VecCUSPCopyToGPUSome_Public - Copies certain entries down to the GPU from the CPU of a vector

   Input Parameters:
.    v - the vector
.    indices - the requested indices, this should be created with CUSPIndicesCreate()

*/
PetscErrorCode VecCUSPCopyToGPUSome_Public(Vec v,PetscCUSPIndices ci)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyToGPUSome(v,ci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  VecCUSPCopyFromGPUSome_Public - Copies certain entries up to the CPU from the GPU of a vector

  Input Parameters:
 +    v - the vector
 -    indices - the requested indices, this should be created with CUSPIndicesCreate()
*/
PetscErrorCode VecCUSPCopyFromGPUSome_Public(Vec v,PetscCUSPIndices ci)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPUSome(v,ci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqCUSP(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetRandom_SeqCUSP_Private(xin,r);CHKERRQ(ierr);
  xin->valid_GPU_array = PETSC_CUSP_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqCUSP(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecResetArray_SeqCUSP_Private(vin);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_CUSP_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_SeqCUSP(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_CUSP_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_SeqCUSP(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_CUSP_CPU;
  PetscFunctionReturn(0);
}

/*@
 VecCreateSeqCUSP - Creates a standard, sequential array-style vector.

 Collective on MPI_Comm

 Input Parameter:
 .  comm - the communicator, should be PETSC_COMM_SELF
 .  n - the vector length

 Output Parameter:
 .  V - the vector

 Notes:
 Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
 same type as an existing vector.

 Level: intermediate

 Concepts: vectors^creating sequential

 .seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
 @*/
PetscErrorCode VecCreateSeqCUSP(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQCUSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_SeqCUSP(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqCUSP(PetscObjectComm((PetscObject)win),win->map->n,V);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_SeqCUSP(Vec V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQCUSP on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQCUSP);CHKERRQ(ierr);

  V->ops->dot                    = VecDot_SeqCUSP;
  V->ops->norm                   = VecNorm_SeqCUSP;
  V->ops->tdot                   = VecTDot_SeqCUSP;
  V->ops->scale                  = VecScale_SeqCUSP;
  V->ops->copy                   = VecCopy_SeqCUSP;
  V->ops->set                    = VecSet_SeqCUSP;
  V->ops->swap                   = VecSwap_SeqCUSP;
  V->ops->axpy                   = VecAXPY_SeqCUSP;
  V->ops->axpby                  = VecAXPBY_SeqCUSP;
  V->ops->axpbypcz               = VecAXPBYPCZ_SeqCUSP;
  V->ops->pointwisemult          = VecPointwiseMult_SeqCUSP;
  V->ops->pointwisedivide        = VecPointwiseDivide_SeqCUSP;
  V->ops->setrandom              = VecSetRandom_SeqCUSP;
  V->ops->dot_local              = VecDot_SeqCUSP;
  V->ops->tdot_local             = VecTDot_SeqCUSP;
  V->ops->norm_local             = VecNorm_SeqCUSP;
  V->ops->mdot_local             = VecMDot_SeqCUSP;
  V->ops->maxpy                  = VecMAXPY_SeqCUSP;
  V->ops->mdot                   = VecMDot_SeqCUSP;
  V->ops->aypx                   = VecAYPX_SeqCUSP;
  V->ops->waxpy                  = VecWAXPY_SeqCUSP;
  V->ops->dotnorm2               = VecDotNorm2_SeqCUSP;
  V->ops->placearray             = VecPlaceArray_SeqCUSP;
  V->ops->replacearray           = VecReplaceArray_SeqCUSP;
  V->ops->resetarray             = VecResetArray_SeqCUSP;
  V->ops->destroy                = VecDestroy_SeqCUSP;
  V->ops->duplicate              = VecDuplicate_SeqCUSP;
  V->ops->conjugate              = VecConjugate_SeqCUSP;
  V->ops->getlocalvector         = VecGetLocalVector_SeqCUSP;
  V->ops->restorelocalvector     = VecRestoreLocalVector_SeqCUSP;
  V->ops->getlocalvectorread     = VecGetLocalVector_SeqCUSP;
  V->ops->restorelocalvectorread = VecRestoreLocalVector_SeqCUSP;

  ierr = VecCUSPAllocateCheck(V);CHKERRQ(ierr);
  V->valid_GPU_array = PETSC_CUSP_GPU;
  ierr = VecSet(V,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
