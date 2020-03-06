/*
 Implementation of the sequential cuda vectors.

 This file contains the code that can be compiled with a C
 compiler.  The companion file veccuda2.cu contains the code that
 must be compiled with nvcc or a C++ compiler.
 */

#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>

/*
    Allocates space for the vector array on the Host if it does not exist.
    Does NOT change the PetscCUDAFlag for the vector
    Does NOT zero the CUDA array
 */
PetscErrorCode VecCUDAAllocateCheckHost(Vec v)
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
    ierr = PetscMalloc1(n,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      v->offloadmask = PETSC_OFFLOAD_CPU;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUDA_Private(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheckHost(xin);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheckHost(yin);CHKERRQ(ierr);
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscArraycpy(ya,xa,xin->map->n);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqCUDA_Private(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) { ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr); }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUDA_Private(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  if (vs) {
    if (vs->array_allocated) { ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr); }
    ierr = PetscFree(vs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqCUDA_Private(Vec vin)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUDAAllocateCheck_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUDACopyToGPU_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    VecCUDACopyToGPUSome_Public - Copies certain entries down to the GPU from the CPU of a vector

   Input Parameters:
 +  v    - the vector
 .  ci   - the requested indices, this should be created with CUDAIndicesCreate()
 -  mode - vec scatter mode used in VecScatterBegin/End
*/
PetscErrorCode VecCUDACopyToGPUSome_Public(Vec v,PetscCUDAIndices ci,ScatterMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyToGPUSome(v,ci,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  VecCUDACopyFromGPUSome_Public - Copies certain entries up to the CPU from the GPU of a vector

  Input Parameters:
 +  v    - the vector
 .  ci   - the requested indices, this should be created with CUDAIndicesCreate()
 -  mode - vec scatter mode used in VecScatterBegin/End
*/
PetscErrorCode VecCUDACopyFromGPUSome_Public(Vec v,PetscCUDAIndices ci,ScatterMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyFromGPUSome(v,ci,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqCUDA(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetRandom_SeqCUDA_Private(xin,r);CHKERRQ(ierr);
  xin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqCUDA(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecResetArray_SeqCUDA_Private(vin);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_SeqCUDA(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_SeqCUDA(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

/*@
 VecCreateSeqCUDA - Creates a standard, sequential array-style vector.

 Collective

 Input Parameter:
 +  comm - the communicator, should be PETSC_COMM_SELF
 -  n - the vector length

 Output Parameter:
 .  v - the vector

 Notes:
 Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
 same type as an existing vector.

 Level: intermediate

 .seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
 @*/
PetscErrorCode VecCreateSeqCUDA(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQCUDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_SeqCUDA(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqCUDA(PetscObjectComm((PetscObject)win),win->map->n,V);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_SeqCUDA(Vec V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(V->map);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(V);CHKERRQ(ierr);
  ierr = VecCreate_SeqCUDA_Private(V,((Vec_CUDA*)V->spptr)->GPUarray_allocated);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheckHost(V);CHKERRQ(ierr);
  ierr = VecSet(V,0.0);CHKERRQ(ierr);
  ierr = VecSet_Seq(V,0.0);CHKERRQ(ierr);
  V->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqCUDAWithArray - Creates a CUDA sequential array-style vector,
   where the user provides the array space to store the vector values. The array
   provided must be a GPU array.

   Collective

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  array - GPU memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecCUDAPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateMPICUDAWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
          VecCreateGhost(), VecCreateSeq(), VecCUDAPlaceArray(), VecCreateSeqWithArray(),
          VecCreateMPIWithArray()
@*/
PetscErrorCode  VecCreateSeqCUDAWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar array[],Vec *V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*V,bs);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  ierr = VecCreate_SeqCUDA_Private(*V,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWrite_SeqCUDA(Vec v,PetscScalar **vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheckHost(v);CHKERRQ(ierr);
  v->offloadmask = PETSC_OFFLOAD_CPU;
  *vv = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_SeqCUDA(Vec V,PetscBool pin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->boundtocpu = pin;
  if (pin) {
    ierr = VecCUDACopyFromGPU(V);CHKERRQ(ierr);
    V->offloadmask                 = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    V->ops->dot                    = VecDot_Seq;
    V->ops->norm                   = VecNorm_Seq;
    V->ops->tdot                   = VecTDot_Seq;
    V->ops->scale                  = VecScale_Seq;
    V->ops->copy                   = VecCopy_Seq;
    V->ops->set                    = VecSet_Seq;
    V->ops->swap                   = VecSwap_Seq;
    V->ops->axpy                   = VecAXPY_Seq;
    V->ops->axpby                  = VecAXPBY_Seq;
    V->ops->axpbypcz               = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult          = VecPointwiseMult_Seq;
    V->ops->pointwisedivide        = VecPointwiseDivide_Seq;
    V->ops->setrandom              = VecSetRandom_Seq;
    V->ops->dot_local              = VecDot_Seq;
    V->ops->tdot_local             = VecTDot_Seq;
    V->ops->norm_local             = VecNorm_Seq;
    V->ops->mdot_local             = VecMDot_Seq;
    V->ops->mtdot_local            = VecMTDot_Seq;
    V->ops->maxpy                  = VecMAXPY_Seq;
    V->ops->mdot                   = VecMDot_Seq;
    V->ops->mtdot                  = VecMTDot_Seq;
    V->ops->aypx                   = VecAYPX_Seq;
    V->ops->waxpy                  = VecWAXPY_Seq;
    V->ops->dotnorm2               = NULL;
    V->ops->placearray             = VecPlaceArray_Seq;
    V->ops->replacearray           = VecReplaceArray_Seq;
    V->ops->resetarray             = VecResetArray_Seq;
    V->ops->duplicate              = VecDuplicate_Seq;
    V->ops->conjugate              = VecConjugate_Seq;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;
  } else {
    V->ops->dot                    = VecDot_SeqCUDA;
    V->ops->norm                   = VecNorm_SeqCUDA;
    V->ops->tdot                   = VecTDot_SeqCUDA;
    V->ops->scale                  = VecScale_SeqCUDA;
    V->ops->copy                   = VecCopy_SeqCUDA;
    V->ops->set                    = VecSet_SeqCUDA;
    V->ops->swap                   = VecSwap_SeqCUDA;
    V->ops->axpy                   = VecAXPY_SeqCUDA;
    V->ops->axpby                  = VecAXPBY_SeqCUDA;
    V->ops->axpbypcz               = VecAXPBYPCZ_SeqCUDA;
    V->ops->pointwisemult          = VecPointwiseMult_SeqCUDA;
    V->ops->pointwisedivide        = VecPointwiseDivide_SeqCUDA;
    V->ops->setrandom              = VecSetRandom_SeqCUDA;
    V->ops->dot_local              = VecDot_SeqCUDA;
    V->ops->tdot_local             = VecTDot_SeqCUDA;
    V->ops->norm_local             = VecNorm_SeqCUDA;
    V->ops->mdot_local             = VecMDot_SeqCUDA;
    V->ops->maxpy                  = VecMAXPY_SeqCUDA;
    V->ops->mdot                   = VecMDot_SeqCUDA;
    V->ops->aypx                   = VecAYPX_SeqCUDA;
    V->ops->waxpy                  = VecWAXPY_SeqCUDA;
    V->ops->dotnorm2               = VecDotNorm2_SeqCUDA;
    V->ops->placearray             = VecPlaceArray_SeqCUDA;
    V->ops->replacearray           = VecReplaceArray_SeqCUDA;
    V->ops->resetarray             = VecResetArray_SeqCUDA;
    V->ops->destroy                = VecDestroy_SeqCUDA;
    V->ops->duplicate              = VecDuplicate_SeqCUDA;
    V->ops->conjugate              = VecConjugate_SeqCUDA;
    V->ops->getlocalvector         = VecGetLocalVector_SeqCUDA;
    V->ops->restorelocalvector     = VecRestoreLocalVector_SeqCUDA;
    V->ops->getlocalvectorread     = VecGetLocalVector_SeqCUDA;
    V->ops->restorelocalvectorread = VecRestoreLocalVector_SeqCUDA;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqCUDA;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_SeqCUDA_Private(Vec V,const PetscScalar *array)
{
  PetscErrorCode ierr;
  Vec_CUDA       *veccuda;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQCUDA on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQCUDA);CHKERRQ(ierr);
  ierr = VecBindToCPU_SeqCUDA(V,PETSC_FALSE);CHKERRQ(ierr);
  V->ops->bindtocpu = VecBindToCPU_SeqCUDA;

  /* Later, functions check for the Vec_CUDA structure existence, so do not create it without array */
  if (array) {
    if (!V->spptr) {
      ierr = PetscMalloc(sizeof(Vec_CUDA),&V->spptr);CHKERRQ(ierr);
      veccuda = (Vec_CUDA*)V->spptr;
      veccuda->stream = 0; /* using default stream */
      veccuda->GPUarray_allocated = 0;
      veccuda->hostDataRegisteredAsPageLocked = PETSC_FALSE;
      V->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
    }
    veccuda = (Vec_CUDA*)V->spptr;
    veccuda->GPUarray = (PetscScalar*)array;
  }

  PetscFunctionReturn(0);
}
