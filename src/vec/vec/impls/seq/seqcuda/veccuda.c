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
#include <petsc/private/cudavecimpl.h>

PetscErrorCode VecCUDAGetArrays_Private(Vec v,const PetscScalar** x,const PetscScalar** x_d,PetscOffloadMask* flg)
{
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscFunctionBegin;
  if (x) {
    Vec_Seq *h = (Vec_Seq*)v->data;

    *x = h->array;
  }
  if (x_d) {
    Vec_CUDA *d = (Vec_CUDA*)v->spptr;

    *x_d = d ? d->GPUarray : NULL;
  }
  if (flg) *flg = v->offloadmask;
  PetscFunctionReturn(0);
}

/*
    Allocates space for the vector array on the Host if it does not exist.
    Does NOT change the PetscCUDAFlag for the vector
    Does NOT zero the CUDA array
 */
PetscErrorCode VecCUDAAllocateCheckHost(Vec v)
{
  PetscScalar    *array;
  Vec_Seq        *s = (Vec_Seq*)v->data;
  PetscInt       n = v->map->n;

  PetscFunctionBegin;
  if (!s) {
    PetscCall(PetscNewLog((PetscObject)v,&s));
    v->data = s;
  }
  if (!s->array) {
    if (n*sizeof(PetscScalar) > v->minimum_bytes_pinned_memory) {
      PetscCall(PetscMallocSetCUDAHost());
      v->pinned_memory = PETSC_TRUE;
    }
    PetscCall(PetscMalloc1(n,&array));
    PetscCall(PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar)));
    s->array           = array;
    s->array_allocated = array;
    if (n*sizeof(PetscScalar) > v->minimum_bytes_pinned_memory) {
      PetscCall(PetscMallocResetCUDAHost());
    }
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

  PetscFunctionBegin;
  PetscCall(VecCUDAAllocateCheckHost(xin));
  PetscCall(VecCUDAAllocateCheckHost(yin));
  if (xin != yin) {
    PetscCall(VecGetArrayRead(xin,&xa));
    PetscCall(VecGetArray(yin,&ya));
    PetscCall(PetscArraycpy(ya,xa,xin->map->n));
    PetscCall(VecRestoreArrayRead(xin,&xa));
    PetscCall(VecRestoreArray(yin,&ya));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqCUDA(Vec xin,PetscRandom r)
{
  PetscInt       n = xin->map->n;
  PetscBool      iscurand;
  PetscScalar    *xx;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)r,PETSCCURAND,&iscurand));
  if (iscurand) {
    PetscCall(VecCUDAGetArrayWrite(xin,&xx));
  } else {
    PetscCall(VecGetArrayWrite(xin,&xx));
  }
  PetscCall(PetscRandomGetValues(r,n,xx));
  if (iscurand) {
    PetscCall(VecCUDARestoreArrayWrite(xin,&xx));
  } else {
    PetscCall(VecRestoreArrayWrite(xin,&xx));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUDA_Private(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsViewOff(v));
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%" PetscInt_FMT,v->map->n);
#endif
  if (vs) {
    if (vs->array_allocated) {
      if (v->pinned_memory) {
        PetscCall(PetscMallocSetCUDAHost());
      }
      PetscCall(PetscFree(vs->array_allocated));
      if (v->pinned_memory) {
        PetscCall(PetscMallocResetCUDAHost());
        v->pinned_memory = PETSC_FALSE;
      }
    }
    PetscCall(PetscFree(vs));
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

PetscErrorCode VecResetArray_SeqCUDA(Vec vin)
{
  PetscFunctionBegin;
  PetscCall(VecCUDACopyFromGPU(vin));
  PetscCall(VecResetArray_SeqCUDA_Private(vin));
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_SeqCUDA(Vec vin,const PetscScalar *a)
{
  PetscFunctionBegin;
  PetscCall(VecCUDACopyFromGPU(vin));
  PetscCall(VecPlaceArray_Seq(vin,a));
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_SeqCUDA(Vec vin,const PetscScalar *a)
{
  Vec_Seq        *vs = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  if (vs->array != vs->array_allocated) {
    /* make sure the users array has the latest values */
    PetscCall(VecCUDACopyFromGPU(vin));
  }
  if (vs->array_allocated) {
    if (vin->pinned_memory) {
      PetscCall(PetscMallocSetCUDAHost());
    }
    PetscCall(PetscFree(vs->array_allocated));
    if (vin->pinned_memory) {
      PetscCall(PetscMallocResetCUDAHost());
    }
  }
  vin->pinned_memory = PETSC_FALSE;
  vs->array_allocated = vs->array = (PetscScalar*)a;
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

 .seealso: VecCreateMPICUDA(), VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
 @*/
PetscErrorCode VecCreateSeqCUDA(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecCreate(comm,v));
  PetscCall(VecSetSizes(*v,n,n));
  PetscCall(VecSetType(*v,VECSEQCUDA));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_SeqCUDA(Vec win,Vec *V)
{
  PetscFunctionBegin;
  PetscCall(VecCreateSeqCUDA(PetscObjectComm((PetscObject)win),win->map->n,V));
  PetscCall(PetscLayoutReference(win->map,&(*V)->map));
  PetscCall(PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist));
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_SeqCUDA(Vec V)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(PetscLayoutSetUp(V->map));
  PetscCall(VecCUDAAllocateCheck(V));
  PetscCall(VecCreate_SeqCUDA_Private(V,((Vec_CUDA*)V->spptr)->GPUarray_allocated));
  PetscCall(VecSet_SeqCUDA(V,0.0));
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqCUDAWithArray - Creates a CUDA sequential array-style vector,
   where the user provides the array space to store the vector values. The array
   provided must be a GPU array.

   Collective

   Input Parameters:
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
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(VecCreate(comm,V));
  PetscCall(VecSetSizes(*V,n,n));
  PetscCall(VecSetBlockSize(*V,bs));
  PetscCall(VecCreate_SeqCUDA_Private(*V,array));
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqCUDAWithArrays - Creates a CUDA sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  cpuarray - CPU memory where the vector elements are to be stored.
-  gpuarray - GPU memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   If both cpuarray and gpuarray are provided, the caller must ensure that
   the provided arrays have identical values.

   PETSc does NOT free the provided arrays when the vector is destroyed via
   VecDestroy(). The user should not free the array until the vector is
   destroyed.

   Level: intermediate

.seealso: VecCreateMPICUDAWithArrays(), VecCreate(), VecCreateSeqWithArray(),
          VecCUDAPlaceArray(), VecCreateSeqCUDAWithArray(),
          VecCUDAAllocateCheckHost()
@*/
PetscErrorCode  VecCreateSeqCUDAWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *V)
{
  PetscFunctionBegin;
  // set V's gpuarray to be gpuarray, do not allocate memory on host yet.
  PetscCall(VecCreateSeqCUDAWithArray(comm,bs,n,gpuarray,V));

  if (cpuarray && gpuarray) {
    Vec_Seq *s = (Vec_Seq*)((*V)->data);
    s->array = (PetscScalar*)cpuarray;
    (*V)->offloadmask = PETSC_OFFLOAD_BOTH;
  } else if (cpuarray) {
    Vec_Seq *s = (Vec_Seq*)((*V)->data);
    s->array = (PetscScalar*)cpuarray;
    (*V)->offloadmask = PETSC_OFFLOAD_CPU;
  } else if (gpuarray) {
    (*V)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*V)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArray_SeqCUDA(Vec v,PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(VecCUDACopyFromGPU(v));
  *a = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArray_SeqCUDA(Vec v,PetscScalar **a)
{
  PetscFunctionBegin;
  v->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWrite_SeqCUDA(Vec v,PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(VecCUDAAllocateCheckHost(v));
  *a   = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayAndMemType_SeqCUDA(Vec v,PetscScalar** a,PetscMemType *mtype)
{
  PetscFunctionBegin;
  PetscCall(VecCUDACopyToGPU(v));
  *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  if (mtype) *mtype = ((Vec_CUDA*)v->spptr)->nvshmem ? PETSC_MEMTYPE_NVSHMEM : PETSC_MEMTYPE_CUDA;
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayAndMemType_SeqCUDA(Vec v,PetscScalar** a)
{
  PetscFunctionBegin;
  v->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWriteAndMemType_SeqCUDA(Vec v,PetscScalar** a,PetscMemType *mtype)
{
  PetscFunctionBegin;
  /* Allocate memory (not zeroed) on device if not yet, but no need to sync data from host to device */
  PetscCall(VecCUDAAllocateCheck(v));
  *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  if (mtype) *mtype = ((Vec_CUDA*)v->spptr)->nvshmem ? PETSC_MEMTYPE_NVSHMEM : PETSC_MEMTYPE_CUDA;
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_SeqCUDA(Vec V,PetscBool bind)
{
  PetscFunctionBegin;
  V->boundtocpu = bind;
  if (bind) {
    PetscCall(VecCUDACopyFromGPU(V));
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
    V->ops->replacearray           = VecReplaceArray_SeqCUDA;
    V->ops->resetarray             = VecResetArray_Seq;
    V->ops->duplicate              = VecDuplicate_Seq;
    V->ops->conjugate              = VecConjugate_Seq;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;
    V->ops->getarrayandmemtype     = NULL;
    V->ops->getarraywriteandmemtype= NULL;
    V->ops->restorearrayandmemtype = NULL;
    V->ops->max                    = VecMax_Seq;
    V->ops->min                    = VecMin_Seq;
    V->ops->reciprocal             = VecReciprocal_Default;
    V->ops->sum                    = NULL;
    V->ops->shift                  = NULL;
    /* default random number generator */
    PetscCall(PetscFree(V->defaultrandtype));
    PetscCall(PetscStrallocpy(PETSCRANDER48,&V->defaultrandtype));
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
    V->ops->getlocalvectorread     = VecGetLocalVectorRead_SeqCUDA;
    V->ops->restorelocalvectorread = VecRestoreLocalVectorRead_SeqCUDA;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqCUDA;
    V->ops->getarray               = VecGetArray_SeqCUDA;
    V->ops->restorearray           = VecRestoreArray_SeqCUDA;
    V->ops->getarrayandmemtype     = VecGetArrayAndMemType_SeqCUDA;
    V->ops->getarraywriteandmemtype= VecGetArrayWriteAndMemType_SeqCUDA;
    V->ops->restorearrayandmemtype = VecRestoreArrayAndMemType_SeqCUDA;
    V->ops->max                    = VecMax_SeqCUDA;
    V->ops->min                    = VecMin_SeqCUDA;
    V->ops->reciprocal             = VecReciprocal_SeqCUDA;
    V->ops->sum                    = VecSum_SeqCUDA;
    V->ops->shift                  = VecShift_SeqCUDA;

    /* default random number generator */
    PetscCall(PetscFree(V->defaultrandtype));
    PetscCall(PetscStrallocpy(PETSCCURAND,&V->defaultrandtype));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_SeqCUDA_Private(Vec V,const PetscScalar *array)
{
  Vec_CUDA       *veccuda;
  PetscMPIInt    size;
  PetscBool      option_set;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)V),&size));
  PetscCheck(size <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQCUDA on more than one process");
  PetscCall(VecCreate_Seq_Private(V,0));
  PetscCall(PetscObjectChangeTypeName((PetscObject)V,VECSEQCUDA));
  PetscCall(VecBindToCPU_SeqCUDA(V,PETSC_FALSE));
  V->ops->bindtocpu = VecBindToCPU_SeqCUDA;

  /* Later, functions check for the Vec_CUDA structure existence, so do not create it without array */
  if (array) {
    if (!V->spptr) {
      PetscReal pinned_memory_min;
      PetscCall(PetscCalloc(sizeof(Vec_CUDA),&V->spptr));
      veccuda = (Vec_CUDA*)V->spptr;
      V->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

      pinned_memory_min = 0;
      /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
         Note: This same code duplicated in VecCUDAAllocateCheck() and VecCreate_MPICUDA_Private(). Is there a good way to avoid this? */
      PetscOptionsBegin(PetscObjectComm((PetscObject)V),((PetscObject)V)->prefix,"VECCUDA Options","Vec");
      PetscCall(PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&option_set));
      if (option_set) V->minimum_bytes_pinned_memory = pinned_memory_min;
      PetscOptionsEnd();
    }
    veccuda = (Vec_CUDA*)V->spptr;
    veccuda->GPUarray = (PetscScalar*)array;
    V->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}
