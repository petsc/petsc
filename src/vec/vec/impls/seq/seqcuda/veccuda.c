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
    if (n*sizeof(PetscScalar) > v->minimum_bytes_pinned_memory) {
      ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
      v->pinned_memory = PETSC_TRUE;
    }
    ierr = PetscMalloc1(n,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
    if (n*sizeof(PetscScalar) > v->minimum_bytes_pinned_memory) {
      ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
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

PetscErrorCode VecSetRandom_SeqCUDA(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n;
  PetscBool      iscurand;
  PetscScalar    *xx;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)r,PETSCCURAND,&iscurand);CHKERRQ(ierr);
  if (iscurand) {
    ierr = VecCUDAGetArrayWrite(xin,&xx);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayWrite(xin,&xx);CHKERRQ(ierr);
  }
  ierr = PetscRandomGetValues(r,n,xx);CHKERRQ(ierr);
  if (iscurand) {
    ierr = VecCUDARestoreArrayWrite(xin,&xx);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArrayWrite(xin,&xx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUDA_Private(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%" PetscInt_FMT,v->map->n);
#endif
  if (vs) {
    if (vs->array_allocated) {
      if (v->pinned_memory) {
        ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
      }
      ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
      if (v->pinned_memory) {
        ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
        v->pinned_memory = PETSC_FALSE;
      }
    }
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
  Vec_Seq        *vs = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  if (vs->array != vs->array_allocated) {
    /* make sure the users array has the latest values */
    ierr = VecCUDACopyFromGPU(vin);CHKERRQ(ierr);
  }
  if (vs->array_allocated) {
    if (vin->pinned_memory) {
      ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
    }
    ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
    if (vin->pinned_memory) {
      ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
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
  ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(V->map);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(V);CHKERRQ(ierr);
  ierr = VecCreate_SeqCUDA_Private(V,((Vec_CUDA*)V->spptr)->GPUarray_allocated);CHKERRQ(ierr);
  ierr = VecSet_SeqCUDA(V,0.0);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*V,bs);CHKERRQ(ierr);
  ierr = VecCreate_SeqCUDA_Private(*V,array);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // set V's gpuarray to be gpuarray, do not allocate memory on host yet.
  ierr = VecCreateSeqCUDAWithArray(comm,bs,n,gpuarray,V);CHKERRQ(ierr);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyFromGPU(v);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheckHost(v);CHKERRQ(ierr);
  *a   = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayAndMemType_SeqCUDA(Vec v,PetscScalar** a,PetscMemType *mtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Allocate memory (not zeroed) on device if not yet, but no need to sync data from host to device */
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  if (mtype) *mtype = ((Vec_CUDA*)v->spptr)->nvshmem ? PETSC_MEMTYPE_NVSHMEM : PETSC_MEMTYPE_CUDA;
  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_SeqCUDA(Vec V,PetscBool bind)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->boundtocpu = bind;
  if (bind) {
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
    ierr = PetscFree(V->defaultrandtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCRANDER48,&V->defaultrandtype);CHKERRQ(ierr);
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
    ierr = PetscFree(V->defaultrandtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCCURAND,&V->defaultrandtype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_SeqCUDA_Private(Vec V,const PetscScalar *array)
{
  PetscErrorCode ierr;
  Vec_CUDA       *veccuda;
  PetscMPIInt    size;
  PetscBool      option_set;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQCUDA on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQCUDA);CHKERRQ(ierr);
  ierr = VecBindToCPU_SeqCUDA(V,PETSC_FALSE);CHKERRQ(ierr);
  V->ops->bindtocpu = VecBindToCPU_SeqCUDA;

  /* Later, functions check for the Vec_CUDA structure existence, so do not create it without array */
  if (array) {
    if (!V->spptr) {
      PetscReal pinned_memory_min;
      ierr = PetscCalloc(sizeof(Vec_CUDA),&V->spptr);CHKERRQ(ierr);
      veccuda = (Vec_CUDA*)V->spptr;
      V->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

      pinned_memory_min = 0;
      /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
         Note: This same code duplicated in VecCUDAAllocateCheck() and VecCreate_MPICUDA_Private(). Is there a good way to avoid this? */
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)V),((PetscObject)V)->prefix,"VECCUDA Options","Vec");CHKERRQ(ierr);
      ierr = PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&option_set);CHKERRQ(ierr);
      if (option_set) V->minimum_bytes_pinned_memory = pinned_memory_min;
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    veccuda = (Vec_CUDA*)V->spptr;
    veccuda->GPUarray = (PetscScalar*)array;
    V->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}
