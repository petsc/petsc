#ifndef __CUDAVECIMPL 
#define __CUDAVECIMPL

#include "private/vecimpl.h"
#include <cublas.h>
#include <cusp/blas.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>

#define CUSPARRAY cusp::array1d<PetscScalar,cusp::device_memory>
#define CUSPINTARRAYGPU cusp::array1d<PetscInt,cusp::device_memory>
#define CUSPINTARRAYCPU cusp::array1d<PetscInt,cusp::host_memory>

EXTERN PetscErrorCode VecDotNorm2_SeqCUDA(Vec,Vec,PetscScalar *, PetscScalar *);
EXTERN PetscErrorCode VecPointwiseDivide_SeqCUDA(Vec,Vec,Vec);
EXTERN PetscErrorCode VecWAXPY_SeqCUDA(Vec,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode VecMDot_SeqCUDA(Vec,PetscInt,const Vec[],PetscScalar *);
EXTERN PetscErrorCode VecSet_SeqCUDA(Vec,PetscScalar);
EXTERN PetscErrorCode VecMAXPY_SeqCUDA(Vec,PetscInt,const PetscScalar *,Vec *);
EXTERN PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode VecPointwiseMult_SeqCUDA(Vec,Vec,Vec);
EXTERN PetscErrorCode VecPlaceArray_SeqCUDA(Vec,const PetscScalar *);
EXTERN PetscErrorCode VecResetArray_SeqCUDA(Vec);
EXTERN PetscErrorCode VecReplaceArray_SeqCUDA(Vec,const PetscScalar *);
EXTERN PetscErrorCode VecDot_SeqCUDA(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecTDot_SeqCUDA(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecScale_SeqCUDA(Vec,PetscScalar);
EXTERN PetscErrorCode VecCopy_SeqCUDA(Vec,Vec);
EXTERN PetscErrorCode VecSwap_SeqCUDA(Vec,Vec);
EXTERN PetscErrorCode VecAXPY_SeqCUDA(Vec,PetscScalar,Vec);
EXTERN PetscErrorCode VecAXPBY_SeqCUDA(Vec,PetscScalar,PetscScalar,Vec);
EXTERN PetscErrorCode VecDuplicate_SeqCUDA(Vec,Vec *);
EXTERN PetscErrorCode VecNorm_SeqCUDA(Vec,NormType,PetscReal*);
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_SeqCUDA(Vec);
EXTERN_C_END
EXTERN PetscErrorCode VecView_Seq(Vec,PetscViewer);
EXTERN PetscErrorCode VecDestroy_SeqCUDA(Vec);
EXTERN PetscErrorCode VecAYPX_SeqCUDA(Vec,PetscScalar,Vec);
EXTERN PetscErrorCode VecSetRandom_SeqCUDA(Vec,PetscRandom);

EXTERN PetscErrorCode VecCUDACopyToGPU_Public(Vec);
EXTERN PetscErrorCode VecCUDAAllocateCheck_Public(Vec);
EXTERN PetscErrorCode VecCUDACopyToGPUSome_Public(Vec,CUSPINTARRAYCPU*,CUSPINTARRAYGPU*);

EXTERN PetscBool  synchronizeCUDA;
#define CHKERRCUDA(err) if (err != CUBLAS_STATUS_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error %d",err)

#define VecCUDACastToRawPtr(x) thrust::raw_pointer_cast(&(x)[0])

#define WaitForGPU() synchronizeCUDA ? cudaThreadSynchronize() : 0

struct Vec_CUDA {
  /* eventually we should probably move the GPU flag into here */
  CUSPARRAY*       GPUarray;  /* this always holds the GPU data */
};

#undef __FUNCT__
#define __FUNCT__ "VecCUDAAllocateCheck"
PETSC_STATIC_INLINE PetscErrorCode VecCUDAAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  Vec_Seq   *s;

  PetscFunctionBegin;
  if (v->valid_GPU_array == PETSC_CUDA_UNALLOCATED){
    try {
        v->spptr = new Vec_CUDA;
	((Vec_CUDA*)v->spptr)->GPUarray = new CUSPARRAY;
	((Vec_CUDA*)v->spptr)->GPUarray->resize((PetscBLASInt)v->map->n);
	ierr = WaitForGPU();CHKERRQ(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    } 
	
    s = (Vec_Seq*)v->data;
    if (s->array == 0){
      v->valid_GPU_array = PETSC_CUDA_GPU;
    } else{
      v->valid_GPU_array = PETSC_CUDA_CPU;
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCUDACopyToGPU"
/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PETSC_STATIC_INLINE PetscErrorCode VecCUDACopyToGPU(Vec v)
{
  PetscBLASInt   cn = v->map->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUDA_CPU){
    ierr = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    try{
      ((Vec_CUDA*)v->spptr)->GPUarray->assign(*(PetscScalar**)v->data,*(PetscScalar**)v->data + cn);
      ierr = WaitForGPU();CHKERRQ(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    } 
    ierr = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUDA_BOTH;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDACopyToGPUSome"
PETSC_STATIC_INLINE PetscErrorCode VecCUDACopyToGPUSome(Vec v,CUSPINTARRAYCPU *indicesCPU,CUSPINTARRAYGPU *indicesGPU)
{
  Vec_Seq        *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  s = (Vec_Seq *)v->data;
  if (v->valid_GPU_array == PETSC_CUDA_CPU) {
    ierr = PetscLogEventBegin(VEC_CUDACopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
    thrust::copy(
		 thrust::make_permutation_iterator(s->array,indicesCPU->begin()),
		 thrust::make_permutation_iterator(s->array,indicesCPU->end()),
		 thrust::make_permutation_iterator(((Vec_CUDA *)v->spptr)->GPUarray->begin(),indicesGPU->begin()));
    ierr = PetscLogEventEnd(VEC_CUDACopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  v->valid_GPU_array = PETSC_CUDA_GPU;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDAGetArrayReadWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUDAGetArrayReadWrite(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
  *a = ((Vec_CUDA *)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDARestoreArrayReadWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUDARestoreArrayReadWrite(Vec v, CUSPARRAY** a)
{
  PetscFunctionBegin;
  if (v->valid_GPU_array != PETSC_CUDA_UNALLOCATED){
    v->valid_GPU_array = PETSC_CUDA_GPU;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDAGetArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecCUDAGetArrayRead(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
  *a = ((Vec_CUDA *)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "VecCUDARestoreArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecCUDARestoreArrayRead(Vec v, CUSPARRAY** a)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "VecCUDAGetArrayWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUDAGetArrayWrite(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  *a = ((Vec_CUDA *)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "VecCUDARestoreArrayWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUDARestoreArrayWrite(Vec v, CUSPARRAY* a)
{
  PetscFunctionBegin;
  if (v->valid_GPU_array != PETSC_CUDA_UNALLOCATED){
    v->valid_GPU_array = PETSC_CUDA_GPU;
  }
  PetscFunctionReturn(0);
}
#endif
