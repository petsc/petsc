#ifndef __CUSPVECIMPL
#define __CUSPVECIMPL

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

extern PetscErrorCode VecDotNorm2_SeqCUSP(Vec,Vec,PetscScalar *, PetscScalar *);
extern PetscErrorCode VecPointwiseDivide_SeqCUSP(Vec,Vec,Vec);
extern PetscErrorCode VecWAXPY_SeqCUSP(Vec,PetscScalar,Vec,Vec);
extern PetscErrorCode VecMDot_SeqCUSP(Vec,PetscInt,const Vec[],PetscScalar *);
extern PetscErrorCode VecSet_SeqCUSP(Vec,PetscScalar);
extern PetscErrorCode VecMAXPY_SeqCUSP(Vec,PetscInt,const PetscScalar *,Vec *);
extern PetscErrorCode VecAXPBYPCZ_SeqCUSP(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
extern PetscErrorCode VecPointwiseMult_SeqCUSP(Vec,Vec,Vec);
extern PetscErrorCode VecPlaceArray_SeqCUSP(Vec,const PetscScalar *);
extern PetscErrorCode VecResetArray_SeqCUSP(Vec);
extern PetscErrorCode VecReplaceArray_SeqCUSP(Vec,const PetscScalar *);
extern PetscErrorCode VecDot_SeqCUSP(Vec,Vec,PetscScalar *);
extern PetscErrorCode VecTDot_SeqCUSP(Vec,Vec,PetscScalar *);
extern PetscErrorCode VecScale_SeqCUSP(Vec,PetscScalar);
extern PetscErrorCode VecCopy_SeqCUSP(Vec,Vec);
extern PetscErrorCode VecSwap_SeqCUSP(Vec,Vec);
extern PetscErrorCode VecAXPY_SeqCUSP(Vec,PetscScalar,Vec);
extern PetscErrorCode VecAXPBY_SeqCUSP(Vec,PetscScalar,PetscScalar,Vec);
extern PetscErrorCode VecDuplicate_SeqCUSP(Vec,Vec *);
extern PetscErrorCode VecNorm_SeqCUSP(Vec,NormType,PetscReal*);
EXTERN_C_BEGIN
extern PetscErrorCode  VecCreate_SeqCUSP(Vec);
EXTERN_C_END
extern PetscErrorCode VecView_Seq(Vec,PetscViewer);
extern PetscErrorCode VecDestroy_SeqCUSP(Vec);
extern PetscErrorCode VecAYPX_SeqCUSP(Vec,PetscScalar,Vec);
extern PetscErrorCode VecSetRandom_SeqCUSP(Vec,PetscRandom);

extern PetscErrorCode VecCUSPCopyToGPU_Public(Vec);
extern PetscErrorCode VecCUSPAllocateCheck_Public(Vec);
extern PetscErrorCode VecCUSPCopyToGPUSome_Public(Vec,CUSPINTARRAYCPU*,CUSPINTARRAYGPU*);

extern PetscBool  synchronizeCUSP;
#define CHKERRCUSP(err) if (err != CUBLAS_STATUS_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error %d",err)

#define VecCUSPCastToRawPtr(x) thrust::raw_pointer_cast(&(x)[0])

#define WaitForGPU() synchronizeCUSP ? cudaThreadSynchronize() : 0

struct Vec_CUSP {
  /* eventually we should probably move the GPU flag into here */
  CUSPARRAY*       GPUarray;  /* this always holds the GPU data */
};

#undef __FUNCT__
#define __FUNCT__ "VecCUSPAllocateCheck"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  Vec_Seq        *s = (Vec_Seq*)v->data;;

  PetscFunctionBegin;
  if (v->valid_GPU_array == PETSC_CUSP_UNALLOCATED){
    try {
      v->spptr = new Vec_CUSP;
      ((Vec_CUSP*)v->spptr)->GPUarray = new CUSPARRAY;
      ((Vec_CUSP*)v->spptr)->GPUarray->resize((PetscBLASInt)v->map->n);
      ierr = WaitForGPU();CHKERRQ(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    if (s->array == 0){
      v->valid_GPU_array = PETSC_CUSP_GPU;
    } else{
      v->valid_GPU_array = PETSC_CUSP_CPU;
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCUSPCopyToGPU"
/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PETSC_STATIC_INLINE PetscErrorCode VecCUSPCopyToGPU(Vec v)
{
  PetscBLASInt   cn = v->map->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUSP_CPU){
    ierr = PetscLogEventBegin(VEC_CUSPCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    try{
      ((Vec_CUSP*)v->spptr)->GPUarray->assign(*(PetscScalar**)v->data,*(PetscScalar**)v->data + cn);
      ierr = WaitForGPU();CHKERRQ(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogEventEnd(VEC_CUSPCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUSP_BOTH;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPCopyToGPUSome"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPCopyToGPUSome(Vec v,CUSPINTARRAYCPU *indicesCPU,CUSPINTARRAYGPU *indicesGPU)
{
  Vec_Seq        *s = (Vec_Seq *)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUSP_CPU) {
    ierr = PetscLogEventBegin(VEC_CUSPCopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
    thrust::copy(thrust::make_permutation_iterator(s->array,indicesCPU->begin()),
		 thrust::make_permutation_iterator(s->array,indicesCPU->end()),
		 thrust::make_permutation_iterator(((Vec_CUSP *)v->spptr)->GPUarray->begin(),indicesGPU->begin()));
    ierr = PetscLogEventEnd(VEC_CUSPCopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  v->valid_GPU_array = PETSC_CUSP_GPU;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPGetArrayReadWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPGetArrayReadWrite(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *a   = 0;
  ierr = VecCUSPCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_CUSP *)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPRestoreArrayReadWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPRestoreArrayReadWrite(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v->valid_GPU_array != PETSC_CUSP_UNALLOCATED){
    v->valid_GPU_array = PETSC_CUSP_GPU;
  }
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPGetArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPGetArrayRead(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *a   = 0;
  ierr = VecCUSPCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_CUSP *)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPRestoreArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPRestoreArrayRead(Vec v, CUSPARRAY** a)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPGetArrayWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPGetArrayWrite(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *a   = 0;
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_CUSP *)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPRestoreArrayWrite"
PETSC_STATIC_INLINE PetscErrorCode VecCUSPRestoreArrayWrite(Vec v, CUSPARRAY** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v->valid_GPU_array != PETSC_CUSP_UNALLOCATED){
    v->valid_GPU_array = PETSC_CUSP_GPU;
  }
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
