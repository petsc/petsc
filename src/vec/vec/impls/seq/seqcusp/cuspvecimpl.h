#ifndef __CUSPVECIMPL
#define __CUSPVECIMPL

#include <petsc-private/vecimpl.h>

#include <algorithm>
#include <vector>
#include <string>

#include <cublas.h>
#include <cusp/blas.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>

#define CUSPARRAY cusp::array1d<PetscScalar,cusp::device_memory>
#define CUSPARRAYCPU cusp::array1d<PetscScalar,cusp::host_memory>
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
extern PetscErrorCode VecCUSPCopyToGPU(Vec);
extern PetscErrorCode VecCUSPAllocateCheck(Vec);
extern PetscErrorCode VecCUSPAllocateCheckHost(Vec);
EXTERN_C_BEGIN
extern PetscErrorCode  VecCreate_SeqCUSP(Vec);
EXTERN_C_END
extern PetscErrorCode VecView_Seq(Vec,PetscViewer);
extern PetscErrorCode VecDestroy_SeqCUSP(Vec);
extern PetscErrorCode VecAYPX_SeqCUSP(Vec,PetscScalar,Vec);
extern PetscErrorCode VecSetRandom_SeqCUSP(Vec,PetscRandom);

extern PetscErrorCode VecCUSPCopyToGPU_Public(Vec);
extern PetscErrorCode VecCUSPAllocateCheck_Public(Vec);

#ifdef PETSC_HAVE_TXPETSCGPU
#include "tx_vector_interface.h"
#endif

struct  _p_PetscCUSPIndices {
#ifdef PETSC_HAVE_TXPETSCGPU
  GPU_Indices<PetscInt, PetscScalar> * sendIndices;
  GPU_Indices<PetscInt, PetscScalar> * recvIndices;
#else
  CUSPINTARRAYCPU sendIndicesCPU;
  CUSPINTARRAYGPU sendIndicesGPU;

  CUSPINTARRAYCPU recvIndicesCPU;
  CUSPINTARRAYGPU recvIndicesGPU;
#endif
};

#ifdef PETSC_HAVE_TXPETSCGPU
extern PetscErrorCode VecCUSPCopySomeToContiguousBufferGPU(Vec, PetscCUSPIndices);
extern PetscErrorCode VecCUSPCopySomeFromContiguousBufferGPU(Vec, PetscCUSPIndices);
#endif

#define CHKERRCUSP(err) if (((int)err) != (int)CUBLAS_STATUS_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error %d",err)

#define VecCUSPCastToRawPtr(x) thrust::raw_pointer_cast(&(x)[0])

#define WaitForGPU() PetscCUSPSynchronize ? cudaThreadSynchronize() : 0

struct Vec_CUSP {
  CUSPARRAY*       GPUarray;  /* this always holds the GPU data */  
#ifdef PETSC_HAVE_TXPETSCGPU
  GPU_Vector<PetscInt, PetscScalar> * GPUvector; /* this always holds the GPU data */
#endif
};


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
  v->valid_GPU_array = PETSC_CUSP_GPU;
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
  v->valid_GPU_array = PETSC_CUSP_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
