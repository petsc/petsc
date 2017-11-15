#if !defined(__CUSPVECIMPL)
#define __CUSPVECIMPL

#include <petsccusp.h>
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

PETSC_INTERN PetscErrorCode VecDotNorm2_SeqCUSP(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqCUSP(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqCUSP(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqCUSP(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecSet_SeqCUSP(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqCUSP(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqCUSP(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqCUSP(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqCUSP(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqCUSP(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqCUSP(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqCUSP(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqCUSP(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqCUSP(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecCopy_SeqCUSP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqCUSP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecAXPY_SeqCUSP(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqCUSP(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqCUSP(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecNorm_SeqCUSP(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecCUSPCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecCUSPAllocateCheck(Vec);
PETSC_INTERN PetscErrorCode VecCUSPAllocateCheckHost(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqCUSP(Vec);
PETSC_INTERN PetscErrorCode VecView_Seq(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecDestroy_SeqCUSP(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqCUSP(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqCUSP(Vec,PetscRandom);

PETSC_INTERN PetscErrorCode VecCUSPCopyToGPU_Public(Vec);
PETSC_INTERN PetscErrorCode VecCUSPAllocateCheck_Public(Vec);

#if defined(PETSC_HAVE_TXPETSCGPU)
#include "tx_vector_interface.h"
#endif

struct  _p_PetscCUSPIndices {
#if defined(PETSC_HAVE_TXPETSCGPU)
  GPU_Indices<PetscInt, PetscScalar> * sendIndices;
  GPU_Indices<PetscInt, PetscScalar> * recvIndices;
#else
  CUSPINTARRAYCPU sendIndicesCPU;
  CUSPINTARRAYGPU sendIndicesGPU;

  CUSPINTARRAYCPU recvIndicesCPU;
  CUSPINTARRAYGPU recvIndicesGPU;
#endif
};

#if defined(PETSC_HAVE_TXPETSCGPU)
PETSC_INTERN PetscErrorCode VecCUSPCopySomeToContiguousBufferGPU(Vec, PetscCUSPIndices);
PETSC_INTERN PetscErrorCode VecCUSPCopySomeFromContiguousBufferGPU(Vec, PetscCUSPIndices);
#endif

#define CHKERRCUSP(err) if (((int)err) != (int)CUBLAS_STATUS_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error %d",err)

#define VecCUSPCastToRawPtr(x) thrust::raw_pointer_cast(&(x)[0])

#define WaitForGPU() PetscCUSPSynchronize ? cudaThreadSynchronize() : 0

struct Vec_CUSP {
  CUSPARRAY *GPUarray;        /* this always holds the GPU data */
#if defined(PETSC_HAVE_TXPETSCGPU)
  GPU_Vector<PetscInt, PetscScalar> * GPUvector; /* this always holds the GPU data */
#endif
};

#endif
