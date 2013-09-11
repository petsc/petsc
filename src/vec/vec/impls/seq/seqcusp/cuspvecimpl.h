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

struct  _p_PetscCUSPIndices {
  PetscInt ns;
  PetscInt sendLowestIndex;
  PetscInt nr;
  PetscInt recvLowestIndex;
};

#define CHKERRCUSP(err) if (((int)err) != (int)CUBLAS_STATUS_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error %d",err)

#define VecCUSPCastToRawPtr(x) thrust::raw_pointer_cast(&(x)[0])

#define WaitForGPU() PetscCUSPSynchronize ? cudaThreadSynchronize() : 0

struct Vec_CUSP {
  CUSPARRAY *GPUarray;        /* this always holds the GPU data */
  cudaStream_t stream;        /* A stream for doing asynchronous data transfers */
  PetscBool hostDataRegisteredAsPageLocked;
};

#endif
