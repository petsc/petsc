#if !defined(__CUDAVECIMPL)
#define __CUDAVECIMPL

#if defined(__CUDACC__)

#include <petsccuda.h>
#include <petsc/private/vecimpl.h>

#include <cublas_v2.h>

#define WaitForGPU() PetscCUDASynchronize ? cudaThreadSynchronize() : 0

struct Vec_CUDA {
  PetscScalar  *GPUarray;      /* this always holds the GPU data */
  cudaStream_t stream;        /* A stream for doing asynchronous data transfers */
  PetscBool    hostDataRegisteredAsPageLocked;
};

#endif

#include <cuda_runtime.h>

PETSC_INTERN PetscErrorCode VecDotNorm2_SeqCUDA(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqCUDA(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqCUDA(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqCUDA(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecSet_SeqCUDA(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqCUDA(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqCUDA(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqCUDA(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqCUDA(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqCUDA(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqCUDA(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqCUDA(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqCUDA(Vec,PetscScalar);
PETSC_EXTERN PetscErrorCode VecCopy_SeqCUDA(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqCUDA(Vec,Vec);
PETSC_INTERN PetscErrorCode VecAXPY_SeqCUDA(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqCUDA(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqCUDA(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecConjugate_SeqCUDA(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqCUDA(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecCUDACopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecCUDAAllocateCheck(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqCUDA(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqCUDA(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqCUDA(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqCUDA(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqCUDA(Vec,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqCUDA(Vec,Vec);
PETSC_INTERN PetscErrorCode VecCopy_SeqCUDA_Private(Vec xin,Vec yin);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqCUDA_Private(Vec xin,PetscRandom r);
PETSC_INTERN PetscErrorCode VecDestroy_SeqCUDA_Private(Vec v);
PETSC_INTERN PetscErrorCode VecResetArray_SeqCUDA_Private(Vec vin);
PETSC_INTERN PetscErrorCode VecCUDACopyToGPU_Public(Vec);
PETSC_INTERN PetscErrorCode VecCUDAAllocateCheck_Public(Vec);
PETSC_INTERN PetscErrorCode VecCUDACopyToGPUSome(Vec v, PetscCUDAIndices ci);
PETSC_INTERN PetscErrorCode VecCUDACopyFromGPUSome(Vec v, PetscCUDAIndices ci);

PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesCreate_PtoP(PetscInt, PetscInt*,PetscInt, PetscInt*,PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesCreate_StoS(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesDestroy(PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDA_StoS(Vec,Vec,PetscCUDAIndices,InsertMode,ScatterMode);

typedef enum {VEC_SCATTER_CUDA_STOS, VEC_SCATTER_CUDA_PTOP} VecCUDAScatterType;
typedef enum {VEC_SCATTER_CUDA_GENERAL, VEC_SCATTER_CUDA_STRIDED} VecCUDASequentialScatterMode;

struct  _p_VecScatterCUDAIndices_PtoP {
  PetscInt ns;
  PetscInt sendLowestIndex;
  PetscInt nr;
  PetscInt recvLowestIndex;
};

struct  _p_VecScatterCUDAIndices_StoS {
  /* from indices data */
  PetscInt *fslots;
  PetscInt fromFirst;
  PetscInt fromStep;
  VecCUDASequentialScatterMode fromMode;

  /* to indices data */
  PetscInt *tslots;
  PetscInt toFirst;
  PetscInt toStep;
  VecCUDASequentialScatterMode toMode;

  PetscInt n;
  PetscInt MAX_BLOCKS;
  PetscInt MAX_CORESIDENT_THREADS;
  cudaStream_t stream;
};

struct  _p_PetscCUDAIndices {
  void * scatter;
  VecCUDAScatterType scatterType;
};

/* complex single */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXaxpy cublasCaxpy
#define cublasXscal cublasCscal
#define cublasXdot cublasCdotc
#define cublasXdotu cublasCdotu
#define cublasXswap cublasCswap
#define cublasXnrm2 cublasCnrm2
#define cublasIXamax cublasIcamax
#define cublasXasum cublasCasum
#else /* complex double */
#define cublasXaxpy cublasZaxpy
#define cublasXscal cublasZscal
#define cublasXdot cublasZdotc
#define cublasXdotu cublasZdotu
#define cublasXswap cublasZswap
#define cublasXnrm2 cublasZnrm2
#define cublasIXamax cublasIzamax
#define cublasXasum cublasZasum
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXaxpy cublasSaxpy
#define cublasXscal cublasSscal
#define cublasXdot cublasSdot
#define cublasXdotu cublasSdot
#define cublasXswap cublasSswap
#define cublasXnrm2 cublasSnrm2
#define cublasIXamax cublasIsamax
#define cublasXasum cublasSasum
#else /* real double */
#define cublasXaxpy cublasDaxpy
#define cublasXscal cublasDscal
#define cublasXdot cublasDdot
#define cublasXdotu cublasDdot
#define cublasXswap cublasDswap
#define cublasXnrm2 cublasDnrm2
#define cublasIXamax cublasIdamax
#define cublasXasum cublasDasum
#endif
#endif

#endif
