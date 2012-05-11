#ifndef __CUSPARSEMATIMPL 
#define __CUSPARSEMATIMPL

#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

#include <cusparse_v2.h>

/* New Way */
#include "tx_sparse_interface.h"

// this is such a hack ... but I don't know of another way to pass this variable
// from one GPU_Matrix_Ifc class to another. This is necessary for the parallel
//  SpMV. Essentially, I need to use the same stream variable in two different
//  data structures. I do this by creating a single instance of that stream
//  and reuse it.
cudaStream_t theBodyStream=0;

#include <algorithm>
#include <vector>
#include <string>
#include <thrust/sort.h>
#include <thrust/fill.h>

// Single instance of the cusparse handle for the class.
cusparseHandle_t MAT_cusparseHandle=0;

struct Mat_SeqAIJCUSPARSETriFactors {
  GPU_Matrix_Ifc* loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  GPU_Matrix_Ifc* upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
  CUSPARRAY* tempvec;
  const GPUStorageFormat  format;  /* the storage format for the matrix on the device */
};

struct Mat_SeqAIJCUSPARSE {
  GPU_Matrix_Ifc*   mat; /* pointer to the matrix on the GPU */
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt          nonzerorow; /* number of nonzero rows ... used in the flop calculations */
  const GPUStorageFormat  format;   /* the storage format for the matrix on the device */
};

extern PetscErrorCode MatCUSPARSECopyToGPU(Mat);
//extern PetscErrorCode MatGetFactor_seqaij_cusparse(Mat,MatFactorType,Mat*);
//extern PetscErrorCode MatFactorGetSolverPackage_seqaij_cusparse(Mat,const MatSolverPackage *);
//extern PetscErrorCode MatCUSPARSECopyFromGPU(Mat, CUSPMATRIX *);
#endif
