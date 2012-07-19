#ifndef __CUSPARSEMATIMPL 
#define __CUSPARSEMATIMPL

#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

#include <cusparse_v2.h>

/* New Way */
#include "tx_sparse_interface.h"

#include <algorithm>
#include <vector>
#include <string>
#include <thrust/sort.h>
#include <thrust/fill.h>

// Single instance of the cusparse handle for the class.
cusparseHandle_t MAT_cusparseHandle=0;

MatCUSPARSEStorageFormat cusparseMatSolveStorageFormat=MAT_CUSPARSE_CSR;

struct Mat_SeqAIJCUSPARSETriFactors {
  GPU_Matrix_Ifc* loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  GPU_Matrix_Ifc* upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
  CUSPARRAY* tempvec;
  MatCUSPARSEStorageFormat  format;  /* the storage format for the matrix on the device */
};

struct Mat_SeqAIJCUSPARSE {
  GPU_Matrix_Ifc*   mat; /* pointer to the matrix on the GPU */
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt          nonzerorow; /* number of nonzero rows ... used in the flop calculations */
  MatCUSPARSEStorageFormat format;   /* the storage format for the matrix on the device */
};

extern PetscErrorCode MatCUSPARSECopyToGPU(Mat);
//extern PetscErrorCode MatGetFactor_seqaij_cusparse(Mat,MatFactorType,Mat*);
//extern PetscErrorCode MatFactorGetSolverPackage_seqaij_cusparse(Mat,const MatSolverPackage *);
//extern PetscErrorCode MatCUSPARSECopyFromGPU(Mat, CUSPMATRIX *);
#endif
