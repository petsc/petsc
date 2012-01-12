#ifndef __CUSPMATIMPL 
#define __CUSPMATIMPL

#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
/*for MatCreateSeqAIJCUSPFromTriple*/
#include <cusp/coo_matrix.h>

#define CUSPMATRIX cusp::csr_matrix<PetscInt,PetscScalar,cusp::device_memory>

struct Mat_SeqAIJCUSP {
  CUSPMATRIX*       mat; /* pointer to the matrix on the GPU */
  CUSPINTARRAYGPU*  indices; /*pointer to an array containing the nonzero row indices, should usecprow be true*/
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
};

extern PetscErrorCode MatCUSPCopyToGPU(Mat);
extern PetscErrorCode MatCUSPCopyFromGPU(Mat, CUSPMATRIX *);
#endif
