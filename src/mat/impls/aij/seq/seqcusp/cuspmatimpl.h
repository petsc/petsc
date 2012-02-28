#ifndef __CUSPMATIMPL 
#define __CUSPMATIMPL

#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

/*for MatCreateSeqAIJCUSPFromTriple*/
#include <cusp/coo_matrix.h>
/* for everything else */
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

/* need the thrust version */
#include <thrust/version.h>

/* Old way */
#define CUSPMATRIX cusp::csr_matrix<PetscInt,PetscScalar,cusp::device_memory>

/* New Way */
#ifdef PETSC_HAVE_TXPETSCGPU
#include "tx_spmv_interface.h"
#endif /* PETSC_HAVE_TXPETSCGPU */

struct Mat_SeqAIJCUSP {
  CUSPMATRIX*       mat; /* pointer to the matrix on the GPU */
  CUSPINTARRAYGPU*  indices; /*pointer to an array containing the nonzero row indices, should usecprow be true*/
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt            nonzerorow; /* number of nonzero rows ... used in the flop calculations */
};


extern PetscErrorCode MatCUSPCopyToGPU(Mat);
extern PetscErrorCode MatCUSPCopyFromGPU(Mat, CUSPMATRIX *);
#endif
