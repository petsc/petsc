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
#include "tx_sparse_interface.h"

struct Mat_SeqAIJCUSP {
  GPU_Matrix_Ifc*   mat; /* pointer to the matrix on the GPU */
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt          nonzerorow; /* number of nonzero rows ... used in the flop calculations */
  MatCUSPStorageFormat     format;   /* the storage format for the matrix on the device */
};

#else // PETSC_HAVE_TXPETSCGPU not defined!

struct Mat_SeqAIJCUSP {
  CUSPMATRIX*       mat; /* pointer to the matrix on the GPU */
  CUSPINTARRAYGPU*  indices; /*pointer to an array containing the nonzero row indices, should usecprow be true*/
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt          nonzerorow; /* number of nonzero rows ... used in the flop calculations */
};
#endif


extern PetscErrorCode MatCUSPCopyToGPU(Mat);
extern PetscErrorCode MatCUSPCopyFromGPU(Mat, CUSPMATRIX *);
#endif
