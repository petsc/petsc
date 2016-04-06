#if !defined(__CUSPMATIMPL)
#define __CUSPMATIMPL

#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

/*for MatCreateSeqAIJCUSPFromTriple*/
#include <cusp/coo_matrix.h>
/* for everything else */
#include <cusp/csr_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/multiply.h>

/* need the thrust version */
#include <thrust/version.h>

#include <algorithm>
#include <vector>
#include <string>
#include <thrust/sort.h>
#include <thrust/fill.h>


/* Old way */
#define CUSPMATRIX cusp::csr_matrix<PetscInt,PetscScalar,cusp::device_memory>
#define CUSPMATRIXELL cusp::ell_matrix<PetscInt,PetscScalar,cusp::device_memory>
#define CUSPMATRIXDIA cusp::dia_matrix<PetscInt,PetscScalar,cusp::device_memory>

struct Mat_SeqAIJCUSP {
  void                 *mat; /* pointer to the matrix on the GPU */
  CUSPINTARRAYGPU      *indices; /*pointer to an array containing the nonzero row indices, should usecprow be true*/
  CUSPARRAY            *tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt             nonzerorow;   /* number of nonzero rows ... used in the flop calculations */
  MatCUSPStorageFormat format;   /* the storage format for the matrix on the device */
  cudaStream_t         stream;   /* a stream for the parallel SpMV ... this is not owned and should not be deleted */
};

PETSC_EXTERN PetscErrorCode MatCUSPCopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatCUSPCopyFromGPU(Mat, CUSPMATRIX*);
PETSC_INTERN PetscErrorCode MatCUSPSetStream(Mat, const cudaStream_t stream);
#endif
