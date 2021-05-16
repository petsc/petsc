#if !defined(__VIENNACLMATIMPL)
#define __VIENNACLMATIMPL

#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/

/* Pulls in some ViennaCL includes as well as VIENNACL_WITH_OPENCL: */
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

/* for everything else */
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/compressed_compressed_matrix.hpp"

typedef viennacl::compressed_matrix<PetscScalar>   ViennaCLAIJMatrix;
typedef viennacl::compressed_compressed_matrix<PetscScalar>   ViennaCLCompressedAIJMatrix;

struct Mat_SeqAIJViennaCL {
  Mat_SeqAIJViennaCL() : tempvec(NULL), mat(NULL), compressed_mat(NULL) {}
  ViennaCLVector               *tempvec;
  ViennaCLAIJMatrix            *mat;  /* pointer to the matrix on the GPU */
  ViennaCLCompressedAIJMatrix  *compressed_mat; /* compressed CSR */
};

PETSC_INTERN PetscErrorCode MatViennaCLCopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatViennaCLCopyFromGPU(Mat, ViennaCLAIJMatrix*);
#endif
