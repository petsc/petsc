#if !defined(__VIENNACLMATIMPL)
#define __VIENNACLMATIMPL

#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/

/* Pulls in some ViennaCL includes as well as VIENNACL_WITH_OPENCL: */
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

/* for everything else */
#include "viennacl/compressed_matrix.hpp"


typedef viennacl::compressed_matrix<PetscScalar>   ViennaCLAIJMatrix;


struct Mat_SeqAIJViennaCL {
  ViennaCLAIJMatrix      *mat;  /* pointer to the matrix on the GPU */
};

PETSC_INTERN PetscErrorCode MatViennaCLCopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatViennaCLCopyFromGPU(Mat, ViennaCLAIJMatrix*);
#endif
