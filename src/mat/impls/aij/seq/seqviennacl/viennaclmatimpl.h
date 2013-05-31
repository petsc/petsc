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

#undef __FUNCT__
#define __FUNCT__ "MatSetFromOptions_SeqViennaCL"
static PetscErrorCode MatSetFromOptions_SeqViennaCL(Mat A)
{
  PetscErrorCode       ierr;
  PetscBool            flg;

  PetscFunctionBegin;
  ViennaCLSetFromOptions((PetscObject)A);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatViennaCLCopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatViennaCLCopyFromGPU(Mat, ViennaCLAIJMatrix*);
#endif
