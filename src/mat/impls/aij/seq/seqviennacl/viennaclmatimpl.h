#if !defined(__VIENNACLMATIMPL)
#define __VIENNACLMATIMPL

#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

/* for everything else */
#include "viennacl/compressed_matrix.hpp"


/* Old way */
typedef viennacl::compressed_matrix<PetscScalar>   ViennaCLAIJMatrix;


struct Mat_SeqAIJViennaCL {
  ViennaCLAIJMatrix      *mat;  /* pointer to the matrix on the GPU */
};


PETSC_INTERN PetscErrorCode MatViennaCLCopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatViennaCLCopyFromGPU(Mat, ViennaCLAIJMatrix*);
#endif
