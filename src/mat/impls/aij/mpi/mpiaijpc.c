
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode MatGetDiagonalBlock_MPIAIJ(Mat A, Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPIAIJ *)A->data)->A;
  PetscFunctionReturn(0);
}
