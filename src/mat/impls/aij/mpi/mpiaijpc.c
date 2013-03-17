
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_MPIAIJ"
PetscErrorCode  MatGetDiagonalBlock_MPIAIJ(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPIAIJ*)A->data)->A;
  PetscFunctionReturn(0);
}

