
#include <../src/mat/impls/aij/mpi/mpiaij.h>

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_MPIAIJ"
PetscErrorCode  MatGetDiagonalBlock_MPIAIJ(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPIAIJ *)A->data)->A;
  PetscFunctionReturn(0);
}
EXTERN_C_END
