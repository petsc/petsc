/*$Id: mpiaijpc.c,v 1.40 1999/01/27 19:47:24 bsmith Exp bsmith $*/
#include "src/mat/impls/aij/mpi/mpiaij.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatGetDiagonalBlock_MPIAIJ"
int MatGetDiagonalBlock_MPIAIJ(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscFunctionBegin;
  *a      = ((Mat_MPIAIJ *)A->data)->A;
  *iscopy = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
