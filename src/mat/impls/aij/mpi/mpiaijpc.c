/*$Id: mpiaijpc.c,v 1.41 1999/10/24 14:02:16 bsmith Exp bsmith $*/
#include "src/mat/impls/aij/mpi/mpiaij.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatGetDiagonalBlock_MPIAIJ"
int MatGetDiagonalBlock_MPIAIJ(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscFunctionBegin;
  *a      = ((Mat_MPIAIJ *)A->data)->A;
  *iscopy = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
