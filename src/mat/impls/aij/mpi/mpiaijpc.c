/*$Id: mpiaijpc.c,v 1.44 2001/01/15 21:45:38 bsmith Exp balay $*/
#include "src/mat/impls/aij/mpi/mpiaij.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonalBlock_MPIAIJ"
int MatGetDiagonalBlock_MPIAIJ(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscFunctionBegin;
  *a      = ((Mat_MPIAIJ *)A->data)->A;
  *iscopy = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
