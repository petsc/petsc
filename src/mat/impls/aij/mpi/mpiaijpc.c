#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpiaijpc.c,v 1.39 1998/10/13 15:18:20 bsmith Exp bsmith $";
#endif
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
