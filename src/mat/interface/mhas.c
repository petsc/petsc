#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mhas.c,v 1.10 1997/07/09 20:53:23 balay Exp bsmith $";
#endif


#include "petsc.h"
#include "src/mat/matimpl.h"        /*I "mat.h" I*/
       
#undef __FUNC__  
#define __FUNC__ "MatHasOperation"
/*@
    MatHasOperation - Determines if the given matrix supports the particular
    operation.

   Input Parameters:
.  mat - the matrix
.  op - the operation, for example, MATOP_GET_DIAGONAL

   Output Parameter:
.  has - either PETSC_TRUE or PETSC_FALSE

   Notes:
   See the file petsc/include/mat.h for a complete list of matrix
   operations, which all have the form MATOP_<OPERATION>, where
   <OPERATION> is the name (in all capital letters) of the
   user-level routine.  E.g., MatNorm() -> MATOP_NORM.

.keywords: matrix, has, operation

.seealso: MatCreateShell()
@*/
int MatHasOperation(Mat mat,MatOperation op,PetscTruth *has)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (((void **)&mat->ops)[op]) {*has =  PETSC_TRUE;}
  else {*has = PETSC_FALSE;}
  return 0;
}
