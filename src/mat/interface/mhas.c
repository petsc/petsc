
#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/

/*@
    MatHasOperation - Determines whether the given matrix supports the particular
    operation.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  op - the operation, for example, MATOP_GET_DIAGONAL

   Output Parameter:
.  has - either PETSC_TRUE or PETSC_FALSE

   Level: advanced

   Notes:
   See the file include/petscmat.h for a complete list of matrix
   operations, which all have the form MATOP_<OPERATION>, where
   <OPERATION> is the name (in all capital letters) of the
   user-level routine.  E.g., MatNorm() -> MATOP_NORM.

.keywords: matrix, has, operation

.seealso: MatCreateShell()
@*/
PetscErrorCode  MatHasOperation(Mat mat,MatOperation op,PetscBool  *has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(has,3);
  if (mat->ops->hasoperation) {
    PetscErrorCode ierr;
    ierr = (*mat->ops->hasoperation)(mat,op,has);CHKERRQ(ierr);
  } else {
    if (((void**)mat->ops)[op]) *has =  PETSC_TRUE;
    else {
      if (op == MATOP_CREATE_SUBMATRIX) {
        PetscErrorCode ierr;
        PetscMPIInt    size;

        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRQ(ierr);
        if (size == 1) {
          ierr = MatHasOperation(mat,MATOP_CREATE_SUBMATRICES,has);CHKERRQ(ierr);
        } else {
          *has = PETSC_FALSE;
        }
      } else {
        *has = PETSC_FALSE;
      }
    }
  }
  PetscFunctionReturn(0);
}
