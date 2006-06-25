#define PETSCMAT_DLL

#include "src/mat/matimpl.h"        /*I "petscmat.h" I*/
       
#undef __FUNCT__  
#define __FUNCT__ "MatHasOperation"
/*@
    MatHasOperation - Determines whether the given matrix supports the particular
    operation.

   Collective on Mat

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
PetscErrorCode PETSCMAT_DLLEXPORT MatHasOperation(Mat mat,MatOperation op,PetscTruth *has)
{
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(has,3);
  if (((void **)mat->ops)[op]) {*has =  PETSC_TRUE;}
  else {*has = PETSC_FALSE;}
  PetscFunctionReturn(0);
}
