#ifndef lint
static char vcid[] = "$Id: mhas.c,v 1.2 1996/02/01 17:29:02 bsmith Exp bsmith $";
#endif


#include "petsc.h"
#include "matimpl.h"        /*I "mat.h" I*/
       
/*@
     MatHasOperation - Determines if the given matrix supports
            the particular operation.

  Input Parameters:
.  mat - the matrix
.  op - the operation, for example, MAT_GET_DIAGONAL

  Output Parameters:
.  has - either PETSC_TRUE or PETSC_FALSE

@*/
int MatHasOperation(Mat mat,MatOperation op,PetscTruth *has)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (((void **)&mat->ops)[op]) {*has =  PETSC_TRUE;}
  else {*has = PETSC_FALSE;}
  return 0;
}
