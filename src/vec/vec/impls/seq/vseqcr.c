
/*
   Implements the sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I  "petscvec.h"   I*/

/*@
   VecCreateSeq - Creates a standard, sequential array-style vector.

   Collective

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
-  n - the vector length

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: intermediate

.seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
PetscErrorCode  VecCreateSeq(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
