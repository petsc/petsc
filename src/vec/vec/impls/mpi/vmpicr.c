
/*
   This file contains routines for Parallel vector operations.
 */

#include <petscvec.h>   /*I  "petscvec.h"   I*/

/*@
   VecCreateMPI - Creates a parallel vector.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
-  N - global vector length (or PETSC_DETERMINE to have calculated if n is given)

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: intermediate

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPIWithArray(), VecCreateGhostWithArray(), VecMPISetGhost()

@*/
PetscErrorCode  VecCreateMPI(MPI_Comm comm,PetscInt n,PetscInt N,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,N);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECMPI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
