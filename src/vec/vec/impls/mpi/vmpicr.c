
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
  PetscFunctionBegin;
  PetscCall(VecCreate(comm,v));
  PetscCall(VecSetSizes(*v,n,N));
  PetscCall(VecSetType(*v,VECMPI));
  PetscFunctionReturn(0);
}
