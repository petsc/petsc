
/*
   Implements the sequential vectors.
*/

#include <petsc-private/vecimpl.h>          /*I  "petscvec.h"   I*/
#include <../src/vec/vec/impls/dvecimpl.h> 

#undef __FUNCT__  
#define __FUNCT__ "VecCreateSeq"
/*@
   VecCreateSeq - Creates a standard, sequential array-style vector.

   Collective on MPI_Comm

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
-  n - the vector length 

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: intermediate

   Concepts: vectors^creating sequential

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
