/*
   Implements the sequential vectors.
*/

#include "vecimpl.h"          /*I  "petscvec.h"   I*/
#include "src/vec/impls/dvecimpl.h" 

#undef __FUNCT__  
#define __FUNCT__ "VecCreateSeq"
/*@C
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
int VecCreateSeq(MPI_Comm comm,int n,Vec *v)
{
  int ierr,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(1,"Comm_size > 1; sequential vector can be created on 1 processor only");
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
