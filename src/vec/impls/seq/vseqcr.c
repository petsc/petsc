#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vseqcr.c,v 1.7 1999/05/04 20:30:44 balay Exp bsmith $";
#endif
/*
   Implements the sequential vectors.
*/

#include "src/vec/vecimpl.h"          /*I  "vec.h"   I*/
#include "src/vec/impls/dvecimpl.h" 

#undef __FUNC__  
#define __FUNC__ "VecCreateSeq"
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

.keywords: vector, sequential, create, BLAS

.seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
int VecCreateSeq(MPI_Comm comm, int n, Vec *v)
{
  int ierr,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(1,1,"Cannot only create sequential vectors on 1 processor");
  ierr = VecCreate(comm,n,n,v);CHKERRQ(ierr);
  ierr = VecSetType(*v,VEC_SEQ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
