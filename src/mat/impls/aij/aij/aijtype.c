
#include "private/matimpl.h"

/*MC
   MATAIJ - MATAIJ = "aij" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJ when constructed with a single process communicator,
   and MATMPIAIJ otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aij - sets the matrix type to "aij" during a call to MatSetFromOptions()

  Developer Notes: Subclasses include MATAIJCUSP, MATAIJPERM, MATAIJCRL, and also automatically switches over to use inodes when 
   enough exist.

  Level: beginner

.seealso: MatCreateMPIAIJ(), MatCreateSeqAIJ(), MATSEQAIJ,MATMPIAIJ
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJ"
PetscErrorCode  MatCreate_AIJ(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATAIJCRL - MATAIJCRL = "aijcrl" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJCRL when constructed with a single process communicator,
   and MATMPIAIJCRL otherwise.  As a result, for single process communicators, 
   MatSeqAIJSetPreallocation() is supported, and similarly MatMPIAIJSetPreallocation() is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aijcrl - sets the matrix type to "aijcrl" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJCRL,MATSEQAIJCRL,MATMPIAIJCRL, MATSEQAIJCRL, MATMPIAIJCRL
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJCRL"
PetscErrorCode  MatCreate_AIJCRL(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJCRL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIAIJCRL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
