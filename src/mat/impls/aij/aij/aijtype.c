#define PETSCMAT_DLL

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

  Level: beginner

.seealso: MatCreateMPIAIJ,MATSEQAIJ,MATMPIAIJ
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_AIJ(Mat A) 
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
   MATCRL - MATCRL = "crl" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQCRL when constructed with a single process communicator,
   and MATMPICRL otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation() is supported, and similarly MatMPIAIJSetPreallocation() is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type crl - sets the matrix type to "crl" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPICRL,MATSEQCRL,MATMPICRL, MATSEQCRL, MATMPICRL
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_CRL"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_CRL(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQCRL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPICRL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
