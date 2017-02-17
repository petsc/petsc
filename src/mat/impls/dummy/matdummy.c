#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/
#include <../src/sys/utils/hash.h>

typedef struct {
  PetscHashJK ht;
  PetscInt   *dnz, *onz;
} Mat_Preallocator;

PetscErrorCode MatDestroy_Dummy(Mat A)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  printf("MatDestroy_Dummy...\n");
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) A, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_Dummy(Mat A, MatOption op, PetscBool flg)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
   MATDUMMY - A matrix type to be used for reusing specific internal data structure.

  Level: advanced

.seealso: Mat

M*/

PETSC_EXTERN PetscErrorCode MatCreate_Dummy(Mat A)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* matrix ops */
  ierr = PetscMemzero(A->ops, sizeof(struct _MatOps));CHKERRQ(ierr);
  A->ops->destroy                 = MatDestroy_Dummy;

  /* special MATPREALLOCATOR functions */
  ierr = PetscObjectChangeTypeName((PetscObject) A, MATDUMMY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
