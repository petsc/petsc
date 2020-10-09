#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>

PetscErrorCode MatDestroySubMatrix_Dummy(Mat C)
{
  PetscErrorCode ierr;
  Mat_SubSppt    *submatj = (Mat_SubSppt*)C->data;

  PetscFunctionBegin;
  ierr = submatj->destroy(C);CHKERRQ(ierr);
  ierr = MatDestroySubMatrix_Private(submatj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroySubMatrices_Dummy(PetscInt n, Mat *mat[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy dummy submatrices (*mat)[n]...(*mat)[n+nstages-1] used for reuse struct Mat_SubSppt */
  if ((*mat)[n]) {
    PetscBool      isdummy;
    ierr = PetscObjectTypeCompare((PetscObject)(*mat)[n],MATDUMMY,&isdummy);CHKERRQ(ierr);
    if (isdummy) {
      Mat_SubSppt* smat = (Mat_SubSppt*)((*mat)[n]->data); /* singleis and nstages are saved in (*mat)[n]->data */

      if (smat && !smat->singleis) {
        PetscInt i,nstages=smat->nstages;
        for (i=0; i<nstages; i++) {
          ierr = MatDestroy(&(*mat)[n+i]);CHKERRQ(ierr);
        }
      }
    }
  }

  /* memory is allocated even if n = 0 */
  ierr = PetscFree(*mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Dummy(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATDUMMY - A matrix type to be used for reusing specific internal data structure.

  Level: developer

.seealso: Mat

M*/

PETSC_EXTERN PetscErrorCode MatCreate_Dummy(Mat A)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* matrix ops */
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->ops->destroy            = MatDestroy_Dummy;
  A->ops->destroysubmatrices = MatDestroySubMatrices_Dummy;

  /* special MATPREALLOCATOR functions */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATDUMMY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
