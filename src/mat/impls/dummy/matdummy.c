#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>

PetscErrorCode MatDestroySubMatrices_SeqXAIJ(PetscInt n, Mat *mat[])
{
  PetscFunctionBegin;
  /* Destroy dummy submatrices (*mat)[n]...(*mat)[n+nstages-1] used for reuse struct Mat_SubSppt */
  if ((*mat)[n]) {
    PetscBool isdummy;
    PetscCall(PetscObjectTypeCompare((PetscObject)(*mat)[n], MATDUMMY, &isdummy));
    if (isdummy) {
      Mat_SubSppt *smat = (Mat_SubSppt *)(*mat)[n]->data; /* singleis and nstages are saved in (*mat)[n]->data */

      if (smat && !smat->singleis) n += smat->nstages;
    }
  }

  PetscCall(MatDestroyMatrices(n, mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Dummy(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatDestroySubMatrix_Private((Mat_SubSppt *)A->data));
  A->data = NULL;
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATDUMMY - A matrix type to be used for reusing specific internal data structure.

  Level: developer

.seealso: `Mat`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Dummy(Mat A)
{
  PetscFunctionBegin;
  /* matrix ops */
  PetscCall(PetscMemzero(A->ops, sizeof(struct _MatOps)));
  A->ops->destroy            = MatDestroy_Dummy;
  A->ops->destroysubmatrices = MatDestroySubMatrices_SeqXAIJ;

  /* special MATPREALLOCATOR functions */
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATDUMMY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
