#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>

PetscErrorCode MatDestroySubMatrix_Dummy(Mat C)
{
  Mat_SubSppt    *submatj = (Mat_SubSppt*)C->data;

  PetscFunctionBegin;
  PetscCall(submatj->destroy(C));
  PetscCall(MatDestroySubMatrix_Private(submatj));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroySubMatrices_Dummy(PetscInt n, Mat *mat[])
{
  PetscFunctionBegin;
  /* Destroy dummy submatrices (*mat)[n]...(*mat)[n+nstages-1] used for reuse struct Mat_SubSppt */
  if ((*mat)[n]) {
    PetscBool      isdummy;
    PetscCall(PetscObjectTypeCompare((PetscObject)(*mat)[n],MATDUMMY,&isdummy));
    if (isdummy) {
      Mat_SubSppt* smat = (Mat_SubSppt*)((*mat)[n]->data); /* singleis and nstages are saved in (*mat)[n]->data */

      if (smat && !smat->singleis) {
        PetscInt i,nstages=smat->nstages;
        for (i=0; i<nstages; i++) {
          PetscCall(MatDestroy(&(*mat)[n+i]));
        }
      }
    }
  }

  /* memory is allocated even if n = 0 */
  PetscCall(PetscFree(*mat));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Dummy(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,NULL));
  PetscFunctionReturn(0);
}

/*MC
   MATDUMMY - A matrix type to be used for reusing specific internal data structure.

  Level: developer

.seealso: Mat

M*/

PETSC_EXTERN PetscErrorCode MatCreate_Dummy(Mat A)
{
  PetscFunctionBegin;
  /* matrix ops */
  PetscCall(PetscMemzero(A->ops,sizeof(struct _MatOps)));
  A->ops->destroy            = MatDestroy_Dummy;
  A->ops->destroysubmatrices = MatDestroySubMatrices_Dummy;

  /* special MATPREALLOCATOR functions */
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATDUMMY));
  PetscFunctionReturn(0);
}
