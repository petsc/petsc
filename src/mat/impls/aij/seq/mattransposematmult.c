
/*
  Defines matrix-matrix product routines for
          C = A^T * B and C = A * B^t
  with A SeqAIJ and B SeqDense
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

PetscErrorCode MatDestroy_SeqDense_MatTransMatMult(void *data)
{
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult *)data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&atb->mA));
  PetscCall(VecDestroy(&atb->bt));
  PetscCall(VecDestroy(&atb->ct));
  PetscCall(PetscFree(atb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTMatTMultNumeric_SeqAIJ_SeqDense(Mat, Mat, Mat);

PETSC_INTERN PetscErrorCode MatTMatTMultSymbolic_SeqAIJ_SeqDense(Mat A, Mat B, PetscReal fill, Mat C)
{
  Mat_MatTransMatMult *atb;
  PetscBool            cisdense;
  PetscInt             dofm;

  PetscFunctionBegin;
  MatCheckProduct(C, 4);
  PetscCheck(!C->product->data, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Extra product struct not empty");
  PetscCheck(C->product->type == MATPRODUCT_ABt || C->product->type == MATPRODUCT_AtB, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not for product type %s", MatProductTypes[C->product->type]);

  /* create output dense matrix C */
  if (C->product->type == MATPRODUCT_AtB) {
    PetscCall(MatSetSizes(C, A->cmap->n, B->cmap->N, A->cmap->n, B->cmap->N));
    dofm = B->cmap->n;
  } else {
    PetscCall(MatSetSizes(C, A->rmap->n, B->rmap->N, A->rmap->n, B->rmap->N));
    dofm = B->rmap->n;
  }
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C, &cisdense, MATSEQDENSE, MATSEQDENSECUDA, ""));
  if (!cisdense) PetscCall(MatSetType(C, ((PetscObject)B)->type_name));
  PetscCall(MatSetUp(C));

  /* create additional data structure for the product */
  PetscCall(PetscNew(&atb));
  PetscCall(MatCreateMAIJ(A, dofm, &atb->mA));
  PetscCall(MatCreateVecs(atb->mA, &atb->ct, &atb->bt));
  C->product->data    = atb;
  C->product->destroy = MatDestroy_SeqDense_MatTransMatMult;

  if (C->product->type == MATPRODUCT_AtB) {
    C->ops->transposematmultnumeric = MatTMatTMultNumeric_SeqAIJ_SeqDense;
  } else {
    C->ops->mattransposemultnumeric = MatTMatTMultNumeric_SeqAIJ_SeqDense;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatTMatTMultNumeric_SeqAIJ_SeqDense(Mat A, Mat B, Mat C)
{
  PetscInt             i, j, m = A->rmap->n, n = A->cmap->n, blda, clda;
  PetscInt             mdof = C->cmap->N;
  const PetscScalar   *Barray;
  PetscScalar         *Carray;
  Mat_MatTransMatMult *atb;
  Vec                  bt, ct;

  PetscFunctionBegin;
  MatCheckProduct(C, 3);
  PetscCheck(C->product->type == MATPRODUCT_ABt || C->product->type == MATPRODUCT_AtB, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not for product type %s", MatProductTypes[C->product->type]);
  atb = (Mat_MatTransMatMult *)C->product->data;
  PetscCheck(atb, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing product struct");
  bt = atb->bt;
  ct = atb->ct;

  PetscCall(MatDenseGetArrayRead(B, &Barray));
  PetscCall(MatDenseGetLDA(B, &blda));
  PetscCall(MatDenseGetArrayWrite(C, &Carray));
  PetscCall(MatDenseGetLDA(C, &clda));
  if (C->product->type == MATPRODUCT_AtB) { /* transpose local array of B, then copy it to vector bt */
    const PetscScalar *ctarray;
    PetscScalar       *btarray;

    PetscCall(VecGetArrayWrite(bt, &btarray));
    for (j = 0; j < mdof; j++) {
      for (i = 0; i < m; i++) btarray[i * mdof + j] = Barray[j * blda + i];
    }
    PetscCall(VecRestoreArrayWrite(bt, &btarray));

    /* compute ct = mA^T * cb */
    PetscCall(MatMultTranspose(atb->mA, bt, ct));

    /* transpose local array of ct to matrix C */
    PetscCall(VecGetArrayRead(ct, &ctarray));
    for (j = 0; j < mdof; j++) {
      for (i = 0; i < n; i++) Carray[j * clda + i] = ctarray[i * mdof + j];
    }
    PetscCall(VecRestoreArrayRead(ct, &ctarray));
  } else {
    const PetscScalar *btarray;
    PetscScalar       *ctarray;

    if (blda == B->rmap->n) {
      PetscCall(VecPlaceArray(ct, Barray));
    } else {
      PetscInt bn = B->cmap->n;
      PetscInt bm = B->rmap->n;

      PetscCall(VecGetArrayWrite(ct, &ctarray));
      for (j = 0; j < bn; j++) {
        for (i = 0; i < bm; i++) ctarray[j * bm + i] = Barray[j * blda + i];
      }
      PetscCall(VecRestoreArrayWrite(ct, &ctarray));
    }

    PetscCall(MatMult(atb->mA, ct, bt));
    if (blda == B->rmap->n) PetscCall(VecResetArray(ct));
    PetscCall(VecGetArrayRead(bt, &btarray));
    for (j = 0; j < mdof; j++) {
      for (i = 0; i < m; i++) Carray[j * clda + i] = btarray[i * mdof + j];
    }
    PetscCall(VecRestoreArrayRead(bt, &btarray));
  }
  PetscCall(MatDenseRestoreArrayRead(B, &Barray));
  PetscCall(MatDenseRestoreArray(C, &Carray));
  PetscFunctionReturn(PETSC_SUCCESS);
}
