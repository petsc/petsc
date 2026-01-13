/*
  Defines matrix-matrix-matrix product routines for SeqAIJ matrices
          D = A * B * C
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/

static PetscErrorCode MatProductCtxDestroy_SeqAIJ_MatMatMatMult(PetscCtxRt data)
{
  MatProductCtx_MatMatMatMult *matmatmatmult = *(MatProductCtx_MatMatMatMult **)data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&matmatmatmult->BC));
  PetscCall(PetscFree(matmatmatmult));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Mat A, Mat B, Mat C, PetscReal fill, Mat D)
{
  Mat                          BC;
  MatProductCtx_MatMatMatMult *matmatmatmult;
  char                        *alg;

  PetscFunctionBegin;
  MatCheckProduct(D, 5);
  PetscCheck(!D->product->data, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatCreate(PETSC_COMM_SELF, &BC));
  PetscCall(MatMatMultSymbolic_SeqAIJ_SeqAIJ(B, C, fill, BC));

  PetscCall(PetscStrallocpy(D->product->alg, &alg));
  PetscCall(MatProductSetAlgorithm(D, "sorted")); /* set alg for D = A*BC */
  PetscCall(MatMatMultSymbolic_SeqAIJ_SeqAIJ(A, BC, fill, D));
  PetscCall(MatProductSetAlgorithm(D, alg)); /* resume original algorithm */
  PetscCall(PetscFree(alg));

  /* create struct MatProductCtx_MatMatMatMult and attached it to D */
  PetscCheck(!D->product->data, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not yet coded");
  PetscCall(PetscNew(&matmatmatmult));
  matmatmatmult->BC   = BC;
  D->product->data    = matmatmatmult;
  D->product->destroy = MatProductCtxDestroy_SeqAIJ_MatMatMatMult;

  D->ops->matmatmultnumeric = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(Mat A, Mat B, Mat C, Mat D)
{
  MatProductCtx_MatMatMatMult *matmatmatmult;
  Mat                          BC;

  PetscFunctionBegin;
  MatCheckProduct(D, 4);
  PetscCheck(D->product->data, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "Product data empty");
  matmatmatmult = (MatProductCtx_MatMatMatMult *)D->product->data;
  BC            = matmatmatmult->BC;
  PetscCheck(BC, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "Missing BC mat");
  PetscCall((*BC->ops->matmultnumeric)(B, C, BC));
  PetscCall((*D->ops->matmultnumeric)(A, BC, D));
  PetscFunctionReturn(PETSC_SUCCESS);
}
