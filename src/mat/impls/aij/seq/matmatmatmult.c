/*
  Defines matrix-matrix-matrix product routines for SeqAIJ matrices
          D = A * B * C
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/

PetscErrorCode MatDestroy_SeqAIJ_MatMatMatMult(Mat A)
{
  Mat_SeqAIJ        *a            = (Mat_SeqAIJ*)A->data;
  Mat_MatMatMatMult *matmatmatmult=a->matmatmatmult;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&matmatmatmult->BC);CHKERRQ(ierr);
  ierr = matmatmatmult->destroy(A);CHKERRQ(ierr);
  ierr = PetscFree(matmatmatmult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat D)
{
  PetscErrorCode    ierr;
  Mat               BC;
  Mat_MatMatMatMult *matmatmatmult;
  Mat_SeqAIJ        *d;
  Mat_Product       *product = D->product;
  MatProductAlgorithm alg=product->alg;

  PetscFunctionBegin;
  if (!product) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Data struc Mat_Product is not created, call MatProductCreate() first");
  ierr = MatCreate(PETSC_COMM_SELF,&BC);CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(B,C,fill,BC);CHKERRQ(ierr);

  ierr = MatProductSetAlgorithm(D,"sorted");CHKERRQ(ierr); /* set alg for D = A*BC */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,BC,fill,D);CHKERRQ(ierr);
  D->product->alg = alg; /* resume original algorithm for D */

  /* create struct Mat_MatMatMatMult and attached it to D */
  ierr = PetscNew(&matmatmatmult);CHKERRQ(ierr);

  matmatmatmult->BC      = BC;
  matmatmatmult->destroy = D->ops->destroy;
  d                      = (Mat_SeqAIJ*)D->data;
  d->matmatmatmult       = matmatmatmult;

  D->ops->matmatmultnumeric = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ;
  D->ops->destroy           = MatDestroy_SeqAIJ_MatMatMatMult;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJ        *d            =(Mat_SeqAIJ*)D->data;
  Mat_MatMatMatMult *matmatmatmult=d->matmatmatmult;
  Mat               BC            = matmatmatmult->BC;

  PetscFunctionBegin;
  ierr = (BC->ops->matmultnumeric)(B,C,BC);CHKERRQ(ierr);
  ierr = (D->ops->matmultnumeric)(A,BC,D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

