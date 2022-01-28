/*
  Defines matrix-matrix-matrix product routines for SeqAIJ matrices
          D = A * B * C
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/

PetscErrorCode MatDestroy_SeqAIJ_MatMatMatMult(void* data)
{
  Mat_MatMatMatMult *matmatmatmult = (Mat_MatMatMatMult*)data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&matmatmatmult->BC);CHKERRQ(ierr);
  ierr = PetscFree(matmatmatmult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat D)
{
  PetscErrorCode    ierr;
  Mat               BC;
  Mat_MatMatMatMult *matmatmatmult;
  char              *alg;

  PetscFunctionBegin;
  MatCheckProduct(D,5);
  PetscAssertFalse(D->product->data,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Product data not empty");
  ierr = MatCreate(PETSC_COMM_SELF,&BC);CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(B,C,fill,BC);CHKERRQ(ierr);

  ierr = PetscStrallocpy(D->product->alg,&alg);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(D,"sorted");CHKERRQ(ierr); /* set alg for D = A*BC */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,BC,fill,D);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(D,alg);CHKERRQ(ierr); /* resume original algorithm */
  ierr = PetscFree(alg);CHKERRQ(ierr);

  /* create struct Mat_MatMatMatMult and attached it to D */
  PetscAssertFalse(D->product->data,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not yet coded");
  ierr = PetscNew(&matmatmatmult);CHKERRQ(ierr);
  matmatmatmult->BC   = BC;
  D->product->data    = matmatmatmult;
  D->product->destroy = MatDestroy_SeqAIJ_MatMatMatMult;

  D->ops->matmatmultnumeric = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode    ierr;
  Mat_MatMatMatMult *matmatmatmult;
  Mat               BC;

  PetscFunctionBegin;
  MatCheckProduct(D,4);
  PetscAssertFalse(!D->product->data,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Product data empty");
  matmatmatmult = (Mat_MatMatMatMult*)D->product->data;
  BC = matmatmatmult->BC;
  PetscAssertFalse(!BC,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Missing BC mat");
  PetscAssertFalse(!BC->ops->matmultnumeric,PetscObjectComm((PetscObject)BC),PETSC_ERR_PLIB,"Missing numeric operation");
  ierr = (*BC->ops->matmultnumeric)(B,C,BC);CHKERRQ(ierr);
  PetscAssertFalse(!D->ops->matmultnumeric,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Missing numeric operation");
  ierr = (*D->ops->matmultnumeric)(A,BC,D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
