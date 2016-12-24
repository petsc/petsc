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

PETSC_INTERN PetscErrorCode MatMatMatMult_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,MatReuse scall,PetscReal fill,Mat *D)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatMatMultSymbolic,A,B,C,0);CHKERRQ(ierr);
    ierr = MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(A,B,C,fill,D);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMatMultSymbolic,A,B,C,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  ierr = ((*D)->ops->matmatmultnumeric)(A,B,C,*D);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat *D)
{
  PetscErrorCode    ierr;
  Mat               BC;
  Mat_MatMatMatMult *matmatmatmult;
  Mat_SeqAIJ        *d;
  PetscBool         scalable=PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)B);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-matmatmatmult_scalable","Use a scalable but slower D=A*B*C","",scalable,&scalable,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (scalable) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(A,BC,fill,D);CHKERRQ(ierr);
  } else {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,BC,fill,D);CHKERRQ(ierr);
  }

  /* create struct Mat_MatMatMatMult and attached it to *D */
  ierr = PetscNew(&matmatmatmult);CHKERRQ(ierr);

  matmatmatmult->BC      = BC;
  matmatmatmult->destroy = (*D)->ops->destroy;
  d                      = (Mat_SeqAIJ*)(*D)->data;
  d->matmatmatmult       = matmatmatmult;

  (*D)->ops->matmatmultnumeric = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ;
  (*D)->ops->destroy           = MatDestroy_SeqAIJ_MatMatMatMult;
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
