/*
  Defines matrix-matrix-matrix product routines for MPIAIJ matrices
          D = A * B * C
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIAIJ_MatMatMatMult"
PetscErrorCode MatDestroy_MPIAIJ_MatMatMatMult(Mat A)
{
  Mat_MPIAIJ        *a            = (Mat_MPIAIJ*)A->data;
  Mat_MatMatMatMult *matmatmatmult=a->matmatmatmult;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&matmatmatmult->BC);CHKERRQ(ierr);
  ierr = matmatmatmult->destroy(A);CHKERRQ(ierr);
  ierr = PetscFree(matmatmatmult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMult_MPIAIJ_MPIAIJ_MPIAIJ"
PETSC_INTERN PetscErrorCode MatMatMatMult_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,MatReuse scall,PetscReal fill,Mat *D)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatMatMultSymbolic,A,B,C,0);CHKERRQ(ierr);
    ierr = MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(A,B,C,fill,D);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMatMultSymbolic,A,B,C,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  ierr = MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ(A,B,C,*D);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat *D)
{
  PetscErrorCode    ierr;
  Mat               BC;
  Mat_MatMatMatMult *matmatmatmult;
  Mat_MPIAIJ        *d;
  PetscBool         scalable=PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)B);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-matmatmatmult_scalable","Use a scalable but slower D=A*B*C","",scalable,&scalable,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (scalable) {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,BC,fill,D);CHKERRQ(ierr);
  } else {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(A,BC,fill,D);CHKERRQ(ierr);
  }

  /* create struct Mat_MatMatMatMult and attached it to *D */
  ierr = PetscNew(&matmatmatmult);CHKERRQ(ierr);

  matmatmatmult->BC      = BC;
  matmatmatmult->destroy = (*D)->ops->destroy;
  d                      = (Mat_MPIAIJ*)(*D)->data;
  d->matmatmatmult       = matmatmatmult;

  (*D)->ops->matmatmultnumeric = MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ;
  (*D)->ops->destroy           = MatDestroy_MPIAIJ_MatMatMatMult;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *d             = (Mat_MPIAIJ*)D->data;
  Mat_MatMatMatMult *matmatmatmult = d->matmatmatmult;
  Mat               BC             = matmatmatmult->BC;

  PetscFunctionBegin;
  ierr = (BC->ops->matmultnumeric)(B,C,BC);CHKERRQ(ierr);
  ierr = (D->ops->matmultnumeric)(A,BC,D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
