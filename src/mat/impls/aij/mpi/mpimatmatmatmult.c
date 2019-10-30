/*
  Defines matrix-matrix-matrix product routines for MPIAIJ matrices
          D = A * B * C
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatTransposeMatMatMultSymbolic_AIJ_AIJ_AIJ_wHYPRE(Mat,Mat,Mat,PetscReal,Mat*);
PETSC_INTERN PetscErrorCode MatTransposeMatMatMultNumeric_AIJ_AIJ_AIJ_wHYPRE(Mat,Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatMatMatMult_Transpose_AIJ_AIJ(Mat R,Mat A,Mat P,MatReuse scall, PetscReal fill, Mat *RAP)
{
  Mat            Rt;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatTransposeGetMat(R,&Rt);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)Rt,&flg,MATSEQAIJ,MATMPIAIJ,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)Rt),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)Rt)->type_name);
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatMatMultSymbolic,R,A,P,0);CHKERRQ(ierr);
    ierr = MatTransposeMatMatMultSymbolic_AIJ_AIJ_AIJ_wHYPRE(Rt,A,P,fill,RAP);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMatMultSymbolic,R,A,P,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMatMultNumeric,R,A,P,0);CHKERRQ(ierr);
  ierr = MatTransposeMatMatMultNumeric_AIJ_AIJ_AIJ_wHYPRE(Rt,A,P,*RAP);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMatMultNumeric,R,A,P,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatFreeIntermediateDataStructures_MPIAIJ_BC(Mat ABC)
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ*)ABC->data;
  Mat_MatMatMatMult *matmatmatmult = a->matmatmatmult;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!matmatmatmult) PetscFunctionReturn(0);

  ierr = MatDestroy(&matmatmatmult->BC);CHKERRQ(ierr);
  ABC->ops->destroy = matmatmatmult->destroy;
  ierr = PetscFree(a->matmatmatmult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJ_MatMatMatMult(Mat A)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = (*A->ops->freeintermediatedatastructures)(A);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  ierr = ((*D)->ops->matmatmultnumeric)(A,B,C,*D);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat *D)
{
  PetscErrorCode    ierr;
  Mat               BC;
  Mat_MatMatMatMult *matmatmatmult;
  Mat_MPIAIJ        *d;
  PetscBool         flg;
  const char        *algTypes[2] = {"scalable","nonscalable"};
  PetscInt          nalg=2,alg=0; /* set default algorithm */

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)B),((PetscObject)B)->prefix,"MatMatMatMult","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-matmatmatmult_via","Algorithmic approach","MatMatMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  switch (alg) {
  case 0: /* scalable */
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = MatZeroEntries(BC);CHKERRQ(ierr); /* initialize value entries of BC */
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,BC,fill,D);CHKERRQ(ierr);
    break;
  case 1:
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = MatZeroEntries(BC);CHKERRQ(ierr); /* initialize value entries of BC */
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(A,BC,fill,D);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Not supported");
  }

  /* create struct Mat_MatMatMatMult and attached it to *D */
  ierr = PetscNew(&matmatmatmult);CHKERRQ(ierr);

  matmatmatmult->BC      = BC;
  matmatmatmult->destroy = (*D)->ops->destroy;
  d                      = (Mat_MPIAIJ*)(*D)->data;
  d->matmatmatmult       = matmatmatmult;

  (*D)->ops->matmatmultnumeric = MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ;
  (*D)->ops->destroy           = MatDestroy_MPIAIJ_MatMatMatMult;
  (*D)->ops->freeintermediatedatastructures = MatFreeIntermediateDataStructures_MPIAIJ_BC;
  PetscFunctionReturn(0);
}

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

PetscErrorCode MatRARt_MPIAIJ_MPIAIJ(Mat A,Mat R,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  Mat            Rt;

  PetscFunctionBegin;
  ierr = MatTranspose(R,MAT_INITIAL_MATRIX,&Rt);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(R,A,Rt,fill,C);CHKERRQ(ierr);
  }
  ierr = MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ(R,A,Rt,*C);CHKERRQ(ierr);
  ierr = MatDestroy(&Rt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
