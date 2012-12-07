/*
  Defines matrix-matrix-matrix product routines for SeqAIJ matrices
          D = A * B * C
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJ_MatMatMatMult"
PetscErrorCode MatDestroy_SeqAIJ_MatMatMatMult(Mat A)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_MatMatMatMult  *matmatmatmult=a->matmatmatmult;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&matmatmatmult->BC);CHKERRQ(ierr);
  ierr = matmatmatmult->destroy(A);CHKERRQ(ierr);
  ierr = PetscFree(matmatmatmult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMult_SeqAIJ_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMatMult_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,MatReuse scall,PetscReal fill,Mat *D)
{
  PetscErrorCode ierr; 

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_MatMatMultSymbolic,A,B,C,0);CHKERRQ(ierr);
    ierr = MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(A,B,C,fill,D);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMatMultSymbolic,A,B,C,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  ierr = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(A,B,C,*D);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat *D)
{
  PetscErrorCode     ierr;
  Mat                BC;
  Mat_MatMatMatMult  *matmatmatmult;
  Mat_SeqAIJ         *d;
  PetscBool          scalable=PETSC_TRUE;
  PetscLogDouble     t0,t1,t2;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)B);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-matmatmatmult_scalable","Use a scalable but slower D=A*B*C","",scalable,&scalable,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (scalable){
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = PetscGetTime(&t1);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(A,BC,fill,D);CHKERRQ(ierr);
    ierr = PetscGetTime(&t2);CHKERRQ(ierr);
    printf("  Mat %d %d, 3MultSymbolic_SeqAIJ_Scalable time: %g + %g = %g\n",A->rmap->N,A->cmap->N,t1-t0,t2-t1,t2-t0);
  } else {
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(B,C,fill,&BC);CHKERRQ(ierr);
    ierr = PetscGetTime(&t1);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,BC,fill,D);CHKERRQ(ierr);
    ierr = PetscGetTime(&t2);CHKERRQ(ierr);
    printf("  Mat %d %d, 3MultSymbolic_SeqAIJ time: %g + %g = %g\n",A->rmap->N,A->cmap->N,t1-t0,t2-t1,t2-t0);
  }

  /* create struct Mat_MatMatMatMult and attached it to *D */
  ierr = PetscNew(Mat_MatMatMatMult,&matmatmatmult);CHKERRQ(ierr);
  matmatmatmult->BC      = BC;
  matmatmatmult->destroy = (*D)->ops->destroy;
  d                      = (Mat_SeqAIJ*)(*D)->data;
  d->matmatmatmult       = matmatmatmult;

  (*D)->ops->matmatmultnumeric = MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ;
  (*D)->ops->destroy           = MatDestroy_SeqAIJ_MatMatMatMult;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJ        *d=(Mat_SeqAIJ*)D->data;
  Mat_MatMatMatMult *matmatmatmult=d->matmatmatmult;
  Mat               BC= matmatmatmult->BC;
  PetscLogDouble    t0,t1,t2;
  
  PetscFunctionBegin;
  ierr = PetscGetTime(&t0);CHKERRQ(ierr);
  ierr = (BC->ops->matmultnumeric)(B,C,BC);CHKERRQ(ierr); 
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  ierr = (D->ops->matmultnumeric)(A,BC,D);CHKERRQ(ierr);
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  printf("  3MultNumeric_SeqAIJ time: %g + %g = %g\n",t1-t0,t2-t1,t2-t0);
  PetscFunctionReturn(0);
}
