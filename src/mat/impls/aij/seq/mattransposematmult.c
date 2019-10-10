
/*
  Defines matrix-matrix product routines 
          C = A^T * B
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

PetscErrorCode MatDestroy_SeqDense_MatTransMatMult(Mat A)
{
  PetscErrorCode      ierr;
  Mat_SeqDense        *a = (Mat_SeqDense*)A->data;
  Mat_MatTransMatMult *atb = a->atb;

  PetscFunctionBegin;
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->bt);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->ct);CHKERRQ(ierr);
  ierr = (atb->destroy)(A);CHKERRQ(ierr);
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMult_SeqAIJ_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_TransposeMatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatTransposeMatMultSymbolic_SeqAIJ_SeqDense(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_TransposeMatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_TransposeMatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatTransposeMatMultNumeric_SeqAIJ_SeqDense(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_TransposeMatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  PetscInt            m=A->rmap->n,n=A->cmap->n,BN=B->cmap->N;
  Mat_MatTransMatMult *atb;
  Mat                 Cdense;
  Vec                 bt,ct;
  Mat_SeqDense        *c;

  PetscFunctionBegin;
  ierr = PetscNew(&atb);CHKERRQ(ierr);

  /* create output dense matrix C = A^T*B */
  ierr = MatCreate(PETSC_COMM_SELF,&Cdense);CHKERRQ(ierr);
  ierr = MatSetSizes(Cdense,n,BN,n,BN);CHKERRQ(ierr);
  ierr = MatSetType(Cdense,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Cdense,NULL);CHKERRQ(ierr);

  /* create vectors bt and ct to hold locally transposed arrays of B and C */
  ierr = VecCreate(PETSC_COMM_SELF,&bt);CHKERRQ(ierr);
  ierr = VecSetSizes(bt,m*BN,m*BN);CHKERRQ(ierr);
  ierr = VecSetType(bt,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&ct);CHKERRQ(ierr);
  ierr = VecSetSizes(ct,n*BN,n*BN);CHKERRQ(ierr);
  ierr = VecSetType(ct,VECSTANDARD);CHKERRQ(ierr);
  atb->bt = bt;
  atb->ct = ct;

  *C                                   = Cdense;
  c                                    = (Mat_SeqDense*)Cdense->data;
  c->atb                               = atb;
  atb->destroy                         = Cdense->ops->destroy;
  Cdense->ops->destroy                 = MatDestroy_SeqDense_MatTransMatMult;
  Cdense->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqAIJ_SeqDense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode      ierr;
  PetscInt            i,j,k,m=A->rmap->n,n=A->cmap->n,BN=B->cmap->N;
  const PetscScalar   *Barray,*ctarray;
  PetscScalar         *Carray,*btarray;
  Mat_SeqDense        *c=(Mat_SeqDense*)C->data;
  Mat_MatTransMatMult *atb=c->atb;
  Vec                 bt=atb->bt,ct=atb->ct;

  PetscFunctionBegin;
  /* create MAIJ matrix mA from A -- should be done in symbolic phase */
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(A,BN,&atb->mA);CHKERRQ(ierr);

  /* transpose local arry of B, then copy it to vector bt */
  ierr = MatDenseGetArrayRead(B,&Barray);CHKERRQ(ierr);
  ierr = VecGetArray(bt,&btarray);CHKERRQ(ierr);

  k=0;
  for (j=0; j<BN; j++) {
    for (i=0; i<m; i++) btarray[i*BN + j] = Barray[k++]; 
  }
  ierr = VecRestoreArray(bt,&btarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&Barray);CHKERRQ(ierr);
  
  /* compute ct = mA^T * cb */
  ierr = MatMultTranspose(atb->mA,bt,ct);CHKERRQ(ierr);

  /* transpose local arry of ct to matrix C */
  ierr = MatDenseGetArray(C,&Carray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ct,&ctarray);CHKERRQ(ierr);
  k = 0;
  for (j=0; j<BN; j++) {
    for (i=0; i<n; i++) Carray[k++] = ctarray[i*BN + j];
  }
  ierr = VecRestoreArrayRead(ct,&ctarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(C,&Carray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
