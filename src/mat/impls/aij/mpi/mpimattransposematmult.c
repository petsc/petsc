
/*
  Defines matrix-matrix product routines for pairs of MPIAIJ matrices
          C = A^T * B
  The routines are slightly modified from MatTransposeMatMultxxx_SeqAIJ_SeqDense().
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/dense/mpi/mpidense.h>

PetscErrorCode MatDestroy_MPIDense_MatTransMatMult(Mat A)
{
  PetscErrorCode      ierr;
  Mat_MPIDense        *a = (Mat_MPIDense*)A->data;
  Mat_MatTransMatMult *atb = a->atb;

  PetscFunctionBegin;
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->bt);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->ct);CHKERRQ(ierr);
  ierr = (atb->destroy)(A);CHKERRQ(ierr);
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  PetscInt            m=A->rmap->n,n=A->cmap->n,BN=B->cmap->N;
  Mat_MatTransMatMult *atb;
  Vec                 bt,ct;
  Mat_MPIDense        *c;

  PetscFunctionBegin;
  ierr = PetscNew(&atb);CHKERRQ(ierr);

  /* create output dense matrix C = A^T*B */
  ierr = MatSetSizes(C,n,PETSC_DECIDE,PETSC_DECIDE,BN);CHKERRQ(ierr);
  ierr = MatSetType(C,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(C,NULL);CHKERRQ(ierr);

  /* create vectors bt and ct to hold locally transposed arrays of B and C */
  ierr = VecCreate(PetscObjectComm((PetscObject)A),&bt);CHKERRQ(ierr);
  ierr = VecSetSizes(bt,m*BN,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(bt,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)A),&ct);CHKERRQ(ierr);
  ierr = VecSetSizes(ct,n*BN,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(ct,VECSTANDARD);CHKERRQ(ierr);
  atb->bt = bt;
  atb->ct = ct;

  c                               = (Mat_MPIDense*)C->data;
  c->atb                          = atb;
  atb->destroy                    = C->ops->destroy;
  C->ops->destroy                 = MatDestroy_MPIDense_MatTransMatMult;
  C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIDense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode      ierr;
  const PetscScalar   *Barray,*ctarray;
  PetscScalar         *Carray,*btarray;
  Mat_MPIDense        *b=(Mat_MPIDense*)B->data,*c=(Mat_MPIDense*)C->data;
  Mat_SeqDense        *bseq=(Mat_SeqDense*)(b->A)->data,*cseq=(Mat_SeqDense*)(c->A)->data;
  PetscInt            i,j,m=A->rmap->n,n=A->cmap->n,ldb=bseq->lda,BN=B->cmap->N,ldc=cseq->lda;
  Mat_MatTransMatMult *atb=c->atb;
  Vec                 bt=atb->bt,ct=atb->ct;

  PetscFunctionBegin;
  /* create MAIJ matrix mA from A -- should be done in symbolic phase */
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(A,BN,&atb->mA);CHKERRQ(ierr);

  /* transpose local arry of B, then copy it to vector bt */
  ierr = MatDenseGetArrayRead(B,&Barray);CHKERRQ(ierr);
  ierr = VecGetArray(bt,&btarray);CHKERRQ(ierr);

  for (j=0; j<BN; j++) {
    for (i=0; i<m; i++) btarray[i*BN + j] = Barray[j*ldb + i];
  }
  ierr = VecRestoreArray(bt,&btarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&Barray);CHKERRQ(ierr);

  /* compute ct = mA^T * cb */
  ierr = MatMultTranspose(atb->mA,bt,ct);CHKERRQ(ierr);

  /* transpose local array of ct to matrix C */
  ierr = MatDenseGetArray(C,&Carray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ct,&ctarray);CHKERRQ(ierr);

  for (j=0; j<BN; j++) {
    for (i=0; i<n; i++) Carray[j*ldc + i] = ctarray[i*BN + j];
  }
  ierr = VecRestoreArrayRead(ct,&ctarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(C,&Carray);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
