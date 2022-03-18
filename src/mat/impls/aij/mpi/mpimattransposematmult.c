
/*
  Defines matrix-matrix product routines for pairs of MPIAIJ matrices
          C = A^T * B
  The routines are slightly modified from MatTransposeMatMultxxx_SeqAIJ_SeqDense().
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/dense/mpi/mpidense.h>

PetscErrorCode MatDestroy_MPIDense_MatTransMatMult(void *data)
{
  PetscErrorCode      ierr;
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult*)data;

  PetscFunctionBegin;
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->bt);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->ct);CHKERRQ(ierr);
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat_MatTransMatMult *atb;
  PetscBool           cisdense;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheckFalse(C->product->data,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");

  /* create output dense matrix C = A^T*B */
  ierr = MatSetSizes(C,A->cmap->n,B->cmap->n,A->cmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATMPIDENSE,MATMPIDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    ierr = MatSetType(C,((PetscObject)B)->type_name);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* create additional data structure for the product */
  ierr = PetscNew(&atb);CHKERRQ(ierr);
  if (B->cmap->N) {
    ierr = MatCreateMAIJ(A,B->cmap->N,&atb->mA);CHKERRQ(ierr);
    if (!atb->mA->assembled) {
      ierr = MatAssemblyBegin(atb->mA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(atb->mA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(atb->mA,&atb->ct,&atb->bt);CHKERRQ(ierr);
  }
  C->product->data    = atb;
  C->product->destroy = MatDestroy_MPIDense_MatTransMatMult;

  C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIDense;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode      ierr;
  const PetscScalar   *Barray,*ctarray;
  PetscScalar         *Carray,*btarray;
  PetscInt            i,j,m=A->rmap->n,n=A->cmap->n,ldb,BN=B->cmap->N,ldc;
  Mat_MatTransMatMult *atb;
  Vec                 bt,ct;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  atb = (Mat_MatTransMatMult *)C->product->data;
  PetscCheckFalse(!atb,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  if (!BN) {
    ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  bt = atb->bt;
  ct = atb->ct;

  /* transpose local array of B, then copy it to vector bt */
  ierr = MatDenseGetArrayRead(B,&Barray);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(B,&ldb);CHKERRQ(ierr);
  ierr = VecGetArray(bt,&btarray);CHKERRQ(ierr);
  for (j=0; j<BN; j++)
    for (i=0; i<m; i++)
      btarray[i*BN + j] = Barray[j*ldb + i];
  ierr = VecRestoreArray(bt,&btarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&Barray);CHKERRQ(ierr);

  /* compute ct = mA^T * cb */
  ierr = MatMultTranspose(atb->mA,bt,ct);CHKERRQ(ierr);

  /* transpose local array of ct to matrix C */
  ierr = MatDenseGetArray(C,&Carray);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&ldc);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ct,&ctarray);CHKERRQ(ierr);
  for (j=0; j<BN; j++)
    for (i=0; i<n; i++)
      Carray[j*ldc + i] = ctarray[i*BN + j];
  ierr = VecRestoreArrayRead(ct,&ctarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(C,&Carray);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
