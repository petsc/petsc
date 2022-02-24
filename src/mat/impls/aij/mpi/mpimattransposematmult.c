
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
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult*)data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&atb->mA));
  CHKERRQ(VecDestroy(&atb->bt));
  CHKERRQ(VecDestroy(&atb->ct));
  CHKERRQ(PetscFree(atb));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  Mat_MatTransMatMult *atb;
  PetscBool           cisdense;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheckFalse(C->product->data,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");

  /* create output dense matrix C = A^T*B */
  CHKERRQ(MatSetSizes(C,A->cmap->n,B->cmap->n,A->cmap->N,B->cmap->N));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATMPIDENSE,MATMPIDENSECUDA,""));
  if (!cisdense) {
    CHKERRQ(MatSetType(C,((PetscObject)B)->type_name));
  }
  CHKERRQ(MatSetUp(C));

  /* create additional data structure for the product */
  CHKERRQ(PetscNew(&atb));
  if (B->cmap->N) {
    CHKERRQ(MatCreateMAIJ(A,B->cmap->N,&atb->mA));
    if (!atb->mA->assembled) {
      CHKERRQ(MatAssemblyBegin(atb->mA,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(atb->mA,MAT_FINAL_ASSEMBLY));
    }
    CHKERRQ(MatCreateVecs(atb->mA,&atb->ct,&atb->bt));
  }
  C->product->data    = atb;
  C->product->destroy = MatDestroy_MPIDense_MatTransMatMult;

  C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIDense;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat A,Mat B,Mat C)
{
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
    CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
  }
  bt = atb->bt;
  ct = atb->ct;

  /* transpose local array of B, then copy it to vector bt */
  CHKERRQ(MatDenseGetArrayRead(B,&Barray));
  CHKERRQ(MatDenseGetLDA(B,&ldb));
  CHKERRQ(VecGetArray(bt,&btarray));
  for (j=0; j<BN; j++)
    for (i=0; i<m; i++)
      btarray[i*BN + j] = Barray[j*ldb + i];
  CHKERRQ(VecRestoreArray(bt,&btarray));
  CHKERRQ(MatDenseRestoreArrayRead(B,&Barray));

  /* compute ct = mA^T * cb */
  CHKERRQ(MatMultTranspose(atb->mA,bt,ct));

  /* transpose local array of ct to matrix C */
  CHKERRQ(MatDenseGetArray(C,&Carray));
  CHKERRQ(MatDenseGetLDA(C,&ldc));
  CHKERRQ(VecGetArrayRead(ct,&ctarray));
  for (j=0; j<BN; j++)
    for (i=0; i<n; i++)
      Carray[j*ldc + i] = ctarray[i*BN + j];
  CHKERRQ(VecRestoreArrayRead(ct,&ctarray));
  CHKERRQ(MatDenseRestoreArray(C,&Carray));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
