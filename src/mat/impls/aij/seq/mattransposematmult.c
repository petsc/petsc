
/*
  Defines matrix-matrix product routines for
          C = A^T * B and C = A * B^t
  with A SeqAIJ and B SeqDense
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

PetscErrorCode MatDestroy_SeqDense_MatTransMatMult(void *data)
{
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult *)data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&atb->mA));
  CHKERRQ(VecDestroy(&atb->bt));
  CHKERRQ(VecDestroy(&atb->ct));
  CHKERRQ(PetscFree(atb));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTMatTMultNumeric_SeqAIJ_SeqDense(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatTMatTMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  Mat_MatTransMatMult *atb;
  PetscBool           cisdense;
  PetscInt            dofm;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheckFalse(C->product->data,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");
  PetscCheckFalse(C->product->type != MATPRODUCT_ABt && C->product->type != MATPRODUCT_AtB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not for product type %s",MatProductTypes[C->product->type]);

  /* create output dense matrix C */
  if (C->product->type == MATPRODUCT_AtB) {
    CHKERRQ(MatSetSizes(C,A->cmap->n,B->cmap->N,A->cmap->n,B->cmap->N));
    dofm = B->cmap->n;
  } else {
    CHKERRQ(MatSetSizes(C,A->rmap->n,B->rmap->N,A->rmap->n,B->rmap->N));
    dofm = B->rmap->n;
  }
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,""));
  if (!cisdense) {
    CHKERRQ(MatSetType(C,((PetscObject)B)->type_name));
  }
  CHKERRQ(MatSetUp(C));

  /* create additional data structure for the product */
  CHKERRQ(PetscNew(&atb));
  CHKERRQ(MatCreateMAIJ(A,dofm,&atb->mA));
  CHKERRQ(MatCreateVecs(atb->mA,&atb->ct,&atb->bt));
  C->product->data    = atb;
  C->product->destroy = MatDestroy_SeqDense_MatTransMatMult;

  if (C->product->type == MATPRODUCT_AtB) {
    C->ops->transposematmultnumeric = MatTMatTMultNumeric_SeqAIJ_SeqDense;
  } else {
    C->ops->mattransposemultnumeric = MatTMatTMultNumeric_SeqAIJ_SeqDense;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatTMatTMultNumeric_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  PetscInt            i,j,m=A->rmap->n,n=A->cmap->n,blda,clda;
  PetscInt            mdof = C->cmap->N;
  const PetscScalar   *Barray;
  PetscScalar         *Carray;
  Mat_MatTransMatMult *atb;
  Vec                 bt,ct;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheckFalse(C->product->type != MATPRODUCT_ABt && C->product->type != MATPRODUCT_AtB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not for product type %s",MatProductTypes[C->product->type]);
  atb = (Mat_MatTransMatMult *)C->product->data;
  PetscCheckFalse(!atb,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  bt = atb->bt;
  ct = atb->ct;

  CHKERRQ(MatDenseGetArrayRead(B,&Barray));
  CHKERRQ(MatDenseGetLDA(B,&blda));
  CHKERRQ(MatDenseGetArrayWrite(C,&Carray));
  CHKERRQ(MatDenseGetLDA(C,&clda));
  if (C->product->type == MATPRODUCT_AtB) { /* transpose local array of B, then copy it to vector bt */
    const PetscScalar *ctarray;
    PetscScalar       *btarray;

    CHKERRQ(VecGetArrayWrite(bt,&btarray));
    for (j=0; j<mdof; j++) {
      for (i=0; i<m; i++) btarray[i*mdof + j] = Barray[j*blda + i];
    }
    CHKERRQ(VecRestoreArrayWrite(bt,&btarray));

    /* compute ct = mA^T * cb */
    CHKERRQ(MatMultTranspose(atb->mA,bt,ct));

    /* transpose local array of ct to matrix C */
    CHKERRQ(VecGetArrayRead(ct,&ctarray));
    for (j=0; j<mdof; j++) {
      for (i=0; i<n; i++) Carray[j*clda + i] = ctarray[i*mdof + j];
    }
    CHKERRQ(VecRestoreArrayRead(ct,&ctarray));
  } else {
    const PetscScalar *btarray;
    PetscScalar       *ctarray;

    if (blda == B->rmap->n) {
      CHKERRQ(VecPlaceArray(ct,Barray));
    } else {
      PetscInt bn = B->cmap->n;
      PetscInt bm = B->rmap->n;

      CHKERRQ(VecGetArrayWrite(ct,&ctarray));
      for (j=0; j<bn; j++) {
        for (i=0; i<bm; i++) ctarray[j*bm + i] = Barray[j*blda + i];
      }
      CHKERRQ(VecRestoreArrayWrite(ct,&ctarray));
    }

    CHKERRQ(MatMult(atb->mA,ct,bt));
    if (blda == B->rmap->n) {
      CHKERRQ(VecResetArray(ct));
    }
    CHKERRQ(VecGetArrayRead(bt,&btarray));
    for (j=0; j<mdof; j++) {
      for (i=0; i<m; i++) Carray[j*clda + i] = btarray[i*mdof + j];
    }
    CHKERRQ(VecRestoreArrayRead(bt,&btarray));
  }
  CHKERRQ(MatDenseRestoreArrayRead(B,&Barray));
  CHKERRQ(MatDenseRestoreArray(C,&Carray));
  PetscFunctionReturn(0);
}
