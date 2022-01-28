
/*
  Defines matrix-matrix product routines for
          C = A^T * B and C = A * B^t
  with A SeqAIJ and B SeqDense
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

PetscErrorCode MatDestroy_SeqDense_MatTransMatMult(void *data)
{
  PetscErrorCode      ierr;
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult *)data;

  PetscFunctionBegin;
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->bt);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->ct);CHKERRQ(ierr);
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTMatTMultNumeric_SeqAIJ_SeqDense(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatTMatTMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat_MatTransMatMult *atb;
  PetscBool           cisdense;
  PetscInt            dofm;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");
  if (C->product->type != MATPRODUCT_ABt && C->product->type != MATPRODUCT_AtB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not for product type %s",MatProductTypes[C->product->type]);

  /* create output dense matrix C */
  if (C->product->type == MATPRODUCT_AtB) {
    ierr = MatSetSizes(C,A->cmap->n,B->cmap->N,A->cmap->n,B->cmap->N);CHKERRQ(ierr);
    dofm = B->cmap->n;
  } else {
    ierr = MatSetSizes(C,A->rmap->n,B->rmap->N,A->rmap->n,B->rmap->N);CHKERRQ(ierr);
    dofm = B->rmap->n;
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    ierr = MatSetType(C,((PetscObject)B)->type_name);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* create additional data structure for the product */
  ierr = PetscNew(&atb);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(A,dofm,&atb->mA);CHKERRQ(ierr);
  ierr = MatCreateVecs(atb->mA,&atb->ct,&atb->bt);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;
  PetscInt            i,j,m=A->rmap->n,n=A->cmap->n,blda,clda;
  PetscInt            mdof = C->cmap->N;
  const PetscScalar   *Barray;
  PetscScalar         *Carray;
  Mat_MatTransMatMult *atb;
  Vec                 bt,ct;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (C->product->type != MATPRODUCT_ABt && C->product->type != MATPRODUCT_AtB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not for product type %s",MatProductTypes[C->product->type]);
  atb = (Mat_MatTransMatMult *)C->product->data;
  if (!atb) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  bt = atb->bt;
  ct = atb->ct;

  ierr = MatDenseGetArrayRead(B,&Barray);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&Carray);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  if (C->product->type == MATPRODUCT_AtB) { /* transpose local array of B, then copy it to vector bt */
    const PetscScalar *ctarray;
    PetscScalar       *btarray;

    ierr = VecGetArrayWrite(bt,&btarray);CHKERRQ(ierr);
    for (j=0; j<mdof; j++) {
      for (i=0; i<m; i++) btarray[i*mdof + j] = Barray[j*blda + i];
    }
    ierr = VecRestoreArrayWrite(bt,&btarray);CHKERRQ(ierr);

    /* compute ct = mA^T * cb */
    ierr = MatMultTranspose(atb->mA,bt,ct);CHKERRQ(ierr);

    /* transpose local array of ct to matrix C */
    ierr = VecGetArrayRead(ct,&ctarray);CHKERRQ(ierr);
    for (j=0; j<mdof; j++) {
      for (i=0; i<n; i++) Carray[j*clda + i] = ctarray[i*mdof + j];
    }
    ierr = VecRestoreArrayRead(ct,&ctarray);CHKERRQ(ierr);
  } else {
    const PetscScalar *btarray;
    PetscScalar       *ctarray;

    if (blda == B->rmap->n) {
      ierr = VecPlaceArray(ct,Barray);CHKERRQ(ierr);
    } else {
      PetscInt bn = B->cmap->n;
      PetscInt bm = B->rmap->n;

      ierr = VecGetArrayWrite(ct,&ctarray);CHKERRQ(ierr);
      for (j=0; j<bn; j++) {
        for (i=0; i<bm; i++) ctarray[j*bm + i] = Barray[j*blda + i];
      }
      ierr = VecRestoreArrayWrite(ct,&ctarray);CHKERRQ(ierr);
    }

    ierr = MatMult(atb->mA,ct,bt);CHKERRQ(ierr);
    if (blda == B->rmap->n) {
      ierr = VecResetArray(ct);CHKERRQ(ierr);
    }
    ierr = VecGetArrayRead(bt,&btarray);CHKERRQ(ierr);
    for (j=0; j<mdof; j++) {
      for (i=0; i<m; i++) Carray[j*clda + i] = btarray[i*mdof + j];
    }
    ierr = VecRestoreArrayRead(bt,&btarray);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArrayRead(B,&Barray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(C,&Carray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
