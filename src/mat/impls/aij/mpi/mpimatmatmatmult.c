/*
  Defines matrix-matrix-matrix product routines for MPIAIJ matrices
          D = A * B * C
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatTransposeMatMatMultSymbolic_AIJ_AIJ_AIJ_wHYPRE(Mat,Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMatMultNumeric_AIJ_AIJ_AIJ_wHYPRE(Mat,Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatProductNumeric_ABC_Transpose_AIJ_AIJ(Mat RAP)
{
  PetscErrorCode ierr;
  Mat_Product    *product = RAP->product;
  Mat            Rt,R=product->A,A=product->B,P=product->C;

  PetscFunctionBegin;
  ierr = MatTransposeGetMat(R,&Rt);CHKERRQ(ierr);
  ierr = MatTransposeMatMatMultNumeric_AIJ_AIJ_AIJ_wHYPRE(Rt,A,P,RAP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_ABC_Transpose_AIJ_AIJ(Mat RAP)
{
  PetscErrorCode ierr;
  Mat_Product    *product = RAP->product;
  Mat            Rt,R=product->A,A=product->B,P=product->C;
  PetscBool      flg;

  PetscFunctionBegin;
  /* local sizes of matrices will be checked by the calling subroutines */
  ierr = MatTransposeGetMat(R,&Rt);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)Rt,&flg,MATSEQAIJ,MATSEQAIJMKL,MATMPIAIJ,NULL);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)Rt),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)Rt)->type_name);
  ierr = MatTransposeMatMatMultSymbolic_AIJ_AIJ_AIJ_wHYPRE(Rt,A,P,product->fill,RAP);CHKERRQ(ierr);
  RAP->ops->productnumeric = MatProductNumeric_ABC_Transpose_AIJ_AIJ;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Transpose_AIJ_AIJ(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_ABC) {
    C->ops->productsymbolic = MatProductSymbolic_ABC_Transpose_AIJ_AIJ;
  } else SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProduct type %s is not supported for Transpose, AIJ and AIJ matrices",MatProductTypes[product->type]);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat D)
{
  PetscErrorCode ierr;
  Mat            BC;
  PetscBool      scalable;
  Mat_Product    *product;

  PetscFunctionBegin;
  MatCheckProduct(D,4);
  PetscCheckFalse(D->product->data,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Product data not empty");
  product = D->product;
  ierr = MatProductCreate(B,C,NULL,&BC);CHKERRQ(ierr);
  ierr = MatProductSetType(BC,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = PetscStrcmp(product->alg,"scalable",&scalable);CHKERRQ(ierr);
  if (scalable) {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(B,C,fill,BC);CHKERRQ(ierr);
    ierr = MatZeroEntries(BC);CHKERRQ(ierr); /* initialize value entries of BC */
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,BC,fill,D);CHKERRQ(ierr);
  } else {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(B,C,fill,BC);CHKERRQ(ierr);
    ierr = MatZeroEntries(BC);CHKERRQ(ierr); /* initialize value entries of BC */
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(A,BC,fill,D);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&product->Dwork);CHKERRQ(ierr);
  product->Dwork = BC;

  D->ops->matmatmultnumeric = MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;
  Mat_Product    *product;
  Mat            BC;

  PetscFunctionBegin;
  MatCheckProduct(D,4);
  PetscCheckFalse(!D->product->data,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Product data empty");
  product = D->product;
  BC = product->Dwork;
  PetscCheckFalse(!BC->ops->matmultnumeric,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Missing numeric operation");
  ierr = (*BC->ops->matmultnumeric)(B,C,BC);CHKERRQ(ierr);
  PetscCheckFalse(!D->ops->matmultnumeric,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Missing numeric operation");
  ierr = (*D->ops->matmultnumeric)(A,BC,D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------- */
PetscErrorCode MatDestroy_MPIAIJ_RARt(void *data)
{
  PetscErrorCode ierr;
  Mat_RARt       *rart = (Mat_RARt*)data;

  PetscFunctionBegin;
  ierr = MatDestroy(&rart->Rt);CHKERRQ(ierr);
  if (rart->destroy) {
    ierr = (*rart->destroy)(rart->data);CHKERRQ(ierr);
  }
  ierr = PetscFree(rart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_RARt_MPIAIJ_MPIAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_RARt       *rart;
  Mat            A,R,Rt;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheckFalse(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;
  A    = C->product->A;
  R    = C->product->B;
  Rt   = rart->Rt;
  ierr = MatTranspose(R,MAT_REUSE_MATRIX,&Rt);CHKERRQ(ierr);
  if (rart->data) C->product->data = rart->data;
  ierr = (*C->ops->matmatmultnumeric)(R,A,Rt,C);CHKERRQ(ierr);
  C->product->data = rart;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_RARt_MPIAIJ_MPIAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat            A,R,Rt;
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheckFalse(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A    = C->product->A;
  R    = C->product->B;
  ierr = MatTranspose(R,MAT_INITIAL_MATRIX,&Rt);CHKERRQ(ierr);
  /* product->Dwork is used to store A*Rt in MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ() */
  ierr = MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(R,A,Rt,C->product->fill,C);CHKERRQ(ierr);
  C->ops->productnumeric = MatProductNumeric_RARt_MPIAIJ_MPIAIJ;

  /* create a supporting struct */
  ierr = PetscNew(&rart);CHKERRQ(ierr);
  rart->Rt      = Rt;
  rart->data    = C->product->data;
  rart->destroy = C->product->destroy;
  C->product->data    = rart;
  C->product->destroy = MatDestroy_MPIAIJ_RARt;
  PetscFunctionReturn(0);
}
