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
  Mat_Product    *product = RAP->product;
  Mat            Rt,R=product->A,A=product->B,P=product->C;

  PetscFunctionBegin;
  CHKERRQ(MatTransposeGetMat(R,&Rt));
  CHKERRQ(MatTransposeMatMatMultNumeric_AIJ_AIJ_AIJ_wHYPRE(Rt,A,P,RAP));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_ABC_Transpose_AIJ_AIJ(Mat RAP)
{
  Mat_Product    *product = RAP->product;
  Mat            Rt,R=product->A,A=product->B,P=product->C;
  PetscBool      flg;

  PetscFunctionBegin;
  /* local sizes of matrices will be checked by the calling subroutines */
  CHKERRQ(MatTransposeGetMat(R,&Rt));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)Rt,&flg,MATSEQAIJ,MATSEQAIJMKL,MATMPIAIJ,NULL));
  PetscCheck(flg,PetscObjectComm((PetscObject)Rt),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)Rt)->type_name);
  CHKERRQ(MatTransposeMatMatMultSymbolic_AIJ_AIJ_AIJ_wHYPRE(Rt,A,P,product->fill,RAP));
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
  Mat            BC;
  PetscBool      scalable;
  Mat_Product    *product;

  PetscFunctionBegin;
  MatCheckProduct(D,4);
  PetscCheck(!D->product->data,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Product data not empty");
  product = D->product;
  CHKERRQ(MatProductCreate(B,C,NULL,&BC));
  CHKERRQ(MatProductSetType(BC,MATPRODUCT_AB));
  CHKERRQ(PetscStrcmp(product->alg,"scalable",&scalable));
  if (scalable) {
    CHKERRQ(MatMatMultSymbolic_MPIAIJ_MPIAIJ(B,C,fill,BC));
    CHKERRQ(MatZeroEntries(BC)); /* initialize value entries of BC */
    CHKERRQ(MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,BC,fill,D));
  } else {
    CHKERRQ(MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(B,C,fill,BC));
    CHKERRQ(MatZeroEntries(BC)); /* initialize value entries of BC */
    CHKERRQ(MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(A,BC,fill,D));
  }
  CHKERRQ(MatDestroy(&product->Dwork));
  product->Dwork = BC;

  D->ops->matmatmultnumeric = MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,Mat D)
{
  Mat_Product    *product;
  Mat            BC;

  PetscFunctionBegin;
  MatCheckProduct(D,4);
  PetscCheck(D->product->data,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Product data empty");
  product = D->product;
  BC = product->Dwork;
  PetscCheck(BC->ops->matmultnumeric,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Missing numeric operation");
  CHKERRQ((*BC->ops->matmultnumeric)(B,C,BC));
  PetscCheck(D->ops->matmultnumeric,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Missing numeric operation");
  CHKERRQ((*D->ops->matmultnumeric)(A,BC,D));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------- */
PetscErrorCode MatDestroy_MPIAIJ_RARt(void *data)
{
  Mat_RARt       *rart = (Mat_RARt*)data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&rart->Rt));
  if (rart->destroy) {
    CHKERRQ((*rart->destroy)(rart->data));
  }
  CHKERRQ(PetscFree(rart));
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_RARt_MPIAIJ_MPIAIJ(Mat C)
{
  Mat_RARt       *rart;
  Mat            A,R,Rt;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  rart = (Mat_RARt*)C->product->data;
  A    = C->product->A;
  R    = C->product->B;
  Rt   = rart->Rt;
  CHKERRQ(MatTranspose(R,MAT_REUSE_MATRIX,&Rt));
  if (rart->data) C->product->data = rart->data;
  CHKERRQ((*C->ops->matmatmultnumeric)(R,A,Rt,C));
  C->product->data = rart;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_RARt_MPIAIJ_MPIAIJ(Mat C)
{
  Mat            A,R,Rt;
  Mat_RARt       *rart;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A    = C->product->A;
  R    = C->product->B;
  CHKERRQ(MatTranspose(R,MAT_INITIAL_MATRIX,&Rt));
  /* product->Dwork is used to store A*Rt in MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ() */
  CHKERRQ(MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(R,A,Rt,C->product->fill,C));
  C->ops->productnumeric = MatProductNumeric_RARt_MPIAIJ_MPIAIJ;

  /* create a supporting struct */
  CHKERRQ(PetscNew(&rart));
  rart->Rt      = Rt;
  rart->data    = C->product->data;
  rart->destroy = C->product->destroy;
  C->product->data    = rart;
  C->product->destroy = MatDestroy_MPIAIJ_RARt;
  PetscFunctionReturn(0);
}
