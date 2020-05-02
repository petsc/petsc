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
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)Rt),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)Rt)->type_name);
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
  } else SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProduct type %s is not supported for Transpose, AIJ and AIJ matrices",MatProductTypes[product->type]);
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

PetscErrorCode MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat D)
{
  PetscErrorCode ierr;
  Mat            BC;
  PetscBool      scalable;
  Mat_Product    *product = D->product;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&BC);CHKERRQ(ierr);
  if (product) {
    ierr = PetscStrcmp(product->alg,"scalable",&scalable);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_ARG_NULL,"Call MatProductCreate() first");

  if (scalable) {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(B,C,fill,BC);CHKERRQ(ierr);
    ierr = MatZeroEntries(BC);CHKERRQ(ierr); /* initialize value entries of BC */
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,BC,fill,D);CHKERRQ(ierr);
  } else {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(B,C,fill,BC);CHKERRQ(ierr);
    ierr = MatZeroEntries(BC);CHKERRQ(ierr); /* initialize value entries of BC */
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(A,BC,fill,D);CHKERRQ(ierr);
  }
  product->Dwork = BC;

  D->ops->matmatmultnumeric = MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ;
  D->ops->freeintermediatedatastructures = MatFreeIntermediateDataStructures_MPIAIJ_BC;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;
  Mat_Product    *product = D->product;
  Mat            BC = product->Dwork;

  PetscFunctionBegin;
  ierr = (BC->ops->matmultnumeric)(B,C,BC);CHKERRQ(ierr);
  ierr = (D->ops->matmultnumeric)(A,BC,D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------- */
PetscErrorCode MatDestroy_MPIAIJ_RARt(Mat C)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *c    = (Mat_MPIAIJ*)C->data;
  Mat_RARt       *rart = c->rart;

  PetscFunctionBegin;
  ierr = MatDestroy(&rart->Rt);CHKERRQ(ierr);

  C->ops->destroy = rart->destroy;
  if (C->ops->destroy) {
    ierr = (*C->ops->destroy)(C);CHKERRQ(ierr);
  }
  ierr = PetscFree(rart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_RARt_MPIAIJ_MPIAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  Mat_RARt       *rart = c->rart;
  Mat_Product    *product = C->product;
  Mat            A=product->A,R=product->B,Rt=rart->Rt;

  PetscFunctionBegin;
  ierr = MatTranspose(R,MAT_REUSE_MATRIX,&Rt);CHKERRQ(ierr);
  ierr = (C->ops->matmatmultnumeric)(R,A,Rt,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_RARt_MPIAIJ_MPIAIJ(Mat C)
{
  PetscErrorCode      ierr;
  Mat_Product         *product = C->product;
  Mat                 A=product->A,R=product->B,Rt;
  PetscReal           fill=product->fill;
  Mat_RARt            *rart;
  Mat_MPIAIJ          *c;

  PetscFunctionBegin;
  ierr = MatTranspose(R,MAT_INITIAL_MATRIX,&Rt);CHKERRQ(ierr);
  /* product->Dwork is used to store A*Rt in MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ() */
  ierr = MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(R,A,Rt,fill,C);CHKERRQ(ierr);
  C->ops->productnumeric = MatProductNumeric_RARt_MPIAIJ_MPIAIJ;

  /* create a supporting struct */
  ierr     = PetscNew(&rart);CHKERRQ(ierr);
  c        = (Mat_MPIAIJ*)C->data;
  c->rart  = rart;
  rart->Rt = Rt;
  rart->destroy   = C->ops->destroy;
  C->ops->destroy = MatDestroy_MPIAIJ_RARt;
  PetscFunctionReturn(0);
}
