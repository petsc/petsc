
/*
    Routines for matrix products. Calling procedure:

    MatProductCreate(A,B,C,&D); or MatProductCreateWithMat(A,B,C,D);
    MatProductSetType(D, MATPRODUCT_AB/AtB/ABt/PtAP/RARt/ABC);
    MatProductSetAlgorithm(D, alg);
    MatProductSetFill(D,fill);
    MatProductSetFromOptions(D);
      -> MatProductSetFromOptions_producttype(D):
           # Check matrix global sizes
           -> MatProductSetFromOptions_Atype_Btype_Ctype(D);
                ->MatProductSetFromOptions_Atype_Btype_Ctype_productype(D):
                    # Check matrix local sizes for mpi matrices
                    # Set default algorithm
                    # Get runtime option
                    # Set D->ops->productsymbolic = MatProductSymbolic_productype_Atype_Btype_Ctype;

    PetscLogEventBegin()
    MatProductSymbolic(D):
      # Call MatxxxSymbolic_Atype_Btype_Ctype();
      # Set D->ops->productnumeric = MatProductNumeric_productype_Atype_Btype_Ctype;
    PetscLogEventEnd()

    PetscLogEventBegin()
    MatProductNumeric(D);
      # Call (D->ops->matxxxnumeric)();
    PetscLogEventEnd()
*/

#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/

const char *const MatProductTypes[] = {"AB","AtB","ABt","PtAP","RARt","ABC","MatProductType","MAT_Product_",0};

static PetscErrorCode MatProductNumeric_PtAP_Basic(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            P = product->B,AP = product->Dwork;

  PetscFunctionBegin;
  /* AP = A*P */
  ierr = MatProductNumeric(AP);CHKERRQ(ierr);
  /* C = P^T*AP */
  if (!C->ops->transposematmultnumeric) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing numeric stage");
  ierr = (*C->ops->transposematmultnumeric)(P,AP,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_PtAP_Basic(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,P=product->B,AP;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  /* AP = A*P */
  ierr = MatProductCreate(A,P,NULL,&AP);CHKERRQ(ierr);
  ierr = MatProductSetType(AP,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(AP,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(AP,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(AP);CHKERRQ(ierr);
  ierr = MatProductSymbolic(AP);CHKERRQ(ierr);

  /* C = P^T*AP */
  ierr = MatProductSetType(C,MATPRODUCT_AtB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);
  product->A = P;
  product->B = AP;
  ierr = MatProductSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatProductSymbolic(C);CHKERRQ(ierr);

  /* resume user's original input matrix setting for A and B */
  product->A     = A;
  product->B     = P;
  product->Dwork = AP;

  C->ops->productnumeric = MatProductNumeric_PtAP_Basic;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_RARt_Basic(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            R=product->B,RA=product->Dwork;

  PetscFunctionBegin;
  /* RA = R*A */
  ierr = MatProductNumeric(RA);CHKERRQ(ierr);
  /* C = RA*R^T */
  if (!C->ops->mattransposemultnumeric) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing numeric stage");
  ierr = (*C->ops->mattransposemultnumeric)(RA,R,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_RARt_Basic(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,R=product->B,RA;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  /* RA = R*A */
  ierr = MatProductCreate(R,A,NULL,&RA);CHKERRQ(ierr);
  ierr = MatProductSetType(RA,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(RA,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(RA,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(RA);CHKERRQ(ierr);
  ierr = MatProductSymbolic(RA);CHKERRQ(ierr);

  /* C = RA*R^T */
  ierr = MatProductSetType(C,MATPRODUCT_ABt);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);
  product->A = RA;
  ierr = MatProductSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatProductSymbolic(C);CHKERRQ(ierr);

  /* resume user's original input matrix setting for A */
  product->A     = A;
  product->Dwork = RA; /* save here so it will be destroyed with product C */
  C->ops->productnumeric = MatProductNumeric_RARt_Basic;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_ABC_Basic(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,BC=product->Dwork;

  PetscFunctionBegin;
  /* Numeric BC = B*C */
  ierr = MatProductNumeric(BC);CHKERRQ(ierr);
  /* Numeric mat = A*BC */
  if (!mat->ops->transposematmultnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric stage");
  ierr = (*mat->ops->matmultnumeric)(A,BC,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_ABC_Basic(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            B=product->B,C=product->C,BC;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  /* Symbolic BC = B*C */
  ierr = MatProductCreate(B,C,NULL,&BC);CHKERRQ(ierr);
  ierr = MatProductSetType(BC,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(BC,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(BC,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(BC);CHKERRQ(ierr);
  ierr = MatProductSymbolic(BC);CHKERRQ(ierr);

  /* Symbolic mat = A*BC */
  ierr = MatProductSetType(mat,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(mat,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);
  product->B     = BC;
  product->Dwork = BC;
  ierr = MatProductSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatProductSymbolic(mat);CHKERRQ(ierr);

  /* resume user's original input matrix setting for B */
  product->B = B;
  mat->ops->productnumeric = MatProductNumeric_ABC_Basic;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_Basic(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_PtAP:
    ierr = PetscInfo2((PetscObject)mat, "MatProduct_Basic_PtAP() for A %s, P %s is used\n",((PetscObject)product->A)->type_name,((PetscObject)product->B)->type_name);CHKERRQ(ierr);
    ierr = MatProductSymbolic_PtAP_Basic(mat);CHKERRQ(ierr);
    break;
  case MATPRODUCT_RARt:
    ierr = PetscInfo2((PetscObject)mat, "MatProduct_Basic_RARt() for A %s, R %s is used\n",((PetscObject)product->A)->type_name,((PetscObject)product->B)->type_name);CHKERRQ(ierr);
    ierr = MatProductSymbolic_RARt_Basic(mat);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABC:
    ierr = PetscInfo3((PetscObject)mat, "MatProduct_Basic_ABC() for A %s, B %s, C %s is used\n",((PetscObject)product->A)->type_name,((PetscObject)product->B)->type_name,((PetscObject)product->C)->type_name);CHKERRQ(ierr);
    ierr = MatProductSymbolic_ABC_Basic(mat);CHKERRQ(ierr);
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"ProductType is not supported");
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */
/*@C
   MatProductReplaceMats - Replace input matrices for a matrix product.

   Collective on Mat

   Input Parameters:
+  A - the matrix or NULL if not being replaced
.  B - the matrix or NULL if not being replaced
.  C - the matrix or NULL if not being replaced
-  D - the matrix product

   Level: intermediate

   Notes:
     Input matrix must have exactly same data structure as replaced one.

.seealso: MatProductCreate()
@*/
PetscErrorCode MatProductReplaceMats(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;
  Mat_Product    *product=D->product;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(D,MAT_CLASSID,4);
  if (!product) SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_ORDER,"Mat D does not have struct 'product'. Call MatProductReplaceProduct()");
  if (A) {
    PetscValidHeaderSpecific(A,MAT_CLASSID,1);
    if (!product->Areplaced) {
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr); /* take ownership of input */
      ierr = MatDestroy(&product->A);CHKERRQ(ierr); /* release old reference */
      product->A = A;
    } else SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_SUP,"Matrix A was changed by a PETSc internal routine, cannot be replaced");
  }
  if (B) {
    PetscValidHeaderSpecific(B,MAT_CLASSID,2);
    if (!product->Breplaced) {
      ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr); /* take ownership of input */
      ierr = MatDestroy(&product->B);CHKERRQ(ierr); /* release old reference */
      product->B = B;
    } else SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_SUP,"Matrix B was changed by a PETSc internal routine, cannot be replaced");
  }
  if (C) {
    PetscValidHeaderSpecific(C,MAT_CLASSID,3);
    ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr); /* take ownership of input */
    ierr = MatDestroy(&product->C);CHKERRQ(ierr); /* release old reference */
    product->C = C;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_X_Dense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A = product->A, B = product->B;
  PetscInt       k, K = B->cmap->N;
  PetscBool      t = PETSC_TRUE,iscuda = PETSC_FALSE;
  PetscBool      Bcpu = PETSC_TRUE, Ccpu = PETSC_TRUE;
  char           *Btype = NULL,*Ctype = NULL;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    t = PETSC_FALSE;
  case MATPRODUCT_AtB:
    break;
  default: SETERRQ3(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProductNumeric type %s not supported for %s and %s matrices",MatProductTypes[product->type],((PetscObject)A)->type_name,((PetscObject)B)->type_name);
  }
  if (PetscDefined(HAVE_CUDA)) {
    VecType vtype;

    ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
    ierr = PetscStrcmp(vtype,VECCUDA,&iscuda);CHKERRQ(ierr);
    if (!iscuda) {
      ierr = PetscStrcmp(vtype,VECSEQCUDA,&iscuda);CHKERRQ(ierr);
    }
    if (!iscuda) {
      ierr = PetscStrcmp(vtype,VECMPICUDA,&iscuda);CHKERRQ(ierr);
    }
    if (iscuda) { /* Make sure we have up-to-date data on the GPU */
      ierr = PetscStrallocpy(((PetscObject)B)->type_name,&Btype);CHKERRQ(ierr);
      ierr = PetscStrallocpy(((PetscObject)C)->type_name,&Ctype);CHKERRQ(ierr);
      ierr = MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
      if (!C->assembled) { /* need to flag the matrix as assembled, otherwise MatConvert will complain */
        ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
      ierr = MatConvert(C,MATDENSECUDA,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
    } else { /* Make sure we have up-to-date data on the CPU */
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_VIENNACL)
      Bcpu = B->boundtocpu;
      Ccpu = C->boundtocpu;
#endif
      ierr = MatBindToCPU(B,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatBindToCPU(C,PETSC_TRUE);CHKERRQ(ierr);
    }
  }
  for (k=0;k<K;k++) {
    Vec x,y;

    ierr = MatDenseGetColumnVecRead(B,k,&x);CHKERRQ(ierr);
    ierr = MatDenseGetColumnVecWrite(C,k,&y);CHKERRQ(ierr);
    if (t) {
      ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);
    } else {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreColumnVecRead(B,k,&x);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecWrite(C,k,&y);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (PetscDefined(HAVE_CUDA)) {
    if (iscuda) {
      ierr = MatConvert(B,Btype,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
      ierr = MatConvert(C,Ctype,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
    } else {
      ierr = MatBindToCPU(B,Bcpu);CHKERRQ(ierr);
      ierr = MatBindToCPU(C,Ccpu);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(Btype);CHKERRQ(ierr);
  ierr = PetscFree(Ctype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_X_Dense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A = product->A, B = product->B;
  PetscBool      isdense;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatSetSizes(C,A->rmap->n,B->cmap->n,A->rmap->N,B->cmap->N);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatSetSizes(C,A->cmap->n,B->cmap->n,A->cmap->N,B->cmap->N);CHKERRQ(ierr);
    break;
  default: SETERRQ3(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProductSymbolic type %s not supported for %s and %s matrices",MatProductTypes[product->type],((PetscObject)A)->type_name,((PetscObject)B)->type_name);
  }
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)C,&isdense,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
  if (!isdense) {
    ierr = MatSetType(C,((PetscObject)B)->type_name);CHKERRQ(ierr);
    /* If matrix type of C was not set or not dense, we need to reset this pointers */
    C->ops->productsymbolic = MatProductSymbolic_X_Dense;
    C->ops->productnumeric  = MatProductNumeric_X_Dense;
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_X_Dense(Mat C)
{
  Mat_Product *product = C->product;
  Mat         A = product->A, B = product->B;

  PetscFunctionBegin;
  C->ops->productsymbolic = NULL;
  C->ops->productnumeric  = NULL;
  switch (product->type) {
  case MATPRODUCT_AB:
  case MATPRODUCT_AtB:
    C->ops->productsymbolic = MatProductSymbolic_X_Dense;
    C->ops->productnumeric  = MatProductNumeric_X_Dense;
    break;
  default: SETERRQ3(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProduct type %s not supported for %s and %s matrices",MatProductTypes[product->type],((PetscObject)A)->type_name,((PetscObject)B)->type_name);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_AB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;
  PetscBool      sametype;
  PetscErrorCode (*fA)(Mat);
  PetscErrorCode (*fB)(Mat);
  PetscErrorCode (*f)(Mat)=NULL;
  PetscBool      A_istrans,B_istrans;

  PetscFunctionBegin;
  /* Check matrix global sizes */
  if (B->rmap->N!=A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->cmap->N);

  fA = A->ops->productsetfromoptions;
  fB = B->ops->productsetfromoptions;

  ierr = PetscStrcmp(((PetscObject)A)->type_name,((PetscObject)B)->type_name,&sametype);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,&A_istrans);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&B_istrans);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"for A %s, B %s\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);

  if (fB == fA && sametype && (!A_istrans || !B_istrans)) {
    ierr = PetscInfo(mat,"  sametype and matching op\n");CHKERRQ(ierr);
    f = fB;
  } else {
    char      mtypes[256];
    PetscBool At_istrans=PETSC_TRUE,Bt_istrans=PETSC_TRUE;
    Mat       At = NULL,Bt = NULL;

    if (A_istrans && !B_istrans) {
      ierr = MatTransposeGetMat(A,&At);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)At,MATTRANSPOSEMAT,&At_istrans);CHKERRQ(ierr);
      if (At_istrans) { /* mat = ATT * B */
        Mat Att = NULL;
        ierr = MatTransposeGetMat(At,&Att);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)Att);CHKERRQ(ierr);
        ierr = MatDestroy(&product->A);CHKERRQ(ierr);
        A                  = Att;
        product->A         = Att; /* use Att for matproduct */
        product->Areplaced = PETSC_TRUE; /* Att = A, but has native matrix type */
      } else { /* !At_istrans: mat = At^T*B */
        ierr = PetscObjectReference((PetscObject)At);CHKERRQ(ierr);
        ierr = MatDestroy(&product->A);CHKERRQ(ierr);
        A                  = At;
        product->A         = At;
        product->Areplaced = PETSC_TRUE;
        product->type      = MATPRODUCT_AtB;
      }
    } else if (!A_istrans && B_istrans) {
      ierr = MatTransposeGetMat(B,&Bt);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)Bt,MATTRANSPOSEMAT,&Bt_istrans);CHKERRQ(ierr);
      if (Bt_istrans) { /* mat = A * BTT */
        Mat Btt = NULL;
        ierr = MatTransposeGetMat(Bt,&Btt);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)Btt);CHKERRQ(ierr);
        ierr = MatDestroy(&product->B);CHKERRQ(ierr);
        B                  = Btt;
        product->B         = Btt; /* use Btt for matproduct */
        product->Breplaced = PETSC_TRUE;
      } else { /* !Bt_istrans */
        /* mat = A*Bt^T */
        ierr = PetscObjectReference((PetscObject)Bt);CHKERRQ(ierr);
        ierr = MatDestroy(&product->B);CHKERRQ(ierr);
        B                  = Bt;
        product->B         = Bt;
        product->Breplaced = PETSC_TRUE;
        product->type = MATPRODUCT_ABt;
      }
    } else if (A_istrans && B_istrans) { /* mat = At^T * Bt^T */
      ierr = MatTransposeGetMat(A,&At);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)At,MATTRANSPOSEMAT,&At_istrans);CHKERRQ(ierr);
      ierr = MatTransposeGetMat(B,&Bt);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)Bt,MATTRANSPOSEMAT,&Bt_istrans);CHKERRQ(ierr);
      if (At_istrans && Bt_istrans) {
        Mat Att= NULL,Btt = NULL;
        ierr = MatTransposeGetMat(At,&Att);CHKERRQ(ierr);
        ierr = MatTransposeGetMat(Bt,&Btt);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)Att);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)Btt);CHKERRQ(ierr);
        ierr = MatDestroy(&product->A);CHKERRQ(ierr);
        ierr = MatDestroy(&product->B);CHKERRQ(ierr);
        A             = Att;
        product->A    = Att; product->Areplaced = PETSC_TRUE;
        B             = Btt;
        product->B    = Btt; product->Breplaced = PETSC_TRUE;
      } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Not supported yet");
    }

    /* query MatProductSetFromOptions_Atype_Btype */
    ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)A)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
    ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    }
  }

  /* last chance, we can still compute the product if B is of type dense */
  if (!f && (product->type == MATPRODUCT_AB || product->type == MATPRODUCT_AtB)) {
    PetscBool isdense;

    ierr = PetscObjectBaseTypeCompareAny((PetscObject)B,&isdense,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
    if (isdense) {
      f    = MatProductSetFromOptions_X_Dense;
      ierr = PetscInfo(mat,"  using basic looping over columns of a dense matrix\n");CHKERRQ(ierr);
    }
  }

  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else { /* reset pointers since the matrix types combination is not available */
    ierr = PetscInfo2(mat,"  for A %s and B %s is not supported\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
    mat->ops->productsymbolic = NULL;
    mat->ops->productnumeric  = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_AtB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;
  PetscBool      sametype;
  PetscErrorCode (*fA)(Mat);
  PetscErrorCode (*fB)(Mat);
  PetscErrorCode (*f)(Mat)=NULL;

  PetscFunctionBegin;
  /* Check matrix global sizes */
  if (B->rmap->N!=A->rmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->rmap->N);

  fA = A->ops->productsetfromoptions;
  fB = B->ops->productsetfromoptions;

  ierr = PetscStrcmp(((PetscObject)A)->type_name,((PetscObject)B)->type_name,&sametype);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"for A %s, B %s\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);

  if (fB == fA && sametype) {
    ierr = PetscInfo(mat,"  sametype and matching op\n");CHKERRQ(ierr);
    f = fB;
  } else {
    char      mtypes[256];
    PetscBool istrans;
    ierr = PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,&istrans);CHKERRQ(ierr);
    if (!istrans) {
      /* query MatProductSetFromOptions_Atype_Btype */
      ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)A)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    } else {
      Mat T = NULL;
      SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MatProductSetFromOptions_AtB for A %s and B %s is not supported",((PetscObject)A)->type_name,((PetscObject)B)->type_name);

      ierr = MatTransposeGetMat(A,&T);CHKERRQ(ierr);
      ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)T)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);

      product->type = MATPRODUCT_AtB;
      ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    }

    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    }
  }

  /* last chance, we can still compute the product if B is of type dense */
  if (!f && (product->type == MATPRODUCT_AB || product->type == MATPRODUCT_AtB)) {
    PetscBool isdense;

    ierr = PetscObjectBaseTypeCompareAny((PetscObject)B,&isdense,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
    if (isdense) {
      f    = MatProductSetFromOptions_X_Dense;
      ierr = PetscInfo(mat,"  using basic looping over columns of a dense matrix\n");CHKERRQ(ierr);
    }
  }

  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else { /* reset pointers since the matrix types combination is not available */
    ierr = PetscInfo2(mat,"  for A %s and B %s is not supported\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
    mat->ops->productsymbolic = NULL;
    mat->ops->productnumeric  = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_ABt(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;
  PetscBool      sametype;
  PetscErrorCode (*fA)(Mat);
  PetscErrorCode (*fB)(Mat);
  PetscErrorCode (*f)(Mat)=NULL;

  PetscFunctionBegin;
  /* Check matrix global sizes */
  if (B->cmap->N!=A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, AN %D != BN %D",A->cmap->N,B->cmap->N);

  fA = A->ops->productsetfromoptions;
  fB = B->ops->productsetfromoptions;

  ierr = PetscStrcmp(((PetscObject)A)->type_name,((PetscObject)B)->type_name,&sametype);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"for A %s, B %s\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
  if (fB == fA && sametype) {
    ierr = PetscInfo(mat,"  sametype and matching op\n");CHKERRQ(ierr);
    f = fB;
  } else {
    char      mtypes[256];
    PetscBool istrans;
    ierr = PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,&istrans);CHKERRQ(ierr);
    if (!istrans) {
      /* query MatProductSetFromOptions_Atype_Btype */
      ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)A)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    } else {
      Mat T = NULL;
      SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MatProductSetFromOptions_ABt for A %s and B %s is not supported",((PetscObject)A)->type_name,((PetscObject)B)->type_name);

      ierr = MatTransposeGetMat(A,&T);CHKERRQ(ierr);
      ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)T)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);

      product->type = MATPRODUCT_ABt;
      ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s (ABt)? %p\n",mtypes,f);CHKERRQ(ierr);
    }

    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s (ABt)? %p\n",mtypes,f);CHKERRQ(ierr);
    }
  }

  /* last chance, we can still compute the product if B is of type dense */
  if (!f && (product->type == MATPRODUCT_AB || product->type == MATPRODUCT_AtB)) {
    PetscBool isdense;

    ierr = PetscObjectBaseTypeCompareAny((PetscObject)B,&isdense,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
    if (isdense) {
      f    = MatProductSetFromOptions_X_Dense;
      ierr = PetscInfo(mat,"  using basic looping over columns of a dense matrix\n");CHKERRQ(ierr);
    }
  }

  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else { /* reset pointers since the matrix types combination is not available */
    ierr = PetscInfo2(mat,"  for A %s and B %s is not supported\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
    mat->ops->productsymbolic = NULL;
    mat->ops->productnumeric  = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_PtAP(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;
  PetscBool      sametype;
  PetscErrorCode (*fA)(Mat);
  PetscErrorCode (*fB)(Mat);
  PetscErrorCode (*f)(Mat)=NULL;

  PetscFunctionBegin;
  /* Check matrix global sizes */
  if (A->rmap->N != A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix A must be square, %D != %D",A->rmap->N,A->cmap->N);
  if (B->rmap->N != A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->cmap->N);

  fA = A->ops->productsetfromoptions;
  fB = B->ops->productsetfromoptions;

  ierr = PetscInfo2(mat,"for A %s, P %s\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcmp(((PetscObject)A)->type_name,((PetscObject)B)->type_name,&sametype);CHKERRQ(ierr);
  if (fB == fA && sametype) {
    ierr = PetscInfo(mat,"  sametype and matching op\n");CHKERRQ(ierr);
    f = fB;
  } else {
    /* query MatProductSetFromOptions_Atype_Btype */
    char  mtypes[256];
    ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)A)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
    ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);

    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    }
  }

  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo2(mat,"  for A %s, P %s uses MatProduct_Basic() implementation\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
    mat->ops->productsymbolic = MatProductSymbolic_Basic;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_RARt(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;
  PetscBool      sametype;
  PetscErrorCode (*fA)(Mat);
  PetscErrorCode (*fB)(Mat);
  PetscErrorCode (*f)(Mat)=NULL;

  PetscFunctionBegin;
  /* Check matrix global sizes */
  if (A->rmap->N != B->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix A must be square, %D != %D",A->rmap->N,A->cmap->N);
  if (B->cmap->N != A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->cmap->N,A->cmap->N);

  fA = A->ops->productsetfromoptions;
  fB = B->ops->productsetfromoptions;

  ierr = PetscStrcmp(((PetscObject)A)->type_name,((PetscObject)B)->type_name,&sametype);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"for A %s, B %s\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
  if (fB == fA && sametype) {
    ierr = PetscInfo(mat,"  sametype and matching op\n");CHKERRQ(ierr);
    f = fB;
  } else {
    /* query MatProductSetFromOptions_Atype_Btype */
    char  mtypes[256];
    ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)A)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
    ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);

    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    }
  }

  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo2(mat,"  for A %s, P %s uses MatProduct_Basic() implementation\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name);CHKERRQ(ierr);
    mat->ops->productsymbolic = MatProductSymbolic_Basic;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_ABC(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B,C=product->C;
  PetscErrorCode (*fA)(Mat);
  PetscErrorCode (*fB)(Mat);
  PetscErrorCode (*fC)(Mat);
  PetscErrorCode (*f)(Mat)=NULL;

  PetscFunctionBegin;
  /* Check matrix global sizes */
  if (B->rmap->N!= A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->cmap->N);
  if (C->rmap->N!= B->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",C->rmap->N,B->cmap->N);

  fA = A->ops->productsetfromoptions;
  fB = B->ops->productsetfromoptions;
  fC = C->ops->productsetfromoptions;
  ierr = PetscInfo3(mat,"for A %s, B %s and C %s\n",((PetscObject)A)->type_name,((PetscObject)B)->type_name,((PetscObject)C)->type_name);CHKERRQ(ierr);
  if (fA == fB && fA == fC && fA) {
    ierr = PetscInfo(mat,"  matching op\n");CHKERRQ(ierr);
    f = fA;
  } else {
    /* query MatProductSetFromOptions_Atype_Btype_Ctype */
    char  mtypes[256];
    ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)A)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)C)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);

    ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
    ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    }
    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)C,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s? %p\n",mtypes,f);CHKERRQ(ierr);
    }
  }

  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else { /* use MatProductSymbolic/Numeric_Basic() */
    ierr = PetscInfo3(mat,"  for A %s, B %s and C %s uses MatProduct_Basic() implementation",((PetscObject)A)->type_name,((PetscObject)B)->type_name,((PetscObject)C)->type_name);CHKERRQ(ierr);
    mat->ops->productsymbolic = MatProductSymbolic_Basic;
  }
  PetscFunctionReturn(0);
}

/*@C
   MatProductSetFromOptions - Creates a matrix product where the type, the algorithm etc are determined from the options database.

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix

   Level: beginner

.seealso: MatSetFromOptions()
@*/
PetscErrorCode MatProductSetFromOptions(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (mat->ops->productsetfromoptions) {
    ierr = (*mat->ops->productsetfromoptions)(mat);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Call MatProductSetType() first");
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */
PetscErrorCode MatProductNumeric_AB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->matmultnumeric) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->matmultnumeric)(A,B,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_AtB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->transposematmultnumeric) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_TransposeMatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->transposematmultnumeric)(A,B,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_TransposeMatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_ABt(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->mattransposemultnumeric) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->mattransposemultnumeric)(A,B,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_PtAP(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->ptapnumeric) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_PtAPNumeric,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->ptapnumeric)(A,B,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_RARt(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->rartnumeric) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_RARtNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->rartnumeric)(A,B,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_RARtNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_ABC(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B,C=product->C;

  PetscFunctionBegin;
  if (!mat->ops->matmatmultnumeric) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  ierr = (*mat->ops->matmatmultnumeric)(A,B,C,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMatMultNumeric,A,B,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatProductNumeric - Implement a matrix product with numerical values.

   Collective on Mat

   Input Parameters:
.  mat - the matrix to hold a product

   Output Parameters:
.  mat - the matrix product

   Level: intermediate

.seealso: MatProductCreate(), MatSetType()
@*/
PetscErrorCode MatProductNumeric(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (mat->ops->productnumeric) {
    ierr = (*mat->ops->productnumeric)(mat);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Call MatProductSymbolic() first");
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */
PetscErrorCode MatProductSymbolic_AB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->matmultsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
  ierr = (*mat->ops->matmultsymbolic)(A,B,product->fill,mat);CHKERRQ(ierr);
  mat->ops->productnumeric = MatProductNumeric_AB;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_AtB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->transposematmultsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
  ierr = (*mat->ops->transposematmultsymbolic)(A,B,product->fill,mat);CHKERRQ(ierr);
  mat->ops->productnumeric = MatProductNumeric_AtB;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_ABt(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->mattransposemultsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
  ierr = (*mat->ops->mattransposemultsymbolic)(A,B,product->fill,mat);CHKERRQ(ierr);
  mat->ops->productnumeric = MatProductNumeric_ABt;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_ABC(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B,C=product->C;

  PetscFunctionBegin;
  if (!mat->ops->matmatmultsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
  ierr = (*mat->ops->matmatmultsymbolic)(A,B,C,product->fill,mat);CHKERRQ(ierr);
  mat->ops->productnumeric = MatProductNumeric_ABC;
  PetscFunctionReturn(0);
}

/*@
   MatProductSymbolic - Perform the symbolic portion of a matrix product, this creates a data structure for use with the numerical produce.

   Collective on Mat

   Input Parameters:
.  mat - the matrix to hold a product

   Output Parameters:
.  mat - the matrix product data structure

   Level: intermediate

.seealso: MatProductCreate(), MatSetType(), MatProductNumeric(), MatProductType, MatProductAlgorithm
@*/
PetscErrorCode MatProductSymbolic(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  MatProductType productype = product->type;
  PetscLogEvent  eventtype=-1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);

  /* log event */
  switch (productype) {
  case MATPRODUCT_AB:
    eventtype = MAT_MatMultSymbolic;
    break;
  case MATPRODUCT_AtB:
    eventtype = MAT_TransposeMatMultSymbolic;
    break;
  case MATPRODUCT_ABt:
    eventtype = MAT_MatTransposeMultSymbolic;
    break;
  case MATPRODUCT_PtAP:
    eventtype = MAT_PtAPSymbolic;
    break;
  case MATPRODUCT_RARt:
    eventtype = MAT_RARtSymbolic;
    break;
  case MATPRODUCT_ABC:
    eventtype = MAT_MatMatMultSymbolic;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MATPRODUCT type is not supported");
  }

  if (mat->ops->productsymbolic) {
    ierr = PetscLogEventBegin(eventtype,mat,0,0,0);CHKERRQ(ierr);
    ierr = (*mat->ops->productsymbolic)(mat);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(eventtype,mat,0,0,0);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Call MatProductSetFromOptions() first");
  PetscFunctionReturn(0);
}

/*@
   MatProductSetFill - Set an expected fill of the matrix product.

   Collective on Mat

   Input Parameters:
+  mat - the matrix product
-  fill - expected fill as ratio of nnz(mat)/(nnz(A) + nnz(B) + nnz(C)); use PETSC_DEFAULT if you do not have a good estimate. If the product is a dense matrix, this is irrelevent.

   Level: intermediate

.seealso: MatProductSetType(), MatProductSetAlgorithm(), MatProductCreate()
@*/
PetscErrorCode MatProductSetFill(Mat mat,PetscReal fill)
{
  Mat_Product *product = mat->product;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (!product) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Data struct Mat_Product is not created, call MatProductCreate() first");
  if (fill == PETSC_DEFAULT || fill == PETSC_DECIDE) product->fill = 2.0;
  else product->fill = fill;
  PetscFunctionReturn(0);
}

/*@
   MatProductSetAlgorithm - Requests a particular algorithm for a matrix product implementation.

   Collective on Mat

   Input Parameters:
+  mat - the matrix product
-  alg - particular implementation algorithm of the matrix product, e.g., MATPRODUCTALGORITHM_DEFAULT.

   Level: intermediate

.seealso: MatProductSetType(), MatProductSetFill(), MatProductCreate()
@*/
PetscErrorCode MatProductSetAlgorithm(Mat mat,MatProductAlgorithm alg)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (!product) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Data struct Mat_Product is not created, call MatProductCreate() first");
  ierr = PetscFree(product->alg);CHKERRQ(ierr);
  ierr = PetscStrallocpy(alg,&product->alg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatProductSetType - Sets a particular matrix product type, for example Mat*Mat.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  productype   - matrix product type, e.g., MATPRODUCT_AB,MATPRODUCT_AtB,MATPRODUCT_ABt,MATPRODUCT_PtAP,MATPRODUCT_RARt,MATPRODUCT_ABC.

   Level: intermediate

.seealso: MatProductCreate(), MatProductType, MatProductAlgorithm
@*/
PetscErrorCode MatProductSetType(Mat mat,MatProductType productype)
{
  Mat_Product *product = mat->product;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,productype,2);
  if (!product) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Data struct Mat_Product is not created, call MatProductCreate() first");
  product->type = productype;

  switch (productype) {
  case MATPRODUCT_AB:
    mat->ops->productsetfromoptions = MatProductSetFromOptions_AB;
    break;
  case MATPRODUCT_AtB:
    mat->ops->productsetfromoptions = MatProductSetFromOptions_AtB;
    break;
  case MATPRODUCT_ABt:
    mat->ops->productsetfromoptions = MatProductSetFromOptions_ABt;
    break;
  case MATPRODUCT_PtAP:
    mat->ops->productsetfromoptions = MatProductSetFromOptions_PtAP;
    break;
  case MATPRODUCT_RARt:
    mat->ops->productsetfromoptions = MatProductSetFromOptions_RARt;
    break;
  case MATPRODUCT_ABC:
    mat->ops->productsetfromoptions = MatProductSetFromOptions_ABC;
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"ProductType %s is not supported",MatProductTypes[product->type]);
  }
  PetscFunctionReturn(0);
}

/*@
   MatProductClear - Clears matrix product internal structure.

   Collective on Mat

   Input Parameters:
.  mat - the product matrix

   Level: intermediate
@*/
PetscErrorCode MatProductClear(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (product) {
    /* release reference */
    ierr = MatDestroy(&product->A);CHKERRQ(ierr);
    ierr = MatDestroy(&product->B);CHKERRQ(ierr);
    ierr = MatDestroy(&product->C);CHKERRQ(ierr);
    ierr = PetscFree(product->alg);CHKERRQ(ierr);
    ierr = MatDestroy(&product->Dwork);CHKERRQ(ierr);
    ierr = PetscFree(mat->product);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Create a supporting struct and attach it to the matrix product */
PetscErrorCode MatProductCreate_Private(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;
  Mat_Product    *product=NULL;

  PetscFunctionBegin;
  ierr = PetscNewLog(D,&product);CHKERRQ(ierr);
  product->A        = A;
  product->B        = B;
  product->C        = C;
  product->Dwork    = NULL;
  product->fill     = 2.0; /* PETSC_DEFAULT */
  product->Areplaced = PETSC_FALSE;
  product->Breplaced = PETSC_FALSE;
  product->api_user  = PETSC_FALSE;
  D->product         = product;

  ierr = MatProductSetAlgorithm(D,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);

  /* take ownership */
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatProductCreateWithMat - Setup a given matrix as a matrix product.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix (optional)
-  D - the matrix which will be used as a product

   Output Parameters:
.  D - the product matrix

   Level: intermediate

.seealso: MatProductCreate()
@*/
PetscErrorCode MatProductCreateWithMat(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  MatCheckPreallocated(A,1);
  if (!A->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(B,2);
  MatCheckPreallocated(B,2);
  if (!B->assembled) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factortype) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  if (C) {
    PetscValidHeaderSpecific(C,MAT_CLASSID,3);
    PetscValidType(C,3);
    MatCheckPreallocated(C,3);
    if (!C->assembled) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
    if (C->factortype) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  }

  PetscValidHeaderSpecific(D,MAT_CLASSID,4);
  PetscValidType(D,4);
  MatCheckPreallocated(D,4);
  if (!D->assembled) SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (D->factortype) SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  /* Create a supporting struct and attach it to D */
  ierr = MatProductCreate_Private(A,B,C,D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatProductCreate - create a matrix product object that can be used to compute various matrix times matrix operations.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  C - the third matrix (optional)

   Output Parameters:
.  D - the product matrix

   Level: intermediate

.seealso: MatProductCreateWithMat(), MatProductSetType(), MatProductSetAlgorithm()
@*/
PetscErrorCode MatProductCreate(Mat A,Mat B,Mat C,Mat *D)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  MatCheckPreallocated(A,1);
  if (!A->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(B,2);
  MatCheckPreallocated(B,2);
  if (!B->assembled) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factortype) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  if (C) {
    PetscValidHeaderSpecific(C,MAT_CLASSID,3);
    PetscValidType(C,3);
    MatCheckPreallocated(C,3);
    if (!C->assembled) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
    if (C->factortype) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  }

  PetscValidPointer(D,4);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),D);CHKERRQ(ierr);
  ierr = MatProductCreate_Private(A,B,C,*D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
