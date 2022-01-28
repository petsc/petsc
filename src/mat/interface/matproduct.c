
/*
    Routines for matrix products. Calling procedure:

    MatProductCreate(A,B,C,&D); or MatProductCreateWithMat(A,B,C,D)
    MatProductSetType(D, MATPRODUCT_AB/AtB/ABt/PtAP/RARt/ABC)
    MatProductSetAlgorithm(D, alg)
    MatProductSetFill(D,fill)
    MatProductSetFromOptions(D)
      -> MatProductSetFromOptions_Private(D)
           # Check matrix global sizes
           if the matrices have the same setfromoptions routine, use it
           if not, try:
             -> Query MatProductSetFromOptions_Atype_Btype_Ctype_C(D) from A, B and C (in order)
             if found -> run the specific setup that must set the symbolic operation (these callbacks should never fail)
           if callback not found or no symbolic operation set
             -> Query MatProductSetFromOptions_anytype_C(D) from A, B and C (in order) (e.g, matrices may have inner matrices like MATTRANPOSEMAT)
           if dispatch found but combination still not present do
             -> check if B is dense and product type AtB or AB -> if true, basic looping of dense columns
             -> check if triple product (PtAP, RARt or ABC) -> if true, set the Basic routines

    #  The setfromoptions calls MatProductSetFromOptions_Atype_Btype_Ctype should
    #    Check matrix local sizes for mpi matrices
    #    Set default algorithm
    #    Get runtime option
    #    Set D->ops->productsymbolic = MatProductSymbolic_productype_Atype_Btype_Ctype if found

    MatProductSymbolic(D)
      # Call MatProductSymbolic_productype_Atype_Btype_Ctype()
        the callback must set the numeric phase D->ops->productnumeric = MatProductNumeric_productype_Atype_Btype_Ctype

    MatProductNumeric(D)
      # Call the numeric phase

    # The symbolic phases are allowed to set extra data structures and attach those to the product
    # this additional data can be reused between multiple numeric phases with the same matrices
    # if not needed, call
    MatProductClear(D)
*/

#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/

const char *const MatProductTypes[] = {"UNSPECIFIED","AB","AtB","ABt","PtAP","RARt","ABC"};

/* these are basic implementations relying on the old function pointers
 * they are dangerous and should be removed in the future */
static PetscErrorCode MatProductNumeric_PtAP_Unsafe(Mat C)
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

static PetscErrorCode MatProductSymbolic_PtAP_Unsafe(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,P=product->B,AP;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  ierr = PetscInfo2((PetscObject)C,"for A %s, P %s is used\n",((PetscObject)product->A)->type_name,((PetscObject)product->B)->type_name);CHKERRQ(ierr);
  /* AP = A*P */
  ierr = MatProductCreate(A,P,NULL,&AP);CHKERRQ(ierr);
  ierr = MatProductSetType(AP,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(AP,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(AP,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(AP);CHKERRQ(ierr);
  ierr = MatProductSymbolic(AP);CHKERRQ(ierr);

  /* C = P^T*AP */
  ierr = MatProductSetType(C,MATPRODUCT_AtB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  product->A = P;
  product->B = AP;
  ierr = MatProductSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatProductSymbolic(C);CHKERRQ(ierr);

  /* resume user's original input matrix setting for A and B */
  product->A     = A;
  product->B     = P;
  product->Dwork = AP;

  C->ops->productnumeric = MatProductNumeric_PtAP_Unsafe;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_RARt_Unsafe(Mat C)
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

static PetscErrorCode MatProductSymbolic_RARt_Unsafe(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,R=product->B,RA;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  ierr = PetscInfo2((PetscObject)C,"for A %s, R %s is used\n",((PetscObject)product->A)->type_name,((PetscObject)product->B)->type_name);CHKERRQ(ierr);
  /* RA = R*A */
  ierr = MatProductCreate(R,A,NULL,&RA);CHKERRQ(ierr);
  ierr = MatProductSetType(RA,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(RA,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(RA,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(RA);CHKERRQ(ierr);
  ierr = MatProductSymbolic(RA);CHKERRQ(ierr);

  /* C = RA*R^T */
  ierr = MatProductSetType(C,MATPRODUCT_ABt);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  product->A = RA;
  ierr = MatProductSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatProductSymbolic(C);CHKERRQ(ierr);

  /* resume user's original input matrix setting for A */
  product->A     = A;
  product->Dwork = RA; /* save here so it will be destroyed with product C */
  C->ops->productnumeric = MatProductNumeric_RARt_Unsafe;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_ABC_Unsafe(Mat mat)
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

static PetscErrorCode MatProductSymbolic_ABC_Unsafe(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            B=product->B,C=product->C,BC;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  ierr = PetscInfo3((PetscObject)mat,"for A %s, B %s, C %s is used\n",((PetscObject)product->A)->type_name,((PetscObject)product->B)->type_name,((PetscObject)product->C)->type_name);CHKERRQ(ierr);
  /* Symbolic BC = B*C */
  ierr = MatProductCreate(B,C,NULL,&BC);CHKERRQ(ierr);
  ierr = MatProductSetType(BC,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(BC,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(BC,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(BC);CHKERRQ(ierr);
  ierr = MatProductSymbolic(BC);CHKERRQ(ierr);

  /* Symbolic mat = A*BC */
  ierr = MatProductSetType(mat,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(mat,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  product->B     = BC;
  product->Dwork = BC;
  ierr = MatProductSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatProductSymbolic(mat);CHKERRQ(ierr);

  /* resume user's original input matrix setting for B */
  product->B = B;
  mat->ops->productnumeric = MatProductNumeric_ABC_Unsafe;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_Unsafe(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_PtAP:
    ierr = MatProductSymbolic_PtAP_Unsafe(mat);CHKERRQ(ierr);
    break;
  case MATPRODUCT_RARt:
    ierr = MatProductSymbolic_RARt_Unsafe(mat);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABC:
    ierr = MatProductSymbolic_ABC_Unsafe(mat);CHKERRQ(ierr);
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"ProductType %s is not supported",MatProductTypes[product->type]);
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
     To reuse the symbolic phase, input matrices must have exactly the same data structure as the replaced one.
     If the type of any of the input matrices is different than what was previously used, or their symmetry changed but
     the symbolic phase took advantage of their symmetry, the product is cleared and MatProductSetFromOptions()/MatProductSymbolic() are invoked again.

.seealso: MatProductCreate(), MatProductSetFromOptions(), MatProductSymbolic(). MatProductClear()
@*/
PetscErrorCode MatProductReplaceMats(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;
  Mat_Product    *product;
  PetscBool      flgA = PETSC_TRUE,flgB = PETSC_TRUE,flgC = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(D,MAT_CLASSID,4);
  MatCheckProduct(D,4);
  product = D->product;
  if (A) {
    PetscValidHeaderSpecific(A,MAT_CLASSID,1);
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)product->A,((PetscObject)A)->type_name,&flgA);CHKERRQ(ierr);
    if (product->symbolic_used_the_fact_A_is_symmetric && !A->symmetric) { /* symbolic was built around a symmetric A, but the new A is not anymore */
      flgA = PETSC_FALSE;
      product->symbolic_used_the_fact_A_is_symmetric = PETSC_FALSE; /* reinit */
    }
    ierr = MatDestroy(&product->A);CHKERRQ(ierr);
    product->A = A;
  }
  if (B) {
    PetscValidHeaderSpecific(B,MAT_CLASSID,2);
    ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)product->B,((PetscObject)B)->type_name,&flgB);CHKERRQ(ierr);
    if (product->symbolic_used_the_fact_B_is_symmetric && !B->symmetric) {
      flgB = PETSC_FALSE;
      product->symbolic_used_the_fact_B_is_symmetric = PETSC_FALSE; /* reinit */
    }
    ierr = MatDestroy(&product->B);CHKERRQ(ierr);
    product->B = B;
  }
  if (C) {
    PetscValidHeaderSpecific(C,MAT_CLASSID,3);
    ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)product->C,((PetscObject)C)->type_name,&flgC);CHKERRQ(ierr);
    if (product->symbolic_used_the_fact_C_is_symmetric && !C->symmetric) {
      flgC = PETSC_FALSE;
      product->symbolic_used_the_fact_C_is_symmetric = PETSC_FALSE; /* reinit */
    }
    ierr = MatDestroy(&product->C);CHKERRQ(ierr);
    product->C = C;
  }
  /* Any of the replaced mats is of a different type, reset */
  if (!flgA || !flgB || !flgC) {
    if (D->product->destroy) {
      ierr = (*D->product->destroy)(D->product->data);CHKERRQ(ierr);
    }
    D->product->destroy = NULL;
    D->product->data = NULL;
    if (D->ops->productnumeric || D->ops->productsymbolic) {
      ierr = MatProductSetFromOptions(D);CHKERRQ(ierr);
      ierr = MatProductSymbolic(D);CHKERRQ(ierr);
    }
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
  default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProductNumeric type %s not supported for %s and %s matrices",MatProductTypes[product->type],((PetscObject)A)->type_name,((PetscObject)B)->type_name);
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
  default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProductSymbolic type %s not supported for %s and %s matrices",MatProductTypes[product->type],((PetscObject)A)->type_name,((PetscObject)B)->type_name);
  }
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)C,&isdense,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
  if (!isdense) {
    ierr = MatSetType(C,((PetscObject)B)->type_name);CHKERRQ(ierr);
    /* If matrix type of C was not set or not dense, we need to reset the pointer */
    C->ops->productsymbolic = MatProductSymbolic_X_Dense;
  }
  C->ops->productnumeric = MatProductNumeric_X_Dense;
  ierr = MatSetUp(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* a single driver to query the dispatching */
static PetscErrorCode MatProductSetFromOptions_Private(Mat mat)
{
  PetscErrorCode    ierr;
  Mat_Product       *product = mat->product;
  PetscInt          Am,An,Bm,Bn,Cm,Cn;
  Mat               A = product->A,B = product->B,C = product->C;
  const char* const Bnames[] = { "B", "R", "P" };
  const char*       bname;
  PetscErrorCode    (*fA)(Mat);
  PetscErrorCode    (*fB)(Mat);
  PetscErrorCode    (*fC)(Mat);
  PetscErrorCode    (*f)(Mat)=NULL;

  PetscFunctionBegin;
  mat->ops->productsymbolic = NULL;
  mat->ops->productnumeric = NULL;
  if (product->type == MATPRODUCT_UNSPECIFIED) PetscFunctionReturn(0);
  if (!A) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing A mat");
  if (!B) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing B mat");
  if (product->type == MATPRODUCT_ABC && !C) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing C mat");
  if (product->type != MATPRODUCT_ABC) C = NULL; /* do not use C if not needed */
  if (product->type == MATPRODUCT_RARt) bname = Bnames[1];
  else if (product->type == MATPRODUCT_PtAP) bname = Bnames[2];
  else bname = Bnames[0];

  /* Check matrices sizes */
  Am = A->rmap->N;
  An = A->cmap->N;
  Bm = B->rmap->N;
  Bn = B->cmap->N;
  Cm = C ? C->rmap->N : 0;
  Cn = C ? C->cmap->N : 0;
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_ABt) { PetscInt t = Bn; Bn = Bm; Bm = t; }
  if (product->type == MATPRODUCT_AtB) { PetscInt t = An; An = Am; Am = t; }
  if (An != Bm) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Matrix dimensions of A and %s are incompatible for MatProductType %s: A %" PetscInt_FMT "x%" PetscInt_FMT ", %s %" PetscInt_FMT "x%" PetscInt_FMT,bname,MatProductTypes[product->type],A->rmap->N,A->cmap->N,bname,B->rmap->N,B->cmap->N);
  if (Cm && Cm != Bn) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Matrix dimensions of B and C are incompatible for MatProductType %s: B %" PetscInt_FMT "x%" PetscInt_FMT ", C %" PetscInt_FMT "x%" PetscInt_FMT,MatProductTypes[product->type],B->rmap->N,B->cmap->N,Cm,Cn);

  fA = A->ops->productsetfromoptions;
  fB = B->ops->productsetfromoptions;
  fC = C ? C->ops->productsetfromoptions : fA;
  if (C) {
    ierr = PetscInfo5(mat,"MatProductType %s for A %s, %s %s, C %s\n",MatProductTypes[product->type],((PetscObject)A)->type_name,bname,((PetscObject)B)->type_name,((PetscObject)C)->type_name);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo4(mat,"MatProductType %s for A %s, %s %s\n",MatProductTypes[product->type],((PetscObject)A)->type_name,bname,((PetscObject)B)->type_name);CHKERRQ(ierr);
  }
  if (fA == fB && fA == fC && fA) {
    ierr = PetscInfo(mat,"  matching op\n");CHKERRQ(ierr);
    ierr = (*fA)(mat);CHKERRQ(ierr);
  }
  /* We may have found f but it did not succeed */
  if (!mat->ops->productsymbolic) { /* query MatProductSetFromOptions_Atype_Btype_Ctype */
    char  mtypes[256];
    ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)A)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
    ierr = PetscStrlcat(mtypes,((PetscObject)B)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    if (C) {
      ierr = PetscStrlcat(mtypes,"_",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscStrlcat(mtypes,((PetscObject)C)->type_name,sizeof(mtypes));CHKERRQ(ierr);
    }
    ierr = PetscStrlcat(mtypes,"_C",sizeof(mtypes));CHKERRQ(ierr);

    ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
    ierr = PetscInfo2(mat,"  querying %s from A? %p\n",mtypes,f);CHKERRQ(ierr);
    if (!f) {
      ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo3(mat,"  querying %s from %s? %p\n",mtypes,bname,f);CHKERRQ(ierr);
    }
    if (!f && C) {
      ierr = PetscObjectQueryFunction((PetscObject)C,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s from C? %p\n",mtypes,f);CHKERRQ(ierr);
    }
    if (f) { ierr = (*f)(mat);CHKERRQ(ierr); }

    /* We may have found f but it did not succeed */
    /* some matrices (i.e. MATTRANSPOSE, MATSHELL constructed from MatConvert), knows what to do with their inner matrices */
    if (!mat->ops->productsymbolic) {
      ierr = PetscStrncpy(mtypes,"MatProductSetFromOptions_anytype_C",sizeof(mtypes));CHKERRQ(ierr);
      ierr = PetscObjectQueryFunction((PetscObject)A,mtypes,&f);CHKERRQ(ierr);
      ierr = PetscInfo2(mat,"  querying %s from A? %p\n",mtypes,f);CHKERRQ(ierr);
      if (!f) {
        ierr = PetscObjectQueryFunction((PetscObject)B,mtypes,&f);CHKERRQ(ierr);
        ierr = PetscInfo3(mat,"  querying %s from %s? %p\n",mtypes,bname,f);CHKERRQ(ierr);
      }
      if (!f && C) {
        ierr = PetscObjectQueryFunction((PetscObject)C,mtypes,&f);CHKERRQ(ierr);
        ierr = PetscInfo2(mat,"  querying %s from C? %p\n",mtypes,f);CHKERRQ(ierr);
      }
    }
    if (f) { ierr = (*f)(mat);CHKERRQ(ierr); }
  }

  /* We may have found f but it did not succeed */
  if (!mat->ops->productsymbolic) {
    /* we can still compute the product if B is of type dense */
    if (product->type == MATPRODUCT_AB || product->type == MATPRODUCT_AtB) {
      PetscBool isdense;

      ierr = PetscObjectBaseTypeCompareAny((PetscObject)B,&isdense,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
      if (isdense) {

        mat->ops->productsymbolic = MatProductSymbolic_X_Dense;
        ierr = PetscInfo(mat,"  using basic looping over columns of a dense matrix\n");CHKERRQ(ierr);
      }
    } else if (product->type != MATPRODUCT_ABt) { /* use MatProductSymbolic/Numeric_Unsafe() for triple products only */
      /*
         TODO: this should be changed to a proper setfromoptions, not setting the symbolic pointer here, because we do not know if
               the combination will succeed. In order to be sure, we need MatProductGetProductType to return the type of the result
               before computing the symbolic phase
      */
      ierr = PetscInfo(mat,"  symbolic product not supported, using MatProductSymbolic_Unsafe() implementation\n");CHKERRQ(ierr);
      mat->ops->productsymbolic = MatProductSymbolic_Unsafe;
    }
  }
  if (!mat->ops->productsymbolic) {
    ierr = PetscInfo(mat,"  symbolic product is not supported\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   MatProductSetFromOptions - Creates a matrix product where the type, the algorithm etc are determined from the options database.

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix

   Options Database Keys:
.    -mat_product_clear - Clear intermediate data structures after MatProductNumeric() has been called

   Level: intermediate

.seealso: MatSetFromOptions(), MatProductCreate(), MatProductCreateWithMat()
@*/
PetscErrorCode MatProductSetFromOptions(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  MatCheckProduct(mat,1);
  if (mat->product->data) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Cannot call MatProductSetFromOptions with already present data");
  ierr = PetscObjectOptionsBegin((PetscObject)mat);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_product_clear","Clear intermediate data structures after MatProductNumeric() has been called","MatProductClear",mat->product->clear,&mat->product->clear,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsDeprecated("-mat_freeintermediatedatastructures","-mat_product_clear","3.13","Or call MatProductClear() after MatProductNumeric()");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = MatProductSetFromOptions_Private(mat);CHKERRQ(ierr);
  if (!mat->product) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing product after setup phase");
  PetscFunctionReturn(0);
}

/*@C
   MatProductView - View a MatProduct

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix obtained with MatProductCreate() or MatProductCreateWithMat()

   Level: intermediate

.seealso: MatView(), MatProductCreate(), MatProductCreateWithMat()
@*/
PetscErrorCode MatProductView(Mat mat, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (!mat->product) PetscFunctionReturn(0);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mat),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(mat,1,viewer,2);
  if (mat->product->view) {
    ierr = (*mat->product->view)(mat,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */
/* these are basic implementations relying on the old function pointers
 * they are dangerous and should be removed in the future */
PetscErrorCode MatProductNumeric_AB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->matmultnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = (*mat->ops->matmultnumeric)(A,B,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_AtB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->transposematmultnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = (*mat->ops->transposematmultnumeric)(A,B,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_ABt(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->mattransposemultnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = (*mat->ops->mattransposemultnumeric)(A,B,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_PtAP(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->ptapnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = (*mat->ops->ptapnumeric)(A,B,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_RARt(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->rartnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = (*mat->ops->rartnumeric)(A,B,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductNumeric_ABC(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B,C=product->C;

  PetscFunctionBegin;
  if (!mat->ops->matmatmultnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric implementation of product %s for mat %s",MatProductTypes[product->type],((PetscObject)mat)->type_name);
  ierr = (*mat->ops->matmatmultnumeric)(A,B,C,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */

/*@
   MatProductNumeric - Implement a matrix product with numerical values.

   Collective on Mat

   Input/Output Parameter:
.  mat - the matrix holding the product

   Level: intermediate

   Notes: MatProductSymbolic() must have been called on mat before calling this function

.seealso: MatProductCreate(), MatSetType(), MatProductSymbolic()
@*/
PetscErrorCode MatProductNumeric(Mat mat)
{
  PetscErrorCode ierr;
  PetscLogEvent  eventtype = -1;
  PetscBool      missing = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  MatCheckProduct(mat,1);
  switch (mat->product->type) {
  case MATPRODUCT_AB:
    eventtype = MAT_MatMultNumeric;
    break;
  case MATPRODUCT_AtB:
    eventtype = MAT_TransposeMatMultNumeric;
    break;
  case MATPRODUCT_ABt:
    eventtype = MAT_MatTransposeMultNumeric;
    break;
  case MATPRODUCT_PtAP:
    eventtype = MAT_PtAPNumeric;
    break;
  case MATPRODUCT_RARt:
    eventtype = MAT_RARtNumeric;
    break;
  case MATPRODUCT_ABC:
    eventtype = MAT_MatMatMultNumeric;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"ProductType %s is not supported",MatProductTypes[mat->product->type]);
  }

  if (mat->ops->productnumeric) {
    ierr = PetscLogEventBegin(eventtype,mat,0,0,0);CHKERRQ(ierr);
    ierr = (*mat->ops->productnumeric)(mat);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(eventtype,mat,0,0,0);CHKERRQ(ierr);
  } else missing = PETSC_TRUE;

  if (missing || !mat->product) {
    char errstr[256];

    if (mat->product->type == MATPRODUCT_ABC) {
      ierr = PetscSNPrintf(errstr,256,"%s with A %s, B %s, C %s",MatProductTypes[mat->product->type],((PetscObject)mat->product->A)->type_name,((PetscObject)mat->product->B)->type_name,((PetscObject)mat->product->C)->type_name);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(errstr,256,"%s with A %s, B %s",MatProductTypes[mat->product->type],((PetscObject)mat->product->A)->type_name,((PetscObject)mat->product->B)->type_name);CHKERRQ(ierr);
    }
    if (missing) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Unspecified numeric phase for product %s",errstr);
    if (!mat->product) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing struct after symbolic phase for product %s",errstr);
  }

  if (mat->product->clear) {
    ierr = MatProductClear(mat);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */
/* these are basic implementations relying on the old function pointers
 * they are dangerous and should be removed in the future */
PetscErrorCode MatProductSymbolic_AB(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (!mat->ops->matmultsymbolic) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
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
  if (!mat->ops->transposematmultsymbolic) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
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
  if (!mat->ops->mattransposemultsymbolic) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
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
  if (!mat->ops->matmatmultsymbolic) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing symbolic implementation of product %s",MatProductTypes[product->type]);
  ierr = (*mat->ops->matmatmultsymbolic)(A,B,C,product->fill,mat);CHKERRQ(ierr);
  mat->ops->productnumeric = MatProductNumeric_ABC;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */

/*@
   MatProductSymbolic - Perform the symbolic portion of a matrix product, this creates a data structure for use with the numerical produce.

   Collective on Mat

   Input/Output Parameter:
.  mat - the matrix to hold a product

   Level: intermediate

   Notes: MatProductSetFromOptions() must have been called on mat before calling this function

.seealso: MatProductCreate(), MatProductCreateWithMat(), MatProductSetFromOptions(), MatProductNumeric(), MatProductSetType(), MatProductSetAlgorithm()
@*/
PetscErrorCode MatProductSymbolic(Mat mat)
{
  PetscErrorCode ierr;
  PetscLogEvent  eventtype = -1;
  PetscBool      missing = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  MatCheckProduct(mat,1);
  if (mat->product->data) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Cannot run symbolic phase. Product data not empty");
  switch (mat->product->type) {
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
  default: SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"ProductType %s is not supported",MatProductTypes[mat->product->type]);
  }
  mat->ops->productnumeric = NULL;
  if (mat->ops->productsymbolic) {
    ierr = PetscLogEventBegin(eventtype,mat,0,0,0);CHKERRQ(ierr);
    ierr = (*mat->ops->productsymbolic)(mat);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(eventtype,mat,0,0,0);CHKERRQ(ierr);
  } else missing = PETSC_TRUE;

  if (missing || !mat->product || !mat->ops->productnumeric) {
    char errstr[256];

    if (mat->product->type == MATPRODUCT_ABC) {
      ierr = PetscSNPrintf(errstr,256,"%s with A %s, B %s, C %s",MatProductTypes[mat->product->type],((PetscObject)mat->product->A)->type_name,((PetscObject)mat->product->B)->type_name,((PetscObject)mat->product->C)->type_name);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(errstr,256,"%s with A %s, B %s",MatProductTypes[mat->product->type],((PetscObject)mat->product->A)->type_name,((PetscObject)mat->product->B)->type_name);CHKERRQ(ierr);
    }
    if (missing) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Unspecified symbolic phase for product %s. Call MatProductSetFromOptions() first",errstr);
    if (!mat->ops->productnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Unspecified numeric phase for product %s",errstr);
    if (!mat->product) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing struct after symbolic phase for product %s",errstr);
  }
  PetscFunctionReturn(0);
}

/*@
   MatProductSetFill - Set an expected fill of the matrix product.

   Collective on Mat

   Input Parameters:
+  mat - the matrix product
-  fill - expected fill as ratio of nnz(mat)/(nnz(A) + nnz(B) + nnz(C)); use PETSC_DEFAULT if you do not have a good estimate. If the product is a dense matrix, this is irrelevant.

   Level: intermediate

.seealso: MatProductSetType(), MatProductSetAlgorithm(), MatProductCreate()
@*/
PetscErrorCode MatProductSetFill(Mat mat,PetscReal fill)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  MatCheckProduct(mat,1);
  if (fill == PETSC_DEFAULT || fill == PETSC_DECIDE) mat->product->fill = 2.0;
  else mat->product->fill = fill;
  PetscFunctionReturn(0);
}

/*@
   MatProductSetAlgorithm - Requests a particular algorithm for a matrix product implementation.

   Collective on Mat

   Input Parameters:
+  mat - the matrix product
-  alg - particular implementation algorithm of the matrix product, e.g., MATPRODUCTALGORITHMDEFAULT.

   Options Database Key:
.  -mat_product_algorithm <algorithm> - Sets the algorithm; use -help for a list
    of available algorithms (for instance, scalable, outerproduct, etc.)

   Level: intermediate

.seealso: MatProductSetType(), MatProductSetFill(), MatProductCreate(), MatProductAlgorithm
@*/
PetscErrorCode MatProductSetAlgorithm(Mat mat,MatProductAlgorithm alg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  MatCheckProduct(mat,1);
  ierr = PetscFree(mat->product->alg);CHKERRQ(ierr);
  ierr = PetscStrallocpy(alg,&mat->product->alg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatProductSetType - Sets a particular matrix product type

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  productype   - matrix product type, e.g., MATPRODUCT_AB,MATPRODUCT_AtB,MATPRODUCT_ABt,MATPRODUCT_PtAP,MATPRODUCT_RARt,MATPRODUCT_ABC.

   Level: intermediate

.seealso: MatProductCreate(), MatProductType
@*/
PetscErrorCode MatProductSetType(Mat mat,MatProductType productype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  MatCheckProduct(mat,1);
  PetscValidLogicalCollectiveEnum(mat,productype,2);
  if (productype != mat->product->type) {
    if (mat->product->destroy) {
      ierr = (*mat->product->destroy)(mat->product->data);CHKERRQ(ierr);
    }
    mat->product->destroy = NULL;
    mat->product->data = NULL;
    mat->ops->productsymbolic = NULL;
    mat->ops->productnumeric  = NULL;
  }
  mat->product->type = productype;
  PetscFunctionReturn(0);
}

/*@
   MatProductClear - Clears matrix product internal structure.

   Collective on Mat

   Input Parameters:
.  mat - the product matrix

   Level: intermediate

   Notes: this function should be called to remove any intermediate data used by the product
          After having called this function, MatProduct operations can no longer be used on mat
@*/
PetscErrorCode MatProductClear(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Product    *product = mat->product;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (product) {
    ierr = MatDestroy(&product->A);CHKERRQ(ierr);
    ierr = MatDestroy(&product->B);CHKERRQ(ierr);
    ierr = MatDestroy(&product->C);CHKERRQ(ierr);
    ierr = PetscFree(product->alg);CHKERRQ(ierr);
    ierr = MatDestroy(&product->Dwork);CHKERRQ(ierr);
    if (product->destroy) {
      ierr = (*product->destroy)(product->data);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(mat->product);CHKERRQ(ierr);
  mat->ops->productsymbolic = NULL;
  mat->ops->productnumeric = NULL;
  PetscFunctionReturn(0);
}

/* Create a supporting struct and attach it to the matrix product */
PetscErrorCode MatProductCreate_Private(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;
  Mat_Product    *product=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(D,MAT_CLASSID,4);
  if (D->product) SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"Product already present");
  ierr = PetscNewLog(D,&product);CHKERRQ(ierr);
  product->A        = A;
  product->B        = B;
  product->C        = C;
  product->type     = MATPRODUCT_UNSPECIFIED;
  product->Dwork    = NULL;
  product->api_user = PETSC_FALSE;
  product->clear    = PETSC_FALSE;
  D->product        = product;

  ierr = MatProductSetAlgorithm(D,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(D,PETSC_DEFAULT);CHKERRQ(ierr);

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

   Notes:
     Any product data attached to D will be cleared

   Level: intermediate

.seealso: MatProductCreate(), MatProductClear()
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
  ierr = MatProductClear(D);CHKERRQ(ierr);
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
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(B,2);
  if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix A");
  if (B->factortype) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix B");

  if (C) {
    PetscValidHeaderSpecific(C,MAT_CLASSID,3);
    PetscValidType(C,3);
    if (C->factortype) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix C");
  }

  PetscValidPointer(D,4);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),D);CHKERRQ(ierr);
  ierr = MatProductCreate_Private(A,B,C,*D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   These are safe basic implementations of ABC, RARt and PtAP
   that do not rely on mat->ops->matmatop function pointers.
   They only use the MatProduct API and are currently used by
   cuSPARSE and KOKKOS-KERNELS backends
*/
typedef struct {
  Mat BC;
  Mat ABC;
} MatMatMatPrivate;

static PetscErrorCode MatDestroy_MatMatMatPrivate(void *data)
{
  PetscErrorCode   ierr;
  MatMatMatPrivate *mmdata = (MatMatMatPrivate *)data;

  PetscFunctionBegin;
  ierr = MatDestroy(&mmdata->BC);CHKERRQ(ierr);
  ierr = MatDestroy(&mmdata->ABC);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_ABC_Basic(Mat mat)
{
  PetscErrorCode   ierr;
  Mat_Product      *product = mat->product;
  MatMatMatPrivate *mmabc;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  if (!mat->product->data) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Product data empty");
  mmabc = (MatMatMatPrivate *)mat->product->data;
  if (!mmabc->BC->ops->productnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric stage");
  /* use function pointer directly to prevent logging */
  ierr = (*mmabc->BC->ops->productnumeric)(mmabc->BC);CHKERRQ(ierr);
  /* swap ABC product stuff with that of ABC for the numeric phase on mat */
  mat->product = mmabc->ABC->product;
  mat->ops->productnumeric = mmabc->ABC->ops->productnumeric;
  if (!mat->ops->productnumeric) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Missing numeric stage");
  /* use function pointer directly to prevent logging */
  ierr = (*mat->ops->productnumeric)(mat);CHKERRQ(ierr);
  mat->ops->productnumeric = MatProductNumeric_ABC_Basic;
  mat->product = product;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_ABC_Basic(Mat mat)
{
  PetscErrorCode   ierr;
  Mat_Product      *product = mat->product;
  Mat              A, B ,C;
  MatProductType   p1,p2;
  MatMatMatPrivate *mmabc;
  const char       *prefix;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  if (mat->product->data) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Product data not empty");
  ierr = MatGetOptionsPrefix(mat,&prefix);CHKERRQ(ierr);
  ierr = PetscNew(&mmabc);CHKERRQ(ierr);
  product->data    = mmabc;
  product->destroy = MatDestroy_MatMatMatPrivate;
  switch (product->type) {
  case MATPRODUCT_PtAP:
    p1 = MATPRODUCT_AB;
    p2 = MATPRODUCT_AtB;
    A = product->B;
    B = product->A;
    C = product->B;
    break;
  case MATPRODUCT_RARt:
    p1 = MATPRODUCT_ABt;
    p2 = MATPRODUCT_AB;
    A = product->B;
    B = product->A;
    C = product->B;
    break;
  case MATPRODUCT_ABC:
    p1 = MATPRODUCT_AB;
    p2 = MATPRODUCT_AB;
    A = product->A;
    B = product->B;
    C = product->C;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Not for ProductType %s",MatProductTypes[product->type]);
  }
  ierr = MatProductCreate(B,C,NULL,&mmabc->BC);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(mmabc->BC,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(mmabc->BC,"P1_");CHKERRQ(ierr);
  ierr = MatProductSetType(mmabc->BC,p1);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(mmabc->BC,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(mmabc->BC,product->fill);CHKERRQ(ierr);
  mmabc->BC->product->api_user = product->api_user;
  ierr = MatProductSetFromOptions(mmabc->BC);CHKERRQ(ierr);
  if (!mmabc->BC->ops->productsymbolic) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Symbolic ProductType %s not supported with %s and %s",MatProductTypes[p1],((PetscObject)B)->type_name,((PetscObject)C)->type_name);
  /* use function pointer directly to prevent logging */
  ierr = (*mmabc->BC->ops->productsymbolic)(mmabc->BC);CHKERRQ(ierr);

  ierr = MatProductCreate(A,mmabc->BC,NULL,&mmabc->ABC);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(mmabc->ABC,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(mmabc->ABC,"P2_");CHKERRQ(ierr);
  ierr = MatProductSetType(mmabc->ABC,p2);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(mmabc->ABC,MATPRODUCTALGORITHMDEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(mmabc->ABC,product->fill);CHKERRQ(ierr);
  mmabc->ABC->product->api_user = product->api_user;
  ierr = MatProductSetFromOptions(mmabc->ABC);CHKERRQ(ierr);
  if (!mmabc->ABC->ops->productsymbolic) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Symbolic ProductType %s not supported with %s and %s",MatProductTypes[p2],((PetscObject)A)->type_name,((PetscObject)mmabc->BC)->type_name);
  /* swap ABC product stuff with that of ABC for the symbolic phase on mat */
  mat->product = mmabc->ABC->product;
  mat->ops->productsymbolic = mmabc->ABC->ops->productsymbolic;
  /* use function pointer directly to prevent logging */
  ierr = (*mat->ops->productsymbolic)(mat);CHKERRQ(ierr);
  mmabc->ABC->ops->productnumeric = mat->ops->productnumeric;
  mat->ops->productsymbolic       = MatProductSymbolic_ABC_Basic;
  mat->ops->productnumeric        = MatProductNumeric_ABC_Basic;
  mat->product                    = product;
  PetscFunctionReturn(0);
}

/*@
   MatProductGetType - Returns the type of the MatProduct.

   Not collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  mtype - the MatProduct type

   Level: intermediate

.seealso: MatProductCreateWithMat(), MatProductSetType(), MatProductCreate()
@*/
PetscErrorCode MatProductGetType(Mat mat, MatProductType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(mtype,2);
  *mtype = MATPRODUCT_UNSPECIFIED;
  if (mat->product) *mtype = mat->product->type;
  PetscFunctionReturn(0);
}

/*@
   MatProductGetMats - Returns the matrices associated with the MatProduct.

   Not collective

   Input Parameter:
.  mat - the product matrix

   Output Parameters:
+  A - the first matrix
.  B - the second matrix
-  C - the third matrix (optional)

   Level: intermediate

.seealso: MatProductCreateWithMat(), MatProductSetType(), MatProductSetAlgorithm()
@*/
PetscErrorCode MatProductGetMats(Mat mat, Mat *A, Mat *B, Mat *C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (A) *A = mat->product ? mat->product->A : NULL;
  if (B) *B = mat->product ? mat->product->B : NULL;
  if (C) *C = mat->product ? mat->product->C : NULL;
  PetscFunctionReturn(0);
}
