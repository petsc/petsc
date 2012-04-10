
#include <petsc-private/matimpl.h>          /*I "petscmat.h" I*/
#include <petscksp.h>                              /*I "petscksp.h" I*/

typedef struct {
  Mat A,Ap,B,C,D;
  KSP ksp;
  Vec work1,work2;
} Mat_SchurComplement;


#undef __FUNCT__  
#define __FUNCT__ "MatView_SchurComplement"
PetscErrorCode MatView_SchurComplement(Mat N,PetscViewer viewer)
{
  Mat_SchurComplement  *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Schur complement A11 - A10 inv(A00) A01\n");CHKERRQ(ierr);
  if (Na->D) {
    ierr = PetscViewerASCIIPrintf(viewer,"A11\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = MatView(Na->D,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"A11 = 0\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"A10\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = MatView(Na->C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"KSP of A00\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = KSPView(Na->ksp,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"A01\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = MatView(Na->B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
           A11 - A10 ksp(A00,Ap00)  A01 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SchurComplement"
PetscErrorCode MatMult_SchurComplement(Mat N,Vec x,Vec y)
{
  Mat_SchurComplement  *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (!Na->work1) {ierr = MatGetVecs(Na->A,&Na->work1,PETSC_NULL);CHKERRQ(ierr);}
  if (!Na->work2) {ierr = MatGetVecs(Na->A,&Na->work2,PETSC_NULL);CHKERRQ(ierr);}
  ierr = MatMult(Na->B,x,Na->work1);CHKERRQ(ierr);
  ierr = KSPSolve(Na->ksp,Na->work1,Na->work2);CHKERRQ(ierr);
  ierr = MatMult(Na->C,Na->work2,y);CHKERRQ(ierr);
  ierr = VecScale(y,-1.0);CHKERRQ(ierr);
  if (Na->D) {
    ierr = MatMultAdd(Na->D,x,y,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetFromOptions_SchurComplement"
PetscErrorCode MatSetFromOptions_SchurComplement(Mat N)
{
  Mat_SchurComplement  *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions(Na->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SchurComplement"
PetscErrorCode MatDestroy_SchurComplement(Mat N)
{
  Mat_SchurComplement  *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->Ap);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->B);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->C);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->D);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->work1);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->work2);CHKERRQ(ierr);
  ierr = KSPDestroy(&Na->ksp);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSchurComplement"
/*@
      MatCreateSchurComplement - Creates a new matrix object that behaves like the Schur complement of a matrix

   Collective on Mat

   Input Parameter:
.   A00,A01,A10,A11  - the four parts of the original matrix (A00 is optional)

   Output Parameter:
.   N - the matrix that the Schur complement A11 - A10 ksp(A00) A01

   Level: intermediate

   Notes: The Schur complement is NOT actually formed! Rather this 
          object performs the matrix-vector product by using the the formula for
          the Schur complement and a KSP solver to approximate the action of inv(A)

          All four matrices must have the same MPI communicator

          A00 and  A11 must be square matrices

.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatSchurComplementUpdate(), MatCreateTranspose(), MatGetSchurComplement()

@*/
PetscErrorCode  MatCreateSchurComplement(Mat A00,Mat Ap00,Mat A01,Mat A10,Mat A11,Mat *N)
{
  PetscErrorCode       ierr;
  PetscInt             m,n;
  Mat_SchurComplement  *Na;  

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A00,MAT_CLASSID,1);
  PetscValidHeaderSpecific(Ap00,MAT_CLASSID,2);
  PetscValidHeaderSpecific(A01,MAT_CLASSID,3);
  PetscValidHeaderSpecific(A10,MAT_CLASSID,4);
  PetscCheckSameComm(A00,1,Ap00,2);
  PetscCheckSameComm(A00,1,A01,3);
  PetscCheckSameComm(A00,1,A10,4);
  if (A00->rmap->n != A00->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A00 %D do not equal local columns %D",A00->rmap->n,A00->cmap->n);
  if (A00->rmap->n != Ap00->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A00 %D do not equal local rows of Ap00 %D",A00->rmap->n,Ap00->rmap->n);
  if (Ap00->rmap->n != Ap00->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of Ap00 %D do not equal local columns %D",Ap00->rmap->n,Ap00->cmap->n);
  if (A00->cmap->n != A01->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of A00 %D do not equal local rows of A01 %D",A00->cmap->n,A01->rmap->n);
  if (A10->cmap->n != A00->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of A10 %D do not equal local rows of A00 %D",A10->cmap->n,A00->rmap->n);
  if (A11) {
    PetscValidHeaderSpecific(A11,MAT_CLASSID,5);
    PetscCheckSameComm(A00,1,A11,5);
    if (A11->rmap->n != A11->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A11 %D do not equal local columns %D",A11->rmap->n,A11->cmap->n);
    if (A10->rmap->n != A11->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A10 %D do not equal local rows A11 %D",A10->rmap->n,A11->rmap->n);
  }

  ierr = MatGetLocalSize(A01,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A10,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)A00)->comm,N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATSCHURCOMPLEMENT);CHKERRQ(ierr);
  
  ierr      = PetscNewLog(*N,Mat_SchurComplement,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  ierr      = PetscObjectReference((PetscObject)A00);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)Ap00);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)A01);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)A10);CHKERRQ(ierr);
  Na->A     = A00;
  Na->Ap    = Ap00;
  Na->B     = A01;
  Na->C     = A10;
  Na->D     = A11;
  if (A11) {
    ierr = PetscObjectReference((PetscObject)A11);CHKERRQ(ierr);
  }

  (*N)->ops->destroy        = MatDestroy_SchurComplement;
  (*N)->ops->view           = MatView_SchurComplement;
  (*N)->ops->mult           = MatMult_SchurComplement;
  (*N)->ops->setfromoptions = MatSetFromOptions_SchurComplement;
  (*N)->assembled           = PETSC_TRUE;
  (*N)->preallocated        = PETSC_TRUE;

  ierr = PetscLayoutSetUp((*N)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*N)->cmap);CHKERRQ(ierr);

  ierr = KSPCreate(((PetscObject)A00)->comm,&Na->ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(Na->ksp,A00,Ap00,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSchurComplementGetKSP"
/*@
      MatSchurComplementGetKSP - Creates gets the KSP object that is used in the Schur complement matrix

   Not Collective

   Input Parameter:
.   A - matrix created with MatCreateSchurComplement()

   Output Parameter:
.   ksp - the linear solver object

   Options Database:
-     -fieldsplit_0_XXX sets KSP and PC options for the A block solver inside the Schur complement

   Level: intermediate

   Notes: 
.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatCreateSchurComplement()

@*/
PetscErrorCode  MatSchurComplementGetKSP(Mat A,KSP *ksp)
{
  Mat_SchurComplement  *Na;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(ksp,2);
  Na = (Mat_SchurComplement*)A->data;
  *ksp = Na->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSchurComplementUpdate"
/*@
      MatSchurComplementUpdate - Updates the Schur complement matrix object with new submatrices

   Collective on Mat

   Input Parameters:
+   N - the matrix obtained with MatCreateSchurComplement()
.   A,B,C,D  - the four parts of the original matrix (D is optional)
-   str - either SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER

 
   Level: intermediate

   Notes: All four matrices must have the same MPI communicator

          A and  D must be square matrices

          All of the matrices provided must have the same sizes as was used with MatCreateSchurComplement()
          though they need not be the same matrices

.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatCreateSchurComplement()

@*/
PetscErrorCode  MatSchurComplementUpdate(Mat N,Mat A,Mat Ap,Mat B,Mat C,Mat D,MatStructure str)
{
  PetscErrorCode       ierr;
  Mat_SchurComplement  *Na = (Mat_SchurComplement*)N->data;  

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidHeaderSpecific(C,MAT_CLASSID,3);
  PetscCheckSameComm(A,1,Ap,2);
  PetscCheckSameComm(A,1,B,3);
  PetscCheckSameComm(A,1,C,4);
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A %D do not equal local columns %D",A->rmap->n,A->cmap->n);
  if (A->rmap->n != Ap->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A %D do not equal local rows of Ap %D",A->rmap->n,Ap->rmap->n);
  if (Ap->rmap->n != Ap->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of Ap %D do not equal local columns %D",Ap->rmap->n,Ap->cmap->n);
  if (A->cmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of A %D do not equal local rows of B %D",A->cmap->n,B->rmap->n);
  if (C->cmap->n != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of C %D do not equal local rows of A %D",C->cmap->n,A->rmap->n);
  if (D) {
    PetscValidHeaderSpecific(D,MAT_CLASSID,5);
    PetscCheckSameComm(A,1,D,5);
    if (D->rmap->n != D->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of D %D do not equal local columns %D",D->rmap->n,D->cmap->n);
    if (C->rmap->n != D->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of C %D do not equal local rows D %D",C->rmap->n,D->rmap->n);
  }

  ierr   = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr   = PetscObjectReference((PetscObject)Ap);CHKERRQ(ierr);
  ierr   = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
  ierr   = PetscObjectReference((PetscObject)C);CHKERRQ(ierr);
  if (D) {
    ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr);
  }

  ierr   = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr   = MatDestroy(&Na->Ap);CHKERRQ(ierr);
  ierr   = MatDestroy(&Na->B);CHKERRQ(ierr);
  ierr   = MatDestroy(&Na->C);CHKERRQ(ierr);
  ierr   = MatDestroy(&Na->D);CHKERRQ(ierr);

  Na->A  = A;
  Na->Ap = Ap;
  Na->B  = B;
  Na->C  = C;
  Na->D  = D;

  ierr = KSPSetOperators(Na->ksp,A,Ap,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSchurComplementGetSubmatrices"
/*@C
  MatSchurComplementGetSubmatrices - Get the individual submatrices in the Schur complement

  Collective on Mat

  Input Parameters:
+ N - the matrix obtained with MatCreateSchurComplement()
- A,B,C,D  - the four parts of the original matrix (D is optional)

  Note:
  D is optional, and thus can be PETSC_NULL

  Level: intermediate

.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatCreateSchurComplement(), MatSchurComplementUpdate()
@*/
PetscErrorCode  MatSchurComplementGetSubmatrices(Mat N,Mat *A,Mat *Ap,Mat *B,Mat *C,Mat *D)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *) N->data;  
  PetscErrorCode       ierr;
  PetscBool            flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(N,MAT_CLASSID,1);
  ierr = PetscTypeCompare((PetscObject)N,MATSCHURCOMPLEMENT,&flg);CHKERRQ(ierr);
  if (flg) {
    if (A)  *A  = Na->A;
    if (Ap) *Ap = Na->Ap;
    if (B)  *B  = Na->B;
    if (C)  *C  = Na->C;
    if (D)  *D  = Na->D;
  } else {
    if (A)  *A  = 0;
    if (Ap) *Ap = 0;
    if (B)  *B  = 0;
    if (C)  *C  = 0;
    if (D)  *D  = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSchurComplement_Basic"
/* Developer Notes: This should be implemented with a MatCreate_SchurComplement() as that is the standard design for new Mat classes. */
PetscErrorCode MatGetSchurComplement_Basic(Mat mat,IS isrow0,IS iscol0,IS isrow1,IS iscol1,MatReuse mreuse,Mat *newmat,MatReuse preuse,Mat *newpmat)
{
  PetscErrorCode ierr;
  Mat A=0,Ap=0,B=0,C=0,D=0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow0,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol0,IS_CLASSID,3);
  PetscValidHeaderSpecific(isrow1,IS_CLASSID,4);
  PetscValidHeaderSpecific(iscol1,IS_CLASSID,5);
  if (mreuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newmat,MAT_CLASSID,7);
  if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newpmat,MAT_CLASSID,9);
  PetscValidType(mat,1);
  if (mat->factortype) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  if (mreuse != MAT_IGNORE_MATRIX) {
    /* Use MatSchurComplement */
    if (mreuse == MAT_REUSE_MATRIX) {
      ierr = MatSchurComplementGetSubmatrices(*newmat,&A,&Ap,&B,&C,&D);CHKERRQ(ierr);
      if (!A || !Ap || !B || !C) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Attempting to reuse matrix but Schur complement matrices unset");
      if (A != Ap) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Preconditioning matrix does not match operator");
      ierr = MatDestroy(&Ap);CHKERRQ(ierr); /* get rid of extra reference */
    }
    ierr = MatGetSubMatrix(mat,isrow0,iscol0,mreuse,&A);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(mat,isrow0,iscol1,mreuse,&B);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(mat,isrow1,iscol0,mreuse,&C);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(mat,isrow1,iscol1,mreuse,&D);CHKERRQ(ierr);
    switch (mreuse) {
    case MAT_INITIAL_MATRIX:
      ierr = MatCreateSchurComplement(A,A,B,C,D,newmat);CHKERRQ(ierr);
      break;
    case MAT_REUSE_MATRIX:
      ierr = MatSchurComplementUpdate(*newmat,A,A,B,C,D,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Unrecognized value of mreuse");
    }
  }
  if (preuse != MAT_IGNORE_MATRIX) {
    /* Use the diagonal part of A to form D - C inv(diag(A)) B */
    Mat Ad,AdB,S;
    Vec diag;
    PetscInt i,m,n,mstart,mend;
    PetscScalar *x;

    /* We could compose these with newpmat so that the matrices can be reused. */
    if (!A) {ierr = MatGetSubMatrix(mat,isrow0,iscol0,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);}
    if (!B) {ierr = MatGetSubMatrix(mat,isrow0,iscol1,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);}
    if (!C) {ierr = MatGetSubMatrix(mat,isrow1,iscol0,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);}
    if (!D) {ierr = MatGetSubMatrix(mat,isrow1,iscol1,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);}

    ierr = MatGetVecs(A,&diag,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetDiagonal(A,diag);CHKERRQ(ierr);
    ierr = VecReciprocal(diag);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    /* We need to compute S = D - C inv(diag(A)) B.  For row-oriented formats, it is easy to scale the rows of B and
     * for column-oriented formats the columns of C can be scaled.  Would skip creating a silly diagonal matrix. */
    ierr = MatCreate(((PetscObject)A)->comm,&Ad);CHKERRQ(ierr);
    ierr = MatSetSizes(Ad,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(Ad,((PetscObject)mat)->prefix);CHKERRQ(ierr);
    ierr = MatAppendOptionsPrefix(Ad,"diag_");CHKERRQ(ierr);
    ierr = MatSetFromOptions(Ad);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Ad,1,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(Ad,1,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Ad,&mstart,&mend);CHKERRQ(ierr);
    ierr = VecGetArray(diag,&x);CHKERRQ(ierr);
    for (i=mstart; i<mend; i++) {
      ierr = MatSetValue(Ad,i,i,x[i-mstart],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(diag,&x);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Ad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecDestroy(&diag);CHKERRQ(ierr);

    ierr = MatMatMult(Ad,B,MAT_INITIAL_MATRIX,1,&AdB);CHKERRQ(ierr);
    S = (preuse == MAT_REUSE_MATRIX) ? *newpmat : (Mat)0;
    ierr = MatMatMult(C,AdB,preuse,PETSC_DEFAULT,&S);CHKERRQ(ierr);
    ierr = MatAYPX(S,-1,D,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    *newpmat = S;
    ierr = MatDestroy(&Ad);CHKERRQ(ierr);
    ierr = MatDestroy(&AdB);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSchurComplement"
/*@
    MatGetSchurComplement - Obtain the Schur complement from eliminating part of the matrix in another part.

    Collective on Mat

    Input Parameters:
+   mat - Matrix in which the complement is to be taken
.   isrow0 - rows to eliminate
.   iscol0 - columns to eliminate, (isrow0,iscol0) should be square and nonsingular
.   isrow1 - rows in which the Schur complement is formed
.   iscol1 - columns in which the Schur complement is formed
.   mreuse - MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX, use MAT_IGNORE_MATRIX to put nothing in newmat
-   preuse - MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX, use MAT_IGNORE_MATRIX to put nothing in newpmat

    Output Parameters:
+   newmat - exact Schur complement, often of type MATSCHURCOMPLEMENT which is difficult to use for preconditioning
-   newpmat - approximate Schur complement suitable for preconditioning

    Note:
    Since the real Schur complement is usually dense, providing a good approximation to newpmat usually requires
    application-specific information.  The default for assembled matrices is to use the diagonal of the (0,0) block
    which will rarely produce a scalable algorithm.

    Sometimes users would like to provide problem-specific data in the Schur complement, usually only for special row
    and column index sets.  In that case, the user should call PetscObjectComposeFunctionDynamic() to set
    "MatNestGetSubMat_C" to their function.  If their function needs to fall back to the default implementation, it
    should call MatGetSchurComplement_Basic().

    Level: advanced

    Concepts: matrices^submatrices

.seealso: MatGetSubMatrix(), PCFIELDSPLIT, MatCreateSchurComplement()
@*/
PetscErrorCode  MatGetSchurComplement(Mat mat,IS isrow0,IS iscol0,IS isrow1,IS iscol1,MatReuse mreuse,Mat *newmat,MatReuse preuse,Mat *newpmat)
{
  PetscErrorCode ierr,(*f)(Mat,IS,IS,IS,IS,MatReuse,Mat*,MatReuse,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow0,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol0,IS_CLASSID,3);
  PetscValidHeaderSpecific(isrow1,IS_CLASSID,4);
  PetscValidHeaderSpecific(iscol1,IS_CLASSID,5);
  if (mreuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newmat,MAT_CLASSID,7);
  if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newpmat,MAT_CLASSID,9);
  PetscValidType(mat,1);
  if (mat->factortype) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatGetSchurComplement_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,isrow0,iscol0,isrow1,iscol1,mreuse,newmat,preuse,newpmat);CHKERRQ(ierr);
  } else {
    ierr = MatGetSchurComplement_Basic(mat,isrow0,iscol0,isrow1,iscol1,mreuse,newmat,preuse,newpmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
