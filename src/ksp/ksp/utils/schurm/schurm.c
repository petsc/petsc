#include <../src/ksp/ksp/utils/schurm/schurm.h> /*I "petscksp.h" I*/

const char *const MatSchurComplementAinvTypes[] = {"DIAG","LUMP","BLOCKDIAG","MatSchurComplementAinvType","MAT_SCHUR_COMPLEMENT_AINV_",NULL};

PetscErrorCode MatCreateVecs_SchurComplement(Mat N,Vec *right,Vec *left)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (Na->D) {
    ierr = MatCreateVecs(Na->D,right,left);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (right) {
    ierr = MatCreateVecs(Na->B,right,NULL);CHKERRQ(ierr);
  }
  if (left) {
    ierr = MatCreateVecs(Na->C,NULL,left);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SchurComplement(Mat N,PetscViewer viewer)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode      ierr;

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
           A11^T - A01^T ksptrans(A00,Ap00) A10^T
*/
PetscErrorCode MatMultTranspose_SchurComplement(Mat N,Vec x,Vec y)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (!Na->work1) {ierr = MatCreateVecs(Na->A,&Na->work1,NULL);CHKERRQ(ierr);}
  if (!Na->work2) {ierr = MatCreateVecs(Na->A,&Na->work2,NULL);CHKERRQ(ierr);}
  ierr = MatMultTranspose(Na->C,x,Na->work1);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(Na->ksp,Na->work1,Na->work2);CHKERRQ(ierr);
  ierr = MatMultTranspose(Na->B,Na->work2,y);CHKERRQ(ierr);
  ierr = VecScale(y,-1.0);CHKERRQ(ierr);
  if (Na->D) {
    ierr = MatMultTransposeAdd(Na->D,x,y,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
           A11 - A10 ksp(A00,Ap00) A01
*/
PetscErrorCode MatMult_SchurComplement(Mat N,Vec x,Vec y)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (!Na->work1) {ierr = MatCreateVecs(Na->A,&Na->work1,NULL);CHKERRQ(ierr);}
  if (!Na->work2) {ierr = MatCreateVecs(Na->A,&Na->work2,NULL);CHKERRQ(ierr);}
  ierr = MatMult(Na->B,x,Na->work1);CHKERRQ(ierr);
  ierr = KSPSolve(Na->ksp,Na->work1,Na->work2);CHKERRQ(ierr);
  ierr = MatMult(Na->C,Na->work2,y);CHKERRQ(ierr);
  ierr = VecScale(y,-1.0);CHKERRQ(ierr);
  if (Na->D) {
    ierr = MatMultAdd(Na->D,x,y,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
           A11 - A10 ksp(A00,Ap00) A01
*/
PetscErrorCode MatMultAdd_SchurComplement(Mat N,Vec x,Vec y,Vec z)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (!Na->work1) {ierr = MatCreateVecs(Na->A,&Na->work1,NULL);CHKERRQ(ierr);}
  if (!Na->work2) {ierr = MatCreateVecs(Na->A,&Na->work2,NULL);CHKERRQ(ierr);}
  ierr = MatMult(Na->B,x,Na->work1);CHKERRQ(ierr);
  ierr = KSPSolve(Na->ksp,Na->work1,Na->work2);CHKERRQ(ierr);
  if (y == z) {
    ierr = VecScale(Na->work2,-1.0);CHKERRQ(ierr);
    ierr = MatMultAdd(Na->C,Na->work2,z,z);CHKERRQ(ierr);
  } else {
    ierr = MatMult(Na->C,Na->work2,z);CHKERRQ(ierr);
    ierr = VecAYPX(z,-1.0,y);CHKERRQ(ierr);
  }
  if (Na->D) {
    ierr = MatMultAdd(Na->D,x,z,z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_SchurComplement(PetscOptionItems *PetscOptionsObject,Mat N)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MatSchurComplementOptions");CHKERRQ(ierr);
  Na->ainvtype = MAT_SCHUR_COMPLEMENT_AINV_DIAG;
  ierr = PetscOptionsEnum("-mat_schur_complement_ainv_type","Type of approximation for inv(A00) used when assembling Sp = A11 - A10 inv(A00) A01","MatSchurComplementSetAinvType",MatSchurComplementAinvTypes,(PetscEnum)Na->ainvtype,(PetscEnum*)&Na->ainvtype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(Na->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SchurComplement(Mat N)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode      ierr;

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

/*@
      MatCreateSchurComplement - Creates a new matrix object that behaves like the Schur complement of a matrix

   Collective on A00

   Input Parameters:
+   A00,A01,A10,A11  - the four parts of the original matrix A = [A00 A01; A10 A11] (A11 is optional)
-   Ap00             - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A^{-1}

   Output Parameter:
.   S - the matrix that the Schur complement S = A11 - A10 ksp(A00,Ap00) A01

   Level: intermediate

   Notes:
    The Schur complement is NOT actually formed! Rather, this
          object performs the matrix-vector product by using formula S = A11 - A10 A^{-1} A01
          for Schur complement S and a KSP solver to approximate the action of A^{-1}.

          All four matrices must have the same MPI communicator.

          A00 and  A11 must be square matrices.

          MatGetSchurComplement() takes as arguments the index sets for the submatrices and returns both the virtual Schur complement (what this returns) plus
          a sparse approximation to the true Schur complement (useful for building a preconditioner for the Schur complement).

          MatSchurComplementGetPmat() can be called on the output of this function to generate an explicit approximation to the Schur complement.

    Developer Notes:
    The API that includes MatGetSchurComplement(), MatCreateSchurComplement(), MatSchurComplementGetPmat() should be refactored to
    remove redundancy and be clearer and simplier.


.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatSchurComplementUpdateSubMatrices(), MatCreateTranspose(), MatGetSchurComplement(),
          MatSchurComplementGetPmat()

@*/
PetscErrorCode  MatCreateSchurComplement(Mat A00,Mat Ap00,Mat A01,Mat A10,Mat A11,Mat *S)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A00),S);CHKERRQ(ierr);
  ierr = MatSetType(*S,MATSCHURCOMPLEMENT);CHKERRQ(ierr);
  ierr = MatSchurComplementSetSubMatrices(*S,A00,Ap00,A01,A10,A11);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
      MatSchurComplementSetSubMatrices - Sets the matrices that define the Schur complement

   Collective on S

   Input Parameter:
+   S                - matrix obtained with MatCreateSchurComplement (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01
.   A00,A01,A10,A11  - the four parts of A = [A00 A01; A10 A11] (A11 is optional)
-   Ap00             - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A^{-1}.

   Level: intermediate

   Notes:
    The Schur complement is NOT actually formed! Rather, this
          object performs the matrix-vector product by using formula S = A11 - A10 A^{-1} A01
          for Schur complement S and a KSP solver to approximate the action of A^{-1}.

          All four matrices must have the same MPI communicator.

          A00 and  A11 must be square matrices.

.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatSchurComplementUpdateSubMatrices(), MatCreateTranspose(), MatCreateSchurComplement(), MatGetSchurComplement()

@*/
PetscErrorCode  MatSchurComplementSetSubMatrices(Mat S,Mat A00,Mat Ap00,Mat A01,Mat A10,Mat A11)
{
  PetscErrorCode      ierr;
  Mat_SchurComplement *Na = (Mat_SchurComplement*)S->data;
  PetscBool           isschur;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)S,MATSCHURCOMPLEMENT,&isschur);CHKERRQ(ierr);
  if (!isschur) PetscFunctionReturn(0);
  if (S->assembled) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_ARG_WRONGSTATE,"Use MatSchurComplementUpdateSubMatrices() for already used matrix");
  PetscValidHeaderSpecific(A00,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Ap00,MAT_CLASSID,3);
  PetscValidHeaderSpecific(A01,MAT_CLASSID,4);
  PetscValidHeaderSpecific(A10,MAT_CLASSID,5);
  PetscCheckSameComm(A00,2,Ap00,3);
  PetscCheckSameComm(A00,2,A01,4);
  PetscCheckSameComm(A00,2,A10,5);
  if (A00->rmap->n != A00->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A00 %D do not equal local columns %D",A00->rmap->n,A00->cmap->n);
  if (A00->rmap->n != Ap00->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A00 %D do not equal local rows of Ap00 %D",A00->rmap->n,Ap00->rmap->n);
  if (Ap00->rmap->n != Ap00->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of Ap00 %D do not equal local columns %D",Ap00->rmap->n,Ap00->cmap->n);
  if (A00->cmap->n != A01->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of A00 %D do not equal local rows of A01 %D",A00->cmap->n,A01->rmap->n);
  if (A10->cmap->n != A00->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of A10 %D do not equal local rows of A00 %D",A10->cmap->n,A00->rmap->n);
  if (A11) {
    PetscValidHeaderSpecific(A11,MAT_CLASSID,6);
    PetscCheckSameComm(A00,2,A11,6);
    if (A10->rmap->n != A11->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A10 %D do not equal local rows A11 %D",A10->rmap->n,A11->rmap->n);
  }

  ierr   = MatSetSizes(S,A10->rmap->n,A01->cmap->n,A10->rmap->N,A01->cmap->N);CHKERRQ(ierr);
  ierr   = PetscObjectReference((PetscObject)A00);CHKERRQ(ierr);
  ierr   = PetscObjectReference((PetscObject)Ap00);CHKERRQ(ierr);
  ierr   = PetscObjectReference((PetscObject)A01);CHKERRQ(ierr);
  ierr   = PetscObjectReference((PetscObject)A10);CHKERRQ(ierr);
  Na->A  = A00;
  Na->Ap = Ap00;
  Na->B  = A01;
  Na->C  = A10;
  Na->D  = A11;
  if (A11) {
    ierr = PetscObjectReference((PetscObject)A11);CHKERRQ(ierr);
  }
  ierr = MatSetUp(S);CHKERRQ(ierr);
  ierr = KSPSetOperators(Na->ksp,A00,Ap00);CHKERRQ(ierr);
  S->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  MatSchurComplementGetKSP - Gets the KSP object that is used to invert A00 in the Schur complement matrix S = A11 - A10 ksp(A00,Ap00) A01

  Not Collective

  Input Parameter:
. S - matrix obtained with MatCreateSchurComplement() (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01

  Output Parameter:
. ksp - the linear solver object

  Options Database:
. -fieldsplit_<splitname_0>_XXX sets KSP and PC options for the 0-split solver inside the Schur complement used in PCFieldSplit; default <splitname_0> is 0.

  Level: intermediate

.seealso: MatSchurComplementSetKSP(), MatCreateSchurComplement(), MatCreateNormal(), MatMult(), MatCreate()
@*/
PetscErrorCode MatSchurComplementGetKSP(Mat S, KSP *ksp)
{
  Mat_SchurComplement *Na;
  PetscBool           isschur;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)S,MATSCHURCOMPLEMENT,&isschur);CHKERRQ(ierr);
  if (!isschur) SETERRQ1(PetscObjectComm((PetscObject)S),PETSC_ERR_ARG_WRONG,"Not for type %s",((PetscObject)S)->type_name);
  PetscValidPointer(ksp,2);
  Na   = (Mat_SchurComplement*) S->data;
  *ksp = Na->ksp;
  PetscFunctionReturn(0);
}

/*@
  MatSchurComplementSetKSP - Sets the KSP object that is used to invert A00 in the Schur complement matrix S = A11 - A10 ksp(A00,Ap00) A01

  Not Collective

  Input Parameters:
+ S   - matrix created with MatCreateSchurComplement()
- ksp - the linear solver object

  Level: developer

  Developer Notes:
    This is used in PCFieldSplit to reuse the 0-split KSP to implement ksp(A00,Ap00) in S.
    The KSP operators are overwritten with A00 and Ap00 currently set in S.

.seealso: MatSchurComplementGetKSP(), MatCreateSchurComplement(), MatCreateNormal(), MatMult(), MatCreate(), MATSCHURCOMPLEMENT
@*/
PetscErrorCode MatSchurComplementSetKSP(Mat S, KSP ksp)
{
  Mat_SchurComplement *Na;
  PetscErrorCode      ierr;
  PetscBool           isschur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)S,MATSCHURCOMPLEMENT,&isschur);CHKERRQ(ierr);
  if (!isschur) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  Na      = (Mat_SchurComplement*) S->data;
  ierr    = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr    = KSPDestroy(&Na->ksp);CHKERRQ(ierr);
  Na->ksp = ksp;
  ierr    = KSPSetOperators(Na->ksp, Na->A, Na->Ap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
      MatSchurComplementUpdateSubMatrices - Updates the Schur complement matrix object with new submatrices

   Collective on S

   Input Parameters:
+   S                - matrix obtained with MatCreateSchurComplement() (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01
.   A00,A01,A10,A11  - the four parts of A = [A00 A01; A10 A11] (A11 is optional)
-   Ap00             - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A^{-1}.

   Level: intermediate

   Notes:
    All four matrices must have the same MPI communicator

          A00 and  A11 must be square matrices

          All of the matrices provided must have the same sizes as was used with MatCreateSchurComplement() or MatSchurComplementSetSubMatrices()
          though they need not be the same matrices.

.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatCreateSchurComplement()

@*/
PetscErrorCode  MatSchurComplementUpdateSubMatrices(Mat S,Mat A00,Mat Ap00,Mat A01,Mat A10,Mat A11)
{
  PetscErrorCode      ierr;
  Mat_SchurComplement *Na = (Mat_SchurComplement*)S->data;
  PetscBool           isschur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)S,MATSCHURCOMPLEMENT,&isschur);CHKERRQ(ierr);
  if (!isschur) PetscFunctionReturn(0);
  if (!S->assembled) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_ARG_WRONGSTATE,"Use MatSchurComplementSetSubMatrices() for a new matrix");
  PetscValidHeaderSpecific(A00,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Ap00,MAT_CLASSID,3);
  PetscValidHeaderSpecific(A01,MAT_CLASSID,4);
  PetscValidHeaderSpecific(A10,MAT_CLASSID,5);
  PetscCheckSameComm(A00,2,Ap00,3);
  PetscCheckSameComm(A00,2,A01,4);
  PetscCheckSameComm(A00,2,A10,5);
  if (A00->rmap->n != A00->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A00 %D do not equal local columns %D",A00->rmap->n,A00->cmap->n);
  if (A00->rmap->n != Ap00->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A00 %D do not equal local rows of Ap00 %D",A00->rmap->n,Ap00->rmap->n);
  if (Ap00->rmap->n != Ap00->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of Ap00 %D do not equal local columns %D",Ap00->rmap->n,Ap00->cmap->n);
  if (A00->cmap->n != A01->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of A00 %D do not equal local rows of A01 %D",A00->cmap->n,A01->rmap->n);
  if (A10->cmap->n != A00->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local columns of A10 %D do not equal local rows of A00 %D",A10->cmap->n,A00->rmap->n);
  if (A11) {
    PetscValidHeaderSpecific(A11,MAT_CLASSID,6);
    PetscCheckSameComm(A00,2,A11,6);
    if (A10->rmap->n != A11->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local rows of A10 %D do not equal local rows A11 %D",A10->rmap->n,A11->rmap->n);
  }

  ierr = PetscObjectReference((PetscObject)A00);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Ap00);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)A01);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)A10);CHKERRQ(ierr);
  if (A11) {
    ierr = PetscObjectReference((PetscObject)A11);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->Ap);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->B);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->C);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->D);CHKERRQ(ierr);

  Na->A  = A00;
  Na->Ap = Ap00;
  Na->B  = A01;
  Na->C  = A10;
  Na->D  = A11;

  ierr = KSPSetOperators(Na->ksp,A00,Ap00);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
  MatSchurComplementGetSubMatrices - Get the individual submatrices in the Schur complement

  Collective on S

  Input Parameter:
. S                - matrix obtained with MatCreateSchurComplement() (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01

  Output Parameters:
+ A00,A01,A10,A11  - the four parts of the original matrix A = [A00 A01; A10 A11] (A11 is optional)
- Ap00             - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A^{-1}.

  Note: A11 is optional, and thus can be NULL.  The submatrices are not increfed before they are returned and should not be modified or destroyed.

  Level: intermediate

.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP(), MatCreateSchurComplement(), MatSchurComplementUpdateSubMatrices()
@*/
PetscErrorCode  MatSchurComplementGetSubMatrices(Mat S,Mat *A00,Mat *Ap00,Mat *A01,Mat *A10,Mat *A11)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement*) S->data;
  PetscErrorCode      ierr;
  PetscBool           flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)S,MATSCHURCOMPLEMENT,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)S),PETSC_ERR_ARG_WRONG,"Not for type %s",((PetscObject)S)->type_name);
  if (A00) *A00 = Na->A;
  if (Ap00) *Ap00 = Na->Ap;
  if (A01) *A01 = Na->B;
  if (A10) *A10 = Na->C;
  if (A11) *A11 = Na->D;
  PetscFunctionReturn(0);
}

#include <petsc/private/kspimpl.h>

/*@
  MatSchurComplementComputeExplicitOperator - Compute the Schur complement matrix explicitly

  Collective on M

  Input Parameter:
. M - the matrix obtained with MatCreateSchurComplement()

  Output Parameter:
. S - the Schur complement matrix

  Note: This can be expensive, so it is mainly for testing

  Level: advanced

.seealso: MatCreateSchurComplement(), MatSchurComplementUpdate()
@*/
PetscErrorCode MatSchurComplementComputeExplicitOperator(Mat M, Mat *S)
{
  Mat            B, C, D, Bd, AinvBd;
  KSP            ksp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSchurComplementGetSubMatrices(M, NULL, NULL, &B, &C, &D);CHKERRQ(ierr);
  ierr = MatSchurComplementGetKSP(M, &ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = MatConvert(B, MATDENSE, MAT_INITIAL_MATRIX, &Bd);CHKERRQ(ierr);
  ierr = MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd);CHKERRQ(ierr);
  ierr = KSPMatSolve(ksp, Bd, AinvBd);CHKERRQ(ierr);
  ierr = MatDestroy(&Bd);CHKERRQ(ierr);
  ierr = MatChop(AinvBd, PETSC_SMALL);CHKERRQ(ierr);
  ierr = MatMatMult(C, AinvBd, MAT_INITIAL_MATRIX, PETSC_DEFAULT, S);CHKERRQ(ierr);
  ierr = MatDestroy(&AinvBd);CHKERRQ(ierr);
  if (D) {
    ierr = MatAXPY(*S, -1.0, D, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatConvert(*S, MATAIJ, MAT_INPLACE_MATRIX, S);CHKERRQ(ierr);
  ierr = MatScale(*S, -1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Developer Notes:
    This should be implemented with a MatCreate_SchurComplement() as that is the standard design for new Mat classes. */
PetscErrorCode MatGetSchurComplement_Basic(Mat mat,IS isrow0,IS iscol0,IS isrow1,IS iscol1,MatReuse mreuse,Mat *newmat,MatSchurComplementAinvType ainvtype, MatReuse preuse,Mat *newpmat)
{
  PetscErrorCode ierr;
  Mat            A=NULL,Ap=NULL,B=NULL,C=NULL,D=NULL;
  MatReuse       reuse;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(isrow0,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol0,IS_CLASSID,3);
  PetscValidHeaderSpecific(isrow1,IS_CLASSID,4);
  PetscValidHeaderSpecific(iscol1,IS_CLASSID,5);
  PetscValidLogicalCollectiveEnum(mat,mreuse,6);
  PetscValidLogicalCollectiveEnum(mat,ainvtype,8);
  PetscValidLogicalCollectiveEnum(mat,preuse,9);
  if (mreuse == MAT_IGNORE_MATRIX && preuse == MAT_IGNORE_MATRIX) PetscFunctionReturn(0);
  if (mreuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newmat,MAT_CLASSID,7);
  if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newpmat,MAT_CLASSID,10);

  if (mat->factortype) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  reuse = MAT_INITIAL_MATRIX;
  if (mreuse == MAT_REUSE_MATRIX) {
    ierr = MatSchurComplementGetSubMatrices(*newmat,&A,&Ap,&B,&C,&D);CHKERRQ(ierr);
    if (!A || !Ap || !B || !C) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Attempting to reuse matrix but Schur complement matrices unset");
    if (A != Ap) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Preconditioning matrix does not match operator");
    ierr = MatDestroy(&Ap);CHKERRQ(ierr); /* get rid of extra reference */
    reuse = MAT_REUSE_MATRIX;
  }
  ierr = MatCreateSubMatrix(mat,isrow0,iscol0,reuse,&A);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(mat,isrow0,iscol1,reuse,&B);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(mat,isrow1,iscol0,reuse,&C);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(mat,isrow1,iscol1,reuse,&D);CHKERRQ(ierr);
  switch (mreuse) {
  case MAT_INITIAL_MATRIX:
    ierr = MatCreateSchurComplement(A,A,B,C,D,newmat);CHKERRQ(ierr);
    break;
  case MAT_REUSE_MATRIX:
    ierr = MatSchurComplementUpdateSubMatrices(*newmat,A,A,B,C,D);CHKERRQ(ierr);
    break;
  default:
    if (mreuse != MAT_IGNORE_MATRIX) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Unrecognized value of mreuse %d",(int)mreuse);
  }
  if (preuse != MAT_IGNORE_MATRIX) {
    ierr = MatCreateSchurComplementPmat(A,B,C,D,ainvtype,preuse,newpmat);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    MatGetSchurComplement - Obtain the Schur complement from eliminating part of the matrix in another part.

    Collective on A

    Input Parameters:
+   A      - matrix in which the complement is to be taken
.   isrow0 - rows to eliminate
.   iscol0 - columns to eliminate, (isrow0,iscol0) should be square and nonsingular
.   isrow1 - rows in which the Schur complement is formed
.   iscol1 - columns in which the Schur complement is formed
.   mreuse - MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX, use MAT_IGNORE_MATRIX to put nothing in S
.   ainvtype - the type of approximation used for the inverse of the (0,0) block used in forming Sp:
                       MAT_SCHUR_COMPLEMENT_AINV_DIAG, MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG, or MAT_SCHUR_COMPLEMENT_AINV_LUMP
-   preuse - MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX, use MAT_IGNORE_MATRIX to put nothing in Sp

    Output Parameters:
+   S      - exact Schur complement, often of type MATSCHURCOMPLEMENT which is difficult to use for preconditioning
-   Sp     - approximate Schur complement from which a preconditioner can be built

    Note:
    Since the real Schur complement is usually dense, providing a good approximation to newpmat usually requires
    application-specific information.  The default for assembled matrices is to use the inverse of the diagonal of
    the (0,0) block A00 in place of A00^{-1}. This rarely produce a scalable algorithm. Optionally, A00 can be lumped
    before forming inv(diag(A00)).

    Sometimes users would like to provide problem-specific data in the Schur complement, usually only for special row
    and column index sets.  In that case, the user should call PetscObjectComposeFunction() on the *S matrix and pass mreuse of MAT_REUSE_MATRIX to set
    "MatGetSchurComplement_C" to their function.  If their function needs to fall back to the default implementation, it
    should call MatGetSchurComplement_Basic().

    MatCreateSchurComplement() takes as arguments the four submatrices and returns the virtual Schur complement (what this returns in S).

    MatSchurComplementGetPmat() takes the virtual Schur complement and returns an explicit approximate Schur complement (what this returns in Sp).

    In other words calling MatCreateSchurComplement() followed by MatSchurComplementGetPmat() produces the same output as this function but with slightly different
    inputs. The actually submatrices of the original block matrix instead of index sets to the submatrices.

    Developer Notes:
    The API that includes MatGetSchurComplement(), MatCreateSchurComplement(), MatSchurComplementGetPmat() should be refactored to
    remove redundancy and be clearer and simplier.

    Level: advanced

.seealso: MatCreateSubMatrix(), PCFIELDSPLIT, MatCreateSchurComplement(), MatSchurComplementAinvType
@*/
PetscErrorCode  MatGetSchurComplement(Mat A,IS isrow0,IS iscol0,IS isrow1,IS iscol1,MatReuse mreuse,Mat *S,MatSchurComplementAinvType ainvtype,MatReuse preuse,Mat *Sp)
{
  PetscErrorCode ierr,(*f)(Mat,IS,IS,IS,IS,MatReuse,Mat*,MatReuse,Mat*) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow0,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol0,IS_CLASSID,3);
  PetscValidHeaderSpecific(isrow1,IS_CLASSID,4);
  PetscValidHeaderSpecific(iscol1,IS_CLASSID,5);
  PetscValidLogicalCollectiveEnum(A,mreuse,6);
  if (mreuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*S,MAT_CLASSID,7);
  PetscValidLogicalCollectiveEnum(A,ainvtype,8);
  PetscValidLogicalCollectiveEnum(A,preuse,9);
  if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*Sp,MAT_CLASSID,10);
  PetscValidType(A,1);
  if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  f = NULL;
  if (mreuse == MAT_REUSE_MATRIX) { /* This is the only situation, in which we can demand that the user pass a non-NULL pointer to non-garbage in S. */
    ierr = PetscObjectQueryFunction((PetscObject)*S,"MatGetSchurComplement_C",&f);CHKERRQ(ierr);
  }
  if (f) {
    ierr = (*f)(A,isrow0,iscol0,isrow1,iscol1,mreuse,S,preuse,Sp);CHKERRQ(ierr);
  } else {
    ierr = MatGetSchurComplement_Basic(A,isrow0,iscol0,isrow1,iscol1,mreuse,S,ainvtype,preuse,Sp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    MatSchurComplementSetAinvType - set the type of approximation used for the inverse of the (0,0) block used in forming Sp in MatSchurComplementGetPmat()

    Not collective.

    Input Parameters:
+   S        - matrix obtained with MatCreateSchurComplement() (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01
-   ainvtype - type of approximation used to form A00inv from A00 when assembling Sp = A11 - A10 A00inv A01:
                      MAT_SCHUR_COMPLEMENT_AINV_DIAG, MAT_SCHUR_COMPLEMENT_AINV_LUMP, or MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG

    Options database:
    -mat_schur_complement_ainv_type diag | lump | blockdiag

    Note:
    Since the real Schur complement is usually dense, providing a good approximation to newpmat usually requires
    application-specific information.  The default for assembled matrices is to use the inverse of the diagonal of
    the (0,0) block A00 in place of A00^{-1}. This rarely produces a scalable algorithm. Optionally, A00 can be lumped
    before forming inv(diag(A00)).

    Level: advanced

.seealso: MatSchurComplementAinvType, MatCreateSchurComplement(), MatGetSchurComplement(), MatSchurComplementGetPmat(), MatSchurComplementGetAinvType()
@*/
PetscErrorCode  MatSchurComplementSetAinvType(Mat S,MatSchurComplementAinvType ainvtype)
{
  PetscErrorCode      ierr;
  PetscBool           isschur;
  Mat_SchurComplement *schur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)S,MATSCHURCOMPLEMENT,&isschur);CHKERRQ(ierr);
  if (!isschur) PetscFunctionReturn(0);
  PetscValidLogicalCollectiveEnum(S,ainvtype,2);
  schur = (Mat_SchurComplement*)S->data;
  if (ainvtype != MAT_SCHUR_COMPLEMENT_AINV_DIAG && ainvtype != MAT_SCHUR_COMPLEMENT_AINV_LUMP && ainvtype != MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown MatSchurComplementAinvType: %d",(int)ainvtype);
  schur->ainvtype = ainvtype;
  PetscFunctionReturn(0);
}

/*@
    MatSchurComplementGetAinvType - get the type of approximation for the inverse of the (0,0) block used in forming Sp in MatSchurComplementGetPmat()

    Not collective.

    Input Parameter:
.   S      - matrix obtained with MatCreateSchurComplement() (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01

    Output Parameter:
.   ainvtype - type of approximation used to form A00inv from A00 when assembling Sp = A11 - A10 A00inv A01:
                      MAT_SCHUR_COMPLEMENT_AINV_DIAG, MAT_SCHUR_COMPLEMENT_AINV_LUMP, or MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG

    Note:
    Since the real Schur complement is usually dense, providing a good approximation to newpmat usually requires
    application-specific information.  The default for assembled matrices is to use the inverse of the diagonal of
    the (0,0) block A00 in place of A00^{-1}. This rarely produce a scalable algorithm. Optionally, A00 can be lumped
    before forming inv(diag(A00)).

    Level: advanced

.seealso: MatSchurComplementAinvType, MatCreateSchurComplement(), MatGetSchurComplement(), MatSchurComplementGetPmat(), MatSchurComplementSetAinvType()
@*/
PetscErrorCode  MatSchurComplementGetAinvType(Mat S,MatSchurComplementAinvType *ainvtype)
{
  PetscErrorCode      ierr;
  PetscBool           isschur;
  Mat_SchurComplement *schur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)S,MATSCHURCOMPLEMENT,&isschur);CHKERRQ(ierr);
  if (!isschur) SETERRQ1(PetscObjectComm((PetscObject)S),PETSC_ERR_ARG_WRONG,"Not for type %s",((PetscObject)S)->type_name);
  schur = (Mat_SchurComplement*)S->data;
  if (ainvtype) *ainvtype = schur->ainvtype;
  PetscFunctionReturn(0);
}

/*@
    MatCreateSchurComplementPmat - create a preconditioning matrix for the Schur complement by assembling Sp = A11 - A10 inv(diag(A00)) A01

    Collective on A00

    Input Parameters:
+   A00,A01,A10,A11      - the four parts of the original matrix A = [A00 A01; A10 A11] (A01,A10, and A11 are optional, implying zero matrices)
.   ainvtype             - type of approximation for inv(A00) used when forming Sp = A11 - A10 inv(A00) A01
-   preuse               - MAT_INITIAL_MATRIX for a new Sp, or MAT_REUSE_MATRIX to reuse an existing Sp, or MAT_IGNORE_MATRIX to put nothing in Sp

    Output Parameter:
-   Spmat                - approximate Schur complement suitable for preconditioning S = A11 - A10 inv(diag(A00)) A01

    Note:
    Since the real Schur complement is usually dense, providing a good approximation to newpmat usually requires
    application-specific information.  The default for assembled matrices is to use the inverse of the diagonal of
    the (0,0) block A00 in place of A00^{-1}. This rarely produce a scalable algorithm. Optionally, A00 can be lumped
    before forming inv(diag(A00)).

    Level: advanced

.seealso: MatCreateSchurComplement(), MatGetSchurComplement(), MatSchurComplementGetPmat(), MatSchurComplementAinvType
@*/
PetscErrorCode  MatCreateSchurComplementPmat(Mat A00,Mat A01,Mat A10,Mat A11,MatSchurComplementAinvType ainvtype,MatReuse preuse,Mat *Spmat)
{
  PetscErrorCode ierr;
  PetscInt       N00;

  PetscFunctionBegin;
  /* Use an appropriate approximate inverse of A00 to form A11 - A10 inv(diag(A00)) A01; a NULL A01, A10 or A11 indicates a zero matrix. */
  /* TODO: Perhaps should create an appropriately-sized zero matrix of the same type as A00? */
  if ((!A01 || !A10) & !A11) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot assemble Spmat: A01, A10 and A11 are all NULL.");
  PetscValidLogicalCollectiveEnum(A11,preuse,6);
  if (preuse == MAT_IGNORE_MATRIX) PetscFunctionReturn(0);

  /* A zero size A00 or empty A01 or A10 imply S = A11. */
  ierr = MatGetSize(A00,&N00,NULL);CHKERRQ(ierr);
  if (!A01 || !A10 || !N00) {
    if (preuse == MAT_INITIAL_MATRIX) {
      ierr = MatDuplicate(A11,MAT_COPY_VALUES,Spmat);CHKERRQ(ierr);
    } else { /* MAT_REUSE_MATRIX */
      /* TODO: when can we pass SAME_NONZERO_PATTERN? */
      ierr = MatCopy(A11,*Spmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  } else {
    Mat AdB;
    Vec diag;

    if (ainvtype == MAT_SCHUR_COMPLEMENT_AINV_LUMP || ainvtype == MAT_SCHUR_COMPLEMENT_AINV_DIAG) {
      ierr = MatDuplicate(A01,MAT_COPY_VALUES,&AdB);CHKERRQ(ierr);
      ierr = MatCreateVecs(A00,&diag,NULL);CHKERRQ(ierr);
      if (ainvtype == MAT_SCHUR_COMPLEMENT_AINV_LUMP) {
        ierr = MatGetRowSum(A00,diag);CHKERRQ(ierr);
      } else {
        ierr = MatGetDiagonal(A00,diag);CHKERRQ(ierr);
      }
      ierr = VecReciprocal(diag);CHKERRQ(ierr);
      ierr = MatDiagonalScale(AdB,diag,NULL);CHKERRQ(ierr);
      ierr = VecDestroy(&diag);CHKERRQ(ierr);
    } else if (ainvtype == MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG) {
      Mat      A00_inv;
      MatType  type;
      MPI_Comm comm;

      ierr = PetscObjectGetComm((PetscObject)A00,&comm);CHKERRQ(ierr);
      ierr = MatGetType(A00,&type);CHKERRQ(ierr);
      ierr = MatCreate(comm,&A00_inv);CHKERRQ(ierr);
      ierr = MatSetType(A00_inv,type);CHKERRQ(ierr);
      ierr = MatInvertBlockDiagonalMat(A00,A00_inv);CHKERRQ(ierr);
      ierr = MatMatMult(A00_inv,A01,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AdB);CHKERRQ(ierr);
      ierr = MatDestroy(&A00_inv);CHKERRQ(ierr);
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown MatSchurComplementAinvType: %D", ainvtype);
    /* Cannot really reuse Spmat in MatMatMult() because of MatAYPX() -->
         MatAXPY() --> MatHeaderReplace() --> MatDestroy_XXX_MatMatMult()  */
    ierr     = MatDestroy(Spmat);CHKERRQ(ierr);
    ierr     = MatMatMult(A10,AdB,MAT_INITIAL_MATRIX,PETSC_DEFAULT,Spmat);CHKERRQ(ierr);
    if (!A11) {
      ierr = MatScale(*Spmat,-1.0);CHKERRQ(ierr);
    } else {
      /* TODO: when can we pass SAME_NONZERO_PATTERN? */
      ierr = MatAYPX(*Spmat,-1,A11,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&AdB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSchurComplementGetPmat_Basic(Mat S,MatReuse preuse,Mat *Spmat)
{
  Mat A,B,C,D;
  Mat_SchurComplement *schur = (Mat_SchurComplement *)S->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (preuse == MAT_IGNORE_MATRIX) PetscFunctionReturn(0);
  ierr = MatSchurComplementGetSubMatrices(S,&A,NULL,&B,&C,&D);CHKERRQ(ierr);
  if (!A) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_ARG_WRONGSTATE,"Schur complement component matrices unset");
  ierr = MatCreateSchurComplementPmat(A,B,C,D,schur->ainvtype,preuse,Spmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    MatSchurComplementGetPmat - Obtain a preconditioning matrix for the Schur complement by assembling Sp = A11 - A10 inv(diag(A00)) A01

    Collective on S

    Input Parameters:
+   S      - matrix obtained with MatCreateSchurComplement() (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01
-   preuse - MAT_INITIAL_MATRIX for a new Sp, or MAT_REUSE_MATRIX to reuse an existing Sp, or MAT_IGNORE_MATRIX to put nothing in Sp

    Output Parameter:
-   Sp     - approximate Schur complement suitable for preconditioning S = A11 - A10 inv(diag(A00)) A01

    Note:
    Since the real Schur complement is usually dense, providing a good approximation to newpmat usually requires
    application-specific information.  The default for assembled matrices is to use the inverse of the diagonal of
    the (0,0) block A00 in place of A00^{-1}. This rarely produce a scalable algorithm. Optionally, A00 can be lumped
    before forming inv(diag(A00)).

    Sometimes users would like to provide problem-specific data in the Schur complement, usually only
    for special row and column index sets.  In that case, the user should call PetscObjectComposeFunction() to set
    "MatSchurComplementGetPmat_C" to their function.  If their function needs to fall back to the default implementation,
    it should call MatSchurComplementGetPmat_Basic().

    Developer Notes:
    The API that includes MatGetSchurComplement(), MatCreateSchurComplement(), MatSchurComplementGetPmat() should be refactored to
    remove redundancy and be clearer and simplier.

    Level: advanced

.seealso: MatCreateSubMatrix(), PCFIELDSPLIT, MatGetSchurComplement(), MatCreateSchurComplement(), MatSchurComplementSetAinvType()
@*/
PetscErrorCode  MatSchurComplementGetPmat(Mat S,MatReuse preuse,Mat *Sp)
{
  PetscErrorCode ierr,(*f)(Mat,MatReuse,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_CLASSID,1);
  PetscValidType(S,1);
  PetscValidLogicalCollectiveEnum(S,preuse,2);
  if (preuse != MAT_IGNORE_MATRIX) PetscValidPointer(Sp,3);
  if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*Sp,MAT_CLASSID,3);
  if (S->factortype) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = PetscObjectQueryFunction((PetscObject)S,"MatSchurComplementGetPmat_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(S,preuse,Sp);CHKERRQ(ierr);
  } else {
    ierr = MatSchurComplementGetPmat_Basic(S,preuse,Sp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SchurComplement(Mat N)
{
  PetscErrorCode      ierr;
  Mat_SchurComplement *Na;

  PetscFunctionBegin;
  ierr    = PetscNewLog(N,&Na);CHKERRQ(ierr);
  N->data = (void*) Na;

  N->ops->destroy        = MatDestroy_SchurComplement;
  N->ops->getvecs        = MatCreateVecs_SchurComplement;
  N->ops->view           = MatView_SchurComplement;
  N->ops->mult           = MatMult_SchurComplement;
  N->ops->multtranspose  = MatMultTranspose_SchurComplement;
  N->ops->multadd        = MatMultAdd_SchurComplement;
  N->ops->setfromoptions = MatSetFromOptions_SchurComplement;
  N->assembled           = PETSC_FALSE;
  N->preallocated        = PETSC_FALSE;

  ierr = KSPCreate(PetscObjectComm((PetscObject)N),&Na->ksp);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)N,MATSCHURCOMPLEMENT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
