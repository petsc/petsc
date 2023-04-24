#include <../src/ksp/ksp/utils/schurm/schurm.h> /*I "petscksp.h" I*/

const char *const MatSchurComplementAinvTypes[] = {"DIAG", "LUMP", "BLOCKDIAG", "FULL", "MatSchurComplementAinvType", "MAT_SCHUR_COMPLEMENT_AINV_", NULL};

PetscErrorCode MatCreateVecs_SchurComplement(Mat N, Vec *right, Vec *left)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)N->data;

  PetscFunctionBegin;
  if (Na->D) {
    PetscCall(MatCreateVecs(Na->D, right, left));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (right) PetscCall(MatCreateVecs(Na->B, right, NULL));
  if (left) PetscCall(MatCreateVecs(Na->C, NULL, left));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatView_SchurComplement(Mat N, PetscViewer viewer)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)N->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Schur complement A11 - A10 inv(A00) A01\n"));
  if (Na->D) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "A11\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(MatView(Na->D, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "A11 = 0\n"));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "A10\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(MatView(Na->C, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "KSP of A00\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(KSPView(Na->ksp, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "A01\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(MatView(Na->B, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
           A11^T - A01^T ksptrans(A00,Ap00) A10^T
*/
PetscErrorCode MatMultTranspose_SchurComplement(Mat N, Vec x, Vec y)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)N->data;

  PetscFunctionBegin;
  if (!Na->work1) PetscCall(MatCreateVecs(Na->A, &Na->work1, NULL));
  if (!Na->work2) PetscCall(MatCreateVecs(Na->A, &Na->work2, NULL));
  PetscCall(MatMultTranspose(Na->C, x, Na->work1));
  PetscCall(KSPSolveTranspose(Na->ksp, Na->work1, Na->work2));
  PetscCall(MatMultTranspose(Na->B, Na->work2, y));
  PetscCall(VecScale(y, -1.0));
  if (Na->D) PetscCall(MatMultTransposeAdd(Na->D, x, y, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
           A11 - A10 ksp(A00,Ap00) A01
*/
PetscErrorCode MatMult_SchurComplement(Mat N, Vec x, Vec y)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)N->data;

  PetscFunctionBegin;
  if (!Na->work1) PetscCall(MatCreateVecs(Na->A, &Na->work1, NULL));
  if (!Na->work2) PetscCall(MatCreateVecs(Na->A, &Na->work2, NULL));
  PetscCall(MatMult(Na->B, x, Na->work1));
  PetscCall(KSPSolve(Na->ksp, Na->work1, Na->work2));
  PetscCall(MatMult(Na->C, Na->work2, y));
  PetscCall(VecScale(y, -1.0));
  if (Na->D) PetscCall(MatMultAdd(Na->D, x, y, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
           A11 - A10 ksp(A00,Ap00) A01
*/
PetscErrorCode MatMultAdd_SchurComplement(Mat N, Vec x, Vec y, Vec z)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)N->data;

  PetscFunctionBegin;
  if (!Na->work1) PetscCall(MatCreateVecs(Na->A, &Na->work1, NULL));
  if (!Na->work2) PetscCall(MatCreateVecs(Na->A, &Na->work2, NULL));
  PetscCall(MatMult(Na->B, x, Na->work1));
  PetscCall(KSPSolve(Na->ksp, Na->work1, Na->work2));
  if (y == z) {
    PetscCall(VecScale(Na->work2, -1.0));
    PetscCall(MatMultAdd(Na->C, Na->work2, z, z));
  } else {
    PetscCall(MatMult(Na->C, Na->work2, z));
    PetscCall(VecAYPX(z, -1.0, y));
  }
  if (Na->D) PetscCall(MatMultAdd(Na->D, x, z, z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetFromOptions_SchurComplement(Mat N, PetscOptionItems *PetscOptionsObject)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)N->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MatSchurComplementOptions");
  Na->ainvtype = MAT_SCHUR_COMPLEMENT_AINV_DIAG;
  PetscCall(PetscOptionsEnum("-mat_schur_complement_ainv_type", "Type of approximation for DIAGFORM(A00) used when assembling Sp = A11 - A10 inv(DIAGFORM(A00)) A01", "MatSchurComplementSetAinvType", MatSchurComplementAinvTypes, (PetscEnum)Na->ainvtype,
                             (PetscEnum *)&Na->ainvtype, NULL));
  PetscOptionsHeadEnd();
  PetscCall(KSPSetFromOptions(Na->ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_SchurComplement(Mat N)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)N->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&Na->A));
  PetscCall(MatDestroy(&Na->Ap));
  PetscCall(MatDestroy(&Na->B));
  PetscCall(MatDestroy(&Na->C));
  PetscCall(MatDestroy(&Na->D));
  PetscCall(VecDestroy(&Na->work1));
  PetscCall(VecDestroy(&Na->work2));
  PetscCall(KSPDestroy(&Na->ksp));
  PetscCall(PetscFree(N->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_schurcomplement_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_schurcomplement_mpidense_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
      MatCreateSchurComplement - Creates a new `Mat` that behaves like the Schur complement of a matrix

   Collective

   Input Parameters:
+   A00  - the upper-left block of the original matrix A = [A00 A01; A10 A11]
.   Ap00 - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A00^{-1}
.   A01  - the upper-right block of the original matrix A = [A00 A01; A10 A11]
.   A10  - the lower-left block of the original matrix A = [A00 A01; A10 A11]
-   A11  - (optional) the lower-right block of the original matrix A = [A00 A01; A10 A11]

   Output Parameter:
.   S - the matrix that behaves as the Schur complement S = A11 - A10 ksp(A00,Ap00) A01

   Level: intermediate

   Notes:
    The Schur complement is NOT explicitly formed! Rather, this function returns a virtual Schur complement
    that can compute the matrix-vector product by using formula S = A11 - A10 A^{-1} A01
    for Schur complement S and a `KSP` solver to approximate the action of A^{-1}.

    All four matrices must have the same MPI communicator.

    `A00` and  `A11` must be square matrices.

    `MatGetSchurComplement()` takes as arguments the index sets for the submatrices and returns both the virtual Schur complement (what this returns) plus
    a sparse approximation to the Schur complement (useful for building a preconditioner for the Schur complement) which can be obtained from this
    matrix with `MatSchurComplementGetPmat()`

    Developer Note:
    The API that includes `MatGetSchurComplement()`, `MatCreateSchurComplement()`, `MatSchurComplementGetPmat()` should be refactored to
    remove redundancy and be clearer and simpler.

.seealso: [](chapter_ksp), `MatCreateNormal()`, `MatMult()`, `MatCreate()`, `MatSchurComplementGetKSP()`, `MatSchurComplementUpdateSubMatrices()`, `MatCreateTranspose()`, `MatGetSchurComplement()`,
          `MatSchurComplementGetPmat()`, `MatSchurComplementSetSubMatrices()`
@*/
PetscErrorCode MatCreateSchurComplement(Mat A00, Mat Ap00, Mat A01, Mat A10, Mat A11, Mat *S)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A00), S));
  PetscCall(MatSetType(*S, MATSCHURCOMPLEMENT));
  PetscCall(MatSchurComplementSetSubMatrices(*S, A00, Ap00, A01, A10, A11));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
      MatSchurComplementSetSubMatrices - Sets the matrices that define the Schur complement

   Collective

   Input Parameters:
+   S                - matrix obtained with `MatSetType`(S,`MATSCHURCOMPLEMENT`)
+   A00  - the upper-left block of the original matrix A = [A00 A01; A10 A11]
.   Ap00 - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A00^{-1}
.   A01  - the upper-right block of the original matrix A = [A00 A01; A10 A11]
.   A10  - the lower-left block of the original matrix A = [A00 A01; A10 A11]
-   A11  - (optional) the lower-right block of the original matrix A = [A00 A01; A10 A11]

   Level: intermediate

   Notes:
     The Schur complement is NOT explicitly formed! Rather, this
     object performs the matrix-vector product of the Schur complement by using formula S = A11 - A10 ksp(A00,Ap00) A01

     All four matrices must have the same MPI communicator.

     `A00` and `A11` must be square matrices.

     This is to be used in the context of code such as
.vb
     MatSetType(S,MATSCHURCOMPLEMENT);
     MatSchurComplementSetSubMatrices(S,...);
.ve
    while `MatSchurComplementUpdateSubMatrices()` should only be called after `MatCreateSchurComplement()` or `MatSchurComplementSetSubMatrices()`

.seealso: [](chapter_ksp), `Mat`, `MatCreateNormal()`, `MatMult()`, `MatCreate()`, `MatSchurComplementGetKSP()`, `MatSchurComplementUpdateSubMatrices()`, `MatCreateTranspose()`, `MatCreateSchurComplement()`, `MatGetSchurComplement()`
@*/
PetscErrorCode MatSchurComplementSetSubMatrices(Mat S, Mat A00, Mat Ap00, Mat A01, Mat A10, Mat A11)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)S->data;
  PetscBool            isschur;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)S, MATSCHURCOMPLEMENT, &isschur));
  if (!isschur) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!S->assembled, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_WRONGSTATE, "Use MatSchurComplementUpdateSubMatrices() for already used matrix");
  PetscValidHeaderSpecific(A00, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(Ap00, MAT_CLASSID, 3);
  PetscValidHeaderSpecific(A01, MAT_CLASSID, 4);
  PetscValidHeaderSpecific(A10, MAT_CLASSID, 5);
  PetscCheckSameComm(A00, 2, Ap00, 3);
  PetscCheckSameComm(A00, 2, A01, 4);
  PetscCheckSameComm(A00, 2, A10, 5);
  PetscCheck(A00->rmap->n == A00->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of A00 %" PetscInt_FMT " do not equal local columns %" PetscInt_FMT, A00->rmap->n, A00->cmap->n);
  PetscCheck(A00->rmap->n == Ap00->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of A00 %" PetscInt_FMT " do not equal local rows of Ap00 %" PetscInt_FMT, A00->rmap->n, Ap00->rmap->n);
  PetscCheck(Ap00->rmap->n == Ap00->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of Ap00 %" PetscInt_FMT " do not equal local columns %" PetscInt_FMT, Ap00->rmap->n, Ap00->cmap->n);
  PetscCheck(A00->cmap->n == A01->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local columns of A00 %" PetscInt_FMT " do not equal local rows of A01 %" PetscInt_FMT, A00->cmap->n, A01->rmap->n);
  PetscCheck(A10->cmap->n == A00->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local columns of A10 %" PetscInt_FMT " do not equal local rows of A00 %" PetscInt_FMT, A10->cmap->n, A00->rmap->n);
  if (A11) {
    PetscValidHeaderSpecific(A11, MAT_CLASSID, 6);
    PetscCheckSameComm(A00, 2, A11, 6);
    PetscCheck(A10->rmap->n == A11->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of A10 %" PetscInt_FMT " do not equal local rows A11 %" PetscInt_FMT, A10->rmap->n, A11->rmap->n);
  }

  PetscCall(MatSetSizes(S, A10->rmap->n, A01->cmap->n, A10->rmap->N, A01->cmap->N));
  PetscCall(PetscObjectReference((PetscObject)A00));
  PetscCall(PetscObjectReference((PetscObject)Ap00));
  PetscCall(PetscObjectReference((PetscObject)A01));
  PetscCall(PetscObjectReference((PetscObject)A10));
  Na->A  = A00;
  Na->Ap = Ap00;
  Na->B  = A01;
  Na->C  = A10;
  Na->D  = A11;
  if (A11) PetscCall(PetscObjectReference((PetscObject)A11));
  PetscCall(MatSetUp(S));
  PetscCall(KSPSetOperators(Na->ksp, A00, Ap00));
  S->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSchurComplementGetKSP - Gets the `KSP` object that is used to solve with A00 in the Schur complement matrix S = A11 - A10 ksp(A00,Ap00) A01

  Not Collective

  Input Parameter:
. S - matrix obtained with `MatCreateSchurComplement()` (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01

  Output Parameter:
. ksp - the linear solver object

  Options Database Key:
. -fieldsplit_<splitname_0>_XXX sets KSP and PC options for the 0-split solver inside the Schur complement used in `PCFIELDSPLIT`; default <splitname_0> is 0.

  Level: intermediate

.seealso: [](chapter_ksp), `Mat`, `MatSchurComplementSetKSP()`, `MatCreateSchurComplement()`, `MatCreateNormal()`, `MatMult()`, `MatCreate()`
@*/
PetscErrorCode MatSchurComplementGetKSP(Mat S, KSP *ksp)
{
  Mat_SchurComplement *Na;
  PetscBool            isschur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)S, MATSCHURCOMPLEMENT, &isschur));
  PetscCheck(isschur, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_WRONG, "Not for type %s", ((PetscObject)S)->type_name);
  PetscValidPointer(ksp, 2);
  Na   = (Mat_SchurComplement *)S->data;
  *ksp = Na->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSchurComplementSetKSP - Sets the `KSP` object that is used to solve with A00 in the Schur complement matrix S = A11 - A10 ksp(A00,Ap00) A01

  Not Collective

  Input Parameters:
+ S   - matrix created with `MatCreateSchurComplement()`
- ksp - the linear solver object

  Level: developer

  Developer Note:
    This is used in `PCFIELDSPLIT` to reuse the 0-split `KSP` to implement ksp(A00,Ap00) in S.
    The `KSP` operators are overwritten with A00 and Ap00 currently set in S.

.seealso: [](chapter_ksp), `Mat`, `MatSchurComplementGetKSP()`, `MatCreateSchurComplement()`, `MatCreateNormal()`, `MatMult()`, `MatCreate()`, `MATSCHURCOMPLEMENT`
@*/
PetscErrorCode MatSchurComplementSetKSP(Mat S, KSP ksp)
{
  Mat_SchurComplement *Na;
  PetscBool            isschur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)S, MATSCHURCOMPLEMENT, &isschur));
  if (!isschur) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  Na = (Mat_SchurComplement *)S->data;
  PetscCall(PetscObjectReference((PetscObject)ksp));
  PetscCall(KSPDestroy(&Na->ksp));
  Na->ksp = ksp;
  PetscCall(KSPSetOperators(Na->ksp, Na->A, Na->Ap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
      MatSchurComplementUpdateSubMatrices - Updates the Schur complement matrix object with new submatrices

   Collective

   Input Parameters:
+   S                - matrix obtained with `MatCreateSchurComplement()` (or `MatSchurSetSubMatrices()`) and implementing the action of A11 - A10 ksp(A00,Ap00) A01
.   A00  - the upper-left block of the original matrix A = [A00 A01; A10 A11]
.   Ap00 - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A00^{-1}
.   A01  - the upper-right block of the original matrix A = [A00 A01; A10 A11]
.   A10  - the lower-left block of the original matrix A = [A00 A01; A10 A11]
-   A11  - (optional) the lower-right block of the original matrix A = [A00 A01; A10 A11]

   Level: intermediate

   Notes:
     All four matrices must have the same MPI communicator

     `A00` and  `A11` must be square matrices

     All of the matrices provided must have the same sizes as was used with `MatCreateSchurComplement()` or `MatSchurComplementSetSubMatrices()`
     though they need not be the same matrices.

     This can only be called after `MatCreateSchurComplement()` or `MatSchurComplementSetSubMatrices()`, it cannot be called immediately after `MatSetType`(S,`MATSCHURCOMPLEMENT`);

   Developer Note:
     This code is almost identical to `MatSchurComplementSetSubMatrices()`. The API should be refactored.

.seealso: [](chapter_ksp), `Mat`, `MatCreateNormal()`, `MatMult()`, `MatCreate()`, `MatSchurComplementGetKSP()`, `MatCreateSchurComplement()`
@*/
PetscErrorCode MatSchurComplementUpdateSubMatrices(Mat S, Mat A00, Mat Ap00, Mat A01, Mat A10, Mat A11)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)S->data;
  PetscBool            isschur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)S, MATSCHURCOMPLEMENT, &isschur));
  if (!isschur) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(S->assembled, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_WRONGSTATE, "Use MatSchurComplementSetSubMatrices() for a new matrix");
  PetscValidHeaderSpecific(A00, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(Ap00, MAT_CLASSID, 3);
  PetscValidHeaderSpecific(A01, MAT_CLASSID, 4);
  PetscValidHeaderSpecific(A10, MAT_CLASSID, 5);
  PetscCheckSameComm(A00, 2, Ap00, 3);
  PetscCheckSameComm(A00, 2, A01, 4);
  PetscCheckSameComm(A00, 2, A10, 5);
  PetscCheck(A00->rmap->n == A00->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of A00 %" PetscInt_FMT " do not equal local columns %" PetscInt_FMT, A00->rmap->n, A00->cmap->n);
  PetscCheck(A00->rmap->n == Ap00->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of A00 %" PetscInt_FMT " do not equal local rows of Ap00 %" PetscInt_FMT, A00->rmap->n, Ap00->rmap->n);
  PetscCheck(Ap00->rmap->n == Ap00->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of Ap00 %" PetscInt_FMT " do not equal local columns %" PetscInt_FMT, Ap00->rmap->n, Ap00->cmap->n);
  PetscCheck(A00->cmap->n == A01->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local columns of A00 %" PetscInt_FMT " do not equal local rows of A01 %" PetscInt_FMT, A00->cmap->n, A01->rmap->n);
  PetscCheck(A10->cmap->n == A00->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local columns of A10 %" PetscInt_FMT " do not equal local rows of A00 %" PetscInt_FMT, A10->cmap->n, A00->rmap->n);
  if (A11) {
    PetscValidHeaderSpecific(A11, MAT_CLASSID, 6);
    PetscCheckSameComm(A00, 2, A11, 6);
    PetscCheck(A10->rmap->n == A11->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows of A10 %" PetscInt_FMT " do not equal local rows A11 %" PetscInt_FMT, A10->rmap->n, A11->rmap->n);
  }

  PetscCall(PetscObjectReference((PetscObject)A00));
  PetscCall(PetscObjectReference((PetscObject)Ap00));
  PetscCall(PetscObjectReference((PetscObject)A01));
  PetscCall(PetscObjectReference((PetscObject)A10));
  if (A11) PetscCall(PetscObjectReference((PetscObject)A11));

  PetscCall(MatDestroy(&Na->A));
  PetscCall(MatDestroy(&Na->Ap));
  PetscCall(MatDestroy(&Na->B));
  PetscCall(MatDestroy(&Na->C));
  PetscCall(MatDestroy(&Na->D));

  Na->A  = A00;
  Na->Ap = Ap00;
  Na->B  = A01;
  Na->C  = A10;
  Na->D  = A11;

  PetscCall(KSPSetOperators(Na->ksp, A00, Ap00));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSchurComplementGetSubMatrices - Get the individual submatrices in the Schur complement

  Collective

  Input Parameter:
. S    - matrix obtained with `MatCreateSchurComplement()` (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01

  Output Parameters:
+ A00  - the upper-left block of the original matrix A = [A00 A01; A10 A11]
. Ap00 - preconditioning matrix for use in ksp(A00,Ap00) to approximate the action of A^{-1}
. A01  - the upper-right block of the original matrix A = [A00 A01; A10 A11]
. A10  - the lower-left block of the original matrix A = [A00 A01; A10 A11]
- A11  - (optional) the lower-right block of the original matrix A = [A00 A01; A10 A11]

  Level: intermediate

  Note:
  `A11` is optional, and thus can be `NULL`.

  The reference counts of the submatrices are not increased before they are returned and the matrices should not be modified or destroyed.

.seealso: [](chapter_ksp), `MatCreateNormal()`, `MatMult()`, `MatCreate()`, `MatSchurComplementGetKSP()`, `MatCreateSchurComplement()`, `MatSchurComplementUpdateSubMatrices()`
@*/
PetscErrorCode MatSchurComplementGetSubMatrices(Mat S, Mat *A00, Mat *Ap00, Mat *A01, Mat *A10, Mat *A11)
{
  Mat_SchurComplement *Na = (Mat_SchurComplement *)S->data;
  PetscBool            flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)S, MATSCHURCOMPLEMENT, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_WRONG, "Not for type %s", ((PetscObject)S)->type_name);
  if (A00) *A00 = Na->A;
  if (Ap00) *Ap00 = Na->Ap;
  if (A01) *A01 = Na->B;
  if (A10) *A10 = Na->C;
  if (A11) *A11 = Na->D;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/kspimpl.h>

/*@
  MatSchurComplementComputeExplicitOperator - Compute the Schur complement matrix explicitly

  Collective

  Input Parameter:
. M - the matrix obtained with `MatCreateSchurComplement()`

  Output Parameter:
. S - the Schur complement matrix

  Notes:
    This can be expensive, so it is mainly for testing

    Use `MatSchurComplementGetPmat()` to get a sparse approximation for the Schur complement suitable for use in building a preconditioner

  Level: advanced

.seealso: [](chapter_ksp), `MatCreateSchurComplement()`, `MatSchurComplementUpdate()`, `MatSchurComplementGetPmat()`
@*/
PetscErrorCode MatSchurComplementComputeExplicitOperator(Mat A, Mat *S)
{
  Mat       B, C, D, E = NULL, Bd, AinvBd;
  KSP       ksp;
  PetscInt  n, N, m, M;
  PetscBool flg = PETSC_FALSE, set, symm;

  PetscFunctionBegin;
  PetscCall(MatSchurComplementGetSubMatrices(A, NULL, NULL, &B, &C, &D));
  PetscCall(MatSchurComplementGetKSP(A, &ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(MatConvert(B, MATDENSE, MAT_INITIAL_MATRIX, &Bd));
  PetscCall(MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd));
  PetscCall(KSPMatSolve(ksp, Bd, AinvBd));
  PetscCall(MatDestroy(&Bd));
  PetscCall(MatChop(AinvBd, PETSC_SMALL));
  if (D) {
    PetscCall(MatGetLocalSize(D, &m, &n));
    PetscCall(MatGetSize(D, &M, &N));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A), m, n, M, N, NULL, S));
  }
  PetscCall(MatMatMult(C, AinvBd, D ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, PETSC_DEFAULT, S));
  PetscCall(MatDestroy(&AinvBd));
  if (D) {
    PetscCall(PetscObjectTypeCompareAny((PetscObject)D, &flg, MATSEQSBAIJ, MATMPISBAIJ, ""));
    if (flg) {
      PetscCall(MatIsSymmetricKnown(A, &set, &symm));
      if (!set || !symm) PetscCall(MatConvert(D, MATBAIJ, MAT_INITIAL_MATRIX, &E)); /* convert the (1,1) block to nonsymmetric storage for MatAXPY() */
    }
    PetscCall(MatAXPY(*S, -1.0, E ? E : D, DIFFERENT_NONZERO_PATTERN)); /* calls Mat[Get|Restore]RowUpperTriangular(), so only the upper triangular part is valid with symmetric storage */
  }
  PetscCall(MatConvert(*S, !E && flg ? MATSBAIJ : MATAIJ, MAT_INPLACE_MATRIX, S)); /* if A is symmetric and the (1,1) block is a MatSBAIJ, return S as a MatSBAIJ */
  PetscCall(MatScale(*S, -1.0));
  PetscCall(MatDestroy(&E));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Developer Notes:
    This should be implemented with a MatCreate_SchurComplement() as that is the standard design for new Mat classes. */
PetscErrorCode MatGetSchurComplement_Basic(Mat mat, IS isrow0, IS iscol0, IS isrow1, IS iscol1, MatReuse mreuse, Mat *S, MatSchurComplementAinvType ainvtype, MatReuse preuse, Mat *Sp)
{
  Mat      A = NULL, Ap = NULL, B = NULL, C = NULL, D = NULL;
  MatReuse reuse;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscValidType(mat, 1);
  PetscValidHeaderSpecific(isrow0, IS_CLASSID, 2);
  PetscValidHeaderSpecific(iscol0, IS_CLASSID, 3);
  PetscValidHeaderSpecific(isrow1, IS_CLASSID, 4);
  PetscValidHeaderSpecific(iscol1, IS_CLASSID, 5);
  PetscValidLogicalCollectiveEnum(mat, mreuse, 6);
  PetscValidLogicalCollectiveEnum(mat, ainvtype, 8);
  PetscValidLogicalCollectiveEnum(mat, preuse, 9);
  if (mreuse == MAT_IGNORE_MATRIX && preuse == MAT_IGNORE_MATRIX) PetscFunctionReturn(PETSC_SUCCESS);
  if (mreuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*S, MAT_CLASSID, 7);
  if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*Sp, MAT_CLASSID, 10);

  PetscCheck(!mat->factortype, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");

  reuse = MAT_INITIAL_MATRIX;
  if (mreuse == MAT_REUSE_MATRIX) {
    PetscCall(MatSchurComplementGetSubMatrices(*S, &A, &Ap, &B, &C, &D));
    PetscCheck(A && Ap && B && C, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Attempting to reuse matrix but Schur complement matrices unset");
    PetscCheck(A == Ap, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Preconditioning matrix does not match operator");
    PetscCall(MatDestroy(&Ap)); /* get rid of extra reference */
    reuse = MAT_REUSE_MATRIX;
  }
  PetscCall(MatCreateSubMatrix(mat, isrow0, iscol0, reuse, &A));
  PetscCall(MatCreateSubMatrix(mat, isrow0, iscol1, reuse, &B));
  PetscCall(MatCreateSubMatrix(mat, isrow1, iscol0, reuse, &C));
  PetscCall(MatCreateSubMatrix(mat, isrow1, iscol1, reuse, &D));
  switch (mreuse) {
  case MAT_INITIAL_MATRIX:
    PetscCall(MatCreateSchurComplement(A, A, B, C, D, S));
    break;
  case MAT_REUSE_MATRIX:
    PetscCall(MatSchurComplementUpdateSubMatrices(*S, A, A, B, C, D));
    break;
  default:
    PetscCheck(mreuse == MAT_IGNORE_MATRIX, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Unrecognized value of mreuse %d", (int)mreuse);
  }
  if (preuse != MAT_IGNORE_MATRIX) PetscCall(MatCreateSchurComplementPmat(A, B, C, D, ainvtype, preuse, Sp));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    MatGetSchurComplement - Obtain the Schur complement from eliminating part of the matrix in another part.

    Collective

    Input Parameters:
+   A      - matrix in which the complement is to be taken
.   isrow0 - rows to eliminate
.   iscol0 - columns to eliminate, (isrow0,iscol0) should be square and nonsingular
.   isrow1 - rows in which the Schur complement is formed
.   iscol1 - columns in which the Schur complement is formed
.   mreuse - `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`, use `MAT_IGNORE_MATRIX` to put nothing in S
.   ainvtype - the type of approximation used for the inverse of the (0,0) block used in forming Sp:
                       `MAT_SCHUR_COMPLEMENT_AINV_DIAG`, `MAT_SCHUR_COMPLEMENT_AINV_LUMP`, `MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG`, or `MAT_SCHUR_COMPLEMENT_AINV_FULL`
-   preuse - `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`, use `MAT_IGNORE_MATRIX` to put nothing in Sp

    Output Parameters:
+   S      - exact Schur complement, often of type `MATSCHURCOMPLEMENT` which is difficult to use for preconditioning
-   Sp     - approximate Schur complement from which a preconditioner can be built A11 - A10 inv(DIAGFORM(A00)) A01

    Level: advanced

    Notes:
    Since the real Schur complement is usually dense, providing a good approximation to Sp usually requires
    application-specific information.

    Sometimes users would like to provide problem-specific data in the Schur complement, usually only for special row
    and column index sets.  In that case, the user should call `PetscObjectComposeFunction()` on the *S matrix and pass mreuse of `MAT_REUSE_MATRIX` to set
    "MatGetSchurComplement_C" to their function.  If their function needs to fall back to the default implementation, it
    should call `MatGetSchurComplement_Basic()`.

    `MatCreateSchurComplement()` takes as arguments the four submatrices and returns the virtual Schur complement (what this function returns in S).

    `MatSchurComplementGetPmat()` takes the virtual Schur complement and returns an explicit approximate Schur complement (what this returns in Sp).

    In other words calling `MatCreateSchurComplement()` followed by `MatSchurComplementGetPmat()` produces the same output as this function but with slightly different
    inputs. The actually submatrices of the original block matrix instead of index sets to the submatrices.

    Developer Note:
    The API that includes `MatGetSchurComplement()`, `MatCreateSchurComplement()`, `MatSchurComplementGetPmat()` should be refactored to
    remove redundancy and be clearer and simpler.

.seealso: [](chapter_ksp), `MatCreateSubMatrix()`, `PCFIELDSPLIT`, `MatCreateSchurComplement()`, `MatSchurComplementAinvType`
@*/
PetscErrorCode MatGetSchurComplement(Mat A, IS isrow0, IS iscol0, IS isrow1, IS iscol1, MatReuse mreuse, Mat *S, MatSchurComplementAinvType ainvtype, MatReuse preuse, Mat *Sp)
{
  PetscErrorCode (*f)(Mat, IS, IS, IS, IS, MatReuse, Mat *, MatReuse, Mat *) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(isrow0, IS_CLASSID, 2);
  PetscValidHeaderSpecific(iscol0, IS_CLASSID, 3);
  PetscValidHeaderSpecific(isrow1, IS_CLASSID, 4);
  PetscValidHeaderSpecific(iscol1, IS_CLASSID, 5);
  PetscValidLogicalCollectiveEnum(A, mreuse, 6);
  if (mreuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*S, MAT_CLASSID, 7);
  PetscValidLogicalCollectiveEnum(A, ainvtype, 8);
  PetscValidLogicalCollectiveEnum(A, preuse, 9);
  if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*Sp, MAT_CLASSID, 10);
  PetscValidType(A, 1);
  PetscCheck(!A->factortype, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  if (mreuse == MAT_REUSE_MATRIX) { /* This is the only situation, in which we can demand that the user pass a non-NULL pointer to non-garbage in S. */
    PetscCall(PetscObjectQueryFunction((PetscObject)*S, "MatGetSchurComplement_C", &f));
  }
  if (f) PetscCall((*f)(A, isrow0, iscol0, isrow1, iscol1, mreuse, S, preuse, Sp));
  else PetscCall(MatGetSchurComplement_Basic(A, isrow0, iscol0, isrow1, iscol1, mreuse, S, ainvtype, preuse, Sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    MatSchurComplementSetAinvType - set the type of approximation used for the inverse of the (0,0) block used in forming Sp in `MatSchurComplementGetPmat()`

    Not Collective

    Input Parameters:
+   S        - matrix obtained with `MatCreateSchurComplement()` (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01
-   ainvtype - type of approximation to be used to form approximate Schur complement Sp = A11 - A10 inv(DIAGFORM(A00)) A01:
                      `MAT_SCHUR_COMPLEMENT_AINV_DIAG`, `MAT_SCHUR_COMPLEMENT_AINV_LUMP`, `MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG`, or `MAT_SCHUR_COMPLEMENT_AINV_FULL`

    Options Database Key:
    -mat_schur_complement_ainv_type diag | lump | blockdiag | full

    Level: advanced

.seealso: [](chapter_ksp), `MatSchurComplementAinvType`, `MatCreateSchurComplement()`, `MatGetSchurComplement()`, `MatSchurComplementGetPmat()`, `MatSchurComplementGetAinvType()`
@*/
PetscErrorCode MatSchurComplementSetAinvType(Mat S, MatSchurComplementAinvType ainvtype)
{
  PetscBool            isschur;
  Mat_SchurComplement *schur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)S, MATSCHURCOMPLEMENT, &isschur));
  if (!isschur) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidLogicalCollectiveEnum(S, ainvtype, 2);
  schur = (Mat_SchurComplement *)S->data;
  PetscCheck(ainvtype == MAT_SCHUR_COMPLEMENT_AINV_DIAG || ainvtype == MAT_SCHUR_COMPLEMENT_AINV_LUMP || ainvtype == MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG || ainvtype == MAT_SCHUR_COMPLEMENT_AINV_FULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown MatSchurComplementAinvType: %d", (int)ainvtype);
  schur->ainvtype = ainvtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    MatSchurComplementGetAinvType - get the type of approximation for the inverse of the (0,0) block used in forming Sp in `MatSchurComplementGetPmat()`

    Not Collective

    Input Parameter:
.   S      - matrix obtained with `MatCreateSchurComplement()` (or equivalent) and implementing the action of A11 - A10 ksp(A00,Ap00) A01

    Output Parameter:
.   ainvtype - type of approximation used to form approximate Schur complement Sp = A11 - A10 inv(DIAGFORM(A00)) A01:
                      `MAT_SCHUR_COMPLEMENT_AINV_DIAG`, `MAT_SCHUR_COMPLEMENT_AINV_LUMP`, `MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG`, or `MAT_SCHUR_COMPLEMENT_AINV_FULL`

    Level: advanced

.seealso: [](chapter_ksp), `MatSchurComplementAinvType`, `MatCreateSchurComplement()`, `MatGetSchurComplement()`, `MatSchurComplementGetPmat()`, `MatSchurComplementSetAinvType()`
@*/
PetscErrorCode MatSchurComplementGetAinvType(Mat S, MatSchurComplementAinvType *ainvtype)
{
  PetscBool            isschur;
  Mat_SchurComplement *schur;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)S, MATSCHURCOMPLEMENT, &isschur));
  PetscCheck(isschur, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_WRONG, "Not for type %s", ((PetscObject)S)->type_name);
  schur = (Mat_SchurComplement *)S->data;
  if (ainvtype) *ainvtype = schur->ainvtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    MatCreateSchurComplementPmat - create a preconditioning matrix for the Schur complement by explicitly assembling the sparse matrix
        Sp = A11 - A10 inv(DIAGFORM(A00)) A01

    Collective

    Input Parameters:
+   A00      - the upper-left part of the original matrix A = [A00 A01; A10 A11]
.   A01      - (optional) the upper-right part of the original matrix A = [A00 A01; A10 A11]
.   A10      - (optional) the lower-left part of the original matrix A = [A00 A01; A10 A11]
.   A11      - (optional) the lower-right part of the original matrix A = [A00 A01; A10 A11]
.   ainvtype - type of approximation for DIAGFORM(A00) used when forming Sp = A11 - A10 inv(DIAGFORM(A00)) A01. See MatSchurComplementAinvType.
-   preuse   - `MAT_INITIAL_MATRIX` for a new Sp, or `MAT_REUSE_MATRIX` to reuse an existing Sp, or `MAT_IGNORE_MATRIX` to put nothing in Sp

    Output Parameter:
-   Sp    - approximate Schur complement suitable for preconditioning the true Schur complement S = A11 - A10 inv(A00) A01

    Level: advanced

.seealso: [](chapter_ksp), `MatCreateSchurComplement()`, `MatGetSchurComplement()`, `MatSchurComplementGetPmat()`, `MatSchurComplementAinvType`
@*/
PetscErrorCode MatCreateSchurComplementPmat(Mat A00, Mat A01, Mat A10, Mat A11, MatSchurComplementAinvType ainvtype, MatReuse preuse, Mat *Sp)
{
  PetscInt N00;

  PetscFunctionBegin;
  /* Use an appropriate approximate inverse of A00 to form A11 - A10 inv(DIAGFORM(A00)) A01; a NULL A01, A10 or A11 indicates a zero matrix. */
  /* TODO: Perhaps should create an appropriately-sized zero matrix of the same type as A00? */
  PetscValidLogicalCollectiveEnum(A11, preuse, 6);
  if (preuse == MAT_IGNORE_MATRIX) PetscFunctionReturn(PETSC_SUCCESS);

  /* A zero size A00 or empty A01 or A10 imply S = A11. */
  PetscCall(MatGetSize(A00, &N00, NULL));
  if (!A01 || !A10 || !N00) {
    if (preuse == MAT_INITIAL_MATRIX) {
      PetscCall(MatDuplicate(A11, MAT_COPY_VALUES, Sp));
    } else { /* MAT_REUSE_MATRIX */
      /* TODO: when can we pass SAME_NONZERO_PATTERN? */
      PetscCall(MatCopy(A11, *Sp, DIFFERENT_NONZERO_PATTERN));
    }
  } else {
    Mat AdB;
    Vec diag;

    if (ainvtype == MAT_SCHUR_COMPLEMENT_AINV_LUMP || ainvtype == MAT_SCHUR_COMPLEMENT_AINV_DIAG) {
      PetscCall(MatDuplicate(A01, MAT_COPY_VALUES, &AdB));
      PetscCall(MatCreateVecs(A00, &diag, NULL));
      if (ainvtype == MAT_SCHUR_COMPLEMENT_AINV_LUMP) {
        PetscCall(MatGetRowSum(A00, diag));
      } else {
        PetscCall(MatGetDiagonal(A00, diag));
      }
      PetscCall(VecReciprocal(diag));
      PetscCall(MatDiagonalScale(AdB, diag, NULL));
      PetscCall(VecDestroy(&diag));
    } else if (ainvtype == MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG) {
      Mat      A00_inv;
      MatType  type;
      MPI_Comm comm;

      PetscCall(PetscObjectGetComm((PetscObject)A00, &comm));
      PetscCall(MatGetType(A00, &type));
      PetscCall(MatCreate(comm, &A00_inv));
      PetscCall(MatSetType(A00_inv, type));
      PetscCall(MatInvertBlockDiagonalMat(A00, A00_inv));
      PetscCall(MatMatMult(A00_inv, A01, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AdB));
      PetscCall(MatDestroy(&A00_inv));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown MatSchurComplementAinvType: %d", ainvtype);
    /* Cannot really reuse Sp in MatMatMult() because of MatAYPX() -->
         MatAXPY() --> MatHeaderReplace() --> MatDestroy_XXX_MatMatMult()  */
    if (preuse == MAT_REUSE_MATRIX) PetscCall(MatDestroy(Sp));
    PetscCall(MatMatMult(A10, AdB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Sp));
    if (!A11) {
      PetscCall(MatScale(*Sp, -1.0));
    } else {
      /* TODO: when can we pass SAME_NONZERO_PATTERN? */
      PetscCall(MatAYPX(*Sp, -1, A11, DIFFERENT_NONZERO_PATTERN));
    }
    PetscCall(MatDestroy(&AdB));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSchurComplementGetPmat_Basic(Mat S, MatReuse preuse, Mat *Sp)
{
  Mat                  A, B, C, D;
  Mat_SchurComplement *schur = (Mat_SchurComplement *)S->data;

  PetscFunctionBegin;
  if (preuse == MAT_IGNORE_MATRIX) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatSchurComplementGetSubMatrices(S, &A, NULL, &B, &C, &D));
  PetscCheck(A, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_WRONGSTATE, "Schur complement component matrices unset");
  if (schur->ainvtype != MAT_SCHUR_COMPLEMENT_AINV_FULL) PetscCall(MatCreateSchurComplementPmat(A, B, C, D, schur->ainvtype, preuse, Sp));
  else {
    if (preuse == MAT_REUSE_MATRIX) PetscCall(MatDestroy(Sp));
    PetscCall(MatSchurComplementComputeExplicitOperator(S, Sp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    MatSchurComplementGetPmat - Obtain a preconditioning matrix for the Schur complement by assembling Sp = A11 - A10 inv(DIAGFORM(A00)) A01

    Collective

    Input Parameters:
+   S      - matrix obtained with MatCreateSchurComplement() (or equivalent) that implements the action of A11 - A10 ksp(A00,Ap00) A01
-   preuse - `MAT_INITIAL_MATRIX` for a new Sp, or `MAT_REUSE_MATRIX` to reuse an existing Sp, or `MAT_IGNORE_MATRIX` to put nothing in Sp

    Output Parameter:
-   Sp     - approximate Schur complement suitable for preconditioning the exact Schur complement S = A11 - A10 inv(A00) A01

    Level: advanced

    Notes:
    The approximation of Sp depends on the the argument passed to to `MatSchurComplementSetAinvType()`
    `MAT_SCHUR_COMPLEMENT_AINV_DIAG`, `MAT_SCHUR_COMPLEMENT_AINV_LUMP`, `MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG`, or `MAT_SCHUR_COMPLEMENT_AINV_FULL`
    -mat_schur_complement_ainv_type <diag,lump,blockdiag,full>

    Sometimes users would like to provide problem-specific data in the Schur complement, usually only
    for special row and column index sets.  In that case, the user should call `PetscObjectComposeFunction()` to set
    "MatSchurComplementGetPmat_C" to their function.  If their function needs to fall back to the default implementation,
    it should call `MatSchurComplementGetPmat_Basic()`.

    Developer Note:
    The API that includes `MatGetSchurComplement()`, `MatCreateSchurComplement()`, `MatSchurComplementGetPmat()` should be refactored to
    remove redundancy and be clearer and simpler.

    This routine should be called `MatSchurComplementCreatePmat()`

.seealso: [](chapter_ksp), `MatCreateSubMatrix()`, `PCFIELDSPLIT`, `MatGetSchurComplement()`, `MatCreateSchurComplement()`, `MatSchurComplementSetAinvType()`
@*/
PetscErrorCode MatSchurComplementGetPmat(Mat S, MatReuse preuse, Mat *Sp)
{
  PetscErrorCode (*f)(Mat, MatReuse, Mat *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S, MAT_CLASSID, 1);
  PetscValidType(S, 1);
  PetscValidLogicalCollectiveEnum(S, preuse, 2);
  if (preuse != MAT_IGNORE_MATRIX) {
    PetscValidPointer(Sp, 3);
    if (preuse == MAT_INITIAL_MATRIX) *Sp = NULL;
    if (preuse == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*Sp, MAT_CLASSID, 3);
  }
  PetscCheck(!S->factortype, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");

  PetscCall(PetscObjectQueryFunction((PetscObject)S, "MatSchurComplementGetPmat_C", &f));
  if (f) PetscCall((*f)(S, preuse, Sp));
  else PetscCall(MatSchurComplementGetPmat_Basic(S, preuse, Sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_SchurComplement_Dense(Mat C)
{
  Mat_Product         *product = C->product;
  Mat_SchurComplement *Na      = (Mat_SchurComplement *)product->A->data;
  Mat                  work1, work2;
  PetscScalar         *v;
  PetscInt             lda;

  PetscFunctionBegin;
  PetscCall(MatMatMult(Na->B, product->B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &work1));
  PetscCall(MatDuplicate(work1, MAT_DO_NOT_COPY_VALUES, &work2));
  PetscCall(KSPMatSolve(Na->ksp, work1, work2));
  PetscCall(MatDestroy(&work1));
  PetscCall(MatDenseGetArrayWrite(C, &v));
  PetscCall(MatDenseGetLDA(C, &lda));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)C), C->rmap->n, C->cmap->n, C->rmap->N, C->cmap->N, v, &work1));
  PetscCall(MatDenseSetLDA(work1, lda));
  PetscCall(MatMatMult(Na->C, work2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &work1));
  PetscCall(MatDenseRestoreArrayWrite(C, &v));
  PetscCall(MatDestroy(&work2));
  PetscCall(MatDestroy(&work1));
  if (Na->D) {
    PetscCall(MatMatMult(Na->D, product->B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &work1));
    PetscCall(MatAYPX(C, -1.0, work1, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&work1));
  } else PetscCall(MatScale(C, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_SchurComplement_Dense(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, B = product->B;
  PetscInt     m = A->rmap->n, n = B->cmap->n, M = A->rmap->N, N = B->cmap->N;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(MatSetSizes(C, m, n, M, N));
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)C, &flg, MATSEQDENSE, MATMPIDENSE, ""));
  if (!flg) {
    PetscCall(MatSetType(C, ((PetscObject)B)->type_name));
    C->ops->productsymbolic = MatProductSymbolic_SchurComplement_Dense;
  }
  PetscCall(MatSetUp(C));
  C->ops->productnumeric = MatProductNumeric_SchurComplement_Dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_Dense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_SchurComplement_Dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_SchurComplement_Dense(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  PetscCheck(product->type == MATPRODUCT_AB, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Not for product type %s", MatProductTypes[product->type]);
  PetscCall(MatProductSetFromOptions_Dense_AB(C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSCHURCOMPLEMENT -  "schurcomplement" - Matrix type that behaves like the Schur complement of a matrix.

  Level: intermediate

.seealso: [](chapter_matrices), `Mat`, `MatCreate()`, `MatType`, `MatCreateSchurComplement()`, `MatSchurComplementComputeExplicitOperator()`,
          `MatSchurComplementGetSubMatrices()`, `MatSchurComplementGetKSP()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SchurComplement(Mat N)
{
  Mat_SchurComplement *Na;

  PetscFunctionBegin;
  PetscCall(PetscNew(&Na));
  N->data = (void *)Na;

  N->ops->destroy        = MatDestroy_SchurComplement;
  N->ops->getvecs        = MatCreateVecs_SchurComplement;
  N->ops->view           = MatView_SchurComplement;
  N->ops->mult           = MatMult_SchurComplement;
  N->ops->multtranspose  = MatMultTranspose_SchurComplement;
  N->ops->multadd        = MatMultAdd_SchurComplement;
  N->ops->setfromoptions = MatSetFromOptions_SchurComplement;
  N->assembled           = PETSC_FALSE;
  N->preallocated        = PETSC_FALSE;

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)N), &Na->ksp));
  PetscCall(PetscObjectChangeTypeName((PetscObject)N, MATSCHURCOMPLEMENT));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_schurcomplement_seqdense_C", MatProductSetFromOptions_SchurComplement_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_schurcomplement_mpidense_C", MatProductSetFromOptions_SchurComplement_Dense));
  PetscFunctionReturn(PETSC_SUCCESS);
}
