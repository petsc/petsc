
#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

PETSC_EXTERN PetscErrorCode VecGetRootType_Private(Vec, VecType *);

typedef struct {
  Mat A;            /* sparse matrix */
  Mat U, V;         /* dense tall-skinny matrices */
  Vec c;            /* sequential vector containing the diagonal of C */
  Vec work1, work2; /* sequential vectors that hold partial products */
  Vec xl, yl;       /* auxiliary sequential vectors for matmult operation */
} Mat_LRC;

static PetscErrorCode MatMult_LRC_kernel(Mat N, Vec x, Vec y, PetscBool transpose)
{
  Mat_LRC    *Na = (Mat_LRC *)N->data;
  PetscMPIInt size;
  Mat         U, V;

  PetscFunctionBegin;
  U = transpose ? Na->V : Na->U;
  V = transpose ? Na->U : Na->V;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)N), &size));
  if (size == 1) {
    PetscCall(MatMultHermitianTranspose(V, x, Na->work1));
    if (Na->c) PetscCall(VecPointwiseMult(Na->work1, Na->c, Na->work1));
    if (Na->A) {
      if (transpose) {
        PetscCall(MatMultTranspose(Na->A, x, y));
      } else {
        PetscCall(MatMult(Na->A, x, y));
      }
      PetscCall(MatMultAdd(U, Na->work1, y, y));
    } else {
      PetscCall(MatMult(U, Na->work1, y));
    }
  } else {
    Mat                Uloc, Vloc;
    Vec                yl, xl;
    const PetscScalar *w1;
    PetscScalar       *w2;
    PetscInt           nwork;
    PetscMPIInt        mpinwork;

    xl = transpose ? Na->yl : Na->xl;
    yl = transpose ? Na->xl : Na->yl;
    PetscCall(VecGetLocalVector(y, yl));
    PetscCall(MatDenseGetLocalMatrix(U, &Uloc));
    PetscCall(MatDenseGetLocalMatrix(V, &Vloc));

    /* multiply the local part of V with the local part of x */
    PetscCall(VecGetLocalVectorRead(x, xl));
    PetscCall(MatMultHermitianTranspose(Vloc, xl, Na->work1));
    PetscCall(VecRestoreLocalVectorRead(x, xl));

    /* form the sum of all the local multiplies: this is work2 = V'*x =
       sum_{all processors} work1 */
    PetscCall(VecGetArrayRead(Na->work1, &w1));
    PetscCall(VecGetArrayWrite(Na->work2, &w2));
    PetscCall(VecGetLocalSize(Na->work1, &nwork));
    PetscCall(PetscMPIIntCast(nwork, &mpinwork));
    PetscCall(MPIU_Allreduce(w1, w2, mpinwork, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)N)));
    PetscCall(VecRestoreArrayRead(Na->work1, &w1));
    PetscCall(VecRestoreArrayWrite(Na->work2, &w2));

    if (Na->c) { /* work2 = C*work2 */
      PetscCall(VecPointwiseMult(Na->work2, Na->c, Na->work2));
    }

    if (Na->A) {
      /* form y = A*x or A^t*x */
      if (transpose) {
        PetscCall(MatMultTranspose(Na->A, x, y));
      } else {
        PetscCall(MatMult(Na->A, x, y));
      }
      /* multiply-add y = y + U*work2 */
      PetscCall(MatMultAdd(Uloc, Na->work2, yl, yl));
    } else {
      /* multiply y = U*work2 */
      PetscCall(MatMult(Uloc, Na->work2, yl));
    }

    PetscCall(VecRestoreLocalVector(y, yl));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LRC(Mat N, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMult_LRC_kernel(N, x, y, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_LRC(Mat N, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMult_LRC_kernel(N, x, y, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_LRC(Mat N)
{
  Mat_LRC *Na = (Mat_LRC *)N->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&Na->A));
  PetscCall(MatDestroy(&Na->U));
  PetscCall(MatDestroy(&Na->V));
  PetscCall(VecDestroy(&Na->c));
  PetscCall(VecDestroy(&Na->work1));
  PetscCall(VecDestroy(&Na->work2));
  PetscCall(VecDestroy(&Na->xl));
  PetscCall(VecDestroy(&Na->yl));
  PetscCall(PetscFree(N->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatLRCGetMats_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLRCGetMats_LRC(Mat N, Mat *A, Mat *U, Vec *c, Mat *V)
{
  Mat_LRC *Na = (Mat_LRC *)N->data;

  PetscFunctionBegin;
  if (A) *A = Na->A;
  if (U) *U = Na->U;
  if (c) *c = Na->c;
  if (V) *V = Na->V;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatLRCGetMats - Returns the constituents of an LRC matrix

   Collective

   Input Parameter:
.  N - matrix of type `MATLRC`

   Output Parameters:
+  A - the (sparse) matrix
.  U - first dense rectangular (tall and skinny) matrix
.  c - a sequential vector containing the diagonal of C
-  V - second dense rectangular (tall and skinny) matrix

   Level: intermediate

   Notes:
   The returned matrices need not be destroyed by the caller.

   `U`, `c`, `V` may be `NULL` if not needed

.seealso: [](chapter_matrices), `Mat`, `MATLRC`, `MatCreateLRC()`
@*/
PetscErrorCode MatLRCGetMats(Mat N, Mat *A, Mat *U, Vec *c, Mat *V)
{
  PetscFunctionBegin;
  PetscUseMethod(N, "MatLRCGetMats_C", (Mat, Mat *, Mat *, Vec *, Mat *), (N, A, U, c, V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATLRC -  "lrc" - a matrix object that behaves like A + U*C*V'

  Note:
   The matrix A + U*C*V' is not formed! Rather the matrix  object performs the matrix-vector product `MatMult()`, by first multiplying by
   A and then adding the other term.

  Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MatCreateLRC()`, `MatMult()`, `MatLRCGetMats()`
M*/

/*@
   MatCreateLRC - Creates a new matrix object that behaves like A + U*C*V' of type `MATLRC`

   Collective

   Input Parameters:
+  A    - the (sparse) matrix (can be `NULL`)
.  U    - dense rectangular (tall and skinny) matrix
.  V    - dense rectangular (tall and skinny) matrix
-  c    - a vector containing the diagonal of C (can be `NULL`)

   Output Parameter:
.  N    - the matrix that represents A + U*C*V'

   Level: intermediate

   Notes:
   The matrix A + U*C*V' is not formed! Rather the new matrix
   object performs the matrix-vector product `MatMult()`, by first multiplying by
   A and then adding the other term.

   `C` is a diagonal matrix (represented as a vector) of order k,
   where k is the number of columns of both `U` and `V`.

   If `A` is `NULL` then the new object behaves like a low-rank matrix U*C*V'.

   Use `V`=`U` (or `V`=`NULL`) for a symmetric low-rank correction, A + U*C*U'.

   If `c` is `NULL` then the low-rank correction is just U*V'.
   If a sequential `c` vector is used for a parallel matrix,
   PETSc assumes that the values of the vector are consistently set across processors.

.seealso: [](chapter_matrices), `Mat`, `MATLRC`, `MatLRCGetMats()`
@*/
PetscErrorCode MatCreateLRC(Mat A, Mat U, Vec c, Mat V, Mat *N)
{
  PetscBool   match;
  PetscInt    m, n, k, m1, n1, k1;
  Mat_LRC    *Na;
  Mat         Uloc;
  PetscMPIInt size, csize = 0;

  PetscFunctionBegin;
  if (A) PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  if (c) PetscValidHeaderSpecific(c, VEC_CLASSID, 3);
  if (V) {
    PetscValidHeaderSpecific(V, MAT_CLASSID, 4);
    PetscCheckSameComm(U, 2, V, 4);
  }
  if (A) PetscCheckSameComm(A, 1, U, 2);

  if (!V) V = U;
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)U, &match, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(match, PetscObjectComm((PetscObject)U), PETSC_ERR_SUP, "Matrix U must be of type dense, found %s", ((PetscObject)U)->type_name);
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)V, &match, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(match, PetscObjectComm((PetscObject)U), PETSC_ERR_SUP, "Matrix V must be of type dense, found %s", ((PetscObject)V)->type_name);
  PetscCall(PetscStrcmp(U->defaultvectype, V->defaultvectype, &match));
  PetscCheck(match, PetscObjectComm((PetscObject)U), PETSC_ERR_ARG_WRONG, "Matrix U and V must have the same VecType %s != %s", U->defaultvectype, V->defaultvectype);
  if (A) {
    PetscCall(PetscStrcmp(A->defaultvectype, U->defaultvectype, &match));
    PetscCheck(match, PetscObjectComm((PetscObject)U), PETSC_ERR_ARG_WRONG, "Matrix A and U must have the same VecType %s != %s", A->defaultvectype, U->defaultvectype);
  }

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)U), &size));
  PetscCall(MatGetSize(U, NULL, &k));
  PetscCall(MatGetSize(V, NULL, &k1));
  PetscCheck(k == k1, PetscObjectComm((PetscObject)U), PETSC_ERR_ARG_INCOMP, "U and V have different number of columns (%" PetscInt_FMT " vs %" PetscInt_FMT ")", k, k1);
  PetscCall(MatGetLocalSize(U, &m, NULL));
  PetscCall(MatGetLocalSize(V, &n, NULL));
  if (A) {
    PetscCall(MatGetLocalSize(A, &m1, &n1));
    PetscCheck(m == m1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local dimensions of U %" PetscInt_FMT " and A %" PetscInt_FMT " do not match", m, m1);
    PetscCheck(n == n1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local dimensions of V %" PetscInt_FMT " and A %" PetscInt_FMT " do not match", n, n1);
  }
  if (c) {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)c), &csize));
    PetscCall(VecGetSize(c, &k1));
    PetscCheck(k == k1, PetscObjectComm((PetscObject)c), PETSC_ERR_ARG_INCOMP, "The length of c %" PetscInt_FMT " does not match the number of columns of U and V (%" PetscInt_FMT ")", k1, k);
    PetscCheck(csize == 1 || csize == size, PetscObjectComm((PetscObject)c), PETSC_ERR_ARG_INCOMP, "U and c must have the same communicator size %d != %d", size, csize);
  }

  PetscCall(MatCreate(PetscObjectComm((PetscObject)U), N));
  PetscCall(MatSetSizes(*N, m, n, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetVecType(*N, U->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATLRC));
  /* Flag matrix as symmetric if A is symmetric and U == V */
  PetscCall(MatSetOption(*N, MAT_SYMMETRIC, (PetscBool)((A ? A->symmetric == PETSC_BOOL3_TRUE : PETSC_TRUE) && U == V)));

  PetscCall(PetscNew(&Na));
  (*N)->data = (void *)Na;
  Na->A      = A;
  Na->U      = U;
  Na->c      = c;
  Na->V      = V;

  PetscCall(PetscObjectReference((PetscObject)A));
  PetscCall(PetscObjectReference((PetscObject)Na->U));
  PetscCall(PetscObjectReference((PetscObject)Na->V));
  PetscCall(PetscObjectReference((PetscObject)c));

  PetscCall(MatDenseGetLocalMatrix(Na->U, &Uloc));
  PetscCall(MatCreateVecs(Uloc, &Na->work1, NULL));
  if (size != 1) {
    Mat Vloc;

    if (Na->c && csize != 1) { /* scatter parallel vector to sequential */
      VecScatter sct;

      PetscCall(VecScatterCreateToAll(Na->c, &sct, &c));
      PetscCall(VecScatterBegin(sct, Na->c, c, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sct, Na->c, c, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterDestroy(&sct));
      PetscCall(VecDestroy(&Na->c));
      Na->c = c;
    }
    PetscCall(MatDenseGetLocalMatrix(Na->V, &Vloc));
    PetscCall(VecDuplicate(Na->work1, &Na->work2));
    PetscCall(MatCreateVecs(Vloc, NULL, &Na->xl));
    PetscCall(MatCreateVecs(Uloc, NULL, &Na->yl));
  }

  /* Internally create a scaling vector if roottypes do not match */
  if (Na->c) {
    VecType rt1, rt2;

    PetscCall(VecGetRootType_Private(Na->work1, &rt1));
    PetscCall(VecGetRootType_Private(Na->c, &rt2));
    PetscCall(PetscStrcmp(rt1, rt2, &match));
    if (!match) {
      PetscCall(VecDuplicate(Na->c, &c));
      PetscCall(VecCopy(Na->c, c));
      PetscCall(VecDestroy(&Na->c));
      Na->c = c;
    }
  }

  (*N)->ops->destroy       = MatDestroy_LRC;
  (*N)->ops->mult          = MatMult_LRC;
  (*N)->ops->multtranspose = MatMultTranspose_LRC;

  (*N)->assembled    = PETSC_TRUE;
  (*N)->preallocated = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)(*N), "MatLRCGetMats_C", MatLRCGetMats_LRC));
  PetscCall(MatSetUp(*N));
  PetscFunctionReturn(PETSC_SUCCESS);
}
