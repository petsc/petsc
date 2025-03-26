#include <petsc/private/petscimpl.h>
#include <petscmat.h>
#include <petscblaslapack.h>
#include <petscdevice.h>
#include "lmproducts.h"
#include "blas_cyclic/blas_cyclic.h"

PetscLogEvent LMPROD_Mult, LMPROD_Solve, LMPROD_Update;

PETSC_INTERN PetscErrorCode LMProductsCreate(LMBasis basis, LMBlockType block_type, LMProducts *dots)
{
  PetscInt m, m_local;

  PetscFunctionBegin;
  PetscAssertPointer(basis, 1);
  PetscValidHeaderSpecific(basis->vecs, MAT_CLASSID, 1);
  PetscCheck(block_type >= 0 && block_type < LMBLOCK_END, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Invalid LMBlockType");
  PetscCall(PetscNew(dots));
  (*dots)->m = m      = basis->m;
  (*dots)->block_type = block_type;
  PetscCall(MatGetLocalSize(basis->vecs, NULL, &m_local));
  (*dots)->m_local = m_local;
  if (block_type == LMBLOCK_DIAGONAL) {
    VecType vec_type;

    PetscCall(MatCreateVecs(basis->vecs, &(*dots)->diagonal_global, NULL));
    PetscCall(VecCreateLocalVector((*dots)->diagonal_global, &(*dots)->diagonal_local));
    PetscCall(VecGetType((*dots)->diagonal_local, &vec_type));
    PetscCall(VecCreate(PETSC_COMM_SELF, &(*dots)->diagonal_dup));
    PetscCall(VecSetSizes((*dots)->diagonal_dup, m, m));
    PetscCall(VecSetType((*dots)->diagonal_dup, vec_type));
    PetscCall(VecSetUp((*dots)->diagonal_dup));
  } else {
    VecType vec_type;

    PetscCall(MatGetVecType(basis->vecs, &vec_type));
    PetscCall(MatCreateDenseFromVecType(PetscObjectComm((PetscObject)basis->vecs), vec_type, m_local, m_local, m, m, m_local, NULL, &(*dots)->full));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsDestroy(LMProducts *dots_p)
{
  PetscFunctionBegin;
  LMProducts dots = *dots_p;
  if (dots == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&dots->full));
  PetscCall(VecDestroy(&dots->diagonal_dup));
  PetscCall(VecDestroy(&dots->diagonal_local));
  PetscCall(VecDestroy(&dots->diagonal_global));
  PetscCall(VecDestroy(&dots->rhs_local));
  PetscCall(VecDestroy(&dots->lhs_local));
  PetscCall(PetscFree(dots));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMProductsPrepare_Internal(LMProducts dots, PetscObjectId operator_id, PetscObjectState operator_state, PetscInt oldest, PetscInt next)
{
  PetscFunctionBegin;
  if (dots->operator_id != operator_id || dots->operator_state != operator_state) {
    // invalidate the block
    dots->operator_id    = operator_id;
    dots->operator_state = operator_state;
    dots->k              = oldest;
  }
  dots->k = PetscMax(oldest, dots->k);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMProductsPrepareFromBases(LMProducts dots, LMBasis X, LMBasis Y)
{
  PetscInt      oldest, next;
  PetscObjectId operator_id    = (X->operator_id == 0) ? Y->operator_id : X->operator_id;
  PetscObjectId operator_state = (X->operator_id == 0) ? Y->operator_state : X->operator_state;

  PetscFunctionBegin;
  PetscCall(LMBasisGetRange(X, &oldest, &next));
  PetscCall(LMProductsPrepare_Internal(dots, operator_id, operator_state, oldest, next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsPrepare(LMProducts dots, Mat op, PetscInt oldest, PetscInt next)
{
  PetscObjectId    operator_id;
  PetscObjectState operator_state;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)op, &operator_id));
  PetscCall(PetscObjectStateGet((PetscObject)op, &operator_state));
  PetscCall(LMProductsPrepare_Internal(dots, operator_id, operator_state, oldest, next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMProductsUpdate_Internal(LMProducts dots, LMBasis X, LMBasis Y, PetscInt oldest, PetscInt next)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)X->vecs);
  PetscInt start;

  PetscFunctionBegin;
  PetscAssert(X->m == Y->m && X->m == dots->m, comm, PETSC_ERR_ARG_INCOMP, "X vecs, Y vecs, and dots incompatible in size, (%d, %d, %d)", (int)X->m, (int)Y->m, (int)dots->m);
  PetscAssert(X->k == Y->k, comm, PETSC_ERR_ARG_INCOMP, "X and Y vecs are incompatible in state, (%d, %d)", (int)X->k, (int)Y->k);
  PetscAssert(dots->k <= X->k, comm, PETSC_ERR_ARG_INCOMP, "Dot products are ahead of X and Y, (%d, %d)", (int)dots->k, (int)X->k);
  PetscAssert(X->operator_id == 0 || Y->operator_id == 0 || X->operator_id == Y->operator_id, comm, PETSC_ERR_ARG_INCOMP, "X and Y vecs are from different operators");
  PetscAssert(X->operator_id != Y->operator_id || Y->operator_state == X->operator_state, comm, PETSC_ERR_ARG_INCOMP, "X and Y vecs are from different operator states");

  PetscCall(LMProductsPrepareFromBases(dots, X, Y));

  start = dots->k;
  if (start == next) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(LMPROD_Update, NULL, NULL, NULL, NULL));
  switch (dots->block_type) {
  case LMBLOCK_DIAGONAL:
    for (PetscInt i = start; i < next; i++) {
      Vec         x, y;
      PetscScalar xTy;

      PetscCall(LMBasisGetVecRead(X, i, &x));
      y = x;
      if (Y != X) PetscCall(LMBasisGetVecRead(Y, i, &y));
      PetscCall(VecDot(y, x, &xTy));
      if (Y != X) PetscCall(LMBasisRestoreVecRead(Y, i, &y));
      PetscCall(LMBasisRestoreVecRead(X, i, &x));
      PetscCall(LMProductsInsertNextDiagonalValue(dots, i, xTy));
    }
    break;
  case LMBLOCK_STRICT_UPPER_TRIANGLE: {
    Mat local;

    PetscCall(MatDenseGetLocalMatrix(dots->full, &local));
    // we have to proceed index by index because we want to zero each row after we compute the corresponding column
    for (PetscInt i = start; i < next; i++) {
      Mat row;
      Vec column, y;

      PetscCall(LMBasisGetVecRead(Y, i, &y));
      PetscCall(MatDenseGetColumnVec(dots->full, i % dots->m, &column));
      PetscCall(LMBasisGEMVH(X, oldest, next, 1.0, y, 0.0, column));
      PetscCall(MatDenseRestoreColumnVec(dots->full, i % dots->m, &column));
      PetscCall(LMBasisRestoreVecRead(Y, i, &y));

      // zero out the new row
      if (dots->m_local) {
        PetscCall(MatDenseGetSubMatrix(local, i % dots->m, (i % dots->m) + 1, PETSC_DECIDE, PETSC_DECIDE, &row));
        PetscCall(MatZeroEntries(row));
        PetscCall(MatDenseRestoreSubMatrix(local, &row));
      }
    }
  } break;
  case LMBLOCK_UPPER_TRIANGLE: {
    PetscInt mid       = next - (next % dots->m);
    PetscInt start_idx = start % dots->m;
    PetscInt next_idx  = ((next - 1) % dots->m) + 1;

    if (next_idx > start_idx) {
      PetscCall(LMBasisGEMMH(X, oldest, next, Y, start, next, 1.0, 0.0, dots->full));
    } else {
      PetscCall(LMBasisGEMMH(X, oldest, mid, Y, start, mid, 1.0, 0.0, dots->full));
      PetscCall(LMBasisGEMMH(X, oldest, next, Y, mid, next, 1.0, 0.0, dots->full));
    }
  } break;
  case LMBLOCK_FULL:
    PetscCall(LMBasisGEMMH(X, oldest, next, Y, start, next, 1.0, 0.0, dots->full));
    PetscCall(LMBasisGEMMH(X, start, next, Y, oldest, start, 1.0, 0.0, dots->full));
    break;
  default:
    PetscUnreachable();
  }
  dots->k = next;
  if (dots->debug) {
    const PetscScalar *values = NULL;
    PetscInt           lda;
    PetscInt           N;

    PetscCall(MatGetSize(X->vecs, &N, NULL));
    if (dots->block_type == LMBLOCK_DIAGONAL) {
      lda = 0;
      if (dots->update_diagonal_global) {
        PetscCall(VecGetArrayRead(dots->diagonal_global, &values));
      } else {
        PetscCall(VecGetArrayRead(dots->diagonal_dup, &values));
      }
    } else {
      PetscCall(MatDenseGetLDA(dots->full, &lda));
      PetscCall(MatDenseGetArrayRead(dots->full, &values));
    }
    for (PetscInt i = oldest; i < next; i++) {
      Vec       x_i_, x_i;
      PetscReal x_norm;
      PetscInt  j_start = oldest;
      PetscInt  j_end   = next;

      PetscCall(LMBasisGetVecRead(X, i, &x_i_));
      PetscCall(VecNorm(x_i_, NORM_1, &x_norm));
      PetscCall(VecDuplicate(x_i_, &x_i));
      PetscCall(VecCopy(x_i_, x_i));
      PetscCall(LMBasisRestoreVecRead(X, i, &x_i_));

      switch (dots->block_type) {
      case LMBLOCK_DIAGONAL:
        j_start = i;
        j_end   = i + 1;
        break;
      case LMBLOCK_UPPER_TRIANGLE:
        j_start = i;
        break;
      case LMBLOCK_STRICT_UPPER_TRIANGLE:
        j_start = i + 1;
        break;
      default:
        break;
      }
      for (PetscInt j = j_start; j < j_end; j++) {
        Vec         y_j;
        PetscScalar dot_true, dot = 0.0, diff;
        PetscReal   y_norm;

        PetscCall(LMBasisGetVecRead(Y, j, &y_j));
        PetscCall(VecDot(y_j, x_i, &dot_true));
        PetscCall(VecNorm(y_j, NORM_1, &y_norm));
        if (dots->m_local) dot = values[(j % dots->m) * lda + (i % dots->m)];
        PetscCallMPI(MPI_Bcast(&dot, 1, MPIU_SCALAR, 0, comm));
        diff = dot_true - dot;
        if (PetscDefined(USE_COMPLEX)) {
          PetscCheck(PetscAbsScalar(diff) <= PETSC_SMALL * N * x_norm * y_norm, comm, PETSC_ERR_PLIB, "LMProducts debug: dots[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g + i*%g != VecDot() = %g + i*%g", i, j, (double)PetscRealPart(dot), (double)PetscImaginaryPart(dot), (double)PetscRealPart(dot_true), (double)PetscImaginaryPart(dot_true));
        } else {
          PetscCheck(PetscAbsScalar(diff) <= PETSC_SMALL * N * x_norm * y_norm, comm, PETSC_ERR_PLIB, "LMProducts debug: dots[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g != VecDot() = %g", i, j, (double)PetscRealPart(dot), (double)PetscRealPart(dot_true));
        }
        PetscCall(LMBasisRestoreVecRead(Y, j, &y_j));
      }

      PetscCall(VecDestroy(&x_i));
    }

    if (dots->block_type == LMBLOCK_DIAGONAL) {
      if (dots->update_diagonal_global) {
        PetscCall(VecRestoreArrayRead(dots->diagonal_global, &values));
      } else {
        PetscCall(VecRestoreArrayRead(dots->diagonal_dup, &values));
      }
    } else {
      PetscCall(MatDenseRestoreArrayRead(dots->full, &values));
    }
  }
  PetscCall(PetscLogEventEnd(LMPROD_Update, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// dots = X^H Y
PETSC_INTERN PetscErrorCode LMProductsUpdate(LMProducts dots, LMBasis X, LMBasis Y)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(LMBasisGetRange(X, &oldest, &next));
  PetscCall(LMProductsUpdate_Internal(dots, X, Y, oldest, next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsCopy(LMProducts src, LMProducts dest)
{
  PetscFunctionBegin;
  PetscCheck(dest->m == src->m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot copy to LMProducts of different size");
  PetscCheck(dest->m_local == src->m_local, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot copy to LMProducts of different size");
  PetscCheck(dest->block_type == src->block_type, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot copy to LMProducts of different block type");
  dest->k       = src->k;
  dest->m_local = src->m_local;
  if (src->full) PetscCall(MatCopy(src->full, dest->full, DIFFERENT_NONZERO_PATTERN));
  if (src->diagonal_dup) PetscCall(VecCopy(src->diagonal_dup, dest->diagonal_dup));
  if (src->diagonal_global) PetscCall(VecCopy(src->diagonal_global, dest->diagonal_global));
  dest->update_diagonal_global = src->update_diagonal_global;
  dest->operator_id            = src->operator_id;
  dest->operator_state         = src->operator_state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsScale(LMProducts dots, PetscScalar scale)
{
  PetscFunctionBegin;
  if (dots->full) PetscCall(MatScale(dots->full, scale));
  if (dots->diagonal_dup) PetscCall(VecScale(dots->diagonal_dup, scale));
  if (dots->diagonal_global) PetscCall(VecScale(dots->diagonal_global, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsGetLocalMatrix(LMProducts dots, Mat *G_local, PetscInt *k, PetscBool *local_is_nonempty)
{
  PetscFunctionBegin;
  PetscCheck(dots->block_type != LMBLOCK_DIAGONAL, PETSC_COMM_SELF, PETSC_ERR_SUP, "Asking for full matrix of diagonal products");
  PetscCall(MatDenseGetLocalMatrix(dots->full, G_local));
  if (k) *k = dots->k;
  if (local_is_nonempty) *local_is_nonempty = (dots->m_local == dots->m) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsRestoreLocalMatrix(LMProducts dots, Mat *G_local, PetscInt *k)
{
  PetscFunctionBegin;
  if (G_local) *G_local = NULL;
  if (k) dots->k = *k;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMProductsGetUpdatedDiagonal(LMProducts dots, Vec *diagonal)
{
  PetscFunctionBegin;
  if (!dots->update_diagonal_global) {
    PetscCall(VecGetLocalVector(dots->diagonal_global, dots->diagonal_local));
    if (dots->m_local) PetscCall(VecCopy(dots->diagonal_dup, dots->diagonal_local));
    PetscCall(VecRestoreLocalVector(dots->diagonal_global, dots->diagonal_local));
    dots->update_diagonal_global = PETSC_TRUE;
  }
  if (diagonal) *diagonal = dots->diagonal_global;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsGetLocalDiagonal(LMProducts dots, Vec *D_local)
{
  PetscFunctionBegin;
  PetscCall(LMProductsGetUpdatedDiagonal(dots, NULL));
  PetscCall(VecGetLocalVector(dots->diagonal_global, dots->diagonal_local));
  *D_local = dots->diagonal_local;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsRestoreLocalDiagonal(LMProducts dots, Vec *D_local)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreLocalVector(dots->diagonal_global, dots->diagonal_local));
  *D_local = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsGetNextColumn(LMProducts dots, Vec *col)
{
  PetscFunctionBegin;
  PetscCheck(dots->block_type != LMBLOCK_DIAGONAL, PETSC_COMM_SELF, PETSC_ERR_SUP, "Asking for column of diagonal products");
  PetscCall(MatDenseGetColumnVecWrite(dots->full, dots->k % dots->m, col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsRestoreNextColumn(LMProducts dots, Vec *col)
{
  PetscFunctionBegin;
  PetscCall(MatDenseRestoreColumnVecWrite(dots->full, dots->k % dots->m, col));
  dots->k++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// copy conj(triu(G)) into tril(G)
PETSC_INTERN PetscErrorCode LMProductsMakeHermitian(Mat local, PetscInt oldest, PetscInt next)
{
  PetscInt m;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(local, &m, NULL));
  if (m) {
    // TODO: implement on device?
    PetscScalar *a;
    PetscInt     lda;

    PetscCall(MatDenseGetLDA(local, &lda));
    PetscCall(MatDenseGetArray(local, &a));
    for (PetscInt j_ = oldest; j_ < next; j_++) {
      PetscInt j = j_ % m;

      a[j + j * lda] = PetscRealPart(a[j + j * lda]);
      for (PetscInt i_ = j_ + 1; i_ < next; i_++) {
        PetscInt i = i_ % m;

        a[i + j * lda] = PetscConj(a[j + i * lda]);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsSolve(LMProducts dots, PetscInt oldest, PetscInt next, Vec b, Vec x, PetscBool hermitian_transpose)
{
  PetscInt dots_oldest = PetscMax(0, dots->k - dots->m);
  PetscInt dots_next   = dots->k;
  Mat      local;
  Vec      diag = NULL;

  PetscFunctionBegin;
  PetscCheck(oldest >= dots_oldest && next <= dots_next, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid indices");
  if (oldest >= next) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(LMPROD_Solve, NULL, NULL, NULL, NULL));
  if (!dots->rhs_local) PetscCall(VecCreateLocalVector(b, &dots->rhs_local));
  if (!dots->lhs_local) PetscCall(VecDuplicate(dots->rhs_local, &dots->lhs_local));
  switch (dots->block_type) {
  case LMBLOCK_DIAGONAL:
    PetscCall(LMProductsGetUpdatedDiagonal(dots, &diag));
    PetscCall(VecDSVCyclic(hermitian_transpose, oldest, next, diag, b, x));
    break;
  case LMBLOCK_UPPER_TRIANGLE:
    PetscCall(MatSeqDenseTRSVCyclic(hermitian_transpose, oldest, next, dots->full, b, x));
    break;
  default: {
    PetscCall(MatDenseGetLocalMatrix(dots->full, &local));
    PetscCall(VecGetLocalVector(b, dots->rhs_local));
    PetscCall(VecGetLocalVector(x, dots->lhs_local));
    if (dots->m_local) {
      if (!hermitian_transpose) {
        PetscCall(MatSolve(local, dots->rhs_local, dots->lhs_local));
      } else {
        Vec rhs_conj = dots->rhs_local;

        if (PetscDefined(USE_COMPLEX)) {
          PetscCall(VecDuplicate(dots->rhs_local, &rhs_conj));
          PetscCall(VecCopy(dots->rhs_local, rhs_conj));
          PetscCall(VecConjugate(rhs_conj));
        }
        PetscCall(MatSolveTranspose(local, rhs_conj, dots->lhs_local));
        if (PetscDefined(USE_COMPLEX)) {
          PetscCall(VecConjugate(dots->lhs_local));
          PetscCall(VecDestroy(&rhs_conj));
        }
      }
    }
    if (x != b) PetscCall(VecRestoreLocalVector(x, dots->lhs_local));
    PetscCall(VecRestoreLocalVector(b, dots->rhs_local));
  } break;
  }
  PetscCall(PetscLogEventEnd(LMPROD_Solve, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsMult(LMProducts dots, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec x, PetscScalar beta, Vec y, PetscBool hermitian_transpose)
{
  PetscInt dots_oldest = PetscMax(0, dots->k - dots->m);
  PetscInt dots_next   = dots->k;
  Vec      diag        = NULL;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(LMPROD_Mult, NULL, NULL, NULL, NULL));
  PetscCheck(oldest >= dots_oldest && next <= dots_next, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid indices");
  switch (dots->block_type) {
  case LMBLOCK_DIAGONAL: {
    PetscCall(LMProductsGetUpdatedDiagonal(dots, &diag));
    PetscCall(VecDMVCyclic(hermitian_transpose, oldest, next, alpha, diag, x, beta, y));
  } break;
  case LMBLOCK_STRICT_UPPER_TRIANGLE: // the lower triangle has been zeroed, MatMult() is safe
  case LMBLOCK_FULL:
    PetscCall(MatSeqDenseGEMVCyclic(hermitian_transpose, oldest, next, alpha, dots->full, x, beta, y));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  }
  PetscCall(PetscLogEventEnd(LMPROD_Mult, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsMultHermitian(LMProducts dots, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec x, PetscScalar beta, Vec y)
{
  PetscInt dots_oldest = PetscMax(0, dots->k - dots->m);
  PetscInt dots_next   = dots->k;

  PetscFunctionBegin;
  PetscCheck(oldest >= dots_oldest && next <= dots_next, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid indices");
  if (dots->block_type == LMBLOCK_DIAGONAL) PetscCall(LMProductsMult(dots, oldest, next, alpha, x, beta, y, PETSC_FALSE));
  else {
    PetscCall(PetscLogEventBegin(LMPROD_Mult, NULL, NULL, NULL, NULL));
    PetscCall(MatSeqDenseHEMVCyclic(oldest, next, alpha, dots->full, x, beta, y));
    PetscCall(PetscLogEventEnd(LMPROD_Mult, NULL, NULL, NULL, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsReset(LMProducts dots)
{
  PetscFunctionBegin;
  if (dots) {
    dots->k              = 0;
    dots->operator_id    = 0;
    dots->operator_state = 0;
    if (dots->full) {
      Mat full_local;

      PetscCall(MatDenseGetLocalMatrix(dots->full, &full_local));
      PetscCall(MatSetUnfactored(full_local));
      PetscCall(MatZeroEntries(full_local));
    }
    if (dots->diagonal_global) PetscCall(VecZeroEntries(dots->diagonal_dup));
    if (dots->diagonal_dup) PetscCall(VecZeroEntries(dots->diagonal_dup));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsGetDiagonalValue(LMProducts dots, PetscInt i, PetscScalar *v)
{
  PetscFunctionBegin;
  PetscInt oldest = PetscMax(0, dots->k - dots->m);
  PetscInt next   = dots->k;
  PetscInt idx    = i % dots->m;
  PetscCheck(i >= oldest && i < next, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserting value %d out of range [%d, %d)", (int)i, (int)oldest, (int)next);
  PetscCall(VecGetValues(dots->diagonal_dup, 1, &idx, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsInsertNextDiagonalValue(LMProducts dots, PetscInt i, PetscScalar v)
{
  PetscInt idx = i % dots->m;

  PetscFunctionBegin;
  PetscCheck(i == dots->k, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT " is not the next index (%" PetscInt_FMT ")", i, dots->k);
  PetscCall(VecSetValue(dots->diagonal_dup, idx, v, INSERT_VALUES));
  if (dots->update_diagonal_global) {
    PetscScalar *array;
    PetscMemType memtype;

    PetscCall(VecGetArrayAndMemType(dots->diagonal_global, &array, &memtype));
    if (dots->m_local > 0) {
      if (PetscMemTypeHost(memtype)) {
        array[idx] = v;
        PetscCall(VecRestoreArrayAndMemType(dots->diagonal_global, &array));
      } else {
        PetscCall(VecRestoreArrayAndMemType(dots->diagonal_global, &array));
        PetscCall(VecGetLocalVector(dots->diagonal_global, dots->diagonal_local));
        if (dots->m_local) PetscCall(VecCopy(dots->diagonal_dup, dots->diagonal_local));
        PetscCall(VecRestoreLocalVector(dots->diagonal_global, dots->diagonal_local));
      }
    } else {
      PetscCall(VecRestoreArrayAndMemType(dots->diagonal_global, &array));
    }
  }
  dots->k++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMProductsOnesOnUnusedDiagonal(Mat A, PetscInt oldest, PetscInt next)
{
  PetscInt m;
  Mat      sub;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, NULL));
  // we could handle the general case but this is the only case used by MatLMVM
  PetscCheck((next < m && oldest == 0) || next - oldest == m, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "General case not implemented");
  if (next - oldest == m) PetscFunctionReturn(PETSC_SUCCESS); // nothing to do if all entries are used
  PetscCall(MatDenseGetSubMatrix(A, next, m, next, m, &sub));
  PetscCall(MatShift(sub, 1.0));
  PetscCall(MatDenseRestoreSubMatrix(A, &sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}
