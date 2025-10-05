#include <petsc/private/petscimpl.h>
#include "lmbasis.h"
#include "blas_cyclic/blas_cyclic.h"

PetscLogEvent LMBASIS_GEMM, LMBASIS_GEMV, LMBASIS_GEMVH;

PetscErrorCode LMBasisCreate(Vec v, PetscInt m, LMBasis *basis_p)
{
  PetscInt    n, N;
  PetscMPIInt rank;
  Mat         backing;
  VecType     type;
  LMBasis     basis;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(v, m, 2);
  PetscCheck(m >= 0, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_OUTOFRANGE, "Requested window size %" PetscInt_FMT " is not >= 0", m);
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetSize(v, &N));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank));
  PetscCall(VecGetType(v, &type));
  PetscCall(MatCreateDenseFromVecType(PetscObjectComm((PetscObject)v), type, n, rank == 0 ? m : 0, N, m, n, NULL, &backing));
  PetscCall(PetscNew(&basis));
  *basis_p    = basis;
  basis->m    = m;
  basis->k    = 0;
  basis->vecs = backing;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMBasisGetVec_Internal(LMBasis basis, PetscInt idx, PetscMemoryAccessMode mode, Vec *single, PetscBool check_idx)
{
  PetscFunctionBegin;
  PetscAssertPointer(basis, 1);
  if (check_idx) {
    PetscValidLogicalCollectiveInt(basis->vecs, idx, 2);
    PetscCheck(idx < basis->k, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for index %" PetscInt_FMT " >= number of inserted vecs %" PetscInt_FMT, idx, basis->k);
    PetscInt earliest = PetscMax(0, basis->k - basis->m);
    PetscCheck(idx >= earliest, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for index %" PetscInt_FMT " < the earliest retained index % " PetscInt_FMT, idx, earliest);
  }
  PetscAssert(mode == PETSC_MEMORY_ACCESS_READ || mode == PETSC_MEMORY_ACCESS_WRITE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "READ_WRITE access not implemented");
  if (mode == PETSC_MEMORY_ACCESS_READ) {
    PetscCall(MatDenseGetColumnVecRead(basis->vecs, idx % basis->m, single));
  } else {
    PetscCall(MatDenseGetColumnVecWrite(basis->vecs, idx % basis->m, single));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetVec(LMBasis basis, PetscInt idx, PetscMemoryAccessMode mode, Vec *single)
{
  PetscFunctionBegin;
  PetscCall(LMBasisGetVec_Internal(basis, idx, mode, single, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreVec(LMBasis basis, PetscInt idx, PetscMemoryAccessMode mode, Vec *single)
{
  PetscFunctionBegin;
  PetscAssertPointer(basis, 1);
  PetscAssert(mode == PETSC_MEMORY_ACCESS_READ || mode == PETSC_MEMORY_ACCESS_WRITE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "READ_WRITE access not implemented");
  if (mode == PETSC_MEMORY_ACCESS_READ) {
    PetscCall(MatDenseRestoreColumnVecRead(basis->vecs, idx % basis->m, single));
  } else {
    PetscCall(MatDenseRestoreColumnVecWrite(basis->vecs, idx % basis->m, single));
  }
  *single = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetVecRead(LMBasis B, PetscInt i, Vec *b)
{
  return LMBasisGetVec(B, i, PETSC_MEMORY_ACCESS_READ, b);
}
PETSC_INTERN PetscErrorCode LMBasisRestoreVecRead(LMBasis B, PetscInt i, Vec *b)
{
  return LMBasisRestoreVec(B, i, PETSC_MEMORY_ACCESS_READ, b);
}

PETSC_INTERN PetscErrorCode LMBasisGetNextVec(LMBasis basis, Vec *single)
{
  PetscFunctionBegin;
  PetscCall(LMBasisGetVec_Internal(basis, basis->k, PETSC_MEMORY_ACCESS_WRITE, single, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreNextVec(LMBasis basis, Vec *single)
{
  PetscFunctionBegin;
  PetscAssertPointer(basis, 1);
  PetscCall(LMBasisRestoreVec(basis, basis->k++, PETSC_MEMORY_ACCESS_WRITE, single));
  // basis is updated, invalidate cached product
  basis->cached_vec_id    = 0;
  basis->cached_vec_state = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisSetNextVec(LMBasis basis, Vec single)
{
  Vec next;

  PetscFunctionBegin;
  PetscCall(LMBasisGetNextVec(basis, &next));
  PetscCall(VecCopy(single, next));
  PetscCall(LMBasisRestoreNextVec(basis, &next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisDestroy(LMBasis *basis_p)
{
  LMBasis basis = *basis_p;

  PetscFunctionBegin;
  *basis_p = NULL;
  if (basis == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(LMBasisReset(basis));
  PetscCheck(basis->work_vecs_in_use == NULL, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_WRONGSTATE, "Work vecs are still checked out at destruction");
  {
    VecLink head = basis->work_vecs_available;

    while (head) {
      VecLink next = head->next;

      PetscCall(VecDestroy(&head->vec));
      PetscCall(PetscFree(head));
      head = next;
    }
  }
  PetscCheck(basis->work_rows_in_use == NULL, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_WRONGSTATE, "Work rows are still checked out at destruction");
  {
    VecLink head = basis->work_rows_available;

    while (head) {
      VecLink next = head->next;

      PetscCall(VecDestroy(&head->vec));
      PetscCall(PetscFree(head));
      head = next;
    }
  }
  PetscCall(MatDestroy(&basis->vecs));
  PetscCall(PetscFree(basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetWorkVec(LMBasis basis, Vec *vec_p)
{
  VecLink link;

  PetscFunctionBegin;
  if (!basis->work_vecs_available) {
    PetscCall(PetscNew(&basis->work_vecs_available));
    PetscCall(MatCreateVecs(basis->vecs, NULL, &basis->work_vecs_available->vec));
  }
  link                       = basis->work_vecs_available;
  basis->work_vecs_available = link->next;
  link->next                 = basis->work_vecs_in_use;
  basis->work_vecs_in_use    = link;

  *vec_p    = link->vec;
  link->vec = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreWorkVec(LMBasis basis, Vec *vec_p)
{
  Vec     v    = *vec_p;
  VecLink link = NULL;

  PetscFunctionBegin;
  *vec_p = NULL;
  PetscCheck(basis->work_vecs_in_use, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_WRONGSTATE, "Trying to check in a vec that wasn't checked out");
  link                       = basis->work_vecs_in_use;
  basis->work_vecs_in_use    = link->next;
  link->next                 = basis->work_vecs_available;
  basis->work_vecs_available = link;

  PetscAssert(link->vec == NULL, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_PLIB, "Link not ready to return vector");
  link->vec = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisCreateRow(LMBasis basis, Vec *row_p)
{
  PetscFunctionBegin;
  PetscCall(MatCreateVecs(basis->vecs, row_p, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetWorkRow(LMBasis basis, Vec *row_p)
{
  VecLink link;

  PetscFunctionBegin;
  if (!basis->work_rows_available) {
    PetscCall(PetscNew(&basis->work_rows_available));
    PetscCall(MatCreateVecs(basis->vecs, &basis->work_rows_available->vec, NULL));
  }
  link                       = basis->work_rows_available;
  basis->work_rows_available = link->next;
  link->next                 = basis->work_rows_in_use;
  basis->work_rows_in_use    = link;

  PetscCall(VecZeroEntries(link->vec));
  *row_p    = link->vec;
  link->vec = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreWorkRow(LMBasis basis, Vec *row_p)
{
  Vec     v    = *row_p;
  VecLink link = NULL;

  PetscFunctionBegin;
  *row_p = NULL;
  PetscCheck(basis->work_rows_in_use, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_WRONGSTATE, "Trying to check in a row that wasn't checked out");
  link                       = basis->work_rows_in_use;
  basis->work_rows_in_use    = link->next;
  link->next                 = basis->work_rows_available;
  basis->work_rows_available = link;

  PetscAssert(link->vec == NULL, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_PLIB, "Link not ready to return vector");
  link->vec = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisCopy(LMBasis basis_a, LMBasis basis_b)
{
  PetscFunctionBegin;
  PetscCheck(basis_a->m == basis_b->m, PetscObjectComm((PetscObject)basis_a), PETSC_ERR_ARG_SIZ, "Copy target has different number of vecs, %" PetscInt_FMT " != %" PetscInt_FMT, basis_b->m, basis_a->m);
  basis_b->k = basis_a->k;
  PetscCall(MatCopy(basis_a->vecs, basis_b->vecs, SAME_NONZERO_PATTERN));
  basis_b->cached_vec_id    = basis_a->cached_vec_id;
  basis_b->cached_vec_state = basis_a->cached_vec_state;
  if (basis_a->cached_product) {
    if (!basis_b->cached_product) PetscCall(VecDuplicate(basis_a->cached_product, &basis_b->cached_product));
    PetscCall(VecCopy(basis_a->cached_product, basis_b->cached_product));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetRange(LMBasis basis, PetscInt *oldest, PetscInt *next)
{
  PetscFunctionBegin;
  *next   = basis->k;
  *oldest = PetscMax(0, basis->k - basis->m);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMBasisMultCheck(LMBasis A, PetscInt oldest, PetscInt next)
{
  PetscInt basis_oldest, basis_next;

  PetscFunctionBegin;
  PetscCall(LMBasisGetRange(A, &basis_oldest, &basis_next));
  PetscCheck(oldest >= basis_oldest && next <= basis_next, PetscObjectComm((PetscObject)A->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for vec that hasn't been computed or is no longer stored");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGEMV(LMBasis A, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec x, PetscScalar beta, Vec y)
{
  PetscInt lim        = next - oldest;
  PetscInt next_idx   = ((next - 1) % A->m) + 1;
  PetscInt oldest_idx = oldest % A->m;
  Vec      x_work     = NULL;
  Vec      x_         = x;

  PetscFunctionBegin;
  if (lim <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(LMBASIS_GEMV, NULL, NULL, NULL, NULL));
  PetscCall(LMBasisMultCheck(A, oldest, next));
  if (alpha != 1.0) {
    PetscCall(LMBasisGetWorkRow(A, &x_work));
    PetscCall(VecAXPBYCyclic(oldest, next, alpha, x, 0.0, x_work));
    x_ = x_work;
  }
  if (beta != 1.0 && beta != 0.0) PetscCall(VecScale(y, beta));
  if (lim == A->m) {
    // all vectors are used
    if (beta == 0.0) PetscCall(MatMult(A->vecs, x_, y));
    else PetscCall(MatMultAdd(A->vecs, x_, y, y));
  } else if (oldest_idx < next_idx) {
    // contiguous vectors are used
    if (beta == 0.0) PetscCall(MatMultColumnRange(A->vecs, x_, y, oldest_idx, next_idx));
    else PetscCall(MatMultAddColumnRange(A->vecs, x_, y, y, oldest_idx, next_idx));
  } else {
    if (beta == 0.0) PetscCall(MatMultColumnRange(A->vecs, x_, y, 0, next_idx));
    else PetscCall(MatMultAddColumnRange(A->vecs, x_, y, y, 0, next_idx));
    PetscCall(MatMultAddColumnRange(A->vecs, x_, y, y, oldest_idx, A->m));
  }
  if (alpha != 1.0) PetscCall(LMBasisRestoreWorkRow(A, &x_work));
  PetscCall(PetscLogEventEnd(LMBASIS_GEMV, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGEMVH(LMBasis A, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec x, PetscScalar beta, Vec y)
{
  PetscInt lim        = next - oldest;
  PetscInt next_idx   = ((next - 1) % A->m) + 1;
  PetscInt oldest_idx = oldest % A->m;
  Vec      y_         = y;

  PetscFunctionBegin;
  if (lim <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(LMBasisMultCheck(A, oldest, next));
  if (A->cached_product && A->cached_vec_id != 0 && A->cached_vec_state != 0) {
    // see if x is the cached input vector
    PetscObjectId    x_id;
    PetscObjectState x_state;

    PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
    PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
    if (x_id == A->cached_vec_id && x_state == A->cached_vec_state) {
      PetscCall(VecAXPBYCyclic(oldest, next, alpha, A->cached_product, beta, y));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCall(PetscLogEventBegin(LMBASIS_GEMVH, NULL, NULL, NULL, NULL));
  if (alpha != 1.0 || (beta != 1.0 && beta != 0.0)) PetscCall(LMBasisGetWorkRow(A, &y_));
  if (lim == A->m) {
    // all vectors are used
    if (alpha == 1.0 && beta == 1.0) PetscCall(MatMultHermitianTransposeAdd(A->vecs, x, y_, y_));
    else PetscCall(MatMultHermitianTranspose(A->vecs, x, y_));
  } else if (oldest_idx < next_idx) {
    // contiguous vectors are used
    if (alpha == 1.0 && beta == 1.0) PetscCall(MatMultHermitianTransposeAddColumnRange(A->vecs, x, y_, y_, oldest_idx, next_idx));
    else PetscCall(MatMultHermitianTransposeColumnRange(A->vecs, x, y_, oldest_idx, next_idx));
  } else {
    if (alpha == 1.0 && beta == 1.0) {
      PetscCall(MatMultHermitianTransposeAddColumnRange(A->vecs, x, y_, y_, 0, next_idx));
      PetscCall(MatMultHermitianTransposeAddColumnRange(A->vecs, x, y_, y_, oldest_idx, A->m));
    } else {
      PetscCall(MatMultHermitianTransposeColumnRange(A->vecs, x, y_, 0, next_idx));
      PetscCall(MatMultHermitianTransposeColumnRange(A->vecs, x, y_, oldest_idx, A->m));
    }
  }
  if (alpha != 1.0 || (beta != 1.0 && beta != 0.0)) {
    PetscCall(VecAXPBYCyclic(oldest, next, alpha, y_, beta, y));
    PetscCall(LMBasisRestoreWorkRow(A, &y_));
  }
  PetscCall(PetscLogEventEnd(LMBASIS_GEMVH, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMBasisGEMMH_Internal(Mat A, Mat B, PetscScalar alpha, PetscScalar beta, Mat G)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(A));
  if (beta != 0.0) {
    Mat G_alloc;

    if (beta != 1.0) PetscCall(MatScale(G, beta));
    PetscCall(MatTransposeMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DECIDE, &G_alloc));
    PetscCall(MatAXPY(G, alpha, G_alloc, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&G_alloc));
  } else {
    PetscCall(MatProductClear(G));
    PetscCall(MatProductCreateWithMat(A, B, NULL, G));
    PetscCall(MatProductSetType(G, MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(G));
    PetscCall(MatProductSymbolic(G));
    PetscCall(MatProductNumeric(G));
    if (alpha != 1.0) PetscCall(MatScale(G, alpha));
  }
  if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGEMMH(LMBasis A, PetscInt a_oldest, PetscInt a_next, LMBasis B, PetscInt b_oldest, PetscInt b_next, PetscScalar alpha, PetscScalar beta, Mat G)
{
  PetscInt a_lim = a_next - a_oldest;
  PetscInt b_lim = b_next - b_oldest;

  PetscFunctionBegin;
  if (a_lim <= 0 || b_lim <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(LMBASIS_GEMM, NULL, NULL, NULL, NULL));
  PetscCall(LMBasisMultCheck(A, a_oldest, a_next));
  PetscCall(LMBasisMultCheck(B, b_oldest, b_next));
  if (b_lim == 1) {
    Vec b;
    Vec g;

    PetscCall(LMBasisGetVecRead(B, b_oldest, &b));
    PetscCall(MatDenseGetColumnVec(G, b_oldest % B->m, &g));
    PetscCall(LMBasisGEMVH(A, a_oldest, a_next, alpha, b, beta, g));
    PetscCall(MatDenseRestoreColumnVec(G, b_oldest % B->m, &g));
    PetscCall(LMBasisRestoreVecRead(B, b_oldest, &b));
  } else if (a_lim == 1) {
    Vec a;
    Vec g;

    PetscCall(LMBasisGetVecRead(A, a_oldest, &a));
    PetscCall(LMBasisGetWorkRow(B, &g));
    PetscCall(LMBasisGEMVH(B, b_oldest, b_next, 1.0, a, 0.0, g));
    if (PetscDefined(USE_COMPLEX)) PetscCall(VecConjugate(g));
    PetscCall(MatSeqDenseRowAXPBYCyclic(b_oldest, b_next, alpha, g, beta, G, a_oldest));
    PetscCall(LMBasisRestoreWorkRow(B, &g));
    PetscCall(LMBasisRestoreVecRead(A, a_oldest, &a));
  } else {
    PetscInt a_next_idx        = ((a_next - 1) % A->m) + 1;
    PetscInt a_oldest_idx      = a_oldest % A->m;
    PetscInt b_next_idx        = ((b_next - 1) % B->m) + 1;
    PetscInt b_oldest_idx      = b_oldest % B->m;
    PetscInt a_intervals[2][2] = {
      {0,            a_next_idx},
      {a_oldest_idx, A->m      }
    };
    PetscInt b_intervals[2][2] = {
      {0,            b_next_idx},
      {b_oldest_idx, B->m      }
    };
    PetscInt a_num_intervals = 2;
    PetscInt b_num_intervals = 2;

    if (a_lim == A->m || a_oldest_idx < a_next_idx) {
      a_num_intervals = 1;
      if (a_lim == A->m) {
        a_intervals[0][0] = 0;
        a_intervals[0][1] = A->m;
      } else {
        a_intervals[0][0] = a_oldest_idx;
        a_intervals[0][1] = a_next_idx;
      }
    }
    if (b_lim == B->m || b_oldest_idx < b_next_idx) {
      b_num_intervals = 1;
      if (b_lim == B->m) {
        b_intervals[0][0] = 0;
        b_intervals[0][1] = B->m;
      } else {
        b_intervals[0][0] = b_oldest_idx;
        b_intervals[0][1] = b_next_idx;
      }
    }
    for (PetscInt i = 0; i < a_num_intervals; i++) {
      Mat sub_A = A->vecs;
      Mat sub_A_;

      if (a_intervals[i][0] != 0 || a_intervals[i][1] != A->m) PetscCall(MatDenseGetSubMatrix(A->vecs, PETSC_DECIDE, PETSC_DECIDE, a_intervals[i][0], a_intervals[i][1], &sub_A));
      sub_A_ = sub_A;

      for (PetscInt j = 0; j < b_num_intervals; j++) {
        Mat sub_B = B->vecs;
        Mat sub_G = G;

        if (b_intervals[j][0] != 0 || b_intervals[j][1] != B->m) {
          if (sub_A_ == sub_A && sub_A != A->vecs && B->vecs == A->vecs) {
            /* We're hampered by the fact that you can only get one submatrix from a MatDense at a time.  This case
             * should not happen often, copying here is acceptable */
            PetscCall(MatDuplicate(sub_A, MAT_COPY_VALUES, &sub_A_));
            PetscCall(MatDenseRestoreSubMatrix(A->vecs, &sub_A));
            sub_A = A->vecs;
          }
          PetscCall(MatDenseGetSubMatrix(B->vecs, PETSC_DECIDE, PETSC_DECIDE, b_intervals[j][0], b_intervals[j][1], &sub_B));
        }

        if (sub_A_ != A->vecs || sub_B != B->vecs) PetscCall(MatDenseGetSubMatrix(G, a_intervals[i][0], a_intervals[i][1], b_intervals[j][0], b_intervals[j][1], &sub_G));

        PetscCall(LMBasisGEMMH_Internal(sub_A_, sub_B, alpha, beta, sub_G));

        if (sub_G != G) PetscCall(MatDenseRestoreSubMatrix(G, &sub_G));
        if (sub_B != B->vecs) PetscCall(MatDenseRestoreSubMatrix(B->vecs, &sub_B));
      }

      if (sub_A_ != sub_A) PetscCall(MatDestroy(&sub_A_));
      if (sub_A != A->vecs) PetscCall(MatDenseRestoreSubMatrix(A->vecs, &sub_A));
    }
  }
  PetscCall(PetscLogEventEnd(LMBASIS_GEMM, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisReset(LMBasis basis)
{
  PetscFunctionBegin;
  if (basis) {
    basis->k = 0;
    PetscCall(VecDestroy(&basis->cached_product));
    basis->cached_vec_id    = 0;
    basis->cached_vec_state = 0;
    basis->operator_id      = 0;
    basis->operator_state   = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisSetCachedProduct(LMBasis A, Vec x, Vec Ax)
{
  PetscFunctionBegin;
  if (x == NULL) {
    A->cached_vec_id    = 0;
    A->cached_vec_state = 0;
  } else {
    PetscCall(PetscObjectGetId((PetscObject)x, &A->cached_vec_id));
    PetscCall(PetscObjectStateGet((PetscObject)x, &A->cached_vec_state));
  }
  PetscCall(PetscObjectReference((PetscObject)Ax));
  PetscCall(VecDestroy(&A->cached_product));
  A->cached_product = Ax;
  PetscFunctionReturn(PETSC_SUCCESS);
}
