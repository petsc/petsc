
/*
    Factorization code for BAIJ format.
*/

#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petscbt.h>
#include <../src/mat/utils/freespace.h>

/* ----------------------------------------------------------------*/
extern PetscErrorCode MatDuplicateNoCreate_SeqBAIJ(Mat, Mat, MatDuplicateOption, PetscBool);

/*
   This is not much faster than MatLUFactorNumeric_SeqBAIJ_N() but the solve is faster at least sometimes
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_15_NaturalOrdering(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat              C = B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
  PetscInt         i, j, k, ipvt[15];
  const PetscInt   n = a->mbs, *ai = a->i, *aj = a->j, *bi = b->i, *bj = b->j, *ajtmp, *bjtmp, *bdiag = b->diag, *pj;
  PetscInt         nz, nzL, row;
  MatScalar       *rtmp, *pc, *mwork, *pv, *vv, work[225];
  const MatScalar *v, *aa = a->a;
  PetscInt         bs2 = a->bs2, bs = A->rmap->bs, flg;
  PetscInt         sol_ver;
  PetscBool        allowzeropivot, zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(PetscOptionsGetInt(NULL, ((PetscObject)A)->prefix, "-sol_ver", &sol_ver, NULL));

  /* generate work space needed by the factorization */
  PetscCall(PetscMalloc2(bs2 * n, &rtmp, bs2, &mwork));
  PetscCall(PetscArrayzero(rtmp, bs2 * n));

  for (i = 0; i < n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i + 1] - bi[i];
    bjtmp = bj + bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* U part */
    nz    = bdiag[i] - bdiag[i + 1];
    bjtmp = bj + bdiag[i + 1] + 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* load in initial (unfactored row) */
    nz    = ai[i + 1] - ai[i];
    ajtmp = aj + ai[i];
    v     = aa + bs2 * ai[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(rtmp + bs2 * ajtmp[j], v + bs2 * j, bs2));

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i + 1] - bi[i];
    for (k = 0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2 * row;
      for (flg = 0, j = 0; j < bs2; j++) {
        if (pc[j] != 0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2 * bdiag[row];
        PetscKernel_A_gets_A_times_B(bs, pc, pv, mwork);
        /* PetscCall(PetscKernel_A_gets_A_times_B_15(pc,pv,mwork)); */
        pj = b->j + bdiag[row + 1] + 1; /* beginning of U(row,:) */
        pv = b->a + bs2 * (bdiag[row + 1] + 1);
        nz = bdiag[row] - bdiag[row + 1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j = 0; j < nz; j++) {
          vv = rtmp + bs2 * pj[j];
          PetscKernel_A_gets_A_minus_B_times_C(bs, vv, pc, pv);
          /* PetscCall(PetscKernel_A_gets_A_minus_B_times_C_15(vv,pc,pv)); */
          pv += bs2;
        }
        PetscCall(PetscLogFlops(2.0 * bs2 * bs * (nz + 1) - bs2)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2 * bi[i];
    pj = b->j + bi[i];
    nz = bi[i + 1] - bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv = b->a + bs2 * bdiag[i];
    pj = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv, rtmp + bs2 * pj[0], bs2));
    PetscCall(PetscKernel_A_gets_inverse_A_15(pv, ipvt, work, info->shiftamount, allowzeropivot, &zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2 * (bdiag[i + 1] + 1);
    pj = b->j + bdiag[i + 1] + 1;
    nz = bdiag[i] - bdiag[i + 1] - 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));
  }

  PetscCall(PetscFree2(rtmp, mwork));

  C->ops->solve          = MatSolve_SeqBAIJ_15_NaturalOrdering_ver1;
  C->ops->solvetranspose = MatSolve_SeqBAIJ_N_NaturalOrdering;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333 * bs * bs2 * b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_N(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat             C = B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
  IS              isrow = b->row, isicol = b->icol;
  const PetscInt *r, *ic;
  PetscInt        i, j, k, n = a->mbs, *ai = a->i, *aj = a->j, *bi = b->i, *bj = b->j;
  PetscInt       *ajtmp, *bjtmp, nz, nzL, row, *bdiag = b->diag, *pj;
  MatScalar      *rtmp, *pc, *mwork, *v, *pv, *aa     = a->a;
  PetscInt        bs = A->rmap->bs, bs2 = a->bs2, *v_pivots, flg;
  MatScalar      *v_work;
  PetscBool       col_identity, row_identity, both_identity;
  PetscBool       allowzeropivot, zeropivotdetected;

  PetscFunctionBegin;
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(isicol, &ic));
  allowzeropivot = PetscNot(A->erroriffailure);

  PetscCall(PetscCalloc1(bs2 * n, &rtmp));

  /* generate work space needed by dense LU factorization */
  PetscCall(PetscMalloc3(bs, &v_work, bs2, &mwork, bs, &v_pivots));

  for (i = 0; i < n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i + 1] - bi[i];
    bjtmp = bj + bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* U part */
    nz    = bdiag[i] - bdiag[i + 1];
    bjtmp = bj + bdiag[i + 1] + 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* load in initial (unfactored row) */
    nz    = ai[r[i] + 1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + bs2 * ai[r[i]];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(rtmp + bs2 * ic[ajtmp[j]], v + bs2 * j, bs2));

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i + 1] - bi[i];
    for (k = 0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2 * row;
      for (flg = 0, j = 0; j < bs2; j++) {
        if (pc[j] != 0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2 * bdiag[row];
        PetscKernel_A_gets_A_times_B(bs, pc, pv, mwork); /* *pc = *pc * (*pv); */
        pj = b->j + bdiag[row + 1] + 1;                  /* beginning of U(row,:) */
        pv = b->a + bs2 * (bdiag[row + 1] + 1);
        nz = bdiag[row] - bdiag[row + 1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j = 0; j < nz; j++) PetscKernel_A_gets_A_minus_B_times_C(bs, rtmp + bs2 * pj[j], pc, pv + bs2 * j);
        PetscCall(PetscLogFlops(2.0 * bs2 * bs * (nz + 1) - bs2)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2 * bi[i];
    pj = b->j + bi[i];
    nz = bi[i + 1] - bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv = b->a + bs2 * bdiag[i];
    pj = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv, rtmp + bs2 * pj[0], bs2));

    PetscCall(PetscKernel_A_gets_inverse_A(bs, pv, v_pivots, v_work, allowzeropivot, &zeropivotdetected));
    if (zeropivotdetected) B->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2 * (bdiag[i + 1] + 1);
    pj = b->j + bdiag[i + 1] + 1;
    nz = bdiag[i] - bdiag[i + 1] - 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));
  }

  PetscCall(PetscFree(rtmp));
  PetscCall(PetscFree3(v_work, mwork, v_pivots));
  PetscCall(ISRestoreIndices(isicol, &ic));
  PetscCall(ISRestoreIndices(isrow, &r));

  PetscCall(ISIdentity(isrow, &row_identity));
  PetscCall(ISIdentity(isicol, &col_identity));

  both_identity = (PetscBool)(row_identity && col_identity);
  if (both_identity) {
    switch (bs) {
    case 9:
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
      C->ops->solve = MatSolve_SeqBAIJ_9_NaturalOrdering;
#else
      C->ops->solve = MatSolve_SeqBAIJ_N_NaturalOrdering;
#endif
      break;
    case 11:
      C->ops->solve = MatSolve_SeqBAIJ_11_NaturalOrdering;
      break;
    case 12:
      C->ops->solve = MatSolve_SeqBAIJ_12_NaturalOrdering;
      break;
    case 13:
      C->ops->solve = MatSolve_SeqBAIJ_13_NaturalOrdering;
      break;
    case 14:
      C->ops->solve = MatSolve_SeqBAIJ_14_NaturalOrdering;
      break;
    default:
      C->ops->solve = MatSolve_SeqBAIJ_N_NaturalOrdering;
      break;
    }
  } else {
    C->ops->solve = MatSolve_SeqBAIJ_N;
  }
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_N;

  C->assembled = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333 * bs * bs2 * b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
   ilu(0) with natural ordering under new data structure.
   See MatILUFactorSymbolic_SeqAIJ_ilu0() for detailed description
   because this code is almost identical to MatILUFactorSymbolic_SeqAIJ_ilu0_inplace().
*/

PetscErrorCode MatILUFactorSymbolic_SeqBAIJ_ilu0(Mat fact, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *)A->data, *b;
  PetscInt     n = a->mbs, *ai = a->i, *aj, *adiag = a->diag, bs2 = a->bs2;
  PetscInt     i, j, nz, *bi, *bj, *bdiag, bi_temp;

  PetscFunctionBegin;
  PetscCall(MatDuplicateNoCreate_SeqBAIJ(fact, A, MAT_DO_NOT_COPY_VALUES, PETSC_FALSE));
  b = (Mat_SeqBAIJ *)(fact)->data;

  /* allocate matrix arrays for new data structure */
  PetscCall(PetscMalloc3(bs2 * ai[n] + 1, &b->a, ai[n] + 1, &b->j, n + 1, &b->i));

  b->singlemalloc    = PETSC_TRUE;
  b->free_a          = PETSC_TRUE;
  b->free_ij         = PETSC_TRUE;
  fact->preallocated = PETSC_TRUE;
  fact->assembled    = PETSC_TRUE;
  if (!b->diag) { PetscCall(PetscMalloc1(n + 1, &b->diag)); }
  bdiag = b->diag;

  if (n > 0) PetscCall(PetscArrayzero(b->a, bs2 * ai[n]));

  /* set bi and bj with new data structure */
  bi = b->i;
  bj = b->j;

  /* L part */
  bi[0] = 0;
  for (i = 0; i < n; i++) {
    nz        = adiag[i] - ai[i];
    bi[i + 1] = bi[i] + nz;
    aj        = a->j + ai[i];
    for (j = 0; j < nz; j++) {
      *bj = aj[j];
      bj++;
    }
  }

  /* U part */
  bi_temp  = bi[n];
  bdiag[n] = bi[n] - 1;
  for (i = n - 1; i >= 0; i--) {
    nz      = ai[i + 1] - adiag[i] - 1;
    bi_temp = bi_temp + nz + 1;
    aj      = a->j + adiag[i] + 1;
    for (j = 0; j < nz; j++) {
      *bj = aj[j];
      bj++;
    }
    /* diag[i] */
    *bj = i;
    bj++;
    bdiag[i] = bi_temp - 1;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatILUFactorSymbolic_SeqBAIJ(Mat fact, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data, *b;
  IS                 isicol;
  const PetscInt    *r, *ic;
  PetscInt           n = a->mbs, *ai = a->i, *aj = a->j, d;
  PetscInt          *bi, *cols, nnz, *cols_lvl;
  PetscInt          *bdiag, prow, fm, nzbd, reallocs = 0, dcount = 0;
  PetscInt           i, levels, diagonal_fill;
  PetscBool          col_identity, row_identity, both_identity;
  PetscReal          f;
  PetscInt           nlnk, *lnk, *lnk_lvl = NULL;
  PetscBT            lnkbt;
  PetscInt           nzi, *bj, **bj_ptr, **bjlvl_ptr;
  PetscFreeSpaceList free_space = NULL, current_space = NULL;
  PetscFreeSpaceList free_space_lvl = NULL, current_space_lvl = NULL;
  PetscBool          missing;
  PetscInt           bs = A->rmap->bs, bs2 = a->bs2;

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT, A->rmap->n, A->cmap->n);
  if (bs > 1) { /* check shifttype */
    PetscCheck(info->shifttype != MAT_SHIFT_NONZERO && info->shifttype != MAT_SHIFT_POSITIVE_DEFINITE, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only MAT_SHIFT_NONE and MAT_SHIFT_INBLOCKS are supported for BAIJ matrix");
  }

  PetscCall(MatMissingDiagonal(A, &missing, &d));
  PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry %" PetscInt_FMT, d);

  f             = info->fill;
  levels        = (PetscInt)info->levels;
  diagonal_fill = (PetscInt)info->diagonal_fill;

  PetscCall(ISInvertPermutation(iscol, PETSC_DECIDE, &isicol));

  PetscCall(ISIdentity(isrow, &row_identity));
  PetscCall(ISIdentity(iscol, &col_identity));

  both_identity = (PetscBool)(row_identity && col_identity);

  if (!levels && both_identity) {
    /* special case: ilu(0) with natural ordering */
    PetscCall(MatILUFactorSymbolic_SeqBAIJ_ilu0(fact, A, isrow, iscol, info));
    PetscCall(MatSeqBAIJSetNumericFactorization(fact, both_identity));

    fact->factortype               = MAT_FACTOR_ILU;
    (fact)->info.factor_mallocs    = 0;
    (fact)->info.fill_ratio_given  = info->fill;
    (fact)->info.fill_ratio_needed = 1.0;

    b       = (Mat_SeqBAIJ *)(fact)->data;
    b->row  = isrow;
    b->col  = iscol;
    b->icol = isicol;
    PetscCall(PetscObjectReference((PetscObject)isrow));
    PetscCall(PetscObjectReference((PetscObject)iscol));
    b->pivotinblocks = (info->pivotinblocks) ? PETSC_TRUE : PETSC_FALSE;

    PetscCall(PetscMalloc1((n + 1) * bs, &b->solve_work));
    PetscFunctionReturn(0);
  }

  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(isicol, &ic));

  /* get new row pointers */
  PetscCall(PetscMalloc1(n + 1, &bi));
  bi[0] = 0;
  /* bdiag is location of diagonal in factor */
  PetscCall(PetscMalloc1(n + 1, &bdiag));
  bdiag[0] = 0;

  PetscCall(PetscMalloc2(n, &bj_ptr, n, &bjlvl_ptr));

  /* create a linked list for storing column indices of the active row */
  nlnk = n + 1;
  PetscCall(PetscIncompleteLLCreate(n, n, nlnk, lnk, lnk_lvl, lnkbt));

  /* initial FreeSpace size is f*(ai[n]+1) */
  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(f, ai[n] + 1), &free_space));
  current_space = free_space;
  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(f, ai[n] + 1), &free_space_lvl));
  current_space_lvl = free_space_lvl;

  for (i = 0; i < n; i++) {
    nzi = 0;
    /* copy current row into linked list */
    nnz = ai[r[i] + 1] - ai[r[i]];
    PetscCheck(nnz, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Empty row in matrix: row in original ordering %" PetscInt_FMT " in permuted ordering %" PetscInt_FMT, r[i], i);
    cols   = aj + ai[r[i]];
    lnk[i] = -1; /* marker to indicate if diagonal exists */
    PetscCall(PetscIncompleteLLInit(nnz, cols, n, ic, &nlnk, lnk, lnk_lvl, lnkbt));
    nzi += nlnk;

    /* make sure diagonal entry is included */
    if (diagonal_fill && lnk[i] == -1) {
      fm = n;
      while (lnk[fm] < i) fm = lnk[fm];
      lnk[i]     = lnk[fm]; /* insert diagonal into linked list */
      lnk[fm]    = i;
      lnk_lvl[i] = 0;
      nzi++;
      dcount++;
    }

    /* add pivot rows into the active row */
    nzbd = 0;
    prow = lnk[n];
    while (prow < i) {
      nnz      = bdiag[prow];
      cols     = bj_ptr[prow] + nnz + 1;
      cols_lvl = bjlvl_ptr[prow] + nnz + 1;
      nnz      = bi[prow + 1] - bi[prow] - nnz - 1;

      PetscCall(PetscILULLAddSorted(nnz, cols, levels, cols_lvl, prow, &nlnk, lnk, lnk_lvl, lnkbt, prow));
      nzi += nlnk;
      prow = lnk[prow];
      nzbd++;
    }
    bdiag[i]  = nzbd;
    bi[i + 1] = bi[i] + nzi;

    /* if free space is not available, make more free space */
    if (current_space->local_remaining < nzi) {
      nnz = PetscIntMultTruncate(2, PetscIntMultTruncate(nzi, (n - i))); /* estimated and max additional space needed */
      PetscCall(PetscFreeSpaceGet(nnz, &current_space));
      PetscCall(PetscFreeSpaceGet(nnz, &current_space_lvl));
      reallocs++;
    }

    /* copy data into free_space and free_space_lvl, then initialize lnk */
    PetscCall(PetscIncompleteLLClean(n, n, nzi, lnk, lnk_lvl, current_space->array, current_space_lvl->array, lnkbt));

    bj_ptr[i]    = current_space->array;
    bjlvl_ptr[i] = current_space_lvl->array;

    /* make sure the active row i has diagonal entry */
    PetscCheck(*(bj_ptr[i] + bdiag[i]) == i, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Row %" PetscInt_FMT " has missing diagonal in factored matrix\ntry running with -pc_factor_nonzeros_along_diagonal or -pc_factor_diagonal_fill", i);

    current_space->array += nzi;
    current_space->local_used += nzi;
    current_space->local_remaining -= nzi;

    current_space_lvl->array += nzi;
    current_space_lvl->local_used += nzi;
    current_space_lvl->local_remaining -= nzi;
  }

  PetscCall(ISRestoreIndices(isrow, &r));
  PetscCall(ISRestoreIndices(isicol, &ic));

  /* copy free_space into bj and free free_space; set bi, bj, bdiag in new datastructure; */
  PetscCall(PetscMalloc1(bi[n] + 1, &bj));
  PetscCall(PetscFreeSpaceContiguous_LU(&free_space, bj, n, bi, bdiag));

  PetscCall(PetscIncompleteLLDestroy(lnk, lnkbt));
  PetscCall(PetscFreeSpaceDestroy(free_space_lvl));
  PetscCall(PetscFree2(bj_ptr, bjlvl_ptr));

#if defined(PETSC_USE_INFO)
  {
    PetscReal af = ((PetscReal)(bdiag[0] + 1)) / ((PetscReal)ai[n]);
    PetscCall(PetscInfo(A, "Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n", reallocs, (double)f, (double)af));
    PetscCall(PetscInfo(A, "Run with -[sub_]pc_factor_fill %g or use \n", (double)af));
    PetscCall(PetscInfo(A, "PCFactorSetFill([sub]pc,%g);\n", (double)af));
    PetscCall(PetscInfo(A, "for best performance.\n"));
    if (diagonal_fill) PetscCall(PetscInfo(A, "Detected and replaced %" PetscInt_FMT " missing diagonals\n", dcount));
  }
#endif

  /* put together the new matrix */
  PetscCall(MatSeqBAIJSetPreallocation(fact, bs, MAT_SKIP_ALLOCATION, NULL));

  b               = (Mat_SeqBAIJ *)(fact)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;

  PetscCall(PetscMalloc1(bs2 * (bdiag[0] + 1), &b->a));

  b->j         = bj;
  b->i         = bi;
  b->diag      = bdiag;
  b->free_diag = PETSC_TRUE;
  b->ilen      = NULL;
  b->imax      = NULL;
  b->row       = isrow;
  b->col       = iscol;
  PetscCall(PetscObjectReference((PetscObject)isrow));
  PetscCall(PetscObjectReference((PetscObject)iscol));
  b->icol = isicol;

  PetscCall(PetscMalloc1(bs * n + bs, &b->solve_work));
  /* In b structure:  Free imax, ilen, old a, old j.
     Allocate bdiag, solve_work, new a, new j */
  b->maxnz = b->nz = bdiag[0] + 1;

  fact->info.factor_mallocs    = reallocs;
  fact->info.fill_ratio_given  = f;
  fact->info.fill_ratio_needed = ((PetscReal)(bdiag[0] + 1)) / ((PetscReal)ai[n]);

  PetscCall(MatSeqBAIJSetNumericFactorization(fact, both_identity));
  PetscFunctionReturn(0);
}

/*
     This code is virtually identical to MatILUFactorSymbolic_SeqAIJ
   except that the data structure of Mat_SeqAIJ is slightly different.
   Not a good example of code reuse.
*/
PetscErrorCode MatILUFactorSymbolic_SeqBAIJ_inplace(Mat fact, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *)A->data, *b;
  IS              isicol;
  const PetscInt *r, *ic, *ai = a->i, *aj = a->j, *xi;
  PetscInt        prow, n = a->mbs, *ainew, *ajnew, jmax, *fill, nz, *im, *ajfill, *flev, *xitmp;
  PetscInt       *dloc, idx, row, m, fm, nzf, nzi, reallocate = 0, dcount = 0;
  PetscInt        incrlev, nnz, i, bs = A->rmap->bs, bs2 = a->bs2, levels, diagonal_fill, dd;
  PetscBool       col_identity, row_identity, both_identity, flg;
  PetscReal       f;

  PetscFunctionBegin;
  PetscCall(MatMissingDiagonal_SeqBAIJ(A, &flg, &dd));
  PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix A is missing diagonal entry in row %" PetscInt_FMT, dd);

  f             = info->fill;
  levels        = (PetscInt)info->levels;
  diagonal_fill = (PetscInt)info->diagonal_fill;

  PetscCall(ISInvertPermutation(iscol, PETSC_DECIDE, &isicol));

  PetscCall(ISIdentity(isrow, &row_identity));
  PetscCall(ISIdentity(iscol, &col_identity));
  both_identity = (PetscBool)(row_identity && col_identity);

  if (!levels && both_identity) { /* special case copy the nonzero structure */
    PetscCall(MatDuplicateNoCreate_SeqBAIJ(fact, A, MAT_DO_NOT_COPY_VALUES, PETSC_TRUE));
    PetscCall(MatSeqBAIJSetNumericFactorization_inplace(fact, both_identity));

    fact->factortype = MAT_FACTOR_ILU;
    b                = (Mat_SeqBAIJ *)fact->data;
    b->row           = isrow;
    b->col           = iscol;
    PetscCall(PetscObjectReference((PetscObject)isrow));
    PetscCall(PetscObjectReference((PetscObject)iscol));
    b->icol          = isicol;
    b->pivotinblocks = (info->pivotinblocks) ? PETSC_TRUE : PETSC_FALSE;

    PetscCall(PetscMalloc1((n + 1) * bs, &b->solve_work));
    PetscFunctionReturn(0);
  }

  /* general case perform the symbolic factorization */
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(isicol, &ic));

  /* get new row pointers */
  PetscCall(PetscMalloc1(n + 1, &ainew));
  ainew[0] = 0;
  /* don't know how many column pointers are needed so estimate */
  jmax = (PetscInt)(f * ai[n] + 1);
  PetscCall(PetscMalloc1(jmax, &ajnew));
  /* ajfill is level of fill for each fill entry */
  PetscCall(PetscMalloc1(jmax, &ajfill));
  /* fill is a linked list of nonzeros in active row */
  PetscCall(PetscMalloc1(n + 1, &fill));
  /* im is level for each filled value */
  PetscCall(PetscMalloc1(n + 1, &im));
  /* dloc is location of diagonal in factor */
  PetscCall(PetscMalloc1(n + 1, &dloc));
  dloc[0] = 0;
  for (prow = 0; prow < n; prow++) {
    /* copy prow into linked list */
    nzf = nz = ai[r[prow] + 1] - ai[r[prow]];
    PetscCheck(nz, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Empty row in matrix: row in original ordering %" PetscInt_FMT " in permuted ordering %" PetscInt_FMT, r[prow], prow);
    xi         = aj + ai[r[prow]];
    fill[n]    = n;
    fill[prow] = -1; /* marker for diagonal entry */
    while (nz--) {
      fm  = n;
      idx = ic[*xi++];
      do {
        m  = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m]   = idx;
      fill[idx] = fm;
      im[idx]   = 0;
    }

    /* make sure diagonal entry is included */
    if (diagonal_fill && fill[prow] == -1) {
      fm = n;
      while (fill[fm] < prow) fm = fill[fm];
      fill[prow] = fill[fm]; /* insert diagonal into linked list */
      fill[fm]   = prow;
      im[prow]   = 0;
      nzf++;
      dcount++;
    }

    nzi = 0;
    row = fill[n];
    while (row < prow) {
      incrlev = im[row] + 1;
      nz      = dloc[row];
      xi      = ajnew + ainew[row] + nz + 1;
      flev    = ajfill + ainew[row] + nz + 1;
      nnz     = ainew[row + 1] - ainew[row] - nz - 1;
      fm      = row;
      while (nnz-- > 0) {
        idx = *xi++;
        if (*flev + incrlev > levels) {
          flev++;
          continue;
        }
        do {
          m  = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          im[idx]   = *flev + incrlev;
          fill[m]   = idx;
          fill[idx] = fm;
          fm        = idx;
          nzf++;
        } else if (im[idx] > *flev + incrlev) im[idx] = *flev + incrlev;
        flev++;
      }
      row = fill[row];
      nzi++;
    }
    /* copy new filled row into permanent storage */
    ainew[prow + 1] = ainew[prow] + nzf;
    if (ainew[prow + 1] > jmax) {
      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      PetscInt maxadd = jmax;
      /* maxadd = (int)(((f*ai[n]+1)*(n-prow+5))/n); */
      if (maxadd < nzf) maxadd = (n - prow) * (nzf + 1);
      jmax += maxadd;

      /* allocate a longer ajnew and ajfill */
      PetscCall(PetscMalloc1(jmax, &xitmp));
      PetscCall(PetscArraycpy(xitmp, ajnew, ainew[prow]));
      PetscCall(PetscFree(ajnew));
      ajnew = xitmp;
      PetscCall(PetscMalloc1(jmax, &xitmp));
      PetscCall(PetscArraycpy(xitmp, ajfill, ainew[prow]));
      PetscCall(PetscFree(ajfill));
      ajfill = xitmp;
      reallocate++; /* count how many reallocations are needed */
    }
    xitmp      = ajnew + ainew[prow];
    flev       = ajfill + ainew[prow];
    dloc[prow] = nzi;
    fm         = fill[n];
    while (nzf--) {
      *xitmp++ = fm;
      *flev++  = im[fm];
      fm       = fill[fm];
    }
    /* make sure row has diagonal entry */
    PetscCheck(ajnew[ainew[prow] + dloc[prow]] == prow, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Row %" PetscInt_FMT " has missing diagonal in factored matrix\n\
                                                        try running with -pc_factor_nonzeros_along_diagonal or -pc_factor_diagonal_fill",
               prow);
  }
  PetscCall(PetscFree(ajfill));
  PetscCall(ISRestoreIndices(isrow, &r));
  PetscCall(ISRestoreIndices(isicol, &ic));
  PetscCall(PetscFree(fill));
  PetscCall(PetscFree(im));

#if defined(PETSC_USE_INFO)
  {
    PetscReal af = ((PetscReal)ainew[n]) / ((PetscReal)ai[n]);
    PetscCall(PetscInfo(A, "Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n", reallocate, (double)f, (double)af));
    PetscCall(PetscInfo(A, "Run with -pc_factor_fill %g or use \n", (double)af));
    PetscCall(PetscInfo(A, "PCFactorSetFill(pc,%g);\n", (double)af));
    PetscCall(PetscInfo(A, "for best performance.\n"));
    if (diagonal_fill) PetscCall(PetscInfo(A, "Detected and replaced %" PetscInt_FMT " missing diagonals\n", dcount));
  }
#endif

  /* put together the new matrix */
  PetscCall(MatSeqBAIJSetPreallocation(fact, bs, MAT_SKIP_ALLOCATION, NULL));
  b = (Mat_SeqBAIJ *)fact->data;

  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;

  PetscCall(PetscMalloc1(bs2 * ainew[n], &b->a));

  b->j = ajnew;
  b->i = ainew;
  for (i = 0; i < n; i++) dloc[i] += ainew[i];
  b->diag          = dloc;
  b->free_diag     = PETSC_TRUE;
  b->ilen          = NULL;
  b->imax          = NULL;
  b->row           = isrow;
  b->col           = iscol;
  b->pivotinblocks = (info->pivotinblocks) ? PETSC_TRUE : PETSC_FALSE;

  PetscCall(PetscObjectReference((PetscObject)isrow));
  PetscCall(PetscObjectReference((PetscObject)iscol));
  b->icol = isicol;
  PetscCall(PetscMalloc1(bs * n + bs, &b->solve_work));
  /* In b structure:  Free imax, ilen, old a, old j.
     Allocate dloc, solve_work, new a, new j */
  b->maxnz = b->nz = ainew[n];

  fact->info.factor_mallocs    = reallocate;
  fact->info.fill_ratio_given  = f;
  fact->info.fill_ratio_needed = ((PetscReal)ainew[n]) / ((PetscReal)ai[prow]);

  PetscCall(MatSeqBAIJSetNumericFactorization_inplace(fact, both_identity));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE(Mat A)
{
  /* Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data; */
  /* int i,*AJ=a->j,nz=a->nz; */

  PetscFunctionBegin;
  /* Undo Column scaling */
  /*    while (nz--) { */
  /*      AJ[i] = AJ[i]/4; */
  /*    } */
  /* This should really invoke a push/pop logic, but we don't have that yet. */
  A->ops->setunfactored = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE_usj(Mat A)
{
  Mat_SeqBAIJ    *a  = (Mat_SeqBAIJ *)A->data;
  PetscInt       *AJ = a->j, nz = a->nz;
  unsigned short *aj = (unsigned short *)AJ;

  PetscFunctionBegin;
  /* Is this really necessary? */
  while (nz--) { AJ[nz] = (int)((unsigned int)aj[nz]); /* First extend, then convert to signed. */ }
  A->ops->setunfactored = NULL;
  PetscFunctionReturn(0);
}
