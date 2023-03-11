
/*
    Factorization code for BAIJ format.
*/
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
   This is used to set the numeric factorization for both LU and ILU symbolic factorization
*/
PetscErrorCode MatSeqBAIJSetNumericFactorization(Mat fact, PetscBool natural)
{
  PetscFunctionBegin;
  if (natural) {
    switch (fact->rmap->bs) {
    case 1:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;
      break;
    case 2:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering;
      break;
    case 3:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering;
      break;
    case 4:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering;
      break;
    case 5:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering;
      break;
    case 6:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering;
      break;
    case 7:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering;
      break;
    case 9:
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_9_NaturalOrdering;
#else
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;
#endif
      break;
    case 15:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_15_NaturalOrdering;
      break;
    default:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;
      break;
    }
  } else {
    switch (fact->rmap->bs) {
    case 1:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;
      break;
    case 2:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2;
      break;
    case 3:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3;
      break;
    case 4:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4;
      break;
    case 5:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5;
      break;
    case 6:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6;
      break;
    case 7:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7;
      break;
    default:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqBAIJSetNumericFactorization_inplace(Mat inA, PetscBool natural)
{
  PetscFunctionBegin;
  if (natural) {
    switch (inA->rmap->bs) {
    case 1:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1_inplace;
      break;
    case 2:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering_inplace;
      break;
    case 3:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering_inplace;
      break;
    case 4:
#if defined(PETSC_USE_REAL_MAT_SINGLE)
    {
      PetscBool sse_enabled_local;
      PetscCall(PetscSSEIsEnabled(inA->comm, &sse_enabled_local, NULL));
      if (sse_enabled_local) {
  #if defined(PETSC_HAVE_SSE)
        int i, *AJ = a->j, nz = a->nz, n = a->mbs;
        if (n == (unsigned short)n) {
          unsigned short *aj = (unsigned short *)AJ;
          for (i = 0; i < nz; i++) aj[i] = (unsigned short)AJ[i];

          inA->ops->setunfactored   = MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE_usj;
          inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE_usj;

          PetscCall(PetscInfo(inA, "Using special SSE, in-place natural ordering, ushort j index factor BS=4\n"));
        } else {
          /* Scale the column indices for easier indexing in MatSolve. */
          /*            for (i=0;i<nz;i++) { */
          /*              AJ[i] = AJ[i]*4; */
          /*            } */
          inA->ops->setunfactored   = MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE;
          inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE;

          PetscCall(PetscInfo(inA, "Using special SSE, in-place natural ordering, int j index factor BS=4\n"));
        }
  #else
        /* This should never be reached.  If so, problem in PetscSSEIsEnabled. */
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "SSE Hardware unavailable");
  #endif
      } else {
        inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_inplace;
      }
    }
#else
      inA->ops->lufactornumeric  = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_inplace;
#endif
    break;
    case 5:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering_inplace;
      break;
    case 6:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering_inplace;
      break;
    case 7:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering_inplace;
      break;
    default:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N_inplace;
      break;
    }
  } else {
    switch (inA->rmap->bs) {
    case 1:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1_inplace;
      break;
    case 2:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2_inplace;
      break;
    case 3:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3_inplace;
      break;
    case 4:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_inplace;
      break;
    case 5:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_inplace;
      break;
    case 6:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6_inplace;
      break;
    case 7:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7_inplace;
      break;
    default:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N_inplace;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    The symbolic factorization code is identical to that for AIJ format,
  except for very small changes since this is now a SeqBAIJ datastructure.
  NOT good code reuse.
*/
#include <petscbt.h>
#include <../src/mat/utils/freespace.h>

PetscErrorCode MatLUFactorSymbolic_SeqBAIJ(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data, *b;
  PetscInt           n = a->mbs, bs = A->rmap->bs, bs2 = a->bs2;
  PetscBool          row_identity, col_identity, both_identity;
  IS                 isicol;
  const PetscInt    *r, *ic;
  PetscInt           i, *ai = a->i, *aj = a->j;
  PetscInt          *bi, *bj, *ajtmp;
  PetscInt          *bdiag, row, nnz, nzi, reallocs = 0, nzbd, *im;
  PetscReal          f;
  PetscInt           nlnk, *lnk, k, **bi_ptr;
  PetscFreeSpaceList free_space = NULL, current_space = NULL;
  PetscBT            lnkbt;
  PetscBool          missing;

  PetscFunctionBegin;
  PetscCheck(A->rmap->N == A->cmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "matrix must be square");
  PetscCall(MatMissingDiagonal(A, &missing, &i));
  PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry %" PetscInt_FMT, i);

  if (bs > 1) { /* check shifttype */
    PetscCheck(info->shifttype != (PetscReal)MAT_SHIFT_NONZERO && info->shifttype != (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only MAT_SHIFT_NONE and MAT_SHIFT_INBLOCKS are supported for BAIJ matrix");
  }

  PetscCall(ISInvertPermutation(iscol, PETSC_DECIDE, &isicol));
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(isicol, &ic));

  /* get new row and diagonal pointers, must be allocated separately because they will be given to the Mat_SeqAIJ and freed separately */
  PetscCall(PetscMalloc1(n + 1, &bi));
  PetscCall(PetscMalloc1(n + 1, &bdiag));
  bi[0] = bdiag[0] = 0;

  /* linked list for storing column indices of the active row */
  nlnk = n + 1;
  PetscCall(PetscLLCreate(n, n, nlnk, lnk, lnkbt));

  PetscCall(PetscMalloc2(n + 1, &bi_ptr, n + 1, &im));

  /* initial FreeSpace size is f*(ai[n]+1) */
  f = info->fill;
  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(f, ai[n] + 1), &free_space));

  current_space = free_space;

  for (i = 0; i < n; i++) {
    /* copy previous fill into linked list */
    nzi   = 0;
    nnz   = ai[r[i] + 1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    PetscCall(PetscLLAddPerm(nnz, ajtmp, ic, n, &nlnk, lnk, lnkbt));
    nzi += nlnk;

    /* add pivot rows into linked list */
    row = lnk[n];
    while (row < i) {
      nzbd  = bdiag[row] + 1;     /* num of entries in the row with column index <= row */
      ajtmp = bi_ptr[row] + nzbd; /* points to the entry next to the diagonal */
      PetscCall(PetscLLAddSortedLU(ajtmp, row, &nlnk, lnk, lnkbt, i, nzbd, im));
      nzi += nlnk;
      row = lnk[row];
    }
    bi[i + 1] = bi[i] + nzi;
    im[i]     = nzi;

    /* mark bdiag */
    nzbd = 0;
    nnz  = nzi;
    k    = lnk[n];
    while (nnz-- && k < i) {
      nzbd++;
      k = lnk[k];
    }
    bdiag[i] = nzbd; /* note : bdaig[i] = nnzL as input for PetscFreeSpaceContiguous_LU() */

    /* if free space is not available, make more free space */
    if (current_space->local_remaining < nzi) {
      nnz = PetscIntMultTruncate(2, PetscIntMultTruncate(n - i, nzi)); /* estimated and max additional space needed */
      PetscCall(PetscFreeSpaceGet(nnz, &current_space));
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    PetscCall(PetscLLClean(n, n, nzi, lnk, current_space->array, lnkbt));

    bi_ptr[i] = current_space->array;
    current_space->array += nzi;
    current_space->local_used += nzi;
    current_space->local_remaining -= nzi;
  }

  PetscCall(ISRestoreIndices(isrow, &r));
  PetscCall(ISRestoreIndices(isicol, &ic));

  /* copy free_space into bj and free free_space; set bi, bj, bdiag in new datastructure; */
  PetscCall(PetscMalloc1(bi[n] + 1, &bj));
  PetscCall(PetscFreeSpaceContiguous_LU(&free_space, bj, n, bi, bdiag));
  PetscCall(PetscLLDestroy(lnk, lnkbt));
  PetscCall(PetscFree2(bi_ptr, im));

  /* put together the new matrix */
  PetscCall(MatSeqBAIJSetPreallocation(B, bs, MAT_SKIP_ALLOCATION, NULL));
  b = (Mat_SeqBAIJ *)(B)->data;

  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;

  PetscCall(PetscMalloc1((bdiag[0] + 1) * bs2, &b->a));
  b->j             = bj;
  b->i             = bi;
  b->diag          = bdiag;
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

  b->maxnz = b->nz = bdiag[0] + 1;

  B->factortype            = MAT_FACTOR_LU;
  B->info.factor_mallocs   = reallocs;
  B->info.fill_ratio_given = f;

  if (ai[n] != 0) {
    B->info.fill_ratio_needed = ((PetscReal)(bdiag[0] + 1)) / ((PetscReal)ai[n]);
  } else {
    B->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[n] != 0) {
    PetscReal af = B->info.fill_ratio_needed;
    PetscCall(PetscInfo(A, "Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n", reallocs, (double)f, (double)af));
    PetscCall(PetscInfo(A, "Run with -pc_factor_fill %g or use \n", (double)af));
    PetscCall(PetscInfo(A, "PCFactorSetFill(pc,%g);\n", (double)af));
    PetscCall(PetscInfo(A, "for best performance.\n"));
  } else {
    PetscCall(PetscInfo(A, "Empty matrix\n"));
  }
#endif

  PetscCall(ISIdentity(isrow, &row_identity));
  PetscCall(ISIdentity(iscol, &col_identity));

  both_identity = (PetscBool)(row_identity && col_identity);

  PetscCall(MatSeqBAIJSetNumericFactorization(B, both_identity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
// unused
static PetscErrorCode MatLUFactorSymbolic_SeqBAIJ_inplace(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data, *b;
  PetscInt           n = a->mbs, bs = A->rmap->bs, bs2 = a->bs2;
  PetscBool          row_identity, col_identity, both_identity;
  IS                 isicol;
  const PetscInt    *r, *ic;
  PetscInt           i, *ai = a->i, *aj = a->j;
  PetscInt          *bi, *bj, *ajtmp;
  PetscInt          *bdiag, row, nnz, nzi, reallocs = 0, nzbd, *im;
  PetscReal          f;
  PetscInt           nlnk, *lnk, k, **bi_ptr;
  PetscFreeSpaceList free_space = NULL, current_space = NULL;
  PetscBT            lnkbt;
  PetscBool          missing;

  PetscFunctionBegin;
  PetscCheck(A->rmap->N == A->cmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "matrix must be square");
  PetscCall(MatMissingDiagonal(A, &missing, &i));
  PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry %" PetscInt_FMT, i);

  PetscCall(ISInvertPermutation(iscol, PETSC_DECIDE, &isicol));
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(isicol, &ic));

  /* get new row and diagonal pointers, must be allocated separately because they will be given to the Mat_SeqAIJ and freed separately */
  PetscCall(PetscMalloc1(n + 1, &bi));
  PetscCall(PetscMalloc1(n + 1, &bdiag));

  bi[0] = bdiag[0] = 0;

  /* linked list for storing column indices of the active row */
  nlnk = n + 1;
  PetscCall(PetscLLCreate(n, n, nlnk, lnk, lnkbt));

  PetscCall(PetscMalloc2(n + 1, &bi_ptr, n + 1, &im));

  /* initial FreeSpace size is f*(ai[n]+1) */
  f = info->fill;
  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(f, ai[n] + 1), &free_space));
  current_space = free_space;

  for (i = 0; i < n; i++) {
    /* copy previous fill into linked list */
    nzi   = 0;
    nnz   = ai[r[i] + 1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    PetscCall(PetscLLAddPerm(nnz, ajtmp, ic, n, &nlnk, lnk, lnkbt));
    nzi += nlnk;

    /* add pivot rows into linked list */
    row = lnk[n];
    while (row < i) {
      nzbd  = bdiag[row] - bi[row] + 1; /* num of entries in the row with column index <= row */
      ajtmp = bi_ptr[row] + nzbd;       /* points to the entry next to the diagonal */
      PetscCall(PetscLLAddSortedLU(ajtmp, row, &nlnk, lnk, lnkbt, i, nzbd, im));
      nzi += nlnk;
      row = lnk[row];
    }
    bi[i + 1] = bi[i] + nzi;
    im[i]     = nzi;

    /* mark bdiag */
    nzbd = 0;
    nnz  = nzi;
    k    = lnk[n];
    while (nnz-- && k < i) {
      nzbd++;
      k = lnk[k];
    }
    bdiag[i] = bi[i] + nzbd;

    /* if free space is not available, make more free space */
    if (current_space->local_remaining < nzi) {
      nnz = PetscIntMultTruncate(n - i, nzi); /* estimated and max additional space needed */
      PetscCall(PetscFreeSpaceGet(nnz, &current_space));
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    PetscCall(PetscLLClean(n, n, nzi, lnk, current_space->array, lnkbt));

    bi_ptr[i] = current_space->array;
    current_space->array += nzi;
    current_space->local_used += nzi;
    current_space->local_remaining -= nzi;
  }
  #if defined(PETSC_USE_INFO)
  if (ai[n] != 0) {
    PetscReal af = ((PetscReal)bi[n]) / ((PetscReal)ai[n]);
    PetscCall(PetscInfo(A, "Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n", reallocs, (double)f, (double)af));
    PetscCall(PetscInfo(A, "Run with -pc_factor_fill %g or use \n", (double)af));
    PetscCall(PetscInfo(A, "PCFactorSetFill(pc,%g);\n", (double)af));
    PetscCall(PetscInfo(A, "for best performance.\n"));
  } else {
    PetscCall(PetscInfo(A, "Empty matrix\n"));
  }
  #endif

  PetscCall(ISRestoreIndices(isrow, &r));
  PetscCall(ISRestoreIndices(isicol, &ic));

  /* destroy list of free space and other temporary array(s) */
  PetscCall(PetscMalloc1(bi[n] + 1, &bj));
  PetscCall(PetscFreeSpaceContiguous(&free_space, bj));
  PetscCall(PetscLLDestroy(lnk, lnkbt));
  PetscCall(PetscFree2(bi_ptr, im));

  /* put together the new matrix */
  PetscCall(MatSeqBAIJSetPreallocation(B, bs, MAT_SKIP_ALLOCATION, NULL));
  b = (Mat_SeqBAIJ *)(B)->data;

  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;

  PetscCall(PetscMalloc1((bi[n] + 1) * bs2, &b->a));
  b->j             = bj;
  b->i             = bi;
  b->diag          = bdiag;
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

  b->maxnz = b->nz = bi[n];

  (B)->factortype            = MAT_FACTOR_LU;
  (B)->info.factor_mallocs   = reallocs;
  (B)->info.fill_ratio_given = f;

  if (ai[n] != 0) {
    (B)->info.fill_ratio_needed = ((PetscReal)bi[n]) / ((PetscReal)ai[n]);
  } else {
    (B)->info.fill_ratio_needed = 0.0;
  }

  PetscCall(ISIdentity(isrow, &row_identity));
  PetscCall(ISIdentity(iscol, &col_identity));

  both_identity = (PetscBool)(row_identity && col_identity);

  PetscCall(MatSeqBAIJSetNumericFactorization_inplace(B, both_identity));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
