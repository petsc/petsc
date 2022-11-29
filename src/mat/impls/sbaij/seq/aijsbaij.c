
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqSBAIJ_SeqAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat           B;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ *)A->data;
  Mat_SeqAIJ   *b;
  PetscInt     *ai = a->i, *aj = a->j, m = A->rmap->N, n = A->cmap->n, i, j, k, *bi, *bj, *rowlengths, nz, *rowstart, itmp;
  PetscInt      bs = A->rmap->bs, bs2 = bs * bs, mbs = A->rmap->N / bs, diagcnt = 0;
  MatScalar    *av, *bv;
#if defined(PETSC_USE_COMPLEX)
  const int aconj = A->hermitian == PETSC_BOOL3_TRUE ? 1 : 0;
#else
  const int aconj = 0;
#endif

  PetscFunctionBegin;
  /* compute rowlengths of newmat */
  PetscCall(PetscMalloc2(m, &rowlengths, m + 1, &rowstart));

  for (i = 0; i < mbs; i++) rowlengths[i * bs] = 0;
  k = 0;
  for (i = 0; i < mbs; i++) {
    nz = ai[i + 1] - ai[i];
    if (nz) {
      rowlengths[k] += nz; /* no. of upper triangular blocks */
      if (*aj == i) {
        aj++;
        diagcnt++;
        nz--;
      }                          /* skip diagonal */
      for (j = 0; j < nz; j++) { /* no. of lower triangular blocks */
        rowlengths[(*aj) * bs]++;
        aj++;
      }
    }
    rowlengths[k] *= bs;
    for (j = 1; j < bs; j++) rowlengths[k + j] = rowlengths[k];
    k += bs;
  }

  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetSizes(B, m, n, m, n));
    PetscCall(MatSetType(B, MATSEQAIJ));
    PetscCall(MatSeqAIJSetPreallocation(B, 0, rowlengths));
    PetscCall(MatSetBlockSize(B, A->rmap->bs));
  } else B = *newmat;

  b  = (Mat_SeqAIJ *)(B->data);
  bi = b->i;
  bj = b->j;
  bv = b->a;

  /* set b->i */
  bi[0]       = 0;
  rowstart[0] = 0;
  for (i = 0; i < mbs; i++) {
    for (j = 0; j < bs; j++) {
      b->ilen[i * bs + j]      = rowlengths[i * bs];
      rowstart[i * bs + j + 1] = rowstart[i * bs + j] + rowlengths[i * bs];
    }
    bi[i + 1] = bi[i] + rowlengths[i * bs] / bs;
  }
  PetscCheck(bi[mbs] == 2 * a->nz - diagcnt, PETSC_COMM_SELF, PETSC_ERR_PLIB, "bi[mbs]: %" PetscInt_FMT " != 2*a->nz-diagcnt: %" PetscInt_FMT, bi[mbs], 2 * a->nz - diagcnt);

  /* set b->j and b->a */
  aj = a->j;
  av = a->a;
  for (i = 0; i < mbs; i++) {
    nz = ai[i + 1] - ai[i];
    /* diagonal block */
    if (nz && *aj == i) {
      nz--;
      for (j = 0; j < bs; j++) { /* row i*bs+j */
        itmp = i * bs + j;
        for (k = 0; k < bs; k++) { /* col i*bs+k */
          *(bj + rowstart[itmp]) = (*aj) * bs + k;
          *(bv + rowstart[itmp]) = *(av + k * bs + j);
          rowstart[itmp]++;
        }
      }
      aj++;
      av += bs2;
    }

    while (nz--) {
      /* lower triangular blocks */
      for (j = 0; j < bs; j++) { /* row (*aj)*bs+j */
        itmp = (*aj) * bs + j;
        for (k = 0; k < bs; k++) { /* col i*bs+k */
          *(bj + rowstart[itmp]) = i * bs + k;
          *(bv + rowstart[itmp]) = aconj ? PetscConj(*(av + j * bs + k)) : *(av + j * bs + k);
          rowstart[itmp]++;
        }
      }
      /* upper triangular blocks */
      for (j = 0; j < bs; j++) { /* row i*bs+j */
        itmp = i * bs + j;
        for (k = 0; k < bs; k++) { /* col (*aj)*bs+k */
          *(bj + rowstart[itmp]) = (*aj) * bs + k;
          *(bv + rowstart[itmp]) = *(av + k * bs + j);
          rowstart[itmp]++;
        }
      }
      aj++;
      av += bs2;
    }
  }
  PetscCall(PetscFree2(rowlengths, rowstart));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqSBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat           B;
  Mat_SeqAIJ   *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqSBAIJ *b;
  PetscInt     *ai = a->i, *aj, m = A->rmap->N, n = A->cmap->N, i, j, *bi, *bj, *rowlengths, bs = PetscAbs(A->rmap->bs);
  MatScalar    *av, *bv;
  PetscBool     miss = PETSC_FALSE;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscCheck(A->symmetric == PETSC_BOOL3_TRUE, PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "Matrix must be symmetric. Call MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE)");
#else
  PetscCheck(A->symmetric == PETSC_BOOL3_TRUE || A->hermitian == PETSC_BOOL3_TRUE, PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "Matrix must be either symmetric or hermitian. Call MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE) and/or MatSetOption(mat,MAT_HERMITIAN,PETSC_TRUE)");
#endif
  PetscCheck(n == m, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square");

  PetscCall(PetscMalloc1(m / bs, &rowlengths));
  for (i = 0; i < m / bs; i++) {
    if (a->diag[i * bs] == ai[i * bs + 1]) {              /* missing diagonal */
      rowlengths[i] = (ai[i * bs + 1] - ai[i * bs]) / bs; /* allocate some extra space */
      miss          = PETSC_TRUE;
    } else {
      rowlengths[i] = (ai[i * bs + 1] - a->diag[i * bs]) / bs;
    }
  }
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetSizes(B, m, n, m, n));
    PetscCall(MatSetType(B, MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(B, bs, 0, rowlengths));
  } else B = *newmat;

  if (bs == 1 && !miss) {
    b  = (Mat_SeqSBAIJ *)(B->data);
    bi = b->i;
    bj = b->j;
    bv = b->a;

    bi[0] = 0;
    for (i = 0; i < m; i++) {
      aj = a->j + a->diag[i];
      av = a->a + a->diag[i];
      for (j = 0; j < rowlengths[i]; j++) {
        *bj = *aj;
        bj++;
        aj++;
        *bv = *av;
        bv++;
        av++;
      }
      bi[i + 1]  = bi[i] + rowlengths[i];
      b->ilen[i] = rowlengths[i];
    }
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  } else {
    PetscCall(MatSetOption(B, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
    /* reuse may not be equal to MAT_REUSE_MATRIX, but the basic converter will reallocate or replace newmat if this value is not used */
    /* if reuse is equal to MAT_INITIAL_MATRIX, it has been appropriately preallocated before                                          */
    /*                      MAT_INPLACE_MATRIX, it will be replaced with MatHeaderReplace below                                        */
    PetscCall(MatConvert_Basic(A, newtype, MAT_REUSE_MATRIX, &B));
  }
  PetscCall(PetscFree(rowlengths));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else *newmat = B;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqSBAIJ_SeqBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat           B;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ *)A->data;
  Mat_SeqBAIJ  *b;
  PetscInt     *ai = a->i, *aj = a->j, m = A->rmap->N, n = A->cmap->n, i, k, *bi, *bj, *browlengths, nz, *browstart, itmp;
  PetscInt      bs = A->rmap->bs, bs2 = bs * bs, mbs = m / bs, col, row;
  MatScalar    *av, *bv;
#if defined(PETSC_USE_COMPLEX)
  const int aconj = A->hermitian == PETSC_BOOL3_TRUE ? 1 : 0;
#else
  const int aconj = 0;
#endif

  PetscFunctionBegin;
  /* compute browlengths of newmat */
  PetscCall(PetscMalloc2(mbs, &browlengths, mbs, &browstart));
  for (i = 0; i < mbs; i++) browlengths[i] = 0;
  for (i = 0; i < mbs; i++) {
    nz = ai[i + 1] - ai[i];
    aj++;                      /* skip diagonal */
    for (k = 1; k < nz; k++) { /* no. of lower triangular blocks */
      browlengths[*aj]++;
      aj++;
    }
    browlengths[i] += nz; /* no. of upper triangular blocks */
  }

  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetSizes(B, m, n, m, n));
    PetscCall(MatSetType(B, MATSEQBAIJ));
    PetscCall(MatSeqBAIJSetPreallocation(B, bs, 0, browlengths));
  } else B = *newmat;

  b  = (Mat_SeqBAIJ *)(B->data);
  bi = b->i;
  bj = b->j;
  bv = b->a;

  /* set b->i */
  bi[0] = 0;
  for (i = 0; i < mbs; i++) {
    b->ilen[i]   = browlengths[i];
    bi[i + 1]    = bi[i] + browlengths[i];
    browstart[i] = bi[i];
  }
  PetscCheck(bi[mbs] == 2 * a->nz - mbs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "bi[mbs]: %" PetscInt_FMT " != 2*a->nz - mbs: %" PetscInt_FMT, bi[mbs], 2 * a->nz - mbs);

  /* set b->j and b->a */
  aj = a->j;
  av = a->a;
  for (i = 0; i < mbs; i++) {
    /* diagonal block */
    *(bj + browstart[i]) = *aj;
    aj++;

    itmp = bs2 * browstart[i];
    for (k = 0; k < bs2; k++) {
      *(bv + itmp + k) = *av;
      av++;
    }
    browstart[i]++;

    nz = ai[i + 1] - ai[i] - 1;
    while (nz--) {
      /* lower triangular blocks - transpose blocks of A */
      *(bj + browstart[*aj]) = i; /* block col index */

      itmp = bs2 * browstart[*aj]; /* row index */
      for (col = 0; col < bs; col++) {
        k = col;
        for (row = 0; row < bs; row++) {
          bv[itmp + col * bs + row] = aconj ? PetscConj(av[k]) : av[k];
          k += bs;
        }
      }
      browstart[*aj]++;

      /* upper triangular blocks */
      *(bj + browstart[i]) = *aj;
      aj++;

      itmp = bs2 * browstart[i];
      for (k = 0; k < bs2; k++) bv[itmp + k] = av[k];
      av += bs2;
      browstart[i]++;
    }
  }
  PetscCall(PetscFree2(browlengths, browstart));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else *newmat = B;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqSBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat           B;
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ *)A->data;
  Mat_SeqSBAIJ *b;
  PetscInt     *ai = a->i, *aj, m = A->rmap->N, n = A->cmap->n, i, j, k, *bi, *bj, *browlengths;
  PetscInt      bs = A->rmap->bs, bs2 = bs * bs, mbs = m / bs, dd;
  MatScalar    *av, *bv;
  PetscBool     flg;

  PetscFunctionBegin;
  PetscCheck(A->symmetric, PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "Matrix must be symmetric. Call MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE)");
  PetscCheck(n == m, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square");
  PetscCall(MatMissingDiagonal_SeqBAIJ(A, &flg, &dd)); /* check for missing diagonals, then mark diag */
  PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal %" PetscInt_FMT, dd);

  PetscCall(PetscMalloc1(mbs, &browlengths));
  for (i = 0; i < mbs; i++) browlengths[i] = ai[i + 1] - a->diag[i];

  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetSizes(B, m, n, m, n));
    PetscCall(MatSetType(B, MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(B, bs, 0, browlengths));
  } else B = *newmat;

  b  = (Mat_SeqSBAIJ *)(B->data);
  bi = b->i;
  bj = b->j;
  bv = b->a;

  bi[0] = 0;
  for (i = 0; i < mbs; i++) {
    aj = a->j + a->diag[i];
    av = a->a + (a->diag[i]) * bs2;
    for (j = 0; j < browlengths[i]; j++) {
      *bj = *aj;
      bj++;
      aj++;
      for (k = 0; k < bs2; k++) {
        *bv = *av;
        bv++;
        av++;
      }
    }
    bi[i + 1]  = bi[i] + browlengths[i];
    b->ilen[i] = browlengths[i];
  }
  PetscCall(PetscFree(browlengths));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else *newmat = B;
  PetscFunctionReturn(0);
}
