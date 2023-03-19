
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/dense/seq/dense.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petscbt.h>
#include <petscblaslapack.h>

PetscErrorCode MatIncreaseOverlap_SeqSBAIJ(Mat A, PetscInt is_max, IS is[], PetscInt ov)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ *)A->data;
  PetscInt        brow, i, j, k, l, mbs, n, *nidx, isz, bcol, bcol_max, start, end, *ai, *aj, bs;
  const PetscInt *idx;
  PetscBT         table_out, table_in;

  PetscFunctionBegin;
  PetscCheck(ov >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative overlap specified");
  mbs = a->mbs;
  ai  = a->i;
  aj  = a->j;
  bs  = A->rmap->bs;
  PetscCall(PetscBTCreate(mbs, &table_out));
  PetscCall(PetscMalloc1(mbs + 1, &nidx));
  PetscCall(PetscBTCreate(mbs, &table_in));

  for (i = 0; i < is_max; i++) { /* for each is */
    isz = 0;
    PetscCall(PetscBTMemzero(mbs, table_out));

    /* Extract the indices, assume there can be duplicate entries */
    PetscCall(ISGetIndices(is[i], &idx));
    PetscCall(ISGetLocalSize(is[i], &n));

    /* Enter these into the temp arrays i.e mark table_out[brow], enter brow into new index */
    bcol_max = 0;
    for (j = 0; j < n; ++j) {
      brow = idx[j] / bs; /* convert the indices into block indices */
      PetscCheck(brow < mbs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index greater than mat-dim");
      if (!PetscBTLookupSet(table_out, brow)) {
        nidx[isz++] = brow;
        if (bcol_max < brow) bcol_max = brow;
      }
    }
    PetscCall(ISRestoreIndices(is[i], &idx));
    PetscCall(ISDestroy(&is[i]));

    k = 0;
    for (j = 0; j < ov; j++) { /* for each overlap */
      /* set table_in for lookup - only mark entries that are added onto nidx in (j-1)-th overlap */
      PetscCall(PetscBTMemzero(mbs, table_in));
      for (l = k; l < isz; l++) PetscCall(PetscBTSet(table_in, nidx[l]));

      n = isz; /* length of the updated is[i] */
      for (brow = 0; brow < mbs; brow++) {
        start = ai[brow];
        end   = ai[brow + 1];
        if (PetscBTLookup(table_in, brow)) { /* brow is on nidx - row search: collect all bcol in this brow */
          for (l = start; l < end; l++) {
            bcol = aj[l];
            if (!PetscBTLookupSet(table_out, bcol)) {
              nidx[isz++] = bcol;
              if (bcol_max < bcol) bcol_max = bcol;
            }
          }
          k++;
          if (k >= n) break; /* for (brow=0; brow<mbs; brow++) */
        } else {             /* brow is not on nidx - col search: add brow onto nidx if there is a bcol in nidx */
          for (l = start; l < end; l++) {
            bcol = aj[l];
            if (bcol > bcol_max) break;
            if (PetscBTLookup(table_in, bcol)) {
              if (!PetscBTLookupSet(table_out, brow)) nidx[isz++] = brow;
              break; /* for l = start; l<end ; l++) */
            }
          }
        }
      }
    } /* for each overlap */
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, isz, nidx, PETSC_COPY_VALUES, is + i));
  } /* for each is */
  PetscCall(PetscBTDestroy(&table_out));
  PetscCall(PetscFree(nidx));
  PetscCall(PetscBTDestroy(&table_in));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Bseq is non-symmetric SBAIJ matrix, only used internally by PETSc.
        Zero some ops' to avoid invalid usse */
PetscErrorCode MatSeqSBAIJZeroOps_Private(Mat Bseq)
{
  PetscFunctionBegin;
  PetscCall(MatSetOption(Bseq, MAT_SYMMETRIC, PETSC_FALSE));
  Bseq->ops->mult                   = NULL;
  Bseq->ops->multadd                = NULL;
  Bseq->ops->multtranspose          = NULL;
  Bseq->ops->multtransposeadd       = NULL;
  Bseq->ops->lufactor               = NULL;
  Bseq->ops->choleskyfactor         = NULL;
  Bseq->ops->lufactorsymbolic       = NULL;
  Bseq->ops->choleskyfactorsymbolic = NULL;
  Bseq->ops->getinertia             = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* same as MatCreateSubMatrices_SeqBAIJ(), except cast Mat_SeqSBAIJ */
PetscErrorCode MatCreateSubMatrix_SeqSBAIJ_Private(Mat A, IS isrow, IS iscol, MatReuse scall, Mat *B)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ *)A->data, *c;
  PetscInt       *smap, i, k, kstart, kend, oldcols = a->nbs, *lens;
  PetscInt        row, mat_i, *mat_j, tcol, *mat_ilen;
  const PetscInt *irow, *icol;
  PetscInt        nrows, ncols, *ssmap, bs = A->rmap->bs, bs2 = a->bs2;
  PetscInt       *aj = a->j, *ai = a->i;
  MatScalar      *mat_a;
  Mat             C;
  PetscBool       flag;

  PetscFunctionBegin;

  PetscCall(ISGetIndices(isrow, &irow));
  PetscCall(ISGetIndices(iscol, &icol));
  PetscCall(ISGetLocalSize(isrow, &nrows));
  PetscCall(ISGetLocalSize(iscol, &ncols));

  PetscCall(PetscCalloc1(1 + oldcols, &smap));
  ssmap = smap;
  PetscCall(PetscMalloc1(1 + nrows, &lens));
  for (i = 0; i < ncols; i++) smap[icol[i]] = i + 1;
  /* determine lens of each row */
  for (i = 0; i < nrows; i++) {
    kstart  = ai[irow[i]];
    kend    = kstart + a->ilen[irow[i]];
    lens[i] = 0;
    for (k = kstart; k < kend; k++) {
      if (ssmap[aj[k]]) lens[i]++;
    }
  }
  /* Create and fill new matrix */
  if (scall == MAT_REUSE_MATRIX) {
    c = (Mat_SeqSBAIJ *)((*B)->data);

    PetscCheck(c->mbs == nrows && c->nbs == ncols && (*B)->rmap->bs == bs, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Submatrix wrong size");
    PetscCall(PetscArraycmp(c->ilen, lens, c->mbs, &flag));
    PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot reuse matrix. wrong no of nonzeros");
    PetscCall(PetscArrayzero(c->ilen, c->mbs));
    C = *B;
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &C));
    PetscCall(MatSetSizes(C, nrows * bs, ncols * bs, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(C, ((PetscObject)A)->type_name));
    PetscCall(MatSeqSBAIJSetPreallocation(C, bs, 0, lens));
  }
  c = (Mat_SeqSBAIJ *)(C->data);
  for (i = 0; i < nrows; i++) {
    row      = irow[i];
    kstart   = ai[row];
    kend     = kstart + a->ilen[row];
    mat_i    = c->i[i];
    mat_j    = c->j + mat_i;
    mat_a    = c->a + mat_i * bs2;
    mat_ilen = c->ilen + i;
    for (k = kstart; k < kend; k++) {
      if ((tcol = ssmap[a->j[k]])) {
        *mat_j++ = tcol - 1;
        PetscCall(PetscArraycpy(mat_a, a->a + k * bs2, bs2));
        mat_a += bs2;
        (*mat_ilen)++;
      }
    }
  }
  /* sort */
  {
    MatScalar *work;

    PetscCall(PetscMalloc1(bs2, &work));
    for (i = 0; i < nrows; i++) {
      PetscInt ilen;
      mat_i = c->i[i];
      mat_j = c->j + mat_i;
      mat_a = c->a + mat_i * bs2;
      ilen  = c->ilen[i];
      PetscCall(PetscSortIntWithDataArray(ilen, mat_j, mat_a, bs2 * sizeof(MatScalar), work));
    }
    PetscCall(PetscFree(work));
  }

  /* Free work space */
  PetscCall(ISRestoreIndices(iscol, &icol));
  PetscCall(PetscFree(smap));
  PetscCall(PetscFree(lens));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(ISRestoreIndices(isrow, &irow));
  *B = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateSubMatrix_SeqSBAIJ(Mat A, IS isrow, IS iscol, MatReuse scall, Mat *B)
{
  IS is1, is2;

  PetscFunctionBegin;
  PetscCall(ISCompressIndicesGeneral(A->rmap->N, A->rmap->n, A->rmap->bs, 1, &isrow, &is1));
  if (isrow == iscol) {
    is2 = is1;
    PetscCall(PetscObjectReference((PetscObject)is2));
  } else PetscCall(ISCompressIndicesGeneral(A->cmap->N, A->cmap->n, A->cmap->bs, 1, &iscol, &is2));
  PetscCall(MatCreateSubMatrix_SeqSBAIJ_Private(A, is1, is2, scall, B));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  if (isrow != iscol) {
    PetscBool isequal;
    PetscCall(ISEqual(isrow, iscol, &isequal));
    if (!isequal) PetscCall(MatSeqSBAIJZeroOps_Private(*B));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateSubMatrices_SeqSBAIJ(Mat A, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *B[])
{
  PetscInt i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) PetscCall(PetscCalloc1(n + 1, B));

  for (i = 0; i < n; i++) PetscCall(MatCreateSubMatrix_SeqSBAIJ(A, irow[i], icol[i], scall, &(*B)[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Should check that shapes of vectors and matrices match */
PetscErrorCode MatMult_SeqSBAIJ_2(Mat A, Vec xx, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, zero = 0.0;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecSet(zz, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (*ib == i) { /* (diag of A)*x */
      z[2 * i] += v[0] * x1 + v[2] * x2;
      z[2 * i + 1] += v[2] * x1 + v[3] * x2;
      v += 4;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 4 * n, 4 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 2;
      z[cval] += v[0] * x1 + v[1] * x2;
      z[cval + 1] += v[2] * x1 + v[3] * x2;
      /* (strict upper triangular part of A)*x  */
      z[2 * i] += v[0] * x[cval] + v[2] * x[cval + 1];
      z[2 * i + 1] += v[1] * x[cval] + v[3] * x[cval + 1];
      v += 4;
    }
    xb += 2;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(8.0 * (a->nz * 2.0 - nonzerorow) - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_SeqSBAIJ_3(Mat A, Vec xx, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, zero = 0.0;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecSet(zz, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (*ib == i) { /* (diag of A)*x */
      z[3 * i] += v[0] * x1 + v[3] * x2 + v[6] * x3;
      z[3 * i + 1] += v[3] * x1 + v[4] * x2 + v[7] * x3;
      z[3 * i + 2] += v[6] * x1 + v[7] * x2 + v[8] * x3;
      v += 9;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 9 * n, 9 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 3;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3;
      z[cval + 1] += v[3] * x1 + v[4] * x2 + v[5] * x3;
      z[cval + 2] += v[6] * x1 + v[7] * x2 + v[8] * x3;
      /* (strict upper triangular part of A)*x  */
      z[3 * i] += v[0] * x[cval] + v[3] * x[cval + 1] + v[6] * x[cval + 2];
      z[3 * i + 1] += v[1] * x[cval] + v[4] * x[cval + 1] + v[7] * x[cval + 2];
      z[3 * i + 2] += v[2] * x[cval] + v[5] * x[cval + 1] + v[8] * x[cval + 2];
      v += 9;
    }
    xb += 3;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(18.0 * (a->nz * 2.0 - nonzerorow) - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_SeqSBAIJ_4(Mat A, Vec xx, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4, zero = 0.0;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecSet(zz, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (*ib == i) { /* (diag of A)*x */
      z[4 * i] += v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      z[4 * i + 1] += v[4] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      z[4 * i + 2] += v[8] * x1 + v[9] * x2 + v[10] * x3 + v[14] * x4;
      z[4 * i + 3] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4;
      v += 16;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 16 * n, 16 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 4;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4;
      z[cval + 1] += v[4] * x1 + v[5] * x2 + v[6] * x3 + v[7] * x4;
      z[cval + 2] += v[8] * x1 + v[9] * x2 + v[10] * x3 + v[11] * x4;
      z[cval + 3] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4;
      /* (strict upper triangular part of A)*x  */
      z[4 * i] += v[0] * x[cval] + v[4] * x[cval + 1] + v[8] * x[cval + 2] + v[12] * x[cval + 3];
      z[4 * i + 1] += v[1] * x[cval] + v[5] * x[cval + 1] + v[9] * x[cval + 2] + v[13] * x[cval + 3];
      z[4 * i + 2] += v[2] * x[cval] + v[6] * x[cval + 1] + v[10] * x[cval + 2] + v[14] * x[cval + 3];
      z[4 * i + 3] += v[3] * x[cval] + v[7] * x[cval + 1] + v[11] * x[cval + 2] + v[15] * x[cval + 3];
      v += 16;
    }
    xb += 4;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(32.0 * (a->nz * 2.0 - nonzerorow) - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_SeqSBAIJ_5(Mat A, Vec xx, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4, x5, zero = 0.0;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecSet(zz, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    x5   = xb[4];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (*ib == i) { /* (diag of A)*x */
      z[5 * i] += v[0] * x1 + v[5] * x2 + v[10] * x3 + v[15] * x4 + v[20] * x5;
      z[5 * i + 1] += v[5] * x1 + v[6] * x2 + v[11] * x3 + v[16] * x4 + v[21] * x5;
      z[5 * i + 2] += v[10] * x1 + v[11] * x2 + v[12] * x3 + v[17] * x4 + v[22] * x5;
      z[5 * i + 3] += v[15] * x1 + v[16] * x2 + v[17] * x3 + v[18] * x4 + v[23] * x5;
      z[5 * i + 4] += v[20] * x1 + v[21] * x2 + v[22] * x3 + v[23] * x4 + v[24] * x5;
      v += 25;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 25 * n, 25 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 5;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5;
      z[cval + 1] += v[5] * x1 + v[6] * x2 + v[7] * x3 + v[8] * x4 + v[9] * x5;
      z[cval + 2] += v[10] * x1 + v[11] * x2 + v[12] * x3 + v[13] * x4 + v[14] * x5;
      z[cval + 3] += v[15] * x1 + v[16] * x2 + v[17] * x3 + v[18] * x4 + v[19] * x5;
      z[cval + 4] += v[20] * x1 + v[21] * x2 + v[22] * x3 + v[23] * x4 + v[24] * x5;
      /* (strict upper triangular part of A)*x  */
      z[5 * i] += v[0] * x[cval] + v[5] * x[cval + 1] + v[10] * x[cval + 2] + v[15] * x[cval + 3] + v[20] * x[cval + 4];
      z[5 * i + 1] += v[1] * x[cval] + v[6] * x[cval + 1] + v[11] * x[cval + 2] + v[16] * x[cval + 3] + v[21] * x[cval + 4];
      z[5 * i + 2] += v[2] * x[cval] + v[7] * x[cval + 1] + v[12] * x[cval + 2] + v[17] * x[cval + 3] + v[22] * x[cval + 4];
      z[5 * i + 3] += v[3] * x[cval] + v[8] * x[cval + 1] + v[13] * x[cval + 2] + v[18] * x[cval + 3] + v[23] * x[cval + 4];
      z[5 * i + 4] += v[4] * x[cval] + v[9] * x[cval + 1] + v[14] * x[cval + 2] + v[19] * x[cval + 3] + v[24] * x[cval + 4];
      v += 25;
    }
    xb += 5;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(50.0 * (a->nz * 2.0 - nonzerorow) - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_SeqSBAIJ_6(Mat A, Vec xx, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4, x5, x6, zero = 0.0;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecSet(zz, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    x5   = xb[4];
    x6   = xb[5];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (*ib == i) { /* (diag of A)*x */
      z[6 * i] += v[0] * x1 + v[6] * x2 + v[12] * x3 + v[18] * x4 + v[24] * x5 + v[30] * x6;
      z[6 * i + 1] += v[6] * x1 + v[7] * x2 + v[13] * x3 + v[19] * x4 + v[25] * x5 + v[31] * x6;
      z[6 * i + 2] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[20] * x4 + v[26] * x5 + v[32] * x6;
      z[6 * i + 3] += v[18] * x1 + v[19] * x2 + v[20] * x3 + v[21] * x4 + v[27] * x5 + v[33] * x6;
      z[6 * i + 4] += v[24] * x1 + v[25] * x2 + v[26] * x3 + v[27] * x4 + v[28] * x5 + v[34] * x6;
      z[6 * i + 5] += v[30] * x1 + v[31] * x2 + v[32] * x3 + v[33] * x4 + v[34] * x5 + v[35] * x6;
      v += 36;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 36 * n, 36 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 6;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5 + v[5] * x6;
      z[cval + 1] += v[6] * x1 + v[7] * x2 + v[8] * x3 + v[9] * x4 + v[10] * x5 + v[11] * x6;
      z[cval + 2] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4 + v[16] * x5 + v[17] * x6;
      z[cval + 3] += v[18] * x1 + v[19] * x2 + v[20] * x3 + v[21] * x4 + v[22] * x5 + v[23] * x6;
      z[cval + 4] += v[24] * x1 + v[25] * x2 + v[26] * x3 + v[27] * x4 + v[28] * x5 + v[29] * x6;
      z[cval + 5] += v[30] * x1 + v[31] * x2 + v[32] * x3 + v[33] * x4 + v[34] * x5 + v[35] * x6;
      /* (strict upper triangular part of A)*x  */
      z[6 * i] += v[0] * x[cval] + v[6] * x[cval + 1] + v[12] * x[cval + 2] + v[18] * x[cval + 3] + v[24] * x[cval + 4] + v[30] * x[cval + 5];
      z[6 * i + 1] += v[1] * x[cval] + v[7] * x[cval + 1] + v[13] * x[cval + 2] + v[19] * x[cval + 3] + v[25] * x[cval + 4] + v[31] * x[cval + 5];
      z[6 * i + 2] += v[2] * x[cval] + v[8] * x[cval + 1] + v[14] * x[cval + 2] + v[20] * x[cval + 3] + v[26] * x[cval + 4] + v[32] * x[cval + 5];
      z[6 * i + 3] += v[3] * x[cval] + v[9] * x[cval + 1] + v[15] * x[cval + 2] + v[21] * x[cval + 3] + v[27] * x[cval + 4] + v[33] * x[cval + 5];
      z[6 * i + 4] += v[4] * x[cval] + v[10] * x[cval + 1] + v[16] * x[cval + 2] + v[22] * x[cval + 3] + v[28] * x[cval + 4] + v[34] * x[cval + 5];
      z[6 * i + 5] += v[5] * x[cval] + v[11] * x[cval + 1] + v[17] * x[cval + 2] + v[23] * x[cval + 3] + v[29] * x[cval + 4] + v[35] * x[cval + 5];
      v += 36;
    }
    xb += 6;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(72.0 * (a->nz * 2.0 - nonzerorow) - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_SeqSBAIJ_7(Mat A, Vec xx, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4, x5, x6, x7, zero = 0.0;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecSet(zz, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    x5   = xb[4];
    x6   = xb[5];
    x7   = xb[6];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (*ib == i) { /* (diag of A)*x */
      z[7 * i] += v[0] * x1 + v[7] * x2 + v[14] * x3 + v[21] * x4 + v[28] * x5 + v[35] * x6 + v[42] * x7;
      z[7 * i + 1] += v[7] * x1 + v[8] * x2 + v[15] * x3 + v[22] * x4 + v[29] * x5 + v[36] * x6 + v[43] * x7;
      z[7 * i + 2] += v[14] * x1 + v[15] * x2 + v[16] * x3 + v[23] * x4 + v[30] * x5 + v[37] * x6 + v[44] * x7;
      z[7 * i + 3] += v[21] * x1 + v[22] * x2 + v[23] * x3 + v[24] * x4 + v[31] * x5 + v[38] * x6 + v[45] * x7;
      z[7 * i + 4] += v[28] * x1 + v[29] * x2 + v[30] * x3 + v[31] * x4 + v[32] * x5 + v[39] * x6 + v[46] * x7;
      z[7 * i + 5] += v[35] * x1 + v[36] * x2 + v[37] * x3 + v[38] * x4 + v[39] * x5 + v[40] * x6 + v[47] * x7;
      z[7 * i + 6] += v[42] * x1 + v[43] * x2 + v[44] * x3 + v[45] * x4 + v[46] * x5 + v[47] * x6 + v[48] * x7;
      v += 49;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 49 * n, 49 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 7;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5 + v[5] * x6 + v[6] * x7;
      z[cval + 1] += v[7] * x1 + v[8] * x2 + v[9] * x3 + v[10] * x4 + v[11] * x5 + v[12] * x6 + v[13] * x7;
      z[cval + 2] += v[14] * x1 + v[15] * x2 + v[16] * x3 + v[17] * x4 + v[18] * x5 + v[19] * x6 + v[20] * x7;
      z[cval + 3] += v[21] * x1 + v[22] * x2 + v[23] * x3 + v[24] * x4 + v[25] * x5 + v[26] * x6 + v[27] * x7;
      z[cval + 4] += v[28] * x1 + v[29] * x2 + v[30] * x3 + v[31] * x4 + v[32] * x5 + v[33] * x6 + v[34] * x7;
      z[cval + 5] += v[35] * x1 + v[36] * x2 + v[37] * x3 + v[38] * x4 + v[39] * x5 + v[40] * x6 + v[41] * x7;
      z[cval + 6] += v[42] * x1 + v[43] * x2 + v[44] * x3 + v[45] * x4 + v[46] * x5 + v[47] * x6 + v[48] * x7;
      /* (strict upper triangular part of A)*x  */
      z[7 * i] += v[0] * x[cval] + v[7] * x[cval + 1] + v[14] * x[cval + 2] + v[21] * x[cval + 3] + v[28] * x[cval + 4] + v[35] * x[cval + 5] + v[42] * x[cval + 6];
      z[7 * i + 1] += v[1] * x[cval] + v[8] * x[cval + 1] + v[15] * x[cval + 2] + v[22] * x[cval + 3] + v[29] * x[cval + 4] + v[36] * x[cval + 5] + v[43] * x[cval + 6];
      z[7 * i + 2] += v[2] * x[cval] + v[9] * x[cval + 1] + v[16] * x[cval + 2] + v[23] * x[cval + 3] + v[30] * x[cval + 4] + v[37] * x[cval + 5] + v[44] * x[cval + 6];
      z[7 * i + 3] += v[3] * x[cval] + v[10] * x[cval + 1] + v[17] * x[cval + 2] + v[24] * x[cval + 3] + v[31] * x[cval + 4] + v[38] * x[cval + 5] + v[45] * x[cval + 6];
      z[7 * i + 4] += v[4] * x[cval] + v[11] * x[cval + 1] + v[18] * x[cval + 2] + v[25] * x[cval + 3] + v[32] * x[cval + 4] + v[39] * x[cval + 5] + v[46] * x[cval + 6];
      z[7 * i + 5] += v[5] * x[cval] + v[12] * x[cval + 1] + v[19] * x[cval + 2] + v[26] * x[cval + 3] + v[33] * x[cval + 4] + v[40] * x[cval + 5] + v[47] * x[cval + 6];
      z[7 * i + 6] += v[6] * x[cval] + v[13] * x[cval + 1] + v[20] * x[cval + 2] + v[27] * x[cval + 3] + v[34] * x[cval + 4] + v[41] * x[cval + 5] + v[48] * x[cval + 6];
      v += 49;
    }
    xb += 7;
    ai++;
  }
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(98.0 * (a->nz * 2.0 - nonzerorow) - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    This will not work with MatScalar == float because it calls the BLAS
*/
PetscErrorCode MatMult_SeqSBAIJ_N(Mat A, Vec xx, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, *z_ptr, *zb, *work, *workt, zero = 0.0;
  const PetscScalar *x, *x_ptr, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, bs = A->rmap->bs, j, n, bs2 = a->bs2, ncols, k;
  const PetscInt    *idx, *aj, *ii;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecSet(zz, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  x_ptr = x;
  z_ptr = z;

  aj = a->j;
  v  = a->a;
  ii = a->i;

  if (!a->mult_work) PetscCall(PetscMalloc1(A->rmap->N + 1, &a->mult_work));
  work = a->mult_work;

  for (i = 0; i < mbs; i++) {
    n     = ii[1] - ii[0];
    ncols = n * bs;
    workt = work;
    idx   = aj + ii[0];
    nonzerorow += (n > 0);

    /* upper triangular part */
    for (j = 0; j < n; j++) {
      xb = x_ptr + bs * (*idx++);
      for (k = 0; k < bs; k++) workt[k] = xb[k];
      workt += bs;
    }
    /* z(i*bs:(i+1)*bs-1) += A(i,:)*x */
    PetscKernel_w_gets_w_plus_Ar_times_v(bs, ncols, work, v, z);

    /* strict lower triangular part */
    idx = aj + ii[0];
    if (n && *idx == i) {
      ncols -= bs;
      v += bs2;
      idx++;
      n--;
    }

    if (ncols > 0) {
      workt = work;
      PetscCall(PetscArrayzero(workt, ncols));
      PetscKernel_w_gets_w_plus_trans_Ar_times_v(bs, ncols, x, v, workt);
      for (j = 0; j < n; j++) {
        zb = z_ptr + bs * (*idx++);
        for (k = 0; k < bs; k++) zb[k] += workt[k];
        workt += bs;
      }
    }
    x += bs;
    v += n * bs2;
    z += bs;
    ii++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(2.0 * (a->nz * 2.0 - nonzerorow) * bs2 - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_1(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;
#if defined(PETSC_USE_COMPLEX)
  const int aconj = A->hermitian == PETSC_BOOL3_TRUE;
#else
  const int aconj = 0;
#endif

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));
  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th row of A */
    x1   = xb[0];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (n && *ib == i) { /* (diag of A)*x */
      z[i] += *v++ * x[*ib++];
      jmin++;
    }
    if (aconj) {
      for (j = jmin; j < n; j++) {
        cval = *ib;
        z[cval] += PetscConj(*v) * x1; /* (strict lower triangular part of A)*x  */
        z[i] += *v++ * x[*ib++];       /* (strict upper triangular part of A)*x  */
      }
    } else {
      for (j = jmin; j < n; j++) {
        cval = *ib;
        z[cval] += *v * x1;      /* (strict lower triangular part of A)*x  */
        z[i] += *v++ * x[*ib++]; /* (strict upper triangular part of A)*x  */
      }
    }
    xb++;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(2.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_2(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (n && *ib == i) { /* (diag of A)*x */
      z[2 * i] += v[0] * x1 + v[2] * x2;
      z[2 * i + 1] += v[2] * x1 + v[3] * x2;
      v += 4;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 4 * n, 4 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 2;
      z[cval] += v[0] * x1 + v[1] * x2;
      z[cval + 1] += v[2] * x1 + v[3] * x2;
      /* (strict upper triangular part of A)*x  */
      z[2 * i] += v[0] * x[cval] + v[2] * x[cval + 1];
      z[2 * i + 1] += v[1] * x[cval] + v[3] * x[cval + 1];
      v += 4;
    }
    xb += 2;
    ai++;
  }
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(4.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_3(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (n && *ib == i) { /* (diag of A)*x */
      z[3 * i] += v[0] * x1 + v[3] * x2 + v[6] * x3;
      z[3 * i + 1] += v[3] * x1 + v[4] * x2 + v[7] * x3;
      z[3 * i + 2] += v[6] * x1 + v[7] * x2 + v[8] * x3;
      v += 9;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 9 * n, 9 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 3;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3;
      z[cval + 1] += v[3] * x1 + v[4] * x2 + v[5] * x3;
      z[cval + 2] += v[6] * x1 + v[7] * x2 + v[8] * x3;
      /* (strict upper triangular part of A)*x  */
      z[3 * i] += v[0] * x[cval] + v[3] * x[cval + 1] + v[6] * x[cval + 2];
      z[3 * i + 1] += v[1] * x[cval] + v[4] * x[cval + 1] + v[7] * x[cval + 2];
      z[3 * i + 2] += v[2] * x[cval] + v[5] * x[cval + 1] + v[8] * x[cval + 2];
      v += 9;
    }
    xb += 3;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(18.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_4(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (n && *ib == i) { /* (diag of A)*x */
      z[4 * i] += v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      z[4 * i + 1] += v[4] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      z[4 * i + 2] += v[8] * x1 + v[9] * x2 + v[10] * x3 + v[14] * x4;
      z[4 * i + 3] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4;
      v += 16;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 16 * n, 16 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 4;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4;
      z[cval + 1] += v[4] * x1 + v[5] * x2 + v[6] * x3 + v[7] * x4;
      z[cval + 2] += v[8] * x1 + v[9] * x2 + v[10] * x3 + v[11] * x4;
      z[cval + 3] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4;
      /* (strict upper triangular part of A)*x  */
      z[4 * i] += v[0] * x[cval] + v[4] * x[cval + 1] + v[8] * x[cval + 2] + v[12] * x[cval + 3];
      z[4 * i + 1] += v[1] * x[cval] + v[5] * x[cval + 1] + v[9] * x[cval + 2] + v[13] * x[cval + 3];
      z[4 * i + 2] += v[2] * x[cval] + v[6] * x[cval + 1] + v[10] * x[cval + 2] + v[14] * x[cval + 3];
      z[4 * i + 3] += v[3] * x[cval] + v[7] * x[cval + 1] + v[11] * x[cval + 2] + v[15] * x[cval + 3];
      v += 16;
    }
    xb += 4;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(32.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_5(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4, x5;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    x5   = xb[4];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (n && *ib == i) { /* (diag of A)*x */
      z[5 * i] += v[0] * x1 + v[5] * x2 + v[10] * x3 + v[15] * x4 + v[20] * x5;
      z[5 * i + 1] += v[5] * x1 + v[6] * x2 + v[11] * x3 + v[16] * x4 + v[21] * x5;
      z[5 * i + 2] += v[10] * x1 + v[11] * x2 + v[12] * x3 + v[17] * x4 + v[22] * x5;
      z[5 * i + 3] += v[15] * x1 + v[16] * x2 + v[17] * x3 + v[18] * x4 + v[23] * x5;
      z[5 * i + 4] += v[20] * x1 + v[21] * x2 + v[22] * x3 + v[23] * x4 + v[24] * x5;
      v += 25;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 25 * n, 25 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 5;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5;
      z[cval + 1] += v[5] * x1 + v[6] * x2 + v[7] * x3 + v[8] * x4 + v[9] * x5;
      z[cval + 2] += v[10] * x1 + v[11] * x2 + v[12] * x3 + v[13] * x4 + v[14] * x5;
      z[cval + 3] += v[15] * x1 + v[16] * x2 + v[17] * x3 + v[18] * x4 + v[19] * x5;
      z[cval + 4] += v[20] * x1 + v[21] * x2 + v[22] * x3 + v[23] * x4 + v[24] * x5;
      /* (strict upper triangular part of A)*x  */
      z[5 * i] += v[0] * x[cval] + v[5] * x[cval + 1] + v[10] * x[cval + 2] + v[15] * x[cval + 3] + v[20] * x[cval + 4];
      z[5 * i + 1] += v[1] * x[cval] + v[6] * x[cval + 1] + v[11] * x[cval + 2] + v[16] * x[cval + 3] + v[21] * x[cval + 4];
      z[5 * i + 2] += v[2] * x[cval] + v[7] * x[cval + 1] + v[12] * x[cval + 2] + v[17] * x[cval + 3] + v[22] * x[cval + 4];
      z[5 * i + 3] += v[3] * x[cval] + v[8] * x[cval + 1] + v[13] * x[cval + 2] + v[18] * x[cval + 3] + v[23] * x[cval + 4];
      z[5 * i + 4] += v[4] * x[cval] + v[9] * x[cval + 1] + v[14] * x[cval + 2] + v[19] * x[cval + 3] + v[24] * x[cval + 4];
      v += 25;
    }
    xb += 5;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(50.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_6(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4, x5, x6;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    x5   = xb[4];
    x6   = xb[5];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (n && *ib == i) { /* (diag of A)*x */
      z[6 * i] += v[0] * x1 + v[6] * x2 + v[12] * x3 + v[18] * x4 + v[24] * x5 + v[30] * x6;
      z[6 * i + 1] += v[6] * x1 + v[7] * x2 + v[13] * x3 + v[19] * x4 + v[25] * x5 + v[31] * x6;
      z[6 * i + 2] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[20] * x4 + v[26] * x5 + v[32] * x6;
      z[6 * i + 3] += v[18] * x1 + v[19] * x2 + v[20] * x3 + v[21] * x4 + v[27] * x5 + v[33] * x6;
      z[6 * i + 4] += v[24] * x1 + v[25] * x2 + v[26] * x3 + v[27] * x4 + v[28] * x5 + v[34] * x6;
      z[6 * i + 5] += v[30] * x1 + v[31] * x2 + v[32] * x3 + v[33] * x4 + v[34] * x5 + v[35] * x6;
      v += 36;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 36 * n, 36 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 6;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5 + v[5] * x6;
      z[cval + 1] += v[6] * x1 + v[7] * x2 + v[8] * x3 + v[9] * x4 + v[10] * x5 + v[11] * x6;
      z[cval + 2] += v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4 + v[16] * x5 + v[17] * x6;
      z[cval + 3] += v[18] * x1 + v[19] * x2 + v[20] * x3 + v[21] * x4 + v[22] * x5 + v[23] * x6;
      z[cval + 4] += v[24] * x1 + v[25] * x2 + v[26] * x3 + v[27] * x4 + v[28] * x5 + v[29] * x6;
      z[cval + 5] += v[30] * x1 + v[31] * x2 + v[32] * x3 + v[33] * x4 + v[34] * x5 + v[35] * x6;
      /* (strict upper triangular part of A)*x  */
      z[6 * i] += v[0] * x[cval] + v[6] * x[cval + 1] + v[12] * x[cval + 2] + v[18] * x[cval + 3] + v[24] * x[cval + 4] + v[30] * x[cval + 5];
      z[6 * i + 1] += v[1] * x[cval] + v[7] * x[cval + 1] + v[13] * x[cval + 2] + v[19] * x[cval + 3] + v[25] * x[cval + 4] + v[31] * x[cval + 5];
      z[6 * i + 2] += v[2] * x[cval] + v[8] * x[cval + 1] + v[14] * x[cval + 2] + v[20] * x[cval + 3] + v[26] * x[cval + 4] + v[32] * x[cval + 5];
      z[6 * i + 3] += v[3] * x[cval] + v[9] * x[cval + 1] + v[15] * x[cval + 2] + v[21] * x[cval + 3] + v[27] * x[cval + 4] + v[33] * x[cval + 5];
      z[6 * i + 4] += v[4] * x[cval] + v[10] * x[cval + 1] + v[16] * x[cval + 2] + v[22] * x[cval + 3] + v[28] * x[cval + 4] + v[34] * x[cval + 5];
      z[6 * i + 5] += v[5] * x[cval] + v[11] * x[cval + 1] + v[17] * x[cval + 2] + v[23] * x[cval + 3] + v[29] * x[cval + 4] + v[35] * x[cval + 5];
      v += 36;
    }
    xb += 6;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(72.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_7(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, x1, x2, x3, x4, x5, x6, x7;
  const PetscScalar *x, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, n, cval, j, jmin;
  const PetscInt    *aj = a->j, *ai = a->i, *ib;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v  = a->a;
  xb = x;

  for (i = 0; i < mbs; i++) {
    n    = ai[1] - ai[0]; /* length of i_th block row of A */
    x1   = xb[0];
    x2   = xb[1];
    x3   = xb[2];
    x4   = xb[3];
    x5   = xb[4];
    x6   = xb[5];
    x7   = xb[6];
    ib   = aj + *ai;
    jmin = 0;
    nonzerorow += (n > 0);
    if (n && *ib == i) { /* (diag of A)*x */
      z[7 * i] += v[0] * x1 + v[7] * x2 + v[14] * x3 + v[21] * x4 + v[28] * x5 + v[35] * x6 + v[42] * x7;
      z[7 * i + 1] += v[7] * x1 + v[8] * x2 + v[15] * x3 + v[22] * x4 + v[29] * x5 + v[36] * x6 + v[43] * x7;
      z[7 * i + 2] += v[14] * x1 + v[15] * x2 + v[16] * x3 + v[23] * x4 + v[30] * x5 + v[37] * x6 + v[44] * x7;
      z[7 * i + 3] += v[21] * x1 + v[22] * x2 + v[23] * x3 + v[24] * x4 + v[31] * x5 + v[38] * x6 + v[45] * x7;
      z[7 * i + 4] += v[28] * x1 + v[29] * x2 + v[30] * x3 + v[31] * x4 + v[32] * x5 + v[39] * x6 + v[46] * x7;
      z[7 * i + 5] += v[35] * x1 + v[36] * x2 + v[37] * x3 + v[38] * x4 + v[39] * x5 + v[40] * x6 + v[47] * x7;
      z[7 * i + 6] += v[42] * x1 + v[43] * x2 + v[44] * x3 + v[45] * x4 + v[46] * x5 + v[47] * x6 + v[48] * x7;
      v += 49;
      jmin++;
    }
    PetscPrefetchBlock(ib + jmin + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 49 * n, 49 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    for (j = jmin; j < n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval = ib[j] * 7;
      z[cval] += v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5 + v[5] * x6 + v[6] * x7;
      z[cval + 1] += v[7] * x1 + v[8] * x2 + v[9] * x3 + v[10] * x4 + v[11] * x5 + v[12] * x6 + v[13] * x7;
      z[cval + 2] += v[14] * x1 + v[15] * x2 + v[16] * x3 + v[17] * x4 + v[18] * x5 + v[19] * x6 + v[20] * x7;
      z[cval + 3] += v[21] * x1 + v[22] * x2 + v[23] * x3 + v[24] * x4 + v[25] * x5 + v[26] * x6 + v[27] * x7;
      z[cval + 4] += v[28] * x1 + v[29] * x2 + v[30] * x3 + v[31] * x4 + v[32] * x5 + v[33] * x6 + v[34] * x7;
      z[cval + 5] += v[35] * x1 + v[36] * x2 + v[37] * x3 + v[38] * x4 + v[39] * x5 + v[40] * x6 + v[41] * x7;
      z[cval + 6] += v[42] * x1 + v[43] * x2 + v[44] * x3 + v[45] * x4 + v[46] * x5 + v[47] * x6 + v[48] * x7;
      /* (strict upper triangular part of A)*x  */
      z[7 * i] += v[0] * x[cval] + v[7] * x[cval + 1] + v[14] * x[cval + 2] + v[21] * x[cval + 3] + v[28] * x[cval + 4] + v[35] * x[cval + 5] + v[42] * x[cval + 6];
      z[7 * i + 1] += v[1] * x[cval] + v[8] * x[cval + 1] + v[15] * x[cval + 2] + v[22] * x[cval + 3] + v[29] * x[cval + 4] + v[36] * x[cval + 5] + v[43] * x[cval + 6];
      z[7 * i + 2] += v[2] * x[cval] + v[9] * x[cval + 1] + v[16] * x[cval + 2] + v[23] * x[cval + 3] + v[30] * x[cval + 4] + v[37] * x[cval + 5] + v[44] * x[cval + 6];
      z[7 * i + 3] += v[3] * x[cval] + v[10] * x[cval + 1] + v[17] * x[cval + 2] + v[24] * x[cval + 3] + v[31] * x[cval + 4] + v[38] * x[cval + 5] + v[45] * x[cval + 6];
      z[7 * i + 4] += v[4] * x[cval] + v[11] * x[cval + 1] + v[18] * x[cval + 2] + v[25] * x[cval + 3] + v[32] * x[cval + 4] + v[39] * x[cval + 5] + v[46] * x[cval + 6];
      z[7 * i + 5] += v[5] * x[cval] + v[12] * x[cval + 1] + v[19] * x[cval + 2] + v[26] * x[cval + 3] + v[33] * x[cval + 4] + v[40] * x[cval + 5] + v[47] * x[cval + 6];
      z[7 * i + 6] += v[6] * x[cval] + v[13] * x[cval + 1] + v[20] * x[cval + 2] + v[27] * x[cval + 3] + v[34] * x[cval + 4] + v[41] * x[cval + 5] + v[48] * x[cval + 6];
      v += 49;
    }
    xb += 7;
    ai++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(98.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSBAIJ_N(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z, *z_ptr = NULL, *zb, *work, *workt;
  const PetscScalar *x, *x_ptr, *xb;
  const MatScalar   *v;
  PetscInt           mbs = a->mbs, i, bs = A->rmap->bs, j, n, bs2 = a->bs2, ncols, k;
  const PetscInt    *idx, *aj, *ii;
  PetscInt           nonzerorow = 0;

  PetscFunctionBegin;
  PetscCall(VecCopy(yy, zz));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayRead(xx, &x));
  x_ptr = x;
  PetscCall(VecGetArray(zz, &z));
  z_ptr = z;

  aj = a->j;
  v  = a->a;
  ii = a->i;

  if (!a->mult_work) PetscCall(PetscMalloc1(A->rmap->n + 1, &a->mult_work));
  work = a->mult_work;

  for (i = 0; i < mbs; i++) {
    n     = ii[1] - ii[0];
    ncols = n * bs;
    workt = work;
    idx   = aj + ii[0];
    nonzerorow += (n > 0);

    /* upper triangular part */
    for (j = 0; j < n; j++) {
      xb = x_ptr + bs * (*idx++);
      for (k = 0; k < bs; k++) workt[k] = xb[k];
      workt += bs;
    }
    /* z(i*bs:(i+1)*bs-1) += A(i,:)*x */
    PetscKernel_w_gets_w_plus_Ar_times_v(bs, ncols, work, v, z);

    /* strict lower triangular part */
    idx = aj + ii[0];
    if (n && *idx == i) {
      ncols -= bs;
      v += bs2;
      idx++;
      n--;
    }
    if (ncols > 0) {
      workt = work;
      PetscCall(PetscArrayzero(workt, ncols));
      PetscKernel_w_gets_w_plus_trans_Ar_times_v(bs, ncols, x, v, workt);
      for (j = 0; j < n; j++) {
        zb = z_ptr + bs * (*idx++);
        for (k = 0; k < bs; k++) zb[k] += workt[k];
        workt += bs;
      }
    }

    x += bs;
    v += n * bs2;
    z += bs;
    ii++;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));

  PetscCall(PetscLogFlops(2.0 * (a->nz * 2.0 - nonzerorow)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatScale_SeqSBAIJ(Mat inA, PetscScalar alpha)
{
  Mat_SeqSBAIJ *a      = (Mat_SeqSBAIJ *)inA->data;
  PetscScalar   oalpha = alpha;
  PetscBLASInt  one    = 1, totalnz;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(a->bs2 * a->nz, &totalnz));
  PetscCallBLAS("BLASscal", BLASscal_(&totalnz, &oalpha, a->a, &one));
  PetscCall(PetscLogFlops(totalnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatNorm_SeqSBAIJ(Mat A, NormType type, PetscReal *norm)
{
  Mat_SeqSBAIJ    *a        = (Mat_SeqSBAIJ *)A->data;
  const MatScalar *v        = a->a;
  PetscReal        sum_diag = 0.0, sum_off = 0.0, *sum;
  PetscInt         i, j, k, bs = A->rmap->bs, bs2 = a->bs2, k1, mbs = a->mbs, jmin, jmax, nexti, ik, *jl, *il;
  const PetscInt  *aj = a->j, *col;

  PetscFunctionBegin;
  if (!a->nz) {
    *norm = 0.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (type == NORM_FROBENIUS) {
    for (k = 0; k < mbs; k++) {
      jmin = a->i[k];
      jmax = a->i[k + 1];
      col  = aj + jmin;
      if (jmax - jmin > 0 && *col == k) { /* diagonal block */
        for (i = 0; i < bs2; i++) {
          sum_diag += PetscRealPart(PetscConj(*v) * (*v));
          v++;
        }
        jmin++;
      }
      for (j = jmin; j < jmax; j++) { /* off-diagonal blocks */
        for (i = 0; i < bs2; i++) {
          sum_off += PetscRealPart(PetscConj(*v) * (*v));
          v++;
        }
      }
    }
    *norm = PetscSqrtReal(sum_diag + 2 * sum_off);
    PetscCall(PetscLogFlops(2.0 * bs2 * a->nz));
  } else if (type == NORM_INFINITY || type == NORM_1) { /* maximum row/column sum */
    PetscCall(PetscMalloc3(bs, &sum, mbs, &il, mbs, &jl));
    for (i = 0; i < mbs; i++) jl[i] = mbs;
    il[0] = 0;

    *norm = 0.0;
    for (k = 0; k < mbs; k++) { /* k_th block row */
      for (j = 0; j < bs; j++) sum[j] = 0.0;
      /*-- col sum --*/
      i = jl[k]; /* first |A(i,k)| to be added */
      /* jl[k]=i: first nozero element in row i for submatrix A(1:k,k:n) (active window)
                  at step k */
      while (i < mbs) {
        nexti = jl[i]; /* next block row to be added */
        ik    = il[i]; /* block index of A(i,k) in the array a */
        for (j = 0; j < bs; j++) {
          v = a->a + ik * bs2 + j * bs;
          for (k1 = 0; k1 < bs; k1++) {
            sum[j] += PetscAbsScalar(*v);
            v++;
          }
        }
        /* update il, jl */
        jmin = ik + 1; /* block index of array a: points to the next nonzero of A in row i */
        jmax = a->i[i + 1];
        if (jmin < jmax) {
          il[i] = jmin;
          j     = a->j[jmin];
          jl[i] = jl[j];
          jl[j] = i;
        }
        i = nexti;
      }
      /*-- row sum --*/
      jmin = a->i[k];
      jmax = a->i[k + 1];
      for (i = jmin; i < jmax; i++) {
        for (j = 0; j < bs; j++) {
          v = a->a + i * bs2 + j;
          for (k1 = 0; k1 < bs; k1++) {
            sum[j] += PetscAbsScalar(*v);
            v += bs;
          }
        }
      }
      /* add k_th block row to il, jl */
      col = aj + jmin;
      if (jmax - jmin > 0 && *col == k) jmin++;
      if (jmin < jmax) {
        il[k] = jmin;
        j     = a->j[jmin];
        jl[k] = jl[j];
        jl[j] = k;
      }
      for (j = 0; j < bs; j++) {
        if (sum[j] > *norm) *norm = sum[j];
      }
    }
    PetscCall(PetscFree3(sum, il, jl));
    PetscCall(PetscLogFlops(PetscMax(mbs * a->nz - 1, 0)));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for this norm yet");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatEqual_SeqSBAIJ(Mat A, Mat B, PetscBool *flg)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ *)A->data, *b = (Mat_SeqSBAIJ *)B->data;

  PetscFunctionBegin;
  /* If the  matrix/block dimensions are not equal, or no of nonzeros or shift */
  if ((A->rmap->N != B->rmap->N) || (A->cmap->n != B->cmap->n) || (A->rmap->bs != B->rmap->bs) || (a->nz != b->nz)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* if the a->i are the same */
  PetscCall(PetscArraycmp(a->i, b->i, a->mbs + 1, flg));
  if (!*flg) PetscFunctionReturn(PETSC_SUCCESS);

  /* if a->j are the same */
  PetscCall(PetscArraycmp(a->j, b->j, a->nz, flg));
  if (!*flg) PetscFunctionReturn(PETSC_SUCCESS);

  /* if a->a are the same */
  PetscCall(PetscArraycmp(a->a, b->a, (a->nz) * (A->rmap->bs) * (A->rmap->bs), flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetDiagonal_SeqSBAIJ(Mat A, Vec v)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ *)A->data;
  PetscInt         i, j, k, row, bs, ambs, bs2;
  const PetscInt  *ai, *aj;
  PetscScalar     *x, zero = 0.0;
  const MatScalar *aa, *aa_j;

  PetscFunctionBegin;
  bs = A->rmap->bs;
  PetscCheck(!A->factortype || bs <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix with bs>1");

  aa   = a->a;
  ambs = a->mbs;

  if (A->factortype == MAT_FACTOR_CHOLESKY || A->factortype == MAT_FACTOR_ICC) {
    PetscInt *diag = a->diag;
    aa             = a->a;
    ambs           = a->mbs;
    PetscCall(VecGetArray(v, &x));
    for (i = 0; i < ambs; i++) x[i] = 1.0 / aa[diag[i]];
    PetscCall(VecRestoreArray(v, &x));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ai  = a->i;
  aj  = a->j;
  bs2 = a->bs2;
  PetscCall(VecSet(v, zero));
  if (!a->nz) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArray(v, &x));
  for (i = 0; i < ambs; i++) {
    j = ai[i];
    if (aj[j] == i) { /* if this is a diagonal element */
      row  = i * bs;
      aa_j = aa + j * bs2;
      for (k = 0; k < bs2; k += (bs + 1), row++) x[row] = aa_j[k];
    }
  }
  PetscCall(VecRestoreArray(v, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDiagonalScale_SeqSBAIJ(Mat A, Vec ll, Vec rr)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar        x;
  const PetscScalar *l, *li, *ri;
  MatScalar         *aa, *v;
  PetscInt           i, j, k, lm, M, m, mbs, tmp, bs, bs2;
  const PetscInt    *ai, *aj;
  PetscBool          flg;

  PetscFunctionBegin;
  if (ll != rr) {
    PetscCall(VecEqual(ll, rr, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "For symmetric format, left and right scaling vectors must be same");
  }
  if (!ll) PetscFunctionReturn(PETSC_SUCCESS);
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  m   = A->rmap->N;
  bs  = A->rmap->bs;
  mbs = a->mbs;
  bs2 = a->bs2;

  PetscCall(VecGetArrayRead(ll, &l));
  PetscCall(VecGetLocalSize(ll, &lm));
  PetscCheck(lm == m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Left scaling vector wrong length");
  for (i = 0; i < mbs; i++) { /* for each block row */
    M  = ai[i + 1] - ai[i];
    li = l + i * bs;
    v  = aa + bs2 * ai[i];
    for (j = 0; j < M; j++) { /* for each block */
      ri = l + bs * aj[ai[i] + j];
      for (k = 0; k < bs; k++) {
        x = ri[k];
        for (tmp = 0; tmp < bs; tmp++) (*v++) *= li[tmp] * x;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(ll, &l));
  PetscCall(PetscLogFlops(2.0 * a->nz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetInfo_SeqSBAIJ(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ *)A->data;

  PetscFunctionBegin;
  info->block_size   = a->bs2;
  info->nz_allocated = a->bs2 * a->maxnz; /*num. of nonzeros in upper triangular part */
  info->nz_used      = a->bs2 * a->nz;    /*num. of nonzeros in upper triangular part */
  info->nz_unneeded  = info->nz_allocated - info->nz_used;
  info->assemblies   = A->num_ass;
  info->mallocs      = A->info.mallocs;
  info->memory       = 0; /* REVIEW ME */
  if (A->factortype) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatZeroEntries_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(PetscArrayzero(a->a, a->bs2 * a->i[a->mbs]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetRowMaxAbs_SeqSBAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ *)A->data;
  PetscInt         i, j, n, row, col, bs, mbs;
  const PetscInt  *ai, *aj;
  PetscReal        atmp;
  const MatScalar *aa;
  PetscScalar     *x;
  PetscInt         ncols, brow, bcol, krow, kcol;

  PetscFunctionBegin;
  PetscCheck(!idx, PETSC_COMM_SELF, PETSC_ERR_SUP, "Send email to petsc-maint@mcs.anl.gov");
  PetscCheck(!A->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  bs  = A->rmap->bs;
  aa  = a->a;
  ai  = a->i;
  aj  = a->j;
  mbs = a->mbs;

  PetscCall(VecSet(v, 0.0));
  PetscCall(VecGetArray(v, &x));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == A->rmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  for (i = 0; i < mbs; i++) {
    ncols = ai[1] - ai[0];
    ai++;
    brow = bs * i;
    for (j = 0; j < ncols; j++) {
      bcol = bs * (*aj);
      for (kcol = 0; kcol < bs; kcol++) {
        col = bcol + kcol; /* col index */
        for (krow = 0; krow < bs; krow++) {
          atmp = PetscAbsScalar(*aa);
          aa++;
          row = brow + krow; /* row index */
          if (PetscRealPart(x[row]) < atmp) x[row] = atmp;
          if (*aj > i && PetscRealPart(x[col]) < atmp) x[col] = atmp;
        }
      }
      aj++;
    }
  }
  PetscCall(VecRestoreArray(v, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultSymbolic_SeqSBAIJ_SeqDense(Mat A, Mat B, PetscReal fill, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultSymbolic_SeqDense_SeqDense(A, B, 0.0, C));
  C->ops->matmultnumeric = MatMatMultNumeric_SeqSBAIJ_SeqDense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMult_SeqSBAIJ_1_Private(Mat A, PetscScalar *b, PetscInt bm, PetscScalar *c, PetscInt cm, PetscInt cn)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z = c;
  const PetscScalar *xb;
  PetscScalar        x1;
  const MatScalar   *v   = a->a, *vv;
  PetscInt           mbs = a->mbs, i, *idx = a->j, *ii = a->i, j, *jj, n, k;
#if defined(PETSC_USE_COMPLEX)
  const int aconj = A->hermitian == PETSC_BOOL3_TRUE;
#else
  const int aconj = 0;
#endif

  PetscFunctionBegin;
  for (i = 0; i < mbs; i++) {
    n = ii[1] - ii[0];
    ii++;
    PetscPrefetchBlock(idx + n, n, 0, PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + n, n, 0, PETSC_PREFETCH_HINT_NTA);   /* Entries for the next row */
    jj = idx;
    vv = v;
    for (k = 0; k < cn; k++) {
      idx = jj;
      v   = vv;
      for (j = 0; j < n; j++) {
        xb = b + (*idx);
        x1 = xb[0 + k * bm];
        z[0 + k * cm] += v[0] * x1;
        if (*idx != i) c[(*idx) + k * cm] += (aconj ? PetscConj(v[0]) : v[0]) * b[i + k * bm];
        v += 1;
        ++idx;
      }
    }
    z += 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMult_SeqSBAIJ_2_Private(Mat A, PetscScalar *b, PetscInt bm, PetscScalar *c, PetscInt cm, PetscInt cn)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z = c;
  const PetscScalar *xb;
  PetscScalar        x1, x2;
  const MatScalar   *v   = a->a, *vv;
  PetscInt           mbs = a->mbs, i, *idx = a->j, *ii = a->i, j, *jj, n, k;

  PetscFunctionBegin;
  for (i = 0; i < mbs; i++) {
    n = ii[1] - ii[0];
    ii++;
    PetscPrefetchBlock(idx + n, n, 0, PETSC_PREFETCH_HINT_NTA);       /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 4 * n, 4 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    jj = idx;
    vv = v;
    for (k = 0; k < cn; k++) {
      idx = jj;
      v   = vv;
      for (j = 0; j < n; j++) {
        xb = b + 2 * (*idx);
        x1 = xb[0 + k * bm];
        x2 = xb[1 + k * bm];
        z[0 + k * cm] += v[0] * x1 + v[2] * x2;
        z[1 + k * cm] += v[1] * x1 + v[3] * x2;
        if (*idx != i) {
          c[2 * (*idx) + 0 + k * cm] += v[0] * b[2 * i + k * bm] + v[1] * b[2 * i + 1 + k * bm];
          c[2 * (*idx) + 1 + k * cm] += v[2] * b[2 * i + k * bm] + v[3] * b[2 * i + 1 + k * bm];
        }
        v += 4;
        ++idx;
      }
    }
    z += 2;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMult_SeqSBAIJ_3_Private(Mat A, PetscScalar *b, PetscInt bm, PetscScalar *c, PetscInt cm, PetscInt cn)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z = c;
  const PetscScalar *xb;
  PetscScalar        x1, x2, x3;
  const MatScalar   *v   = a->a, *vv;
  PetscInt           mbs = a->mbs, i, *idx = a->j, *ii = a->i, j, *jj, n, k;

  PetscFunctionBegin;
  for (i = 0; i < mbs; i++) {
    n = ii[1] - ii[0];
    ii++;
    PetscPrefetchBlock(idx + n, n, 0, PETSC_PREFETCH_HINT_NTA);       /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 9 * n, 9 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    jj = idx;
    vv = v;
    for (k = 0; k < cn; k++) {
      idx = jj;
      v   = vv;
      for (j = 0; j < n; j++) {
        xb = b + 3 * (*idx);
        x1 = xb[0 + k * bm];
        x2 = xb[1 + k * bm];
        x3 = xb[2 + k * bm];
        z[0 + k * cm] += v[0] * x1 + v[3] * x2 + v[6] * x3;
        z[1 + k * cm] += v[1] * x1 + v[4] * x2 + v[7] * x3;
        z[2 + k * cm] += v[2] * x1 + v[5] * x2 + v[8] * x3;
        if (*idx != i) {
          c[3 * (*idx) + 0 + k * cm] += v[0] * b[3 * i + k * bm] + v[3] * b[3 * i + 1 + k * bm] + v[6] * b[3 * i + 2 + k * bm];
          c[3 * (*idx) + 1 + k * cm] += v[1] * b[3 * i + k * bm] + v[4] * b[3 * i + 1 + k * bm] + v[7] * b[3 * i + 2 + k * bm];
          c[3 * (*idx) + 2 + k * cm] += v[2] * b[3 * i + k * bm] + v[5] * b[3 * i + 1 + k * bm] + v[8] * b[3 * i + 2 + k * bm];
        }
        v += 9;
        ++idx;
      }
    }
    z += 3;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMult_SeqSBAIJ_4_Private(Mat A, PetscScalar *b, PetscInt bm, PetscScalar *c, PetscInt cm, PetscInt cn)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z = c;
  const PetscScalar *xb;
  PetscScalar        x1, x2, x3, x4;
  const MatScalar   *v   = a->a, *vv;
  PetscInt           mbs = a->mbs, i, *idx = a->j, *ii = a->i, j, *jj, n, k;

  PetscFunctionBegin;
  for (i = 0; i < mbs; i++) {
    n = ii[1] - ii[0];
    ii++;
    PetscPrefetchBlock(idx + n, n, 0, PETSC_PREFETCH_HINT_NTA);         /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 16 * n, 16 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    jj = idx;
    vv = v;
    for (k = 0; k < cn; k++) {
      idx = jj;
      v   = vv;
      for (j = 0; j < n; j++) {
        xb = b + 4 * (*idx);
        x1 = xb[0 + k * bm];
        x2 = xb[1 + k * bm];
        x3 = xb[2 + k * bm];
        x4 = xb[3 + k * bm];
        z[0 + k * cm] += v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
        z[1 + k * cm] += v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
        z[2 + k * cm] += v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
        z[3 + k * cm] += v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
        if (*idx != i) {
          c[4 * (*idx) + 0 + k * cm] += v[0] * b[4 * i + k * bm] + v[4] * b[4 * i + 1 + k * bm] + v[8] * b[4 * i + 2 + k * bm] + v[12] * b[4 * i + 3 + k * bm];
          c[4 * (*idx) + 1 + k * cm] += v[1] * b[4 * i + k * bm] + v[5] * b[4 * i + 1 + k * bm] + v[9] * b[4 * i + 2 + k * bm] + v[13] * b[4 * i + 3 + k * bm];
          c[4 * (*idx) + 2 + k * cm] += v[2] * b[4 * i + k * bm] + v[6] * b[4 * i + 1 + k * bm] + v[10] * b[4 * i + 2 + k * bm] + v[14] * b[4 * i + 3 + k * bm];
          c[4 * (*idx) + 3 + k * cm] += v[3] * b[4 * i + k * bm] + v[7] * b[4 * i + 1 + k * bm] + v[11] * b[4 * i + 2 + k * bm] + v[15] * b[4 * i + 3 + k * bm];
        }
        v += 16;
        ++idx;
      }
    }
    z += 4;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMult_SeqSBAIJ_5_Private(Mat A, PetscScalar *b, PetscInt bm, PetscScalar *c, PetscInt cm, PetscInt cn)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  PetscScalar       *z = c;
  const PetscScalar *xb;
  PetscScalar        x1, x2, x3, x4, x5;
  const MatScalar   *v   = a->a, *vv;
  PetscInt           mbs = a->mbs, i, *idx = a->j, *ii = a->i, j, *jj, n, k;

  PetscFunctionBegin;
  for (i = 0; i < mbs; i++) {
    n = ii[1] - ii[0];
    ii++;
    PetscPrefetchBlock(idx + n, n, 0, PETSC_PREFETCH_HINT_NTA);         /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + 25 * n, 25 * n, 0, PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    jj = idx;
    vv = v;
    for (k = 0; k < cn; k++) {
      idx = jj;
      v   = vv;
      for (j = 0; j < n; j++) {
        xb = b + 5 * (*idx);
        x1 = xb[0 + k * bm];
        x2 = xb[1 + k * bm];
        x3 = xb[2 + k * bm];
        x4 = xb[3 + k * bm];
        x5 = xb[4 + k * cm];
        z[0 + k * cm] += v[0] * x1 + v[5] * x2 + v[10] * x3 + v[15] * x4 + v[20] * x5;
        z[1 + k * cm] += v[1] * x1 + v[6] * x2 + v[11] * x3 + v[16] * x4 + v[21] * x5;
        z[2 + k * cm] += v[2] * x1 + v[7] * x2 + v[12] * x3 + v[17] * x4 + v[22] * x5;
        z[3 + k * cm] += v[3] * x1 + v[8] * x2 + v[13] * x3 + v[18] * x4 + v[23] * x5;
        z[4 + k * cm] += v[4] * x1 + v[9] * x2 + v[14] * x3 + v[19] * x4 + v[24] * x5;
        if (*idx != i) {
          c[5 * (*idx) + 0 + k * cm] += v[0] * b[5 * i + k * bm] + v[5] * b[5 * i + 1 + k * bm] + v[10] * b[5 * i + 2 + k * bm] + v[15] * b[5 * i + 3 + k * bm] + v[20] * b[5 * i + 4 + k * bm];
          c[5 * (*idx) + 1 + k * cm] += v[1] * b[5 * i + k * bm] + v[6] * b[5 * i + 1 + k * bm] + v[11] * b[5 * i + 2 + k * bm] + v[16] * b[5 * i + 3 + k * bm] + v[21] * b[5 * i + 4 + k * bm];
          c[5 * (*idx) + 2 + k * cm] += v[2] * b[5 * i + k * bm] + v[7] * b[5 * i + 1 + k * bm] + v[12] * b[5 * i + 2 + k * bm] + v[17] * b[5 * i + 3 + k * bm] + v[22] * b[5 * i + 4 + k * bm];
          c[5 * (*idx) + 3 + k * cm] += v[3] * b[5 * i + k * bm] + v[8] * b[5 * i + 1 + k * bm] + v[13] * b[5 * i + 2 + k * bm] + v[18] * b[5 * i + 3 + k * bm] + v[23] * b[5 * i + 4 + k * bm];
          c[5 * (*idx) + 4 + k * cm] += v[4] * b[5 * i + k * bm] + v[9] * b[5 * i + 1 + k * bm] + v[14] * b[5 * i + 2 + k * bm] + v[19] * b[5 * i + 3 + k * bm] + v[24] * b[5 * i + 4 + k * bm];
        }
        v += 25;
        ++idx;
      }
    }
    z += 5;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultNumeric_SeqSBAIJ_SeqDense(Mat A, Mat B, Mat C)
{
  Mat_SeqSBAIJ    *a  = (Mat_SeqSBAIJ *)A->data;
  Mat_SeqDense    *bd = (Mat_SeqDense *)B->data;
  Mat_SeqDense    *cd = (Mat_SeqDense *)C->data;
  PetscInt         cm = cd->lda, cn = B->cmap->n, bm = bd->lda;
  PetscInt         mbs, i, bs = A->rmap->bs, j, n, bs2 = a->bs2;
  PetscBLASInt     bbs, bcn, bbm, bcm;
  PetscScalar     *z = NULL;
  PetscScalar     *c, *b;
  const MatScalar *v;
  const PetscInt  *idx, *ii;
  PetscScalar      _DOne = 1.0;

  PetscFunctionBegin;
  if (!cm || !cn) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(B->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number columns in A %" PetscInt_FMT " not equal rows in B %" PetscInt_FMT, A->cmap->n, B->rmap->n);
  PetscCheck(A->rmap->n == C->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number rows in C %" PetscInt_FMT " not equal rows in A %" PetscInt_FMT, C->rmap->n, A->rmap->n);
  PetscCheck(B->cmap->n == C->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number columns in B %" PetscInt_FMT " not equal columns in C %" PetscInt_FMT, B->cmap->n, C->cmap->n);
  b = bd->v;
  PetscCall(MatZeroEntries(C));
  PetscCall(MatDenseGetArray(C, &c));
  switch (bs) {
  case 1:
    PetscCall(MatMatMult_SeqSBAIJ_1_Private(A, b, bm, c, cm, cn));
    break;
  case 2:
    PetscCall(MatMatMult_SeqSBAIJ_2_Private(A, b, bm, c, cm, cn));
    break;
  case 3:
    PetscCall(MatMatMult_SeqSBAIJ_3_Private(A, b, bm, c, cm, cn));
    break;
  case 4:
    PetscCall(MatMatMult_SeqSBAIJ_4_Private(A, b, bm, c, cm, cn));
    break;
  case 5:
    PetscCall(MatMatMult_SeqSBAIJ_5_Private(A, b, bm, c, cm, cn));
    break;
  default: /* block sizes larger than 5 by 5 are handled by BLAS */
    PetscCall(PetscBLASIntCast(bs, &bbs));
    PetscCall(PetscBLASIntCast(cn, &bcn));
    PetscCall(PetscBLASIntCast(bm, &bbm));
    PetscCall(PetscBLASIntCast(cm, &bcm));
    idx = a->j;
    v   = a->a;
    mbs = a->mbs;
    ii  = a->i;
    z   = c;
    for (i = 0; i < mbs; i++) {
      n = ii[1] - ii[0];
      ii++;
      for (j = 0; j < n; j++) {
        if (*idx != i) PetscCallBLAS("BLASgemm", BLASgemm_("T", "N", &bbs, &bcn, &bbs, &_DOne, v, &bbs, b + bs * i, &bbm, &_DOne, c + bs * (*idx), &bcm));
        PetscCallBLAS("BLASgemm", BLASgemm_("N", "N", &bbs, &bcn, &bbs, &_DOne, v, &bbs, b + bs * (*idx++), &bbm, &_DOne, z, &bcm));
        v += bs2;
      }
      z += bs;
    }
  }
  PetscCall(MatDenseRestoreArray(C, &c));
  PetscCall(PetscLogFlops((2.0 * (a->nz * 2.0 - a->nonzerorowcnt) * bs2 - a->nonzerorowcnt) * cn));
  PetscFunctionReturn(PETSC_SUCCESS);
}
