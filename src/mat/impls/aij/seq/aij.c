/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <petscblaslapack.h>
#include <petscbt.h>
#include <petsc/private/kernels/blocktranspose.h>

/* defines MatSetValues_Seq_Hash(), MatAssemblyEnd_Seq_Hash(), MatSetUp_Seq_Hash() */
#define TYPE AIJ
#define TYPE_BS
#include "../src/mat/impls/aij/seq/seqhashmatsetvalues.h"
#include "../src/mat/impls/aij/seq/seqhashmat.h"
#undef TYPE
#undef TYPE_BS

static PetscErrorCode MatSeqAIJSetTypeFromOptions(Mat A)
{
  PetscBool flg;
  char      type[256];

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)A);
  PetscCall(PetscOptionsFList("-mat_seqaij_type", "Matrix SeqAIJ type", "MatSeqAIJSetType", MatSeqAIJList, "seqaij", type, 256, &flg));
  if (flg) PetscCall(MatSeqAIJSetType(A, type));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetColumnReductions_SeqAIJ(Mat A, PetscInt type, PetscReal *reductions)
{
  PetscInt    i, m, n;
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCall(PetscArrayzero(reductions, n));
  if (type == NORM_2) {
    for (i = 0; i < aij->i[m]; i++) reductions[aij->j[i]] += PetscAbsScalar(aij->a[i] * aij->a[i]);
  } else if (type == NORM_1) {
    for (i = 0; i < aij->i[m]; i++) reductions[aij->j[i]] += PetscAbsScalar(aij->a[i]);
  } else if (type == NORM_INFINITY) {
    for (i = 0; i < aij->i[m]; i++) reductions[aij->j[i]] = PetscMax(PetscAbsScalar(aij->a[i]), reductions[aij->j[i]]);
  } else if (type == REDUCTION_SUM_REALPART || type == REDUCTION_MEAN_REALPART) {
    for (i = 0; i < aij->i[m]; i++) reductions[aij->j[i]] += PetscRealPart(aij->a[i]);
  } else if (type == REDUCTION_SUM_IMAGINARYPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i = 0; i < aij->i[m]; i++) reductions[aij->j[i]] += PetscImaginaryPart(aij->a[i]);
  } else SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Unknown reduction type");

  if (type == NORM_2) {
    for (i = 0; i < n; i++) reductions[i] = PetscSqrtReal(reductions[i]);
  } else if (type == REDUCTION_MEAN_REALPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i = 0; i < n; i++) reductions[i] /= m;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFindOffBlockDiagonalEntries_SeqAIJ(Mat A, IS *is)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  PetscInt        i, m = A->rmap->n, cnt = 0, bs = A->rmap->bs;
  const PetscInt *jj = a->j, *ii = a->i;
  PetscInt       *rows;

  PetscFunctionBegin;
  for (i = 0; i < m; i++) {
    if ((ii[i] != ii[i + 1]) && ((jj[ii[i]] < bs * (i / bs)) || (jj[ii[i + 1] - 1] > bs * ((i + bs) / bs) - 1))) cnt++;
  }
  PetscCall(PetscMalloc1(cnt, &rows));
  cnt = 0;
  for (i = 0; i < m; i++) {
    if ((ii[i] != ii[i + 1]) && ((jj[ii[i]] < bs * (i / bs)) || (jj[ii[i + 1] - 1] > bs * ((i + bs) / bs) - 1))) {
      rows[cnt] = i;
      cnt++;
    }
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cnt, rows, PETSC_OWN_POINTER, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatFindZeroDiagonals_SeqAIJ_Private(Mat A, PetscInt *nrows, PetscInt **zrows)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  const MatScalar *aa;
  PetscInt         i, m = A->rmap->n, cnt = 0;
  const PetscInt  *ii = a->i, *jj = a->j, *diag;
  PetscInt        *rows;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  PetscCall(MatMarkDiagonal_SeqAIJ(A));
  diag = a->diag;
  for (i = 0; i < m; i++) {
    if ((diag[i] >= ii[i + 1]) || (jj[diag[i]] != i) || (aa[diag[i]] == 0.0)) cnt++;
  }
  PetscCall(PetscMalloc1(cnt, &rows));
  cnt = 0;
  for (i = 0; i < m; i++) {
    if ((diag[i] >= ii[i + 1]) || (jj[diag[i]] != i) || (aa[diag[i]] == 0.0)) rows[cnt++] = i;
  }
  *nrows = cnt;
  *zrows = rows;
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFindZeroDiagonals_SeqAIJ(Mat A, IS *zrows)
{
  PetscInt nrows, *rows;

  PetscFunctionBegin;
  *zrows = NULL;
  PetscCall(MatFindZeroDiagonals_SeqAIJ_Private(A, &nrows, &rows));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), nrows, rows, PETSC_OWN_POINTER, zrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFindNonzeroRows_SeqAIJ(Mat A, IS *keptrows)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  const MatScalar *aa;
  PetscInt         m = A->rmap->n, cnt = 0;
  const PetscInt  *ii;
  PetscInt         n, i, j, *rows;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  *keptrows = NULL;
  ii        = a->i;
  for (i = 0; i < m; i++) {
    n = ii[i + 1] - ii[i];
    if (!n) {
      cnt++;
      goto ok1;
    }
    for (j = ii[i]; j < ii[i + 1]; j++) {
      if (aa[j] != 0.0) goto ok1;
    }
    cnt++;
  ok1:;
  }
  if (!cnt) {
    PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc1(A->rmap->n - cnt, &rows));
  cnt = 0;
  for (i = 0; i < m; i++) {
    n = ii[i + 1] - ii[i];
    if (!n) continue;
    for (j = ii[i]; j < ii[i + 1]; j++) {
      if (aa[j] != 0.0) {
        rows[cnt++] = i;
        break;
      }
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cnt, rows, PETSC_OWN_POINTER, keptrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDiagonalSet_SeqAIJ(Mat Y, Vec D, InsertMode is)
{
  Mat_SeqAIJ        *aij = (Mat_SeqAIJ *)Y->data;
  PetscInt           i, m = Y->rmap->n;
  const PetscInt    *diag;
  MatScalar         *aa;
  const PetscScalar *v;
  PetscBool          missing;

  PetscFunctionBegin;
  if (Y->assembled) {
    PetscCall(MatMissingDiagonal_SeqAIJ(Y, &missing, NULL));
    if (!missing) {
      diag = aij->diag;
      PetscCall(VecGetArrayRead(D, &v));
      PetscCall(MatSeqAIJGetArray(Y, &aa));
      if (is == INSERT_VALUES) {
        for (i = 0; i < m; i++) aa[diag[i]] = v[i];
      } else {
        for (i = 0; i < m; i++) aa[diag[i]] += v[i];
      }
      PetscCall(MatSeqAIJRestoreArray(Y, &aa));
      PetscCall(VecRestoreArrayRead(D, &v));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(MatSeqAIJInvalidateDiagonal(Y));
  }
  PetscCall(MatDiagonalSet_Default(Y, D, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetRowIJ_SeqAIJ(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *m, const PetscInt *ia[], const PetscInt *ja[], PetscBool *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt    i, ishift;

  PetscFunctionBegin;
  if (m) *m = A->rmap->n;
  if (!ia) PetscFunctionReturn(PETSC_SUCCESS);
  ishift = 0;
  if (symmetric && A->structurally_symmetric != PETSC_BOOL3_TRUE) {
    PetscCall(MatToSymmetricIJ_SeqAIJ(A->rmap->n, a->i, a->j, PETSC_TRUE, ishift, oshift, (PetscInt **)ia, (PetscInt **)ja));
  } else if (oshift == 1) {
    PetscInt *tia;
    PetscInt  nz = a->i[A->rmap->n];
    /* malloc space and  add 1 to i and j indices */
    PetscCall(PetscMalloc1(A->rmap->n + 1, &tia));
    for (i = 0; i < A->rmap->n + 1; i++) tia[i] = a->i[i] + 1;
    *ia = tia;
    if (ja) {
      PetscInt *tja;
      PetscCall(PetscMalloc1(nz + 1, &tja));
      for (i = 0; i < nz; i++) tja[i] = a->j[i] + 1;
      *ja = tja;
    }
  } else {
    *ia = a->i;
    if (ja) *ja = a->j;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatRestoreRowIJ_SeqAIJ(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *n, const PetscInt *ia[], const PetscInt *ja[], PetscBool *done)
{
  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(PETSC_SUCCESS);
  if ((symmetric && A->structurally_symmetric != PETSC_BOOL3_TRUE) || oshift == 1) {
    PetscCall(PetscFree(*ia));
    if (ja) PetscCall(PetscFree(*ja));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetColumnIJ_SeqAIJ(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *nn, const PetscInt *ia[], const PetscInt *ja[], PetscBool *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt    i, *collengths, *cia, *cja, n = A->cmap->n, m = A->rmap->n;
  PetscInt    nz = a->i[m], row, *jj, mr, col;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(PETSC_SUCCESS);
  if (symmetric) {
    PetscCall(MatToSymmetricIJ_SeqAIJ(A->rmap->n, a->i, a->j, PETSC_TRUE, 0, oshift, (PetscInt **)ia, (PetscInt **)ja));
  } else {
    PetscCall(PetscCalloc1(n, &collengths));
    PetscCall(PetscMalloc1(n + 1, &cia));
    PetscCall(PetscMalloc1(nz, &cja));
    jj = a->j;
    for (i = 0; i < nz; i++) collengths[jj[i]]++;
    cia[0] = oshift;
    for (i = 0; i < n; i++) cia[i + 1] = cia[i] + collengths[i];
    PetscCall(PetscArrayzero(collengths, n));
    jj = a->j;
    for (row = 0; row < m; row++) {
      mr = a->i[row + 1] - a->i[row];
      for (i = 0; i < mr; i++) {
        col = *jj++;

        cja[cia[col] + collengths[col]++ - oshift] = row + oshift;
      }
    }
    PetscCall(PetscFree(collengths));
    *ia = cia;
    *ja = cja;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatRestoreColumnIJ_SeqAIJ(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *n, const PetscInt *ia[], const PetscInt *ja[], PetscBool *done)
{
  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFree(*ia));
  PetscCall(PetscFree(*ja));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 MatGetColumnIJ_SeqAIJ_Color() and MatRestoreColumnIJ_SeqAIJ_Color() are customized from
 MatGetColumnIJ_SeqAIJ() and MatRestoreColumnIJ_SeqAIJ() by adding an output
 spidx[], index of a->a, to be used in MatTransposeColoringCreate_SeqAIJ() and MatFDColoringCreate_SeqXAIJ()
*/
PetscErrorCode MatGetColumnIJ_SeqAIJ_Color(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *nn, const PetscInt *ia[], const PetscInt *ja[], PetscInt *spidx[], PetscBool *done)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  PetscInt        i, *collengths, *cia, *cja, n = A->cmap->n, m = A->rmap->n;
  PetscInt        nz = a->i[m], row, mr, col, tmp;
  PetscInt       *cspidx;
  const PetscInt *jj;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscCalloc1(n, &collengths));
  PetscCall(PetscMalloc1(n + 1, &cia));
  PetscCall(PetscMalloc1(nz, &cja));
  PetscCall(PetscMalloc1(nz, &cspidx));
  jj = a->j;
  for (i = 0; i < nz; i++) collengths[jj[i]]++;
  cia[0] = oshift;
  for (i = 0; i < n; i++) cia[i + 1] = cia[i] + collengths[i];
  PetscCall(PetscArrayzero(collengths, n));
  jj = a->j;
  for (row = 0; row < m; row++) {
    mr = a->i[row + 1] - a->i[row];
    for (i = 0; i < mr; i++) {
      col         = *jj++;
      tmp         = cia[col] + collengths[col]++ - oshift;
      cspidx[tmp] = a->i[row] + i; /* index of a->j */
      cja[tmp]    = row + oshift;
    }
  }
  PetscCall(PetscFree(collengths));
  *ia    = cia;
  *ja    = cja;
  *spidx = cspidx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatRestoreColumnIJ_SeqAIJ_Color(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *n, const PetscInt *ia[], const PetscInt *ja[], PetscInt *spidx[], PetscBool *done)
{
  PetscFunctionBegin;
  PetscCall(MatRestoreColumnIJ_SeqAIJ(A, oshift, symmetric, inodecompressed, n, ia, ja, done));
  PetscCall(PetscFree(*spidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesRow_SeqAIJ(Mat A, PetscInt row, const PetscScalar v[])
{
  Mat_SeqAIJ  *a  = (Mat_SeqAIJ *)A->data;
  PetscInt    *ai = a->i;
  PetscScalar *aa;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(A, &aa));
  PetscCall(PetscArraycpy(aa + ai[row], v, ai[row + 1] - ai[row]));
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    MatSeqAIJSetValuesLocalFast - An optimized version of MatSetValuesLocal() for SeqAIJ matrices with several assumptions

      -   a single row of values is set with each call
      -   no row or column indices are negative or (in error) larger than the number of rows or columns
      -   the values are always added to the matrix, not set
      -   no new locations are introduced in the nonzero structure of the matrix

     This does NOT assume the global column indices are sorted

*/

#include <petsc/private/isimpl.h>
PetscErrorCode MatSeqAIJSetValuesLocalFast(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  PetscInt        low, high, t, row, nrow, i, col, l;
  const PetscInt *rp, *ai = a->i, *ailen = a->ilen, *aj = a->j;
  PetscInt        lastcol = -1;
  MatScalar      *ap, value, *aa;
  const PetscInt *ridx = A->rmap->mapping->indices, *cidx = A->cmap->mapping->indices;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(A, &aa));
  row  = ridx[im[0]];
  rp   = aj + ai[row];
  ap   = aa + ai[row];
  nrow = ailen[row];
  low  = 0;
  high = nrow;
  for (l = 0; l < n; l++) { /* loop over added columns */
    col   = cidx[in[l]];
    value = v[l];

    if (col <= lastcol) low = 0;
    else high = nrow;
    lastcol = col;
    while (high - low > 5) {
      t = (low + high) / 2;
      if (rp[t] > col) high = t;
      else low = t;
    }
    for (i = low; i < high; i++) {
      if (rp[i] == col) {
        ap[i] += value;
        low = i + 1;
        break;
      }
    }
  }
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  return PETSC_SUCCESS;
}

PetscErrorCode MatSetValues_SeqAIJ(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt   *rp, k, low, high, t, ii, row, nrow, i, col, l, rmax, N;
  PetscInt   *imax = a->imax, *ai = a->i, *ailen = a->ilen;
  PetscInt   *aj = a->j, nonew = a->nonew, lastcol = -1;
  MatScalar  *ap = NULL, value = 0.0, *aa;
  PetscBool   ignorezeroentries = a->ignorezeroentries;
  PetscBool   roworiented       = a->roworiented;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(A, &aa));
  for (k = 0; k < m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    PetscCheck(row < A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT, row, A->rmap->n - 1);
    rp = PetscSafePointerPlusOffset(aj, ai[row]);
    if (!A->structure_only) ap = PetscSafePointerPlusOffset(aa, ai[row]);
    rmax = imax[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l = 0; l < n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      PetscCheck(in[l] < A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT, in[l], A->cmap->n - 1);
      col = in[l];
      if (v && !A->structure_only) value = roworiented ? v[l + k * n] : v[k + l * m];
      if (!A->structure_only && value == 0.0 && ignorezeroentries && is == ADD_VALUES && row != col) continue;

      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high - low > 5) {
        t = (low + high) / 2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i = low; i < high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (!A->structure_only) {
            if (is == ADD_VALUES) {
              ap[i] += value;
              (void)PetscLogFlops(1.0);
            } else ap[i] = value;
          }
          low = i + 1;
          goto noinsert;
        }
      }
      if (value == 0.0 && ignorezeroentries && row != col) goto noinsert;
      if (nonew == 1) goto noinsert;
      PetscCheck(nonew != -1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserting a new nonzero at (%" PetscInt_FMT ",%" PetscInt_FMT ") in the matrix", row, col);
      if (A->structure_only) {
        MatSeqXAIJReallocateAIJ_structure_only(A, A->rmap->n, 1, nrow, row, col, rmax, ai, aj, rp, imax, nonew, MatScalar);
      } else {
        MatSeqXAIJReallocateAIJ(A, A->rmap->n, 1, nrow, row, col, rmax, aa, ai, aj, rp, ap, imax, nonew, MatScalar);
      }
      N = nrow++ - 1;
      a->nz++;
      high++;
      /* shift up all the later entries in this row */
      PetscCall(PetscArraymove(rp + i + 1, rp + i, N - i + 1));
      rp[i] = col;
      if (!A->structure_only) {
        PetscCall(PetscArraymove(ap + i + 1, ap + i, N - i + 1));
        ap[i] = value;
      }
      low = i + 1;
    noinsert:;
    }
    ailen[row] = nrow;
  }
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValues_SeqAIJ_SortedFullNoPreallocation(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt   *rp, k, row;
  PetscInt   *ai = a->i;
  PetscInt   *aj = a->j;
  MatScalar  *aa, *ap;

  PetscFunctionBegin;
  PetscCheck(!A->was_assembled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot call on assembled matrix.");
  PetscCheck(m * n + a->nz <= a->maxnz, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of entries in matrix will be larger than maximum nonzeros allocated for %" PetscInt_FMT " in MatSeqAIJSetTotalPreallocation()", a->maxnz);

  PetscCall(MatSeqAIJGetArray(A, &aa));
  for (k = 0; k < m; k++) { /* loop over added rows */
    row = im[k];
    rp  = aj + ai[row];
    ap  = PetscSafePointerPlusOffset(aa, ai[row]);

    PetscCall(PetscMemcpy(rp, in, n * sizeof(PetscInt)));
    if (!A->structure_only) {
      if (v) {
        PetscCall(PetscMemcpy(ap, v, n * sizeof(PetscScalar)));
        v += n;
      } else {
        PetscCall(PetscMemzero(ap, n * sizeof(PetscScalar)));
      }
    }
    a->ilen[row]  = n;
    a->imax[row]  = n;
    a->i[row + 1] = a->i[row] + n;
    a->nz += n;
  }
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSeqAIJSetTotalPreallocation - Sets an upper bound on the total number of expected nonzeros in the matrix.

  Input Parameters:
+ A       - the `MATSEQAIJ` matrix
- nztotal - bound on the number of nonzeros

  Level: advanced

  Notes:
  This can be called if you will be provided the matrix row by row (from row zero) with sorted column indices for each row.
  Simply call `MatSetValues()` after this call to provide the matrix entries in the usual manner. This matrix may be used
  as always with multiple matrix assemblies.

.seealso: [](ch_matrices), `Mat`, `MatSetOption()`, `MAT_SORTED_FULL`, `MatSetValues()`, `MatSeqAIJSetPreallocation()`
@*/
PetscErrorCode MatSeqAIJSetTotalPreallocation(Mat A, PetscInt nztotal)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  a->maxnz = nztotal;
  if (!a->imax) { PetscCall(PetscMalloc1(A->rmap->n, &a->imax)); }
  if (!a->ilen) {
    PetscCall(PetscMalloc1(A->rmap->n, &a->ilen));
  } else {
    PetscCall(PetscMemzero(a->ilen, A->rmap->n * sizeof(PetscInt)));
  }

  /* allocate the matrix space */
  PetscCall(PetscShmgetAllocateArray(A->rmap->n + 1, sizeof(PetscInt), (void **)&a->i));
  PetscCall(PetscShmgetAllocateArray(nztotal, sizeof(PetscInt), (void **)&a->j));
  a->free_ij = PETSC_TRUE;
  if (A->structure_only) {
    a->free_a = PETSC_FALSE;
  } else {
    PetscCall(PetscShmgetAllocateArray(nztotal, sizeof(PetscScalar), (void **)&a->a));
    a->free_a = PETSC_TRUE;
  }
  a->i[0]           = 0;
  A->ops->setvalues = MatSetValues_SeqAIJ_SortedFullNoPreallocation;
  A->preallocated   = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValues_SeqAIJ_SortedFull(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt   *rp, k, row;
  PetscInt   *ai = a->i, *ailen = a->ilen;
  PetscInt   *aj = a->j;
  MatScalar  *aa, *ap;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(A, &aa));
  for (k = 0; k < m; k++) { /* loop over added rows */
    row = im[k];
    PetscCheck(n <= a->imax[row], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Preallocation for row %" PetscInt_FMT " does not match number of columns provided", n);
    rp = aj + ai[row];
    ap = aa + ai[row];
    if (!A->was_assembled) PetscCall(PetscMemcpy(rp, in, n * sizeof(PetscInt)));
    if (!A->structure_only) {
      if (v) {
        PetscCall(PetscMemcpy(ap, v, n * sizeof(PetscScalar)));
        v += n;
      } else {
        PetscCall(PetscMemzero(ap, n * sizeof(PetscScalar)));
      }
    }
    ailen[row] = n;
    a->nz += n;
  }
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetValues_SeqAIJ(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], PetscScalar v[])
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  PetscInt        *rp, k, low, high, t, row, nrow, i, col, l, *aj = a->j;
  PetscInt        *ai = a->i, *ailen = a->ilen;
  const MatScalar *ap, *aa;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  for (k = 0; k < m; k++) { /* loop over rows */
    row = im[k];
    if (row < 0) {
      v += n;
      continue;
    } /* negative row */
    PetscCheck(row < A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT, row, A->rmap->n - 1);
    rp   = PetscSafePointerPlusOffset(aj, ai[row]);
    ap   = PetscSafePointerPlusOffset(aa, ai[row]);
    nrow = ailen[row];
    for (l = 0; l < n; l++) { /* loop over columns */
      if (in[l] < 0) {
        v++;
        continue;
      } /* negative column */
      PetscCheck(in[l] < A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT, in[l], A->cmap->n - 1);
      col  = in[l];
      high = nrow;
      low  = 0; /* assume unsorted */
      while (high - low > 5) {
        t = (low + high) / 2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i = low; i < high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          *v++ = ap[i];
          goto finished;
        }
      }
      *v++ = 0.0;
    finished:;
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_SeqAIJ_Binary(Mat mat, PetscViewer viewer)
{
  Mat_SeqAIJ        *A = (Mat_SeqAIJ *)mat->data;
  const PetscScalar *av;
  PetscInt           header[4], M, N, m, nz, i;
  PetscInt          *rowlens;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));

  M  = mat->rmap->N;
  N  = mat->cmap->N;
  m  = mat->rmap->n;
  nz = A->nz;

  /* write matrix header */
  header[0] = MAT_FILE_CLASSID;
  header[1] = M;
  header[2] = N;
  header[3] = nz;
  PetscCall(PetscViewerBinaryWrite(viewer, header, 4, PETSC_INT));

  /* fill in and store row lengths */
  PetscCall(PetscMalloc1(m, &rowlens));
  for (i = 0; i < m; i++) rowlens[i] = A->i[i + 1] - A->i[i];
  if (PetscDefined(USE_DEBUG)) {
    PetscInt mnz = 0;

    for (i = 0; i < m; i++) mnz += rowlens[i];
    PetscCheck(nz == mnz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Row lens %" PetscInt_FMT " do not sum to nz %" PetscInt_FMT, mnz, nz);
  }
  PetscCall(PetscViewerBinaryWrite(viewer, rowlens, m, PETSC_INT));
  PetscCall(PetscFree(rowlens));
  /* store column indices */
  PetscCall(PetscViewerBinaryWrite(viewer, A->j, nz, PETSC_INT));
  /* store nonzero values */
  PetscCall(MatSeqAIJGetArrayRead(mat, &av));
  PetscCall(PetscViewerBinaryWrite(viewer, av, nz, PETSC_SCALAR));
  PetscCall(MatSeqAIJRestoreArrayRead(mat, &av));

  /* write block size option to the viewer's .info file */
  PetscCall(MatView_Binary_BlockSizes(mat, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_SeqAIJ_ASCII_structonly(Mat A, PetscViewer viewer)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt    i, k, m = A->rmap->N;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
  for (i = 0; i < m; i++) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "row %" PetscInt_FMT ":", i));
    for (k = a->i[i]; k < a->i[i + 1]; k++) PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ") ", a->j[k]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_SeqAIJ_ASCII(Mat A, PetscViewer viewer)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  const PetscScalar *av;
  PetscInt           i, j, m = A->rmap->n;
  const char        *name;
  PetscViewerFormat  format;

  PetscFunctionBegin;
  if (A->structure_only) {
    PetscCall(MatView_SeqAIJ_ASCII_structonly(A, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscViewerGetFormat(viewer, &format));
  // By petsc's rule, even PETSC_VIEWER_ASCII_INFO_DETAIL doesn't print matrix entries
  if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(PETSC_SUCCESS);

  /* trigger copy to CPU if needed */
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscInt nofinalvalue = 0;
    if (m && ((a->i[m] == a->i[m - 1]) || (a->j[a->nz - 1] != A->cmap->n - 1))) {
      /* Need a dummy value to ensure the dimension of the matrix. */
      nofinalvalue = 1;
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%% Size = %" PetscInt_FMT " %" PetscInt_FMT " \n", m, A->cmap->n));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%% Nonzeros = %" PetscInt_FMT " \n", a->nz));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer, "zzz = zeros(%" PetscInt_FMT ",4);\n", a->nz + nofinalvalue));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer, "zzz = zeros(%" PetscInt_FMT ",3);\n", a->nz + nofinalvalue));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer, "zzz = [\n"));

    for (i = 0; i < m; i++) {
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e %18.16e\n", i + 1, a->j[j] + 1, (double)PetscRealPart(a->a[j]), (double)PetscImaginaryPart(a->a[j])));
#else
        PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e\n", i + 1, a->j[j] + 1, (double)a->a[j]));
#endif
      }
    }
    if (nofinalvalue) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e %18.16e\n", m, A->cmap->n, 0., 0.));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e\n", m, A->cmap->n, 0.0));
#endif
    }
    PetscCall(PetscObjectGetName((PetscObject)A, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "];\n %s = spconvert(zzz);\n", name));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    for (i = 0; i < m; i++) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "row %" PetscInt_FMT ":", i));
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g + %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)PetscImaginaryPart(a->a[j])));
        } else if (PetscImaginaryPart(a->a[j]) < 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g - %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)-PetscImaginaryPart(a->a[j])));
        } else if (PetscRealPart(a->a[j]) != 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)PetscRealPart(a->a[j])));
        }
#else
        if (a->a[j] != 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)a->a[j]));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    PetscInt nzd = 0, fshift = 1, *sptr;
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    PetscCall(PetscMalloc1(m + 1, &sptr));
    for (i = 0; i < m; i++) {
      sptr[i] = nzd + 1;
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        if (a->j[j] >= i) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) != 0.0 || PetscRealPart(a->a[j]) != 0.0) nzd++;
#else
          if (a->a[j] != 0.0) nzd++;
#endif
        }
      }
    }
    sptr[m] = nzd + 1;
    PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT " %" PetscInt_FMT "\n\n", m, nzd));
    for (i = 0; i < m + 1; i += 6) {
      if (i + 4 < m) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", sptr[i], sptr[i + 1], sptr[i + 2], sptr[i + 3], sptr[i + 4], sptr[i + 5]));
      } else if (i + 3 < m) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", sptr[i], sptr[i + 1], sptr[i + 2], sptr[i + 3], sptr[i + 4]));
      } else if (i + 2 < m) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", sptr[i], sptr[i + 1], sptr[i + 2], sptr[i + 3]));
      } else if (i + 1 < m) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", sptr[i], sptr[i + 1], sptr[i + 2]));
      } else if (i < m) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT " %" PetscInt_FMT "\n", sptr[i], sptr[i + 1]));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT "\n", sptr[i]));
      }
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscFree(sptr));
    for (i = 0; i < m; i++) {
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        if (a->j[j] >= i) PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT " ", a->j[j] + fshift));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    for (i = 0; i < m; i++) {
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        if (a->j[j] >= i) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) != 0.0 || PetscRealPart(a->a[j]) != 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, " %18.16e %18.16e ", (double)PetscRealPart(a->a[j]), (double)PetscImaginaryPart(a->a[j])));
#else
          if (a->a[j] != 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, " %18.16e ", (double)a->a[j]));
#endif
        }
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  } else if (format == PETSC_VIEWER_ASCII_DENSE) {
    PetscInt    cnt = 0, jcnt;
    PetscScalar value;
#if defined(PETSC_USE_COMPLEX)
    PetscBool realonly = PETSC_TRUE;

    for (i = 0; i < a->i[m]; i++) {
      if (PetscImaginaryPart(a->a[i]) != 0.0) {
        realonly = PETSC_FALSE;
        break;
      }
    }
#endif

    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    for (i = 0; i < m; i++) {
      jcnt = 0;
      for (j = 0; j < A->cmap->n; j++) {
        if (jcnt < a->i[i + 1] - a->i[i] && j == a->j[cnt]) {
          value = a->a[cnt++];
          jcnt++;
        } else {
          value = 0.0;
        }
#if defined(PETSC_USE_COMPLEX)
        if (realonly) {
          PetscCall(PetscViewerASCIIPrintf(viewer, " %7.5e ", (double)PetscRealPart(value)));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, " %7.5e+%7.5e i ", (double)PetscRealPart(value), (double)PetscImaginaryPart(value)));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer, " %7.5e ", (double)value));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  } else if (format == PETSC_VIEWER_ASCII_MATRIXMARKET) {
    PetscInt fshift = 1;
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer, "%%%%MatrixMarket matrix coordinate complex general\n"));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer, "%%%%MatrixMarket matrix coordinate real general\n"));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", m, A->cmap->n, a->nz));
    for (i = 0; i < m; i++) {
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT " %g %g\n", i + fshift, a->j[j] + fshift, (double)PetscRealPart(a->a[j]), (double)PetscImaginaryPart(a->a[j])));
#else
        PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT " %g\n", i + fshift, a->j[j] + fshift, (double)a->a[j]));
#endif
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  } else {
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    if (A->factortype) {
      for (i = 0; i < m; i++) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "row %" PetscInt_FMT ":", i));
        /* L part */
        for (j = a->i[i]; j < a->i[i + 1]; j++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g + %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)PetscImaginaryPart(a->a[j])));
          } else if (PetscImaginaryPart(a->a[j]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g - %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)(-PetscImaginaryPart(a->a[j]))));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)PetscRealPart(a->a[j])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)a->a[j]));
#endif
        }
        /* diagonal */
        j = a->diag[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g + %g i)", a->j[j], (double)PetscRealPart(1 / a->a[j]), (double)PetscImaginaryPart(1 / a->a[j])));
        } else if (PetscImaginaryPart(a->a[j]) < 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g - %g i)", a->j[j], (double)PetscRealPart(1 / a->a[j]), (double)(-PetscImaginaryPart(1 / a->a[j]))));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)PetscRealPart(1 / a->a[j])));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)(1 / a->a[j])));
#endif

        /* U part */
        for (j = a->diag[i + 1] + 1; j < a->diag[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g + %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)PetscImaginaryPart(a->a[j])));
          } else if (PetscImaginaryPart(a->a[j]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g - %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)(-PetscImaginaryPart(a->a[j]))));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)PetscRealPart(a->a[j])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)a->a[j]));
#endif
        }
        PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      }
    } else {
      for (i = 0; i < m; i++) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "row %" PetscInt_FMT ":", i));
        for (j = a->i[i]; j < a->i[i + 1]; j++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g + %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)PetscImaginaryPart(a->a[j])));
          } else if (PetscImaginaryPart(a->a[j]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g - %g i)", a->j[j], (double)PetscRealPart(a->a[j]), (double)-PetscImaginaryPart(a->a[j])));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)PetscRealPart(a->a[j])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT ", %g) ", a->j[j], (double)a->a[j]));
#endif
        }
        PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscdraw.h>
static PetscErrorCode MatView_SeqAIJ_Draw_Zoom(PetscDraw draw, void *Aa)
{
  Mat                A = (Mat)Aa;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscInt           i, j, m = A->rmap->n;
  int                color;
  PetscReal          xl, yl, xr, yr, x_l, x_r, y_l, y_r;
  PetscViewer        viewer;
  PetscViewerFormat  format;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "Zoomviewer", (PetscObject *)&viewer));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscDrawGetCoordinates(draw, &xl, &yl, &xr, &yr));

  /* loop over matrix elements drawing boxes */
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    PetscDrawCollectiveBegin(draw);
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = PETSC_DRAW_BLUE;
    for (i = 0; i < m; i++) {
      y_l = m - i - 1.0;
      y_r = y_l + 1.0;
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        x_l = a->j[j];
        x_r = x_l + 1.0;
        if (PetscRealPart(aa[j]) >= 0.) continue;
        PetscCall(PetscDrawRectangle(draw, x_l, y_l, x_r, y_r, color, color, color, color));
      }
    }
    color = PETSC_DRAW_CYAN;
    for (i = 0; i < m; i++) {
      y_l = m - i - 1.0;
      y_r = y_l + 1.0;
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        x_l = a->j[j];
        x_r = x_l + 1.0;
        if (aa[j] != 0.) continue;
        PetscCall(PetscDrawRectangle(draw, x_l, y_l, x_r, y_r, color, color, color, color));
      }
    }
    color = PETSC_DRAW_RED;
    for (i = 0; i < m; i++) {
      y_l = m - i - 1.0;
      y_r = y_l + 1.0;
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        x_l = a->j[j];
        x_r = x_l + 1.0;
        if (PetscRealPart(aa[j]) <= 0.) continue;
        PetscCall(PetscDrawRectangle(draw, x_l, y_l, x_r, y_r, color, color, color, color));
      }
    }
    PetscDrawCollectiveEnd(draw);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv = 0.0, maxv = 0.0;
    PetscInt  nz = a->nz, count = 0;
    PetscDraw popup;

    for (i = 0; i < nz; i++) {
      if (PetscAbsScalar(aa[i]) > maxv) maxv = PetscAbsScalar(aa[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    PetscCall(PetscDrawGetPopup(draw, &popup));
    PetscCall(PetscDrawScalePopup(popup, minv, maxv));

    PetscDrawCollectiveBegin(draw);
    for (i = 0; i < m; i++) {
      y_l = m - i - 1.0;
      y_r = y_l + 1.0;
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        x_l   = a->j[j];
        x_r   = x_l + 1.0;
        color = PetscDrawRealToColor(PetscAbsScalar(aa[count]), minv, maxv);
        PetscCall(PetscDrawRectangle(draw, x_l, y_l, x_r, y_r, color, color, color, color));
        count++;
      }
    }
    PetscDrawCollectiveEnd(draw);
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscdraw.h>
static PetscErrorCode MatView_SeqAIJ_Draw(Mat A, PetscViewer viewer)
{
  PetscDraw draw;
  PetscReal xr, yr, xl, yl, h, w;
  PetscBool isnull;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
  PetscCall(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(PETSC_SUCCESS);

  xr = A->cmap->n;
  yr = A->rmap->n;
  h  = yr / 10.0;
  w  = xr / 10.0;
  xr += w;
  yr += h;
  xl = -w;
  yl = -h;
  PetscCall(PetscDrawSetCoordinates(draw, xl, yl, xr, yr));
  PetscCall(PetscObjectCompose((PetscObject)A, "Zoomviewer", (PetscObject)viewer));
  PetscCall(PetscDrawZoom(draw, MatView_SeqAIJ_Draw_Zoom, A));
  PetscCall(PetscObjectCompose((PetscObject)A, "Zoomviewer", NULL));
  PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatView_SeqAIJ(Mat A, PetscViewer viewer)
{
  PetscBool iascii, isbinary, isdraw;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  if (iascii) PetscCall(MatView_SeqAIJ_ASCII(A, viewer));
  else if (isbinary) PetscCall(MatView_SeqAIJ_Binary(A, viewer));
  else if (isdraw) PetscCall(MatView_SeqAIJ_Draw(A, viewer));
  PetscCall(MatView_SeqAIJ_Inode(A, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatAssemblyEnd_SeqAIJ(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJ *a      = (Mat_SeqAIJ *)A->data;
  PetscInt    fshift = 0, i, *ai = a->i, *aj = a->j, *imax = a->imax;
  PetscInt    m = A->rmap->n, *ip, N, *ailen = a->ilen, rmax = 0, n;
  MatScalar  *aa    = a->a, *ap;
  PetscReal   ratio = 0.6;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  if (A->was_assembled && A->ass_nonzerostate == A->nonzerostate) {
    /* we need to respect users asking to use or not the inodes routine in between matrix assemblies, e.g., via MatSetOption(A, MAT_USE_INODES, val) */
    PetscCall(MatAssemblyEnd_SeqAIJ_Inode(A, mode)); /* read the sparsity pattern */
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i = 1; i < m; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i - 1] - ailen[i - 1];
    rmax = PetscMax(rmax, ailen[i]);
    if (fshift) {
      ip = aj + ai[i];
      ap = aa + ai[i];
      N  = ailen[i];
      PetscCall(PetscArraymove(ip - fshift, ip, N));
      if (!A->structure_only) PetscCall(PetscArraymove(ap - fshift, ap, N));
    }
    ai[i] = ai[i - 1] + ailen[i - 1];
  }
  if (m) {
    fshift += imax[m - 1] - ailen[m - 1];
    ai[m] = ai[m - 1] + ailen[m - 1];
  }
  /* reset ilen and imax for each row */
  a->nonzerorowcnt = 0;
  if (A->structure_only) {
    PetscCall(PetscFree(a->imax));
    PetscCall(PetscFree(a->ilen));
  } else { /* !A->structure_only */
    for (i = 0; i < m; i++) {
      ailen[i] = imax[i] = ai[i + 1] - ai[i];
      a->nonzerorowcnt += ((ai[i + 1] - ai[i]) > 0);
    }
  }
  a->nz = ai[m];
  PetscCheck(!fshift || a->nounused != -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unused space detected in matrix: %" PetscInt_FMT " X %" PetscInt_FMT ", %" PetscInt_FMT " unneeded", m, A->cmap->n, fshift);
  PetscCall(MatMarkDiagonal_SeqAIJ(A)); // since diagonal info is used a lot, it is helpful to set them up at the end of assembly
  a->diagonaldense = PETSC_TRUE;
  n                = PetscMin(A->rmap->n, A->cmap->n);
  for (i = 0; i < n; i++) {
    if (a->diag[i] >= ai[i + 1]) {
      a->diagonaldense = PETSC_FALSE;
      break;
    }
  }
  PetscCall(PetscInfo(A, "Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: %" PetscInt_FMT " unneeded,%" PetscInt_FMT " used\n", m, A->cmap->n, fshift, a->nz));
  PetscCall(PetscInfo(A, "Number of mallocs during MatSetValues() is %" PetscInt_FMT "\n", a->reallocs));
  PetscCall(PetscInfo(A, "Maximum nonzeros in any row is %" PetscInt_FMT "\n", rmax));

  A->info.mallocs += a->reallocs;
  a->reallocs         = 0;
  A->info.nz_unneeded = (PetscReal)fshift;
  a->rmax             = rmax;

  if (!A->structure_only) PetscCall(MatCheckCompressedRow(A, a->nonzerorowcnt, &a->compressedrow, a->i, m, ratio));
  PetscCall(MatAssemblyEnd_SeqAIJ_Inode(A, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRealPart_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt    i, nz = a->nz;
  MatScalar  *aa;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(A, &aa));
  for (i = 0; i < nz; i++) aa[i] = PetscRealPart(aa[i]);
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatImaginaryPart_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt    i, nz = a->nz;
  MatScalar  *aa;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(A, &aa));
  for (i = 0; i < nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatZeroEntries_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  MatScalar  *aa;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayWrite(A, &aa));
  PetscCall(PetscArrayzero(aa, a->i[A->rmap->n]));
  PetscCall(MatSeqAIJRestoreArrayWrite(A, &aa));
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (A->hash_active) {
    A->ops[0] = a->cops;
    PetscCall(PetscHMapIJVDestroy(&a->ht));
    PetscCall(PetscFree(a->dnz));
    A->hash_active = PETSC_FALSE;
  }

  PetscCall(PetscLogObjectState((PetscObject)A, "Rows=%" PetscInt_FMT ", Cols=%" PetscInt_FMT ", NZ=%" PetscInt_FMT, A->rmap->n, A->cmap->n, a->nz));
  PetscCall(MatSeqXAIJFreeAIJ(A, &a->a, &a->j, &a->i));
  PetscCall(ISDestroy(&a->row));
  PetscCall(ISDestroy(&a->col));
  PetscCall(PetscFree(a->diag));
  PetscCall(PetscFree(a->ibdiag));
  PetscCall(PetscFree(a->imax));
  PetscCall(PetscFree(a->ilen));
  PetscCall(PetscFree(a->ipre));
  PetscCall(PetscFree3(a->idiag, a->mdiag, a->ssor_work));
  PetscCall(PetscFree(a->solve_work));
  PetscCall(ISDestroy(&a->icol));
  PetscCall(PetscFree(a->saved_values));
  PetscCall(PetscFree2(a->compressedrow.i, a->compressedrow.rindex));
  PetscCall(MatDestroy_SeqAIJ_Inode(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatResetHash_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatReset_SeqAIJ(A));
  PetscCall(MatCreate_SeqAIJ_Inode(A));
  PetscCall(MatSetUp_Seq_Hash(A));
  A->nonzerostate++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatReset_SeqAIJ(A));
  PetscCall(PetscFree(A->data));

  /* MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted may allocate this.
     That function is so heavily used (sometimes in an hidden way through multnumeric function pointers)
     that is hard to properly add this data to the MatProduct data. We free it here to avoid
     users reusing the matrix object with different data to incur in obscure segmentation faults
     due to different matrix sizes */
  PetscCall(PetscObjectCompose((PetscObject)A, "__PETSc__ab_dense", NULL));

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "PetscMatlabEnginePut_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "PetscMatlabEngineGet_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJSetColumnIndices_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatStoreValues_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatRetrieveValues_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqsbaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqbaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijperm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijsell_C", NULL));
#if defined(PETSC_HAVE_MKL_SPARSE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijmkl_C", NULL));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijcusparse_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijcusparse_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqaijcusparse_C", NULL));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijhipsparse_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqaijhipsparse_C", NULL));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijkokkos_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijcrl_C", NULL));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_elemental_C", NULL));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_scalapack_C", NULL));
#endif
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_transpose_seqaij_seqaij_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqsell_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_is_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatIsTranspose_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatIsHermitianTranspose_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJSetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatResetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatResetHash_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJSetPreallocationCSR_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatReorderForNonzeroDiagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_is_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdense_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJKron_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  /* these calls do not belong here: the subclasses Duplicate/Destroy are wrong */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaijsell_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaijperm_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaij_seqaijviennacl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijviennacl_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijviennacl_seqaij_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetOption_SeqAIJ(Mat A, MatOption op, PetscBool flg)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    break;
  case MAT_KEEP_NONZERO_PATTERN:
    a->keepnonzeropattern = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
    a->nonew = (flg ? 0 : 1);
    break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    a->nonew = (flg ? -1 : 0);
    break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    a->nonew = (flg ? -2 : 0);
    break;
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
    a->nounused = (flg ? -1 : 0);
    break;
  case MAT_IGNORE_ZERO_ENTRIES:
    a->ignorezeroentries = flg;
    break;
  case MAT_USE_INODES:
    PetscCall(MatSetOption_SeqAIJ_Inode(A, MAT_USE_INODES, flg));
    break;
  case MAT_SUBMAT_SINGLEIS:
    A->submat_singleis = flg;
    break;
  case MAT_SORTED_FULL:
    if (flg) A->ops->setvalues = MatSetValues_SeqAIJ_SortedFull;
    else A->ops->setvalues = MatSetValues_SeqAIJ;
    break;
  case MAT_FORM_EXPLICIT_TRANSPOSE:
    A->form_explicit_transpose = flg;
    break;
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_SeqAIJ(Mat A, Vec v)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscInt           i, j, n, *ai = a->i, *aj = a->j;
  PetscScalar       *x;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  if (A->factortype == MAT_FACTOR_ILU || A->factortype == MAT_FACTOR_LU) {
    PetscInt *diag = a->diag;
    PetscCall(VecGetArrayWrite(v, &x));
    for (i = 0; i < n; i++) x[i] = 1.0 / aa[diag[i]];
    PetscCall(VecRestoreArrayWrite(v, &x));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecGetArrayWrite(v, &x));
  for (i = 0; i < n; i++) {
    x[i] = 0.0;
    for (j = ai[i]; j < ai[i + 1]; j++) {
      if (aj[j] == i) {
        x[i] = aa[j];
        break;
      }
    }
  }
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/mat/impls/aij/seq/ftn-kernels/fmult.h>
PetscErrorCode MatMultTransposeAdd_SeqAIJ(Mat A, Vec xx, Vec zz, Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  const MatScalar   *aa;
  PetscScalar       *y;
  const PetscScalar *x;
  PetscInt           m = A->rmap->n;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
  const MatScalar  *v;
  PetscScalar       alpha;
  PetscInt          n, i, j;
  const PetscInt   *idx, *ii, *ridx = NULL;
  Mat_CompressedRow cprow    = a->compressedrow;
  PetscBool         usecprow = cprow.use;
#endif

  PetscFunctionBegin;
  if (zz != yy) PetscCall(VecCopy(zz, yy));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(yy, &y));
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
  fortranmulttransposeaddaij_(&m, x, a->i, a->j, aa, y);
#else
  if (usecprow) {
    m    = cprow.nrows;
    ii   = cprow.i;
    ridx = cprow.rindex;
  } else {
    ii = a->i;
  }
  for (i = 0; i < m; i++) {
    idx = a->j + ii[i];
    v   = aa + ii[i];
    n   = ii[i + 1] - ii[i];
    if (usecprow) {
      alpha = x[ridx[i]];
    } else {
      alpha = x[i];
    }
    for (j = 0; j < n; j++) y[idx[j]] += alpha * v[j];
  }
#endif
  PetscCall(PetscLogFlops(2.0 * a->nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultTranspose_SeqAIJ(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(VecSet(yy, 0.0));
  PetscCall(MatMultTransposeAdd_SeqAIJ(A, xx, yy, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/mat/impls/aij/seq/ftn-kernels/fmult.h>

PetscErrorCode MatMult_SeqAIJ(Mat A, Vec xx, Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *a_a;
  PetscInt           m = A->rmap->n;
  const PetscInt    *ii, *ridx = NULL;
  PetscBool          usecprow = a->compressedrow.use;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
  #pragma disjoint(*x, *y, *aa)
#endif

  PetscFunctionBegin;
  if (a->inode.use && a->inode.checked) {
    PetscCall(MatMult_SeqAIJ_Inode(A, xx, yy));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatSeqAIJGetArrayRead(A, &a_a));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(yy, &y));
  ii = a->i;
  if (usecprow) { /* use compressed row format */
    PetscCall(PetscArrayzero(y, m));
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    PetscPragmaUseOMPKernels(parallel for)
    for (PetscInt i = 0; i < m; i++) {
      PetscInt           n   = ii[i + 1] - ii[i];
      const PetscInt    *aj  = a->j + ii[i];
      const PetscScalar *aa  = a_a + ii[i];
      PetscScalar        sum = 0.0;
      PetscSparseDensePlusDot(sum, x, aa, aj, n);
      /* for (j=0; j<n; j++) sum += (*aa++)*x[*aj++]; */
      y[ridx[i]] = sum;
    }
  } else { /* do not use compressed row format */
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
    fortranmultaij_(&m, x, ii, a->j, a_a, y);
#else
    PetscPragmaUseOMPKernels(parallel for)
    for (PetscInt i = 0; i < m; i++) {
      PetscInt           n   = ii[i + 1] - ii[i];
      const PetscInt    *aj  = a->j + ii[i];
      const PetscScalar *aa  = a_a + ii[i];
      PetscScalar        sum = 0.0;
      PetscSparseDensePlusDot(sum, x, aa, aj, n);
      y[i] = sum;
    }
#endif
  }
  PetscCall(PetscLogFlops(2.0 * a->nz - a->nonzerorowcnt));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &a_a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// HACK!!!!! Used by src/mat/tests/ex170.c
PETSC_EXTERN PetscErrorCode MatMultMax_SeqAIJ(Mat A, Vec xx, Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *aa, *a_a;
  PetscInt           m = A->rmap->n;
  const PetscInt    *aj, *ii, *ridx   = NULL;
  PetscInt           n, i, nonzerorow = 0;
  PetscScalar        sum;
  PetscBool          usecprow = a->compressedrow.use;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
  #pragma disjoint(*x, *y, *aa)
#endif

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &a_a));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(yy, &y));
  if (usecprow) { /* use compressed row format */
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    for (i = 0; i < m; i++) {
      n   = ii[i + 1] - ii[i];
      aj  = a->j + ii[i];
      aa  = a_a + ii[i];
      sum = 0.0;
      nonzerorow += (n > 0);
      PetscSparseDenseMaxDot(sum, x, aa, aj, n);
      /* for (j=0; j<n; j++) sum += (*aa++)*x[*aj++]; */
      y[*ridx++] = sum;
    }
  } else { /* do not use compressed row format */
    ii = a->i;
    for (i = 0; i < m; i++) {
      n   = ii[i + 1] - ii[i];
      aj  = a->j + ii[i];
      aa  = a_a + ii[i];
      sum = 0.0;
      nonzerorow += (n > 0);
      PetscSparseDenseMaxDot(sum, x, aa, aj, n);
      y[i] = sum;
    }
  }
  PetscCall(PetscLogFlops(2.0 * a->nz - nonzerorow));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &a_a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// HACK!!!!! Used by src/mat/tests/ex170.c
PETSC_EXTERN PetscErrorCode MatMultAddMax_SeqAIJ(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscScalar       *y, *z;
  const PetscScalar *x;
  const MatScalar   *aa, *a_a;
  PetscInt           m = A->rmap->n, *aj, *ii;
  PetscInt           n, i, *ridx = NULL;
  PetscScalar        sum;
  PetscBool          usecprow = a->compressedrow.use;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &a_a));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArrayPair(yy, zz, &y, &z));
  if (usecprow) { /* use compressed row format */
    if (zz != yy) PetscCall(PetscArraycpy(z, y, m));
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    for (i = 0; i < m; i++) {
      n   = ii[i + 1] - ii[i];
      aj  = a->j + ii[i];
      aa  = a_a + ii[i];
      sum = y[*ridx];
      PetscSparseDenseMaxDot(sum, x, aa, aj, n);
      z[*ridx++] = sum;
    }
  } else { /* do not use compressed row format */
    ii = a->i;
    for (i = 0; i < m; i++) {
      n   = ii[i + 1] - ii[i];
      aj  = a->j + ii[i];
      aa  = a_a + ii[i];
      sum = y[i];
      PetscSparseDenseMaxDot(sum, x, aa, aj, n);
      z[i] = sum;
    }
  }
  PetscCall(PetscLogFlops(2.0 * a->nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArrayPair(yy, zz, &y, &z));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &a_a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/mat/impls/aij/seq/ftn-kernels/fmultadd.h>
PetscErrorCode MatMultAdd_SeqAIJ(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscScalar       *y, *z;
  const PetscScalar *x;
  const MatScalar   *a_a;
  const PetscInt    *ii, *ridx = NULL;
  PetscInt           m        = A->rmap->n;
  PetscBool          usecprow = a->compressedrow.use;

  PetscFunctionBegin;
  if (a->inode.use && a->inode.checked) {
    PetscCall(MatMultAdd_SeqAIJ_Inode(A, xx, yy, zz));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatSeqAIJGetArrayRead(A, &a_a));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArrayPair(yy, zz, &y, &z));
  if (usecprow) { /* use compressed row format */
    if (zz != yy) PetscCall(PetscArraycpy(z, y, m));
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    for (PetscInt i = 0; i < m; i++) {
      PetscInt           n   = ii[i + 1] - ii[i];
      const PetscInt    *aj  = a->j + ii[i];
      const PetscScalar *aa  = a_a + ii[i];
      PetscScalar        sum = y[*ridx];
      PetscSparseDensePlusDot(sum, x, aa, aj, n);
      z[*ridx++] = sum;
    }
  } else { /* do not use compressed row format */
    ii = a->i;
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
    fortranmultaddaij_(&m, x, ii, a->j, a_a, y, z);
#else
    PetscPragmaUseOMPKernels(parallel for)
    for (PetscInt i = 0; i < m; i++) {
      PetscInt           n   = ii[i + 1] - ii[i];
      const PetscInt    *aj  = a->j + ii[i];
      const PetscScalar *aa  = a_a + ii[i];
      PetscScalar        sum = y[i];
      PetscSparseDensePlusDot(sum, x, aa, aj, n);
      z[i] = sum;
    }
#endif
  }
  PetscCall(PetscLogFlops(2.0 * a->nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArrayPair(yy, zz, &y, &z));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &a_a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Adds diagonal pointers to sparse matrix nonzero structure.
*/
PetscErrorCode MatMarkDiagonal_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt    i, j, m = A->rmap->n;
  PetscBool   alreadySet = PETSC_TRUE;

  PetscFunctionBegin;
  if (!a->diag) {
    PetscCall(PetscMalloc1(m, &a->diag));
    alreadySet = PETSC_FALSE;
  }
  for (i = 0; i < A->rmap->n; i++) {
    /* If A's diagonal is already correctly set, this fast track enables cheap and repeated MatMarkDiagonal_SeqAIJ() calls */
    if (alreadySet) {
      PetscInt pos = a->diag[i];
      if (pos >= a->i[i] && pos < a->i[i + 1] && a->j[pos] == i) continue;
    }

    a->diag[i] = a->i[i + 1];
    for (j = a->i[i]; j < a->i[i + 1]; j++) {
      if (a->j[j] == i) {
        a->diag[i] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatShift_SeqAIJ(Mat A, PetscScalar v)
{
  Mat_SeqAIJ     *a    = (Mat_SeqAIJ *)A->data;
  const PetscInt *diag = (const PetscInt *)a->diag;
  const PetscInt *ii   = (const PetscInt *)a->i;
  PetscInt        i, *mdiag = NULL;
  PetscInt        cnt = 0; /* how many diagonals are missing */

  PetscFunctionBegin;
  if (!A->preallocated || !a->nz) {
    PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
    PetscCall(MatShift_Basic(A, v));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (a->diagonaldense) {
    cnt = 0;
  } else {
    PetscCall(PetscCalloc1(A->rmap->n, &mdiag));
    for (i = 0; i < A->rmap->n; i++) {
      if (i < A->cmap->n && diag[i] >= ii[i + 1]) { /* 'out of range' rows never have diagonals */
        cnt++;
        mdiag[i] = 1;
      }
    }
  }
  if (!cnt) {
    PetscCall(MatShift_Basic(A, v));
  } else {
    PetscScalar       *olda = a->a; /* preserve pointers to current matrix nonzeros structure and values */
    PetscInt          *oldj = a->j, *oldi = a->i;
    PetscBool          free_a = a->free_a, free_ij = a->free_ij;
    const PetscScalar *Aa;

    PetscCall(MatSeqAIJGetArrayRead(A, &Aa)); // sync the host
    PetscCall(MatSeqAIJRestoreArrayRead(A, &Aa));

    a->a = NULL;
    a->j = NULL;
    a->i = NULL;
    /* increase the values in imax for each row where a diagonal is being inserted then reallocate the matrix data structures */
    for (i = 0; i < PetscMin(A->rmap->n, A->cmap->n); i++) a->imax[i] += mdiag[i];
    PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(A, 0, a->imax));

    /* copy old values into new matrix data structure */
    for (i = 0; i < A->rmap->n; i++) {
      PetscCall(MatSetValues(A, 1, &i, a->imax[i] - mdiag[i], &oldj[oldi[i]], &olda[oldi[i]], ADD_VALUES));
      if (i < A->cmap->n) PetscCall(MatSetValue(A, i, i, v, ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    if (free_a) PetscCall(PetscShmgetDeallocateArray((void **)&olda));
    if (free_ij) PetscCall(PetscShmgetDeallocateArray((void **)&oldj));
    if (free_ij) PetscCall(PetscShmgetDeallocateArray((void **)&oldi));
  }
  PetscCall(PetscFree(mdiag));
  a->diagonaldense = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Checks for missing diagonals
*/
PetscErrorCode MatMissingDiagonal_SeqAIJ(Mat A, PetscBool *missing, PetscInt *d)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  PetscInt   *diag, *ii = a->i, i;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !ii) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    PetscCall(PetscInfo(A, "Matrix has no entries therefore is missing diagonal\n"));
  } else {
    PetscInt n;
    n    = PetscMin(A->rmap->n, A->cmap->n);
    diag = a->diag;
    for (i = 0; i < n; i++) {
      if (diag[i] >= ii[i + 1]) {
        *missing = PETSC_TRUE;
        if (d) *d = i;
        PetscCall(PetscInfo(A, "Matrix is missing diagonal number %" PetscInt_FMT "\n", i));
        break;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscblaslapack.h>
#include <petsc/private/kernels/blockinvert.h>

/*
    Note that values is allocated externally by the PC and then passed into this routine
*/
static PetscErrorCode MatInvertVariableBlockDiagonal_SeqAIJ(Mat A, PetscInt nblocks, const PetscInt *bsizes, PetscScalar *diag)
{
  PetscInt        n = A->rmap->n, i, ncnt = 0, *indx, j, bsizemax = 0, *v_pivots;
  PetscBool       allowzeropivot, zeropivotdetected = PETSC_FALSE;
  const PetscReal shift = 0.0;
  PetscInt        ipvt[5];
  PetscCount      flops = 0;
  PetscScalar     work[25], *v_work;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  for (i = 0; i < nblocks; i++) ncnt += bsizes[i];
  PetscCheck(ncnt == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Total blocksizes %" PetscInt_FMT " doesn't match number matrix rows %" PetscInt_FMT, ncnt, n);
  for (i = 0; i < nblocks; i++) bsizemax = PetscMax(bsizemax, bsizes[i]);
  PetscCall(PetscMalloc1(bsizemax, &indx));
  if (bsizemax > 7) PetscCall(PetscMalloc2(bsizemax, &v_work, bsizemax, &v_pivots));
  ncnt = 0;
  for (i = 0; i < nblocks; i++) {
    for (j = 0; j < bsizes[i]; j++) indx[j] = ncnt + j;
    PetscCall(MatGetValues(A, bsizes[i], indx, bsizes[i], indx, diag));
    switch (bsizes[i]) {
    case 1:
      *diag = 1.0 / (*diag);
      break;
    case 2:
      PetscCall(PetscKernel_A_gets_inverse_A_2(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_2(diag));
      break;
    case 3:
      PetscCall(PetscKernel_A_gets_inverse_A_3(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_3(diag));
      break;
    case 4:
      PetscCall(PetscKernel_A_gets_inverse_A_4(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_4(diag));
      break;
    case 5:
      PetscCall(PetscKernel_A_gets_inverse_A_5(diag, ipvt, work, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_5(diag));
      break;
    case 6:
      PetscCall(PetscKernel_A_gets_inverse_A_6(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_6(diag));
      break;
    case 7:
      PetscCall(PetscKernel_A_gets_inverse_A_7(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_7(diag));
      break;
    default:
      PetscCall(PetscKernel_A_gets_inverse_A(bsizes[i], diag, v_pivots, v_work, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_N(diag, bsizes[i]));
    }
    ncnt += bsizes[i];
    diag += bsizes[i] * bsizes[i];
    flops += 2 * PetscPowInt64(bsizes[i], 3) / 3;
  }
  PetscCall(PetscLogFlops(flops));
  if (bsizemax > 7) PetscCall(PetscFree2(v_work, v_pivots));
  PetscCall(PetscFree(indx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Negative shift indicates do not generate an error if there is a zero diagonal, just invert it anyways
*/
static PetscErrorCode MatInvertDiagonal_SeqAIJ(Mat A, PetscScalar omega, PetscScalar fshift)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  PetscInt         i, *diag, m = A->rmap->n;
  const MatScalar *v;
  PetscScalar     *idiag, *mdiag;

  PetscFunctionBegin;
  if (a->idiagvalid) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatMarkDiagonal_SeqAIJ(A));
  diag = a->diag;
  if (!a->idiag) { PetscCall(PetscMalloc3(m, &a->idiag, m, &a->mdiag, m, &a->ssor_work)); }

  mdiag = a->mdiag;
  idiag = a->idiag;
  PetscCall(MatSeqAIJGetArrayRead(A, &v));
  if (omega == 1.0 && PetscRealPart(fshift) <= 0.0) {
    for (i = 0; i < m; i++) {
      mdiag[i] = v[diag[i]];
      if (!PetscAbsScalar(mdiag[i])) { /* zero diagonal */
        if (PetscRealPart(fshift)) {
          PetscCall(PetscInfo(A, "Zero diagonal on row %" PetscInt_FMT "\n", i));
          A->factorerrortype             = MAT_FACTOR_NUMERIC_ZEROPIVOT;
          A->factorerror_zeropivot_value = 0.0;
          A->factorerror_zeropivot_row   = i;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Zero diagonal on row %" PetscInt_FMT, i);
      }
      idiag[i] = 1.0 / v[diag[i]];
    }
    PetscCall(PetscLogFlops(m));
  } else {
    for (i = 0; i < m; i++) {
      mdiag[i] = v[diag[i]];
      idiag[i] = omega / (fshift + v[diag[i]]);
    }
    PetscCall(PetscLogFlops(2.0 * m));
  }
  a->idiagvalid = PETSC_TRUE;
  PetscCall(MatSeqAIJRestoreArrayRead(A, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSOR_SeqAIJ(Mat A, Vec bb, PetscReal omega, MatSORType flag, PetscReal fshift, PetscInt its, PetscInt lits, Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscScalar       *x, d, sum, *t, scale;
  const MatScalar   *v, *idiag = NULL, *mdiag, *aa;
  const PetscScalar *b, *bs, *xb, *ts;
  PetscInt           n, m = A->rmap->n, i;
  const PetscInt    *idx, *diag;

  PetscFunctionBegin;
  if (a->inode.use && a->inode.checked && omega == 1.0 && fshift == 0.0) {
    PetscCall(MatSOR_SeqAIJ_Inode(A, bb, omega, flag, fshift, its, lits, xx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  its = its * lits;

  if (fshift != a->fshift || omega != a->omega) a->idiagvalid = PETSC_FALSE; /* must recompute idiag[] */
  if (!a->idiagvalid) PetscCall(MatInvertDiagonal_SeqAIJ(A, omega, fshift));
  a->fshift = fshift;
  a->omega  = omega;

  diag  = a->diag;
  t     = a->ssor_work;
  idiag = a->idiag;
  mdiag = a->mdiag;

  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  PetscCall(VecGetArray(xx, &x));
  PetscCall(VecGetArrayRead(bb, &b));
  /* We count flops by assuming the upper triangular and lower triangular parts have the same number of nonzeros */
  if (flag == SOR_APPLY_UPPER) {
    /* apply (U + D/omega) to the vector */
    bs = b;
    for (i = 0; i < m; i++) {
      d   = fshift + mdiag[i];
      n   = a->i[i + 1] - diag[i] - 1;
      idx = a->j + diag[i] + 1;
      v   = aa + diag[i] + 1;
      sum = b[i] * d / omega;
      PetscSparseDensePlusDot(sum, bs, v, idx, n);
      x[i] = sum;
    }
    PetscCall(VecRestoreArray(xx, &x));
    PetscCall(VecRestoreArrayRead(bb, &b));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
    PetscCall(PetscLogFlops(a->nz));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCheck(flag != SOR_APPLY_LOWER, PETSC_COMM_SELF, PETSC_ERR_SUP, "SOR_APPLY_LOWER is not implemented");
  if (flag & SOR_EISENSTAT) {
    /* Let  A = L + U + D; where L is lower triangular,
    U is upper triangular, E = D/omega; This routine applies

            (L + E)^{-1} A (U + E)^{-1}

    to a vector efficiently using Eisenstat's trick.
    */
    scale = (2.0 / omega) - 1.0;

    /*  x = (E + U)^{-1} b */
    for (i = m - 1; i >= 0; i--) {
      n   = a->i[i + 1] - diag[i] - 1;
      idx = a->j + diag[i] + 1;
      v   = aa + diag[i] + 1;
      sum = b[i];
      PetscSparseDenseMinusDot(sum, x, v, idx, n);
      x[i] = sum * idiag[i];
    }

    /*  t = b - (2*E - D)x */
    v = aa;
    for (i = 0; i < m; i++) t[i] = b[i] - scale * (v[*diag++]) * x[i];

    /*  t = (E + L)^{-1}t */
    ts   = t;
    diag = a->diag;
    for (i = 0; i < m; i++) {
      n   = diag[i] - a->i[i];
      idx = a->j + a->i[i];
      v   = aa + a->i[i];
      sum = t[i];
      PetscSparseDenseMinusDot(sum, ts, v, idx, n);
      t[i] = sum * idiag[i];
      /*  x = x + t */
      x[i] += t[i];
    }

    PetscCall(PetscLogFlops(6.0 * m - 1 + 2.0 * a->nz));
    PetscCall(VecRestoreArray(xx, &x));
    PetscCall(VecRestoreArrayRead(bb, &b));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i = 0; i < m; i++) {
        n   = diag[i] - a->i[i];
        idx = a->j + a->i[i];
        v   = aa + a->i[i];
        sum = b[i];
        PetscSparseDenseMinusDot(sum, x, v, idx, n);
        t[i] = sum;
        x[i] = sum * idiag[i];
      }
      xb = t;
      PetscCall(PetscLogFlops(a->nz));
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      for (i = m - 1; i >= 0; i--) {
        n   = a->i[i + 1] - diag[i] - 1;
        idx = a->j + diag[i] + 1;
        v   = aa + diag[i] + 1;
        sum = xb[i];
        PetscSparseDenseMinusDot(sum, x, v, idx, n);
        if (xb == b) {
          x[i] = sum * idiag[i];
        } else {
          x[i] = (1 - omega) * x[i] + sum * idiag[i]; /* omega in idiag */
        }
      }
      PetscCall(PetscLogFlops(a->nz)); /* assumes 1/2 in upper */
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i = 0; i < m; i++) {
        /* lower */
        n   = diag[i] - a->i[i];
        idx = a->j + a->i[i];
        v   = aa + a->i[i];
        sum = b[i];
        PetscSparseDenseMinusDot(sum, x, v, idx, n);
        t[i] = sum; /* save application of the lower-triangular part */
        /* upper */
        n   = a->i[i + 1] - diag[i] - 1;
        idx = a->j + diag[i] + 1;
        v   = aa + diag[i] + 1;
        PetscSparseDenseMinusDot(sum, x, v, idx, n);
        x[i] = (1. - omega) * x[i] + sum * idiag[i]; /* omega in idiag */
      }
      xb = t;
      PetscCall(PetscLogFlops(2.0 * a->nz));
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      for (i = m - 1; i >= 0; i--) {
        sum = xb[i];
        if (xb == b) {
          /* whole matrix (no checkpointing available) */
          n   = a->i[i + 1] - a->i[i];
          idx = a->j + a->i[i];
          v   = aa + a->i[i];
          PetscSparseDenseMinusDot(sum, x, v, idx, n);
          x[i] = (1. - omega) * x[i] + (sum + mdiag[i] * x[i]) * idiag[i];
        } else { /* lower-triangular part has been saved, so only apply upper-triangular */
          n   = a->i[i + 1] - diag[i] - 1;
          idx = a->j + diag[i] + 1;
          v   = aa + diag[i] + 1;
          PetscSparseDenseMinusDot(sum, x, v, idx, n);
          x[i] = (1. - omega) * x[i] + sum * idiag[i]; /* omega in idiag */
        }
      }
      if (xb == b) {
        PetscCall(PetscLogFlops(2.0 * a->nz));
      } else {
        PetscCall(PetscLogFlops(a->nz)); /* assumes 1/2 in upper */
      }
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetInfo_SeqAIJ(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  info->block_size   = 1.0;
  info->nz_allocated = a->maxnz;
  info->nz_used      = a->nz;
  info->nz_unneeded  = (a->maxnz - a->nz);
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

static PetscErrorCode MatZeroRows_SeqAIJ(Mat A, PetscInt N, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscInt           i, m = A->rmap->n - 1;
  const PetscScalar *xx;
  PetscScalar       *bb, *aa;
  PetscInt           d = 0;

  PetscFunctionBegin;
  if (x && b) {
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(b, &bb));
    for (i = 0; i < N; i++) {
      PetscCheck(rows[i] >= 0 && rows[i] <= m, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "row %" PetscInt_FMT " out of range", rows[i]);
      if (rows[i] >= A->cmap->n) continue;
      bb[rows[i]] = diag * xx[rows[i]];
    }
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(b, &bb));
  }

  PetscCall(MatSeqAIJGetArray(A, &aa));
  if (a->keepnonzeropattern) {
    for (i = 0; i < N; i++) {
      PetscCheck(rows[i] >= 0 && rows[i] <= m, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "row %" PetscInt_FMT " out of range", rows[i]);
      PetscCall(PetscArrayzero(&aa[a->i[rows[i]]], a->ilen[rows[i]]));
    }
    if (diag != 0.0) {
      for (i = 0; i < N; i++) {
        d = rows[i];
        if (rows[i] >= A->cmap->n) continue;
        PetscCheck(a->diag[d] < a->i[d + 1], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry in the zeroed row %" PetscInt_FMT, d);
      }
      for (i = 0; i < N; i++) {
        if (rows[i] >= A->cmap->n) continue;
        aa[a->diag[rows[i]]] = diag;
      }
    }
  } else {
    if (diag != 0.0) {
      for (i = 0; i < N; i++) {
        PetscCheck(rows[i] >= 0 && rows[i] <= m, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "row %" PetscInt_FMT " out of range", rows[i]);
        if (a->ilen[rows[i]] > 0) {
          if (rows[i] >= A->cmap->n) {
            a->ilen[rows[i]] = 0;
          } else {
            a->ilen[rows[i]]    = 1;
            aa[a->i[rows[i]]]   = diag;
            a->j[a->i[rows[i]]] = rows[i];
          }
        } else if (rows[i] < A->cmap->n) { /* in case row was completely empty */
          PetscCall(MatSetValues_SeqAIJ(A, 1, &rows[i], 1, &rows[i], &diag, INSERT_VALUES));
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        PetscCheck(rows[i] >= 0 && rows[i] <= m, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "row %" PetscInt_FMT " out of range", rows[i]);
        a->ilen[rows[i]] = 0;
      }
    }
    A->nonzerostate++;
  }
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscUseTypeMethod(A, assemblyend, MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroRowsColumns_SeqAIJ(Mat A, PetscInt N, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscInt           i, j, m = A->rmap->n - 1, d = 0;
  PetscBool          missing, *zeroed, vecs = PETSC_FALSE;
  const PetscScalar *xx;
  PetscScalar       *bb, *aa;

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatSeqAIJGetArray(A, &aa));
  if (x && b) {
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(b, &bb));
    vecs = PETSC_TRUE;
  }
  PetscCall(PetscCalloc1(A->rmap->n, &zeroed));
  for (i = 0; i < N; i++) {
    PetscCheck(rows[i] >= 0 && rows[i] <= m, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "row %" PetscInt_FMT " out of range", rows[i]);
    PetscCall(PetscArrayzero(PetscSafePointerPlusOffset(aa, a->i[rows[i]]), a->ilen[rows[i]]));

    zeroed[rows[i]] = PETSC_TRUE;
  }
  for (i = 0; i < A->rmap->n; i++) {
    if (!zeroed[i]) {
      for (j = a->i[i]; j < a->i[i + 1]; j++) {
        if (a->j[j] < A->rmap->n && zeroed[a->j[j]]) {
          if (vecs) bb[i] -= aa[j] * xx[a->j[j]];
          aa[j] = 0.0;
        }
      }
    } else if (vecs && i < A->cmap->N) bb[i] = diag * xx[i];
  }
  if (x && b) {
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(b, &bb));
  }
  PetscCall(PetscFree(zeroed));
  if (diag != 0.0) {
    PetscCall(MatMissingDiagonal_SeqAIJ(A, &missing, &d));
    if (missing) {
      for (i = 0; i < N; i++) {
        if (rows[i] >= A->cmap->N) continue;
        PetscCheck(!a->nonew || rows[i] < d, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry in row %" PetscInt_FMT " (%" PetscInt_FMT ")", d, rows[i]);
        PetscCall(MatSetValues_SeqAIJ(A, 1, &rows[i], 1, &rows[i], &diag, INSERT_VALUES));
      }
    } else {
      for (i = 0; i < N; i++) aa[a->diag[rows[i]]] = diag;
    }
  }
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscUseTypeMethod(A, assemblyend, MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetRow_SeqAIJ(Mat A, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  *nz = a->i[row + 1] - a->i[row];
  if (v) *v = PetscSafePointerPlusOffset((PetscScalar *)aa, a->i[row]);
  if (idx) {
    if (*nz && a->j) *idx = a->j + a->i[row];
    else *idx = NULL;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatRestoreRow_SeqAIJ(Mat A, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNorm_SeqAIJ(Mat A, NormType type, PetscReal *nrm)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  const MatScalar *v;
  PetscReal        sum = 0.0;
  PetscInt         i, j;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &v));
  if (type == NORM_FROBENIUS) {
#if defined(PETSC_USE_REAL___FP16)
    PetscBLASInt one = 1, nz = a->nz;
    PetscCallBLAS("BLASnrm2", *nrm = BLASnrm2_(&nz, v, &one));
#else
    for (i = 0; i < a->nz; i++) {
      sum += PetscRealPart(PetscConj(*v) * (*v));
      v++;
    }
    *nrm = PetscSqrtReal(sum);
#endif
    PetscCall(PetscLogFlops(2.0 * a->nz));
  } else if (type == NORM_1) {
    PetscReal *tmp;
    PetscInt  *jj = a->j;
    PetscCall(PetscCalloc1(A->cmap->n + 1, &tmp));
    *nrm = 0.0;
    for (j = 0; j < a->nz; j++) {
      tmp[*jj++] += PetscAbsScalar(*v);
      v++;
    }
    for (j = 0; j < A->cmap->n; j++) {
      if (tmp[j] > *nrm) *nrm = tmp[j];
    }
    PetscCall(PetscFree(tmp));
    PetscCall(PetscLogFlops(PetscMax(a->nz - 1, 0)));
  } else if (type == NORM_INFINITY) {
    *nrm = 0.0;
    for (j = 0; j < A->rmap->n; j++) {
      const PetscScalar *v2 = PetscSafePointerPlusOffset(v, a->i[j]);
      sum                   = 0.0;
      for (i = 0; i < a->i[j + 1] - a->i[j]; i++) {
        sum += PetscAbsScalar(*v2);
        v2++;
      }
      if (sum > *nrm) *nrm = sum;
    }
    PetscCall(PetscLogFlops(PetscMax(a->nz - 1, 0)));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for two norm");
  PetscCall(MatSeqAIJRestoreArrayRead(A, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatIsTranspose_SeqAIJ(Mat A, Mat B, PetscReal tol, PetscBool *f)
{
  Mat_SeqAIJ      *aij = (Mat_SeqAIJ *)A->data, *bij = (Mat_SeqAIJ *)B->data;
  PetscInt        *adx, *bdx, *aii, *bii, *aptr, *bptr;
  const MatScalar *va, *vb;
  PetscInt         ma, na, mb, nb, i;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &ma, &na));
  PetscCall(MatGetSize(B, &mb, &nb));
  if (ma != nb || na != mb) {
    *f = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatSeqAIJGetArrayRead(A, &va));
  PetscCall(MatSeqAIJGetArrayRead(B, &vb));
  aii = aij->i;
  bii = bij->i;
  adx = aij->j;
  bdx = bij->j;
  PetscCall(PetscMalloc1(ma, &aptr));
  PetscCall(PetscMalloc1(mb, &bptr));
  for (i = 0; i < ma; i++) aptr[i] = aii[i];
  for (i = 0; i < mb; i++) bptr[i] = bii[i];

  *f = PETSC_TRUE;
  for (i = 0; i < ma; i++) {
    while (aptr[i] < aii[i + 1]) {
      PetscInt    idc, idr;
      PetscScalar vc, vr;
      /* column/row index/value */
      idc = adx[aptr[i]];
      idr = bdx[bptr[idc]];
      vc  = va[aptr[i]];
      vr  = vb[bptr[idc]];
      if (i != idr || PetscAbsScalar(vc - vr) > tol) {
        *f = PETSC_FALSE;
        goto done;
      } else {
        aptr[i]++;
        if (B || i != idc) bptr[idc]++;
      }
    }
  }
done:
  PetscCall(PetscFree(aptr));
  PetscCall(PetscFree(bptr));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &va));
  PetscCall(MatSeqAIJRestoreArrayRead(B, &vb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatIsHermitianTranspose_SeqAIJ(Mat A, Mat B, PetscReal tol, PetscBool *f)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data, *bij = (Mat_SeqAIJ *)B->data;
  PetscInt   *adx, *bdx, *aii, *bii, *aptr, *bptr;
  MatScalar  *va, *vb;
  PetscInt    ma, na, mb, nb, i;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &ma, &na));
  PetscCall(MatGetSize(B, &mb, &nb));
  if (ma != nb || na != mb) {
    *f = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  aii = aij->i;
  bii = bij->i;
  adx = aij->j;
  bdx = bij->j;
  va  = aij->a;
  vb  = bij->a;
  PetscCall(PetscMalloc1(ma, &aptr));
  PetscCall(PetscMalloc1(mb, &bptr));
  for (i = 0; i < ma; i++) aptr[i] = aii[i];
  for (i = 0; i < mb; i++) bptr[i] = bii[i];

  *f = PETSC_TRUE;
  for (i = 0; i < ma; i++) {
    while (aptr[i] < aii[i + 1]) {
      PetscInt    idc, idr;
      PetscScalar vc, vr;
      /* column/row index/value */
      idc = adx[aptr[i]];
      idr = bdx[bptr[idc]];
      vc  = va[aptr[i]];
      vr  = vb[bptr[idc]];
      if (i != idr || PetscAbsScalar(vc - PetscConj(vr)) > tol) {
        *f = PETSC_FALSE;
        goto done;
      } else {
        aptr[i]++;
        if (B || i != idc) bptr[idc]++;
      }
    }
  }
done:
  PetscCall(PetscFree(aptr));
  PetscCall(PetscFree(bptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDiagonalScale_SeqAIJ(Mat A, Vec ll, Vec rr)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  const PetscScalar *l, *r;
  PetscScalar        x;
  MatScalar         *v;
  PetscInt           i, j, m = A->rmap->n, n = A->cmap->n, M, nz = a->nz;
  const PetscInt    *jj;

  PetscFunctionBegin;
  if (ll) {
    /* The local size is used so that VecMPI can be passed to this routine
       by MatDiagonalScale_MPIAIJ */
    PetscCall(VecGetLocalSize(ll, &m));
    PetscCheck(m == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Left scaling vector wrong length");
    PetscCall(VecGetArrayRead(ll, &l));
    PetscCall(MatSeqAIJGetArray(A, &v));
    for (i = 0; i < m; i++) {
      x = l[i];
      M = a->i[i + 1] - a->i[i];
      for (j = 0; j < M; j++) (*v++) *= x;
    }
    PetscCall(VecRestoreArrayRead(ll, &l));
    PetscCall(PetscLogFlops(nz));
    PetscCall(MatSeqAIJRestoreArray(A, &v));
  }
  if (rr) {
    PetscCall(VecGetLocalSize(rr, &n));
    PetscCheck(n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Right scaling vector wrong length");
    PetscCall(VecGetArrayRead(rr, &r));
    PetscCall(MatSeqAIJGetArray(A, &v));
    jj = a->j;
    for (i = 0; i < nz; i++) (*v++) *= r[*jj++];
    PetscCall(MatSeqAIJRestoreArray(A, &v));
    PetscCall(VecRestoreArrayRead(rr, &r));
    PetscCall(PetscLogFlops(nz));
  }
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateSubMatrix_SeqAIJ(Mat A, IS isrow, IS iscol, PetscInt csize, MatReuse scall, Mat *B)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data, *c;
  PetscInt          *smap, i, k, kstart, kend, oldcols = A->cmap->n, *lens;
  PetscInt           row, mat_i, *mat_j, tcol, first, step, *mat_ilen, sum, lensi;
  const PetscInt    *irow, *icol;
  const PetscScalar *aa;
  PetscInt           nrows, ncols;
  PetscInt          *starts, *j_new, *i_new, *aj = a->j, *ai = a->i, ii, *ailen = a->ilen;
  MatScalar         *a_new, *mat_a, *c_a;
  Mat                C;
  PetscBool          stride;

  PetscFunctionBegin;
  PetscCall(ISGetIndices(isrow, &irow));
  PetscCall(ISGetLocalSize(isrow, &nrows));
  PetscCall(ISGetLocalSize(iscol, &ncols));

  PetscCall(PetscObjectTypeCompare((PetscObject)iscol, ISSTRIDE, &stride));
  if (stride) {
    PetscCall(ISStrideGetInfo(iscol, &first, &step));
  } else {
    first = 0;
    step  = 0;
  }
  if (stride && step == 1) {
    /* special case of contiguous rows */
    PetscCall(PetscMalloc2(nrows, &lens, nrows, &starts));
    /* loop over new rows determining lens and starting points */
    for (i = 0; i < nrows; i++) {
      kstart    = ai[irow[i]];
      kend      = kstart + ailen[irow[i]];
      starts[i] = kstart;
      for (k = kstart; k < kend; k++) {
        if (aj[k] >= first) {
          starts[i] = k;
          break;
        }
      }
      sum = 0;
      while (k < kend) {
        if (aj[k++] >= first + ncols) break;
        sum++;
      }
      lens[i] = sum;
    }
    /* create submatrix */
    if (scall == MAT_REUSE_MATRIX) {
      PetscInt n_cols, n_rows;
      PetscCall(MatGetSize(*B, &n_rows, &n_cols));
      PetscCheck(n_rows == nrows && n_cols == ncols, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Reused submatrix wrong size");
      PetscCall(MatZeroEntries(*B));
      C = *B;
    } else {
      PetscInt rbs, cbs;
      PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &C));
      PetscCall(MatSetSizes(C, nrows, ncols, PETSC_DETERMINE, PETSC_DETERMINE));
      PetscCall(ISGetBlockSize(isrow, &rbs));
      PetscCall(ISGetBlockSize(iscol, &cbs));
      PetscCall(MatSetBlockSizes(C, rbs, cbs));
      PetscCall(MatSetType(C, ((PetscObject)A)->type_name));
      PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(C, 0, lens));
    }
    c = (Mat_SeqAIJ *)C->data;

    /* loop over rows inserting into submatrix */
    PetscCall(MatSeqAIJGetArrayWrite(C, &a_new)); // Not 'a_new = c->a-new', since that raw usage ignores offload state of C
    j_new = c->j;
    i_new = c->i;
    PetscCall(MatSeqAIJGetArrayRead(A, &aa));
    for (i = 0; i < nrows; i++) {
      ii    = starts[i];
      lensi = lens[i];
      if (lensi) {
        for (k = 0; k < lensi; k++) *j_new++ = aj[ii + k] - first;
        PetscCall(PetscArraycpy(a_new, aa + starts[i], lensi));
        a_new += lensi;
      }
      i_new[i + 1] = i_new[i] + lensi;
      c->ilen[i]   = lensi;
    }
    PetscCall(MatSeqAIJRestoreArrayWrite(C, &a_new)); // Set C's offload state properly
    PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
    PetscCall(PetscFree2(lens, starts));
  } else {
    PetscCall(ISGetIndices(iscol, &icol));
    PetscCall(PetscCalloc1(oldcols, &smap));
    PetscCall(PetscMalloc1(1 + nrows, &lens));
    for (i = 0; i < ncols; i++) {
      PetscCheck(icol[i] < oldcols, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requesting column beyond largest column icol[%" PetscInt_FMT "] %" PetscInt_FMT " >= A->cmap->n %" PetscInt_FMT, i, icol[i], oldcols);
      smap[icol[i]] = i + 1;
    }

    /* determine lens of each row */
    for (i = 0; i < nrows; i++) {
      kstart  = ai[irow[i]];
      kend    = kstart + a->ilen[irow[i]];
      lens[i] = 0;
      for (k = kstart; k < kend; k++) {
        if (smap[aj[k]]) lens[i]++;
      }
    }
    /* Create and fill new matrix */
    if (scall == MAT_REUSE_MATRIX) {
      PetscBool equal;

      c = (Mat_SeqAIJ *)((*B)->data);
      PetscCheck((*B)->rmap->n == nrows && (*B)->cmap->n == ncols, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot reuse matrix. wrong size");
      PetscCall(PetscArraycmp(c->ilen, lens, (*B)->rmap->n, &equal));
      PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot reuse matrix. wrong number of nonzeros");
      PetscCall(PetscArrayzero(c->ilen, (*B)->rmap->n));
      C = *B;
    } else {
      PetscInt rbs, cbs;
      PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &C));
      PetscCall(MatSetSizes(C, nrows, ncols, PETSC_DETERMINE, PETSC_DETERMINE));
      PetscCall(ISGetBlockSize(isrow, &rbs));
      PetscCall(ISGetBlockSize(iscol, &cbs));
      if (rbs > 1 || cbs > 1) PetscCall(MatSetBlockSizes(C, rbs, cbs));
      PetscCall(MatSetType(C, ((PetscObject)A)->type_name));
      PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(C, 0, lens));
    }
    PetscCall(MatSeqAIJGetArrayRead(A, &aa));

    c = (Mat_SeqAIJ *)C->data;
    PetscCall(MatSeqAIJGetArrayWrite(C, &c_a)); // Not 'c->a', since that raw usage ignores offload state of C
    for (i = 0; i < nrows; i++) {
      row      = irow[i];
      kstart   = ai[row];
      kend     = kstart + a->ilen[row];
      mat_i    = c->i[i];
      mat_j    = PetscSafePointerPlusOffset(c->j, mat_i);
      mat_a    = PetscSafePointerPlusOffset(c_a, mat_i);
      mat_ilen = c->ilen + i;
      for (k = kstart; k < kend; k++) {
        if ((tcol = smap[a->j[k]])) {
          *mat_j++ = tcol - 1;
          *mat_a++ = aa[k];
          (*mat_ilen)++;
        }
      }
    }
    PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
    /* Free work space */
    PetscCall(ISRestoreIndices(iscol, &icol));
    PetscCall(PetscFree(smap));
    PetscCall(PetscFree(lens));
    /* sort */
    for (i = 0; i < nrows; i++) {
      PetscInt ilen;

      mat_i = c->i[i];
      mat_j = PetscSafePointerPlusOffset(c->j, mat_i);
      mat_a = PetscSafePointerPlusOffset(c_a, mat_i);
      ilen  = c->ilen[i];
      PetscCall(PetscSortIntWithScalarArray(ilen, mat_j, mat_a));
    }
    PetscCall(MatSeqAIJRestoreArrayWrite(C, &c_a));
  }
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(C, A->boundtocpu));
#endif
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(ISRestoreIndices(isrow, &irow));
  *B = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetMultiProcBlock_SeqAIJ(Mat mat, MPI_Comm subComm, MatReuse scall, Mat *subMat)
{
  Mat B;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreate(subComm, &B));
    PetscCall(MatSetSizes(B, mat->rmap->n, mat->cmap->n, mat->rmap->n, mat->cmap->n));
    PetscCall(MatSetBlockSizesFromMats(B, mat, mat));
    PetscCall(MatSetType(B, MATSEQAIJ));
    PetscCall(MatDuplicateNoCreate_SeqAIJ(B, mat, MAT_COPY_VALUES, PETSC_TRUE));
    *subMat = B;
  } else {
    PetscCall(MatCopy_SeqAIJ(mat, *subMat, SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactor_SeqAIJ(Mat inA, IS row, IS col, const MatFactorInfo *info)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)inA->data;
  Mat         outA;
  PetscBool   row_identity, col_identity;

  PetscFunctionBegin;
  PetscCheck(info->levels == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only levels=0 supported for in-place ilu");

  PetscCall(ISIdentity(row, &row_identity));
  PetscCall(ISIdentity(col, &col_identity));

  outA             = inA;
  outA->factortype = MAT_FACTOR_LU;
  PetscCall(PetscFree(inA->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC, &inA->solvertype));

  PetscCall(PetscObjectReference((PetscObject)row));
  PetscCall(ISDestroy(&a->row));

  a->row = row;

  PetscCall(PetscObjectReference((PetscObject)col));
  PetscCall(ISDestroy(&a->col));

  a->col = col;

  /* Create the inverse permutation so that it can be used in MatLUFactorNumeric() */
  PetscCall(ISDestroy(&a->icol));
  PetscCall(ISInvertPermutation(col, PETSC_DECIDE, &a->icol));

  if (!a->solve_work) { /* this matrix may have been factored before */
    PetscCall(PetscMalloc1(inA->rmap->n + 1, &a->solve_work));
  }

  PetscCall(MatMarkDiagonal_SeqAIJ(inA));
  if (row_identity && col_identity) {
    PetscCall(MatLUFactorNumeric_SeqAIJ_inplace(outA, inA, info));
  } else {
    PetscCall(MatLUFactorNumeric_SeqAIJ_InplaceWithPerm(outA, inA, info));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatScale_SeqAIJ(Mat inA, PetscScalar alpha)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *)inA->data;
  PetscScalar *v;
  PetscBLASInt one = 1, bnz;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(inA, &v));
  PetscCall(PetscBLASIntCast(a->nz, &bnz));
  PetscCallBLAS("BLASscal", BLASscal_(&bnz, &alpha, v, &one));
  PetscCall(PetscLogFlops(a->nz));
  PetscCall(MatSeqAIJRestoreArray(inA, &v));
  PetscCall(MatSeqAIJInvalidateDiagonal(inA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroySubMatrix_Private(Mat_SubSppt *submatj)
{
  PetscInt i;

  PetscFunctionBegin;
  if (!submatj->id) { /* delete data that are linked only to submats[id=0] */
    PetscCall(PetscFree4(submatj->sbuf1, submatj->ptr, submatj->tmp, submatj->ctr));

    for (i = 0; i < submatj->nrqr; ++i) PetscCall(PetscFree(submatj->sbuf2[i]));
    PetscCall(PetscFree3(submatj->sbuf2, submatj->req_size, submatj->req_source1));

    if (submatj->rbuf1) {
      PetscCall(PetscFree(submatj->rbuf1[0]));
      PetscCall(PetscFree(submatj->rbuf1));
    }

    for (i = 0; i < submatj->nrqs; ++i) PetscCall(PetscFree(submatj->rbuf3[i]));
    PetscCall(PetscFree3(submatj->req_source2, submatj->rbuf2, submatj->rbuf3));
    PetscCall(PetscFree(submatj->pa));
  }

#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&submatj->rmap));
  if (submatj->cmap_loc) PetscCall(PetscFree(submatj->cmap_loc));
  PetscCall(PetscFree(submatj->rmap_loc));
#else
  PetscCall(PetscFree(submatj->rmap));
#endif

  if (!submatj->allcolumns) {
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscHMapIDestroy(&submatj->cmap));
#else
    PetscCall(PetscFree(submatj->cmap));
#endif
  }
  PetscCall(PetscFree(submatj->row2proc));

  PetscCall(PetscFree(submatj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroySubMatrix_SeqAIJ(Mat C)
{
  Mat_SeqAIJ  *c       = (Mat_SeqAIJ *)C->data;
  Mat_SubSppt *submatj = c->submatis1;

  PetscFunctionBegin;
  PetscCall((*submatj->destroy)(C));
  PetscCall(MatDestroySubMatrix_Private(submatj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Note this has code duplication with MatDestroySubMatrices_SeqBAIJ() */
static PetscErrorCode MatDestroySubMatrices_SeqAIJ(PetscInt n, Mat *mat[])
{
  PetscInt     i;
  Mat          C;
  Mat_SeqAIJ  *c;
  Mat_SubSppt *submatj;

  PetscFunctionBegin;
  for (i = 0; i < n; i++) {
    C       = (*mat)[i];
    c       = (Mat_SeqAIJ *)C->data;
    submatj = c->submatis1;
    if (submatj) {
      if (--((PetscObject)C)->refct <= 0) {
        PetscCall(PetscFree(C->factorprefix));
        PetscCall((*submatj->destroy)(C));
        PetscCall(MatDestroySubMatrix_Private(submatj));
        PetscCall(PetscFree(C->defaultvectype));
        PetscCall(PetscFree(C->defaultrandtype));
        PetscCall(PetscLayoutDestroy(&C->rmap));
        PetscCall(PetscLayoutDestroy(&C->cmap));
        PetscCall(PetscHeaderDestroy(&C));
      }
    } else {
      PetscCall(MatDestroy(&C));
    }
  }

  /* Destroy Dummy submatrices created for reuse */
  PetscCall(MatDestroySubMatrices_Dummy(n, mat));

  PetscCall(PetscFree(*mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_SeqAIJ(Mat A, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *B[])
{
  PetscInt i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) PetscCall(PetscCalloc1(n + 1, B));

  for (i = 0; i < n; i++) PetscCall(MatCreateSubMatrix_SeqAIJ(A, irow[i], icol[i], PETSC_DECIDE, scall, &(*B)[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatIncreaseOverlap_SeqAIJ(Mat A, PetscInt is_max, IS is[], PetscInt ov)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  PetscInt        row, i, j, k, l, ll, m, n, *nidx, isz, val;
  const PetscInt *idx;
  PetscInt        start, end, *ai, *aj, bs = A->rmap->bs == A->cmap->bs ? A->rmap->bs : 1;
  PetscBT         table;

  PetscFunctionBegin;
  m  = A->rmap->n / bs;
  ai = a->i;
  aj = a->j;

  PetscCheck(ov >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "illegal negative overlap value used");

  PetscCall(PetscMalloc1(m + 1, &nidx));
  PetscCall(PetscBTCreate(m, &table));

  for (i = 0; i < is_max; i++) {
    /* Initialize the two local arrays */
    isz = 0;
    PetscCall(PetscBTMemzero(m, table));

    /* Extract the indices, assume there can be duplicate entries */
    PetscCall(ISGetIndices(is[i], &idx));
    PetscCall(ISGetLocalSize(is[i], &n));

    if (bs > 1) {
      /* Enter these into the temp arrays. I.e., mark table[row], enter row into new index */
      for (j = 0; j < n; ++j) {
        if (!PetscBTLookupSet(table, idx[j] / bs)) nidx[isz++] = idx[j] / bs;
      }
      PetscCall(ISRestoreIndices(is[i], &idx));
      PetscCall(ISDestroy(&is[i]));

      k = 0;
      for (j = 0; j < ov; j++) { /* for each overlap */
        n = isz;
        for (; k < n; k++) { /* do only those rows in nidx[k], which are not done yet */
          for (ll = 0; ll < bs; ll++) {
            row   = bs * nidx[k] + ll;
            start = ai[row];
            end   = ai[row + 1];
            for (l = start; l < end; l++) {
              val = aj[l] / bs;
              if (!PetscBTLookupSet(table, val)) nidx[isz++] = val;
            }
          }
        }
      }
      PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, isz, nidx, PETSC_COPY_VALUES, is + i));
    } else {
      /* Enter these into the temp arrays. I.e., mark table[row], enter row into new index */
      for (j = 0; j < n; ++j) {
        if (!PetscBTLookupSet(table, idx[j])) nidx[isz++] = idx[j];
      }
      PetscCall(ISRestoreIndices(is[i], &idx));
      PetscCall(ISDestroy(&is[i]));

      k = 0;
      for (j = 0; j < ov; j++) { /* for each overlap */
        n = isz;
        for (; k < n; k++) { /* do only those rows in nidx[k], which are not done yet */
          row   = nidx[k];
          start = ai[row];
          end   = ai[row + 1];
          for (l = start; l < end; l++) {
            val = aj[l];
            if (!PetscBTLookupSet(table, val)) nidx[isz++] = val;
          }
        }
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, isz, nidx, PETSC_COPY_VALUES, is + i));
    }
  }
  PetscCall(PetscBTDestroy(&table));
  PetscCall(PetscFree(nidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPermute_SeqAIJ(Mat A, IS rowp, IS colp, Mat *B)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  PetscInt        i, nz = 0, m = A->rmap->n, n = A->cmap->n;
  const PetscInt *row, *col;
  PetscInt       *cnew, j, *lens;
  IS              icolp, irowp;
  PetscInt       *cwork = NULL;
  PetscScalar    *vwork = NULL;

  PetscFunctionBegin;
  PetscCall(ISInvertPermutation(rowp, PETSC_DECIDE, &irowp));
  PetscCall(ISGetIndices(irowp, &row));
  PetscCall(ISInvertPermutation(colp, PETSC_DECIDE, &icolp));
  PetscCall(ISGetIndices(icolp, &col));

  /* determine lengths of permuted rows */
  PetscCall(PetscMalloc1(m + 1, &lens));
  for (i = 0; i < m; i++) lens[row[i]] = a->i[i + 1] - a->i[i];
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, m, n, m, n));
  PetscCall(MatSetBlockSizesFromMats(*B, A, A));
  PetscCall(MatSetType(*B, ((PetscObject)A)->type_name));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*B, 0, lens));
  PetscCall(PetscFree(lens));

  PetscCall(PetscMalloc1(n, &cnew));
  for (i = 0; i < m; i++) {
    PetscCall(MatGetRow_SeqAIJ(A, i, &nz, &cwork, &vwork));
    for (j = 0; j < nz; j++) cnew[j] = col[cwork[j]];
    PetscCall(MatSetValues_SeqAIJ(*B, 1, &row[i], nz, cnew, vwork, INSERT_VALUES));
    PetscCall(MatRestoreRow_SeqAIJ(A, i, &nz, &cwork, &vwork));
  }
  PetscCall(PetscFree(cnew));

  (*B)->assembled = PETSC_FALSE;

#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*B, A->boundtocpu));
#endif
  PetscCall(MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY));
  PetscCall(ISRestoreIndices(irowp, &row));
  PetscCall(ISRestoreIndices(icolp, &col));
  PetscCall(ISDestroy(&irowp));
  PetscCall(ISDestroy(&icolp));
  if (rowp == colp) PetscCall(MatPropagateSymmetryOptions(A, *B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCopy_SeqAIJ(Mat A, Mat B, MatStructure str)
{
  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
    Mat_SeqAIJ        *b = (Mat_SeqAIJ *)B->data;
    const PetscScalar *aa;
    PetscScalar       *bb;

    PetscCall(MatSeqAIJGetArrayRead(A, &aa));
    PetscCall(MatSeqAIJGetArrayWrite(B, &bb));

    PetscCheck(a->i[A->rmap->n] == b->i[B->rmap->n], PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of nonzeros in two matrices are different %" PetscInt_FMT " != %" PetscInt_FMT, a->i[A->rmap->n], b->i[B->rmap->n]);
    PetscCall(PetscArraycpy(bb, aa, a->i[A->rmap->n]));
    PetscCall(PetscObjectStateIncrease((PetscObject)B));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
    PetscCall(MatSeqAIJRestoreArrayWrite(B, &bb));
  } else {
    PetscCall(MatCopy_Basic(A, B, str));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqAIJGetArray_SeqAIJ(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqAIJRestoreArray_SeqAIJ(Mat A, PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Computes the number of nonzeros per row needed for preallocation when X and Y
   have different nonzero structure.
*/
PetscErrorCode MatAXPYGetPreallocation_SeqX_private(PetscInt m, const PetscInt *xi, const PetscInt *xj, const PetscInt *yi, const PetscInt *yj, PetscInt *nnz)
{
  PetscInt i, j, k, nzx, nzy;

  PetscFunctionBegin;
  /* Set the number of nonzeros in the new matrix */
  for (i = 0; i < m; i++) {
    const PetscInt *xjj = PetscSafePointerPlusOffset(xj, xi[i]), *yjj = PetscSafePointerPlusOffset(yj, yi[i]);
    nzx    = xi[i + 1] - xi[i];
    nzy    = yi[i + 1] - yi[i];
    nnz[i] = 0;
    for (j = 0, k = 0; j < nzx; j++) {                  /* Point in X */
      for (; k < nzy && yjj[k] < xjj[j]; k++) nnz[i]++; /* Catch up to X */
      if (k < nzy && yjj[k] == xjj[j]) k++;             /* Skip duplicate */
      nnz[i]++;
    }
    for (; k < nzy; k++) nnz[i]++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatAXPYGetPreallocation_SeqAIJ(Mat Y, Mat X, PetscInt *nnz)
{
  PetscInt    m = Y->rmap->N;
  Mat_SeqAIJ *x = (Mat_SeqAIJ *)X->data;
  Mat_SeqAIJ *y = (Mat_SeqAIJ *)Y->data;

  PetscFunctionBegin;
  /* Set the number of nonzeros in the new matrix */
  PetscCall(MatAXPYGetPreallocation_SeqX_private(m, x->i, x->j, y->i, y->j, nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatAXPY_SeqAIJ(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_SeqAIJ *x = (Mat_SeqAIJ *)X->data, *y = (Mat_SeqAIJ *)Y->data;

  PetscFunctionBegin;
  if (str == UNKNOWN_NONZERO_PATTERN || (PetscDefined(USE_DEBUG) && str == SAME_NONZERO_PATTERN)) {
    PetscBool e = x->nz == y->nz ? PETSC_TRUE : PETSC_FALSE;
    if (e) {
      PetscCall(PetscArraycmp(x->i, y->i, Y->rmap->n + 1, &e));
      if (e) {
        PetscCall(PetscArraycmp(x->j, y->j, y->nz, &e));
        if (e) str = SAME_NONZERO_PATTERN;
      }
    }
    if (!e) PetscCheck(str != SAME_NONZERO_PATTERN, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "MatStructure is not SAME_NONZERO_PATTERN");
  }
  if (str == SAME_NONZERO_PATTERN) {
    const PetscScalar *xa;
    PetscScalar       *ya, alpha = a;
    PetscBLASInt       one = 1, bnz;

    PetscCall(PetscBLASIntCast(x->nz, &bnz));
    PetscCall(MatSeqAIJGetArray(Y, &ya));
    PetscCall(MatSeqAIJGetArrayRead(X, &xa));
    PetscCallBLAS("BLASaxpy", BLASaxpy_(&bnz, &alpha, xa, &one, ya, &one));
    PetscCall(MatSeqAIJRestoreArrayRead(X, &xa));
    PetscCall(MatSeqAIJRestoreArray(Y, &ya));
    PetscCall(PetscLogFlops(2.0 * bnz));
    PetscCall(MatSeqAIJInvalidateDiagonal(Y));
    PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    PetscCall(MatAXPY_Basic(Y, a, X, str));
  } else {
    Mat       B;
    PetscInt *nnz;
    PetscCall(PetscMalloc1(Y->rmap->N, &nnz));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)Y), &B));
    PetscCall(PetscObjectSetName((PetscObject)B, ((PetscObject)Y)->name));
    PetscCall(MatSetLayouts(B, Y->rmap, Y->cmap));
    PetscCall(MatSetType(B, ((PetscObject)Y)->type_name));
    PetscCall(MatAXPYGetPreallocation_SeqAIJ(Y, X, nnz));
    PetscCall(MatSeqAIJSetPreallocation(B, 0, nnz));
    PetscCall(MatAXPY_BasicWithPreallocation(B, Y, a, X, str));
    PetscCall(MatHeaderMerge(Y, &B));
    PetscCall(MatSeqAIJCheckInode(Y));
    PetscCall(PetscFree(nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConjugate_SeqAIJ(Mat mat)
{
#if defined(PETSC_USE_COMPLEX)
  Mat_SeqAIJ  *aij = (Mat_SeqAIJ *)mat->data;
  PetscInt     i, nz;
  PetscScalar *a;

  PetscFunctionBegin;
  nz = aij->nz;
  PetscCall(MatSeqAIJGetArray(mat, &a));
  for (i = 0; i < nz; i++) a[i] = PetscConj(a[i]);
  PetscCall(MatSeqAIJRestoreArray(mat, &a));
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMaxAbs_SeqAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  PetscInt         i, j, m = A->rmap->n, *ai, *aj, ncols, n;
  PetscReal        atmp;
  PetscScalar     *x;
  const MatScalar *aa, *av;

  PetscFunctionBegin;
  PetscCheck(!A->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  aa = av;
  ai = a->i;
  aj = a->j;

  PetscCall(VecGetArrayWrite(v, &x));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  for (i = 0; i < m; i++) {
    ncols = ai[1] - ai[0];
    ai++;
    x[i] = 0;
    for (j = 0; j < ncols; j++) {
      atmp = PetscAbsScalar(*aa);
      if (PetscAbsScalar(x[i]) < atmp) {
        x[i] = atmp;
        if (idx) idx[i] = *aj;
      }
      aa++;
      aj++;
    }
  }
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowSumAbs_SeqAIJ(Mat A, Vec v)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  PetscInt         i, j, m = A->rmap->n, *ai, ncols, n;
  PetscScalar     *x;
  const MatScalar *aa, *av;

  PetscFunctionBegin;
  PetscCheck(!A->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  aa = av;
  ai = a->i;

  PetscCall(VecGetArrayWrite(v, &x));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  for (i = 0; i < m; i++) {
    ncols = ai[1] - ai[0];
    ai++;
    x[i] = 0;
    for (j = 0; j < ncols; j++) {
      x[i] += PetscAbsScalar(*aa);
      aa++;
    }
  }
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMax_SeqAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  PetscInt         i, j, m = A->rmap->n, *ai, *aj, ncols, n;
  PetscScalar     *x;
  const MatScalar *aa, *av;

  PetscFunctionBegin;
  PetscCheck(!A->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  aa = av;
  ai = a->i;
  aj = a->j;

  PetscCall(VecGetArrayWrite(v, &x));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  for (i = 0; i < m; i++) {
    ncols = ai[1] - ai[0];
    ai++;
    if (ncols == A->cmap->n) { /* row is dense */
      x[i] = *aa;
      if (idx) idx[i] = 0;
    } else { /* row is sparse so already KNOW maximum is 0.0 or higher */
      x[i] = 0.0;
      if (idx) {
        for (j = 0; j < ncols; j++) { /* find first implicit 0.0 in the row */
          if (aj[j] > j) {
            idx[i] = j;
            break;
          }
        }
        /* in case first implicit 0.0 in the row occurs at ncols-th column */
        if (j == ncols && j < A->cmap->n) idx[i] = j;
      }
    }
    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(x[i]) < PetscRealPart(*aa)) {
        x[i] = *aa;
        if (idx) idx[i] = *aj;
      }
      aa++;
      aj++;
    }
  }
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMinAbs_SeqAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  PetscInt         i, j, m = A->rmap->n, *ai, *aj, ncols, n;
  PetscScalar     *x;
  const MatScalar *aa, *av;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  aa = av;
  ai = a->i;
  aj = a->j;

  PetscCall(VecGetArrayWrite(v, &x));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector, %" PetscInt_FMT " vs. %" PetscInt_FMT " rows", m, n);
  for (i = 0; i < m; i++) {
    ncols = ai[1] - ai[0];
    ai++;
    if (ncols == A->cmap->n) { /* row is dense */
      x[i] = *aa;
      if (idx) idx[i] = 0;
    } else { /* row is sparse so already KNOW minimum is 0.0 or higher */
      x[i] = 0.0;
      if (idx) { /* find first implicit 0.0 in the row */
        for (j = 0; j < ncols; j++) {
          if (aj[j] > j) {
            idx[i] = j;
            break;
          }
        }
        /* in case first implicit 0.0 in the row occurs at ncols-th column */
        if (j == ncols && j < A->cmap->n) idx[i] = j;
      }
    }
    for (j = 0; j < ncols; j++) {
      if (PetscAbsScalar(x[i]) > PetscAbsScalar(*aa)) {
        x[i] = *aa;
        if (idx) idx[i] = *aj;
      }
      aa++;
      aj++;
    }
  }
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMin_SeqAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)A->data;
  PetscInt         i, j, m = A->rmap->n, ncols, n;
  const PetscInt  *ai, *aj;
  PetscScalar     *x;
  const MatScalar *aa, *av;

  PetscFunctionBegin;
  PetscCheck(!A->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  aa = av;
  ai = a->i;
  aj = a->j;

  PetscCall(VecGetArrayWrite(v, &x));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  for (i = 0; i < m; i++) {
    ncols = ai[1] - ai[0];
    ai++;
    if (ncols == A->cmap->n) { /* row is dense */
      x[i] = *aa;
      if (idx) idx[i] = 0;
    } else { /* row is sparse so already KNOW minimum is 0.0 or lower */
      x[i] = 0.0;
      if (idx) { /* find first implicit 0.0 in the row */
        for (j = 0; j < ncols; j++) {
          if (aj[j] > j) {
            idx[i] = j;
            break;
          }
        }
        /* in case first implicit 0.0 in the row occurs at ncols-th column */
        if (j == ncols && j < A->cmap->n) idx[i] = j;
      }
    }
    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(x[i]) > PetscRealPart(*aa)) {
        x[i] = *aa;
        if (idx) idx[i] = *aj;
      }
      aa++;
      aj++;
    }
  }
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatInvertBlockDiagonal_SeqAIJ(Mat A, const PetscScalar **values)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  PetscInt        i, bs = A->rmap->bs, mbs = A->rmap->n / bs, ipvt[5], bs2 = bs * bs, *v_pivots, ij[7], *IJ, j;
  MatScalar      *diag, work[25], *v_work;
  const PetscReal shift = 0.0;
  PetscBool       allowzeropivot, zeropivotdetected = PETSC_FALSE;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  if (a->ibdiagvalid) {
    if (values) *values = a->ibdiag;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatMarkDiagonal_SeqAIJ(A));
  if (!a->ibdiag) { PetscCall(PetscMalloc1(bs2 * mbs, &a->ibdiag)); }
  diag = a->ibdiag;
  if (values) *values = a->ibdiag;
  /* factor and invert each block */
  switch (bs) {
  case 1:
    for (i = 0; i < mbs; i++) {
      PetscCall(MatGetValues(A, 1, &i, 1, &i, diag + i));
      if (PetscAbsScalar(diag[i] + shift) < PETSC_MACHINE_EPSILON) {
        if (allowzeropivot) {
          A->factorerrortype             = MAT_FACTOR_NUMERIC_ZEROPIVOT;
          A->factorerror_zeropivot_value = PetscAbsScalar(diag[i]);
          A->factorerror_zeropivot_row   = i;
          PetscCall(PetscInfo(A, "Zero pivot, row %" PetscInt_FMT " pivot %g tolerance %g\n", i, (double)PetscAbsScalar(diag[i]), (double)PETSC_MACHINE_EPSILON));
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Zero pivot, row %" PetscInt_FMT " pivot %g tolerance %g", i, (double)PetscAbsScalar(diag[i]), (double)PETSC_MACHINE_EPSILON);
      }
      diag[i] = (PetscScalar)1.0 / (diag[i] + shift);
    }
    break;
  case 2:
    for (i = 0; i < mbs; i++) {
      ij[0] = 2 * i;
      ij[1] = 2 * i + 1;
      PetscCall(MatGetValues(A, 2, ij, 2, ij, diag));
      PetscCall(PetscKernel_A_gets_inverse_A_2(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_2(diag));
      diag += 4;
    }
    break;
  case 3:
    for (i = 0; i < mbs; i++) {
      ij[0] = 3 * i;
      ij[1] = 3 * i + 1;
      ij[2] = 3 * i + 2;
      PetscCall(MatGetValues(A, 3, ij, 3, ij, diag));
      PetscCall(PetscKernel_A_gets_inverse_A_3(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_3(diag));
      diag += 9;
    }
    break;
  case 4:
    for (i = 0; i < mbs; i++) {
      ij[0] = 4 * i;
      ij[1] = 4 * i + 1;
      ij[2] = 4 * i + 2;
      ij[3] = 4 * i + 3;
      PetscCall(MatGetValues(A, 4, ij, 4, ij, diag));
      PetscCall(PetscKernel_A_gets_inverse_A_4(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_4(diag));
      diag += 16;
    }
    break;
  case 5:
    for (i = 0; i < mbs; i++) {
      ij[0] = 5 * i;
      ij[1] = 5 * i + 1;
      ij[2] = 5 * i + 2;
      ij[3] = 5 * i + 3;
      ij[4] = 5 * i + 4;
      PetscCall(MatGetValues(A, 5, ij, 5, ij, diag));
      PetscCall(PetscKernel_A_gets_inverse_A_5(diag, ipvt, work, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_5(diag));
      diag += 25;
    }
    break;
  case 6:
    for (i = 0; i < mbs; i++) {
      ij[0] = 6 * i;
      ij[1] = 6 * i + 1;
      ij[2] = 6 * i + 2;
      ij[3] = 6 * i + 3;
      ij[4] = 6 * i + 4;
      ij[5] = 6 * i + 5;
      PetscCall(MatGetValues(A, 6, ij, 6, ij, diag));
      PetscCall(PetscKernel_A_gets_inverse_A_6(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_6(diag));
      diag += 36;
    }
    break;
  case 7:
    for (i = 0; i < mbs; i++) {
      ij[0] = 7 * i;
      ij[1] = 7 * i + 1;
      ij[2] = 7 * i + 2;
      ij[3] = 7 * i + 3;
      ij[4] = 7 * i + 4;
      ij[5] = 7 * i + 5;
      ij[6] = 7 * i + 6;
      PetscCall(MatGetValues(A, 7, ij, 7, ij, diag));
      PetscCall(PetscKernel_A_gets_inverse_A_7(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_7(diag));
      diag += 49;
    }
    break;
  default:
    PetscCall(PetscMalloc3(bs, &v_work, bs, &v_pivots, bs, &IJ));
    for (i = 0; i < mbs; i++) {
      for (j = 0; j < bs; j++) IJ[j] = bs * i + j;
      PetscCall(MatGetValues(A, bs, IJ, bs, IJ, diag));
      PetscCall(PetscKernel_A_gets_inverse_A(bs, diag, v_pivots, v_work, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      PetscCall(PetscKernel_A_gets_transpose_A_N(diag, bs));
      diag += bs2;
    }
    PetscCall(PetscFree3(v_work, v_pivots, IJ));
  }
  a->ibdiagvalid = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetRandom_SeqAIJ(Mat x, PetscRandom rctx)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)x->data;
  PetscScalar a, *aa;
  PetscInt    m, n, i, j, col;

  PetscFunctionBegin;
  if (!x->assembled) {
    PetscCall(MatGetSize(x, &m, &n));
    for (i = 0; i < m; i++) {
      for (j = 0; j < aij->imax[i]; j++) {
        PetscCall(PetscRandomGetValue(rctx, &a));
        col = (PetscInt)(n * PetscRealPart(a));
        PetscCall(MatSetValues(x, 1, &i, 1, &col, &a, ADD_VALUES));
      }
    }
  } else {
    PetscCall(MatSeqAIJGetArrayWrite(x, &aa));
    for (i = 0; i < aij->nz; i++) PetscCall(PetscRandomGetValue(rctx, aa + i));
    PetscCall(MatSeqAIJRestoreArrayWrite(x, &aa));
  }
  PetscCall(MatAssemblyBegin(x, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Like MatSetRandom_SeqAIJ, but do not set values on columns in range of [low, high) */
PetscErrorCode MatSetRandomSkipColumnRange_SeqAIJ_Private(Mat x, PetscInt low, PetscInt high, PetscRandom rctx)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)x->data;
  PetscScalar a;
  PetscInt    m, n, i, j, col, nskip;

  PetscFunctionBegin;
  nskip = high - low;
  PetscCall(MatGetSize(x, &m, &n));
  n -= nskip; /* shrink number of columns where nonzeros can be set */
  for (i = 0; i < m; i++) {
    for (j = 0; j < aij->imax[i]; j++) {
      PetscCall(PetscRandomGetValue(rctx, &a));
      col = (PetscInt)(n * PetscRealPart(a));
      if (col >= low) col += nskip; /* shift col rightward to skip the hole */
      PetscCall(MatSetValues(x, 1, &i, 1, &col, &a, ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(x, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _MatOps MatOps_Values = {MatSetValues_SeqAIJ,
                                       MatGetRow_SeqAIJ,
                                       MatRestoreRow_SeqAIJ,
                                       MatMult_SeqAIJ,
                                       /*  4*/ MatMultAdd_SeqAIJ,
                                       MatMultTranspose_SeqAIJ,
                                       MatMultTransposeAdd_SeqAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /* 10*/ NULL,
                                       MatLUFactor_SeqAIJ,
                                       NULL,
                                       MatSOR_SeqAIJ,
                                       MatTranspose_SeqAIJ,
                                       /* 15*/ MatGetInfo_SeqAIJ,
                                       MatEqual_SeqAIJ,
                                       MatGetDiagonal_SeqAIJ,
                                       MatDiagonalScale_SeqAIJ,
                                       MatNorm_SeqAIJ,
                                       /* 20*/ NULL,
                                       MatAssemblyEnd_SeqAIJ,
                                       MatSetOption_SeqAIJ,
                                       MatZeroEntries_SeqAIJ,
                                       /* 24*/ MatZeroRows_SeqAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /* 29*/ MatSetUp_Seq_Hash,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /* 34*/ MatDuplicate_SeqAIJ,
                                       NULL,
                                       NULL,
                                       MatILUFactor_SeqAIJ,
                                       NULL,
                                       /* 39*/ MatAXPY_SeqAIJ,
                                       MatCreateSubMatrices_SeqAIJ,
                                       MatIncreaseOverlap_SeqAIJ,
                                       MatGetValues_SeqAIJ,
                                       MatCopy_SeqAIJ,
                                       /* 44*/ MatGetRowMax_SeqAIJ,
                                       MatScale_SeqAIJ,
                                       MatShift_SeqAIJ,
                                       MatDiagonalSet_SeqAIJ,
                                       MatZeroRowsColumns_SeqAIJ,
                                       /* 49*/ MatSetRandom_SeqAIJ,
                                       MatGetRowIJ_SeqAIJ,
                                       MatRestoreRowIJ_SeqAIJ,
                                       MatGetColumnIJ_SeqAIJ,
                                       MatRestoreColumnIJ_SeqAIJ,
                                       /* 54*/ MatFDColoringCreate_SeqXAIJ,
                                       NULL,
                                       NULL,
                                       MatPermute_SeqAIJ,
                                       NULL,
                                       /* 59*/ NULL,
                                       MatDestroy_SeqAIJ,
                                       MatView_SeqAIJ,
                                       NULL,
                                       NULL,
                                       /* 64*/ MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatGetRowMaxAbs_SeqAIJ,
                                       /* 69*/ MatGetRowMinAbs_SeqAIJ,
                                       NULL,
                                       NULL,
                                       MatFDColoringApply_AIJ,
                                       NULL,
                                       /* 74*/ MatFindZeroDiagonals_SeqAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatLoad_SeqAIJ,
                                       /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /* 84*/ NULL,
                                       MatMatMultNumeric_SeqAIJ_SeqAIJ,
                                       MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy,
                                       NULL,
                                       MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ,
                                       /* 90*/ NULL,
                                       MatProductSetFromOptions_SeqAIJ,
                                       NULL,
                                       NULL,
                                       MatConjugate_SeqAIJ,
                                       /* 94*/ NULL,
                                       MatSetValuesRow_SeqAIJ,
                                       MatRealPart_SeqAIJ,
                                       MatImaginaryPart_SeqAIJ,
                                       NULL,
                                       /* 99*/ NULL,
                                       MatMatSolve_SeqAIJ,
                                       NULL,
                                       MatGetRowMin_SeqAIJ,
                                       NULL,
                                       /*104*/ MatMissingDiagonal_SeqAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*114*/ MatGetMultiProcBlock_SeqAIJ,
                                       MatFindNonzeroRows_SeqAIJ,
                                       MatGetColumnReductions_SeqAIJ,
                                       MatInvertBlockDiagonal_SeqAIJ,
                                       MatInvertVariableBlockDiagonal_SeqAIJ,
                                       /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       MatTransposeMatMultNumeric_SeqAIJ_SeqAIJ,
                                       MatTransposeColoringCreate_SeqAIJ,
                                       /*124*/ MatTransColoringApplySpToDen_SeqAIJ,
                                       MatTransColoringApplyDenToSp_SeqAIJ,
                                       MatRARtNumeric_SeqAIJ_SeqAIJ,
                                       NULL,
                                       NULL,
                                       /*129*/ MatFDColoringSetUp_SeqXAIJ,
                                       MatFindOffBlockDiagonalEntries_SeqAIJ,
                                       MatCreateMPIMatConcatenateSeqMat_SeqAIJ,
                                       MatDestroySubMatrices_SeqAIJ,
                                       NULL,
                                       /*134*/ NULL,
                                       MatCreateGraph_Simple_AIJ,
                                       MatTransposeSymbolic_SeqAIJ,
                                       MatEliminateZeros_SeqAIJ,
                                       MatGetRowSumAbs_SeqAIJ,
                                       /*139*/ NULL,
                                       NULL,
                                       NULL,
                                       MatCopyHashToXAIJ_Seq_Hash};

static PetscErrorCode MatSeqAIJSetColumnIndices_SeqAIJ(Mat mat, PetscInt *indices)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  PetscInt    i, nz, n;

  PetscFunctionBegin;
  nz = aij->maxnz;
  n  = mat->rmap->n;
  for (i = 0; i < nz; i++) aij->j[i] = indices[i];
  aij->nz = nz;
  for (i = 0; i < n; i++) aij->ilen[i] = aij->imax[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Given a sparse matrix with global column indices, compact it by using a local column space.
 * The result matrix helps saving memory in other algorithms, such as MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable()
 */
PetscErrorCode MatSeqAIJCompactOutExtraColumns_SeqAIJ(Mat mat, ISLocalToGlobalMapping *mapping)
{
  Mat_SeqAIJ   *aij = (Mat_SeqAIJ *)mat->data;
  PetscHMapI    gid1_lid1;
  PetscHashIter tpos;
  PetscInt      gid, lid, i, ec, nz = aij->nz;
  PetscInt     *garray, *jj = aij->j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscAssertPointer(mapping, 2);
  /* use a table */
  PetscCall(PetscHMapICreateWithSize(mat->rmap->n, &gid1_lid1));
  ec = 0;
  for (i = 0; i < nz; i++) {
    PetscInt data, gid1 = jj[i] + 1;
    PetscCall(PetscHMapIGetWithDefault(gid1_lid1, gid1, 0, &data));
    if (!data) {
      /* one based table */
      PetscCall(PetscHMapISet(gid1_lid1, gid1, ++ec));
    }
  }
  /* form array of columns we need */
  PetscCall(PetscMalloc1(ec, &garray));
  PetscHashIterBegin(gid1_lid1, tpos);
  while (!PetscHashIterAtEnd(gid1_lid1, tpos)) {
    PetscHashIterGetKey(gid1_lid1, tpos, gid);
    PetscHashIterGetVal(gid1_lid1, tpos, lid);
    PetscHashIterNext(gid1_lid1, tpos);
    gid--;
    lid--;
    garray[lid] = gid;
  }
  PetscCall(PetscSortInt(ec, garray)); /* sort, and rebuild */
  PetscCall(PetscHMapIClear(gid1_lid1));
  for (i = 0; i < ec; i++) PetscCall(PetscHMapISet(gid1_lid1, garray[i] + 1, i + 1));
  /* compact out the extra columns in B */
  for (i = 0; i < nz; i++) {
    PetscInt gid1 = jj[i] + 1;
    PetscCall(PetscHMapIGetWithDefault(gid1_lid1, gid1, 0, &lid));
    lid--;
    jj[i] = lid;
  }
  PetscCall(PetscLayoutDestroy(&mat->cmap));
  PetscCall(PetscHMapIDestroy(&gid1_lid1));
  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)mat), ec, ec, 1, &mat->cmap));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, mat->cmap->bs, mat->cmap->n, garray, PETSC_OWN_POINTER, mapping));
  PetscCall(ISLocalToGlobalMappingSetType(*mapping, ISLOCALTOGLOBALMAPPINGHASH));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSeqAIJSetColumnIndices - Set the column indices for all the rows
  in the matrix.

  Input Parameters:
+ mat     - the `MATSEQAIJ` matrix
- indices - the column indices

  Level: advanced

  Notes:
  This can be called if you have precomputed the nonzero structure of the
  matrix and want to provide it to the matrix object to improve the performance
  of the `MatSetValues()` operation.

  You MUST have set the correct numbers of nonzeros per row in the call to
  `MatCreateSeqAIJ()`, and the columns indices MUST be sorted.

  MUST be called before any calls to `MatSetValues()`

  The indices should start with zero, not one.

.seealso: [](ch_matrices), `Mat`, `MATSEQAIJ`
@*/
PetscErrorCode MatSeqAIJSetColumnIndices(Mat mat, PetscInt *indices)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscAssertPointer(indices, 2);
  PetscUseMethod(mat, "MatSeqAIJSetColumnIndices_C", (Mat, PetscInt *), (mat, indices));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatStoreValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  size_t      nz  = aij->i[mat->rmap->n];

  PetscFunctionBegin;
  PetscCheck(aij->nonew, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");

  /* allocate space for values if not already there */
  if (!aij->saved_values) { PetscCall(PetscMalloc1(nz + 1, &aij->saved_values)); }

  /* copy values over */
  PetscCall(PetscArraycpy(aij->saved_values, aij->a, nz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatStoreValues - Stashes a copy of the matrix values; this allows reusing of the linear part of a Jacobian, while recomputing only the
  nonlinear portion.

  Logically Collect

  Input Parameter:
. mat - the matrix (currently only `MATAIJ` matrices support this option)

  Level: advanced

  Example Usage:
.vb
    Using SNES
    Create Jacobian matrix
    Set linear terms into matrix
    Apply boundary conditions to matrix, at this time matrix must have
      final nonzero structure (i.e. setting the nonlinear terms and applying
      boundary conditions again will not change the nonzero structure
    MatSetOption(mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);
    MatStoreValues(mat);
    Call SNESSetJacobian() with matrix
    In your Jacobian routine
      MatRetrieveValues(mat);
      Set nonlinear terms in matrix

    Without `SNESSolve()`, i.e. when you handle nonlinear solve yourself:
    // build linear portion of Jacobian
    MatSetOption(mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);
    MatStoreValues(mat);
    loop over nonlinear iterations
       MatRetrieveValues(mat);
       // call MatSetValues(mat,...) to set nonliner portion of Jacobian
       // call MatAssemblyBegin/End() on matrix
       Solve linear system with Jacobian
    endloop
.ve

  Notes:
  Matrix must already be assembled before calling this routine
  Must set the matrix option `MatSetOption`(mat,`MAT_NEW_NONZERO_LOCATIONS`,`PETSC_FALSE`); before
  calling this routine.

  When this is called multiple times it overwrites the previous set of stored values
  and does not allocated additional space.

.seealso: [](ch_matrices), `Mat`, `MatRetrieveValues()`
@*/
PetscErrorCode MatStoreValues(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscCheck(mat->assembled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCheck(!mat->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscUseMethod(mat, "MatStoreValues_C", (Mat), (mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRetrieveValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  PetscInt    nz  = aij->i[mat->rmap->n];

  PetscFunctionBegin;
  PetscCheck(aij->nonew, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  PetscCheck(aij->saved_values, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call MatStoreValues(A);first");
  /* copy values over */
  PetscCall(PetscArraycpy(aij->a, aij->saved_values, nz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatRetrieveValues - Retrieves the copy of the matrix values that was stored with `MatStoreValues()`

  Logically Collect

  Input Parameter:
. mat - the matrix (currently only `MATAIJ` matrices support this option)

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatStoreValues()`
@*/
PetscErrorCode MatRetrieveValues(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscCheck(mat->assembled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCheck(!mat->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscUseMethod(mat, "MatRetrieveValues_C", (Mat), (mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateSeqAIJ - Creates a sparse matrix in `MATSEQAIJ` (compressed row) format
  (the default parallel PETSc format).  For good matrix assembly performance
  the user should preallocate the matrix storage by setting the parameter `nz`
  (or the array `nnz`).

  Collective

  Input Parameters:
+ comm - MPI communicator, set to `PETSC_COMM_SELF`
. m    - number of rows
. n    - number of columns
. nz   - number of nonzeros per row (same for all rows)
- nnz  - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

  Output Parameter:
. A - the matrix

  Options Database Keys:
+ -mat_no_inode            - Do not use inodes
- -mat_inode_limit <limit> - Sets inode limit (max limit=5)

  Level: intermediate

  Notes:
  It is recommend to use `MatCreateFromOptions()` instead of this routine

  If `nnz` is given then `nz` is ignored

  The `MATSEQAIJ` format, also called
  compressed row storage, is fully compatible with standard Fortran
  storage.  That is, the stored row and column indices can begin at
  either one (as in Fortran) or zero.

  Specify the preallocated storage with either `nz` or `nnz` (not both).
  Set `nz` = `PETSC_DEFAULT` and `nnz` = `NULL` for PETSc to control dynamic memory
  allocation.

  By default, this format uses inodes (identical nodes) when possible, to
  improve numerical efficiency of matrix-vector products and solves. We
  search for consecutive rows with the same nonzero structure, thereby
  reusing matrix information to achieve increased efficiency.

.seealso: [](ch_matrices), `Mat`, [Sparse Matrix Creation](sec_matsparse), `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`
@*/
PetscErrorCode MatCreateSeqAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt nz, const PetscInt nnz[], Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A, nz, nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSeqAIJSetPreallocation - For good matrix assembly performance
  the user should preallocate the matrix storage by setting the parameter nz
  (or the array nnz).  By setting these parameters accurately, performance
  during matrix assembly can be increased by more than a factor of 50.

  Collective

  Input Parameters:
+ B   - The matrix
. nz  - number of nonzeros per row (same for all rows)
- nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

  Options Database Keys:
+ -mat_no_inode            - Do not use inodes
- -mat_inode_limit <limit> - Sets inode limit (max limit=5)

  Level: intermediate

  Notes:
  If `nnz` is given then `nz` is ignored

  The `MATSEQAIJ` format also called
  compressed row storage, is fully compatible with standard Fortran
  storage.  That is, the stored row and column indices can begin at
  either one (as in Fortran) or zero.  See the users' manual for details.

  Specify the preallocated storage with either `nz` or `nnz` (not both).
  Set nz = `PETSC_DEFAULT` and `nnz` = `NULL` for PETSc to control dynamic memory
  allocation.

  You can call `MatGetInfo()` to get information on how effective the preallocation was;
  for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
  You can also run with the option -info and look for messages with the string
  malloc in them to see if additional memory allocation was needed.

  Developer Notes:
  Use nz of `MAT_SKIP_ALLOCATION` to not allocate any space for the matrix
  entries or columns indices

  By default, this format uses inodes (identical nodes) when possible, to
  improve numerical efficiency of matrix-vector products and solves. We
  search for consecutive rows with the same nonzero structure, thereby
  reusing matrix information to achieve increased efficiency.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MatGetInfo()`,
          `MatSeqAIJSetTotalPreallocation()`
@*/
PetscErrorCode MatSeqAIJSetPreallocation(Mat B, PetscInt nz, const PetscInt nnz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidType(B, 1);
  PetscTryMethod(B, "MatSeqAIJSetPreallocation_C", (Mat, PetscInt, const PetscInt[]), (B, nz, nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJSetPreallocation_SeqAIJ(Mat B, PetscInt nz, const PetscInt *nnz)
{
  Mat_SeqAIJ *b              = (Mat_SeqAIJ *)B->data;
  PetscBool   skipallocation = PETSC_FALSE, realalloc = PETSC_FALSE;
  PetscInt    i;

  PetscFunctionBegin;
  if (B->hash_active) {
    B->ops[0] = b->cops;
    PetscCall(PetscHMapIJVDestroy(&b->ht));
    PetscCall(PetscFree(b->dnz));
    B->hash_active = PETSC_FALSE;
  }
  if (nz >= 0 || nnz) realalloc = PETSC_TRUE;
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  PetscCheck(nz >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nz cannot be less than 0: value %" PetscInt_FMT, nz);
  if (nnz) {
    for (i = 0; i < B->rmap->n; i++) {
      PetscCheck(nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, nnz[i]);
      PetscCheck(nnz[i] <= B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nnz cannot be greater than row length: local row %" PetscInt_FMT " value %" PetscInt_FMT " rowlength %" PetscInt_FMT, i, nnz[i], B->cmap->n);
    }
  }

  B->preallocated = PETSC_TRUE;
  if (!skipallocation) {
    if (!b->imax) { PetscCall(PetscMalloc1(B->rmap->n, &b->imax)); }
    if (!b->ilen) {
      /* b->ilen will count nonzeros in each row so far. */
      PetscCall(PetscCalloc1(B->rmap->n, &b->ilen));
    } else {
      PetscCall(PetscMemzero(b->ilen, B->rmap->n * sizeof(PetscInt)));
    }
    if (!b->ipre) PetscCall(PetscMalloc1(B->rmap->n, &b->ipre));
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 10;
      else if (nz < 0) nz = 1;
      nz = PetscMin(nz, B->cmap->n);
      for (i = 0; i < B->rmap->n; i++) b->imax[i] = nz;
      PetscCall(PetscIntMultError(nz, B->rmap->n, &nz));
    } else {
      PetscInt64 nz64 = 0;
      for (i = 0; i < B->rmap->n; i++) {
        b->imax[i] = nnz[i];
        nz64 += nnz[i];
      }
      PetscCall(PetscIntCast(nz64, &nz));
    }

    /* allocate the matrix space */
    PetscCall(MatSeqXAIJFreeAIJ(B, &b->a, &b->j, &b->i));
    PetscCall(PetscShmgetAllocateArray(nz, sizeof(PetscInt), (void **)&b->j));
    PetscCall(PetscShmgetAllocateArray(B->rmap->n + 1, sizeof(PetscInt), (void **)&b->i));
    b->free_ij = PETSC_TRUE;
    if (B->structure_only) {
      b->free_a = PETSC_FALSE;
    } else {
      PetscCall(PetscShmgetAllocateArray(nz, sizeof(PetscScalar), (void **)&b->a));
      b->free_a = PETSC_TRUE;
    }
    b->i[0] = 0;
    for (i = 1; i < B->rmap->n + 1; i++) b->i[i] = b->i[i - 1] + b->imax[i - 1];
  } else {
    b->free_a  = PETSC_FALSE;
    b->free_ij = PETSC_FALSE;
  }

  if (b->ipre && nnz != b->ipre && b->imax) {
    /* reserve user-requested sparsity */
    PetscCall(PetscArraycpy(b->ipre, b->imax, B->rmap->n));
  }

  b->nz               = 0;
  b->maxnz            = nz;
  B->info.nz_unneeded = (double)b->maxnz;
  if (realalloc) PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  /* We simply deem preallocation has changed nonzero state. Updating the state
     will give clients (like AIJKokkos) a chance to know something has happened.
  */
  B->nonzerostate++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatResetPreallocation_SeqAIJ_Private(Mat A, PetscBool *memoryreset)
{
  Mat_SeqAIJ *a;
  PetscInt    i;
  PetscBool   skipreset;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);

  PetscCheck(A->insertmode == NOT_SET_VALUES, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot reset preallocation after setting some values but not yet calling MatAssemblyBegin()/MatAssemblyEnd()");
  if (A->num_ass == 0) PetscFunctionReturn(PETSC_SUCCESS);

  /* Check local size. If zero, then return */
  if (!A->rmap->n) PetscFunctionReturn(PETSC_SUCCESS);

  a = (Mat_SeqAIJ *)A->data;
  /* if no saved info, we error out */
  PetscCheck(a->ipre, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "No saved preallocation info ");

  PetscCheck(a->i && a->imax && a->ilen, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Memory info is incomplete, and cannot reset preallocation ");

  PetscCall(PetscArraycmp(a->ipre, a->ilen, A->rmap->n, &skipreset));
  if (skipreset) PetscCall(MatZeroEntries(A));
  else {
    PetscCall(PetscArraycpy(a->imax, a->ipre, A->rmap->n));
    PetscCall(PetscArrayzero(a->ilen, A->rmap->n));
    a->i[0] = 0;
    for (i = 1; i < A->rmap->n + 1; i++) a->i[i] = a->i[i - 1] + a->imax[i - 1];
    A->preallocated     = PETSC_TRUE;
    a->nz               = 0;
    a->maxnz            = a->i[A->rmap->n];
    A->info.nz_unneeded = (double)a->maxnz;
    A->was_assembled    = PETSC_FALSE;
    A->assembled        = PETSC_FALSE;
    A->nonzerostate++;
    /* Log that the state of this object has changed; this will help guarantee that preconditioners get re-setup */
    PetscCall(PetscObjectStateIncrease((PetscObject)A));
  }
  if (memoryreset) *memoryreset = (PetscBool)!skipreset;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatResetPreallocation_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatResetPreallocation_SeqAIJ_Private(A, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSeqAIJSetPreallocationCSR - Allocates memory for a sparse sequential matrix in `MATSEQAIJ` format.

  Input Parameters:
+ B - the matrix
. i - the indices into `j` for the start of each row (indices start with zero)
. j - the column indices for each row (indices start with zero) these must be sorted for each row
- v - optional values in the matrix, use `NULL` if not provided

  Level: developer

  Notes:
  The `i`,`j`,`v` values are COPIED with this routine; to avoid the copy use `MatCreateSeqAIJWithArrays()`

  This routine may be called multiple times with different nonzero patterns (or the same nonzero pattern). The nonzero
  structure will be the union of all the previous nonzero structures.

  Developer Notes:
  An optimization could be added to the implementation where it checks if the `i`, and `j` are identical to the current `i` and `j` and
  then just copies the `v` values directly with `PetscMemcpy()`.

  This routine could also take a `PetscCopyMode` argument to allow sharing the values instead of always copying them.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatSeqAIJSetPreallocation()`, `MATSEQAIJ`, `MatResetPreallocation()`
@*/
PetscErrorCode MatSeqAIJSetPreallocationCSR(Mat B, const PetscInt i[], const PetscInt j[], const PetscScalar v[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidType(B, 1);
  PetscTryMethod(B, "MatSeqAIJSetPreallocationCSR_C", (Mat, const PetscInt[], const PetscInt[], const PetscScalar[]), (B, i, j, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJSetPreallocationCSR_SeqAIJ(Mat B, const PetscInt Ii[], const PetscInt J[], const PetscScalar v[])
{
  PetscInt  i;
  PetscInt  m, n;
  PetscInt  nz;
  PetscInt *nnz;

  PetscFunctionBegin;
  PetscCheck(Ii[0] == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Ii[0] must be 0 it is %" PetscInt_FMT, Ii[0]);

  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

  PetscCall(MatGetSize(B, &m, &n));
  PetscCall(PetscMalloc1(m + 1, &nnz));
  for (i = 0; i < m; i++) {
    nz = Ii[i + 1] - Ii[i];
    PetscCheck(nz >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local row %" PetscInt_FMT " has a negative number of columns %" PetscInt_FMT, i, nz);
    nnz[i] = nz;
  }
  PetscCall(MatSeqAIJSetPreallocation(B, 0, nnz));
  PetscCall(PetscFree(nnz));

  for (i = 0; i < m; i++) PetscCall(MatSetValues_SeqAIJ(B, 1, &i, Ii[i + 1] - Ii[i], J + Ii[i], PetscSafePointerPlusOffset(v, Ii[i]), INSERT_VALUES));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSeqAIJKron - Computes `C`, the Kronecker product of `A` and `B`.

  Input Parameters:
+ A     - left-hand side matrix
. B     - right-hand side matrix
- reuse - either `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`

  Output Parameter:
. C - Kronecker product of `A` and `B`

  Level: intermediate

  Note:
  `MAT_REUSE_MATRIX` can only be used when the nonzero structure of the product matrix has not changed from that last call to `MatSeqAIJKron()`.

.seealso: [](ch_matrices), `Mat`, `MatCreateSeqAIJ()`, `MATSEQAIJ`, `MATKAIJ`, `MatReuse`
@*/
PetscErrorCode MatSeqAIJKron(Mat A, Mat B, MatReuse reuse, Mat *C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscValidType(B, 2);
  PetscAssertPointer(C, 4);
  if (reuse == MAT_REUSE_MATRIX) {
    PetscValidHeaderSpecific(*C, MAT_CLASSID, 4);
    PetscValidType(*C, 4);
  }
  PetscTryMethod(A, "MatSeqAIJKron_C", (Mat, Mat, MatReuse, Mat *), (A, B, reuse, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJKron_SeqAIJ(Mat A, Mat B, MatReuse reuse, Mat *C)
{
  Mat                newmat;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ        *b = (Mat_SeqAIJ *)B->data;
  PetscScalar       *v;
  const PetscScalar *aa, *ba;
  PetscInt          *i, *j, m, n, p, q, nnz = 0, am = A->rmap->n, bm = B->rmap->n, an = A->cmap->n, bn = B->cmap->n;
  PetscBool          flg;

  PetscFunctionBegin;
  PetscCheck(!A->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCheck(A->assembled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCheck(!B->factortype, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCheck(B->assembled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQAIJ, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_SUP, "MatType %s", ((PetscObject)B)->type_name);
  PetscCheck(reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX, PETSC_COMM_SELF, PETSC_ERR_SUP, "MatReuse %d", (int)reuse);
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc2(am * bm + 1, &i, a->i[am] * b->i[bm], &j));
    PetscCall(MatCreate(PETSC_COMM_SELF, &newmat));
    PetscCall(MatSetSizes(newmat, am * bm, an * bn, am * bm, an * bn));
    PetscCall(MatSetType(newmat, MATAIJ));
    i[0] = 0;
    for (m = 0; m < am; ++m) {
      for (p = 0; p < bm; ++p) {
        i[m * bm + p + 1] = i[m * bm + p] + (a->i[m + 1] - a->i[m]) * (b->i[p + 1] - b->i[p]);
        for (n = a->i[m]; n < a->i[m + 1]; ++n) {
          for (q = b->i[p]; q < b->i[p + 1]; ++q) j[nnz++] = a->j[n] * bn + b->j[q];
        }
      }
    }
    PetscCall(MatSeqAIJSetPreallocationCSR(newmat, i, j, NULL));
    *C = newmat;
    PetscCall(PetscFree2(i, j));
    nnz = 0;
  }
  PetscCall(MatSeqAIJGetArray(*C, &v));
  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  PetscCall(MatSeqAIJGetArrayRead(B, &ba));
  for (m = 0; m < am; ++m) {
    for (p = 0; p < bm; ++p) {
      for (n = a->i[m]; n < a->i[m + 1]; ++n) {
        for (q = b->i[p]; q < b->i[p + 1]; ++q) v[nnz++] = aa[n] * ba[q];
      }
    }
  }
  PetscCall(MatSeqAIJRestoreArray(*C, &v));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscCall(MatSeqAIJRestoreArrayRead(B, &ba));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/mat/impls/dense/seq/dense.h>
#include <petsc/private/kernels/petscaxpy.h>

/*
    Computes (B'*A')' since computing B*A directly is untenable

               n                       p                          p
        [             ]       [             ]         [                 ]
      m [      A      ]  *  n [       B     ]   =   m [         C       ]
        [             ]       [             ]         [                 ]

*/
PetscErrorCode MatMatMultNumeric_SeqDense_SeqAIJ(Mat A, Mat B, Mat C)
{
  Mat_SeqDense      *sub_a = (Mat_SeqDense *)A->data;
  Mat_SeqAIJ        *sub_b = (Mat_SeqAIJ *)B->data;
  Mat_SeqDense      *sub_c = (Mat_SeqDense *)C->data;
  PetscInt           i, j, n, m, q, p;
  const PetscInt    *ii, *idx;
  const PetscScalar *b, *a, *a_q;
  PetscScalar       *c, *c_q;
  PetscInt           clda = sub_c->lda;
  PetscInt           alda = sub_a->lda;

  PetscFunctionBegin;
  m = A->rmap->n;
  n = A->cmap->n;
  p = B->cmap->n;
  a = sub_a->v;
  b = sub_b->a;
  c = sub_c->v;
  if (clda == m) {
    PetscCall(PetscArrayzero(c, m * p));
  } else {
    for (j = 0; j < p; j++)
      for (i = 0; i < m; i++) c[j * clda + i] = 0.0;
  }
  ii  = sub_b->i;
  idx = sub_b->j;
  for (i = 0; i < n; i++) {
    q = ii[i + 1] - ii[i];
    while (q-- > 0) {
      c_q = c + clda * (*idx);
      a_q = a + alda * i;
      PetscKernelAXPY(c_q, *b, a_q, m);
      idx++;
      b++;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultSymbolic_SeqDense_SeqAIJ(Mat A, Mat B, PetscReal fill, Mat C)
{
  PetscInt  m = A->rmap->n, n = B->cmap->n;
  PetscBool cisdense;

  PetscFunctionBegin;
  PetscCheck(A->cmap->n == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "A->cmap->n %" PetscInt_FMT " != B->rmap->n %" PetscInt_FMT, A->cmap->n, B->rmap->n);
  PetscCall(MatSetSizes(C, m, n, m, n));
  PetscCall(MatSetBlockSizesFromMats(C, A, B));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C, &cisdense, MATSEQDENSE, MATSEQDENSECUDA, MATSEQDENSEHIP, ""));
  if (!cisdense) PetscCall(MatSetType(C, MATDENSE));
  PetscCall(MatSetUp(C));

  C->ops->matmultnumeric = MatMatMultNumeric_SeqDense_SeqAIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSEQAIJ - MATSEQAIJ = "seqaij" - A matrix type to be used for sequential sparse matrices,
   based on compressed sparse row format.

   Options Database Key:
. -mat_type seqaij - sets the matrix type to "seqaij" during a call to MatSetFromOptions()

   Level: beginner

   Notes:
    `MatSetValues()` may be called for this matrix type with a `NULL` argument for the numerical values,
    in this case the values associated with the rows and columns one passes in are set to zero
    in the matrix

    `MatSetOptions`(,`MAT_STRUCTURE_ONLY`,`PETSC_TRUE`) may be called for this matrix type. In this no
    space is allocated for the nonzero entries and any entries passed with `MatSetValues()` are ignored

  Developer Note:
    It would be nice if all matrix formats supported passing `NULL` in for the numerical values

.seealso: [](ch_matrices), `Mat`, `MatCreateSeqAIJ()`, `MatSetFromOptions()`, `MatSetType()`, `MatCreate()`, `MatType`, `MATSELL`, `MATSEQSELL`, `MATMPISELL`
M*/

/*MC
   MATAIJ - MATAIJ = "aij" - A matrix type to be used for sparse matrices.

   This matrix type is identical to `MATSEQAIJ` when constructed with a single process communicator,
   and `MATMPIAIJ` otherwise.  As a result, for single process communicators,
   `MatSeqAIJSetPreallocation()` is supported, and similarly `MatMPIAIJSetPreallocation()` is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Key:
. -mat_type aij - sets the matrix type to "aij" during a call to `MatSetFromOptions()`

  Level: beginner

   Note:
   Subclasses include `MATAIJCUSPARSE`, `MATAIJPERM`, `MATAIJSELL`, `MATAIJMKL`, `MATAIJCRL`, and also automatically switches over to use inodes when
   enough exist.

.seealso: [](ch_matrices), `Mat`, `MatCreateAIJ()`, `MatCreateSeqAIJ()`, `MATSEQAIJ`, `MATMPIAIJ`, `MATSELL`, `MATSEQSELL`, `MATMPISELL`
M*/

/*MC
   MATAIJCRL - MATAIJCRL = "aijcrl" - A matrix type to be used for sparse matrices.

   Options Database Key:
. -mat_type aijcrl - sets the matrix type to "aijcrl" during a call to `MatSetFromOptions()`

  Level: beginner

   Note:
   This matrix type is identical to `MATSEQAIJCRL` when constructed with a single process communicator,
   and `MATMPIAIJCRL` otherwise.  As a result, for single process communicators,
   `MatSeqAIJSetPreallocation()` is supported, and similarly `MatMPIAIJSetPreallocation()` is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

.seealso: [](ch_matrices), `Mat`, `MatCreateMPIAIJCRL`, `MATSEQAIJCRL`, `MATMPIAIJCRL`, `MATSEQAIJCRL`, `MATMPIAIJCRL`
M*/

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCRL(Mat, MatType, MatReuse, Mat *);
#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_Elemental(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_INTERN PetscErrorCode MatConvert_AIJ_ScaLAPACK(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatConvert_AIJ_HYPRE(Mat A, MatType, MatReuse, Mat *);
#endif

PETSC_EXTERN PetscErrorCode MatConvert_SeqAIJ_SeqSELL(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_XAIJ_IS(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_IS_XAIJ(Mat);

/*@C
  MatSeqAIJGetArray - gives read/write access to the array where the data for a `MATSEQAIJ` matrix is stored

  Not Collective

  Input Parameter:
. A - a `MATSEQAIJ` matrix

  Output Parameter:
. array - pointer to the data

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJRestoreArray()`
@*/
PetscErrorCode MatSeqAIJGetArray(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (aij->ops->getarray) {
    PetscCall((*aij->ops->getarray)(A, array));
  } else {
    *array = aij->a;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJRestoreArray - returns access to the array where the data for a `MATSEQAIJ` matrix is stored obtained by `MatSeqAIJGetArray()`

  Not Collective

  Input Parameters:
+ A     - a `MATSEQAIJ` matrix
- array - pointer to the data

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJGetArray()`
@*/
PetscErrorCode MatSeqAIJRestoreArray(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (aij->ops->restorearray) {
    PetscCall((*aij->ops->restorearray)(A, array));
  } else {
    *array = NULL;
  }
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJGetArrayRead - gives read-only access to the array where the data for a `MATSEQAIJ` matrix is stored

  Not Collective; No Fortran Support

  Input Parameter:
. A - a `MATSEQAIJ` matrix

  Output Parameter:
. array - pointer to the data

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJGetArray()`, `MatSeqAIJRestoreArrayRead()`
@*/
PetscErrorCode MatSeqAIJGetArrayRead(Mat A, const PetscScalar *array[])
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (aij->ops->getarrayread) {
    PetscCall((*aij->ops->getarrayread)(A, array));
  } else {
    *array = aij->a;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJRestoreArrayRead - restore the read-only access array obtained from `MatSeqAIJGetArrayRead()`

  Not Collective; No Fortran Support

  Input Parameter:
. A - a `MATSEQAIJ` matrix

  Output Parameter:
. array - pointer to the data

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJGetArray()`, `MatSeqAIJGetArrayRead()`
@*/
PetscErrorCode MatSeqAIJRestoreArrayRead(Mat A, const PetscScalar *array[])
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (aij->ops->restorearrayread) {
    PetscCall((*aij->ops->restorearrayread)(A, array));
  } else {
    *array = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJGetArrayWrite - gives write-only access to the array where the data for a `MATSEQAIJ` matrix is stored

  Not Collective; No Fortran Support

  Input Parameter:
. A - a `MATSEQAIJ` matrix

  Output Parameter:
. array - pointer to the data

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJGetArray()`, `MatSeqAIJRestoreArrayRead()`
@*/
PetscErrorCode MatSeqAIJGetArrayWrite(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (aij->ops->getarraywrite) {
    PetscCall((*aij->ops->getarraywrite)(A, array));
  } else {
    *array = aij->a;
  }
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJRestoreArrayWrite - restore the read-only access array obtained from MatSeqAIJGetArrayRead

  Not Collective; No Fortran Support

  Input Parameter:
. A - a MATSEQAIJ matrix

  Output Parameter:
. array - pointer to the data

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJGetArray()`, `MatSeqAIJGetArrayRead()`
@*/
PetscErrorCode MatSeqAIJRestoreArrayWrite(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (aij->ops->restorearraywrite) {
    PetscCall((*aij->ops->restorearraywrite)(A, array));
  } else {
    *array = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJGetCSRAndMemType - Get the CSR arrays and the memory type of the `MATSEQAIJ` matrix

  Not Collective; No Fortran Support

  Input Parameter:
. mat - a matrix of type `MATSEQAIJ` or its subclasses

  Output Parameters:
+ i     - row map array of the matrix
. j     - column index array of the matrix
. a     - data array of the matrix
- mtype - memory type of the arrays

  Level: developer

  Notes:
  Any of the output parameters can be `NULL`, in which case the corresponding value is not returned.
  If mat is a device matrix, the arrays are on the device. Otherwise, they are on the host.

  One can call this routine on a preallocated but not assembled matrix to just get the memory of the CSR underneath the matrix.
  If the matrix is assembled, the data array `a` is guaranteed to have the latest values of the matrix.

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJGetArray()`, `MatSeqAIJGetArrayRead()`
@*/
PetscErrorCode MatSeqAIJGetCSRAndMemType(Mat mat, const PetscInt *i[], const PetscInt *j[], PetscScalar *a[], PetscMemType *mtype)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCheck(mat->preallocated, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "matrix is not preallocated");
  if (aij->ops->getcsrandmemtype) {
    PetscCall((*aij->ops->getcsrandmemtype)(mat, i, j, a, mtype));
  } else {
    if (i) *i = aij->i;
    if (j) *j = aij->j;
    if (a) *a = aij->a;
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSeqAIJGetMaxRowNonzeros - returns the maximum number of nonzeros in any row

  Not Collective

  Input Parameter:
. A - a `MATSEQAIJ` matrix

  Output Parameter:
. nz - the maximum number of nonzeros in any row

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJRestoreArray()`
@*/
PetscErrorCode MatSeqAIJGetMaxRowNonzeros(Mat A, PetscInt *nz)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  *nz = aij->rmax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCOOStructDestroy_SeqAIJ(void **data)
{
  MatCOOStruct_SeqAIJ *coo = (MatCOOStruct_SeqAIJ *)*data;

  PetscFunctionBegin;
  PetscCall(PetscFree(coo->perm));
  PetscCall(PetscFree(coo->jmap));
  PetscCall(PetscFree(coo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetPreallocationCOO_SeqAIJ(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  MPI_Comm             comm;
  PetscInt            *i, *j;
  PetscInt             M, N, row, iprev;
  PetscCount           k, p, q, nneg, nnz, start, end; /* Index the coo array, so use PetscCount as their type */
  PetscInt            *Ai;                             /* Change to PetscCount once we use it for row pointers */
  PetscInt            *Aj;
  PetscScalar         *Aa;
  Mat_SeqAIJ          *seqaij = (Mat_SeqAIJ *)mat->data;
  MatType              rtype;
  PetscCount          *perm, *jmap;
  MatCOOStruct_SeqAIJ *coo;
  PetscBool            isorted;
  PetscBool            hypre;
  const char          *name;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCall(MatGetSize(mat, &M, &N));
  i = coo_i;
  j = coo_j;
  PetscCall(PetscMalloc1(coo_n, &perm));

  /* Ignore entries with negative row or col indices; at the same time, check if i[] is already sorted (e.g., MatConvert_AlJ_HYPRE results in this case) */
  isorted = PETSC_TRUE;
  iprev   = PETSC_INT_MIN;
  for (k = 0; k < coo_n; k++) {
    if (j[k] < 0) i[k] = -1;
    if (isorted) {
      if (i[k] < iprev) isorted = PETSC_FALSE;
      else iprev = i[k];
    }
    perm[k] = k;
  }

  /* Sort by row if not already */
  if (!isorted) PetscCall(PetscSortIntWithIntCountArrayPair(coo_n, i, j, perm));
  PetscCheck(i == NULL || i[coo_n - 1] < M, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "COO row index %" PetscInt_FMT " is >= the matrix row size %" PetscInt_FMT, i[coo_n - 1], M);

  /* Advance k to the first row with a non-negative index */
  for (k = 0; k < coo_n; k++)
    if (i[k] >= 0) break;
  nneg = k;
  PetscCall(PetscMalloc1(coo_n - nneg + 1, &jmap)); /* +1 to make a CSR-like data structure. jmap[i] originally is the number of repeats for i-th nonzero */
  nnz = 0;                                          /* Total number of unique nonzeros to be counted */
  jmap++;                                           /* Inc jmap by 1 for convenience */

  PetscCall(PetscShmgetAllocateArray(M + 1, sizeof(PetscInt), (void **)&Ai)); /* CSR of A */
  PetscCall(PetscArrayzero(Ai, M + 1));
  PetscCall(PetscShmgetAllocateArray(coo_n - nneg, sizeof(PetscInt), (void **)&Aj)); /* We have at most coo_n-nneg unique nonzeros */

  PetscCall(PetscObjectGetName((PetscObject)mat, &name));
  PetscCall(PetscStrcmp("_internal_COO_mat_for_hypre", name, &hypre));

  /* In each row, sort by column, then unique column indices to get row length */
  Ai++;  /* Inc by 1 for convenience */
  q = 0; /* q-th unique nonzero, with q starting from 0 */
  while (k < coo_n) {
    PetscBool strictly_sorted; // this row is strictly sorted?
    PetscInt  jprev;

    /* get [start,end) indices for this row; also check if cols in this row are strictly sorted */
    row             = i[k];
    start           = k;
    jprev           = PETSC_INT_MIN;
    strictly_sorted = PETSC_TRUE;
    while (k < coo_n && i[k] == row) {
      if (strictly_sorted) {
        if (j[k] <= jprev) strictly_sorted = PETSC_FALSE;
        else jprev = j[k];
      }
      k++;
    }
    end = k;

    /* hack for HYPRE: swap min column to diag so that diagonal values will go first */
    if (hypre) {
      PetscInt  minj    = PETSC_INT_MAX;
      PetscBool hasdiag = PETSC_FALSE;

      if (strictly_sorted) { // fast path to swap the first and the diag
        PetscCount tmp;
        for (p = start; p < end; p++) {
          if (j[p] == row && p != start) {
            j[p]        = j[start]; // swap j[], so that the diagonal value will go first (manipulated by perm[])
            j[start]    = row;
            tmp         = perm[start];
            perm[start] = perm[p]; // also swap perm[] so we can save the call to PetscSortIntWithCountArray() below
            perm[p]     = tmp;
            break;
          }
        }
      } else {
        for (p = start; p < end; p++) {
          hasdiag = (PetscBool)(hasdiag || (j[p] == row));
          minj    = PetscMin(minj, j[p]);
        }

        if (hasdiag) {
          for (p = start; p < end; p++) {
            if (j[p] == minj) j[p] = row;
            else if (j[p] == row) j[p] = minj;
          }
        }
      }
    }
    // sort by columns in a row. perm[] indicates their original order
    if (!strictly_sorted) PetscCall(PetscSortIntWithCountArray(end - start, j + start, perm + start));
    PetscCheck(end == start || j[end - 1] < N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "COO column index %" PetscInt_FMT " is >= the matrix column size %" PetscInt_FMT, j[end - 1], N);

    if (strictly_sorted) { // fast path to set Aj[], jmap[], Ai[], nnz, q
      for (p = start; p < end; p++, q++) {
        Aj[q]   = j[p];
        jmap[q] = 1;
      }
      PetscCall(PetscIntCast(end - start, Ai + row));
      nnz += Ai[row]; // q is already advanced
    } else {
      /* Find number of unique col entries in this row */
      Aj[q]   = j[start]; /* Log the first nonzero in this row */
      jmap[q] = 1;        /* Number of repeats of this nonzero entry */
      Ai[row] = 1;
      nnz++;

      for (p = start + 1; p < end; p++) { /* Scan remaining nonzero in this row */
        if (j[p] != j[p - 1]) {           /* Meet a new nonzero */
          q++;
          jmap[q] = 1;
          Aj[q]   = j[p];
          Ai[row]++;
          nnz++;
        } else {
          jmap[q]++;
        }
      }
      q++; /* Move to next row and thus next unique nonzero */
    }
  }

  Ai--; /* Back to the beginning of Ai[] */
  for (k = 0; k < M; k++) Ai[k + 1] += Ai[k];
  jmap--; // Back to the beginning of jmap[]
  jmap[0] = 0;
  for (k = 0; k < nnz; k++) jmap[k + 1] += jmap[k];

  if (nnz < coo_n - nneg) { /* Reallocate with actual number of unique nonzeros */
    PetscCount *jmap_new;
    PetscInt   *Aj_new;

    PetscCall(PetscMalloc1(nnz + 1, &jmap_new));
    PetscCall(PetscArraycpy(jmap_new, jmap, nnz + 1));
    PetscCall(PetscFree(jmap));
    jmap = jmap_new;

    PetscCall(PetscShmgetAllocateArray(nnz, sizeof(PetscInt), (void **)&Aj_new));
    PetscCall(PetscArraycpy(Aj_new, Aj, nnz));
    PetscCall(PetscShmgetDeallocateArray((void **)&Aj));
    Aj = Aj_new;
  }

  if (nneg) { /* Discard heading entries with negative indices in perm[], as we'll access it from index 0 in MatSetValuesCOO */
    PetscCount *perm_new;

    PetscCall(PetscMalloc1(coo_n - nneg, &perm_new));
    PetscCall(PetscArraycpy(perm_new, perm + nneg, coo_n - nneg));
    PetscCall(PetscFree(perm));
    perm = perm_new;
  }

  PetscCall(MatGetRootType_Private(mat, &rtype));
  PetscCall(PetscShmgetAllocateArray(nnz, sizeof(PetscScalar), (void **)&Aa));
  PetscCall(PetscArrayzero(Aa, nnz));
  PetscCall(MatSetSeqAIJWithArrays_private(PETSC_COMM_SELF, M, N, Ai, Aj, Aa, rtype, mat));

  seqaij->free_a = seqaij->free_ij = PETSC_TRUE; /* Let newmat own Ai, Aj and Aa */

  // Put the COO struct in a container and then attach that to the matrix
  PetscCall(PetscMalloc1(1, &coo));
  PetscCall(PetscIntCast(nnz, &coo->nz));
  coo->n    = coo_n;
  coo->Atot = coo_n - nneg; // Annz is seqaij->nz, so no need to record that again
  coo->jmap = jmap;         // of length nnz+1
  coo->perm = perm;
  PetscCall(PetscObjectContainerCompose((PetscObject)mat, "__PETSc_MatCOOStruct_Host", coo, MatCOOStructDestroy_SeqAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_SeqAIJ(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJ          *aseq = (Mat_SeqAIJ *)A->data;
  PetscCount           i, j, Annz = aseq->nz;
  PetscCount          *perm, *jmap;
  PetscScalar         *Aa;
  PetscContainer       container;
  MatCOOStruct_SeqAIJ *coo;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "__PETSc_MatCOOStruct_Host", (PetscObject *)&container));
  PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not found MatCOOStruct on this matrix");
  PetscCall(PetscContainerGetPointer(container, (void **)&coo));
  perm = coo->perm;
  jmap = coo->jmap;
  PetscCall(MatSeqAIJGetArray(A, &Aa));
  for (i = 0; i < Annz; i++) {
    PetscScalar sum = 0.0;
    for (j = jmap[i]; j < jmap[i + 1]; j++) sum += v[perm[j]];
    Aa[i] = (imode == INSERT_VALUES ? 0.0 : Aa[i]) + sum;
  }
  PetscCall(MatSeqAIJRestoreArray(A, &Aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCUSPARSE(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_HIP)
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJHIPSPARSE(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat, MatType, MatReuse, Mat *);
#endif

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJ(Mat B)
{
  Mat_SeqAIJ *b;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B), &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Comm must be of size 1");

  PetscCall(PetscNew(&b));

  B->data   = (void *)b;
  B->ops[0] = MatOps_Values;
  if (B->sortedfull) B->ops->setvalues = MatSetValues_SeqAIJ_SortedFull;

  b->row                = NULL;
  b->col                = NULL;
  b->icol               = NULL;
  b->reallocs           = 0;
  b->ignorezeroentries  = PETSC_FALSE;
  b->roworiented        = PETSC_TRUE;
  b->nonew              = 0;
  b->diag               = NULL;
  b->solve_work         = NULL;
  B->spptr              = NULL;
  b->saved_values       = NULL;
  b->idiag              = NULL;
  b->mdiag              = NULL;
  b->ssor_work          = NULL;
  b->omega              = 1.0;
  b->fshift             = 0.0;
  b->idiagvalid         = PETSC_FALSE;
  b->ibdiagvalid        = PETSC_FALSE;
  b->keepnonzeropattern = PETSC_FALSE;

  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJ));
#if defined(PETSC_HAVE_MATLAB)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "PetscMatlabEnginePut_C", MatlabEnginePut_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "PetscMatlabEngineGet_C", MatlabEngineGet_SeqAIJ));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSeqAIJSetColumnIndices_C", MatSeqAIJSetColumnIndices_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatStoreValues_C", MatStoreValues_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatRetrieveValues_C", MatRetrieveValues_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqsbaij_C", MatConvert_SeqAIJ_SeqSBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqbaij_C", MatConvert_SeqAIJ_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqaijperm_C", MatConvert_SeqAIJ_SeqAIJPERM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqaijsell_C", MatConvert_SeqAIJ_SeqAIJSELL));
#if defined(PETSC_HAVE_MKL_SPARSE)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqaijmkl_C", MatConvert_SeqAIJ_SeqAIJMKL));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqaijcusparse_C", MatConvert_SeqAIJ_SeqAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaijcusparse_seqaij_C", MatProductSetFromOptions_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqaijcusparse_C", MatProductSetFromOptions_SeqAIJ));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqaijhipsparse_C", MatConvert_SeqAIJ_SeqAIJHIPSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaijhipsparse_seqaij_C", MatProductSetFromOptions_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqaijhipsparse_C", MatProductSetFromOptions_SeqAIJ));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqaijkokkos_C", MatConvert_SeqAIJ_SeqAIJKokkos));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqaijcrl_C", MatConvert_SeqAIJ_SeqAIJCRL));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_elemental_C", MatConvert_SeqAIJ_Elemental));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_scalapack_C", MatConvert_AIJ_ScaLAPACK));
#endif
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_hypre_C", MatConvert_AIJ_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_transpose_seqaij_seqaij_C", MatProductSetFromOptions_Transpose_AIJ_AIJ));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqdense_C", MatConvert_SeqAIJ_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_seqsell_C", MatConvert_SeqAIJ_SeqSELL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaij_is_C", MatConvert_XAIJ_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatIsTranspose_C", MatIsTranspose_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatIsHermitianTranspose_C", MatIsHermitianTranspose_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSeqAIJSetPreallocation_C", MatSeqAIJSetPreallocation_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatResetPreallocation_C", MatResetPreallocation_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatResetHash_C", MatResetHash_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSeqAIJSetPreallocationCSR_C", MatSeqAIJSetPreallocationCSR_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatReorderForNonzeroDiagonal_C", MatReorderForNonzeroDiagonal_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_is_seqaij_C", MatProductSetFromOptions_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqdense_seqaij_C", MatProductSetFromOptions_SeqDense_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqaij_C", MatProductSetFromOptions_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSeqAIJKron_C", MatSeqAIJKron_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetValuesCOO_C", MatSetValuesCOO_SeqAIJ));
  PetscCall(MatCreate_SeqAIJ_Inode(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJ));
  PetscCall(MatSeqAIJSetTypeFromOptions(B)); /* this allows changing the matrix subtype to say MATSEQAIJPERM */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Given a matrix generated with MatGetFactor() duplicates all the information in A into C
*/
PetscErrorCode MatDuplicateNoCreate_SeqAIJ(Mat C, Mat A, MatDuplicateOption cpvalues, PetscBool mallocmatspace)
{
  Mat_SeqAIJ *c = (Mat_SeqAIJ *)C->data, *a = (Mat_SeqAIJ *)A->data;
  PetscInt    m = A->rmap->n, i;

  PetscFunctionBegin;
  PetscCheck(A->assembled || cpvalues == MAT_DO_NOT_COPY_VALUES, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot duplicate unassembled matrix");

  C->factortype    = A->factortype;
  c->row           = NULL;
  c->col           = NULL;
  c->icol          = NULL;
  c->reallocs      = 0;
  c->diagonaldense = a->diagonaldense;

  C->assembled = A->assembled;

  if (A->preallocated) {
    PetscCall(PetscLayoutReference(A->rmap, &C->rmap));
    PetscCall(PetscLayoutReference(A->cmap, &C->cmap));

    if (!A->hash_active) {
      PetscCall(PetscMalloc1(m, &c->imax));
      PetscCall(PetscMemcpy(c->imax, a->imax, m * sizeof(PetscInt)));
      PetscCall(PetscMalloc1(m, &c->ilen));
      PetscCall(PetscMemcpy(c->ilen, a->ilen, m * sizeof(PetscInt)));

      /* allocate the matrix space */
      if (mallocmatspace) {
        PetscCall(PetscShmgetAllocateArray(a->i[m], sizeof(PetscScalar), (void **)&c->a));
        PetscCall(PetscShmgetAllocateArray(a->i[m], sizeof(PetscInt), (void **)&c->j));
        PetscCall(PetscShmgetAllocateArray(m + 1, sizeof(PetscInt), (void **)&c->i));
        PetscCall(PetscArraycpy(c->i, a->i, m + 1));
        c->free_a  = PETSC_TRUE;
        c->free_ij = PETSC_TRUE;
        if (m > 0) {
          PetscCall(PetscArraycpy(c->j, a->j, a->i[m]));
          if (cpvalues == MAT_COPY_VALUES) {
            const PetscScalar *aa;

            PetscCall(MatSeqAIJGetArrayRead(A, &aa));
            PetscCall(PetscArraycpy(c->a, aa, a->i[m]));
            PetscCall(MatSeqAIJGetArrayRead(A, &aa));
          } else {
            PetscCall(PetscArrayzero(c->a, a->i[m]));
          }
        }
      }
      C->preallocated = PETSC_TRUE;
    } else {
      PetscCheck(mallocmatspace, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Cannot malloc matrix memory from a non-preallocated matrix");
      PetscCall(MatSetUp(C));
    }

    c->ignorezeroentries = a->ignorezeroentries;
    c->roworiented       = a->roworiented;
    c->nonew             = a->nonew;
    if (a->diag) {
      PetscCall(PetscMalloc1(m + 1, &c->diag));
      PetscCall(PetscMemcpy(c->diag, a->diag, m * sizeof(PetscInt)));
    } else c->diag = NULL;

    c->solve_work         = NULL;
    c->saved_values       = NULL;
    c->idiag              = NULL;
    c->ssor_work          = NULL;
    c->keepnonzeropattern = a->keepnonzeropattern;

    c->rmax  = a->rmax;
    c->nz    = a->nz;
    c->maxnz = a->nz; /* Since we allocate exactly the right amount */

    c->compressedrow.use   = a->compressedrow.use;
    c->compressedrow.nrows = a->compressedrow.nrows;
    if (a->compressedrow.use) {
      i = a->compressedrow.nrows;
      PetscCall(PetscMalloc2(i + 1, &c->compressedrow.i, i, &c->compressedrow.rindex));
      PetscCall(PetscArraycpy(c->compressedrow.i, a->compressedrow.i, i + 1));
      PetscCall(PetscArraycpy(c->compressedrow.rindex, a->compressedrow.rindex, i));
    } else {
      c->compressedrow.use    = PETSC_FALSE;
      c->compressedrow.i      = NULL;
      c->compressedrow.rindex = NULL;
    }
    c->nonzerorowcnt = a->nonzerorowcnt;
    C->nonzerostate  = A->nonzerostate;

    PetscCall(MatDuplicate_SeqAIJ_Inode(A, cpvalues, &C));
  }
  PetscCall(PetscFunctionListDuplicate(((PetscObject)A)->qlist, &((PetscObject)C)->qlist));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDuplicate_SeqAIJ(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, A->rmap->n, A->cmap->n, A->rmap->n, A->cmap->n));
  if (!(A->rmap->n % A->rmap->bs) && !(A->cmap->n % A->cmap->bs)) PetscCall(MatSetBlockSizesFromMats(*B, A, A));
  PetscCall(MatSetType(*B, ((PetscObject)A)->type_name));
  PetscCall(MatDuplicateNoCreate_SeqAIJ(*B, A, cpvalues, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLoad_SeqAIJ(Mat newMat, PetscViewer viewer)
{
  PetscBool isbinary, ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newMat, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  /* force binary viewer to load .info file if it has not yet done so */
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
  if (isbinary) {
    PetscCall(MatLoad_SeqAIJ_Binary(newMat, viewer));
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(MatLoad_AIJ_HDF5(newMat, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject)newMat), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else {
    SETERRQ(PetscObjectComm((PetscObject)newMat), PETSC_ERR_SUP, "Viewer type %s not yet supported for reading %s matrices", ((PetscObject)viewer)->type_name, ((PetscObject)newMat)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLoad_SeqAIJ_Binary(Mat mat, PetscViewer viewer)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat->data;
  PetscInt    header[4], *rowlens, M, N, nz, sum, rows, cols, i;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));

  /* read in matrix header */
  PetscCall(PetscViewerBinaryRead(viewer, header, 4, NULL, PETSC_INT));
  PetscCheck(header[0] == MAT_FILE_CLASSID, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Not a matrix object in file");
  M  = header[1];
  N  = header[2];
  nz = header[3];
  PetscCheck(M >= 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Matrix row size (%" PetscInt_FMT ") in file is negative", M);
  PetscCheck(N >= 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Matrix column size (%" PetscInt_FMT ") in file is negative", N);
  PetscCheck(nz >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Matrix stored in special format on disk, cannot load as SeqAIJ");

  /* set block sizes from the viewer's .info file */
  PetscCall(MatLoad_Binary_BlockSizes(mat, viewer));
  /* set local and global sizes if not set already */
  if (mat->rmap->n < 0) mat->rmap->n = M;
  if (mat->cmap->n < 0) mat->cmap->n = N;
  if (mat->rmap->N < 0) mat->rmap->N = M;
  if (mat->cmap->N < 0) mat->cmap->N = N;
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));

  /* check if the matrix sizes are correct */
  PetscCall(MatGetSize(mat, &rows, &cols));
  PetscCheck(M == rows && N == cols, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different sizes (%" PetscInt_FMT ", %" PetscInt_FMT ") than the input matrix (%" PetscInt_FMT ", %" PetscInt_FMT ")", M, N, rows, cols);

  /* read in row lengths */
  PetscCall(PetscMalloc1(M, &rowlens));
  PetscCall(PetscViewerBinaryRead(viewer, rowlens, M, NULL, PETSC_INT));
  /* check if sum(rowlens) is same as nz */
  sum = 0;
  for (i = 0; i < M; i++) sum += rowlens[i];
  PetscCheck(sum == nz, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Inconsistent matrix data in file: nonzeros = %" PetscInt_FMT ", sum-row-lengths = %" PetscInt_FMT, nz, sum);
  /* preallocate and check sizes */
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(mat, 0, rowlens));
  PetscCall(MatGetSize(mat, &rows, &cols));
  PetscCheck(M == rows && N == cols, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different length (%" PetscInt_FMT ", %" PetscInt_FMT ") than the input matrix (%" PetscInt_FMT ", %" PetscInt_FMT ")", M, N, rows, cols);
  /* store row lengths */
  PetscCall(PetscArraycpy(a->ilen, rowlens, M));
  PetscCall(PetscFree(rowlens));

  /* fill in "i" row pointers */
  a->i[0] = 0;
  for (i = 0; i < M; i++) a->i[i + 1] = a->i[i] + a->ilen[i];
  /* read in "j" column indices */
  PetscCall(PetscViewerBinaryRead(viewer, a->j, nz, NULL, PETSC_INT));
  /* read in "a" nonzero values */
  PetscCall(PetscViewerBinaryRead(viewer, a->a, nz, NULL, PETSC_SCALAR));

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatEqual_SeqAIJ(Mat A, Mat B, PetscBool *flg)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data, *b = (Mat_SeqAIJ *)B->data;
  const PetscScalar *aa, *ba;
#if defined(PETSC_USE_COMPLEX)
  PetscInt k;
#endif

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->rmap->n != B->rmap->n) || (A->cmap->n != B->cmap->n) || (a->nz != b->nz)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* if the a->i are the same */
  PetscCall(PetscArraycmp(a->i, b->i, A->rmap->n + 1, flg));
  if (!*flg) PetscFunctionReturn(PETSC_SUCCESS);

  /* if a->j are the same */
  PetscCall(PetscArraycmp(a->j, b->j, a->nz, flg));
  if (!*flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  PetscCall(MatSeqAIJGetArrayRead(B, &ba));
  /* if a->a are the same */
#if defined(PETSC_USE_COMPLEX)
  for (k = 0; k < a->nz; k++) {
    if (PetscRealPart(aa[k]) != PetscRealPart(ba[k]) || PetscImaginaryPart(aa[k]) != PetscImaginaryPart(ba[k])) {
      *flg = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
#else
  PetscCall(PetscArraycmp(aa, ba, a->nz, flg));
#endif
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscCall(MatSeqAIJRestoreArrayRead(B, &ba));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateSeqAIJWithArrays - Creates an sequential `MATSEQAIJ` matrix using matrix elements (in CSR format)
  provided by the user.

  Collective

  Input Parameters:
+ comm - must be an MPI communicator of size 1
. m    - number of rows
. n    - number of columns
. i    - row indices; that is i[0] = 0, i[row] = i[row-1] + number of elements in that row of the matrix
. j    - column indices
- a    - matrix values

  Output Parameter:
. mat - the matrix

  Level: intermediate

  Notes:
  The `i`, `j`, and `a` arrays are not copied by this routine, the user must free these arrays
  once the matrix is destroyed and not before

  You cannot set new nonzero locations into this matrix, that will generate an error.

  The `i` and `j` indices are 0 based

  The format which is used for the sparse matrix input, is equivalent to a
  row-major ordering.. i.e for the following matrix, the input data expected is
  as shown
.vb
        1 0 0
        2 0 3
        4 5 6

        i =  {0,1,3,6}  [size = nrow+1  = 3+1]
        j =  {0,0,2,0,1,2}  [size = 6]; values must be sorted for each row
        v =  {1,2,3,4,5,6}  [size = 6]
.ve

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatCreateSeqAIJ()`, `MatCreateMPIAIJWithArrays()`, `MatMPIAIJSetPreallocationCSR()`
@*/
PetscErrorCode MatCreateSeqAIJWithArrays(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt i[], PetscInt j[], PetscScalar a[], Mat *mat)
{
  PetscInt    ii;
  Mat_SeqAIJ *aij;
  PetscInt    jj;

  PetscFunctionBegin;
  PetscCheck(m <= 0 || i[0] == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "i (row indices) must start with 0");
  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetSizes(*mat, m, n, m, n));
  /* PetscCall(MatSetBlockSizes(*mat,,)); */
  PetscCall(MatSetType(*mat, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*mat, MAT_SKIP_ALLOCATION, NULL));
  aij = (Mat_SeqAIJ *)(*mat)->data;
  PetscCall(PetscMalloc1(m, &aij->imax));
  PetscCall(PetscMalloc1(m, &aij->ilen));

  aij->i       = i;
  aij->j       = j;
  aij->a       = a;
  aij->nonew   = -1; /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aij->free_a  = PETSC_FALSE;
  aij->free_ij = PETSC_FALSE;

  for (ii = 0, aij->nonzerorowcnt = 0, aij->rmax = 0; ii < m; ii++) {
    aij->ilen[ii] = aij->imax[ii] = i[ii + 1] - i[ii];
    if (PetscDefined(USE_DEBUG)) {
      PetscCheck(i[ii + 1] - i[ii] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative row length in i (row indices) row = %" PetscInt_FMT " length = %" PetscInt_FMT, ii, i[ii + 1] - i[ii]);
      for (jj = i[ii] + 1; jj < i[ii + 1]; jj++) {
        PetscCheck(j[jj] >= j[jj - 1], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column entry number %" PetscInt_FMT " (actual column %" PetscInt_FMT ") in row %" PetscInt_FMT " is not sorted", jj - i[ii], j[jj], ii);
        PetscCheck(j[jj] != j[jj - 1], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column entry number %" PetscInt_FMT " (actual column %" PetscInt_FMT ") in row %" PetscInt_FMT " is identical to previous entry", jj - i[ii], j[jj], ii);
      }
    }
  }
  if (PetscDefined(USE_DEBUG)) {
    for (ii = 0; ii < aij->i[m]; ii++) {
      PetscCheck(j[ii] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative column index at location = %" PetscInt_FMT " index = %" PetscInt_FMT, ii, j[ii]);
      PetscCheck(j[ii] <= n - 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column index to large at location = %" PetscInt_FMT " index = %" PetscInt_FMT " last column = %" PetscInt_FMT, ii, j[ii], n - 1);
    }
  }

  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateSeqAIJFromTriple - Creates an sequential `MATSEQAIJ` matrix using matrix elements (in COO format)
  provided by the user.

  Collective

  Input Parameters:
+ comm - must be an MPI communicator of size 1
. m    - number of rows
. n    - number of columns
. i    - row indices
. j    - column indices
. a    - matrix values
. nz   - number of nonzeros
- idx  - if the `i` and `j` indices start with 1 use `PETSC_TRUE` otherwise use `PETSC_FALSE`

  Output Parameter:
. mat - the matrix

  Level: intermediate

  Example:
  For the following matrix, the input data expected is as shown (using 0 based indexing)
.vb
        1 0 0
        2 0 3
        4 5 6

        i =  {0,1,1,2,2,2}
        j =  {0,0,2,0,1,2}
        v =  {1,2,3,4,5,6}
.ve

  Note:
  Instead of using this function, users should also consider `MatSetPreallocationCOO()` and `MatSetValuesCOO()`, which allow repeated or remote entries,
  and are particularly useful in iterative applications.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatCreateSeqAIJ()`, `MatCreateSeqAIJWithArrays()`, `MatMPIAIJSetPreallocationCSR()`, `MatSetValuesCOO()`, `MatSetPreallocationCOO()`
@*/
PetscErrorCode MatCreateSeqAIJFromTriple(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt i[], PetscInt j[], PetscScalar a[], Mat *mat, PetscCount nz, PetscBool idx)
{
  PetscInt ii, *nnz, one = 1, row, col;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(m, &nnz));
  for (ii = 0; ii < nz; ii++) nnz[i[ii] - !!idx] += 1;
  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetSizes(*mat, m, n, m, n));
  PetscCall(MatSetType(*mat, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*mat, 0, nnz));
  for (ii = 0; ii < nz; ii++) {
    if (idx) {
      row = i[ii] - 1;
      col = j[ii] - 1;
    } else {
      row = i[ii];
      col = j[ii];
    }
    PetscCall(MatSetValues(*mat, one, &row, one, &col, &a[ii], ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJInvalidateDiagonal(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  a->idiagvalid  = PETSC_FALSE;
  a->ibdiagvalid = PETSC_FALSE;

  PetscCall(MatSeqAIJInvalidateDiagonal_Inode(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqAIJ(MPI_Comm comm, Mat inmat, PetscInt n, MatReuse scall, Mat *outmat)
{
  PetscFunctionBegin;
  PetscCall(MatCreateMPIMatConcatenateSeqMat_MPIAIJ(comm, inmat, n, scall, outmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Permute A into C's *local* index space using rowemb,colemb.
 The embedding are supposed to be injections and the above implies that the range of rowemb is a subset
 of [0,m), colemb is in [0,n).
 If pattern == DIFFERENT_NONZERO_PATTERN, C is preallocated according to A.
 */
PetscErrorCode MatSetSeqMat_SeqAIJ(Mat C, IS rowemb, IS colemb, MatStructure pattern, Mat B)
{
  /* If making this function public, change the error returned in this function away from _PLIB. */
  Mat_SeqAIJ     *Baij;
  PetscBool       seqaij;
  PetscInt        m, n, *nz, i, j, count;
  PetscScalar     v;
  const PetscInt *rowindices, *colindices;

  PetscFunctionBegin;
  if (!B) PetscFunctionReturn(PETSC_SUCCESS);
  /* Check to make sure the target matrix (and embeddings) are compatible with C and each other. */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATSEQAIJ, &seqaij));
  PetscCheck(seqaij, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input matrix is of wrong type");
  if (rowemb) {
    PetscCall(ISGetLocalSize(rowemb, &m));
    PetscCheck(m == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Row IS of size %" PetscInt_FMT " is incompatible with matrix row size %" PetscInt_FMT, m, B->rmap->n);
  } else {
    PetscCheck(C->rmap->n == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input matrix is row-incompatible with the target matrix");
  }
  if (colemb) {
    PetscCall(ISGetLocalSize(colemb, &n));
    PetscCheck(n == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diag col IS of size %" PetscInt_FMT " is incompatible with input matrix col size %" PetscInt_FMT, n, B->cmap->n);
  } else {
    PetscCheck(C->cmap->n == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input matrix is col-incompatible with the target matrix");
  }

  Baij = (Mat_SeqAIJ *)B->data;
  if (pattern == DIFFERENT_NONZERO_PATTERN) {
    PetscCall(PetscMalloc1(B->rmap->n, &nz));
    for (i = 0; i < B->rmap->n; i++) nz[i] = Baij->i[i + 1] - Baij->i[i];
    PetscCall(MatSeqAIJSetPreallocation(C, 0, nz));
    PetscCall(PetscFree(nz));
  }
  if (pattern == SUBSET_NONZERO_PATTERN) PetscCall(MatZeroEntries(C));
  count      = 0;
  rowindices = NULL;
  colindices = NULL;
  if (rowemb) PetscCall(ISGetIndices(rowemb, &rowindices));
  if (colemb) PetscCall(ISGetIndices(colemb, &colindices));
  for (i = 0; i < B->rmap->n; i++) {
    PetscInt row;
    row = i;
    if (rowindices) row = rowindices[i];
    for (j = Baij->i[i]; j < Baij->i[i + 1]; j++) {
      PetscInt col;
      col = Baij->j[count];
      if (colindices) col = colindices[col];
      v = Baij->a[count];
      PetscCall(MatSetValues(C, 1, &row, 1, &col, &v, INSERT_VALUES));
      ++count;
    }
  }
  /* FIXME: set C's nonzerostate correctly. */
  /* Assembly for C is necessary. */
  C->preallocated  = PETSC_TRUE;
  C->assembled     = PETSC_TRUE;
  C->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatEliminateZeros_SeqAIJ(Mat A, PetscBool keep)
{
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *)A->data;
  MatScalar  *aa = a->a;
  PetscInt    m = A->rmap->n, fshift = 0, fshift_prev = 0, i, k;
  PetscInt   *ailen = a->ilen, *imax = a->imax, *ai = a->i, *aj = a->j, rmax = 0;

  PetscFunctionBegin;
  PetscCheck(A->assembled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot eliminate zeros for unassembled matrix");
  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i = 1; i <= m; i++) {
    /* move each nonzero entry back by the amount of zero slots (fshift) before it*/
    for (k = ai[i - 1]; k < ai[i]; k++) {
      if (aa[k] == 0 && (aj[k] != i - 1 || !keep)) fshift++;
      else {
        if (aa[k] == 0 && aj[k] == i - 1) PetscCall(PetscInfo(A, "Keep the diagonal zero at row %" PetscInt_FMT "\n", i - 1));
        aa[k - fshift] = aa[k];
        aj[k - fshift] = aj[k];
      }
    }
    ai[i - 1] -= fshift_prev; // safe to update ai[i-1] now since it will not be used in the next iteration
    fshift_prev = fshift;
    /* reset ilen and imax for each row */
    ailen[i - 1] = imax[i - 1] = ai[i] - fshift - ai[i - 1];
    a->nonzerorowcnt += ((ai[i] - fshift - ai[i - 1]) > 0);
    rmax = PetscMax(rmax, ailen[i - 1]);
  }
  if (fshift) {
    if (m) {
      ai[m] -= fshift;
      a->nz = ai[m];
    }
    PetscCall(PetscInfo(A, "Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; zeros eliminated: %" PetscInt_FMT "; nonzeros left: %" PetscInt_FMT "\n", m, A->cmap->n, fshift, a->nz));
    A->nonzerostate++;
    A->info.nz_unneeded += (PetscReal)fshift;
    a->rmax = rmax;
    if (a->inode.use && a->inode.checked) PetscCall(MatSeqAIJCheckInode(A));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscFunctionList MatSeqAIJList = NULL;

/*@
  MatSeqAIJSetType - Converts a `MATSEQAIJ` matrix to a subtype

  Collective

  Input Parameters:
+ mat    - the matrix object
- matype - matrix type

  Options Database Key:
. -mat_seqaij_type  <method> - for example seqaijcrl

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `PCSetType()`, `VecSetType()`, `MatCreate()`, `MatType`
@*/
PetscErrorCode MatSeqAIJSetType(Mat mat, MatType matype)
{
  PetscBool sametype;
  PetscErrorCode (*r)(Mat, MatType, MatReuse, Mat *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, matype, &sametype));
  if (sametype) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(MatSeqAIJList, matype, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Mat type given: %s", matype);
  PetscCall((*r)(mat, matype, MAT_INPLACE_MATRIX, &mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJRegister -  - Adds a new sub-matrix type for sequential `MATSEQAIJ` matrices

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of a new user-defined matrix type, for example `MATSEQAIJCRL`
- function - routine to convert to subtype

  Level: advanced

  Notes:
  `MatSeqAIJRegister()` may be called multiple times to add several user-defined solvers.

  Then, your matrix can be chosen with the procedural interface at runtime via the option
.vb
  -mat_seqaij_type my_mat
.ve

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJRegisterAll()`
@*/
PetscErrorCode MatSeqAIJRegister(const char sname[], PetscErrorCode (*function)(Mat, MatType, MatReuse, Mat *))
{
  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  PetscCall(PetscFunctionListAdd(&MatSeqAIJList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscBool MatSeqAIJRegisterAllCalled = PETSC_FALSE;

/*@C
  MatSeqAIJRegisterAll - Registers all of the matrix subtypes of `MATSSEQAIJ`

  Not Collective

  Level: advanced

  Note:
  This registers the versions of `MATSEQAIJ` for GPUs

.seealso: [](ch_matrices), `Mat`, `MatRegisterAll()`, `MatSeqAIJRegister()`
@*/
PetscErrorCode MatSeqAIJRegisterAll(void)
{
  PetscFunctionBegin;
  if (MatSeqAIJRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  MatSeqAIJRegisterAllCalled = PETSC_TRUE;

  PetscCall(MatSeqAIJRegister(MATSEQAIJCRL, MatConvert_SeqAIJ_SeqAIJCRL));
  PetscCall(MatSeqAIJRegister(MATSEQAIJPERM, MatConvert_SeqAIJ_SeqAIJPERM));
  PetscCall(MatSeqAIJRegister(MATSEQAIJSELL, MatConvert_SeqAIJ_SeqAIJSELL));
#if defined(PETSC_HAVE_MKL_SPARSE)
  PetscCall(MatSeqAIJRegister(MATSEQAIJMKL, MatConvert_SeqAIJ_SeqAIJMKL));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(MatSeqAIJRegister(MATSEQAIJCUSPARSE, MatConvert_SeqAIJ_SeqAIJCUSPARSE));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(MatSeqAIJRegister(MATSEQAIJHIPSPARSE, MatConvert_SeqAIJ_SeqAIJHIPSPARSE));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(MatSeqAIJRegister(MATSEQAIJKOKKOS, MatConvert_SeqAIJ_SeqAIJKokkos));
#endif
#if defined(PETSC_HAVE_VIENNACL) && defined(PETSC_HAVE_VIENNACL_NO_CUDA)
  PetscCall(MatSeqAIJRegister(MATMPIAIJVIENNACL, MatConvert_SeqAIJ_SeqAIJViennaCL));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Special version for direct calls from Fortran
*/
#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matsetvaluesseqaij_ MATSETVALUESSEQAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matsetvaluesseqaij_ matsetvaluesseqaij
#endif

/* Change these macros so can be used in void function */

/* Change these macros so can be used in void function */
/* Identical to PetscCallVoid, except it assigns to *_ierr */
#undef PetscCall
#define PetscCall(...) \
  do { \
    PetscErrorCode ierr_msv_mpiaij = __VA_ARGS__; \
    if (PetscUnlikely(ierr_msv_mpiaij)) { \
      *_ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_msv_mpiaij, PETSC_ERROR_REPEAT, " "); \
      return; \
    } \
  } while (0)

#undef SETERRQ
#define SETERRQ(comm, ierr, ...) \
  do { \
    *_ierr = PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_INITIAL, __VA_ARGS__); \
    return; \
  } while (0)

PETSC_EXTERN void matsetvaluesseqaij_(Mat *AA, PetscInt *mm, const PetscInt im[], PetscInt *nn, const PetscInt in[], const PetscScalar v[], InsertMode *isis, PetscErrorCode *_ierr)
{
  Mat         A = *AA;
  PetscInt    m = *mm, n = *nn;
  InsertMode  is = *isis;
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *)A->data;
  PetscInt   *rp, k, low, high, t, ii, row, nrow, i, col, l, rmax, N;
  PetscInt   *imax, *ai, *ailen;
  PetscInt   *aj, nonew = a->nonew, lastcol = -1;
  MatScalar  *ap, value, *aa;
  PetscBool   ignorezeroentries = a->ignorezeroentries;
  PetscBool   roworiented       = a->roworiented;

  PetscFunctionBegin;
  MatCheckPreallocated(A, 1);
  imax  = a->imax;
  ai    = a->i;
  ailen = a->ilen;
  aj    = a->j;
  aa    = a->a;

  for (k = 0; k < m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    PetscCheck(row < A->rmap->n, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_OUTOFRANGE, "Row too large");
    rp   = aj + ai[row];
    ap   = aa + ai[row];
    rmax = imax[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l = 0; l < n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      PetscCheck(in[l] < A->cmap->n, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_OUTOFRANGE, "Column too large");
      col = in[l];
      if (roworiented) value = v[l + k * n];
      else value = v[k + l * m];

      if (value == 0.0 && ignorezeroentries && (is == ADD_VALUES)) continue;

      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high - low > 5) {
        t = (low + high) / 2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i = low; i < high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (is == ADD_VALUES) ap[i] += value;
          else ap[i] = value;
          goto noinsert;
        }
      }
      if (value == 0.0 && ignorezeroentries) goto noinsert;
      if (nonew == 1) goto noinsert;
      PetscCheck(nonew != -1, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_OUTOFRANGE, "Inserting a new nonzero in the matrix");
      MatSeqXAIJReallocateAIJ(A, A->rmap->n, 1, nrow, row, col, rmax, aa, ai, aj, rp, ap, imax, nonew, MatScalar);
      N = nrow++ - 1;
      a->nz++;
      high++;
      /* shift up all the later entries in this row */
      for (ii = N; ii >= i; ii--) {
        rp[ii + 1] = rp[ii];
        ap[ii + 1] = ap[ii];
      }
      rp[i] = col;
      ap[i] = value;
    noinsert:;
      low = i + 1;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturnVoid();
}
/* Undefining these here since they were redefined from their original definition above! No
 * other PETSc functions should be defined past this point, as it is impossible to recover the
 * original definitions */
#undef PetscCall
#undef SETERRQ
