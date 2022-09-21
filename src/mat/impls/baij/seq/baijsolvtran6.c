#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSolveTranspose_SeqBAIJ_6_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout;
  const PetscInt    *diag = a->diag, n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, ii, ic, ir, oidx;
  const MatScalar   *aa = a->a, *v;
  PetscScalar        s1, s2, s3, s4, s5, s6, x1, x2, x3, x4, x5, x6, *x, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i = 0; i < n; i++) {
    ic        = 6 * c[i];
    t[ii]     = b[ic];
    t[ii + 1] = b[ic + 1];
    t[ii + 2] = b[ic + 2];
    t[ii + 3] = b[ic + 3];
    t[ii + 4] = b[ic + 4];
    t[ii + 5] = b[ic + 5];
    ii += 6;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i = 0; i < n; i++) {
    v = aa + 36 * diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];
    x2 = t[1 + idx];
    x3 = t[2 + idx];
    x4 = t[3 + idx];
    x5 = t[4 + idx];
    x6 = t[5 + idx];
    s1 = v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5 + v[5] * x6;
    s2 = v[6] * x1 + v[7] * x2 + v[8] * x3 + v[9] * x4 + v[10] * x5 + v[11] * x6;
    s3 = v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4 + v[16] * x5 + v[17] * x6;
    s4 = v[18] * x1 + v[19] * x2 + v[20] * x3 + v[21] * x4 + v[22] * x5 + v[23] * x6;
    s5 = v[24] * x1 + v[25] * x2 + v[26] * x3 + v[27] * x4 + v[28] * x5 + v[29] * x6;
    s6 = v[30] * x1 + v[31] * x2 + v[32] * x3 + v[33] * x4 + v[34] * x5 + v[35] * x6;
    v += 36;

    vi = aj + diag[i] + 1;
    nz = ai[i + 1] - diag[i] - 1;
    while (nz--) {
      oidx = 6 * (*vi++);
      t[oidx] -= v[0] * s1 + v[1] * s2 + v[2] * s3 + v[3] * s4 + v[4] * s5 + v[5] * s6;
      t[oidx + 1] -= v[6] * s1 + v[7] * s2 + v[8] * s3 + v[9] * s4 + v[10] * s5 + v[11] * s6;
      t[oidx + 2] -= v[12] * s1 + v[13] * s2 + v[14] * s3 + v[15] * s4 + v[16] * s5 + v[17] * s6;
      t[oidx + 3] -= v[18] * s1 + v[19] * s2 + v[20] * s3 + v[21] * s4 + v[22] * s5 + v[23] * s6;
      t[oidx + 4] -= v[24] * s1 + v[25] * s2 + v[26] * s3 + v[27] * s4 + v[28] * s5 + v[29] * s6;
      t[oidx + 5] -= v[30] * s1 + v[31] * s2 + v[32] * s3 + v[33] * s4 + v[34] * s5 + v[35] * s6;
      v += 36;
    }
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
    t[5 + idx] = s6;
    idx += 6;
  }
  /* backward solve the L^T */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 36 * diag[i] - 36;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 6 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    s6  = t[5 + idt];
    while (nz--) {
      idx = 6 * (*vi--);
      t[idx] -= v[0] * s1 + v[1] * s2 + v[2] * s3 + v[3] * s4 + v[4] * s5 + v[5] * s6;
      t[idx + 1] -= v[6] * s1 + v[7] * s2 + v[8] * s3 + v[9] * s4 + v[10] * s5 + v[11] * s6;
      t[idx + 2] -= v[12] * s1 + v[13] * s2 + v[14] * s3 + v[15] * s4 + v[16] * s5 + v[17] * s6;
      t[idx + 3] -= v[18] * s1 + v[19] * s2 + v[20] * s3 + v[21] * s4 + v[22] * s5 + v[23] * s6;
      t[idx + 4] -= v[24] * s1 + v[25] * s2 + v[26] * s3 + v[27] * s4 + v[28] * s5 + v[29] * s6;
      t[idx + 5] -= v[30] * s1 + v[31] * s2 + v[32] * s3 + v[33] * s4 + v[34] * s5 + v[35] * s6;
      v -= 36;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i = 0; i < n; i++) {
    ir        = 6 * r[i];
    x[ir]     = t[ii];
    x[ir + 1] = t[ii + 1];
    x[ir + 2] = t[ii + 2];
    x[ir + 3] = t[ii + 3];
    x[ir + 4] = t[ii + 4];
    x[ir + 5] = t[ii + 5];
    ii += 6;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 36 * (a->nz) - 6.0 * A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_6(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *diag = a->diag;
  const PetscInt    *r, *c, *rout, *cout;
  PetscInt           nz, idx, idt, j, i, oidx, ii, ic, ir;
  const PetscInt     bs = A->rmap->bs, bs2 = a->bs2;
  const MatScalar   *aa = a->a, *v;
  PetscScalar        s1, s2, s3, s4, s5, s6, x1, x2, x3, x4, x5, x6, *x, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* copy b into temp work space according to permutation */
  for (i = 0; i < n; i++) {
    ii        = bs * i;
    ic        = bs * c[i];
    t[ii]     = b[ic];
    t[ii + 1] = b[ic + 1];
    t[ii + 2] = b[ic + 2];
    t[ii + 3] = b[ic + 3];
    t[ii + 4] = b[ic + 4];
    t[ii + 5] = b[ic + 5];
  }

  /* forward solve the U^T */
  idx = 0;
  for (i = 0; i < n; i++) {
    v = aa + bs2 * diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];
    x2 = t[1 + idx];
    x3 = t[2 + idx];
    x4 = t[3 + idx];
    x5 = t[4 + idx];
    x6 = t[5 + idx];
    s1 = v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4 + v[4] * x5 + v[5] * x6;
    s2 = v[6] * x1 + v[7] * x2 + v[8] * x3 + v[9] * x4 + v[10] * x5 + v[11] * x6;
    s3 = v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4 + v[16] * x5 + v[17] * x6;
    s4 = v[18] * x1 + v[19] * x2 + v[20] * x3 + v[21] * x4 + v[22] * x5 + v[23] * x6;
    s5 = v[24] * x1 + v[25] * x2 + v[26] * x3 + v[27] * x4 + v[28] * x5 + v[29] * x6;
    s6 = v[30] * x1 + v[31] * x2 + v[32] * x3 + v[33] * x4 + v[34] * x5 + v[35] * x6;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i + 1] - 1;
    for (j = 0; j > -nz; j--) {
      oidx = bs * vi[j];
      t[oidx] -= v[0] * s1 + v[1] * s2 + v[2] * s3 + v[3] * s4 + v[4] * s5 + v[5] * s6;
      t[oidx + 1] -= v[6] * s1 + v[7] * s2 + v[8] * s3 + v[9] * s4 + v[10] * s5 + v[11] * s6;
      t[oidx + 2] -= v[12] * s1 + v[13] * s2 + v[14] * s3 + v[15] * s4 + v[16] * s5 + v[17] * s6;
      t[oidx + 3] -= v[18] * s1 + v[19] * s2 + v[20] * s3 + v[21] * s4 + v[22] * s5 + v[23] * s6;
      t[oidx + 4] -= v[24] * s1 + v[25] * s2 + v[26] * s3 + v[27] * s4 + v[28] * s5 + v[29] * s6;
      t[oidx + 5] -= v[30] * s1 + v[31] * s2 + v[32] * s3 + v[33] * s4 + v[34] * s5 + v[35] * s6;
      v -= bs2;
    }
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
    t[5 + idx] = s6;
    idx += bs;
  }
  /* backward solve the L^T */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + bs2 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idt = bs * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    s6  = t[5 + idt];
    for (j = 0; j < nz; j++) {
      idx = bs * vi[j];
      t[idx] -= v[0] * s1 + v[1] * s2 + v[2] * s3 + v[3] * s4 + v[4] * s5 + v[5] * s6;
      t[idx + 1] -= v[6] * s1 + v[7] * s2 + v[8] * s3 + v[9] * s4 + v[10] * s5 + v[11] * s6;
      t[idx + 2] -= v[12] * s1 + v[13] * s2 + v[14] * s3 + v[15] * s4 + v[16] * s5 + v[17] * s6;
      t[idx + 3] -= v[18] * s1 + v[19] * s2 + v[20] * s3 + v[21] * s4 + v[22] * s5 + v[23] * s6;
      t[idx + 4] -= v[24] * s1 + v[25] * s2 + v[26] * s3 + v[27] * s4 + v[28] * s5 + v[29] * s6;
      t[idx + 5] -= v[30] * s1 + v[31] * s2 + v[32] * s3 + v[33] * s4 + v[34] * s5 + v[35] * s6;
      v += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i = 0; i < n; i++) {
    ii        = bs * i;
    ir        = bs * r[i];
    x[ir]     = t[ii];
    x[ir + 1] = t[ii + 1];
    x[ir + 2] = t[ii + 2];
    x[ir + 3] = t[ii + 3];
    x[ir + 4] = t[ii + 4];
    x[ir + 5] = t[ii + 5];
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * bs2 * (a->nz) - bs * A->cmap->n));
  PetscFunctionReturn(0);
}
