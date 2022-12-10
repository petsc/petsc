#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSolve_SeqBAIJ_N_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout;
  const PetscInt     n = a->mbs, *ai = a->i, *aj = a->j, *vi;
  PetscInt           i, nz;
  const PetscInt     bs = A->rmap->bs, bs2 = a->bs2;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, *s, *t, *ls;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  PetscCall(PetscArraycpy(t, b + bs * (*r++), bs));
  for (i = 1; i < n; i++) {
    v  = aa + bs2 * ai[i];
    vi = aj + ai[i];
    nz = a->diag[i] - ai[i];
    s  = t + bs * i;
    PetscCall(PetscArraycpy(s, b + bs * (*r++), bs));
    while (nz--) {
      PetscKernel_v_gets_v_minus_A_times_w(bs, s, v, t + bs * (*vi++));
      v += bs2;
    }
  }
  /* backward solve the upper triangular */
  ls = a->solve_work + A->cmap->n;
  for (i = n - 1; i >= 0; i--) {
    v  = aa + bs2 * (a->diag[i] + 1);
    vi = aj + a->diag[i] + 1;
    nz = ai[i + 1] - a->diag[i] - 1;
    PetscCall(PetscArraycpy(ls, t + i * bs, bs));
    while (nz--) {
      PetscKernel_v_gets_v_minus_A_times_w(bs, ls, v, t + bs * (*vi++));
      v += bs2;
    }
    PetscKernel_w_gets_A_times_v(bs, ls, aa + bs2 * a->diag[i], t + i * bs);
    PetscCall(PetscArraycpy(x + bs * (*c--), t + i * bs, bs));
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * (a->bs2) * (a->nz) - A->rmap->bs * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_7_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *ai = a->i, *aj = a->j;
  const PetscInt    *rout, *cout, *diag = a->diag, *vi, n = a->mbs;
  PetscInt           i, nz, idx, idt, idc;
  const MatScalar   *aa = a->a, *v;
  PetscScalar        s1, s2, s3, s4, s5, s6, s7, x1, x2, x3, x4, x5, x6, x7, *x, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 7 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  t[4] = b[4 + idx];
  t[5] = b[5 + idx];
  t[6] = b[6 + idx];

  for (i = 1; i < n; i++) {
    v   = aa + 49 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 7 * (*r++);
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    s5  = b[4 + idx];
    s6  = b[5 + idx];
    s7  = b[6 + idx];
    while (nz--) {
      idx = 7 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      x7  = t[6 + idx];
      s1 -= v[0] * x1 + v[7] * x2 + v[14] * x3 + v[21] * x4 + v[28] * x5 + v[35] * x6 + v[42] * x7;
      s2 -= v[1] * x1 + v[8] * x2 + v[15] * x3 + v[22] * x4 + v[29] * x5 + v[36] * x6 + v[43] * x7;
      s3 -= v[2] * x1 + v[9] * x2 + v[16] * x3 + v[23] * x4 + v[30] * x5 + v[37] * x6 + v[44] * x7;
      s4 -= v[3] * x1 + v[10] * x2 + v[17] * x3 + v[24] * x4 + v[31] * x5 + v[38] * x6 + v[45] * x7;
      s5 -= v[4] * x1 + v[11] * x2 + v[18] * x3 + v[25] * x4 + v[32] * x5 + v[39] * x6 + v[46] * x7;
      s6 -= v[5] * x1 + v[12] * x2 + v[19] * x3 + v[26] * x4 + v[33] * x5 + v[40] * x6 + v[47] * x7;
      s7 -= v[6] * x1 + v[13] * x2 + v[20] * x3 + v[27] * x4 + v[34] * x5 + v[41] * x6 + v[48] * x7;
      v += 49;
    }
    idx        = 7 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
    t[5 + idx] = s6;
    t[6 + idx] = s7;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 49 * diag[i] + 49;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 7 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    s6  = t[5 + idt];
    s7  = t[6 + idt];
    while (nz--) {
      idx = 7 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      x7  = t[6 + idx];
      s1 -= v[0] * x1 + v[7] * x2 + v[14] * x3 + v[21] * x4 + v[28] * x5 + v[35] * x6 + v[42] * x7;
      s2 -= v[1] * x1 + v[8] * x2 + v[15] * x3 + v[22] * x4 + v[29] * x5 + v[36] * x6 + v[43] * x7;
      s3 -= v[2] * x1 + v[9] * x2 + v[16] * x3 + v[23] * x4 + v[30] * x5 + v[37] * x6 + v[44] * x7;
      s4 -= v[3] * x1 + v[10] * x2 + v[17] * x3 + v[24] * x4 + v[31] * x5 + v[38] * x6 + v[45] * x7;
      s5 -= v[4] * x1 + v[11] * x2 + v[18] * x3 + v[25] * x4 + v[32] * x5 + v[39] * x6 + v[46] * x7;
      s6 -= v[5] * x1 + v[12] * x2 + v[19] * x3 + v[26] * x4 + v[33] * x5 + v[40] * x6 + v[47] * x7;
      s7 -= v[6] * x1 + v[13] * x2 + v[20] * x3 + v[27] * x4 + v[34] * x5 + v[41] * x6 + v[48] * x7;
      v += 49;
    }
    idc    = 7 * (*c--);
    v      = aa + 49 * diag[i];
    x[idc] = t[idt] = v[0] * s1 + v[7] * s2 + v[14] * s3 + v[21] * s4 + v[28] * s5 + v[35] * s6 + v[42] * s7;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[8] * s2 + v[15] * s3 + v[22] * s4 + v[29] * s5 + v[36] * s6 + v[43] * s7;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[9] * s2 + v[16] * s3 + v[23] * s4 + v[30] * s5 + v[37] * s6 + v[44] * s7;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[10] * s2 + v[17] * s3 + v[24] * s4 + v[31] * s5 + v[38] * s6 + v[45] * s7;
    x[4 + idc] = t[4 + idt] = v[4] * s1 + v[11] * s2 + v[18] * s3 + v[25] * s4 + v[32] * s5 + v[39] * s6 + v[46] * s7;
    x[5 + idc] = t[5 + idt] = v[5] * s1 + v[12] * s2 + v[19] * s3 + v[26] * s4 + v[33] * s5 + v[40] * s6 + v[47] * s7;
    x[6 + idc] = t[6 + idt] = v[6] * s1 + v[13] * s2 + v[20] * s3 + v[27] * s4 + v[34] * s5 + v[41] * s6 + v[48] * s7;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 49 * (a->nz) - 7.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_7(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *ai = a->i, *aj = a->j, *adiag = a->diag;
  const PetscInt     n = a->mbs, *rout, *cout, *vi;
  PetscInt           i, nz, idx, idt, idc, m;
  const MatScalar   *aa = a->a, *v;
  PetscScalar        s1, s2, s3, s4, s5, s6, s7, x1, x2, x3, x4, x5, x6, x7, *x, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  idx  = 7 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  t[4] = b[4 + idx];
  t[5] = b[5 + idx];
  t[6] = b[6 + idx];

  for (i = 1; i < n; i++) {
    v   = aa + 49 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 7 * r[i];
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    s5  = b[4 + idx];
    s6  = b[5 + idx];
    s7  = b[6 + idx];
    for (m = 0; m < nz; m++) {
      idx = 7 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      x7  = t[6 + idx];
      s1 -= v[0] * x1 + v[7] * x2 + v[14] * x3 + v[21] * x4 + v[28] * x5 + v[35] * x6 + v[42] * x7;
      s2 -= v[1] * x1 + v[8] * x2 + v[15] * x3 + v[22] * x4 + v[29] * x5 + v[36] * x6 + v[43] * x7;
      s3 -= v[2] * x1 + v[9] * x2 + v[16] * x3 + v[23] * x4 + v[30] * x5 + v[37] * x6 + v[44] * x7;
      s4 -= v[3] * x1 + v[10] * x2 + v[17] * x3 + v[24] * x4 + v[31] * x5 + v[38] * x6 + v[45] * x7;
      s5 -= v[4] * x1 + v[11] * x2 + v[18] * x3 + v[25] * x4 + v[32] * x5 + v[39] * x6 + v[46] * x7;
      s6 -= v[5] * x1 + v[12] * x2 + v[19] * x3 + v[26] * x4 + v[33] * x5 + v[40] * x6 + v[47] * x7;
      s7 -= v[6] * x1 + v[13] * x2 + v[20] * x3 + v[27] * x4 + v[34] * x5 + v[41] * x6 + v[48] * x7;
      v += 49;
    }
    idx        = 7 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
    t[5 + idx] = s6;
    t[6 + idx] = s7;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 49 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 7 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    s6  = t[5 + idt];
    s7  = t[6 + idt];
    for (m = 0; m < nz; m++) {
      idx = 7 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      x7  = t[6 + idx];
      s1 -= v[0] * x1 + v[7] * x2 + v[14] * x3 + v[21] * x4 + v[28] * x5 + v[35] * x6 + v[42] * x7;
      s2 -= v[1] * x1 + v[8] * x2 + v[15] * x3 + v[22] * x4 + v[29] * x5 + v[36] * x6 + v[43] * x7;
      s3 -= v[2] * x1 + v[9] * x2 + v[16] * x3 + v[23] * x4 + v[30] * x5 + v[37] * x6 + v[44] * x7;
      s4 -= v[3] * x1 + v[10] * x2 + v[17] * x3 + v[24] * x4 + v[31] * x5 + v[38] * x6 + v[45] * x7;
      s5 -= v[4] * x1 + v[11] * x2 + v[18] * x3 + v[25] * x4 + v[32] * x5 + v[39] * x6 + v[46] * x7;
      s6 -= v[5] * x1 + v[12] * x2 + v[19] * x3 + v[26] * x4 + v[33] * x5 + v[40] * x6 + v[47] * x7;
      s7 -= v[6] * x1 + v[13] * x2 + v[20] * x3 + v[27] * x4 + v[34] * x5 + v[41] * x6 + v[48] * x7;
      v += 49;
    }
    idc    = 7 * c[i];
    x[idc] = t[idt] = v[0] * s1 + v[7] * s2 + v[14] * s3 + v[21] * s4 + v[28] * s5 + v[35] * s6 + v[42] * s7;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[8] * s2 + v[15] * s3 + v[22] * s4 + v[29] * s5 + v[36] * s6 + v[43] * s7;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[9] * s2 + v[16] * s3 + v[23] * s4 + v[30] * s5 + v[37] * s6 + v[44] * s7;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[10] * s2 + v[17] * s3 + v[24] * s4 + v[31] * s5 + v[38] * s6 + v[45] * s7;
    x[4 + idc] = t[4 + idt] = v[4] * s1 + v[11] * s2 + v[18] * s3 + v[25] * s4 + v[32] * s5 + v[39] * s6 + v[46] * s7;
    x[5 + idc] = t[5 + idt] = v[5] * s1 + v[12] * s2 + v[19] * s3 + v[26] * s4 + v[33] * s5 + v[40] * s6 + v[47] * s7;
    x[6 + idc] = t[6 + idt] = v[6] * s1 + v[13] * s2 + v[20] * s3 + v[27] * s4 + v[34] * s5 + v[41] * s6 + v[48] * s7;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 49 * (a->nz) - 7.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_6_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout;
  const PetscInt    *diag = a->diag, n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, s5, s6, x1, x2, x3, x4, x5, x6, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 6 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  t[4] = b[4 + idx];
  t[5] = b[5 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 36 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 6 * (*r++);
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    s5  = b[4 + idx];
    s6  = b[5 + idx];
    while (nz--) {
      idx = 6 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      s1 -= v[0] * x1 + v[6] * x2 + v[12] * x3 + v[18] * x4 + v[24] * x5 + v[30] * x6;
      s2 -= v[1] * x1 + v[7] * x2 + v[13] * x3 + v[19] * x4 + v[25] * x5 + v[31] * x6;
      s3 -= v[2] * x1 + v[8] * x2 + v[14] * x3 + v[20] * x4 + v[26] * x5 + v[32] * x6;
      s4 -= v[3] * x1 + v[9] * x2 + v[15] * x3 + v[21] * x4 + v[27] * x5 + v[33] * x6;
      s5 -= v[4] * x1 + v[10] * x2 + v[16] * x3 + v[22] * x4 + v[28] * x5 + v[34] * x6;
      s6 -= v[5] * x1 + v[11] * x2 + v[17] * x3 + v[23] * x4 + v[29] * x5 + v[35] * x6;
      v += 36;
    }
    idx        = 6 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
    t[5 + idx] = s6;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 36 * diag[i] + 36;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 6 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    s6  = t[5 + idt];
    while (nz--) {
      idx = 6 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      s1 -= v[0] * x1 + v[6] * x2 + v[12] * x3 + v[18] * x4 + v[24] * x5 + v[30] * x6;
      s2 -= v[1] * x1 + v[7] * x2 + v[13] * x3 + v[19] * x4 + v[25] * x5 + v[31] * x6;
      s3 -= v[2] * x1 + v[8] * x2 + v[14] * x3 + v[20] * x4 + v[26] * x5 + v[32] * x6;
      s4 -= v[3] * x1 + v[9] * x2 + v[15] * x3 + v[21] * x4 + v[27] * x5 + v[33] * x6;
      s5 -= v[4] * x1 + v[10] * x2 + v[16] * x3 + v[22] * x4 + v[28] * x5 + v[34] * x6;
      s6 -= v[5] * x1 + v[11] * x2 + v[17] * x3 + v[23] * x4 + v[29] * x5 + v[35] * x6;
      v += 36;
    }
    idc    = 6 * (*c--);
    v      = aa + 36 * diag[i];
    x[idc] = t[idt] = v[0] * s1 + v[6] * s2 + v[12] * s3 + v[18] * s4 + v[24] * s5 + v[30] * s6;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[7] * s2 + v[13] * s3 + v[19] * s4 + v[25] * s5 + v[31] * s6;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[8] * s2 + v[14] * s3 + v[20] * s4 + v[26] * s5 + v[32] * s6;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[9] * s2 + v[15] * s3 + v[21] * s4 + v[27] * s5 + v[33] * s6;
    x[4 + idc] = t[4 + idt] = v[4] * s1 + v[10] * s2 + v[16] * s3 + v[22] * s4 + v[28] * s5 + v[34] * s6;
    x[5 + idc] = t[5 + idt] = v[5] * s1 + v[11] * s2 + v[17] * s3 + v[23] * s4 + v[29] * s5 + v[35] * s6;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 36 * (a->nz) - 6.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_6(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, nz, idx, idt, idc, m;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, s5, s6, x1, x2, x3, x4, x5, x6, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  idx  = 6 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  t[4] = b[4 + idx];
  t[5] = b[5 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 36 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 6 * r[i];
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    s5  = b[4 + idx];
    s6  = b[5 + idx];
    for (m = 0; m < nz; m++) {
      idx = 6 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      s1 -= v[0] * x1 + v[6] * x2 + v[12] * x3 + v[18] * x4 + v[24] * x5 + v[30] * x6;
      s2 -= v[1] * x1 + v[7] * x2 + v[13] * x3 + v[19] * x4 + v[25] * x5 + v[31] * x6;
      s3 -= v[2] * x1 + v[8] * x2 + v[14] * x3 + v[20] * x4 + v[26] * x5 + v[32] * x6;
      s4 -= v[3] * x1 + v[9] * x2 + v[15] * x3 + v[21] * x4 + v[27] * x5 + v[33] * x6;
      s5 -= v[4] * x1 + v[10] * x2 + v[16] * x3 + v[22] * x4 + v[28] * x5 + v[34] * x6;
      s6 -= v[5] * x1 + v[11] * x2 + v[17] * x3 + v[23] * x4 + v[29] * x5 + v[35] * x6;
      v += 36;
    }
    idx        = 6 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
    t[5 + idx] = s6;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 36 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 6 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    s6  = t[5 + idt];
    for (m = 0; m < nz; m++) {
      idx = 6 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      x6  = t[5 + idx];
      s1 -= v[0] * x1 + v[6] * x2 + v[12] * x3 + v[18] * x4 + v[24] * x5 + v[30] * x6;
      s2 -= v[1] * x1 + v[7] * x2 + v[13] * x3 + v[19] * x4 + v[25] * x5 + v[31] * x6;
      s3 -= v[2] * x1 + v[8] * x2 + v[14] * x3 + v[20] * x4 + v[26] * x5 + v[32] * x6;
      s4 -= v[3] * x1 + v[9] * x2 + v[15] * x3 + v[21] * x4 + v[27] * x5 + v[33] * x6;
      s5 -= v[4] * x1 + v[10] * x2 + v[16] * x3 + v[22] * x4 + v[28] * x5 + v[34] * x6;
      s6 -= v[5] * x1 + v[11] * x2 + v[17] * x3 + v[23] * x4 + v[29] * x5 + v[35] * x6;
      v += 36;
    }
    idc    = 6 * c[i];
    x[idc] = t[idt] = v[0] * s1 + v[6] * s2 + v[12] * s3 + v[18] * s4 + v[24] * s5 + v[30] * s6;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[7] * s2 + v[13] * s3 + v[19] * s4 + v[25] * s5 + v[31] * s6;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[8] * s2 + v[14] * s3 + v[20] * s4 + v[26] * s5 + v[32] * s6;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[9] * s2 + v[15] * s3 + v[21] * s4 + v[27] * s5 + v[33] * s6;
    x[4 + idc] = t[4 + idt] = v[4] * s1 + v[10] * s2 + v[16] * s3 + v[22] * s4 + v[28] * s5 + v[34] * s6;
    x[5 + idc] = t[5 + idt] = v[5] * s1 + v[11] * s2 + v[17] * s3 + v[23] * s4 + v[29] * s5 + v[35] * s6;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 36 * (a->nz) - 6.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_5_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout, *diag = a->diag;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, s5, x1, x2, x3, x4, x5, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 5 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  t[4] = b[4 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 25 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 5 * (*r++);
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    s5  = b[4 + idx];
    while (nz--) {
      idx = 5 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      s1 -= v[0] * x1 + v[5] * x2 + v[10] * x3 + v[15] * x4 + v[20] * x5;
      s2 -= v[1] * x1 + v[6] * x2 + v[11] * x3 + v[16] * x4 + v[21] * x5;
      s3 -= v[2] * x1 + v[7] * x2 + v[12] * x3 + v[17] * x4 + v[22] * x5;
      s4 -= v[3] * x1 + v[8] * x2 + v[13] * x3 + v[18] * x4 + v[23] * x5;
      s5 -= v[4] * x1 + v[9] * x2 + v[14] * x3 + v[19] * x4 + v[24] * x5;
      v += 25;
    }
    idx        = 5 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 25 * diag[i] + 25;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 5 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    while (nz--) {
      idx = 5 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      s1 -= v[0] * x1 + v[5] * x2 + v[10] * x3 + v[15] * x4 + v[20] * x5;
      s2 -= v[1] * x1 + v[6] * x2 + v[11] * x3 + v[16] * x4 + v[21] * x5;
      s3 -= v[2] * x1 + v[7] * x2 + v[12] * x3 + v[17] * x4 + v[22] * x5;
      s4 -= v[3] * x1 + v[8] * x2 + v[13] * x3 + v[18] * x4 + v[23] * x5;
      s5 -= v[4] * x1 + v[9] * x2 + v[14] * x3 + v[19] * x4 + v[24] * x5;
      v += 25;
    }
    idc    = 5 * (*c--);
    v      = aa + 25 * diag[i];
    x[idc] = t[idt] = v[0] * s1 + v[5] * s2 + v[10] * s3 + v[15] * s4 + v[20] * s5;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[6] * s2 + v[11] * s3 + v[16] * s4 + v[21] * s5;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[7] * s2 + v[12] * s3 + v[17] * s4 + v[22] * s5;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[8] * s2 + v[13] * s3 + v[18] * s4 + v[23] * s5;
    x[4 + idc] = t[4 + idt] = v[4] * s1 + v[9] * s2 + v[14] * s3 + v[19] * s4 + v[24] * s5;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 25 * (a->nz) - 5.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_5(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, nz, idx, idt, idc, m;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, s5, x1, x2, x3, x4, x5, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  idx  = 5 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  t[4] = b[4 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 25 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 5 * r[i];
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    s5  = b[4 + idx];
    for (m = 0; m < nz; m++) {
      idx = 5 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      s1 -= v[0] * x1 + v[5] * x2 + v[10] * x3 + v[15] * x4 + v[20] * x5;
      s2 -= v[1] * x1 + v[6] * x2 + v[11] * x3 + v[16] * x4 + v[21] * x5;
      s3 -= v[2] * x1 + v[7] * x2 + v[12] * x3 + v[17] * x4 + v[22] * x5;
      s4 -= v[3] * x1 + v[8] * x2 + v[13] * x3 + v[18] * x4 + v[23] * x5;
      s5 -= v[4] * x1 + v[9] * x2 + v[14] * x3 + v[19] * x4 + v[24] * x5;
      v += 25;
    }
    idx        = 5 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    t[4 + idx] = s5;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 25 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 5 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    s5  = t[4 + idt];
    for (m = 0; m < nz; m++) {
      idx = 5 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      x5  = t[4 + idx];
      s1 -= v[0] * x1 + v[5] * x2 + v[10] * x3 + v[15] * x4 + v[20] * x5;
      s2 -= v[1] * x1 + v[6] * x2 + v[11] * x3 + v[16] * x4 + v[21] * x5;
      s3 -= v[2] * x1 + v[7] * x2 + v[12] * x3 + v[17] * x4 + v[22] * x5;
      s4 -= v[3] * x1 + v[8] * x2 + v[13] * x3 + v[18] * x4 + v[23] * x5;
      s5 -= v[4] * x1 + v[9] * x2 + v[14] * x3 + v[19] * x4 + v[24] * x5;
      v += 25;
    }
    idc    = 5 * c[i];
    x[idc] = t[idt] = v[0] * s1 + v[5] * s2 + v[10] * s3 + v[15] * s4 + v[20] * s5;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[6] * s2 + v[11] * s3 + v[16] * s4 + v[21] * s5;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[7] * s2 + v[12] * s3 + v[17] * s4 + v[22] * s5;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[8] * s2 + v[13] * s3 + v[18] * s4 + v[23] * s5;
    x[4 + idc] = t[4 + idt] = v[4] * s1 + v[9] * s2 + v[14] * s3 + v[19] * s4 + v[24] * s5;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 25 * (a->nz) - 5.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const PetscInt    *r, *c, *diag = a->diag, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, x1, x2, x3, x4, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 4 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 16 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 4 * (*r++);
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    while (nz--) {
      idx = 4 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idx        = 4 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 16 * diag[i] + 16;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 4 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    while (nz--) {
      idx = 4 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idc    = 4 * (*c--);
    v      = aa + 16 * diag[i];
    x[idc] = t[idt] = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, nz, idx, idt, idc, m;
  const PetscInt    *r, *c, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, x1, x2, x3, x4, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  idx  = 4 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 16 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 4 * r[i];
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    for (m = 0; m < nz; m++) {
      idx = 4 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idx        = 4 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 16 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 4 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    for (m = 0; m < nz; m++) {
      idx = 4 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idc    = 4 * c[i];
    x[idc] = t[idt] = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_Demotion(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const PetscInt    *r, *c, *diag = a->diag, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  MatScalar          s1, s2, s3, s4, x1, x2, x3, x4, *t;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = (MatScalar *)a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 4 * (*r++);
  t[0] = (MatScalar)b[idx];
  t[1] = (MatScalar)b[1 + idx];
  t[2] = (MatScalar)b[2 + idx];
  t[3] = (MatScalar)b[3 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 16 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 4 * (*r++);
    s1  = (MatScalar)b[idx];
    s2  = (MatScalar)b[1 + idx];
    s3  = (MatScalar)b[2 + idx];
    s4  = (MatScalar)b[3 + idx];
    while (nz--) {
      idx = 4 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idx        = 4 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 16 * diag[i] + 16;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 4 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    while (nz--) {
      idx = 4 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idc        = 4 * (*c--);
    v          = aa + 16 * diag[i];
    t[idt]     = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    t[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    t[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    t[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
    x[idc]     = (PetscScalar)t[idt];
    x[1 + idc] = (PetscScalar)t[1 + idt];
    x[2 + idc] = (PetscScalar)t[2 + idt];
    x[3 + idc] = (PetscScalar)t[3 + idt];
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_SSE)

  #include PETSC_HAVE_SSE

PetscErrorCode MatSolve_SeqBAIJ_4_SSE_Demotion(Mat A, Vec bb, Vec xx)
{
  /*
     Note: This code uses demotion of double
     to float when performing the mixed-mode computation.
     This may not be numerically reasonable for all applications.
  */
  Mat_SeqBAIJ    *a     = (Mat_SeqBAIJ *)A->data;
  IS              iscol = a->col, isrow = a->row;
  PetscInt        i, n = a->mbs, *vi, *ai = a->i, *aj = a->j, nz, idx, idt, idc, ai16;
  const PetscInt *r, *c, *diag = a->diag, *rout, *cout;
  MatScalar      *aa = a->a, *v;
  PetscScalar    *x, *b, *t;

  /* Make space in temp stack for 16 Byte Aligned arrays */
  float         ssealignedspace[11], *tmps, *tmpx;
  unsigned long offset;

  PetscFunctionBegin;
  SSE_SCOPE_BEGIN;

  offset = (unsigned long)ssealignedspace % 16;
  if (offset) offset = (16 - offset) / 4;
  tmps = &ssealignedspace[offset];
  tmpx = &ssealignedspace[offset + 4];
  PREFETCH_NTA(aa + 16 * ai[1]);

  PetscCall(VecGetArray(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 4 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  v    = aa + 16 * ai[1];

  for (i = 1; i < n;) {
    PREFETCH_NTA(&v[8]);
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 4 * (*r++);

    /* Demote sum from double to float */
    CONVERT_DOUBLE4_FLOAT4(tmps, &b[idx]);
    LOAD_PS(tmps, XMM7);

    while (nz--) {
      PREFETCH_NTA(&v[16]);
      idx = 4 * (*vi++);

      /* Demote solution (so far) from double to float */
      CONVERT_DOUBLE4_FLOAT4(tmpx, &x[idx]);

      /* 4x4 Matrix-Vector product with negative accumulation: */
      SSE_INLINE_BEGIN_2(tmpx, v)
      SSE_LOAD_PS(SSE_ARG_1, FLOAT_0, XMM6)

      /* First Column */
      SSE_COPY_PS(XMM0, XMM6)
      SSE_SHUFFLE(XMM0, XMM0, 0x00)
      SSE_MULT_PS_M(XMM0, SSE_ARG_2, FLOAT_0)
      SSE_SUB_PS(XMM7, XMM0)

      /* Second Column */
      SSE_COPY_PS(XMM1, XMM6)
      SSE_SHUFFLE(XMM1, XMM1, 0x55)
      SSE_MULT_PS_M(XMM1, SSE_ARG_2, FLOAT_4)
      SSE_SUB_PS(XMM7, XMM1)

      SSE_PREFETCH_NTA(SSE_ARG_2, FLOAT_24)

      /* Third Column */
      SSE_COPY_PS(XMM2, XMM6)
      SSE_SHUFFLE(XMM2, XMM2, 0xAA)
      SSE_MULT_PS_M(XMM2, SSE_ARG_2, FLOAT_8)
      SSE_SUB_PS(XMM7, XMM2)

      /* Fourth Column */
      SSE_COPY_PS(XMM3, XMM6)
      SSE_SHUFFLE(XMM3, XMM3, 0xFF)
      SSE_MULT_PS_M(XMM3, SSE_ARG_2, FLOAT_12)
      SSE_SUB_PS(XMM7, XMM3)
      SSE_INLINE_END_2

      v += 16;
    }
    idx = 4 * i;
    v   = aa + 16 * ai[++i];
    PREFETCH_NTA(v);
    STORE_PS(tmps, XMM7);

    /* Promote result from float to double */
    CONVERT_FLOAT4_DOUBLE4(&t[idx], tmps);
  }
  /* backward solve the upper triangular */
  idt  = 4 * (n - 1);
  ai16 = 16 * diag[n - 1];
  v    = aa + ai16 + 16;
  for (i = n - 1; i >= 0;) {
    PREFETCH_NTA(&v[8]);
    vi = aj + diag[i] + 1;
    nz = ai[i + 1] - diag[i] - 1;

    /* Demote accumulator from double to float */
    CONVERT_DOUBLE4_FLOAT4(tmps, &t[idt]);
    LOAD_PS(tmps, XMM7);

    while (nz--) {
      PREFETCH_NTA(&v[16]);
      idx = 4 * (*vi++);

      /* Demote solution (so far) from double to float */
      CONVERT_DOUBLE4_FLOAT4(tmpx, &t[idx]);

      /* 4x4 Matrix-Vector Product with negative accumulation: */
      SSE_INLINE_BEGIN_2(tmpx, v)
      SSE_LOAD_PS(SSE_ARG_1, FLOAT_0, XMM6)

      /* First Column */
      SSE_COPY_PS(XMM0, XMM6)
      SSE_SHUFFLE(XMM0, XMM0, 0x00)
      SSE_MULT_PS_M(XMM0, SSE_ARG_2, FLOAT_0)
      SSE_SUB_PS(XMM7, XMM0)

      /* Second Column */
      SSE_COPY_PS(XMM1, XMM6)
      SSE_SHUFFLE(XMM1, XMM1, 0x55)
      SSE_MULT_PS_M(XMM1, SSE_ARG_2, FLOAT_4)
      SSE_SUB_PS(XMM7, XMM1)

      SSE_PREFETCH_NTA(SSE_ARG_2, FLOAT_24)

      /* Third Column */
      SSE_COPY_PS(XMM2, XMM6)
      SSE_SHUFFLE(XMM2, XMM2, 0xAA)
      SSE_MULT_PS_M(XMM2, SSE_ARG_2, FLOAT_8)
      SSE_SUB_PS(XMM7, XMM2)

      /* Fourth Column */
      SSE_COPY_PS(XMM3, XMM6)
      SSE_SHUFFLE(XMM3, XMM3, 0xFF)
      SSE_MULT_PS_M(XMM3, SSE_ARG_2, FLOAT_12)
      SSE_SUB_PS(XMM7, XMM3)
      SSE_INLINE_END_2
      v += 16;
    }
    v    = aa + ai16;
    ai16 = 16 * diag[--i];
    PREFETCH_NTA(aa + ai16 + 16);
    /*
       Scale the result by the diagonal 4x4 block,
       which was inverted as part of the factorization
    */
    SSE_INLINE_BEGIN_3(v, tmps, aa + ai16)
    /* First Column */
    SSE_COPY_PS(XMM0, XMM7)
    SSE_SHUFFLE(XMM0, XMM0, 0x00)
    SSE_MULT_PS_M(XMM0, SSE_ARG_1, FLOAT_0)

    /* Second Column */
    SSE_COPY_PS(XMM1, XMM7)
    SSE_SHUFFLE(XMM1, XMM1, 0x55)
    SSE_MULT_PS_M(XMM1, SSE_ARG_1, FLOAT_4)
    SSE_ADD_PS(XMM0, XMM1)

    SSE_PREFETCH_NTA(SSE_ARG_3, FLOAT_24)

    /* Third Column */
    SSE_COPY_PS(XMM2, XMM7)
    SSE_SHUFFLE(XMM2, XMM2, 0xAA)
    SSE_MULT_PS_M(XMM2, SSE_ARG_1, FLOAT_8)
    SSE_ADD_PS(XMM0, XMM2)

    /* Fourth Column */
    SSE_COPY_PS(XMM3, XMM7)
    SSE_SHUFFLE(XMM3, XMM3, 0xFF)
    SSE_MULT_PS_M(XMM3, SSE_ARG_1, FLOAT_12)
    SSE_ADD_PS(XMM0, XMM3)

    SSE_STORE_PS(SSE_ARG_2, FLOAT_0, XMM0)
    SSE_INLINE_END_3

    /* Promote solution from float to double */
    CONVERT_FLOAT4_DOUBLE4(&t[idt], tmps);

    /* Apply reordering to t and stream into x.    */
    /* This way, x doesn't pollute the cache.      */
    /* Be careful with size: 2 doubles = 4 floats! */
    idc = 4 * (*c--);
    SSE_INLINE_BEGIN_2((float *)&t[idt], (float *)&x[idc])
    /*  x[idc]   = t[idt];   x[1+idc] = t[1+idc]; */
    SSE_LOAD_PS(SSE_ARG_1, FLOAT_0, XMM0)
    SSE_STREAM_PS(SSE_ARG_2, FLOAT_0, XMM0)
    /*  x[idc+2] = t[idt+2]; x[3+idc] = t[3+idc]; */
    SSE_LOAD_PS(SSE_ARG_1, FLOAT_4, XMM1)
    SSE_STREAM_PS(SSE_ARG_2, FLOAT_4, XMM1)
    SSE_INLINE_END_2
    v = aa + ai16 + 16;
    idt -= 4;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArray(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  SSE_SCOPE_END;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif

PetscErrorCode MatSolve_SeqBAIJ_3_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const PetscInt    *r, *c, *diag = a->diag, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, x1, x2, x3, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 3 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 9 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 3 * (*r++);
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    while (nz--) {
      idx = 3 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      s1 -= v[0] * x1 + v[3] * x2 + v[6] * x3;
      s2 -= v[1] * x1 + v[4] * x2 + v[7] * x3;
      s3 -= v[2] * x1 + v[5] * x2 + v[8] * x3;
      v += 9;
    }
    idx        = 3 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 9 * diag[i] + 9;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 3 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    while (nz--) {
      idx = 3 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      s1 -= v[0] * x1 + v[3] * x2 + v[6] * x3;
      s2 -= v[1] * x1 + v[4] * x2 + v[7] * x3;
      s3 -= v[2] * x1 + v[5] * x2 + v[8] * x3;
      v += 9;
    }
    idc    = 3 * (*c--);
    v      = aa + 9 * diag[i];
    x[idc] = t[idt] = v[0] * s1 + v[3] * s2 + v[6] * s3;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[4] * s2 + v[7] * s3;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[5] * s2 + v[8] * s3;
  }
  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 9 * (a->nz) - 3.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_3(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, nz, idx, idt, idc, m;
  const PetscInt    *r, *c, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, x1, x2, x3, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  idx  = 3 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 9 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 3 * r[i];
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    for (m = 0; m < nz; m++) {
      idx = 3 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      s1 -= v[0] * x1 + v[3] * x2 + v[6] * x3;
      s2 -= v[1] * x1 + v[4] * x2 + v[7] * x3;
      s3 -= v[2] * x1 + v[5] * x2 + v[8] * x3;
      v += 9;
    }
    idx        = 3 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 9 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 3 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    for (m = 0; m < nz; m++) {
      idx = 3 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      s1 -= v[0] * x1 + v[3] * x2 + v[6] * x3;
      s2 -= v[1] * x1 + v[4] * x2 + v[7] * x3;
      s3 -= v[2] * x1 + v[5] * x2 + v[8] * x3;
      v += 9;
    }
    idc    = 3 * c[i];
    x[idc] = t[idt] = v[0] * s1 + v[3] * s2 + v[6] * s3;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[4] * s2 + v[7] * s3;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[5] * s2 + v[8] * s3;
  }
  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 9 * (a->nz) - 3.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_2_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const PetscInt    *r, *c, *diag = a->diag, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, x1, x2, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 2 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 4 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 2 * (*r++);
    s1  = b[idx];
    s2  = b[1 + idx];
    while (nz--) {
      idx = 2 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    idx        = 2 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 4 * diag[i] + 4;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 2 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    while (nz--) {
      idx = 2 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    idc    = 2 * (*c--);
    v      = aa + 4 * diag[i];
    x[idc] = t[idt] = v[0] * s1 + v[2] * s2;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[3] * s2;
  }
  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 4 * (a->nz) - 2.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_2(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, nz, idx, jdx, idt, idc, m;
  const PetscInt    *r, *c, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, x1, x2, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  idx  = 2 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 4 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 2 * r[i];
    s1  = b[idx];
    s2  = b[1 + idx];
    for (m = 0; m < nz; m++) {
      jdx = 2 * vi[m];
      x1  = t[jdx];
      x2  = t[1 + jdx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    idx        = 2 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 4 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 2 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    for (m = 0; m < nz; m++) {
      idx = 2 * vi[m];
      x1  = t[idx];
      x2  = t[1 + idx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    idc    = 2 * c[i];
    x[idc] = t[idt] = v[0] * s1 + v[2] * s2;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[3] * s2;
  }
  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 4 * (a->nz) - 2.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_1_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz;
  const PetscInt    *r, *c, *diag = a->diag, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  t[0] = b[*r++];
  for (i = 1; i < n; i++) {
    v  = aa + ai[i];
    vi = aj + ai[i];
    nz = diag[i] - ai[i];
    s1 = b[*r++];
    while (nz--) s1 -= (*v++) * t[*vi++];
    t[i] = s1;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v  = aa + diag[i] + 1;
    vi = aj + diag[i] + 1;
    nz = ai[i + 1] - diag[i] - 1;
    s1 = t[i];
    while (nz--) s1 -= (*v++) * t[*vi++];
    x[*c--] = t[i] = aa[diag[i]] * s1;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 1 * (a->nz) - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_1(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  PetscInt           i, n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag, nz;
  const PetscInt    *rout, *cout, *r, *c;
  PetscScalar       *x, *tmp, sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a, *v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  tmp = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  tmp[0] = b[r[0]];
  v      = aa;
  vi     = aj;
  for (i = 1; i < n; i++) {
    nz  = ai[i + 1] - ai[i];
    sum = b[r[i]];
    PetscSparseDenseMinusDot(sum, tmp, v, vi, nz);
    tmp[i] = sum;
    v += nz;
    vi += nz;
  }

  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + adiag[i + 1] + 1;
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    sum = tmp[i];
    PetscSparseDenseMinusDot(sum, tmp, v, vi, nz);
    x[c[i]] = tmp[i] = sum * v[nz]; /* v[nz] = aa[adiag[i]] */
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * a->nz - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}
