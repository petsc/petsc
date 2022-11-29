#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/* ----------------------------------------------------------- */
PetscErrorCode MatSolveTranspose_SeqBAIJ_N_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout, *ai = a->i, *aj = a->j, *vi;
  PetscInt           i, nz, j;
  const PetscInt     n = a->mbs, bs = A->rmap->bs, bs2 = a->bs2;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, *t, *ls;
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
  for (i = 0; i < n; i++) {
    for (j = 0; j < bs; j++) t[i * bs + j] = b[c[i] * bs + j];
  }

  /* forward solve the upper triangular transpose */
  ls = a->solve_work + A->cmap->n;
  for (i = 0; i < n; i++) {
    PetscCall(PetscArraycpy(ls, t + i * bs, bs));
    PetscKernel_w_gets_transA_times_v(bs, ls, aa + bs2 * a->diag[i], t + i * bs);
    v  = aa + bs2 * (a->diag[i] + 1);
    vi = aj + a->diag[i] + 1;
    nz = ai[i + 1] - a->diag[i] - 1;
    while (nz--) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs, t + bs * (*vi++), v, t + i * bs);
      v += bs2;
    }
  }

  /* backward solve the lower triangular transpose */
  for (i = n - 1; i >= 0; i--) {
    v  = aa + bs2 * ai[i];
    vi = aj + ai[i];
    nz = a->diag[i] - ai[i];
    while (nz--) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs, t + bs * (*vi++), v, t + i * bs);
      v += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i = 0; i < n; i++) {
    for (j = 0; j < bs; j++) x[bs * r[i] + j] = t[bs * i + j];
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * (a->bs2) * (a->nz) - A->rmap->bs * A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_N(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout;
  const PetscInt     n = a->mbs, *ai = a->i, *aj = a->j, *vi, *diag = a->diag;
  PetscInt           i, j, nz;
  const PetscInt     bs = A->rmap->bs, bs2 = a->bs2;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, *t, *ls;
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
  for (i = 0; i < n; i++) {
    for (j = 0; j < bs; j++) t[i * bs + j] = b[c[i] * bs + j];
  }

  /* forward solve the upper triangular transpose */
  ls = a->solve_work + A->cmap->n;
  for (i = 0; i < n; i++) {
    PetscCall(PetscArraycpy(ls, t + i * bs, bs));
    PetscKernel_w_gets_transA_times_v(bs, ls, aa + bs2 * diag[i], t + i * bs);
    v  = aa + bs2 * (diag[i] - 1);
    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i + 1] - 1;
    for (j = 0; j > -nz; j--) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs, t + bs * (vi[j]), v, t + i * bs);
      v -= bs2;
    }
  }

  /* backward solve the lower triangular transpose */
  for (i = n - 1; i >= 0; i--) {
    v  = aa + bs2 * ai[i];
    vi = aj + ai[i];
    nz = ai[i + 1] - ai[i];
    for (j = 0; j < nz; j++) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs, t + bs * (vi[j]), v, t + i * bs);
      v += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i = 0; i < n; i++) {
    for (j = 0; j < bs; j++) x[bs * r[i] + j] = t[bs * i + j];
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * (a->bs2) * (a->nz) - A->rmap->bs * A->cmap->n));
  PetscFunctionReturn(0);
}
