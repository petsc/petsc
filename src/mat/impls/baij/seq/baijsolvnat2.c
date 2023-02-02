#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *diag = a->diag;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, x1, x2;
  const PetscScalar *b;
  PetscInt           jdx, idt, idx, nz, i;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));

  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[0];
  x[1] = b[1];
  for (i = 1; i < n; i++) {
    v  = aa + 4 * ai[i];
    vi = aj + ai[i];
    nz = diag[i] - ai[i];
    idx += 2;
    s1 = b[idx];
    s2 = b[1 + idx];
    while (nz--) {
      jdx = 2 * (*vi++);
      x1  = x[jdx];
      x2  = x[1 + jdx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    x[idx]     = s1;
    x[1 + idx] = s2;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 4 * diag[i] + 4;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 2 * i;
    s1  = x[idt];
    s2  = x[1 + idt];
    while (nz--) {
      idx = 2 * (*vi++);
      x1  = x[idx];
      x2  = x[1 + idx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    v          = aa + 4 * diag[i];
    x[idt]     = v[0] * s1 + v[2] * s2;
    x[1 + idt] = v[1] * s1 + v[3] * s2;
  }

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 4 * (a->nz) - 2.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, k, nz, idx, idt, jdx;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, x1, x2;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx];
  x[1] = b[1 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 4 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 2 * i;
    s1  = b[idx];
    s2  = b[1 + idx];
    PetscPrefetchBlock(vi + nz, nz, 0, PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v + 4 * nz, 4 * nz, 0, PETSC_PREFETCH_HINT_NTA);
    for (k = 0; k < nz; k++) {
      jdx = 2 * vi[k];
      x1  = x[jdx];
      x2  = x[1 + jdx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    x[idx]     = s1;
    x[1 + idx] = s2;
  }

  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 4 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 2 * i;
    s1  = x[idt];
    s2  = x[1 + idt];
    PetscPrefetchBlock(vi + nz, nz, 0, PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v + 4 * nz, 4 * nz, 0, PETSC_PREFETCH_HINT_NTA);
    for (k = 0; k < nz; k++) {
      idx = 2 * vi[k];
      x1  = x[idx];
      x2  = x[1 + idx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    /* x = inv_diagonal*x */
    x[idt]     = v[0] * s1 + v[2] * s2;
    x[1 + idt] = v[1] * s1 + v[3] * s2;
  }

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 4 * (a->nz) - 2.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatForwardSolve_SeqBAIJ_2_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, k, nz, idx, jdx;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, x1, x2;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx];
  x[1] = b[1 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 4 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 2 * i;
    s1  = b[idx];
    s2  = b[1 + idx];
    PetscPrefetchBlock(vi + nz, nz, 0, PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v + 4 * nz, 4 * nz, 0, PETSC_PREFETCH_HINT_NTA);
    for (k = 0; k < nz; k++) {
      jdx = 2 * vi[k];
      x1  = x[jdx];
      x2  = x[1 + jdx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    x[idx]     = s1;
    x[1 + idx] = s2;
  }

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(4.0 * (a->nz) - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatBackwardSolve_SeqBAIJ_2_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  const PetscInt     n = a->mbs, *vi, *aj = a->j, *adiag = a->diag;
  PetscInt           i, k, nz, idx, idt;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, x1, x2;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));

  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 4 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 2 * i;
    s1  = b[idt];
    s2  = b[1 + idt];
    PetscPrefetchBlock(vi + nz, nz, 0, PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v + 4 * nz, 4 * nz, 0, PETSC_PREFETCH_HINT_NTA);
    for (k = 0; k < nz; k++) {
      idx = 2 * vi[k];
      x1  = x[idx];
      x2  = x[1 + idx];
      s1 -= v[0] * x1 + v[2] * x2;
      s2 -= v[1] * x1 + v[3] * x2;
      v += 4;
    }
    /* x = inv_diagonal*x */
    x[idt]     = v[0] * s1 + v[2] * s2;
    x[1 + idt] = v[1] * s1 + v[3] * s2;
  }

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(4.0 * a->nz - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}
