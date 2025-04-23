#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a  = (Mat_SeqBAIJ *)A->data;
  PetscInt           n  = a->mbs;
  const PetscInt    *ai = a->i, *aj = a->j;
  const PetscInt    *diag = a->diag;
  const MatScalar   *aa   = a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ)
  {
    static PetscScalar w[2000]; /* very BAD need to fix */
    fortransolvebaij4_(&n, x, ai, aj, diag, aa, b, w);
  }
#elif defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJUNROLL)
  fortransolvebaij4unroll_(&n, x, ai, aj, diag, aa, b);
#else
  {
    PetscScalar      s1, s2, s3, s4, x1, x2, x3, x4;
    const MatScalar *v;
    PetscInt         jdx, idt, idx, nz, i, ai16;
    const PetscInt  *vi;

    /* forward solve the lower triangular */
    idx  = 0;
    x[0] = b[0];
    x[1] = b[1];
    x[2] = b[2];
    x[3] = b[3];
    for (i = 1; i < n; i++) {
      v  = aa + 16 * ai[i];
      vi = aj + ai[i];
      nz = diag[i] - ai[i];
      idx += 4;
      s1 = b[idx];
      s2 = b[1 + idx];
      s3 = b[2 + idx];
      s4 = b[3 + idx];
      while (nz--) {
        jdx = 4 * (*vi++);
        x1  = x[jdx];
        x2  = x[1 + jdx];
        x3  = x[2 + jdx];
        x4  = x[3 + jdx];
        s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
        s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
        s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
        s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
        v += 16;
      }
      x[idx]     = s1;
      x[1 + idx] = s2;
      x[2 + idx] = s3;
      x[3 + idx] = s4;
    }
    /* backward solve the upper triangular */
    idt = 4 * (n - 1);
    for (i = n - 1; i >= 0; i--) {
      ai16 = 16 * diag[i];
      v    = aa + ai16 + 16;
      vi   = aj + diag[i] + 1;
      nz   = ai[i + 1] - diag[i] - 1;
      s1   = x[idt];
      s2   = x[1 + idt];
      s3   = x[2 + idt];
      s4   = x[3 + idt];
      while (nz--) {
        idx = 4 * (*vi++);
        x1  = x[idx];
        x2  = x[1 + idx];
        x3  = x[2 + idx];
        x4  = x[3 + idx];
        s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
        s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
        s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
        s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
        v += 16;
      }
      v          = aa + ai16;
      x[idt]     = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
      x[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
      x[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
      x[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
      idt -= 4;
    }
  }
#endif

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, k, nz, idx, jdx, idt;
  const PetscInt     bs = A->rmap->bs, bs2 = a->bs2;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar        s1, s2, s3, s4, x1, x2, x3, x4;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx];
  x[1] = b[1 + idx];
  x[2] = b[2 + idx];
  x[3] = b[3 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + bs2 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = bs * i;
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    for (k = 0; k < nz; k++) {
      jdx = bs * vi[k];
      x1  = x[jdx];
      x2  = x[1 + jdx];
      x3  = x[2 + jdx];
      x4  = x[3 + jdx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;

      v += bs2;
    }

    x[idx]     = s1;
    x[1 + idx] = s2;
    x[2 + idx] = s3;
    x[3 + idx] = s4;
  }

  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + bs2 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = bs * i;
    s1  = x[idt];
    s2  = x[1 + idt];
    s3  = x[2 + idt];
    s4  = x[3 + idt];

    for (k = 0; k < nz; k++) {
      idx = bs * vi[k];
      x1  = x[idx];
      x2  = x[1 + idx];
      x3  = x[2 + idx];
      x4  = x[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;

      v += bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]     = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    x[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    x[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    x[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
  }

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * bs2 * (a->nz) - bs * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}
