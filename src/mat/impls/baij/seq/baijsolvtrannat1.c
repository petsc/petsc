#include <../src/mat/impls/baij/seq/baij.h>

PetscErrorCode MatSolveTranspose_SeqBAIJ_1_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  const PetscInt    *adiag = a->diag, *ai = a->i, *aj = a->j, *vi;
  PetscInt           i, n = a->mbs, j;
  PetscInt           nz;
  PetscScalar       *x, *tmp, s1;
  const MatScalar   *aa = a->a, *v;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  tmp = a->solve_work;

  /* copy the b into temp work space according to permutation */
  for (i = 0; i < n; i++) tmp[i] = b[i];

  /* forward solve the U^T */
  for (i = 0; i < n; i++) {
    v  = aa + adiag[i + 1] + 1;
    vi = aj + adiag[i + 1] + 1;
    nz = adiag[i] - adiag[i + 1] - 1;
    s1 = tmp[i];
    s1 *= v[nz]; /* multiply by inverse of diagonal entry */
    for (j = 0; j < nz; j++) tmp[vi[j]] -= s1 * v[j];
    tmp[i] = s1;
  }

  /* backward solve the L^T */
  for (i = n - 1; i >= 0; i--) {
    v  = aa + ai[i];
    vi = aj + ai[i];
    nz = ai[i + 1] - ai[i];
    s1 = tmp[i];
    for (j = 0; j < nz; j++) tmp[vi[j]] -= s1 * v[j];
  }

  /* copy tmp into x according to permutation */
  for (i = 0; i < n; i++) x[i] = tmp[i];

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));

  PetscCall(PetscLogFlops(2.0 * a->nz - A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_1_NaturalOrdering_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *)A->data;
  PetscInt         i, nz;
  const PetscInt  *diag = a->diag, n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  const MatScalar *aa = a->a, *v;
  PetscScalar      s1, *x;

  PetscFunctionBegin;
  PetscCall(VecCopy(bb, xx));
  PetscCall(VecGetArray(xx, &x));

  /* forward solve the U^T */
  for (i = 0; i < n; i++) {
    v = aa + diag[i];
    /* multiply by the inverse of the block diagonal */
    s1 = (*v++) * x[i];
    vi = aj + diag[i] + 1;
    nz = ai[i + 1] - diag[i] - 1;
    while (nz--) x[*vi++] -= (*v++) * s1;
    x[i] = s1;
  }
  /* backward solve the L^T */
  for (i = n - 1; i >= 0; i--) {
    v  = aa + diag[i] - 1;
    vi = aj + diag[i] - 1;
    nz = diag[i] - ai[i];
    s1 = x[i];
    while (nz--) x[*vi--] -= (*v--) * s1;
  }
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * (a->nz) - A->cmap->n));
  PetscFunctionReturn(0);
}
