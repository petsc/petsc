
/*
              This file creating by running f2c
            linpack. this version dated 08/14/78
      cleve moler, university of new mexico, argonne national lab.

      Computes the inverse of a matrix given its factors and pivots
    calculated by PetscLINPACKgefa(). Performed in-place for an n by n
    dense matrix.

       Used by the sparse factorization routines in
     src/mat/impls/baij/seq

*/

#include <petscsys.h>
#include <petsc/private/kernels/blockinvert.h>

PETSC_INTERN PetscErrorCode PetscLINPACKgedi(MatScalar *a, PetscInt n, PetscInt *ipvt, MatScalar *work)
{
  PetscInt   i__2, kb, kp1, nm1, i, j, k, l, ll, kn, knp1, jn1;
  MatScalar *aa, *ax, *ay, tmp;
  MatScalar  t;

  PetscFunctionBegin;
  --work;
  --ipvt;
  a -= n + 1;

  /*     compute inverse(u) */

  for (k = 1; k <= n; ++k) {
    kn      = k * n;
    knp1    = kn + k;
    a[knp1] = 1.0 / a[knp1];
    t       = -a[knp1];
    i__2    = k - 1;
    aa      = &a[1 + kn];
    for (ll = 0; ll < i__2; ll++) aa[ll] *= t;
    kp1 = k + 1;
    if (n < kp1) continue;
    ax = aa;
    for (j = kp1; j <= n; ++j) {
      jn1        = j * n;
      t          = a[k + jn1];
      a[k + jn1] = 0.;
      ay         = &a[1 + jn1];
      for (ll = 0; ll < k; ll++) ay[ll] += t * ax[ll];
    }
  }

  /*    form inverse(u)*inverse(l) */

  nm1 = n - 1;
  if (nm1 < 1) PetscFunctionReturn(0);
  for (kb = 1; kb <= nm1; ++kb) {
    k   = n - kb;
    kn  = k * n;
    kp1 = k + 1;
    aa  = a + kn;
    for (i = kp1; i <= n; ++i) {
      work[i] = aa[i];
      aa[i]   = 0.;
    }
    for (j = kp1; j <= n; ++j) {
      t  = work[j];
      ax = &a[j * n + 1];
      ay = &a[kn + 1];
      for (ll = 0; ll < n; ll++) ay[ll] += t * ax[ll];
    }
    l = ipvt[k];
    if (l != k) {
      ax = &a[kn + 1];
      ay = &a[l * n + 1];
      for (ll = 0; ll < n; ll++) {
        tmp    = ax[ll];
        ax[ll] = ay[ll];
        ay[ll] = tmp;
      }
    }
  }
  PetscFunctionReturn(0);
}
