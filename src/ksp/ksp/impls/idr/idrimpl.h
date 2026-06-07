/*
   Private data structure used by the IDR(s) method.
*/
#pragma once

#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

typedef struct {
  PetscInt     s;     /* shadow space dimension; default 4 */
  Vec         *GG;    /* s direction vectors G[0..s-1] */
  Vec         *UU;    /* s update vectors   U[0..s-1] */
  Vec         *PP;    /* s shadow vectors   P[0..s-1] (fixed, random orthonormal) */
  Vec          r;     /* current residual */
  Vec          v;     /* work vector */
  Vec          t;     /* work vector (preconditioned operator result) */
  Vec          guess; /* saved initial guess, used with right preconditioning */
  PetscScalar *M;     /* s*s matrix M[j,k] = <G[k],P[j]>, column-major */
  PetscScalar *f;     /* length s: P^T r */
  PetscScalar *c;     /* length s: solution of M c = f */
  PetscReal    cth;   /* omega stabilization threshold (0 = off, default 0.7) */
} KSP_IDR;
