#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  "Full" memory implementation of only the diagonal terms in a symmetric Broyden approximation.
*/

typedef struct {
  Vec invDnew, invD, BFGS, DFP, U, V, W;    /* work vectors for diagonal scaling */
  PetscReal *yts, *yty, *sts;               /* scalar arrays for recycling dot products */
  PetscReal theta, rho, alpha, beta;        /* convex combination factors for the scalar or diagonal scaling */
  PetscReal delta, delta_min, delta_max, sigma, tol;
  PetscInt sigma_hist;                      /* length of update history to be used for scaling */
  PetscBool allocated;
  PetscBool forward;
} Mat_DiagBrdn;
