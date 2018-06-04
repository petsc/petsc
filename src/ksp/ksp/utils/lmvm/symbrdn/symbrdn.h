#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Limited-memory Symmetric Broyden method for approximating both 
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Vec *P, *Q;                               /* storage vectors for (B_i)*S[i] and (B_i)^{-1}*Y[i] */
  Vec work;
  PetscBool allocated;
  PetscReal *stp, *ytq, *yts, *yty, *sts;   /* scalar arrays for recycling dot products */
  PetscReal phi, *psi;                      /* convex combination factors between DFP and BFGS */
  PetscReal rho, alpha;                     /* convex combination factors for the default J0 scalar */
  PetscInt sigma_hist;                      /* length of update history to be used for default J0 scalar */
} Mat_SymBrdn;

PETSC_INTERN PetscErrorCode MatSymBrdnComputeJ0Scalar(Mat, PetscScalar*);