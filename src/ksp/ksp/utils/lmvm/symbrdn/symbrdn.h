#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Limited-memory Symmetric Broyden method for approximating both 
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Vec *P, *Q;
  Vec work;
  PetscBool allocated;
  PetscReal *stp, *ytq, *yts, *yty, *sts;
  PetscReal phi, *psi;
  PetscReal rho, alpha;
  PetscInt sigma_hist;
} Mat_SymBrdn;

PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVMSymBrdn(PetscOptionItems*, Mat);
PETSC_INTERN PetscErrorCode MatSymBrdnComputeJ0Scalar(Mat, PetscScalar*);