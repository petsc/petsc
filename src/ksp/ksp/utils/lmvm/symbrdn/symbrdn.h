#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Limited-memory Symmetric Broyden method for approximating both 
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Mat D;                                    /* diagonal scaling term */
  Vec *P, *Q;                               /* storage vectors for (B_i)*S[i] and (B_i)^{-1}*Y[i] */
  Vec invDnew, invD, BFGS, DFP, U, V, W;    /* work vectors for diagonal scaling */
  Vec work;
  PetscBool allocated, needP, needQ;
  PetscReal *stp, *ytq, *yts, *yty, *sts;   /* scalar arrays for recycling dot products */
  PetscReal theta, phi, *psi;               /* convex combination factors between DFP and BFGS */
  PetscReal rho, alpha, beta;               /* convex combination factors for the scalar or diagonal scaling */
  PetscReal delta, delta_min, delta_max, sigma;
  PetscInt sigma_hist;                      /* length of update history to be used for scaling */
  PetscInt scale_type;
  PetscInt watchdog, max_seq_rejects;        /* tracker to reset after a certain # of consecutive rejects */
} Mat_SymBrdn;

#define SYMBRDN_SCALE_NONE      0
#define SYMBRDN_SCALE_SCALAR    1
#define SYMBRDN_SCALE_DIAG      2
#define SYMBRDN_SCALE_SIZE      3

static const char *Scale_Table[64] = {"none","scalar","diagonal"};

PETSC_INTERN PetscErrorCode MatSymBrdnApplyJ0Fwd(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSymBrdnApplyJ0Inv(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSymBrdnComputeJ0Diag(Mat);
PETSC_INTERN PetscErrorCode MatSymBrdnComputeJ0Scalar(Mat);

PETSC_INTERN PetscErrorCode MatView_LMVMSymBrdn(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVMSymBrdn(PetscOptionItems*, Mat);
