#include <petsc-private/kspimpl.h>
#include <../src/ksp/ksp/impls/lsqr/lsqr.h>
extern PetscErrorCode  KSPLSQRGetArnorm(KSP,PetscReal*,PetscReal*,PetscReal*);

PetscErrorCode KSPConvergedLSQR(KSP solksp,PetscInt  iter,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  PetscReal       atol;    /* Absolute convergence criterium */
  PetscReal       dtol;    /* Divergence criterium */
  PetscReal       rtol;    /* Relative convergence criterium */
  PetscReal       stop1;   /* Stop if: |r| < rtol*|b| + atol*|A|*|x| */
  PetscReal       stop2;   /* Stop if: |A^t.r|/(|A|*|r|) < atol */
  Vec             x_sol;   /* Current solution vector */

  PetscReal       arnorm, anorm, bnorm, xnorm;  /* Norms of A*residual; matrix A; rhs; solution */

  PetscInt        mxiter;  /* Maximum # of iterations */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *reason = KSP_CONVERGED_ITERATING;
  if (iter == 0) {PetscFunctionReturn(0);};

  if (PetscIsInfOrNanScalar(rnorm)) {
    *reason = KSP_DIVERGED_NAN;
    PetscFunctionReturn(0);
  }

  ierr = KSPGetTolerances( solksp, &rtol, &atol, &dtol, &mxiter );CHKERRQ(ierr);
  if ( iter > mxiter ) {
    *reason = KSP_DIVERGED_ITS;
    PetscFunctionReturn(0);
  }

  ierr = KSPGetSolution( solksp, &x_sol );CHKERRQ(ierr);
  ierr = VecNorm(x_sol, NORM_2 , &xnorm);CHKERRQ(ierr);

  ierr = KSPLSQRGetArnorm( solksp, &arnorm, &bnorm, &anorm);CHKERRQ(ierr);
  if (bnorm > 0.0)  {
    stop1 = rnorm / bnorm;
    rtol = rtol + atol * anorm*xnorm/bnorm;
  } else {
    stop1 = 0.0;
    rtol = 0.0;
  }
  stop2 = 0.0;
  if (rnorm > 0.0) stop2 = arnorm / (anorm * rnorm );

  /* Test for tolerances set by the user */
  if ( stop1 <= rtol ) *reason = KSP_CONVERGED_RTOL;
  if ( stop2 <= atol ) *reason = KSP_CONVERGED_ATOL;

  /* Test for machine precision */
  if (bnorm > 0)  {
    stop1 = stop1 / (1.0 + anorm*xnorm/bnorm);
  } else {
    stop1 = 0.0;
  }
  stop1 = 1.0 + stop1;
  stop2 = 1.0 + stop2;
  if ( stop1 <= 1.0 ) *reason = KSP_CONVERGED_RTOL;
  if ( stop2 <= 1.0 ) *reason = KSP_CONVERGED_ATOL;
  PetscFunctionReturn(0);
}
