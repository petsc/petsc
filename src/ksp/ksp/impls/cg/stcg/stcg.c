#define PETSCKSP_DLL

#include "src/ksp/ksp/kspimpl.h"             /*I "petscksp.h" I*/
#include "src/ksp/ksp/impls/cg/stcg/stcg.h"

#undef __FUNCT__
#define __FUNCT__ "KSPSTCGSetRadius"
/*@
    KSPSTCGSetRadius - Sets the radius of the trust region.

    Collective on KSP

    Input Parameters:
+   ksp    - the iterative context
-   radius - the trust region radius (Infinity is the default)

    Options Database Key:
.   -ksp_stcg_radius <r>

    Level: advanced

.keywords: KSP, STCG, set, trust region radius
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGSetRadius(KSP ksp,PetscReal radius)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  if (radius <= 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Tolerance must be positive");
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPSTCGSetRadius_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, radius); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_STCG"
/*
  KSPSolve_STCG - Use preconditioned conjugate gradient to compute
  an approximate minimizer of the quadratic function

            q(s) = g^T * s + .5 * s^T * H * s

   subject to the trust region constraint

            || s ||_M <= delta,

   where

     delta is the trust region radius,
     g is the gradient vector, and
     H is the Hessian matrix,
     M is the positive definite preconditioner matrix.

   KSPConvergedReason may be
$  KSP_CONVERGED_STCG_NEG_CURVE if convergence is reached along a negative curvature direction,
$  KSP_CONVERGED_STCG_CONSTRAINED if convergence is reached along a constrained step,
$  other KSP converged/diverged reasons


  Notes:
  The preconditioner supplied should be symmetric and positive definite.
*/
PetscErrorCode KSPSolve_STCG(KSP ksp)
{
  PetscErrorCode ierr;
  MatStructure  pflag;
  Mat Qmat, Mmat;
  Vec r, z, p, d;
  PC  pc;
  KSP_STCG *cg;
  PetscReal norm_r, norm_d, norm_dp1, norm_p, dMp;
  PetscReal alpha, beta, kappa, rz, rzm1;
  PetscReal rdm1, rpm1;
  PetscReal r2;
  PetscInt  i, maxit;
  PetscTruth diagonalscale;

  PetscFunctionBegin;
  ierr = PCDiagonalScale(ksp->pc, &diagonalscale); CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ksp->type_name);

  cg       = (KSP_STCG *)ksp->data;
  r2       = cg->radius * cg->radius;
  maxit    = ksp->max_it;
  r        = ksp->work[0];
  z        = ksp->work[1];
  p        = ksp->work[2];
  d        = ksp->vec_sol;
  pc       = ksp->pc;

#if !defined(PETSC_USE_COMPLEX)
#define VecXDot(x,y,a) VecDot(x,y,a)
#else
#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))
#endif

  ksp->its = 0;
  if (cg->radius <= 0.0) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Input error: radius <= 0");
  }

  /* Initialize variables */
  ierr = PCGetOperators(pc, &Qmat, &Mmat, &pflag); CHKERRQ(ierr);

  ierr = VecSet(d, 0.0); CHKERRQ(ierr);			/* d = 0        */
  ierr = VecCopy(ksp->vec_rhs, r); CHKERRQ(ierr);	/* r = -grad    */
  ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);		/* z = M_{-1} r */

  /* Check that preconditioner is positive definite */
  ierr = VecXDot(r, z, &rz); CHKERRQ(ierr);		/* rz = r^T z   */
  if ((rz != rz) || (rz && (rz / rz != rz / rz))) {
    ksp->reason = KSP_DIVERGED_NAN;
    ierr = PetscInfo1(ksp, "KSPSolve_STCG: bad preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

    /* In this case, the preconditioner produced not a number or an         */
    /* infinite value.  We just take the gradient step.                     */
    /* Only needs to be checked once.                                       */

    ierr = VecXDot(r, r, &rz); CHKERRQ(ierr);		/* rz = r^T r   */

    alpha = sqrt(r2 / rz);
    ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);		/* d = d + alpha r */
    PetscFunctionReturn(0);
  }

  if (rz <= 0.0) {
    ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
    ierr = PetscInfo1(ksp, "KSPSolve_STCG: indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

    /* In this case, the preconditioner is indefinite, so we cannot measure */
    /* the direction in the preconditioned norm.  Therefore, we must use    */
    /* an unpreconditioned calculation.  The direction in this case is	    */
    /* uses the right hand side, which should be the negative gradient      */
    /* intersected with the trust region.                                   */

    ierr = VecXDot(r, r, &rz); CHKERRQ(ierr);		/* rz = r^T r   */

    alpha = sqrt(r2 / rz);
    ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);		/* d = d + alpha r */
    PetscFunctionReturn(0);
  }

  /* As far as we know, the preconditioner is positive definite.  Compute   */
  /* the appropriate residual depending on what the user has set.           */
  if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
    ierr = VecNorm(z, NORM_2, &norm_r); CHKERRQ(ierr);	/* norm_r = |z| */
  }
  else if (ksp->normtype == KSP_UNPRECONDITIONED_NORM) {
    ierr = VecNorm(r, NORM_2, &norm_r); CHKERRQ(ierr);	/* norm_r = |r| */
  }
  else if (ksp->normtype == KSP_NATURAL_NORM) {
    norm_r = sqrt(rz);					/* norm_r = |r|_B */
  }
  else {
    norm_r = 0;
  }

  /* Log the residual and call any registered moitor routines */
  KSPLogResidualHistory(ksp, norm_r);
  KSPMonitor(ksp, 0, norm_r);
  ksp->rnorm = norm_r;

  /* Test for convergence */
  ierr = (*ksp->converged)(ksp, 0, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
  if (ksp->reason) {
    PetscFunctionReturn(0);
  }

  /* Compute the initial vectors and variables for trust-region computations */
  ierr = VecCopy(z, p); CHKERRQ(ierr);                   /* p = z       */

  dMp = 0.0;
  norm_p = rz;
  norm_d = 0.0;

  /* Compute the direction */
  ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p   */
  ierr = VecXDot(p, z, &kappa); CHKERRQ(ierr);		/* kappa = p^T z */

  if ((kappa != kappa) || (kappa && (kappa / kappa != kappa / kappa))) {
    ksp->reason = KSP_DIVERGED_NAN;
    ierr = PetscInfo1(ksp, "KSPSolve_STCG: bad matrix: kappa=%g\n", kappa); CHKERRQ(ierr);

    /* In this case, the matrix produced not a number or an infinite value. */
    /* We just take the gradient step.  Only needs to be checked once.      */

    ierr = VecXDot(r, r, &rz); CHKERRQ(ierr);           /* rz = r^T r   */

    alpha = sqrt(r2 / rz);
    ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);         /* d = d + alpha r */
    PetscFunctionReturn(0);
  }

  /* Begin iterating */
  for (i = 0; i <= maxit; i++) {
    ++ksp->its;

    if (kappa <= 0.0) {
      ksp->reason = KSP_CONVERGED_STCG_NEG_CURVE;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: negative curvature: kappa=%g\n", kappa); CHKERRQ(ierr);

      /* In this case, the matrix is indefinite and we have encountered     */
      /* a direction of negative curvature.  Follow the direction to the    */
      /* boundary of the trust region.                                      */

      alpha = (sqrt(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
      ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);        /* d = d + alpha p */
      break;
    }
    alpha = rz / kappa;

    /* Update residual */
    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha z */
    ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);

    /* Check for loss of orthogonality */
    ierr = VecXDot(r, d, &rdm1); CHKERRQ(ierr);		/* r_k^T d_{k-1} */
    ierr = VecXDot(r, p, &rpm1); CHKERRQ(ierr);		/* r_k^T p_{k-1} */
    if (fabs(rdm1) > 1e-5 || fabs(rpm1) > 1e-5) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr = PetscInfo2(ksp, "KSPSolve_STCG: breakdown: rdm1=%g rpm1=%g\n", rdm1, rpm1); CHKERRQ(ierr);

      if (0 == i) {
        /* In this case, we have lost orthogonality in the directions.      */
        /* We stop in this case and return the current direction.  Since    */
	/* we have not updated the direction yet, we need to take a         */
        /* gradient step.                                                   */

        ierr = VecCopy(ksp->vec_rhs, r); CHKERRQ(ierr);	/* r = -grad    */
        ierr = VecXDot(r, r, &rz); CHKERRQ(ierr);	/* rz = r^T r   */

        alpha = sqrt(r2 / rz);
        ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r */
      }
      else {
        /* Otherwise accept the current step */
      }
      break;
    }

#if 0
    ierr = VecXDot(z, d, &rdm1); CHKERRQ(ierr);		/* z_k^T d_{k-1} */
    ierr = VecXDot(z, p, &rpm1); CHKERRQ(ierr);		/* z_k^T p_{k-1} */
#endif

    /* Update the direction */
    norm_dp1 = norm_d + 2.0*alpha*dMp + alpha*alpha*norm_p;
    if (norm_dp1 >= r2) {
      ksp->reason = KSP_CONVERGED_STCG_CONSTRAINED;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: constrained step: radius=%g\n", cg->radius); CHKERRQ(ierr);

      /* In this case, the matrix is  positive definite as far as we know.  */
      /* However, the direction does beyond the trust region.  Follow the   */
      /* direction to the boundary of the trust region.                     */

      alpha = (sqrt(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
      ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);        /* d = d + alpha p */
      break;
    }
    ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);         /* d = d + alpha p */

    /* Check that preconditioner is positive definite */
    rzm1 = rz;
    ierr = VecXDot(r, z, &rz); CHKERRQ(ierr);		/* rz = r^T z   */
    if (rz <= 0.0) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

      /* In this case, the preconditioner is indefinite.  We could follow   */
      /* the direction to the boundary of the trust region, but it seems    */
      /* best to stop at the current point.                                 */
      break;
    }

    /* As far as we know, the matrix and preconditioner are positive        */
    /* definite.  Compute the appropriate residual depending on what the    */
    /* user has set.                                                        */
    if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
      ierr = VecNorm(z, NORM_2, &norm_r); CHKERRQ(ierr);/* norm_r = |z| */
    }
    else if (ksp->normtype == KSP_UNPRECONDITIONED_NORM) {
      ierr = VecNorm(r, NORM_2, &norm_r); CHKERRQ(ierr);/* norm_r = |r| */
    }
    else if (ksp->normtype == KSP_NATURAL_NORM) {
      norm_r = sqrt(rz);				/* norm_r = |r|_B */
    }
    else {
      norm_r = 0;
    }

    /* Log the residual and call any registered moitor routines */
    KSPLogResidualHistory(ksp, norm_r);
    KSPMonitor(ksp, 0, norm_r);
    ksp->rnorm = norm_r;
  
    /* Test for convergence */
    ierr = (*ksp->converged)(ksp, i+1, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
    if (ksp->reason) {                 /* convergence */
      ierr = PetscInfo2(ksp,"KSPSolve_STCG: truncated step: rnorm=%g, radius=%g\n",norm_r,cg->radius); CHKERRQ(ierr);
      break;
    }

    /* Update p and the norms */
    beta = rz / rzm1;

    VecAYPX(p, beta, z);                    /* p = z + beta p */

    dMp = beta*(dMp + alpha*norm_p);
    norm_p = rz + beta*beta*norm_p;
    norm_d = norm_dp1;

    /* Compute new direction */
    ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p   */
    ierr = VecXDot(p, z, &kappa); CHKERRQ(ierr);	/* kappa = p^T z */
  }

  if (!ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_STCG"
PetscErrorCode KSPSetUp_STCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* This implementation of CG only handles left preconditioning
   * so generate an error otherwise.
   */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP, "No right preconditioning for KSPSTCG");
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP, "No symmetric preconditioning for KSPSTCG");
  }

  /* get work vectors needed by CG */
  ierr = KSPDefaultGetWork(ksp, 3); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_STCG"
PetscErrorCode KSPDestroy_STCG(KSP ksp)
{
  KSP_STCG *cgP = (KSP_STCG *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultFreeWork(ksp); CHKERRQ(ierr);

  /* Free the context variable */
  ierr = PetscFree(cgP); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPSTCGSetRadius_STCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGSetRadius_STCG(KSP ksp,PetscReal radius)
{
  KSP_STCG *cgP = (KSP_STCG *)ksp->data;

  PetscFunctionBegin;
  cgP->radius = radius;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_STCG"
PetscErrorCode KSPSetFromOptions_STCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_STCG        *cgP = (KSP_STCG *)ksp->data;
  PetscReal      radius;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP STCG options"); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_stcg_radius", "Trust Region Radius", "KSPSTCGSetRadius", cgP->radius, &radius, &flg); CHKERRQ(ierr);
  if (flg) { ierr = KSPSTCGSetRadius(ksp, radius); CHKERRQ(ierr); }
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPSTCG -   Code to run conjugate gradient method subject to a constraint
         on the solution norm. This is used in Trust Region methods for
         nonlinear equations, SNESTR

   Options Database Keys:
.      -ksp_stcg_radius <r> - Trust Region Radius

   Notes: This is rarely used directly

   Level: developer

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPSTCGSetRadius()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_STCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_STCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_STCG        *cg;

  PetscFunctionBegin;
  ierr = PetscNew(KSP_STCG, &cg); CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp, sizeof(KSP_STCG)); CHKERRQ(ierr);
  cg->radius                     = PETSC_MAX;
  ksp->data                      = (void *)cg;
  ksp->pc_side                   = PC_LEFT;

  /* Sets the functions that are associated with this data structure
   * (in C++ this is the same as defining virtual functions)
   */
  ksp->ops->setup                = KSPSetUp_STCG;
  ksp->ops->solve                = KSPSolve_STCG;
  ksp->ops->destroy              = KSPDestroy_STCG;
  ksp->ops->setfromoptions       = KSPSetFromOptions_STCG;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->view                 = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPSTCGSetRadius_C",
                                    "KSPSTCGSetRadius_STCG",
                                     KSPSTCGSetRadius_STCG); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
