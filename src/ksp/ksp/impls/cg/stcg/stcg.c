#define PETSCKSP_DLL

#include "include/private/kspimpl.h"             /*I "petscksp.h" I*/
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
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGSetRadius(KSP ksp, PetscReal radius)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  if (radius < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Tolerance must be positive");
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPSTCGSetRadius_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, radius); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSTCGGetNormD"
/*@
    KSPSTCGGetNormD - Got norm of the direction.

    Collective on KSP

    Input Parameters:
+   ksp    - the iterative context
-   norm_d - the norm of the direction

    Level: advanced

.keywords: KSP, STCG, get, norm direction
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetNormD(KSP ksp, PetscReal *norm_d)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPSTCGGetNormD_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, norm_d); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSTCGGetObjFcn"
/*@
    KSPSTCGGetObjFcn - Get objective function value.

    Collective on KSP

    Input Parameters:
+   ksp   - the iterative context
-   o_fcn - the objective function value

    Level: advanced

.keywords: KSP, STCG, get, objective function
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetObjFcn(KSP ksp, PetscReal *o_fcn)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPSTCGGetObjFcn_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, o_fcn); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_STCG"
/*
  KSPSolve_STCG - Use preconditioned conjugate gradient to compute
  an approximate minimizer of the quadratic function

            q(s) = g^T * s + 0.5 * s^T * H * s

   subject to the trust region constraint

            || s ||_M <= delta,

   where

     delta is the trust region radius,
     g is the gradient vector, and
     H is the Hessian matrix,
     M is the positive definite preconditioner matrix.

   KSPConvergedReason may be
$  KSP_CONVERGED_CG_NEG_CURVE if convergence is reached along a negative curvature direction,
$  KSP_CONVERGED_CG_CONSTRAINED if convergence is reached along a constrained step,
$  other KSP converged/diverged reasons

  Notes:
  The preconditioner supplied should be symmetric and positive definite.
*/
PetscErrorCode KSPSolve_STCG(KSP ksp)
{
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_ERR_SUP, "STCG is not available for complex systems");
#else
  KSP_STCG *cg = (KSP_STCG *)ksp->data;

  PetscErrorCode ierr;
  MatStructure  pflag;
  Mat Qmat, Mmat;
  Vec r, z, p, d;
  PC  pc;
  PetscReal norm_r, norm_d, norm_dp1, norm_p, dMp;
  PetscReal alpha, beta, kappa, rz, rzm1;
  PetscReal rr, r2, step;
  PetscInt  i, max_cg_its;

  PetscTruth diagonalscale;

  PetscFunctionBegin;

  /***************************************************************************/
  /* Check the arguments and parameters.                                     */
  /***************************************************************************/

  ierr = PCDiagonalScale(ksp->pc, &diagonalscale); CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ksp->type_name);

  if (cg->radius < 0.0) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Input error: radius < 0");
  }

  /***************************************************************************/
  /* Get the workspace vectors and initialize variables                      */
  /***************************************************************************/

  r2 = cg->radius * cg->radius;
  r  = ksp->work[0];
  z  = ksp->work[1];
  p  = ksp->work[2];
  d  = ksp->vec_sol;
  pc = ksp->pc;

  ierr = PCGetOperators(pc, &Qmat, &Mmat, &pflag); CHKERRQ(ierr);

  max_cg_its = ksp->max_it;
  ksp->its = 0;

  /***************************************************************************/
  /* Initialize objective function value                                     */
  /***************************************************************************/

  cg->o_fcn = 0.0;

  /***************************************************************************/
  /* Begin the conjugate gradient method.                                    */
  /***************************************************************************/

  ierr = VecSet(d, 0.0); CHKERRQ(ierr);			/* d = 0             */
  ierr = VecCopy(ksp->vec_rhs, r); CHKERRQ(ierr);	/* r = -grad         */
  ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);		/* z = inv(M) r      */
  cg->norm_d = 0.0;

  /***************************************************************************/
  /* Check the preconditioners for numerical problems and for positive       */
  /* definiteness.  The check for not-a-number and infinite values need be   */
  /* performed only once.                                                    */
  /***************************************************************************/

  ierr = VecDot(r, z, &rz); CHKERRQ(ierr);		/* rz = r^T inv(M) r */
  if (rz - rz != 0.0) {
    /*************************************************************************/
    /* The preconditioner produced not-a-number or an infinite value.  This  */
    /* can appear either due to r having numerical problems or M having      */
    /* numerical problems.  Differentiate between the two and then use the   */
    /* gradient direction.                                                   */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_NAN;
    ierr = VecDot(r, r, &rr); CHKERRQ(ierr);		/* rr = r^T r        */
    if (rr - rr != 0.0) {
      /***********************************************************************/
      /* The right-hand side contains not-a-number or an infinite value.     */
      /* The gradient step does not work; return a zero value for the step.  */
      /***********************************************************************/

      ierr = PetscInfo1(ksp, "KSPSolve_STCG: bad right-hand side: rr=%g\n", rr); CHKERRQ(ierr);
    }
    else {
      /***********************************************************************/
      /* The preconditioner contains not-a-number or an infinite value.      */
      /***********************************************************************/

      ierr = PetscInfo1(ksp, "KSPSolve_STCG: bad preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

      if (cg->radius) {
        alpha = sqrt(r2 / rr);
        ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r   */
        cg->norm_d = cg->radius;
  
	/*********************************************************************/
	/* Compute objective function.                                       */
	/*********************************************************************/

        ierr = KSP_MatMult(ksp, Qmat, d, z); CHKERRQ(ierr)
        ierr = VecAYPX(z, -0.5, ksp->vec_rhs); CHKERRQ(ierr);
        ierr = VecDot(d, z, &cg->o_fcn); CHKERRQ(ierr);
        cg->o_fcn = -cg->o_fcn;
      }
    }
    PetscFunctionReturn(0);
  }

  if (rz <= 0.0) {
    /*************************************************************************/
    /* The preconditioner is indefinite.  Because this is the first          */
    /* and we do not have a direction yet, we use the gradient step.  Note   */
    /* that we cannot use the preconditioned norm when computing the step    */
    /* because the matrix is indefinite.                                     */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
    ierr = PetscInfo1(ksp, "KSPSolve_STCG: indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

    if (cg->radius) {
      ierr = VecDot(r, r, &rr); CHKERRQ(ierr);		/* rr = r^T r        */
      alpha = sqrt(r2 / rr);
      ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r   */
      cg->norm_d = cg->radius;

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z); CHKERRQ(ierr)
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs); CHKERRQ(ierr);
      ierr = VecDot(d, z, &cg->o_fcn); CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
    }
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* As far as we know, the preconditioner is positive semidefinite.  Compute*/
  /* the residual and check for convergence.                                 */
  /***************************************************************************/

  switch(ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = VecNorm(z, NORM_2, &norm_r); CHKERRQ(ierr);	/* norm_r = |z|      */
    break;

  case KSP_NORM_UNPRECONDITIONED:
    ierr = VecNorm(r, NORM_2, &norm_r); CHKERRQ(ierr);	/* norm_r = |r|      */
    break;

  case KSP_NORM_NATURAL:
    norm_r = sqrt(rz);       				/* norm_r = |r|_M    */
    break;

  default:
    norm_r = 0.0;
    break;
  }

  KSPLogResidualHistory(ksp, norm_r);
  KSPMonitor(ksp, 0, norm_r);
  ksp->rnorm = norm_r;

  ierr = (*ksp->converged)(ksp, 0, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
  if (ksp->reason) {
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* We have not converged.  Compute the first direction and check the       */
  /* matrix for numerical problems.                                          */
  /***************************************************************************/

  ierr = VecCopy(z, p); CHKERRQ(ierr);			/* p = z             */
  ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p         */
  ierr = VecDot(p, z, &kappa); CHKERRQ(ierr);		/* kappa = p^T Q p   */
  if (kappa - kappa != 0.0) {
    /*************************************************************************/
    /* The matrix produced not-a-number or an infinite value.  In this case, */
    /* we must stop and use the gradient direction.  This condition need     */
    /* only be checked once.                                                 */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_NAN;
    ierr = PetscInfo1(ksp, "KSPSolve_STCG: bad matrix: kappa=%g\n", kappa); CHKERRQ(ierr);

    if (cg->radius) {
      ierr = VecDot(r, r, &rr); CHKERRQ(ierr);		/* rr = r^T r        */
      alpha = sqrt(r2 / rr);
      ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r   */
      cg->norm_d = cg->radius;

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z); CHKERRQ(ierr)
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs); CHKERRQ(ierr);
      ierr = VecDot(d, z, &cg->o_fcn); CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
    }
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* Initialize variables for calculating the norm of the direction.         */
  /***************************************************************************/

  dMp = 0.0;
  norm_d = 0.0;
  switch(cg->dtype) {
  case STCG_PRECONDITIONED_DIRECTION:
    norm_p = rz;
    break;

  default:
    ierr = VecDot(p, p, &norm_p); CHKERRQ(ierr);
    break;
  }

  /***************************************************************************/
  /* Run the conjugate gradient method until either the problem is solved,   */
  /* we encounter the boundary of the trust region, or the conjugate gradient*/
  /* method breaks down.                                                     */
  /***************************************************************************/

  for (i = 0; i <= max_cg_its; ++i) {
    /*************************************************************************/
    /* Know that kappa is nonzero, because we have not broken down, so we    */
    /* can compute the steplength.                                           */
    /*************************************************************************/

    if (kappa <= 0.0) {
      /***********************************************************************/
      /* In this case, the matrix is indefinite and we have encountered      */
      /* a direction of negative curvature.  Follow the direction to the     */
      /* boundary of the trust region.                                       */
      /***********************************************************************/

      ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: negative curvature: kappa=%g\n", kappa); CHKERRQ(ierr);

      if (cg->radius) {
        step = (sqrt(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
        ierr = VecAXPY(d, step, p); CHKERRQ(ierr);	/* d = d + step p    */
        cg->norm_d = cg->radius;

	/*********************************************************************/
        /* Update objective function.                                        */
	/*********************************************************************/

        cg->o_fcn += step * (0.5 * step * kappa - rz);
      }
      break;
    }
    alpha = rz / kappa;

    /*************************************************************************/
    /* Compute the steplength and check for intersection with the trust      */
    /* region.                                                               */
    /*************************************************************************/

    norm_dp1 = norm_d + alpha*(2.0*dMp + alpha*norm_p);
    if (cg->radius && norm_dp1 >= r2) {
      /***********************************************************************/
      /* In this case, the matrix is positive definite as far as we know.    */
      /* However, the direction does beyond the trust region.  Follow the    */
      /* direction to the boundary of the trust region.                      */
      /***********************************************************************/

      ksp->reason = KSP_CONVERGED_CG_CONSTRAINED;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: constrained step: radius=%g\n", cg->radius); CHKERRQ(ierr);

      step = (sqrt(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
      ierr = VecAXPY(d, step, p); CHKERRQ(ierr);	/* d = d + step p    */
      cg->norm_d = cg->radius;

      /***********************************************************************/
      /* Update objective function.                                          */
      /***********************************************************************/

      cg->o_fcn += step * (0.5 * step * kappa - rz);
      break;
    }

    /*************************************************************************/
    /* Now we can update the direction, residual, and objective value.       */
    /*************************************************************************/

    ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);		/* d = d + alpha p   */
    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha Q p */
    ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);

    switch(cg->dtype) {
    case STCG_PRECONDITIONED_DIRECTION:
      norm_d = norm_dp1;
      break;

    default:
      ierr = VecDot(d, d, &norm_d); CHKERRQ(ierr);
      break;
    }
    cg->norm_d = sqrt(norm_d);

    cg->o_fcn -= 0.5 * alpha * rz;

    /*************************************************************************/
    /* Check that the preconditioner appears positive semidefinite.          */
    /*************************************************************************/

    rzm1 = rz;
    ierr = VecDot(r, z, &rz); CHKERRQ(ierr);		/* rz = r^T z        */
    if (rz <= 0.0) {
      /***********************************************************************/
      /* The preconditioner is indefinite.                                   */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: cg indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* As far as we know, the preconditioner is positive semidefinite.       */
    /* Compute the residual and check for convergence.                       */
    /*************************************************************************/

    switch(ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = VecNorm(z, NORM_2, &norm_r); CHKERRQ(ierr);/* norm_r = |z| */
      break;

    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(r, NORM_2, &norm_r); CHKERRQ(ierr);/* norm_r = |r| */
      break;

    case KSP_NORM_NATURAL:
      norm_r = sqrt(rz);				/* norm_r = |r|_B */
      break;

    default:
      norm_r = 0;
      break;
    }

    KSPLogResidualHistory(ksp, norm_r);
    KSPMonitor(ksp, 0, norm_r);
    ksp->rnorm = norm_r;
  
    ierr = (*ksp->converged)(ksp, i+1, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
    if (ksp->reason) {                 /* convergence */
      ierr = PetscInfo2(ksp,"KSPSolve_STCG: truncated step: rnorm=%g, radius=%g\n",norm_r,cg->radius); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* We have not converged yet.  Check for breakdown, then update p and    */
    /* the norms.                                                            */
    /*************************************************************************/

    beta = rz / rzm1;
    if (fabs(beta) <= 0.0) {
      /***********************************************************************/
      /* Conjugate gradients has broken down.                                */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: breakdown: beta=%g\n", beta); CHKERRQ(ierr);
      break;
    }

    ierr = VecAYPX(p, beta, z); CHKERRQ(ierr);          /* p = z + beta p    */

    switch(cg->dtype) {
    case STCG_PRECONDITIONED_DIRECTION:
      dMp = beta*(dMp + alpha*norm_p);
      norm_p = beta*(rzm1 + beta*norm_p);
      break;

    default:
      ierr = VecDot(d, p, &dMp); CHKERRQ(ierr);
      ierr = VecDot(p, p, &norm_p); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Compute the new direction                                             */
    /*************************************************************************/

    ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p         */
    ierr = VecDot(p, z, &kappa); CHKERRQ(ierr);		/* kappa = p^T Q p   */

    /*************************************************************************/
    /* Update the iteration.                                                 */
    /*************************************************************************/

    ++ksp->its;
  }

  if (!ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_STCG"
PetscErrorCode KSPSetUp_STCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /***************************************************************************/
  /* This implementation of CG only handles left preconditioning so generate */
  /* an error otherwise.                                                     */
  /***************************************************************************/

  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP, "No right preconditioning for KSPSTCG");
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP, "No symmetric preconditioning for KSPSTCG");
  }

  /***************************************************************************/
  /* Set work vectors needed by conjugate gradient method and allocate       */
  /***************************************************************************/

  ierr = KSPDefaultGetWork(ksp, 3); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_STCG"
PetscErrorCode KSPDestroy_STCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

 /***************************************************************************/
  /* Clear composed functions                                                */
  /***************************************************************************/

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPSTCGSetRadius_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPSTCGGetNormD_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPSTCGGetObjFcn_C","",PETSC_NULL);CHKERRQ(ierr);

  /***************************************************************************/
  /* Destroy KSP object.                                                     */
  /***************************************************************************/

  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPSTCGSetRadius_STCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGSetRadius_STCG(KSP ksp, PetscReal radius)
{
  KSP_STCG *cg = (KSP_STCG *)ksp->data;

  PetscFunctionBegin;
  cg->radius = radius;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSTCGGetNormD_STCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetNormD_STCG(KSP ksp, PetscReal *norm_d)
{
  KSP_STCG *cg = (KSP_STCG *)ksp->data;

  PetscFunctionBegin;
  *norm_d = cg->norm_d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSTCGGetObjFcn_STCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetObjFcn_STCG(KSP ksp, PetscReal *o_fcn){
  KSP_STCG *cg = (KSP_STCG *)ksp->data;

  PetscFunctionBegin;
  *o_fcn = cg->o_fcn;
  PetscFunctionReturn(0);
}
EXTERN_C_END

static const char *DType_Table[64] = {
  "preconditioned", "unpreconditioned"
};

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_STCG"
PetscErrorCode KSPSetFromOptions_STCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_STCG *cg = (KSP_STCG *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP STCG options"); CHKERRQ(ierr);

  ierr = PetscOptionsReal("-ksp_stcg_radius", "Trust Region Radius", "KSPSTCGSetRadius", cg->radius, &cg->radius, PETSC_NULL); CHKERRQ(ierr);

  ierr = PetscOptionsEList("-ksp_stcg_dtype", "Norm used for direction", "", DType_Table, STCG_DIRECTION_TYPES, DType_Table[cg->dtype], &cg->dtype, PETSC_NULL); CHKERRQ(ierr);

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

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPSTCGSetRadius(), KSPSTCGGetNormD(), KSPSTCGGetObjFcn()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_STCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_STCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_STCG *cg;

  PetscFunctionBegin;

  ierr = PetscNewLog(ksp,KSP_STCG, &cg); CHKERRQ(ierr);

  cg->radius = PETSC_MAX;
  cg->dtype = STCG_UNPRECONDITIONED_DIRECTION;

  ksp->data = (void *) cg;
  ksp->pc_side = PC_LEFT;

  /***************************************************************************/
  /* Sets the functions that are associated with this data structure         */
  /* (in C++ this is the same as defining virtual functions).                */
  /***************************************************************************/

  ksp->ops->setup                = KSPSetUp_STCG;
  ksp->ops->solve                = KSPSolve_STCG;
  ksp->ops->destroy              = KSPDestroy_STCG;
  ksp->ops->setfromoptions       = KSPSetFromOptions_STCG;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->view                 = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
				           "KSPSTCGSetRadius_C",
                                           "KSPSTCGSetRadius_STCG",
                                            KSPSTCGSetRadius_STCG); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
				           "KSPSTCGGetNormD_C",
                                           "KSPSTCGGetNormD_STCG",
                                            KSPSTCGGetNormD_STCG); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                           "KSPSTCGGetObjFcn_C",
                                           "KSPSTCGGetObjFcn_STCG",
                                            KSPSTCGGetObjFcn_STCG); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
