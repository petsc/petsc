#define PETSCKSP_DLL

#include <math.h>
#include "include/private/kspimpl.h"             /*I "petscksp.h" I*/
#include "include/petscblaslapack.h"
#include "src/ksp/ksp/impls/cg/gltr/gltr.h"

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRSetRadius"
/*@
    KSPGLTRSetRadius - Sets the radius of the trust region.

    Collective on KSP

    Input Parameters:
+   ksp    - the iterative context
-   radius - the trust region radius (Infinity is the default)

    Options Database Key:
.   -ksp_gltr_radius <r>

    Level: advanced

.keywords: KSP, GLTR, set, trust region radius
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRSetRadius(KSP ksp, PetscReal radius)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  if (radius < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Tolerance must be positive");
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPGLTRSetRadius_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, radius); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetNormD"
/*@
    KSPGLTRGetNormD - Get norm of the direction.

    Collective on KSP

    Input Parameters:
+   ksp    - the iterative context
-   norm_d - the norm of the direction

    Level: advanced

.keywords: KSP, GLTR, get, norm direction
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetNormD(KSP ksp,PetscReal *norm_d)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPGLTRGetNormD_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, norm_d); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetObjFcn"
/*@
    KSPGLTRGetObjFcn - Get objective function value.

    Collective on KSP

    Input Parameters:
+   ksp   - the iterative context
-   o_fcn - the objective function value

    Level: advanced

.keywords: KSP, GLTR, get, objective function
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetObjFcn(KSP ksp,PetscReal *o_fcn)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPGLTRGetObjFcn_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, o_fcn); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetMinEig"
/*@
    KSPGLTRGetMinEig - Get minimum eigenvalue.

    Collective on KSP

    Input Parameters:
+   ksp   - the iterative context
-   e_min - the minimum eigenvalue

    Level: advanced

.keywords: KSP, GLTR, get, minimum eigenvalue
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetMinEig(KSP ksp,PetscReal *e_min)
{
  PetscErrorCode ierr, (*f)(KSP, PetscReal *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_COOKIE, 1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp, "KSPGLTRGetMinEig_C", (void (**)(void))&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp, e_min); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_GLTR"
/*
  KSPSolve_GLTR - Use preconditioned conjugate gradient to compute
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
$  KSP_CONVERGED_CG_NEG_CURVE if convergence is reached along a negative curvature direction,
$  KSP_CONVERGED_CG_CONSTRAINED if convergence is reached along a constrained step,
$  other KSP converged/diverged reasons


  Notes:
  The preconditioner supplied should be symmetric and positive definite.
*/

#ifdef PETSC_USE_COMPLEX
PetscErrorCode GLTR_VecDot(Vec x, Vec y, PetscReal *a)
{
  PetscScalar ca;
  PetscErrorCode ierr;

  ierr = VecDot(x,y,&ca);
  *a = PetscRealPart(ca);
  return ierr;
}
#else
PetscErrorCode GLTR_VecDot(Vec x, Vec y, PetscReal *a)
{
  return VecDot(x,y,a);
}
#endif

PetscErrorCode KSPSolve_GLTR(KSP ksp)
{
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_ERR_SUP, "GLTR is not available for complex systems");
#else
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;
  PetscReal *t_soln, *t_diag, *t_offd, *e_valu, *e_vect, *e_rwrk;
  PetscInt  *e_iblk, *e_splt, *e_iwrk;

  PetscErrorCode ierr;
  MatStructure  pflag;
  Mat Qmat, Mmat;
  Vec r, z, p, d;
  PC  pc;
  PetscReal norm_r, norm_d, norm_dp1, norm_p, dMp;
  PetscReal alpha, beta, kappa, rz, rzm1;
  PetscReal rr, r2, piv, step;
  PetscReal vl, vu;
  PetscReal coef1, coef2, coef3, root1, root2, obj1, obj2;
  PetscReal norm_t, norm_w, lambda, pert;
  PetscInt  i, j, max_cg_its, max_lanczos_its, max_newton_its, sigma;
  PetscInt  t_size = 0, il, iu, e_valus, e_splts, info;
  PetscInt  nrhs, nldb;

  KSPConvergedReason reason;
  PetscTruth diagonalscale;

  PetscFunctionBegin;

  /***************************************************************************/
  /* Check the arguments and parameters.                                     */
  /***************************************************************************/

  ierr = PCDiagonalScale(ksp->pc, &diagonalscale); CHKERRQ(ierr);
  if (diagonalscale) {
    SETERRQ1(PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ksp->type_name);
  }

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

  max_cg_its      = cg->max_cg_its;
  max_lanczos_its = cg->max_lanczos_its;
  max_newton_its  = cg->max_newton_its;
  ksp->its = 0;

  /***************************************************************************/
  /* Initialize objective function value and minimum eigenvalue.             */
  /***************************************************************************/

  cg->e_min = 0.0;
  cg->o_fcn = 0.0;

  /***************************************************************************/
  /* The first phase of GLTR performs a standard conjugate gradient method,  */
  /* but stores the values required for the Lanczos matrix.  We switch to    */
  /* the Lanczos when the conjugate gradient method breaks down.             */
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

  ierr = GLTR_VecDot(r, z, &rz); CHKERRQ(ierr);		/* rz = r^T inv(M) r */
  if ((rz != rz) || (rz && (rz / rz != rz / rz))) {
    /*************************************************************************/
    /* The preconditioner produced not-a-number or an infinite value.  This  */
    /* can appear either due to r having numerical problems or M having      */
    /* numerical problems.  Differentiate between the two and then use the   */
    /* gradient direction.                                                   */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_NAN;
    ierr = GLTR_VecDot(r, r, &rr); CHKERRQ(ierr);	/* rr = r^T r        */
    if ((rr != rr) || (rr && (rr / rr != rr / rr))) {
      /***********************************************************************/
      /* The right-hand side contains not-a-number or an infinite value.     */
      /* The gradient step does not work; return a zero value for the step.  */
      /***********************************************************************/

      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: bad right-hand side: rr=%g\n", rr); CHKERRQ(ierr);
    }
    else {
      /***********************************************************************/
      /* The preconditioner contains not-a-number or an infinite value.      */
      /***********************************************************************/

      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: bad preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

      if (cg->radius) {
        alpha = sqrt(r2 / rr);
        ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r   */
        cg->norm_d = cg->radius;
  
	/*********************************************************************/
	/* Compute objective function.                                       */
	/*********************************************************************/

        ierr = KSP_MatMult(ksp, Qmat, d, z); CHKERRQ(ierr)
        ierr = VecAYPX(z, -0.5, ksp->vec_rhs); CHKERRQ(ierr);
        ierr = GLTR_VecDot(d, z, &cg->o_fcn); CHKERRQ(ierr);
        cg->o_fcn = -cg->o_fcn;
      }
    }
    PetscFunctionReturn(0);
  }

  if (rz < 0.0) {
    /*************************************************************************/
    /* The preconditioner is indefinite.  Because this is the first          */
    /* and we do not have a direction yet, we use the gradient step.  Note   */
    /* that we cannot use the preconditioned norm when computing the step    */
    /* because the matrix is indefinite.                                     */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

    if (cg->radius) {
      ierr = GLTR_VecDot(r, r, &rr); CHKERRQ(ierr);	/* rr = r^T r        */
      alpha = sqrt(r2 / rr);
      ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r   */
      cg->norm_d = cg->radius;

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z); CHKERRQ(ierr)
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs); CHKERRQ(ierr);
      ierr = GLTR_VecDot(d, z, &cg->o_fcn); CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
    }
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* As far as we know, the preconditioner is positive semidefinite.  Compute*/
  /* the residual and check for convergence.  Note that there is no choice   */
  /* in the residual that can be used; we must always use the natural        */
  /* residual because this is the only one that can be used by the           */
  /* preconditioned Lanzcos method.                                          */
  /***************************************************************************/

  norm_r = sqrt(rz);					/* norm_r = |r|_M    */
  cg->norm_r[0] = norm_r;

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
  ierr = GLTR_VecDot(p, z, &kappa); CHKERRQ(ierr);	/* kappa = p^T Q p   */
  if ((kappa != kappa) || (kappa && (kappa / kappa != kappa / kappa))) {
    /*************************************************************************/
    /* The matrix produced not-a-number or an infinite value.  In this case, */
    /* we must stop and use the gradient direction.  This condition need     */
    /* only be checked once.                                                 */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_NAN;
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: bad matrix: kappa=%g\n", kappa); CHKERRQ(ierr);

    if (cg->radius) {
      ierr = GLTR_VecDot(r, r, &rr); CHKERRQ(ierr);	/* rr = r^T r        */
      alpha = sqrt(r2 / rr);
      ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r   */
      cg->norm_d = cg->radius;

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z); CHKERRQ(ierr)
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs); CHKERRQ(ierr);
      ierr = GLTR_VecDot(d, z, &cg->o_fcn); CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
    }
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* Check for breakdown of the conjugate gradient method; this occurs when  */
  /* kappa is zero.                                                          */
  /***************************************************************************/

  if (fabs(kappa) <= 0.0) {
    /*************************************************************************/
    /* The curvature is zero.  In this case, we must stop and use the        */
    /* since there the Lanczos matrix is not available.                      */
    /*************************************************************************/
    ksp->reason = KSP_DIVERGED_BREAKDOWN;
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: breakdown: kappa=%g\n", kappa); CHKERRQ(ierr);

    if (cg->radius) {
      ierr = GLTR_VecDot(r, r, &rr); CHKERRQ(ierr);	/* rr = r^T r        */
      alpha = sqrt(r2 / rr);
      ierr = VecAXPY(d, alpha, r); CHKERRQ(ierr);	/* d = d + alpha r   */
      cg->norm_d = cg->radius;

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z); CHKERRQ(ierr)
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs); CHKERRQ(ierr);
      ierr = GLTR_VecDot(d, z, &cg->o_fcn); CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
    }
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* Initialize variables for calculating the norm of the direction and for  */
  /* the Lanczos tridiagonal matrix.  Note that we track the diagonal value  */
  /* of the Cholesky factorization of the Lanczos matrix in order to         */
  /* determine when negative curvature is encountered.                       */
  /***************************************************************************/

  dMp = 0.0;
  norm_p = rz;
  norm_d = 0.0;

  cg->diag[t_size] = kappa / rz;
  cg->offd[t_size] = 0.0;
  ++t_size;

  piv = 1.0;

  /***************************************************************************/
  /* Begin the first part of the GLTR algorithm which runs the conjugate     */
  /* gradient method until either the problem is solved, we encounter the    */
  /* boundary of the trust region, or the conjugate gradient method breaks   */
  /* down.                                                                   */
  /***************************************************************************/

  for (i = 0; i <= max_cg_its; ++i) {
    /*************************************************************************/
    /* Know that kappa is nonzero, because we have not broken down, so we    */
    /* can compute the steplength.                                           */
    /*************************************************************************/

    alpha = rz / kappa;
    cg->alpha[ksp->its] = alpha;

    /*************************************************************************/
    /* Compute the diagonal value of the Cholesky factorization of the       */
    /* Lanczos matrix and check to see if the Lanczos matrix is indefinite.  */
    /* This indicates a direction of negative curvature.                     */
    /*************************************************************************/

    piv = cg->diag[ksp->its] - cg->offd[ksp->its]*cg->offd[ksp->its] / piv;
    if (piv <= 0.0) {
      /***********************************************************************/
      /* In this case, the matrix is indefinite and we have encountered      */
      /* a direction of negative curvature.  Follow the direction to the     */
      /* boundary of the trust region.                                       */
      /***********************************************************************/

      ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: negative curvature: kappa=%g\n", kappa); CHKERRQ(ierr);

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
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: constrained step: radius=%g\n", cg->radius); CHKERRQ(ierr);

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

    norm_d = norm_dp1;
    cg->norm_d = sqrt(norm_d);

    cg->o_fcn -= 0.5 * alpha * rz;

    /*************************************************************************/
    /* Check that the preconditioner appears positive definite.              */
    /*************************************************************************/

    rzm1 = rz;
    ierr = GLTR_VecDot(r, z, &rz); CHKERRQ(ierr);	/* rz = r^T z        */
    if (rz < 0.0) {
      /***********************************************************************/
      /* The preconditioner is indefinite.                                   */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: cg indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* As far as we know, the preconditioner is positive definite.  Compute  */
    /* the residual and check for convergence.  Note that there is no choice */
    /* in the residual that can be used; we must always use the natural      */
    /* residual because this is the only one that can be used by the         */
    /* preconditioned Lanzcos method.                                        */
    /*************************************************************************/

    norm_r = sqrt(rz);					/* norm_r = |r|_M   */
    cg->norm_r[ksp->its+1] = norm_r;

    KSPLogResidualHistory(ksp, norm_r);
    KSPMonitor(ksp, 0, norm_r);
    ksp->rnorm = norm_r;
  
    ierr = (*ksp->converged)(ksp, i+1, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
    if (ksp->reason) {
      ierr = PetscInfo2(ksp,"KSPSolve_GLTR: cg truncated step: rnorm=%g, radius=%g\n",norm_r,cg->radius); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* We have not converged yet, update p and the norms.                    */
    /*************************************************************************/

    beta = rz / rzm1;
    cg->beta[ksp->its] = beta;
    ierr = VecAYPX(p, beta, z); CHKERRQ(ierr);          /* p = z + beta p    */

    dMp = beta*(dMp + alpha*norm_p);
    norm_p = beta*(rzm1 + beta*norm_p);

    /*************************************************************************/
    /* Compute the new direction                                             */
    /*************************************************************************/

    ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p         */
    ierr = GLTR_VecDot(p, z, &kappa); CHKERRQ(ierr);	/* kappa = p^T Q p   */

    /*************************************************************************/
    /* Update the iteration and the Lanczos tridiagonal matrix.              */
    /*************************************************************************/

    ++ksp->its;

    cg->offd[t_size] = sqrt(beta) / fabs(alpha);
    cg->diag[t_size] = kappa / rz + beta / alpha;
    ++t_size;

    /*************************************************************************/
    /* Check for breakdown of the conjugate gradient method; this occurs     */
    /* when kappa is zero.                                                   */
    /*************************************************************************/

    if (fabs(kappa) <= 0.0) {
      /***********************************************************************/
      /* The method breaks down; move along the direction as if the matrix   */
      /* were indefinite.                                                    */
      /***********************************************************************/

      ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: cg breakdown: kappa=%g\n", kappa); CHKERRQ(ierr);

      if (cg->radius) {
        step = (sqrt(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
        ierr = VecAXPY(d, step, p); CHKERRQ(ierr);	/* d = d + step p    */
        cg->norm_d = cg->radius;
      }
      break;
    }
  }

  if (!ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  /***************************************************************************/
  /* Check to see if we need to continue with the Lanczos method.            */
  /***************************************************************************/

  if (!cg->radius) {
    /*************************************************************************/
    /* There is no radius.  Therefore, we cannot move along the boundary.    */
    /*************************************************************************/

    PetscFunctionReturn(0);
  }

  if ((KSP_CONVERGED_CG_CONSTRAINED != ksp->reason) &&
      (KSP_CONVERGED_CG_NEG_CURVE != ksp->reason)) {
    /*************************************************************************/
    /* The method either converged to an interior point or the iteration     */
    /* limit was reached.                                                    */
    /*************************************************************************/

    PetscFunctionReturn(0);
  }
  reason = ksp->reason;

  /***************************************************************************/
  /* Switch to contructing the Lanczos basis by way of the conjugate         */
  /* directions.                                                             */
  /***************************************************************************/
  for (i = 0; i < max_lanczos_its; ++i) {
    /*************************************************************************/
    /* Check for breakdown of the conjugate gradient method; this occurs     */
    /* when kappa is zero.                                                   */
    /*************************************************************************/

    if (fabs(kappa) <= 0.0) {
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: lanczos breakdown: kappa=%g\n", kappa); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Update the direction and residual.                                    */
    /*************************************************************************/
    
    alpha = rz / kappa;
    cg->alpha[ksp->its] = alpha;

    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha Q p */
    ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);

    /*************************************************************************/
    /* Check that the preconditioner appears positive definite.              */
    /*************************************************************************/

    rzm1 = rz;
    ierr = GLTR_VecDot(r, z, &rz); CHKERRQ(ierr);	/* rz = r^T z        */
    if (rz < 0.0) {
      /***********************************************************************/
      /* The preconditioner is indefinite.                                   */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: lanczos indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* As far as we know, the preconditioner is positive definite.  Compute  */
    /* the residual and check for convergence.  Note that there is no choice */
    /* in the residual that can be used; we must always use the natural      */
    /* residual because this is the only one that can be used by the         */
    /* preconditioned Lanzcos method.                                        */
    /*************************************************************************/

    norm_r = sqrt(rz);					/* norm_r = |r|_M   */
    cg->norm_r[ksp->its+1] = norm_r;

    KSPLogResidualHistory(ksp, norm_r);
    KSPMonitor(ksp, 0, norm_r);
    ksp->rnorm = norm_r;
  
    ierr = (*ksp->converged)(ksp, i+1, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
    if (ksp->reason) {
      ierr = PetscInfo2(ksp,"KSPSolve_GLTR: lanczos truncated step: rnorm=%g, radius=%g\n",norm_r,cg->radius); CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Update p and the norms.                                               */
    /*************************************************************************/

    beta = rz / rzm1;
    cg->beta[ksp->its] = beta;
    ierr = VecAYPX(p, beta, z); CHKERRQ(ierr);          /* p = z + beta p    */

    /*************************************************************************/
    /* Compute the new direction                                             */
    /*************************************************************************/

    ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p         */
    ierr = GLTR_VecDot(p, z, &kappa); CHKERRQ(ierr);	/* kappa = p^T Q p   */

    /*************************************************************************/
    /* Update the iteration and the Lanczos tridiagonal matrix.              */
    /*************************************************************************/

    ++ksp->its;

    cg->offd[t_size] = sqrt(beta) / fabs(alpha);
    cg->diag[t_size] = kappa / rz + beta / alpha;
    ++t_size;

    norm_r = sqrt(rz);					/* norm_r = |r|_M   */
  }

  /***************************************************************************/
  /* We have the Lanczos basis, solve the tridiagonal trust-region problem   */
  /* to obtain the Lanczos direction.  We know that the solution lies on     */
  /* the boundary of the trust region.  We start by checking that the        */
  /* workspace allocated is large enough.                                    */
  /***************************************************************************/

  if (t_size > cg->alloced) {
    if (cg->alloced) {
      ierr = PetscFree(cg->rwork); CHKERRQ(ierr);
      ierr = PetscFree(cg->iwork); CHKERRQ(ierr);
      cg->alloced += cg->init_alloc;
    }
    else {
      cg->alloced = cg->init_alloc;
    }

    while (t_size > cg->alloced) {
      cg->alloced += cg->init_alloc;
    }
 
    cg->alloced = PetscMin(cg->alloced, t_size);
    ierr = PetscMalloc(10*sizeof(PetscReal)*cg->alloced, &cg->rwork); CHKERRQ(ierr);
    ierr = PetscMalloc(5*sizeof(PetscInt)*cg->alloced, &cg->iwork); CHKERRQ(ierr);
  }

  /***************************************************************************/
  /* Set up the required vectors.                                            */
  /***************************************************************************/

  t_soln = cg->rwork + 0*t_size;			/* Solution          */
  t_diag = cg->rwork + 1*t_size;			/* Diagonal of T     */
  t_offd = cg->rwork + 2*t_size;			/* Off-diagonal of T */
  e_valu = cg->rwork + 3*t_size;			/* Eigenvalues of T  */
  e_vect = cg->rwork + 4*t_size;			/* Eigenvector of T  */
  e_rwrk = cg->rwork + 5*t_size;			/* Eigen workspace   */
  
  e_iblk = cg->iwork + 0*t_size;			/* Eigen blocks      */
  e_splt = cg->iwork + 1*t_size;			/* Eigen splits      */
  e_iwrk = cg->iwork + 2*t_size;			/* Eigen workspace   */

  /***************************************************************************/
  /* Compute the minimum eigenvalue of T.                                    */
  /***************************************************************************/

  vl = 0.0;
  vu = 0.0;
  il = 1;
  iu = 1;

  LAPACKstebz_("I", "E", &t_size, &vl, &vu, &il, &iu, &cg->eigen_tol,
               cg->diag, cg->offd + 1, &e_valus, &e_splts, e_valu, 
               e_iblk, e_splt, e_rwrk, e_iwrk, &info);

  if ((0 != info) || (1 != e_valus)) {
    /*************************************************************************/
    /* Calculation of the minimum eigenvalue failed.  Return the             */
    /* Steihaug-Toint direction.                                             */
    /*************************************************************************/

    ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute eigenvalue.\n");
    ksp->reason = reason;
    PetscFunctionReturn(0);
  }

  cg->e_min = e_valu[0];

  /***************************************************************************/
  /* Compute the initial value of lambda to make (T + lamba I) positive      */
  /* definite.                                                               */
  /***************************************************************************/

  pert = cg->init_pert;
  if (e_valu[0] <= 0.0) {
    lambda = pert - e_valu[0];
  }
  else {
    lambda = 0.0;
  }

  while(1) {
    for (i = 0; i < t_size; ++i) {
      t_diag[i] = cg->diag[i] + lambda;
      t_offd[i] = cg->offd[i];
    }

    LAPACKpttrf_(&t_size, t_diag, t_offd + 1, &info);
    if (0 == info) {
      break;
    }

    pert += pert;
    lambda = lambda * (1.0 + pert) + pert;
  }

  /***************************************************************************/
  /* Compute the initial step and its norm.                                  */
  /***************************************************************************/

  nrhs = 1;
  nldb = t_size;

  t_soln[0] = -cg->norm_r[0];
  for (i = 1; i < t_size; ++i) {
    t_soln[i] = 0.0;
  }

  LAPACKpttrs_(&t_size, &nrhs, t_diag, t_offd + 1, t_soln, &nldb, &info);
  if (0 != info) {
    /*************************************************************************/
    /* Calculation of the initial step failed; return the Steihaug-Toint     */
    /* direction.                                                            */
    /*************************************************************************/

    ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute step.\n");
    ksp->reason = reason;
    PetscFunctionReturn(0);
  }

  norm_t = 0;
  for (i = 0; i < t_size; ++i) {
    norm_t += t_soln[i] * t_soln[i];
  }
  norm_t = sqrt(norm_t);

  /***************************************************************************/
  /* Determine the case we are in.                                           */
  /***************************************************************************/

  if (norm_t <= cg->radius) {
    /*************************************************************************/
    /* The step is within the trust region; check if we are in the hard case */
    /* and need to move to the boundary by following a direction of negative */
    /* curvature.                                                            */
    /*************************************************************************/
    
    if ((e_valu[0] <= 0.0) && (norm_t < cg->radius)) {
      /***********************************************************************/
      /* This is the hard case; compute the eigenvector associated with the  */
      /* minimum eigenvalue and move along this direction to the boundary.   */
      /***********************************************************************/

      LAPACKstein_(&t_size, cg->diag, cg->offd + 1, &e_valus, e_valu,
		   e_iblk, e_splt, e_vect, &nldb, 
		   e_rwrk, e_iwrk, e_iwrk + t_size, &info);
      if (0 != info) {
	/*********************************************************************/
	/* Calculation of the minimum eigenvalue failed.  Return the         */
	/* Steihaug-Toint direction.                                         */
	/*********************************************************************/
	
	ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute eigenvector.\n");
	ksp->reason = reason;
	PetscFunctionReturn(0);
      }
      
      coef1 = 0.0;
      coef2 = 0.0;
      coef3 = -cg->radius * cg->radius;
      for (i = 0; i < t_size; ++i) {
	coef1 += e_vect[i] * e_vect[i];
	coef2 += e_vect[i] * t_soln[i];
	coef3 += t_soln[i] * t_soln[i];
      }
      
      coef3 = sqrt(coef2 * coef2 - coef1 * coef3);
      root1 = (-coef2 + coef3) / coef1;
      root2 = (-coef2 - coef3) / coef1;

      /***********************************************************************/
      /* Compute objective value for (t_soln + root1 * e_vect)               */
      /***********************************************************************/

      for (i = 0; i < t_size; ++i) {
	e_rwrk[i] = t_soln[i] + root1 * e_vect[i];
      }
      
      obj1 = e_rwrk[0]*(0.5*(cg->diag[0]*e_rwrk[0]+
			     cg->offd[1]*e_rwrk[1])+cg->norm_r[0]);
      for (i = 1; i < t_size - 1; ++i) {
	obj1 += 0.5*e_rwrk[i]*(cg->offd[i]*e_rwrk[i-1]+
			       cg->diag[i]*e_rwrk[i]+
			       cg->offd[i+1]*e_rwrk[i+1]);
      }
      obj1 += 0.5*e_rwrk[i]*(cg->offd[i]*e_rwrk[i-1]+
			     cg->diag[i]*e_rwrk[i]);

      /***********************************************************************/
      /* Compute objective value for (t_soln + root2 * e_vect)               */
      /***********************************************************************/

      for (i = 0; i < t_size; ++i) {
	e_rwrk[i] = t_soln[i] + root2 * e_vect[i];
      }
      
      obj2 = e_rwrk[0]*(0.5*(cg->diag[0]*e_rwrk[0]+
			     cg->offd[1]*e_rwrk[1])+cg->norm_r[0]);
      for (i = 1; i < t_size - 1; ++i) {
	obj2 += 0.5*e_rwrk[i]*(cg->offd[i]*e_rwrk[i-1]+
			       cg->diag[i]*e_rwrk[i]+
			       cg->offd[i+1]*e_rwrk[i+1]);
      }
      obj2 += 0.5*e_rwrk[i]*(cg->offd[i]*e_rwrk[i-1]+
			     cg->diag[i]*e_rwrk[i]);
      
      /***********************************************************************/
      /* Choose the point with the best objective function value.            */
      /***********************************************************************/

      if (obj1 <= obj2) {
	for (i = 0; i < t_size; ++i) {
	  t_soln[i] += root1 * e_vect[i];
	}
      }
      else {
        for (i = 0; i < t_size; ++i) {
          t_soln[i] += root2 * e_vect[i];
        }
      }
    }
    else {
      /***********************************************************************/
      /* The matrix is positive definite or there was no room to move; the   */
      /* solution is already contained in t_soln.                            */
      /***********************************************************************/
    }
  }
  else {
    /*************************************************************************/
    /* The step is outside the trust-region.  Compute the correct value for  */
    /* lambda by performing Newton's method.                                 */
    /*************************************************************************/

    for (i = 0; i < max_newton_its; ++i) {
      /***********************************************************************/
      /* Check for convergence.                                              */
      /***********************************************************************/

      if (fabs(norm_t - cg->radius) <= cg->newton_tol * cg->radius) {
	break;
      }

      /***********************************************************************/
      /* Compute the update.                                                 */
      /***********************************************************************/

      PetscMemcpy(e_rwrk, t_soln, sizeof(PetscReal)*t_size);
      
      LAPACKpttrs_(&t_size, &nrhs, t_diag, t_offd + 1, e_rwrk, &nldb, &info);
      if (0 != info) {
	/*********************************************************************/
	/* Calculation of the step failed; return the Steihaug-Toint         */
	/* direction.                                                        */
	/*********************************************************************/

	ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute step.\n");
	ksp->reason = reason;
	PetscFunctionReturn(0);
      }

      /***********************************************************************/
      /* Modify lambda.                                                      */
      /***********************************************************************/

      norm_w = 0;
      for (j = 0; j < t_size; ++j) {
	norm_w += t_soln[j] * e_rwrk[j];
      }
      
      lambda += (norm_t - cg->radius)/cg->radius * (norm_t * norm_t) / norm_w;

      /***********************************************************************/
      /* Factor T + lambda I                                                 */
      /***********************************************************************/
      
      for (j = 0; j < t_size; ++j) {
	t_diag[j] = cg->diag[j] + lambda;
	t_offd[j] = cg->offd[j];
      }

      LAPACKpttrf_(&t_size, t_diag, t_offd + 1, &info);
      if (0 != info) {
	/*********************************************************************/
	/* Calculation of factorization failed; return the Steihaug-Toint    */
	/* direction.                                                        */
	/*********************************************************************/

	ierr = PetscInfo(ksp, "KSPSolve_GLTR: factorization failed.\n");
	ksp->reason = reason;
	PetscFunctionReturn(0);
      }

      /***********************************************************************/
      /* Compute the new step and its norm.                                  */
      /***********************************************************************/

      t_soln[0] = -cg->norm_r[0];
      for (j = 1; j < t_size; ++j) {
	t_soln[j] = 0.0;
      }

      LAPACKpttrs_(&t_size, &nrhs, t_diag, t_offd + 1, t_soln, &nldb, &info);
      if (0 != info) {
	/*********************************************************************/
	/* Calculation of the step failed; return the Steihaug-Toint         */
	/* direction.                                                        */
	/*********************************************************************/
	
	ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute step.\n");
	ksp->reason = reason;
	PetscFunctionReturn(0);
      }

      norm_t = 0;
      for (j = 0; j < t_size; ++j) {
	norm_t += t_soln[j] * t_soln[j];
      }
      norm_t = sqrt(norm_t);
    }

    /*************************************************************************/
    /* Check for convergence.                                                */
    /*************************************************************************/

    if (fabs(norm_t - cg->radius) > cg->newton_tol * cg->radius) {
      /***********************************************************************/
      /* Newton method failed to converge in iteration limit.                */
      /***********************************************************************/

      ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to converge.\n");
      ksp->reason = reason;
      PetscFunctionReturn(0);
    }
  }

  /***************************************************************************/
  /* Recover the norm of the direction and objective function value.         */
  /***************************************************************************/

  cg->norm_d = norm_t;

  cg->o_fcn = t_soln[0]*(0.5*(cg->diag[0]*t_soln[0]+
			      cg->offd[1]*t_soln[1])+cg->norm_r[0]);
  for (i = 1; i < t_size - 1; ++i) {
    cg->o_fcn += 0.5*t_soln[i]*(cg->offd[i]*t_soln[i-1]+
		 	        cg->diag[i]*t_soln[i]+
			        cg->offd[i+1]*t_soln[i+1]);
  }
  cg->o_fcn += 0.5*t_soln[i]*(cg->offd[i]*t_soln[i-1]+
			      cg->diag[i]*t_soln[i]);

  /***************************************************************************/
  /* Recover the direction.                                                  */
  /***************************************************************************/

  sigma = -1;

  /***************************************************************************/
  /* Start conjugate gradient method from the beginning                      */
  /***************************************************************************/

  ierr = VecCopy(ksp->vec_rhs, r); CHKERRQ(ierr);	/* r = -grad         */
  ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);		/* z = inv(M) r      */

  /***************************************************************************/
  /* Accumulate Q * s                                                        */
  /***************************************************************************/

  ierr = VecCopy(z, d); CHKERRQ(ierr);
  ierr = VecScale(d, sigma * t_soln[0] / cg->norm_r[0]); CHKERRQ(ierr);
 
  /***************************************************************************/
  /* Compute the first direction.                                            */
  /***************************************************************************/

  ierr = VecCopy(z, p); CHKERRQ(ierr);			/* p = z             */
  ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p         */

  for (i = 0; i < ksp->its - 1; ++i) {
    /*************************************************************************/
    /* Update the residual and direction.                                    */
    /*************************************************************************/

    alpha = cg->alpha[i];
    if (alpha >= 0.0) {
      sigma = -sigma;
    }

    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha Q p */
    ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);

    /*************************************************************************/
    /* Accumulate Q * s                                                      */
    /*************************************************************************/

    ierr = VecAXPY(d, sigma * t_soln[i+1] / cg->norm_r[i+1], z);

    /*************************************************************************/
    /* Update p.                                                             */
    /*************************************************************************/

    beta = cg->beta[i];
    ierr = VecAYPX(p, beta, z); CHKERRQ(ierr);          /* p = z + beta p    */
    ierr = KSP_MatMult(ksp, Qmat, p, z); CHKERRQ(ierr);	/* z = Q * p         */
  }

  if (i < ksp->its) {
    /*************************************************************************/
    /* Update the residual and direction.                                    */
    /*************************************************************************/

    alpha = cg->alpha[i];
    if (alpha >= 0.0) {
      sigma = -sigma;
    }

    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha Q p */
    ierr = KSP_PCApply(ksp, r, z); CHKERRQ(ierr);

    /*************************************************************************/
    /* Accumulate Q * s                                                      */
    /*************************************************************************/

    ierr = VecAXPY(d, sigma * t_soln[i+1] / cg->norm_r[i+1], z);
  }

  /***************************************************************************/
  /* Set the termination reason.                                             */
  /***************************************************************************/

  ksp->reason = reason;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_GLTR"
PetscErrorCode KSPSetUp_GLTR(KSP ksp)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;
  PetscInt size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* This implementation of CG only handles left preconditioning
   * so generate an error otherwise.
   */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP, "No right preconditioning for KSPGLTR");
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP, "No symmetric preconditioning for KSPGLTR");
  }

  /* get work vectors needed by CG */
  ierr = KSPDefaultGetWork(ksp, 3); CHKERRQ(ierr);

  /* allocate workspace for Lanczos matrix */
  cg->max_its = cg->max_cg_its + cg->max_lanczos_its + 1;
  size = 5*cg->max_its*sizeof(PetscReal);

  ierr = PetscMalloc(size, &cg->diag); CHKERRQ(ierr);
  ierr = PetscMemzero(cg->diag, size); CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp, size); CHKERRQ(ierr);
  cg->offd   = cg->diag  + cg->max_its;
  cg->alpha  = cg->offd  + cg->max_its;
  cg->beta   = cg->alpha + cg->max_its;
  cg->norm_r = cg->beta  + cg->max_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_GLTR"
PetscErrorCode KSPDestroy_GLTR(KSP ksp)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(cg->diag); CHKERRQ(ierr);
  if (cg->alloced) {
    ierr = PetscFree(cg->rwork);
    ierr = PetscFree(cg->iwork);
  }

  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRSetRadius_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRGetNormD_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRGetObjFcn_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRGetMinEig_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPGLTRSetRadius_GLTR"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRSetRadius_GLTR(KSP ksp,PetscReal radius)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  cg->radius = radius;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetNormD_GLTR"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetNormD_GLTR(KSP ksp,PetscReal *norm_d)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  *norm_d = cg->norm_d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetObjFcn_GLTR"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetObjFcn_GLTR(KSP ksp,PetscReal *o_fcn)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  *o_fcn = cg->o_fcn;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetMinEig_GLTR"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetMinEig_GLTR(KSP ksp,PetscReal *e_min)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  *e_min = cg->e_min;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_GLTR"
PetscErrorCode KSPSetFromOptions_GLTR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP GLTR options"); CHKERRQ(ierr);

  ierr = PetscOptionsReal("-ksp_gltr_radius", "Trust Region Radius", "KSPGLTRSetRadius", cg->radius, &cg->radius, PETSC_NULL); CHKERRQ(ierr);

  ierr = PetscOptionsReal("-ksp_gltr_init_pert", "Initial perturbation", "", cg->init_pert, &cg->init_pert, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_gltr_eigen_tol", "Eigenvalue tolerance", "", cg->eigen_tol, &cg->eigen_tol, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_gltr_newton_tol", "Newton tolerance", "", cg->newton_tol, &cg->newton_tol, PETSC_NULL); CHKERRQ(ierr);

  ierr = PetscOptionsInt("-ksp_gltr_max_cg_its", "Maximum Conjugate Gradient Iters", "", cg->max_cg_its, &cg->max_cg_its, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_gltr_max_lanczos_its", "Maximum Lanczos Iters", "", cg->max_lanczos_its, &cg->max_lanczos_its, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_gltr_max_newton_its", "Maximum Newton Iters", "", cg->max_newton_its, &cg->max_newton_its, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPGLTR -   Code to run conjugate gradient method subject to a constraint
         on the solution norm. This is used in Trust Region methods for
         nonlinear equations, SNESTR

   Options Database Keys:
.      -ksp_gltr_radius <r> - Trust Region Radius

   Notes: This is rarely used directly

   Level: developer

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPGLTRSetRadius(), KSPGLTRGetNormD(), KSPGLTRGetObjFcn(), KSPGLTRGetMinEig()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_GLTR"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_GLTR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_GLTR *cg;

  PetscFunctionBegin;

  ierr = PetscNew(KSP_GLTR, &cg); CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp, sizeof(KSP_GLTR)); CHKERRQ(ierr);

  cg->radius = PETSC_MAX;

  cg->init_pert = 1.0e-8;
  cg->eigen_tol = 1.0e-10;
  cg->newton_tol = 1.0e-6;

  cg->alloced = 0;
  cg->init_alloc = 1024;

  cg->max_cg_its = 10000;
  cg->max_lanczos_its = 20;
  cg->max_newton_its  = 10;

  cg->max_its = cg->max_cg_its + cg->max_lanczos_its + 1;

  ksp->data = (void *) cg;
  ksp->pc_side = PC_LEFT;

  /* Sets the functions that are associated with this data structure
   * (in C++ this is the same as defining virtual functions)
   */
  ksp->ops->setup          = KSPSetUp_GLTR;
  ksp->ops->solve          = KSPSolve_GLTR;
  ksp->ops->destroy        = KSPDestroy_GLTR;
  ksp->ops->setfromoptions = KSPSetFromOptions_GLTR;
  ksp->ops->buildsolution  = KSPDefaultBuildSolution;
  ksp->ops->buildresidual  = KSPDefaultBuildResidual;
  ksp->ops->view           = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                    "KSPGLTRSetRadius_C",
                                    "KSPGLTRSetRadius_GLTR",
                                     KSPGLTRSetRadius_GLTR); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                    "KSPGLTRGetNormD_C",
                                    "KSPGLTRGetNormD_GLTR",
                                     KSPGLTRGetNormD_GLTR); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                    "KSPGLTRGetObjFcn_C",
                                    "KSPGLTRGetObjFcn_GLTR",
                                     KSPGLTRGetObjFcn_GLTR); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                    "KSPGLTRGetMinEig_C",
                                    "KSPGLTRGetMinEig_GLTR",
                                     KSPGLTRGetMinEig_GLTR); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
