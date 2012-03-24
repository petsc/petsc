
#include <petsc-private/kspimpl.h>             /*I "petscksp.h" I*/
#include <../src/ksp/ksp/impls/cg/gltr/gltrimpl.h>
#include <petscblaslapack.h>

#define GLTR_PRECONDITIONED_DIRECTION   0
#define GLTR_UNPRECONDITIONED_DIRECTION 1
#define GLTR_DIRECTION_TYPES            2

static const char *DType_Table[64] = {"preconditioned", "unpreconditioned"};

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRSetRadius"
/*@
    KSPGLTRSetRadius - Sets the radius of the trust region.

    Logically Collective on KSP

    Input Parameters:
+   ksp    - the iterative context
-   radius - the trust region radius (Infinity is the default)

    Options Database Key:
.   -ksp_gltr_radius <r>

    Level: advanced

.keywords: KSP, GLTR, set, trust region radius
@*/
PetscErrorCode  KSPGLTRSetRadius(KSP ksp, PetscReal radius)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (radius < 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_OUTOFRANGE, "Radius negative");
  PetscValidLogicalCollectiveReal(ksp,radius,2);
  ierr = PetscTryMethod(ksp,"KSPGLTRSetRadius_C",(KSP,PetscReal),(ksp,radius));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetNormD"
/*@
    KSPGLTRGetNormD - Get norm of the direction.

    Collective

    Input Parameters:
+   ksp    - the iterative context
-   norm_d - the norm of the direction

    Level: advanced

.keywords: KSP, GLTR, get, norm direction
@*/
PetscErrorCode  KSPGLTRGetNormD(KSP ksp, PetscReal *norm_d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ierr = PetscUseMethod(ksp,"KSPGLTRGetNormD_C",(KSP,PetscReal*),(ksp,norm_d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetObjFcn"
/*@
    KSPGLTRGetObjFcn - Get objective function value.

    Note Collective

    Input Parameters:
+   ksp   - the iterative context
-   o_fcn - the objective function value

    Level: advanced

.keywords: KSP, GLTR, get, objective function
@*/
PetscErrorCode  KSPGLTRGetObjFcn(KSP ksp, PetscReal *o_fcn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ierr = PetscUseMethod(ksp,"KSPGLTRGetObjFcn_C",(KSP,PetscReal*),(ksp,o_fcn));CHKERRQ(ierr);
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
PetscErrorCode  KSPGLTRGetMinEig(KSP ksp, PetscReal *e_min)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ierr = PetscUseMethod(ksp,"KSPGLTRGetMinEig_C",(KSP,PetscReal*),(ksp,e_min));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetLambda"
/*@
    KSPGLTRGetLambda - Get multiplier on trust-region constraint.

    Not Collective

    Input Parameters:
+   ksp    - the iterative context
-   lambda - the multiplier

    Level: advanced

.keywords: KSP, GLTR, get, multiplier
@*/
PetscErrorCode  KSPGLTRGetLambda(KSP ksp, PetscReal *lambda)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ierr = PetscUseMethod(ksp,"KSPGLTRGetLambda_C",(KSP,PetscReal*),(ksp,lambda));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_GLTR"
PetscErrorCode KSPSolve_GLTR(KSP ksp)
{
#ifdef PETSC_USE_COMPLEX
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP, "GLTR is not available for complex systems");
#else
  KSP_GLTR       *cg = (KSP_GLTR *)ksp->data;
  PetscReal      *t_soln, *t_diag, *t_offd, *e_valu, *e_vect, *e_rwrk;
  PetscBLASInt   *e_iblk, *e_splt, *e_iwrk;

  PetscErrorCode ierr;
  MatStructure   pflag;
  Mat            Qmat, Mmat;
  Vec            r, z, p, d;
  PC             pc;

  PetscReal      norm_r, norm_d, norm_dp1, norm_p, dMp;
  PetscReal      alpha, beta, kappa, rz, rzm1;
  PetscReal      rr, r2, piv, step;
  PetscReal      vl, vu;
  PetscReal      coef1, coef2, coef3, root1, root2, obj1, obj2;
  PetscReal      norm_t, norm_w, pert;

  PetscInt       i, j, max_cg_its, max_lanczos_its, max_newton_its, sigma;
  PetscBLASInt   t_size = 0, l_size = 0, il, iu, info;
  PetscBLASInt   nrhs, nldb;

#if !defined(PETSC_MISSING_LAPACK_STEBZ)
  PetscBLASInt  e_valus, e_splts;
#endif
  PetscBool     diagonalscale;

  PetscFunctionBegin;
  /***************************************************************************/
  /* Check the arguments and parameters.                                     */
  /***************************************************************************/

  ierr = PCGetDiagonalScale(ksp->pc, &diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  if (cg->radius < 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_OUTOFRANGE, "Input error: radius < 0");

  /***************************************************************************/
  /* Get the workspace vectors and initialize variables                      */
  /***************************************************************************/

  r2 = cg->radius * cg->radius;
  r  = ksp->work[0];
  z  = ksp->work[1];
  p  = ksp->work[2];
  d  = ksp->vec_sol;
  pc = ksp->pc;

  ierr = PCGetOperators(pc, &Qmat, &Mmat, &pflag);CHKERRQ(ierr);

  ierr = VecGetSize(d, &max_cg_its);CHKERRQ(ierr);
  max_cg_its = PetscMin(max_cg_its, ksp->max_it);
  max_lanczos_its = cg->max_lanczos_its;
  max_newton_its = cg->max_newton_its;
  ksp->its = 0;

  /***************************************************************************/
  /* Initialize objective function direction, and minimum eigenvalue.        */
  /***************************************************************************/

  cg->o_fcn = 0.0;

  ierr = VecSet(d, 0.0);CHKERRQ(ierr);			/* d = 0             */
  cg->norm_d = 0.0;

  cg->e_min = 0.0;
  cg->lambda = 0.0;

  /***************************************************************************/
  /* The first phase of GLTR performs a standard conjugate gradient method,  */
  /* but stores the values required for the Lanczos matrix.  We switch to    */
  /* the Lanczos when the conjugate gradient method breaks down.  Check the  */
  /* right-hand side for numerical problems.  The check for not-a-number and */
  /* infinite values need be performed only once.                            */
  /***************************************************************************/

  ierr = VecCopy(ksp->vec_rhs, r);CHKERRQ(ierr);	/* r = -grad         */
  ierr = VecDot(r, r, &rr);CHKERRQ(ierr);		/* rr = r^T r        */
  if (PetscIsInfOrNanScalar(rr)) {
    /*************************************************************************/
    /* The right-hand side contains not-a-number or an infinite value.       */
    /* The gradient step does not work; return a zero value for the step.    */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_NAN;
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: bad right-hand side: rr=%g\n", rr);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* Check the preconditioner for numerical problems and for positive        */
  /* definiteness.  The check for not-a-number and infinite values need be   */
  /* performed only once.                                                    */
  /***************************************************************************/

  ierr = KSP_PCApply(ksp, r, z);CHKERRQ(ierr);		/* z = inv(M) r      */
  ierr = VecDot(r, z, &rz);CHKERRQ(ierr);		/* rz = r^T inv(M) r */
  if (PetscIsInfOrNanScalar(rz)) {
    /*************************************************************************/
    /* The preconditioner contains not-a-number or an infinite value.        */
    /* Return the gradient direction intersected with the trust region.      */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_NAN;
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: bad preconditioner: rz=%g\n", rz);CHKERRQ(ierr);

    if (cg->radius) {
      if (r2 >= rr) {
        alpha = 1.0;
        cg->norm_d = PetscSqrtReal(rr);
      }
      else {
        alpha = PetscSqrtReal(r2 / rr);
        cg->norm_d = cg->radius;
      }

      ierr = VecAXPY(d, alpha, r);CHKERRQ(ierr);	/* d = d + alpha r   */
  
      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z);CHKERRQ(ierr);
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs);CHKERRQ(ierr);
      ierr = VecDot(d, z, &cg->o_fcn);CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
      ++ksp->its;
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
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: indefinite preconditioner: rz=%g\n", rz);CHKERRQ(ierr);

    if (cg->radius) {
      if (r2 >= rr) {
        alpha = 1.0;
        cg->norm_d = PetscSqrtReal(rr);
      }
      else {
        alpha = PetscSqrtReal(r2 / rr);
        cg->norm_d = cg->radius;
      }

      ierr = VecAXPY(d, alpha, r);CHKERRQ(ierr);	/* d = d + alpha r   */

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z);CHKERRQ(ierr);
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs);CHKERRQ(ierr);
      ierr = VecDot(d, z, &cg->o_fcn);CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
      ++ksp->its;
    }
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* As far as we know, the preconditioner is positive semidefinite.         */
  /* Compute and log the residual.  Check convergence because this           */
  /* initializes things, but do not terminate until at least one conjugate   */
  /* gradient iteration has been performed.                                  */
  /***************************************************************************/

  cg->norm_r[0] = PetscSqrtReal(rz);				/* norm_r = |r|_M    */

  switch(ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = VecNorm(z, NORM_2, &norm_r);CHKERRQ(ierr);	/* norm_r = |z|      */
    break;

  case KSP_NORM_UNPRECONDITIONED:
    norm_r = PetscSqrtReal(rr);					/* norm_r = |r|      */
    break;

  case KSP_NORM_NATURAL:
    norm_r = cg->norm_r[0];				/* norm_r = |r|_M    */
    break;

  default:
    norm_r = 0.0;
    break;
  }

  KSPLogResidualHistory(ksp, norm_r);
  ierr = KSPMonitor(ksp, ksp->its, norm_r);CHKERRQ(ierr);
  ksp->rnorm = norm_r;

  ierr = (*ksp->converged)(ksp, ksp->its, norm_r, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);

  /***************************************************************************/
  /* Compute the first direction and update the iteration.                   */
  /***************************************************************************/

  ierr = VecCopy(z, p);CHKERRQ(ierr);			/* p = z             */
  ierr = KSP_MatMult(ksp, Qmat, p, z);CHKERRQ(ierr);	/* z = Q * p         */
  ++ksp->its;

  /***************************************************************************/
  /* Check the matrix for numerical problems.                                */
  /***************************************************************************/

  ierr = VecDot(p, z, &kappa);CHKERRQ(ierr);		/* kappa = p^T Q p   */
  if (PetscIsInfOrNanScalar(kappa)) {
    /*************************************************************************/
    /* The matrix produced not-a-number or an infinite value.  In this case, */
    /* we must stop and use the gradient direction.  This condition need     */
    /* only be checked once.                                                 */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_NAN;
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: bad matrix: kappa=%g\n", kappa);CHKERRQ(ierr);

    if (cg->radius) {
      if (r2 >= rr) {
        alpha = 1.0;
        cg->norm_d = PetscSqrtReal(rr);
      }
      else {
        alpha = PetscSqrtReal(r2 / rr);
        cg->norm_d = cg->radius;
      }

      ierr = VecAXPY(d, alpha, r);CHKERRQ(ierr);	/* d = d + alpha r   */

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z);CHKERRQ(ierr);
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs);CHKERRQ(ierr);
      ierr = VecDot(d, z, &cg->o_fcn);CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
      ++ksp->its;
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
  norm_d = 0.0;
  switch(cg->dtype) {
  case GLTR_PRECONDITIONED_DIRECTION:
    norm_p = rz;
    break;

  default:
    ierr = VecDot(p, p, &norm_p);CHKERRQ(ierr);
    break;
  }

  cg->diag[t_size] = kappa / rz;
  cg->offd[t_size] = 0.0;
  ++t_size;

  piv = 1.0;

  /***************************************************************************/
  /* Check for breakdown of the conjugate gradient method; this occurs when  */
  /* kappa is zero.                                                          */
  /***************************************************************************/

  if (fabs(kappa) <= 0.0) {
    /*************************************************************************/
    /* The curvature is zero.  In this case, we must stop and use follow     */
    /* the direction of negative curvature since the Lanczos matrix is zero. */
    /*************************************************************************/

    ksp->reason = KSP_DIVERGED_BREAKDOWN;
    ierr = PetscInfo1(ksp, "KSPSolve_GLTR: breakdown: kappa=%g\n", kappa);CHKERRQ(ierr);

    if (cg->radius && norm_p > 0.0) {
      /***********************************************************************/
      /* Follow direction of negative curvature to the boundary of the       */
      /* trust region.                                                       */
      /***********************************************************************/

      step = PetscSqrtReal(r2 / norm_p);
      cg->norm_d = cg->radius;

      ierr = VecAXPY(d, step, p);CHKERRQ(ierr);	/* d = d + step p    */

      /***********************************************************************/
      /* Update objective function.                                          */
      /***********************************************************************/

      cg->o_fcn += step * (0.5 * step * kappa - rz);
    }
    else if (cg->radius) {
      /***********************************************************************/
      /* The norm of the preconditioned direction is zero; use the gradient  */
      /* step.                                                               */
      /***********************************************************************/

      if (r2 >= rr) {
        alpha = 1.0;
        cg->norm_d = PetscSqrtReal(rr);
      }
      else {
        alpha = PetscSqrtReal(r2 / rr);
        cg->norm_d = cg->radius;
      }

      ierr = VecAXPY(d, alpha, r);CHKERRQ(ierr);	/* d = d + alpha r   */

      /***********************************************************************/
      /* Compute objective function.                                         */
      /***********************************************************************/

      ierr = KSP_MatMult(ksp, Qmat, d, z);CHKERRQ(ierr);
      ierr = VecAYPX(z, -0.5, ksp->vec_rhs);CHKERRQ(ierr);
      ierr = VecDot(d, z, &cg->o_fcn);CHKERRQ(ierr);
      cg->o_fcn = -cg->o_fcn;
      ++ksp->its;
    }
    PetscFunctionReturn(0);
  }

  /***************************************************************************/
  /* Begin the first part of the GLTR algorithm which runs the conjugate     */
  /* gradient method until either the problem is solved, we encounter the    */
  /* boundary of the trust region, or the conjugate gradient method breaks   */
  /* down.                                                                   */
  /***************************************************************************/

  while(1) {
    /*************************************************************************/
    /* Know that kappa is nonzero, because we have not broken down, so we    */
    /* can compute the steplength.                                           */
    /*************************************************************************/

    alpha = rz / kappa;
    cg->alpha[l_size] = alpha;

    /*************************************************************************/
    /* Compute the diagonal value of the Cholesky factorization of the       */
    /* Lanczos matrix and check to see if the Lanczos matrix is indefinite.  */
    /* This indicates a direction of negative curvature.                     */
    /*************************************************************************/

    piv = cg->diag[l_size] - cg->offd[l_size]*cg->offd[l_size] / piv;
    if (piv <= 0.0) {
      /***********************************************************************/
      /* In this case, the matrix is indefinite and we have encountered      */
      /* a direction of negative curvature.  Follow the direction to the     */
      /* boundary of the trust region.                                       */
      /***********************************************************************/

      ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: negative curvature: kappa=%g\n", kappa);CHKERRQ(ierr);

      if (cg->radius && norm_p > 0.0) {
        /*********************************************************************/
        /* Follow direction of negative curvature to boundary.               */
        /*********************************************************************/

        step = (PetscSqrtReal(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
        cg->norm_d = cg->radius;

        ierr = VecAXPY(d, step, p);CHKERRQ(ierr);	/* d = d + step p    */

	/*********************************************************************/
        /* Update objective function.                                        */
	/*********************************************************************/

        cg->o_fcn += step * (0.5 * step * kappa - rz);
      }
      else if (cg->radius) {
        /*********************************************************************/
        /* The norm of the direction is zero; there is nothing to follow.    */
        /*********************************************************************/
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
      /* However, the full step goes beyond the trust region.                */
      /***********************************************************************/

      ksp->reason = KSP_CONVERGED_CG_CONSTRAINED;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: constrained step: radius=%g\n", cg->radius);CHKERRQ(ierr);

      if (norm_p > 0.0) {
	/*********************************************************************/
	/* Follow the direction to the boundary of the trust region.         */
	/*********************************************************************/

        step = (PetscSqrtReal(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
        cg->norm_d = cg->radius;

        ierr = VecAXPY(d, step, p);CHKERRQ(ierr);	/* d = d + step p    */

        /*********************************************************************/
        /* Update objective function.                                        */
        /*********************************************************************/

        cg->o_fcn += step * (0.5 * step * kappa - rz);
      }
      else {
	/*********************************************************************/
	/* The norm of the direction is zero; there is nothing to follow.    */
	/*********************************************************************/
      }
      break;
    }

    /*************************************************************************/
    /* Now we can update the direction and residual.                         */
    /*************************************************************************/

    ierr = VecAXPY(d, alpha, p);CHKERRQ(ierr);		/* d = d + alpha p   */
    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha Q p */
    ierr = KSP_PCApply(ksp, r, z);CHKERRQ(ierr);	/* z = inv(M) r      */

    switch(cg->dtype) {
    case GLTR_PRECONDITIONED_DIRECTION:
      norm_d = norm_dp1;
      break;

    default:
      ierr = VecDot(d, d, &norm_d);CHKERRQ(ierr);
      break;
    }
    cg->norm_d = PetscSqrtReal(norm_d);

    /*************************************************************************/
    /* Update objective function.                                            */
    /*************************************************************************/

    cg->o_fcn -= 0.5 * alpha * rz;

    /*************************************************************************/
    /* Check that the preconditioner appears positive semidefinite.          */
    /*************************************************************************/

    rzm1 = rz;
    ierr = VecDot(r, z, &rz);CHKERRQ(ierr);		/* rz = r^T z        */
    if (rz < 0.0) {
      /***********************************************************************/
      /* The preconditioner is indefinite.                                   */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: cg indefinite preconditioner: rz=%g\n", rz);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* As far as we know, the preconditioner is positive semidefinite.       */
    /* Compute the residual and check for convergence.                       */
    /*************************************************************************/

    cg->norm_r[l_size+1] = PetscSqrtReal(rz);			/* norm_r = |r|_M   */

    switch(ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = VecNorm(z, NORM_2, &norm_r);CHKERRQ(ierr);/* norm_r = |z|      */
      break;

    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(r, NORM_2, &norm_r);CHKERRQ(ierr);/* norm_r = |r|      */
      break;

    case KSP_NORM_NATURAL:
      norm_r = cg->norm_r[l_size+1];			/* norm_r = |r|_M    */
      break;

    default:
      norm_r = 0.0;
      break;
    }

    KSPLogResidualHistory(ksp, norm_r);
    ierr = KSPMonitor(ksp, ksp->its, norm_r);CHKERRQ(ierr);
    ksp->rnorm = norm_r;
  
    ierr = (*ksp->converged)(ksp, ksp->its, norm_r, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) {
      /***********************************************************************/
      /* The method has converged.                                           */
      /***********************************************************************/

      ierr = PetscInfo2(ksp, "KSPSolve_GLTR: cg truncated step: rnorm=%g, radius=%g\n", norm_r, cg->radius);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* We have not converged yet.  Check for breakdown.                      */
    /*************************************************************************/

    beta = rz / rzm1;
    if (fabs(beta) <= 0.0) {
      /***********************************************************************/
      /* Conjugate gradients has broken down.                                */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: breakdown: beta=%g\n", beta);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Check iteration limit.                                                */
    /*************************************************************************/

    if (ksp->its >= max_cg_its) {
      ksp->reason = KSP_DIVERGED_ITS;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: iterlim: its=%d\n", ksp->its);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Update p and the norms.                                               */
    /*************************************************************************/

    cg->beta[l_size] = beta;
    ierr = VecAYPX(p, beta, z);CHKERRQ(ierr);		/* p = z + beta p    */

    switch(cg->dtype) {
    case GLTR_PRECONDITIONED_DIRECTION:
      dMp = beta*(dMp + alpha*norm_p);
      norm_p = beta*(rzm1 + beta*norm_p);
      break;

    default:
      ierr = VecDot(d, p, &dMp);CHKERRQ(ierr);
      ierr = VecDot(p, p, &norm_p);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Compute the new direction and update the iteration.                   */
    /*************************************************************************/

    ierr = KSP_MatMult(ksp, Qmat, p, z);CHKERRQ(ierr);	/* z = Q * p         */
    ierr = VecDot(p, z, &kappa);CHKERRQ(ierr);		/* kappa = p^T Q p   */
    ++ksp->its;

    /*************************************************************************/
    /* Update the the Lanczos tridiagonal matrix.                            */
    /*************************************************************************/

    ++l_size;
    cg->offd[t_size] = PetscSqrtReal(beta) / fabs(alpha);
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
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: cg breakdown: kappa=%g\n", kappa);CHKERRQ(ierr);

      if (cg->radius && norm_p > 0.0) {
        /*********************************************************************/
        /* Follow direction to boundary.                                     */
        /*********************************************************************/

        step = (PetscSqrtReal(dMp*dMp+norm_p*(r2-norm_d))-dMp)/norm_p;
        cg->norm_d = cg->radius;

        ierr = VecAXPY(d, step, p);CHKERRQ(ierr);	/* d = d + step p    */

        /*********************************************************************/
        /* Update objective function.                                        */
        /*********************************************************************/

        cg->o_fcn += step * (0.5 * step * kappa - rz);
      }
      else if (cg->radius) {
        /*********************************************************************/
        /* The norm of the direction is zero; there is nothing to follow.    */
        /*********************************************************************/
      }
      break;
    }
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

  if (KSP_CONVERGED_CG_NEG_CURVE != ksp->reason) {
    /*************************************************************************/
    /* The method either converged to an interior point, hit the boundary of */
    /* the trust-region without encountering a direction of negative         */
    /* curvature or the iteration limit was reached.                         */
    /*************************************************************************/

    PetscFunctionReturn(0);
  }

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
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: lanczos breakdown: kappa=%g\n", kappa);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Update the direction and residual.                                    */
    /*************************************************************************/
    
    alpha = rz / kappa;
    cg->alpha[l_size] = alpha;

    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha Q p */
    ierr = KSP_PCApply(ksp, r, z);CHKERRQ(ierr);	/* z = inv(M) r      */

    /*************************************************************************/
    /* Check that the preconditioner appears positive semidefinite.          */
    /*************************************************************************/

    rzm1 = rz;
    ierr = VecDot(r, z, &rz);CHKERRQ(ierr);		/* rz = r^T z        */
    if (rz < 0.0) {
      /***********************************************************************/
      /* The preconditioner is indefinite.                                   */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: lanczos indefinite preconditioner: rz=%g\n", rz);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* As far as we know, the preconditioner is positive definite.  Compute  */
    /* the residual.  Do NOT check for convergence.                          */
    /*************************************************************************/

    cg->norm_r[l_size+1] = PetscSqrtReal(rz);			/* norm_r = |r|_M    */

    switch(ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = VecNorm(z, NORM_2, &norm_r);CHKERRQ(ierr);/* norm_r = |z|      */
      break;

    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(r, NORM_2, &norm_r);CHKERRQ(ierr);/* norm_r = |r|      */
      break;

    case KSP_NORM_NATURAL:
      norm_r = cg->norm_r[l_size+1];			/* norm_r = |r|_M    */
      break;

    default:
      norm_r = 0.0;
      break;
    }

    KSPLogResidualHistory(ksp, norm_r);
    ierr = KSPMonitor(ksp, ksp->its, norm_r);CHKERRQ(ierr);
    ksp->rnorm = norm_r;
  
    /*************************************************************************/
    /* Check for breakdown.                                                  */
    /*************************************************************************/

    beta = rz / rzm1;
    if (fabs(beta) <= 0.0) {
      /***********************************************************************/
      /* Conjugate gradients has broken down.                                */
      /***********************************************************************/

      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr = PetscInfo1(ksp, "KSPSolve_GLTR: breakdown: beta=%g\n", beta);CHKERRQ(ierr);
      break;
    }

    /*************************************************************************/
    /* Update p and the norms.                                               */
    /*************************************************************************/

    cg->beta[l_size] = beta;
    ierr = VecAYPX(p, beta, z);CHKERRQ(ierr);          /* p = z + beta p    */

    /*************************************************************************/
    /* Compute the new direction and update the iteration.                   */
    /*************************************************************************/

    ierr = KSP_MatMult(ksp, Qmat, p, z);CHKERRQ(ierr);	/* z = Q * p         */
    ierr = VecDot(p, z, &kappa);CHKERRQ(ierr);		/* kappa = p^T Q p   */
    ++ksp->its;

    /*************************************************************************/
    /* Update the Lanczos tridiagonal matrix.                                */
    /*************************************************************************/

    ++l_size;
    cg->offd[t_size] = PetscSqrtReal(beta) / fabs(alpha);
    cg->diag[t_size] = kappa / rz + beta / alpha;
    ++t_size;
  }

  /***************************************************************************/
  /* We have the Lanczos basis, solve the tridiagonal trust-region problem   */
  /* to obtain the Lanczos direction.  We know that the solution lies on     */
  /* the boundary of the trust region.  We start by checking that the        */
  /* workspace allocated is large enough.                                    */
  /***************************************************************************/
  /* Note that the current version only computes the solution by using the   */
  /* preconditioned direction.  Need to think about how to do the            */
  /* unpreconditioned direction calculation.                                 */
  /***************************************************************************/

  if (t_size > cg->alloced) {
    if (cg->alloced) {
      ierr = PetscFree(cg->rwork);CHKERRQ(ierr);
      ierr = PetscFree(cg->iwork);CHKERRQ(ierr);
      cg->alloced += cg->init_alloc;
    } else {
      cg->alloced = cg->init_alloc;
    }

    while (t_size > cg->alloced) {
      cg->alloced += cg->init_alloc;
    }
 
    cg->alloced = PetscMin(cg->alloced, t_size);
    ierr = PetscMalloc2(10*cg->alloced,PetscReal, &cg->rwork,5*cg->alloced,PetscBLASInt, &cg->iwork);CHKERRQ(ierr);
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

#if defined(PETSC_MISSING_LAPACK_STEBZ)
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"STEBZ - Lapack routine is unavailable.");
#else
  LAPACKstebz_("I", "E", &t_size, &vl, &vu, &il, &iu, &cg->eigen_tol,
               cg->diag, cg->offd + 1, &e_valus, &e_splts, e_valu, 
               e_iblk, e_splt, e_rwrk, e_iwrk, &info);

  if ((0 != info) || (1 != e_valus)) {
    /*************************************************************************/
    /* Calculation of the minimum eigenvalue failed.  Return the             */
    /* Steihaug-Toint direction.                                             */
    /*************************************************************************/

    ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute eigenvalue.\n");
    ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
    PetscFunctionReturn(0);
  }

  cg->e_min = e_valu[0];

  /***************************************************************************/
  /* Compute the initial value of lambda to make (T + lamba I) positive      */
  /* definite.                                                               */
  /***************************************************************************/

  pert = cg->init_pert;
  if (e_valu[0] < 0.0) {
    cg->lambda = pert - e_valu[0];
  }
#endif

  while(1) {
    for (i = 0; i < t_size; ++i) {
      t_diag[i] = cg->diag[i] + cg->lambda;
      t_offd[i] = cg->offd[i];
    }

#if defined(PETSC_MISSING_LAPACK_PTTRF)
    SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"PTTRF - Lapack routine is unavailable.");
#else
    LAPACKpttrf_(&t_size, t_diag, t_offd + 1, &info);

    if (0 == info) {
      break;
    }

    pert += pert;
    cg->lambda = cg->lambda * (1.0 + pert) + pert;
#endif
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

#if defined(PETSC_MISSING_LAPACK_PTTRS)
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"PTTRS - Lapack routine is unavailable.");
#else
  LAPACKpttrs_(&t_size, &nrhs, t_diag, t_offd + 1, t_soln, &nldb, &info);
#endif

  if (0 != info) {
    /*************************************************************************/
    /* Calculation of the initial step failed; return the Steihaug-Toint     */
    /* direction.                                                            */
    /*************************************************************************/

    ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute step.\n");
    ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
    PetscFunctionReturn(0);
  }

  norm_t = 0.;
  for (i = 0; i < t_size; ++i) {
    norm_t += t_soln[i] * t_soln[i];
  }
  norm_t = PetscSqrtReal(norm_t);

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

#if defined(PETSC_MISSING_LAPACK_STEIN)
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"STEIN - Lapack routine is unavailable.");
#else
      LAPACKstein_(&t_size, cg->diag, cg->offd + 1, &e_valus, e_valu,
		   e_iblk, e_splt, e_vect, &nldb, 
		   e_rwrk, e_iwrk, e_iwrk + t_size, &info);
#endif

      if (0 != info) {
	/*********************************************************************/
	/* Calculation of the minimum eigenvalue failed.  Return the         */
	/* Steihaug-Toint direction.                                         */
	/*********************************************************************/
	
	ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute eigenvector.\n");
	ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
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
      
      coef3 = PetscSqrtReal(coef2 * coef2 - coef1 * coef3);
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

#if defined(PETSC_MISSING_LAPACK_PTTRS)
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"PTTRS - Lapack routine is unavailable.");
#else      
      LAPACKpttrs_(&t_size, &nrhs, t_diag, t_offd + 1, e_rwrk, &nldb, &info);
#endif

      if (0 != info) {
	/*********************************************************************/
	/* Calculation of the step failed; return the Steihaug-Toint         */
	/* direction.                                                        */
	/*********************************************************************/

	ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute step.\n");
	ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
	PetscFunctionReturn(0);
      }

      /***********************************************************************/
      /* Modify lambda.                                                      */
      /***********************************************************************/

      norm_w = 0.;
      for (j = 0; j < t_size; ++j) {
	norm_w += t_soln[j] * e_rwrk[j];
      }
      
      cg->lambda += (norm_t - cg->radius)/cg->radius * (norm_t * norm_t) / norm_w;

      /***********************************************************************/
      /* Factor T + lambda I                                                 */
      /***********************************************************************/
      
      for (j = 0; j < t_size; ++j) {
	t_diag[j] = cg->diag[j] + cg->lambda;
	t_offd[j] = cg->offd[j];
      }

#if defined(PETSC_MISSING_LAPACK_PTTRF)
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"PTTRF - Lapack routine is unavailable.");
#else
      LAPACKpttrf_(&t_size, t_diag, t_offd + 1, &info);
#endif

      if (0 != info) {
	/*********************************************************************/
	/* Calculation of factorization failed; return the Steihaug-Toint    */
	/* direction.                                                        */
	/*********************************************************************/

	ierr = PetscInfo(ksp, "KSPSolve_GLTR: factorization failed.\n");
	ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
	PetscFunctionReturn(0);
      }

      /***********************************************************************/
      /* Compute the new step and its norm.                                  */
      /***********************************************************************/

      t_soln[0] = -cg->norm_r[0];
      for (j = 1; j < t_size; ++j) {
	t_soln[j] = 0.0;
      }

#if defined(PETSC_MISSING_LAPACK_PTTRS)
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"PTTRS - Lapack routine is unavailable.");
#else
      LAPACKpttrs_(&t_size, &nrhs, t_diag, t_offd + 1, t_soln, &nldb, &info);
#endif

      if (0 != info) {
	/*********************************************************************/
	/* Calculation of the step failed; return the Steihaug-Toint         */
	/* direction.                                                        */
	/*********************************************************************/
	
	ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to compute step.\n");
	ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
	PetscFunctionReturn(0);
      }

      norm_t = 0.;
      for (j = 0; j < t_size; ++j) {
	norm_t += t_soln[j] * t_soln[j];
      }
      norm_t = PetscSqrtReal(norm_t);
    }

    /*************************************************************************/
    /* Check for convergence.                                                */
    /*************************************************************************/

    if (fabs(norm_t - cg->radius) > cg->newton_tol * cg->radius) {
      /***********************************************************************/
      /* Newton method failed to converge in iteration limit.                */
      /***********************************************************************/

      ierr = PetscInfo(ksp, "KSPSolve_GLTR: failed to converge.\n");
      ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
      PetscFunctionReturn(0);
    }
  }

  /***************************************************************************/
  /* Recover the norm of the direction and objective function value.         */
  /***************************************************************************/

  cg->norm_d = norm_t;

  cg->o_fcn = t_soln[0]*(0.5*(cg->diag[0]*t_soln[0]+cg->offd[1]*t_soln[1])+cg->norm_r[0]);
  for (i = 1; i < t_size - 1; ++i) {
    cg->o_fcn += 0.5*t_soln[i]*(cg->offd[i]*t_soln[i-1]+cg->diag[i]*t_soln[i]+cg->offd[i+1]*t_soln[i+1]);
  }
  cg->o_fcn += 0.5*t_soln[i]*(cg->offd[i]*t_soln[i-1]+cg->diag[i]*t_soln[i]);

  /***************************************************************************/
  /* Recover the direction.                                                  */
  /***************************************************************************/

  sigma = -1;

  /***************************************************************************/
  /* Start conjugate gradient method from the beginning                      */
  /***************************************************************************/

  ierr = VecCopy(ksp->vec_rhs, r);CHKERRQ(ierr);	/* r = -grad         */
  ierr = KSP_PCApply(ksp, r, z);CHKERRQ(ierr);		/* z = inv(M) r      */

  /***************************************************************************/
  /* Accumulate Q * s                                                        */
  /***************************************************************************/

  ierr = VecCopy(z, d);CHKERRQ(ierr);
  ierr = VecScale(d, sigma * t_soln[0] / cg->norm_r[0]);CHKERRQ(ierr);
 
  /***************************************************************************/
  /* Compute the first direction.                                            */
  /***************************************************************************/

  ierr = VecCopy(z, p);CHKERRQ(ierr);			/* p = z             */
  ierr = KSP_MatMult(ksp, Qmat, p, z);CHKERRQ(ierr);	/* z = Q * p         */
  ++ksp->its;

  for (i = 0; i < l_size - 1; ++i) {
    /*************************************************************************/
    /* Update the residual and direction.                                    */
    /*************************************************************************/

    alpha = cg->alpha[i];
    if (alpha >= 0.0) {
      sigma = -sigma;
    }

    ierr = VecAXPY(r, -alpha, z);			/* r = r - alpha Q p */
    ierr = KSP_PCApply(ksp, r, z);CHKERRQ(ierr);	/* z = inv(M) r      */

    /*************************************************************************/
    /* Accumulate Q * s                                                      */
    /*************************************************************************/

    ierr = VecAXPY(d, sigma * t_soln[i+1] / cg->norm_r[i+1], z);

    /*************************************************************************/
    /* Update p.                                                             */
    /*************************************************************************/

    beta = cg->beta[i];
    ierr = VecAYPX(p, beta, z);CHKERRQ(ierr);          /* p = z + beta p    */
    ierr = KSP_MatMult(ksp, Qmat, p, z);CHKERRQ(ierr);	/* z = Q * p         */
    ++ksp->its;
  }

  /***************************************************************************/
  /* Update the residual and direction.                                      */
  /***************************************************************************/

  alpha = cg->alpha[i];
  if (alpha >= 0.0) {
    sigma = -sigma;
  }

  ierr = VecAXPY(r, -alpha, z);				/* r = r - alpha Q p */
  ierr = KSP_PCApply(ksp, r, z);CHKERRQ(ierr);		/* z = inv(M) r      */

  /***************************************************************************/
  /* Accumulate Q * s                                                        */
  /***************************************************************************/

  ierr = VecAXPY(d, sigma * t_soln[i+1] / cg->norm_r[i+1], z);

  /***************************************************************************/
  /* Set the termination reason.                                             */
  /***************************************************************************/

  ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_GLTR"
PetscErrorCode KSPSetUp_GLTR(KSP ksp)
{
  KSP_GLTR       *cg = (KSP_GLTR *)ksp->data;
  PetscInt       max_its;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /***************************************************************************/
  /* Determine the total maximum number of iterations.                       */
  /***************************************************************************/

  max_its = ksp->max_it + cg->max_lanczos_its + 1;

  /***************************************************************************/
  /* Set work vectors needed by conjugate gradient method and allocate       */
  /* workspace for Lanczos matrix.                                           */
  /***************************************************************************/

  ierr = KSPDefaultGetWork(ksp, 3);CHKERRQ(ierr);

  ierr = PetscMalloc5(max_its,PetscReal,&cg->diag,max_its,PetscReal,&cg->offd,max_its,PetscReal,&cg->alpha,max_its,PetscReal,&cg->beta,max_its,PetscReal,&cg->norm_r);CHKERRQ(ierr);
  ierr = PetscMemzero(cg->diag, max_its*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(cg->offd, max_its*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(cg->alpha, max_its*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(cg->beta, max_its*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(cg->norm_r, max_its*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp, 5*max_its*sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_GLTR"
PetscErrorCode KSPDestroy_GLTR(KSP ksp)
{
  KSP_GLTR       *cg = (KSP_GLTR *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /***************************************************************************/
  /* Free memory allocated for the data.                                     */
  /***************************************************************************/

  ierr = PetscFree5(cg->diag,cg->offd,cg->alpha,cg->beta,cg->norm_r);CHKERRQ(ierr);
  if (cg->alloced) {
    ierr = PetscFree2(cg->rwork,cg->iwork);CHKERRQ(ierr);
  }

  /***************************************************************************/
  /* Clear composed functions                                                */
  /***************************************************************************/

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRSetRadius_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRGetNormD_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRGetObjFcn_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRGetMinEig_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGLTRGetLambda_C","",PETSC_NULL);CHKERRQ(ierr);

  /***************************************************************************/
  /* Destroy KSP object.                                                     */
  /***************************************************************************/
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPGLTRSetRadius_GLTR"
PetscErrorCode  KSPGLTRSetRadius_GLTR(KSP ksp, PetscReal radius)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  cg->radius = radius;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetNormD_GLTR"
PetscErrorCode  KSPGLTRGetNormD_GLTR(KSP ksp, PetscReal *norm_d)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  *norm_d = cg->norm_d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetObjFcn_GLTR"
PetscErrorCode  KSPGLTRGetObjFcn_GLTR(KSP ksp, PetscReal *o_fcn)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  *o_fcn = cg->o_fcn;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetMinEig_GLTR"
PetscErrorCode  KSPGLTRGetMinEig_GLTR(KSP ksp, PetscReal *e_min)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  *e_min = cg->e_min;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGLTRGetLambda_GLTR"
PetscErrorCode  KSPGLTRGetLambda_GLTR(KSP ksp, PetscReal *lambda)
{
  KSP_GLTR *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  *lambda = cg->lambda;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_GLTR"
PetscErrorCode KSPSetFromOptions_GLTR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_GLTR       *cg = (KSP_GLTR *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP GLTR options");CHKERRQ(ierr);

  ierr = PetscOptionsReal("-ksp_gltr_radius", "Trust Region Radius", "KSPGLTRSetRadius", cg->radius, &cg->radius, PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-ksp_gltr_init_pert", "Initial perturbation", "", cg->init_pert, &cg->init_pert, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_gltr_eigen_tol", "Eigenvalue tolerance", "", cg->eigen_tol, &cg->eigen_tol, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_gltr_newton_tol", "Newton tolerance", "", cg->newton_tol, &cg->newton_tol, PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-ksp_gltr_max_lanczos_its", "Maximum Lanczos Iters", "", cg->max_lanczos_its, &cg->max_lanczos_its, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_gltr_max_newton_its", "Maximum Newton Iters", "", cg->max_newton_its, &cg->max_newton_its, PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsEList("-ksp_gltr_dtype", "Norm used for direction", "", DType_Table, GLTR_DIRECTION_TYPES, DType_Table[cg->dtype], &cg->dtype, PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPGLTR -   Code to run conjugate gradient method subject to a constraint
         on the solution norm. This is used in Trust Region methods for
         nonlinear equations, SNESTR

   Options Database Keys:
.      -ksp_gltr_radius <r> - Trust Region Radius

   Notes: This is rarely used directly

  Use preconditioned conjugate gradient to compute
  an approximate minimizer of the quadratic function

            q(s) = g^T * s + .5 * s^T * H * s

   subject to the trust region constraint

            || s || <= delta,

   where

     delta is the trust region radius,
     g is the gradient vector,
     H is the Hessian approximation,
     M is the positive definite preconditioner matrix.

   KSPConvergedReason may be
$  KSP_CONVERGED_CG_NEG_CURVE if convergence is reached along a negative curvature direction,
$  KSP_CONVERGED_CG_CONSTRAINED if convergence is reached along a constrained step,
$  other KSP converged/diverged reasons

  Notes:
  The preconditioner supplied should be symmetric and positive definite.

   Level: developer

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPGLTRSetRadius(), KSPGLTRGetNormD(), KSPGLTRGetObjFcn(), KSPGLTRGetMinEig(), KSPGLTRGetLambda()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_GLTR"
PetscErrorCode  KSPCreate_GLTR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_GLTR       *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp, KSP_GLTR, &cg);CHKERRQ(ierr);
  cg->radius = 0.0;
  cg->dtype = GLTR_UNPRECONDITIONED_DIRECTION;

  cg->init_pert = 1.0e-8;
  cg->eigen_tol = 1.0e-10;
  cg->newton_tol = 1.0e-6;

  cg->alloced = 0;
  cg->init_alloc = 1024;

  cg->max_lanczos_its = 20;
  cg->max_newton_its  = 10;

  ksp->data = (void *) cg;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,1);CHKERRQ(ierr);

  /***************************************************************************/
  /* Sets the functions that are associated with this data structure         */
  /* (in C++ this is the same as defining virtual functions).                */
  /***************************************************************************/

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
                                            KSPGLTRSetRadius_GLTR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                           "KSPGLTRGetNormD_C",
                                           "KSPGLTRGetNormD_GLTR",
                                            KSPGLTRGetNormD_GLTR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                           "KSPGLTRGetObjFcn_C",
                                           "KSPGLTRGetObjFcn_GLTR",
                                            KSPGLTRGetObjFcn_GLTR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                           "KSPGLTRGetMinEig_C",
                                           "KSPGLTRGetMinEig_GLTR",
                                            KSPGLTRGetMinEig_GLTR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,
                                           "KSPGLTRGetLambda_C",
                                           "KSPGLTRGetLambda_GLTR",
                                            KSPGLTRGetLambda_GLTR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
