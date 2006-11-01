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
  MatStructure   pflag;
  Mat            Qmat, Mmat;
  Vec            r, z, p, w, d;
  PC             pc;
  KSP_STCG        *cg;
  PetscReal      zero = 0.0, negone = -1.0;
  PetscReal      norm_r, norm_d, norm_dp1, norm_p, dMp;
  PetscReal      alpha, beta, kappa, rz, rzm1;
  PetscReal      radius;
  PetscInt       i, maxit;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    crz, ckappa;
#endif
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale); CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",ksp->type_name);

  cg       = (KSP_STCG *)ksp->data;
  radius   = cg->radius;
  maxit    = ksp->max_it;
  r        = ksp->work[0];
  z        = ksp->work[1];
  p        = ksp->work[2];
  w        = ksp->work[3];
  d        = ksp->vec_sol;
  pc       = ksp->pc;

  ksp->its = 0;
  if (radius <= zero) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Input error: radius <= 0");

  /* Initialize variables */
  ierr = PCGetOperators(pc, &Qmat, &Mmat, &pflag); CHKERRQ(ierr);

  ierr = VecSet(d, zero); CHKERRQ(ierr);                 /* d = 0        */
  ierr = VecCopy(ksp->vec_rhs, r); CHKERRQ(ierr);        /* r = rhs      */
  ierr = VecScale(r, negone); CHKERRQ(ierr);             /* r = grad     */
  ierr = VecNorm(r, NORM_2, &norm_r); CHKERRQ(ierr);     /* norm_r = |r| */

  KSPLogResidualHistory(ksp, norm_r);
  KSPMonitor(ksp, 0, norm_r);
  ksp->rnorm = norm_r;

  ierr = (*ksp->converged)(ksp, 0, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Compute the initial vectors */
  ierr = PCApply(pc, r, z); CHKERRQ(ierr);               /* z = M_{-1} r */
  ierr = VecCopy(z, p); CHKERRQ(ierr);                   /* p = -z       */
  ierr = VecScale(p, negone); CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(r, z, &crz);CHKERRQ(ierr); rz = PetscRealPart(crz);
#else
  ierr = VecDot(r, z, &rz);CHKERRQ(ierr);                /* rz = r^T z   */
#endif

  if (rz <= zero) {
    ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
    ierr = PetscInfo1(ksp, "KSPSolve_STCG: indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);

    /* Return gradient step */
    ierr = VecCopy(r, z); CHKERRQ(ierr);                   /* No precond   */
    ierr = VecCopy(z, p); CHKERRQ(ierr);                   /* p = -z       */
    ierr = VecScale(p, negone); CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = VecDot(r, z, &crz);CHKERRQ(ierr); rz = PetscRealPart(crz);
#else
    ierr = VecDot(r, z, &rz);CHKERRQ(ierr);                /* rz = r^T z   */
#endif

    alpha = sqrt(rz*radius) / rz;
    ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);            /* d = d + alpha p */    PetscFunctionReturn(0);
  }

  dMp    = 0;
  norm_p = rz;
  norm_d = 0;

  /* Begin iterating */
  for (i=0; i<=maxit; i++) {
    ksp->its++;

    ierr = MatMult(Qmat, p, w); CHKERRQ(ierr);            /* w = Q * p   */
#if defined(PETSC_USE_COMPLEX)
    ierr = VecDot(p, w, &ckappa);CHKERRQ(ierr); kappa = PetscRealPart(ckappa);
#else
    ierr = VecDot(p, w, &kappa); CHKERRQ(ierr);           /* kappa = p^T w */
#endif

    if (kappa <= zero) {
      /* Direction of negative curvature, calculate intersection and sol */

      alpha = (sqrt(dMp*dMp+norm_p*(radius-norm_d))-dMp)/norm_p;
      ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);        /* d = d + alpha p */

      ksp->reason  = KSP_CONVERGED_STCG_NEG_CURVE;  /* negative curvature */
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: negative curvature: radius=%g\n", cg->radius); CHKERRQ(ierr);
      break;
    }

    alpha = rz / kappa;

    norm_dp1 = norm_d + 2.0*alpha*dMp + alpha*alpha*norm_p;
    if (norm_dp1 >= radius) {
      alpha = (sqrt(dMp*dMp+norm_p*(radius-norm_d))-dMp)/norm_p;
      ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);        /* d = d + alpha p */

      ksp->reason  = KSP_CONVERGED_STCG_CONSTRAINED;     /* step */
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: constrained step: radius=%g\n",cg->radius); CHKERRQ(ierr);
      break;
    }

    ierr = VecAXPY(d, alpha, p); CHKERRQ(ierr);         /* d = d + alpha p */
    ierr = VecAXPY(r, alpha, w);                        /* r = r + alpha w */

    ierr = VecNorm(r, NORM_2, &norm_r); CHKERRQ(ierr);
    ksp->rnorm = norm_r;
    KSPLogResidualHistory(ksp, norm_r);
    KSPMonitor(ksp, i+1, norm_r);
    ierr = (*ksp->converged)(ksp, i+1, norm_r, &ksp->reason, ksp->cnvP); CHKERRQ(ierr);
    if (ksp->reason) {                 /* convergence */
      ierr = PetscInfo2(ksp,"KSPSolve_STCG: truncated step: rnorm=%g, radius=%g\n",norm_r,cg->radius); CHKERRQ(ierr);
      break;
    }

    ierr = PCApply(pc, r, z); CHKERRQ(ierr);

    rzm1 = rz;
#if defined(PETSC_USE_COMPLEX)
    ierr = VecDot(r, z, &crz);CHKERRQ(ierr); rz = PetscRealPart(crz);
#else
    ierr = VecDot(r, z, &rz); CHKERRQ(ierr);
#endif

    if (rz <= zero) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr = PetscInfo1(ksp, "KSPSolve_STCG: indefinite preconditioner: rz=%g\n", rz); CHKERRQ(ierr);
      break;
    }

    beta = rz / rzm1;

    VecAXPBY(p, negone, beta, z);                    /* p = beta p - z */
    dMp = beta*dMp + alpha*norm_p;
    norm_p = rz + beta*beta*norm_p;
    norm_d = norm_dp1;
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
  ierr = KSPDefaultGetWork(ksp, 4); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_STCG"
PetscErrorCode KSPDestroy_STCG(KSP ksp)
{
  KSP_STCG        *cgP = (KSP_STCG *)ksp->data;
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
           KSPSTCGGetQuadratic()
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
