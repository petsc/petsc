
#include <../src/ksp/ksp/impls/qcg/qcgimpl.h> /*I "petscksp.h" I*/

static PetscErrorCode KSPQCGQuadraticRoots(Vec, Vec, PetscReal, PetscReal *, PetscReal *);

/*@
    KSPQCGSetTrustRegionRadius - Sets the radius of the trust region for `KSPQCG`

    Logically Collective on ksp

    Input Parameters:
+   ksp   - the iterative context
-   delta - the trust region radius (Infinity is the default)

    Options Database Key:
.   -ksp_qcg_trustregionradius <delta> - trust region radius

    Level: advanced

@*/
PetscErrorCode KSPQCGSetTrustRegionRadius(KSP ksp, PetscReal delta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscCheck(delta >= 0.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Tolerance must be non-negative");
  PetscTryMethod(ksp, "KSPQCGSetTrustRegionRadius_C", (KSP, PetscReal), (ksp, delta));
  PetscFunctionReturn(0);
}

/*@
    KSPQCGGetTrialStepNorm - Gets the norm of a trial step vector in `KSPQCG`.  The WCG step may be
    constrained, so this is not necessarily the length of the ultimate step taken in `KSPQCG`.

    Not Collective

    Input Parameter:
.   ksp - the iterative context

    Output Parameter:
.   tsnorm - the norm

    Level: advanced
@*/
PetscErrorCode KSPQCGGetTrialStepNorm(KSP ksp, PetscReal *tsnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPQCGGetTrialStepNorm_C", (KSP, PetscReal *), (ksp, tsnorm));
  PetscFunctionReturn(0);
}

/*@
    KSPQCGGetQuadratic - Gets the value of the quadratic function, evaluated at the new iterate:

       q(s) = g^T * s + 0.5 * s^T * H * s

    which satisfies the Euclidian Norm trust region constraint

       || D * s || <= delta,

    where

     delta is the trust region radius,
     g is the gradient vector, and
     H is Hessian matrix,
     D is a scaling matrix.

    Collective on ksp

    Input Parameter:
.   ksp - the iterative context

    Output Parameter:
.   quadratic - the quadratic function evaluated at the new iterate

    Level: advanced

.seealso: [](chapter_ksp), `KSPQCG`
@*/
PetscErrorCode KSPQCGGetQuadratic(KSP ksp, PetscReal *quadratic)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPQCGGetQuadratic_C", (KSP, PetscReal *), (ksp, quadratic));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_QCG(KSP ksp)
{
  /*
   Correpondence with documentation above:
      B = g = gradient,
      X = s = step
   Note:  This is not coded correctly for complex arithmetic!
 */

  KSP_QCG    *pcgP = (KSP_QCG *)ksp->data;
  Mat         Amat, Pmat;
  Vec         W, WA, WA2, R, P, ASP, BS, X, B;
  PetscScalar scal, beta, rntrn, step;
  PetscReal   q1, q2, xnorm, step1, step2, rnrm = 0.0, btx, xtax;
  PetscReal   ptasp, rtr, wtasp, bstp;
  PetscReal   dzero = 0.0, bsnrm = 0.0;
  PetscInt    i, maxit;
  PC          pc = ksp->pc;
  PetscBool   diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  PetscCheck(!ksp->transpose_solve, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Currently does not support transpose solve");

  ksp->its = 0;
  maxit    = ksp->max_it;
  WA       = ksp->work[0];
  R        = ksp->work[1];
  P        = ksp->work[2];
  ASP      = ksp->work[3];
  BS       = ksp->work[4];
  W        = ksp->work[5];
  WA2      = ksp->work[6];
  X        = ksp->vec_sol;
  B        = ksp->vec_rhs;

  PetscCheck(pcgP->delta > dzero, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Input error: delta <= 0");

  /* Initialize variables */
  PetscCall(VecSet(W, 0.0)); /* W = 0 */
  PetscCall(VecSet(X, 0.0)); /* X = 0 */
  PetscCall(PCGetOperators(pc, &Amat, &Pmat));

  /* Compute:  BS = D^{-1} B */
  PetscCall(PCApplySymmetricLeft(pc, B, BS));

  if (ksp->normtype != KSP_NORM_NONE) {
    PetscCall(VecNorm(BS, NORM_2, &bsnrm));
    KSPCheckNorm(ksp, bsnrm);
  }
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = bsnrm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPLogResidualHistory(ksp, bsnrm));
  PetscCall(KSPMonitor(ksp, 0, bsnrm));
  PetscCall((*ksp->converged)(ksp, 0, bsnrm, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  /* Compute the initial scaled direction and scaled residual */
  PetscCall(VecCopy(BS, R));
  PetscCall(VecScale(R, -1.0));
  PetscCall(VecCopy(R, P));
  PetscCall(VecDotRealPart(R, R, &rtr));

  for (i = 0; i <= maxit; i++) {
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

    /* Compute:  asp = D^{-T}*A*D^{-1}*p  */
    PetscCall(PCApplySymmetricRight(pc, P, WA));
    PetscCall(KSP_MatMult(ksp, Amat, WA, WA2));
    PetscCall(PCApplySymmetricLeft(pc, WA2, ASP));

    /* Check for negative curvature */
    PetscCall(VecDotRealPart(P, ASP, &ptasp));
    if (ptasp <= dzero) {
      /* Scaled negative curvature direction:  Compute a step so that
        ||w + step*p|| = delta and QS(w + step*p) is least */

      if (!i) {
        PetscCall(VecCopy(P, X));
        PetscCall(VecNorm(X, NORM_2, &xnorm));
        KSPCheckNorm(ksp, xnorm);
        scal = pcgP->delta / xnorm;
        PetscCall(VecScale(X, scal));
      } else {
        /* Compute roots of quadratic */
        PetscCall(KSPQCGQuadraticRoots(W, P, pcgP->delta, &step1, &step2));
        PetscCall(VecDotRealPart(W, ASP, &wtasp));
        PetscCall(VecDotRealPart(BS, P, &bstp));
        PetscCall(VecCopy(W, X));
        q1 = step1 * (bstp + wtasp + .5 * step1 * ptasp);
        q2 = step2 * (bstp + wtasp + .5 * step2 * ptasp);
        if (q1 <= q2) {
          PetscCall(VecAXPY(X, step1, P));
        } else {
          PetscCall(VecAXPY(X, step2, P));
        }
      }
      pcgP->ltsnrm = pcgP->delta;                /* convergence in direction of */
      ksp->reason  = KSP_CONVERGED_CG_NEG_CURVE; /* negative curvature */
      if (!i) {
        PetscCall(PetscInfo(ksp, "negative curvature: delta=%g\n", (double)pcgP->delta));
      } else {
        PetscCall(PetscInfo(ksp, "negative curvature: step1=%g, step2=%g, delta=%g\n", (double)step1, (double)step2, (double)pcgP->delta));
      }

    } else {
      /* Compute step along p */
      step = rtr / ptasp;
      PetscCall(VecCopy(W, X));       /*  x = w  */
      PetscCall(VecAXPY(X, step, P)); /*  x <- step*p + x  */
      PetscCall(VecNorm(X, NORM_2, &pcgP->ltsnrm));
      KSPCheckNorm(ksp, pcgP->ltsnrm);

      if (pcgP->ltsnrm > pcgP->delta) {
        /* Since the trial iterate is outside the trust region,
            evaluate a constrained step along p so that
                    ||w + step*p|| = delta
          The positive step is always better in this case. */
        if (!i) {
          scal = pcgP->delta / pcgP->ltsnrm;
          PetscCall(VecScale(X, scal));
        } else {
          /* Compute roots of quadratic */
          PetscCall(KSPQCGQuadraticRoots(W, P, pcgP->delta, &step1, &step2));
          PetscCall(VecCopy(W, X));
          PetscCall(VecAXPY(X, step1, P)); /*  x <- step1*p + x  */
        }
        pcgP->ltsnrm = pcgP->delta;
        ksp->reason  = KSP_CONVERGED_CG_CONSTRAINED; /* convergence along constrained step */
        if (!i) {
          PetscCall(PetscInfo(ksp, "constrained step: delta=%g\n", (double)pcgP->delta));
        } else {
          PetscCall(PetscInfo(ksp, "constrained step: step1=%g, step2=%g, delta=%g\n", (double)step1, (double)step2, (double)pcgP->delta));
        }

      } else {
        /* Evaluate the current step */
        PetscCall(VecCopy(X, W));          /* update interior iterate */
        PetscCall(VecAXPY(R, -step, ASP)); /* r <- -step*asp + r */
        if (ksp->normtype != KSP_NORM_NONE) {
          PetscCall(VecNorm(R, NORM_2, &rnrm));
          KSPCheckNorm(ksp, rnrm);
        }
        PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
        ksp->rnorm = rnrm;
        PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
        PetscCall(KSPLogResidualHistory(ksp, rnrm));
        PetscCall(KSPMonitor(ksp, i + 1, rnrm));
        PetscCall((*ksp->converged)(ksp, i + 1, rnrm, &ksp->reason, ksp->cnvP));
        if (ksp->reason) { /* convergence for */
          PetscCall(PetscInfo(ksp, "truncated step: step=%g, rnrm=%g, delta=%g\n", (double)PetscRealPart(step), (double)rnrm, (double)pcgP->delta));
        }
      }
    }
    if (ksp->reason) break; /* Convergence has been attained */
    else { /* Compute a new AS-orthogonal direction */ PetscCall(VecDot(R, R, &rntrn));
      beta = rntrn / rtr;
      PetscCall(VecAYPX(P, beta, R)); /*  p <- r + beta*p  */
      rtr = PetscRealPart(rntrn);
    }
  }
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;

  /* Unscale x */
  PetscCall(VecCopy(X, WA2));
  PetscCall(PCApplySymmetricRight(pc, WA2, X));

  PetscCall(KSP_MatMult(ksp, Amat, X, WA));
  PetscCall(VecDotRealPart(B, X, &btx));
  PetscCall(VecDotRealPart(X, WA, &xtax));

  pcgP->quadratic = btx + .5 * xtax;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUp_QCG(KSP ksp)
{
  PetscFunctionBegin;
  /* Get work vectors from user code */
  PetscCall(KSPSetWorkVecs(ksp, 7));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_QCG(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPQCGGetQuadratic_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPQCGGetTrialStepNorm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPQCGSetTrustRegionRadius_C", NULL));
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPQCGSetTrustRegionRadius_QCG(KSP ksp, PetscReal delta)
{
  KSP_QCG *cgP = (KSP_QCG *)ksp->data;

  PetscFunctionBegin;
  cgP->delta = delta;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPQCGGetTrialStepNorm_QCG(KSP ksp, PetscReal *ltsnrm)
{
  KSP_QCG *cgP = (KSP_QCG *)ksp->data;

  PetscFunctionBegin;
  *ltsnrm = cgP->ltsnrm;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPQCGGetQuadratic_QCG(KSP ksp, PetscReal *quadratic)
{
  KSP_QCG *cgP = (KSP_QCG *)ksp->data;

  PetscFunctionBegin;
  *quadratic = cgP->quadratic;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_QCG(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  PetscReal delta;
  KSP_QCG  *cgP = (KSP_QCG *)ksp->data;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP QCG Options");
  PetscCall(PetscOptionsReal("-ksp_qcg_trustregionradius", "Trust Region Radius", "KSPQCGSetTrustRegionRadius", cgP->delta, &delta, &flg));
  if (flg) PetscCall(KSPQCGSetTrustRegionRadius(ksp, delta));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*MC
     KSPQCG -   Code to run conjugate gradient method subject to a constraint on the solution norm.

   Options Database Key:
.      -ksp_qcg_trustregionradius <r> - Trust Region Radius

   Level: developer

   Notes:
    This is rarely used directly, ir is used in Trust Region methods for nonlinear equations, `SNESNEWTONTR`

    Uses preconditioned conjugate gradient to compute
      an approximate minimizer of the quadratic function

            q(s) = g^T * s + .5 * s^T * H * s

   subject to the Euclidean norm trust region constraint

            || D * s || <= delta,

   where

     delta is the trust region radius,
     g is the gradient vector, and
     H is Hessian matrix,
     D is a scaling matrix.

   `KSPConvergedReason` may be
.vb
   KSP_CONVERGED_CG_NEG_CURVE if convergence is reached along a negative curvature direction,
   KSP_CONVERGED_CG_CONSTRAINED if convergence is reached along a constrained step,
.ve
   and other `KSP` converged/diverged reasons

  Notes:
  Currently we allow symmetric preconditioning with the following scaling matrices:
.vb
      `PCNONE`:   D = Identity matrix
      `PCJACOBI`: D = diag [d_1, d_2, ...., d_n], where d_i = sqrt(H[i,i])
      `PCICC`:    D = L^T, implemented with forward and backward solves. Here L is an incomplete Cholesky factor of H.
.ve

  References:
. * - Trond Steihaug, The Conjugate Gradient Method and Trust Regions in Large Scale Optimization,
   SIAM Journal on Numerical Analysis, Vol. 20, No. 3 (Jun., 1983).

.seealso: [](chapter_ksp), 'KSPNASH`, `KSPGLTR`, `KSPSTCG`, `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPQCGSetTrustRegionRadius()`
          `KSPQCGGetTrialStepNorm()`, `KSPQCGGetQuadratic()`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_QCG(KSP ksp)
{
  KSP_QCG *cgP;

  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_SYMMETRIC, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_SYMMETRIC, 1));
  PetscCall(PetscNew(&cgP));

  ksp->data                = (void *)cgP;
  ksp->ops->setup          = KSPSetUp_QCG;
  ksp->ops->setfromoptions = KSPSetFromOptions_QCG;
  ksp->ops->solve          = KSPSolve_QCG;
  ksp->ops->destroy        = KSPDestroy_QCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->view           = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPQCGGetQuadratic_C", KSPQCGGetQuadratic_QCG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPQCGGetTrialStepNorm_C", KSPQCGGetTrialStepNorm_QCG));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPQCGSetTrustRegionRadius_C", KSPQCGSetTrustRegionRadius_QCG));
  cgP->delta = PETSC_MAX_REAL; /* default trust region radius is infinite */
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*
  KSPQCGQuadraticRoots - Computes the roots of the quadratic,
         ||s + step*p|| - delta = 0
   such that step1 >= 0 >= step2.
   where
      delta:
        On entry delta must contain scalar delta.
        On exit delta is unchanged.
      step1:
        On entry step1 need not be specified.
        On exit step1 contains the non-negative root.
      step2:
        On entry step2 need not be specified.
        On exit step2 contains the non-positive root.
   C code is translated from the Fortran version of the MINPACK-2 Project,
   Argonne National Laboratory, Brett M. Averick and Richard G. Carter.
*/
static PetscErrorCode KSPQCGQuadraticRoots(Vec s, Vec p, PetscReal delta, PetscReal *step1, PetscReal *step2)
{
  PetscReal dsq, ptp, pts, rad, sts;

  PetscFunctionBegin;
  PetscCall(VecDotRealPart(p, s, &pts));
  PetscCall(VecDotRealPart(p, p, &ptp));
  PetscCall(VecDotRealPart(s, s, &sts));
  dsq = delta * delta;
  rad = PetscSqrtReal((pts * pts) - ptp * (sts - dsq));
  if (pts > 0.0) {
    *step2 = -(pts + rad) / ptp;
    *step1 = (sts - dsq) / (ptp * *step2);
  } else {
    *step1 = -(pts - rad) / ptp;
    *step2 = (sts - dsq) / (ptp * *step1);
  }
  PetscFunctionReturn(0);
}
