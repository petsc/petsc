
#include <../src/ksp/ksp/impls/qcg/qcgimpl.h>     /*I "petscksp.h" I*/

static PetscErrorCode KSPQCGQuadraticRoots(Vec,Vec,PetscReal,PetscReal*,PetscReal*);

/*@
    KSPQCGSetTrustRegionRadius - Sets the radius of the trust region.

    Logically Collective on ksp

    Input Parameters:
+   ksp   - the iterative context
-   delta - the trust region radius (Infinity is the default)

    Options Database Key:
.   -ksp_qcg_trustregionradius <delta>

    Level: advanced

@*/
PetscErrorCode  KSPQCGSetTrustRegionRadius(KSP ksp,PetscReal delta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (delta < 0.0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be non-negative");
  ierr = PetscTryMethod(ksp,"KSPQCGSetTrustRegionRadius_C",(KSP,PetscReal),(ksp,delta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    KSPQCGGetTrialStepNorm - Gets the norm of a trial step vector.  The WCG step may be
    constrained, so this is not necessarily the length of the ultimate step taken in QCG.

    Not Collective

    Input Parameter:
.   ksp - the iterative context

    Output Parameter:
.   tsnorm - the norm

    Level: advanced
@*/
PetscErrorCode  KSPQCGGetTrialStepNorm(KSP ksp,PetscReal *tsnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPQCGGetTrialStepNorm_C",(KSP,PetscReal*),(ksp,tsnorm));CHKERRQ(ierr);
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
@*/
PetscErrorCode  KSPQCGGetQuadratic(KSP ksp,PetscReal *quadratic)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPQCGGetQuadratic_C",(KSP,PetscReal*),(ksp,quadratic));CHKERRQ(ierr);
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

  KSP_QCG        *pcgP = (KSP_QCG*)ksp->data;
  Mat            Amat,Pmat;
  Vec            W,WA,WA2,R,P,ASP,BS,X,B;
  PetscScalar    scal,beta,rntrn,step;
  PetscReal      q1,q2,xnorm,step1,step2,rnrm = 0.0,btx,xtax;
  PetscReal      ptasp,rtr,wtasp,bstp;
  PetscReal      dzero = 0.0,bsnrm = 0.0;
  PetscErrorCode ierr;
  PetscInt       i,maxit;
  PC             pc = ksp->pc;
  PCSide         side;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  if (ksp->transpose_solve) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Currently does not support transpose solve");

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

  if (pcgP->delta <= dzero) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Input error: delta <= 0");
  ierr = KSPGetPCSide(ksp,&side);CHKERRQ(ierr);
  if (side != PC_SYMMETRIC) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Requires symmetric preconditioner!");

  /* Initialize variables */
  ierr = VecSet(W,0.0);CHKERRQ(ierr);  /* W = 0 */
  ierr = VecSet(X,0.0);CHKERRQ(ierr);  /* X = 0 */
  ierr = PCGetOperators(pc,&Amat,&Pmat);CHKERRQ(ierr);

  /* Compute:  BS = D^{-1} B */
  ierr = PCApplySymmetricLeft(pc,B,BS);CHKERRQ(ierr);

  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(BS,NORM_2,&bsnrm);CHKERRQ(ierr);
    KSPCheckNorm(ksp,bsnrm);
  }
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = bsnrm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,bsnrm);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,bsnrm);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,bsnrm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Compute the initial scaled direction and scaled residual */
  ierr = VecCopy(BS,R);CHKERRQ(ierr);
  ierr = VecScale(R,-1.0);CHKERRQ(ierr);
  ierr = VecCopy(R,P);CHKERRQ(ierr);
  ierr = VecDotRealPart(R,R,&rtr);CHKERRQ(ierr);

  for (i=0; i<=maxit; i++) {
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

    /* Compute:  asp = D^{-T}*A*D^{-1}*p  */
    ierr = PCApplySymmetricRight(pc,P,WA);CHKERRQ(ierr);
    ierr = KSP_MatMult(ksp,Amat,WA,WA2);CHKERRQ(ierr);
    ierr = PCApplySymmetricLeft(pc,WA2,ASP);CHKERRQ(ierr);

    /* Check for negative curvature */
    ierr = VecDotRealPart(P,ASP,&ptasp);CHKERRQ(ierr);
    if (ptasp <= dzero) {

      /* Scaled negative curvature direction:  Compute a step so that
        ||w + step*p|| = delta and QS(w + step*p) is least */

      if (!i) {
        ierr = VecCopy(P,X);CHKERRQ(ierr);
        ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);
        KSPCheckNorm(ksp,xnorm);
        scal = pcgP->delta / xnorm;
        ierr = VecScale(X,scal);CHKERRQ(ierr);
      } else {
        /* Compute roots of quadratic */
        ierr = KSPQCGQuadraticRoots(W,P,pcgP->delta,&step1,&step2);CHKERRQ(ierr);
        ierr = VecDotRealPart(W,ASP,&wtasp);CHKERRQ(ierr);
        ierr = VecDotRealPart(BS,P,&bstp);CHKERRQ(ierr);
        ierr = VecCopy(W,X);CHKERRQ(ierr);
        q1   = step1*(bstp + wtasp + .5*step1*ptasp);
        q2   = step2*(bstp + wtasp + .5*step2*ptasp);
        if (q1 <= q2) {
          ierr = VecAXPY(X,step1,P);CHKERRQ(ierr);
        } else {
          ierr = VecAXPY(X,step2,P);CHKERRQ(ierr);
        }
      }
      pcgP->ltsnrm = pcgP->delta;                       /* convergence in direction of */
      ksp->reason  = KSP_CONVERGED_CG_NEG_CURVE;  /* negative curvature */
      if (!i) {
        ierr = PetscInfo1(ksp,"negative curvature: delta=%g\n",(double)pcgP->delta);CHKERRQ(ierr);
      } else {
        ierr = PetscInfo3(ksp,"negative curvature: step1=%g, step2=%g, delta=%g\n",(double)step1,(double)step2,(double)pcgP->delta);CHKERRQ(ierr);
      }

    } else {
      /* Compute step along p */
      step = rtr/ptasp;
      ierr = VecCopy(W,X);CHKERRQ(ierr);        /*  x = w  */
      ierr = VecAXPY(X,step,P);CHKERRQ(ierr);   /*  x <- step*p + x  */
      ierr = VecNorm(X,NORM_2,&pcgP->ltsnrm);CHKERRQ(ierr);
      KSPCheckNorm(ksp,pcgP->ltsnrm);

      if (pcgP->ltsnrm > pcgP->delta) {
        /* Since the trial iterate is outside the trust region,
            evaluate a constrained step along p so that
                    ||w + step*p|| = delta
          The positive step is always better in this case. */
        if (!i) {
          scal = pcgP->delta / pcgP->ltsnrm;
          ierr = VecScale(X,scal);CHKERRQ(ierr);
        } else {
          /* Compute roots of quadratic */
          ierr = KSPQCGQuadraticRoots(W,P,pcgP->delta,&step1,&step2);CHKERRQ(ierr);
          ierr = VecCopy(W,X);CHKERRQ(ierr);
          ierr = VecAXPY(X,step1,P);CHKERRQ(ierr);  /*  x <- step1*p + x  */
        }
        pcgP->ltsnrm = pcgP->delta;
        ksp->reason  = KSP_CONVERGED_CG_CONSTRAINED; /* convergence along constrained step */
        if (!i) {
          ierr = PetscInfo1(ksp,"constrained step: delta=%g\n",(double)pcgP->delta);CHKERRQ(ierr);
        } else {
          ierr = PetscInfo3(ksp,"constrained step: step1=%g, step2=%g, delta=%g\n",(double)step1,(double)step2,(double)pcgP->delta);CHKERRQ(ierr);
        }

      } else {
        /* Evaluate the current step */
        ierr = VecCopy(X,W);CHKERRQ(ierr);  /* update interior iterate */
        ierr = VecAXPY(R,-step,ASP);CHKERRQ(ierr); /* r <- -step*asp + r */
        if (ksp->normtype != KSP_NORM_NONE) {
          ierr = VecNorm(R,NORM_2,&rnrm);CHKERRQ(ierr);
          KSPCheckNorm(ksp,rnrm);
        }
        ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
        ksp->rnorm = rnrm;
        ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
        ierr = KSPLogResidualHistory(ksp,rnrm);CHKERRQ(ierr);
        ierr = KSPMonitor(ksp,i+1,rnrm);CHKERRQ(ierr);
        ierr = (*ksp->converged)(ksp,i+1,rnrm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
        if (ksp->reason) {                 /* convergence for */
          ierr = PetscInfo3(ksp,"truncated step: step=%g, rnrm=%g, delta=%g\n",(double)PetscRealPart(step),(double)rnrm,(double)pcgP->delta);CHKERRQ(ierr);
        }
      }
    }
    if (ksp->reason) break;  /* Convergence has been attained */
    else {                   /* Compute a new AS-orthogonal direction */
      ierr = VecDot(R,R,&rntrn);CHKERRQ(ierr);
      beta = rntrn/rtr;
      ierr = VecAYPX(P,beta,R);CHKERRQ(ierr);  /*  p <- r + beta*p  */
      rtr  = PetscRealPart(rntrn);
    }
  }
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;

  /* Unscale x */
  ierr = VecCopy(X,WA2);CHKERRQ(ierr);
  ierr = PCApplySymmetricRight(pc,WA2,X);CHKERRQ(ierr);

  ierr = KSP_MatMult(ksp,Amat,X,WA);CHKERRQ(ierr);
  ierr = VecDotRealPart(B,X,&btx);CHKERRQ(ierr);
  ierr = VecDotRealPart(X,WA,&xtax);CHKERRQ(ierr);

  pcgP->quadratic = btx + .5*xtax;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUp_QCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Get work vectors from user code */
  ierr = KSPSetWorkVecs(ksp,7);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_QCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPQCGGetQuadratic_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPQCGGetTrialStepNorm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPQCGSetTrustRegionRadius_C",NULL);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPQCGSetTrustRegionRadius_QCG(KSP ksp,PetscReal delta)
{
  KSP_QCG *cgP = (KSP_QCG*)ksp->data;

  PetscFunctionBegin;
  cgP->delta = delta;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPQCGGetTrialStepNorm_QCG(KSP ksp,PetscReal *ltsnrm)
{
  KSP_QCG *cgP = (KSP_QCG*)ksp->data;

  PetscFunctionBegin;
  *ltsnrm = cgP->ltsnrm;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPQCGGetQuadratic_QCG(KSP ksp,PetscReal *quadratic)
{
  KSP_QCG *cgP = (KSP_QCG*)ksp->data;

  PetscFunctionBegin;
  *quadratic = cgP->quadratic;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_QCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  PetscReal      delta;
  KSP_QCG        *cgP = (KSP_QCG*)ksp->data;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP QCG Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_qcg_trustregionradius","Trust Region Radius","KSPQCGSetTrustRegionRadius",cgP->delta,&delta,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPQCGSetTrustRegionRadius(ksp,delta);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPQCG -   Code to run conjugate gradient method subject to a constraint
         on the solution norm. This is used in Trust Region methods for nonlinear equations, SNESNEWTONTR

   Options Database Keys:
.      -ksp_qcg_trustregionradius <r> - Trust Region Radius

   Notes:
    This is rarely used directly

   Level: developer

  Notes:
    Use preconditioned conjugate gradient to compute
      an approximate minimizer of the quadratic function

            q(s) = g^T * s + .5 * s^T * H * s

   subject to the Euclidean norm trust region constraint

            || D * s || <= delta,

   where

     delta is the trust region radius,
     g is the gradient vector, and
     H is Hessian matrix,
     D is a scaling matrix.

   KSPConvergedReason may be
$  KSP_CONVERGED_CG_NEG_CURVE if convergence is reached along a negative curvature direction,
$  KSP_CONVERGED_CG_CONSTRAINED if convergence is reached along a constrained step,
$  other KSP converged/diverged reasons

  Notes:
  Currently we allow symmetric preconditioning with the following scaling matrices:
      PCNONE:   D = Identity matrix
      PCJACOBI: D = diag [d_1, d_2, ...., d_n], where d_i = sqrt(H[i,i])
      PCICC:    D = L^T, implemented with forward and backward solves.
                Here L is an incomplete Cholesky factor of H.

  References:
.  1. - Trond Steihaug, The Conjugate Gradient Method and Trust Regions in Large Scale Optimization,
   SIAM Journal on Numerical Analysis, Vol. 20, No. 3 (Jun., 1983).

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPQCGSetTrustRegionRadius()
           KSPQCGGetTrialStepNorm(), KSPQCGGetQuadratic()
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_QCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_QCG        *cgP;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_SYMMETRIC,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_SYMMETRIC,1);CHKERRQ(ierr);
  ierr = PetscNewLog(ksp,&cgP);CHKERRQ(ierr);

  ksp->data                = (void*)cgP;
  ksp->ops->setup          = KSPSetUp_QCG;
  ksp->ops->setfromoptions = KSPSetFromOptions_QCG;
  ksp->ops->solve          = KSPSolve_QCG;
  ksp->ops->destroy        = KSPDestroy_QCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->view           = NULL;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPQCGGetQuadratic_C",KSPQCGGetQuadratic_QCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPQCGGetTrialStepNorm_C",KSPQCGGetTrialStepNorm_QCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPQCGSetTrustRegionRadius_C",KSPQCGSetTrustRegionRadius_QCG);CHKERRQ(ierr);
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
static PetscErrorCode KSPQCGQuadraticRoots(Vec s,Vec p,PetscReal delta,PetscReal *step1,PetscReal *step2)
{
  PetscReal      dsq,ptp,pts,rad,sts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDotRealPart(p,s,&pts);CHKERRQ(ierr);
  ierr = VecDotRealPart(p,p,&ptp);CHKERRQ(ierr);
  ierr = VecDotRealPart(s,s,&sts);CHKERRQ(ierr);
  dsq  = delta*delta;
  rad  = PetscSqrtReal((pts*pts) - ptp*(sts - dsq));
  if (pts > 0.0) {
    *step2 = -(pts + rad)/ptp;
    *step1 = (sts - dsq)/(ptp * *step2);
  } else {
    *step1 = -(pts - rad)/ptp;
    *step2 = (sts - dsq)/(ptp * *step1);
  }
  PetscFunctionReturn(0);
}
