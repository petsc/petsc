
/*
            This implements Richardson Iteration.
*/
#include <../src/ksp/ksp/impls/rich/richardsonimpl.h>     /*I "petscksp.h" I*/

PetscErrorCode KSPSetUp_Richardson(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;

  PetscFunctionBegin;
  if (richardsonP->selfscale) {
    ierr = KSPSetWorkVecs(ksp,4);CHKERRQ(ierr);
  } else {
    ierr = KSPSetWorkVecs(ksp,2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPSolve_Richardson(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,maxit;
  PetscReal      rnorm = 0.0,abr;
  PetscScalar    scale,rdot;
  Vec            x,b,r,z,w = NULL,y = NULL;
  PetscInt       xs, ws;
  Mat            Amat,Pmat;
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  PetscBool      exists,diagonalscale;
  MatNullSpace   nullsp;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ksp->its = 0;

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  x    = ksp->vec_sol;
  b    = ksp->vec_rhs;
  ierr = VecGetSize(x,&xs);CHKERRQ(ierr);
  ierr = VecGetSize(ksp->work[0],&ws);CHKERRQ(ierr);
  if (xs != ws) {
    if (richardsonP->selfscale) {
      ierr = KSPSetWorkVecs(ksp,4);CHKERRQ(ierr);
    } else {
      ierr = KSPSetWorkVecs(ksp,2);CHKERRQ(ierr);
    }
  }
  r = ksp->work[0];
  z = ksp->work[1];
  if (richardsonP->selfscale) {
    w = ksp->work[2];
    y = ksp->work[3];
  }
  maxit = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  ierr = PCApplyRichardsonExists(ksp->pc,&exists);CHKERRQ(ierr);
  ierr = MatGetNullSpace(Pmat,&nullsp);CHKERRQ(ierr);
  if (exists && maxit > 0 && richardsonP->scale == 1.0 && (ksp->converged == KSPConvergedDefault || ksp->converged == KSPConvergedSkip) && !ksp->numbermonitors && !ksp->transpose_solve && !nullsp) {
    PCRichardsonConvergedReason reason;
    ierr        = PCApplyRichardson(ksp->pc,b,x,r,ksp->rtol,ksp->abstol,ksp->divtol,maxit,ksp->guess_zero,&ksp->its,&reason);CHKERRQ(ierr);
    ksp->reason = (KSPConvergedReason)reason;
    PetscFunctionReturn(0);
  } else {
    ierr = PetscInfo(ksp,"KSPSolve_Richardson: Warning, skipping optimized PCApplyRichardson()\n");CHKERRQ(ierr);
  }

  if (!ksp->guess_zero) {                          /*   r <- b - A x     */
    ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);
  }

  ksp->its = 0;
  if (richardsonP->selfscale) {
    ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);         /*   z <- B r          */
    for (i=0; i<maxit; i++) {

      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- z'*z     */
      } else rnorm = 0.0;

      KSPCheckNorm(ksp,rnorm);
      ksp->rnorm = rnorm;
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
      ierr  = KSP_PCApplyBAorAB(ksp,z,y,w);CHKERRQ(ierr); /* y = BAz = BABr */
      ierr  = VecDotNorm2(z,y,&rdot,&abr);CHKERRQ(ierr);   /*   rdot = (Br)^T(BABR); abr = (BABr)^T (BABr) */
      scale = rdot/abr;
      ierr  = PetscInfo(ksp,"Self-scale factor %g\n",(double)PetscRealPart(scale));CHKERRQ(ierr);
      ierr  = VecAXPY(x,scale,z);CHKERRQ(ierr);   /*   x  <- x + scale z */
      ierr  = VecAXPY(r,-scale,w);CHKERRQ(ierr);  /*  r <- r - scale*Az */
      ierr  = VecAXPY(z,-scale,y);CHKERRQ(ierr);  /*  z <- z - scale*y */
      ksp->its++;
    }
  } else {
    for (i=0; i<maxit; i++) {

      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- r'*r     */
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);    /*   z <- B r          */
        ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr); /*   rnorm <- z'*z     */
      } else rnorm = 0.0;
      ksp->rnorm = rnorm;
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
      if (ksp->normtype != KSP_NORM_PRECONDITIONED) {
        ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);    /*   z <- B r          */
      }

      ierr = VecAXPY(x,richardsonP->scale,z);CHKERRQ(ierr);    /*   x  <- x + scale z */
      ksp->its++;

      if (i+1 < maxit || ksp->normtype != KSP_NORM_NONE) {
        ierr = KSP_MatMult(ksp,Amat,x,r);CHKERRQ(ierr);      /*   r  <- b - Ax      */
        ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
      }
    }
  }
  if (!ksp->reason) {
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- r'*r     */
    } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);   /*   z <- B r          */
      ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr);     /*   rnorm <- z'*z     */
    } else rnorm = 0.0;

    KSPCheckNorm(ksp,rnorm);
    ksp->rnorm = rnorm;
    ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
        ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
        if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      } else {
        ksp->reason = KSP_CONVERGED_ITS;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_Richardson(KSP ksp,PetscViewer viewer)
{
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (richardsonP->selfscale) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using self-scale best computed damping factor\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  damping factor=%g\n",(double)richardsonP->scale);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_Richardson(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_Richardson *rich = (KSP_Richardson*)ksp->data;
  PetscErrorCode ierr;
  PetscReal      tmp;
  PetscBool      flg,flg2;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP Richardson Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_richardson_scale","damping factor","KSPRichardsonSetScale",rich->scale,&tmp,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPRichardsonSetScale(ksp,tmp);CHKERRQ(ierr); }
  ierr = PetscOptionsBool("-ksp_richardson_self_scale","dynamically determine optimal damping factor","KSPRichardsonSetSelfScale",rich->selfscale,&flg2,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPRichardsonSetSelfScale(ksp,flg2);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_Richardson(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPRichardsonSetScale_C",NULL);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPRichardsonSetScale_Richardson(KSP ksp,PetscReal scale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP        = (KSP_Richardson*)ksp->data;
  richardsonP->scale = scale;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPRichardsonSetSelfScale_Richardson(KSP ksp,PetscBool selfscale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP            = (KSP_Richardson*)ksp->data;
  richardsonP->selfscale = selfscale;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPBuildResidual_Richardson(KSP ksp,Vec t,Vec v,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->normtype == KSP_NORM_NONE) {
    ierr = KSPBuildResidualDefault(ksp,t,v,V);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(ksp->work[0],v);CHKERRQ(ierr);
    *V   = v;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPRICHARDSON - The preconditioned Richardson iterative method

   Options Database Keys:
.   -ksp_richardson_scale - damping factor on the correction (defaults to 1.0)

   Level: beginner

   Notes:
    x^{n+1} = x^{n} + scale*B(b - A x^{n})

          Here B is the application of the preconditioner

          This method often (usually) will not converge unless scale is very small.

   Notes:
    For some preconditioners, currently SOR, the convergence test is skipped to improve speed,
    thus it always iterates the maximum number of iterations you've selected. When -ksp_monitor
    (or any other monitor) is turned on, the norm is computed at each iteration and so the convergence test is run unless
    you specifically call KSPSetNormType(ksp,KSP_NORM_NONE);

         For some preconditioners, currently PCMG and PCHYPRE with BoomerAMG if -ksp_monitor (and also
    any other monitor) is not turned on then the convergence test is done by the preconditioner itself and
    so the solver may run more or fewer iterations then if -ksp_monitor is selected.

    Supports only left preconditioning

    If using direct solvers such as PCLU and PCCHOLESKY one generally uses KSPPREONLY which uses exactly one iteration

$    -ksp_type richardson -pc_type jacobi gives one classically Jacobi preconditioning

  References:
.  1. - L. F. Richardson, "The Approximate Arithmetical Solution by Finite Differences of Physical Problems Involving
   Differential Equations, with an Application to the Stresses in a Masonry Dam",
  Philosophical Transactions of the Royal Society of London. Series A,
  Containing Papers of a Mathematical or Physical Character, Vol. 210, 1911 (1911).

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPRichardsonSetScale(), KSPPREONLY

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_Richardson(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,&richardsonP);CHKERRQ(ierr);
  ksp->data = (void*)richardsonP;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_Richardson;
  ksp->ops->solve          = KSPSolve_Richardson;
  ksp->ops->destroy        = KSPDestroy_Richardson;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_Richardson;
  ksp->ops->view           = KSPView_Richardson;
  ksp->ops->setfromoptions = KSPSetFromOptions_Richardson;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPRichardsonSetScale_C",KSPRichardsonSetScale_Richardson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPRichardsonSetSelfScale_C",KSPRichardsonSetSelfScale_Richardson);CHKERRQ(ierr);

  richardsonP->scale = 1.0;
  PetscFunctionReturn(0);
}
