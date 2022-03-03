
/*
            This implements Richardson Iteration.
*/
#include <../src/ksp/ksp/impls/rich/richardsonimpl.h>     /*I "petscksp.h" I*/

PetscErrorCode KSPSetUp_Richardson(KSP ksp)
{
  KSP_Richardson *richardsonP = (KSP_Richardson*)ksp->data;

  PetscFunctionBegin;
  if (richardsonP->selfscale) {
    CHKERRQ(KSPSetWorkVecs(ksp,4));
  } else {
    CHKERRQ(KSPSetWorkVecs(ksp,2));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPSolve_Richardson(KSP ksp)
{
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
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ksp->its = 0;

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));
  x    = ksp->vec_sol;
  b    = ksp->vec_rhs;
  CHKERRQ(VecGetSize(x,&xs));
  CHKERRQ(VecGetSize(ksp->work[0],&ws));
  if (xs != ws) {
    if (richardsonP->selfscale) {
      CHKERRQ(KSPSetWorkVecs(ksp,4));
    } else {
      CHKERRQ(KSPSetWorkVecs(ksp,2));
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
  CHKERRQ(PCApplyRichardsonExists(ksp->pc,&exists));
  CHKERRQ(MatGetNullSpace(Pmat,&nullsp));
  if (exists && maxit > 0 && richardsonP->scale == 1.0 && (ksp->converged == KSPConvergedDefault || ksp->converged == KSPConvergedSkip) && !ksp->numbermonitors && !ksp->transpose_solve && !nullsp) {
    PCRichardsonConvergedReason reason;
    CHKERRQ(PCApplyRichardson(ksp->pc,b,x,r,ksp->rtol,ksp->abstol,ksp->divtol,maxit,ksp->guess_zero,&ksp->its,&reason));
    ksp->reason = (KSPConvergedReason)reason;
    PetscFunctionReturn(0);
  } else {
    CHKERRQ(PetscInfo(ksp,"KSPSolve_Richardson: Warning, skipping optimized PCApplyRichardson()\n"));
  }

  if (!ksp->guess_zero) {                          /*   r <- b - A x     */
    CHKERRQ(KSP_MatMult(ksp,Amat,x,r));
    CHKERRQ(VecAYPX(r,-1.0,b));
  } else {
    CHKERRQ(VecCopy(b,r));
  }

  ksp->its = 0;
  if (richardsonP->selfscale) {
    CHKERRQ(KSP_PCApply(ksp,r,z));         /*   z <- B r          */
    for (i=0; i<maxit; i++) {

      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        CHKERRQ(VecNorm(r,NORM_2,&rnorm)); /*   rnorm <- r'*r     */
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        CHKERRQ(VecNorm(z,NORM_2,&rnorm)); /*   rnorm <- z'*z     */
      } else rnorm = 0.0;

      KSPCheckNorm(ksp,rnorm);
      ksp->rnorm = rnorm;
      CHKERRQ(KSPMonitor(ksp,i,rnorm));
      CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
      CHKERRQ((*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP));
      if (ksp->reason) break;
      CHKERRQ(KSP_PCApplyBAorAB(ksp,z,y,w)); /* y = BAz = BABr */
      CHKERRQ(VecDotNorm2(z,y,&rdot,&abr));   /*   rdot = (Br)^T(BABR); abr = (BABr)^T (BABr) */
      scale = rdot/abr;
      CHKERRQ(PetscInfo(ksp,"Self-scale factor %g\n",(double)PetscRealPart(scale)));
      CHKERRQ(VecAXPY(x,scale,z));   /*   x  <- x + scale z */
      CHKERRQ(VecAXPY(r,-scale,w));  /*  r <- r - scale*Az */
      CHKERRQ(VecAXPY(z,-scale,y));  /*  z <- z - scale*y */
      ksp->its++;
    }
  } else {
    for (i=0; i<maxit; i++) {

      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        CHKERRQ(VecNorm(r,NORM_2,&rnorm)); /*   rnorm <- r'*r     */
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        CHKERRQ(KSP_PCApply(ksp,r,z));    /*   z <- B r          */
        CHKERRQ(VecNorm(z,NORM_2,&rnorm)); /*   rnorm <- z'*z     */
      } else rnorm = 0.0;
      ksp->rnorm = rnorm;
      CHKERRQ(KSPMonitor(ksp,i,rnorm));
      CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
      CHKERRQ((*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP));
      if (ksp->reason) break;
      if (ksp->normtype != KSP_NORM_PRECONDITIONED) {
        CHKERRQ(KSP_PCApply(ksp,r,z));    /*   z <- B r          */
      }

      CHKERRQ(VecAXPY(x,richardsonP->scale,z));    /*   x  <- x + scale z */
      ksp->its++;

      if (i+1 < maxit || ksp->normtype != KSP_NORM_NONE) {
        CHKERRQ(KSP_MatMult(ksp,Amat,x,r));      /*   r  <- b - Ax      */
        CHKERRQ(VecAYPX(r,-1.0,b));
      }
    }
  }
  if (!ksp->reason) {
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      CHKERRQ(VecNorm(r,NORM_2,&rnorm));     /*   rnorm <- r'*r     */
    } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      CHKERRQ(KSP_PCApply(ksp,r,z));   /*   z <- B r          */
      CHKERRQ(VecNorm(z,NORM_2,&rnorm));     /*   rnorm <- z'*z     */
    } else rnorm = 0.0;

    KSPCheckNorm(ksp,rnorm);
    ksp->rnorm = rnorm;
    CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
    CHKERRQ(KSPMonitor(ksp,i,rnorm));
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
        CHKERRQ((*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP));
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
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    if (richardsonP->selfscale) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using self-scale best computed damping factor\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  damping factor=%g\n",(double)richardsonP->scale));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_Richardson(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_Richardson *rich = (KSP_Richardson*)ksp->data;
  PetscReal      tmp;
  PetscBool      flg,flg2;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"KSP Richardson Options"));
  CHKERRQ(PetscOptionsReal("-ksp_richardson_scale","damping factor","KSPRichardsonSetScale",rich->scale,&tmp,&flg));
  if (flg) CHKERRQ(KSPRichardsonSetScale(ksp,tmp));
  CHKERRQ(PetscOptionsBool("-ksp_richardson_self_scale","dynamically determine optimal damping factor","KSPRichardsonSetSelfScale",rich->selfscale,&flg2,&flg));
  if (flg) CHKERRQ(KSPRichardsonSetSelfScale(ksp,flg2));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_Richardson(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPRichardsonSetScale_C",NULL));
  CHKERRQ(KSPDestroyDefault(ksp));
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
  PetscFunctionBegin;
  if (ksp->normtype == KSP_NORM_NONE) {
    CHKERRQ(KSPBuildResidualDefault(ksp,t,v,V));
  } else {
    CHKERRQ(VecCopy(ksp->work[0],v));
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
. * - L. F. Richardson, "The Approximate Arithmetical Solution by Finite Differences of Physical Problems Involving
   Differential Equations, with an Application to the Stresses in a Masonry Dam",
  Philosophical Transactions of the Royal Society of London. Series A,
  Containing Papers of a Mathematical or Physical Character, Vol. 210, 1911 (1911).

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPRichardsonSetScale(), KSPPREONLY

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_Richardson(KSP ksp)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&richardsonP));
  ksp->data = (void*)richardsonP;

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_Richardson;
  ksp->ops->solve          = KSPSolve_Richardson;
  ksp->ops->destroy        = KSPDestroy_Richardson;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_Richardson;
  ksp->ops->view           = KSPView_Richardson;
  ksp->ops->setfromoptions = KSPSetFromOptions_Richardson;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPRichardsonSetScale_C",KSPRichardsonSetScale_Richardson));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPRichardsonSetSelfScale_C",KSPRichardsonSetSelfScale_Richardson));

  richardsonP->scale = 1.0;
  PetscFunctionReturn(0);
}
