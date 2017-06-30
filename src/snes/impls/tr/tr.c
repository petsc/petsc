
#include <../src/snes/impls/tr/trimpl.h>                /*I   "petscsnes.h"   I*/

typedef struct {
  void *ctx;
  SNES snes;
} SNES_TR_KSPConverged_Ctx;

/*
   This convergence test determines if the two norm of the
   solution lies outside the trust region, if so it halts.
*/
PetscErrorCode SNES_TR_KSPConverged_Private(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx = (SNES_TR_KSPConverged_Ctx*)cctx;
  SNES                     snes = ctx->snes;
  SNES_NEWTONTR            *neP = (SNES_NEWTONTR*)snes->data;
  Vec                      x;
  PetscReal                nrm;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = KSPConvergedDefault(ksp,n,rnorm,reason,ctx->ctx);CHKERRQ(ierr);
  if (*reason) {
    ierr = PetscInfo2(snes,"default convergence test KSP iterations=%D, rnorm=%g\n",n,(double)rnorm);CHKERRQ(ierr);
  }
  /* Determine norm of solution */
  ierr = KSPBuildSolution(ksp,0,&x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm >= neP->delta) {
    ierr    = PetscInfo2(snes,"Ending linear iteration early, delta=%g, length=%g\n",(double)neP->delta,(double)nrm);CHKERRQ(ierr);
    *reason = KSP_CONVERGED_STEP_LENGTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNES_TR_KSPConverged_Destroy(void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx = (SNES_TR_KSPConverged_Ctx*)cctx;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = KSPConvergedDefaultDestroy(ctx->ctx);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/*
   SNES_TR_Converged_Private -test convergence JUST for
   the trust region tolerance.

*/
static PetscErrorCode SNES_TR_Converged_Private(SNES snes,PetscInt it,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  SNES_NEWTONTR  *neP = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *reason = SNES_CONVERGED_ITERATING;
  if (neP->delta < xnorm * snes->deltatol) {
    ierr    = PetscInfo3(snes,"Converged due to trust region param %g<%g*%g\n",(double)neP->delta,(double)xnorm,(double)snes->deltatol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_TR_DELTA;
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr    = PetscInfo1(snes,"Exceeded maximum number of function evaluations: %D\n",snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }
  PetscFunctionReturn(0);
}


/*
   SNESSolve_NEWTONTR - Implements Newton's Method with a very simple trust
   region approach for solving systems of nonlinear equations.


*/
static PetscErrorCode SNESSolve_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR       *neP = (SNES_NEWTONTR*)snes->data;
  Vec                 X,F,Y,G,Ytmp;
  PetscErrorCode      ierr;
  PetscInt            maxits,i,lits;
  PetscReal           rho,fnorm,gnorm,gpnorm,xnorm=0,delta,nrm,ynorm,norm1;
  PetscScalar         cnorm;
  KSP                 ksp;
  SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
  PetscBool           conv   = PETSC_FALSE,breakout = PETSC_FALSE;

  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* work vectors */
  G      = snes->work[1];
  Ytmp   = snes->work[2];

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);          /* F(X) */
  } else snes->vec_func_init_set = PETSC_FALSE;

  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);             /* fnorm <- || F || */
  SNESCheckFunctionNorm(snes,fnorm);
  ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);             /* fnorm <- || F || */
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  delta      = xnorm ? neP->delta0*xnorm : neP->delta0;
  neP->delta = delta;
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  /* Set the stopping criteria to use the More' trick. */
  ierr = PetscOptionsGetBool(((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_tr_ksp_regular_convergence_test",&conv,NULL);CHKERRQ(ierr);
  if (!conv) {
    SNES_TR_KSPConverged_Ctx *ctx;
    ierr      = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr      = PetscNew(&ctx);CHKERRQ(ierr);
    ctx->snes = snes;
    ierr      = KSPConvergedDefaultCreate(&ctx->ctx);CHKERRQ(ierr);
    ierr      = KSPSetConvergenceTest(ksp,SNES_TR_KSPConverged_Private,ctx,SNES_TR_KSPConverged_Destroy);CHKERRQ(ierr);
    ierr      = PetscInfo(snes,"Using Krylov convergence test SNES_TR_KSPConverged_Private\n");CHKERRQ(ierr);
  }

  for (i=0; i<maxits; i++) {

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /* Solve J Y = F, where J is Jacobian matrix */
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    ierr = KSPSolve(snes->ksp,F,Ytmp);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);

    snes->linear_its += lits;

    ierr  = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    ierr  = VecNorm(Ytmp,NORM_2,&nrm);CHKERRQ(ierr);
    norm1 = nrm;
    while (1) {
      ierr = VecCopy(Ytmp,Y);CHKERRQ(ierr);
      nrm  = norm1;

      /* Scale Y if need be and predict new value of F norm */
      if (nrm >= delta) {
        nrm    = delta/nrm;
        gpnorm = (1.0 - nrm)*fnorm;
        cnorm  = nrm;
        ierr   = PetscInfo1(snes,"Scaling direction by %g\n",(double)nrm);CHKERRQ(ierr);
        ierr   = VecScale(Y,cnorm);CHKERRQ(ierr);
        nrm    = gpnorm;
        ynorm  = delta;
      } else {
        gpnorm = 0.0;
        ierr   = PetscInfo(snes,"Direction is in Trust Region\n");CHKERRQ(ierr);
        ynorm  = nrm;
      }
      ierr = VecAYPX(Y,-1.0,X);CHKERRQ(ierr);            /* Y <- X - Y */
      ierr = VecCopy(X,snes->vec_sol_update);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes,Y,G);CHKERRQ(ierr); /*  F(X) */
      ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);      /* gnorm <- || g || */
      if (fnorm == gpnorm) rho = 0.0;
      else rho = (fnorm*fnorm - gnorm*gnorm)/(fnorm*fnorm - gpnorm*gpnorm);

      /* Update size of trust region */
      if      (rho < neP->mu)  delta *= neP->delta1;
      else if (rho < neP->eta) delta *= neP->delta2;
      else                     delta *= neP->delta3;
      ierr = PetscInfo3(snes,"fnorm=%g, gnorm=%g, ynorm=%g\n",(double)fnorm,(double)gnorm,(double)ynorm);CHKERRQ(ierr);
      ierr = PetscInfo3(snes,"gpred=%g, rho=%g, delta=%g\n",(double)gpnorm,(double)rho,(double)delta);CHKERRQ(ierr);

      neP->delta = delta;
      if (rho > neP->sigma) break;
      ierr = PetscInfo(snes,"Trying again in smaller region\n");CHKERRQ(ierr);
      /* check to see if progress is hopeless */
      neP->itflag = PETSC_FALSE;
      ierr        = SNES_TR_Converged_Private(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (!reason) { ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr); }
      if (reason) {
        /* We're not progressing, so return with the current iterate */
        ierr     = SNESMonitor(snes,i+1,fnorm);CHKERRQ(ierr);
        breakout = PETSC_TRUE;
        break;
      }
      snes->numFailures++;
    }
    if (!breakout) {
      /* Update function and solution vectors */
      fnorm = gnorm;
      ierr  = VecCopy(G,F);CHKERRQ(ierr);
      ierr  = VecCopy(Y,X);CHKERRQ(ierr);
      /* Monitor convergence */
      ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
      snes->iter = i+1;
      snes->norm = fnorm;
      ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
      ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
      ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
      /* Test for convergence, xnorm = || X || */
      neP->itflag = PETSC_TRUE;
      if (snes->ops->converged != SNESConvergedSkip) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
      ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (reason) break;
    } else break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!reason) reason = SNES_DIVERGED_MAX_IT;
  }
  ierr         = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->reason = reason;
  ierr         = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
static PetscErrorCode SNESSetUp_NEWTONTR(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetWorkVecs(snes,3);CHKERRQ(ierr);
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESReset_NEWTONTR(SNES snes)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_NEWTONTR(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NEWTONTR(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode SNESSetFromOptions_NEWTONTR(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NEWTONTR  *ctx = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SNES trust region options for nonlinear equations");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trtol","Trust region tolerance","SNESSetTrustRegionTolerance",snes->deltatol,&snes->deltatol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_mu","mu","None",ctx->mu,&ctx->mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_eta","eta","None",ctx->eta,&ctx->eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_sigma","sigma","None",ctx->sigma,&ctx->sigma,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_delta0","delta0","None",ctx->delta0,&ctx->delta0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_delta1","delta1","None",ctx->delta1,&ctx->delta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_delta2","delta2","None",ctx->delta2,&ctx->delta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_delta3","delta3","None",ctx->delta3,&ctx->delta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_NEWTONTR(SNES snes,PetscViewer viewer)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  mu=%g, eta=%g, sigma=%g\n",(double)tr->mu,(double)tr->eta,(double)tr->sigma);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  delta0=%g, delta1=%g, delta2=%g, delta3=%g\n",(double)tr->delta0,(double)tr->delta1,(double)tr->delta2,(double)tr->delta3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
/*MC
      SNESNEWTONTR - Newton based nonlinear solver that uses a trust region

   Options Database:
+    -snes_trtol <tol> Trust region tolerance
.    -snes_tr_mu <mu>
.    -snes_tr_eta <eta>
.    -snes_tr_sigma <sigma>
.    -snes_tr_delta0 <delta0> -  initial size of the trust region is delta0*norm2(x)
.    -snes_tr_delta1 <delta1>
.    -snes_tr_delta2 <delta2>
-    -snes_tr_delta3 <delta3>

   The basic algorithm is taken from "The Minpack Project", by More',
   Sorensen, Garbow, Hillstrom, pages 88-111 of "Sources and Development
   of Mathematical Software", Wayne Cowell, editor.

   Level: intermediate

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESSetTrustRegionTolerance()

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR  *neP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONTR;
  snes->ops->solve          = SNESSolve_NEWTONTR;
  snes->ops->destroy        = SNESDestroy_NEWTONTR;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONTR;
  snes->ops->view           = SNESView_NEWTONTR;
  snes->ops->reset          = SNESReset_NEWTONTR;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  ierr        = PetscNewLog(snes,&neP);CHKERRQ(ierr);
  snes->data  = (void*)neP;
  neP->mu     = 0.25;
  neP->eta    = 0.75;
  neP->delta  = 0.0;
  neP->delta0 = 0.2;
  neP->delta1 = 0.3;
  neP->delta2 = 0.75;
  neP->delta3 = 2.0;
  neP->sigma  = 0.0001;
  neP->itflag = PETSC_FALSE;
  neP->rnorm0 = 0.0;
  neP->ttol   = 0.0;
  PetscFunctionReturn(0);
}

