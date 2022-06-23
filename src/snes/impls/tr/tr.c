#include <../src/snes/impls/tr/trimpl.h>                /*I   "petscsnes.h"   I*/

typedef struct {
  SNES           snes;
  /*  Information on the regular SNES convergence test; which may have been user provided */
  PetscErrorCode (*convtest)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
  PetscErrorCode (*convdestroy)(void*);
  void           *convctx;
} SNES_TR_KSPConverged_Ctx;

static PetscErrorCode SNESTR_KSPConverged_Private(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx = (SNES_TR_KSPConverged_Ctx*)cctx;
  SNES                     snes = ctx->snes;
  SNES_NEWTONTR            *neP = (SNES_NEWTONTR*)snes->data;
  Vec                      x;
  PetscReal                nrm;

  PetscFunctionBegin;
  PetscCall((*ctx->convtest)(ksp,n,rnorm,reason,ctx->convctx));
  if (*reason) {
    PetscCall(PetscInfo(snes,"Default or user provided convergence test KSP iterations=%" PetscInt_FMT ", rnorm=%g\n",n,(double)rnorm));
  }
  /* Determine norm of solution */
  PetscCall(KSPBuildSolution(ksp,NULL,&x));
  PetscCall(VecNorm(x,NORM_2,&nrm));
  if (nrm >= neP->delta) {
    PetscCall(PetscInfo(snes,"Ending linear iteration early, delta=%g, length=%g\n",(double)neP->delta,(double)nrm));
    *reason = KSP_CONVERGED_STEP_LENGTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTR_KSPConverged_Destroy(void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx = (SNES_TR_KSPConverged_Ctx*)cctx;

  PetscFunctionBegin;
  PetscCall((*ctx->convdestroy)(ctx->convctx));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/*
   SNESTR_Converged_Private -test convergence JUST for
   the trust region tolerance.

*/
static PetscErrorCode SNESTR_Converged_Private(SNES snes,PetscInt it,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  SNES_NEWTONTR  *neP = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  *reason = SNES_CONVERGED_ITERATING;
  if (neP->delta < xnorm * snes->deltatol) {
    PetscCall(PetscInfo(snes,"Converged due to trust region param %g<%g*%g\n",(double)neP->delta,(double)xnorm,(double)snes->deltatol));
    *reason = SNES_DIVERGED_TR_DELTA;
  } else if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    PetscCall(PetscInfo(snes,"Exceeded maximum number of function evaluations: %" PetscInt_FMT "\n",snes->max_funcs));
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRSetPreCheck - Sets a user function that is called before the search step has been determined.
       Allows the user a chance to change or override the decision of the line search routine.

   Logically Collective on snes

   Input Parameters:
+  snes - the nonlinear solver object
.  func - [optional] function evaluation routine, see SNESNewtonTRPreCheck()  for the calling sequence
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Note: This function is called BEFORE the function evaluation within the SNESNEWTONTR solver.

.seealso: `SNESNewtonTRPreCheck()`, `SNESNewtonTRGetPreCheck()`, `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRGetPostCheck()`
@*/
PetscErrorCode  SNESNewtonTRSetPreCheck(SNES snes, PetscErrorCode (*func)(SNES,Vec,Vec,PetscBool*,void*),void *ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) tr->precheck    = func;
  if (ctx)  tr->precheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRGetPreCheck - Gets the pre-check function

   Not collective

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  func - [optional] function evaluation routine, see for the calling sequence SNESNewtonTRPreCheck()
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: `SNESNewtonTRSetPreCheck()`, `SNESNewtonTRPreCheck()`
@*/
PetscErrorCode  SNESNewtonTRGetPreCheck(SNES snes, PetscErrorCode (**func)(SNES,Vec,Vec,PetscBool*,void*),void **ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) *func = tr->precheck;
  if (ctx)  *ctx  = tr->precheckctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRSetPostCheck - Sets a user function that is called after the search step has been determined but before the next
       function evaluation. Allows the user a chance to change or override the decision of the line search routine

   Logically Collective on snes

   Input Parameters:
+  snes - the nonlinear solver object
.  func - [optional] function evaluation routine, see SNESNewtonTRPostCheck()  for the calling sequence
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Note: This function is called BEFORE the function evaluation within the SNESNEWTONTR solver while the function set in
   SNESLineSearchSetPostCheck() is called AFTER the function evaluation.

.seealso: `SNESNewtonTRPostCheck()`, `SNESNewtonTRGetPostCheck()`
@*/
PetscErrorCode  SNESNewtonTRSetPostCheck(SNES snes,PetscErrorCode (*func)(SNES,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void *ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) tr->postcheck    = func;
  if (ctx)  tr->postcheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRGetPostCheck - Gets the post-check function

   Not collective

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  func - [optional] function evaluation routine, see for the calling sequence SNESNewtonTRPostCheck()
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRPostCheck()`
@*/
PetscErrorCode  SNESNewtonTRGetPostCheck(SNES snes,PetscErrorCode (**func)(SNES,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void **ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) *func = tr->postcheck;
  if (ctx)  *ctx  = tr->postcheckctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRPreCheck - Called before the step has been determined in SNESNEWTONTR

   Logically Collective on snes

   Input Parameters:
+  snes - the solver
.  X - The last solution
-  Y - The step direction

   Output Parameters:
.  changed_Y - Indicator that the step direction Y has been changed.

   Level: developer

.seealso: `SNESNewtonTRSetPreCheck()`, `SNESNewtonTRGetPreCheck()`
@*/
static PetscErrorCode SNESNewtonTRPreCheck(SNES snes,Vec X,Vec Y,PetscBool *changed_Y)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  if (tr->precheck) {
    PetscCall((*tr->precheck)(snes,X,Y,changed_Y,tr->precheckctx));
    PetscValidLogicalCollectiveBool(snes,*changed_Y,4);
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRPostCheck - Called after the step has been determined in SNESNEWTONTR but before the function evaluation

   Logically Collective on snes

   Input Parameters:
+  snes - the solver
.  X - The last solution
.  Y - The full step direction
-  W - The updated solution, W = X - Y

   Output Parameters:
+  changed_Y - indicator if step has been changed
-  changed_W - Indicator if the new candidate solution W has been changed.

   Notes:
     If Y is changed then W is recomputed as X - Y

   Level: developer

.seealso: `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRGetPostCheck()`
@*/
static PetscErrorCode SNESNewtonTRPostCheck(SNES snes,Vec X,Vec Y,Vec W,PetscBool *changed_Y,PetscBool *changed_W)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  if (tr->postcheck) {
    PetscCall((*tr->postcheck)(snes,X,Y,W,changed_Y,changed_W,tr->postcheckctx));
    PetscValidLogicalCollectiveBool(snes,*changed_Y,5);
    PetscValidLogicalCollectiveBool(snes,*changed_W,6);
  }
  PetscFunctionReturn(0);
}

/*
   SNESSolve_NEWTONTR - Implements Newton's Method with a very simple trust
   region approach for solving systems of nonlinear equations.

*/
static PetscErrorCode SNESSolve_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR            *neP = (SNES_NEWTONTR*)snes->data;
  Vec                      X,F,Y,G,Ytmp,W;
  PetscInt                 maxits,i,lits;
  PetscReal                rho,fnorm,gnorm,gpnorm,xnorm=0,delta,nrm,ynorm,norm1;
  PetscScalar              cnorm;
  KSP                      ksp;
  SNESConvergedReason      reason = SNES_CONVERGED_ITERATING;
  PetscBool                breakout = PETSC_FALSE;
  SNES_TR_KSPConverged_Ctx *ctx;
  PetscErrorCode           (*convtest)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),(*convdestroy)(void*);
  void                     *convctx;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* work vectors */
  G      = snes->work[1];
  Ytmp   = snes->work[2];
  W      = snes->work[3];

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  /* Set the linear stopping criteria to use the More' trick. */
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPGetConvergenceTest(ksp,&convtest,&convctx,&convdestroy));
  if (convtest != SNESTR_KSPConverged_Private) {
    PetscCall(PetscNew(&ctx));
    ctx->snes             = snes;
    PetscCall(KSPGetAndClearConvergenceTest(ksp,&ctx->convtest,&ctx->convctx,&ctx->convdestroy));
    PetscCall(KSPSetConvergenceTest(ksp,SNESTR_KSPConverged_Private,ctx,SNESTR_KSPConverged_Destroy));
    PetscCall(PetscInfo(snes,"Using Krylov convergence test SNESTR_KSPConverged_Private\n"));
  }

  if (!snes->vec_func_init_set) {
    PetscCall(SNESComputeFunction(snes,X,F));          /* F(X) */
  } else snes->vec_func_init_set = PETSC_FALSE;

  PetscCall(VecNorm(F,NORM_2,&fnorm));             /* fnorm <- || F || */
  SNESCheckFunctionNorm(snes,fnorm);
  PetscCall(VecNorm(X,NORM_2,&xnorm));             /* xnorm <- || X || */
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  delta      = xnorm ? neP->delta0*xnorm : neP->delta0;
  neP->delta = delta;
  PetscCall(SNESLogConvergenceHistory(snes,fnorm,0));
  PetscCall(SNESMonitor(snes,0,fnorm));

  /* test convergence */
  PetscCall((*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {

    /* Call general purpose update function */
    if (snes->ops->update) PetscCall((*snes->ops->update)(snes, snes->iter));

    /* Solve J Y = F, where J is Jacobian matrix */
    PetscCall(SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre));
    SNESCheckJacobianDomainerror(snes);
    PetscCall(KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre));
    PetscCall(KSPSolve(snes->ksp,F,Ytmp));
    PetscCall(KSPGetIterationNumber(snes->ksp,&lits));
    snes->linear_its += lits;

    PetscCall(PetscInfo(snes,"iter=%" PetscInt_FMT ", linear solve iterations=%" PetscInt_FMT "\n",snes->iter,lits));
    PetscCall(VecNorm(Ytmp,NORM_2,&nrm));
    norm1 = nrm;

    while (1) {
      PetscBool changed_y;
      PetscBool changed_w;
      PetscCall(VecCopy(Ytmp,Y));
      nrm  = norm1;

      /* Scale Y if need be and predict new value of F norm */
      if (nrm >= delta) {
        nrm    = delta/nrm;
        gpnorm = (1.0 - nrm)*fnorm;
        cnorm  = nrm;
        PetscCall(PetscInfo(snes,"Scaling direction by %g\n",(double)nrm));
        PetscCall(VecScale(Y,cnorm));
        nrm    = gpnorm;
        ynorm  = delta;
      } else {
        gpnorm = 0.0;
        PetscCall(PetscInfo(snes,"Direction is in Trust Region\n"));
        ynorm  = nrm;
      }
      /* PreCheck() allows for updates to Y prior to W <- X - Y */
      PetscCall(SNESNewtonTRPreCheck(snes,X,Y,&changed_y));
      PetscCall(VecWAXPY(W,-1.0,Y,X));         /* W <- X - Y */
      PetscCall(SNESNewtonTRPostCheck(snes,X,Y,W,&changed_y,&changed_w));
      if (changed_y) PetscCall(VecWAXPY(W,-1.0,Y,X));
      PetscCall(VecCopy(Y,snes->vec_sol_update));
      PetscCall(SNESComputeFunction(snes,W,G)); /*  F(X-Y) = G */
      PetscCall(VecNorm(G,NORM_2,&gnorm));      /* gnorm <- || g || */
      SNESCheckFunctionNorm(snes,gnorm);
      if (fnorm == gpnorm) rho = 0.0;
      else rho = (fnorm*fnorm - gnorm*gnorm)/(fnorm*fnorm - gpnorm*gpnorm);

      /* Update size of trust region */
      if      (rho < neP->mu)  delta *= neP->delta1;
      else if (rho < neP->eta) delta *= neP->delta2;
      else                     delta *= neP->delta3;
      PetscCall(PetscInfo(snes,"fnorm=%g, gnorm=%g, ynorm=%g\n",(double)fnorm,(double)gnorm,(double)ynorm));
      PetscCall(PetscInfo(snes,"gpred=%g, rho=%g, delta=%g\n",(double)gpnorm,(double)rho,(double)delta));

      neP->delta = delta;
      if (rho > neP->sigma) break;
      PetscCall(PetscInfo(snes,"Trying again in smaller region\n"));

      /* check to see if progress is hopeless */
      neP->itflag = PETSC_FALSE;
      PetscCall(SNESTR_Converged_Private(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP));
      if (!reason) PetscCall((*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP));
      if (reason == SNES_CONVERGED_SNORM_RELATIVE) reason = SNES_DIVERGED_INNER;
      if (reason) {
        /* We're not progressing, so return with the current iterate */
        PetscCall(SNESMonitor(snes,i+1,fnorm));
        breakout = PETSC_TRUE;
        break;
      }
      snes->numFailures++;
    }
    if (!breakout) {
      /* Update function and solution vectors */
      fnorm = gnorm;
      PetscCall(VecCopy(G,F));
      PetscCall(VecCopy(W,X));
      /* Monitor convergence */
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
      snes->iter = i+1;
      snes->norm = fnorm;
      snes->xnorm = xnorm;
      snes->ynorm = ynorm;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
      PetscCall(SNESLogConvergenceHistory(snes,snes->norm,lits));
      PetscCall(SNESMonitor(snes,snes->iter,snes->norm));
      /* Test for convergence, xnorm = || X || */
      neP->itflag = PETSC_TRUE;
      if (snes->ops->converged != SNESConvergedSkip) PetscCall(VecNorm(X,NORM_2,&xnorm));
      PetscCall((*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP));
      if (reason) break;
    } else break;
  }

  if (i == maxits) {
    PetscCall(PetscInfo(snes,"Maximum number of iterations has been reached: %" PetscInt_FMT "\n",maxits));
    if (!reason) reason = SNES_DIVERGED_MAX_IT;
  }
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->reason = reason;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  if (convtest != SNESTR_KSPConverged_Private) {
    PetscCall(KSPGetAndClearConvergenceTest(ksp,&ctx->convtest,&ctx->convctx,&ctx->convdestroy));
    PetscCall(PetscFree(ctx));
    PetscCall(KSPSetConvergenceTest(ksp,convtest,convctx,convdestroy));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode SNESSetUp_NEWTONTR(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESSetWorkVecs(snes,4));
  PetscCall(SNESSetUpMatrices(snes));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESReset_NEWTONTR(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_NEWTONTR(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESReset_NEWTONTR(snes));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode SNESSetFromOptions_NEWTONTR(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NEWTONTR  *ctx = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SNES trust region options for nonlinear equations");
  PetscCall(PetscOptionsReal("-snes_trtol","Trust region tolerance","SNESSetTrustRegionTolerance",snes->deltatol,&snes->deltatol,NULL));
  PetscCall(PetscOptionsReal("-snes_tr_mu","mu","None",ctx->mu,&ctx->mu,NULL));
  PetscCall(PetscOptionsReal("-snes_tr_eta","eta","None",ctx->eta,&ctx->eta,NULL));
  PetscCall(PetscOptionsReal("-snes_tr_sigma","sigma","None",ctx->sigma,&ctx->sigma,NULL));
  PetscCall(PetscOptionsReal("-snes_tr_delta0","delta0","None",ctx->delta0,&ctx->delta0,NULL));
  PetscCall(PetscOptionsReal("-snes_tr_delta1","delta1","None",ctx->delta1,&ctx->delta1,NULL));
  PetscCall(PetscOptionsReal("-snes_tr_delta2","delta2","None",ctx->delta2,&ctx->delta2,NULL));
  PetscCall(PetscOptionsReal("-snes_tr_delta3","delta3","None",ctx->delta3,&ctx->delta3,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_NEWTONTR(SNES snes,PetscViewer viewer)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Trust region tolerance %g (-snes_trtol)\n",(double)snes->deltatol));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  mu=%g, eta=%g, sigma=%g\n",(double)tr->mu,(double)tr->eta,(double)tr->sigma));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  delta0=%g, delta1=%g, delta2=%g, delta3=%g\n",(double)tr->delta0,(double)tr->delta1,(double)tr->delta2,(double)tr->delta3));
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
/*MC
      SNESNEWTONTR - Newton based nonlinear solver that uses a trust region

   Options Database:
+    -snes_trtol <tol> - trust region tolerance
.    -snes_tr_mu <mu> - trust region parameter
.    -snes_tr_eta <eta> - trust region parameter
.    -snes_tr_sigma <sigma> - trust region parameter
.    -snes_tr_delta0 <delta0> -  initial size of the trust region is delta0*norm2(x)
.    -snes_tr_delta1 <delta1> - trust region parameter
.    -snes_tr_delta2 <delta2> - trust region parameter
-    -snes_tr_delta3 <delta3> - trust region parameter

   The basic algorithm is taken from "The Minpack Project", by More',
   Sorensen, Garbow, Hillstrom, pages 88-111 of "Sources and Development
   of Mathematical Software", Wayne Cowell, editor.

   Level: intermediate

.seealso: `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONLS`, `SNESSetTrustRegionTolerance()`

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR  *neP;

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

  PetscCall(PetscNewLog(snes,&neP));
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
