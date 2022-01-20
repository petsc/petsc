
#include <../src/snes/impls/ntrdc/ntrdcimpl.h>                /*I   "petscsnes.h"   I*/

typedef struct {
  SNES           snes;
  /*  Information on the regular SNES convergence test; which may have been user provided
      Copied from tr.c (maybe able to disposed, but this is a private function) - Heeho
      Same with SNESTR_KSPConverged_Private, SNESTR_KSPConverged_Destroy, and SNESTR_Converged_Private
 */

  PetscErrorCode (*convtest)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
  PetscErrorCode (*convdestroy)(void*);
  void           *convctx;
} SNES_TRDC_KSPConverged_Ctx;

static PetscErrorCode SNESTRDC_KSPConverged_Private(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *cctx)
{
  SNES_TRDC_KSPConverged_Ctx  *ctx = (SNES_TRDC_KSPConverged_Ctx*)cctx;
  SNES                        snes = ctx->snes;
  SNES_NEWTONTRDC             *neP = (SNES_NEWTONTRDC*)snes->data;
  Vec                         x;
  PetscReal                   nrm;
  PetscErrorCode              ierr;

  PetscFunctionBegin;
  ierr = (*ctx->convtest)(ksp,n,rnorm,reason,ctx->convctx);CHKERRQ(ierr);
  if (*reason) {
    ierr = PetscInfo2(snes,"Default or user provided convergence test KSP iterations=%" PetscInt_FMT ", rnorm=%g\n",n,(double)rnorm);CHKERRQ(ierr);
  }
  /* Determine norm of solution */
  ierr = KSPBuildSolution(ksp,NULL,&x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm >= neP->delta) {
    ierr    = PetscInfo2(snes,"Ending linear iteration early, delta=%g, length=%g\n",(double)neP->delta,(double)nrm);CHKERRQ(ierr);
    *reason = KSP_CONVERGED_STEP_LENGTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTRDC_KSPConverged_Destroy(void *cctx)
{
  SNES_TRDC_KSPConverged_Ctx *ctx = (SNES_TRDC_KSPConverged_Ctx*)cctx;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = (*ctx->convdestroy)(ctx->convctx);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/*
   SNESTRDC_Converged_Private -test convergence JUST for
   the trust region tolerance.

*/
static PetscErrorCode SNESTRDC_Converged_Private(SNES snes,PetscInt it,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  SNES_NEWTONTRDC  *neP = (SNES_NEWTONTRDC*)snes->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  *reason = SNES_CONVERGED_ITERATING;
  if (neP->delta < xnorm * snes->deltatol) {
    ierr    = PetscInfo3(snes,"Diverged due to too small a trust region %g<%g*%g\n",(double)neP->delta,(double)xnorm,(double)snes->deltatol);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_TR_DELTA;
  } else if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    ierr    = PetscInfo1(snes,"Exceeded maximum number of function evaluations: %" PetscInt_FMT "\n",snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }
  PetscFunctionReturn(0);
}

/*@
  SNESNewtonTRDCGetRhoFlag - Get whether the solution update is within the trust-region.

  Input Parameters:
. snes - the nonlinear solver object

  Output Parameters:
. rho_flag: PETSC_TRUE if the solution update is in the trust-region; otherwise, PETSC_FALSE

  Level: developer

@*/
PetscErrorCode  SNESNewtonTRDCGetRhoFlag(SNES snes,PetscBool *rho_flag)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidBoolPointer(rho_flag,2);
  *rho_flag = tr->rho_satisfied;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRDCSetPreCheck - Sets a user function that is called before the search step has been determined.
       Allows the user a chance to change or override the trust region decision.

   Logically Collective on snes

   Input Parameters:
+  snes - the nonlinear solver object
.  func - [optional] function evaluation routine, see SNESNewtonTRDCPreCheck()  for the calling sequence
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Note: This function is called BEFORE the function evaluation within the SNESNEWTONTRDC solver.

.seealso: SNESNewtonTRDCPreCheck(), SNESNewtonTRDCGetPreCheck(), SNESNewtonTRDCSetPostCheck(), SNESNewtonTRDCGetPostCheck()
@*/
PetscErrorCode  SNESNewtonTRDCSetPreCheck(SNES snes, PetscErrorCode (*func)(SNES,Vec,Vec,PetscBool*,void*),void *ctx)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) tr->precheck    = func;
  if (ctx)  tr->precheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRDCGetPreCheck - Gets the pre-check function

   Not collective

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  func - [optional] function evaluation routine, see for the calling sequence SNESNewtonTRDCPreCheck()
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: SNESNewtonTRDCSetPreCheck(), SNESNewtonTRDCPreCheck()
@*/
PetscErrorCode  SNESNewtonTRDCGetPreCheck(SNES snes, PetscErrorCode (**func)(SNES,Vec,Vec,PetscBool*,void*),void **ctx)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) *func = tr->precheck;
  if (ctx)  *ctx  = tr->precheckctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRDCSetPostCheck - Sets a user function that is called after the search step has been determined but before the next
       function evaluation. Allows the user a chance to change or override the decision of the line search routine

   Logically Collective on snes

   Input Parameters:
+  snes - the nonlinear solver object
.  func - [optional] function evaluation routine, see SNESNewtonTRDCPostCheck()  for the calling sequence
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Note: This function is called BEFORE the function evaluation within the SNESNEWTONTRDC solver while the function set in
   SNESLineSearchSetPostCheck() is called AFTER the function evaluation.

.seealso: SNESNewtonTRDCPostCheck(), SNESNewtonTRDCGetPostCheck()
@*/
PetscErrorCode  SNESNewtonTRDCSetPostCheck(SNES snes,PetscErrorCode (*func)(SNES,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void *ctx)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) tr->postcheck    = func;
  if (ctx)  tr->postcheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRDCGetPostCheck - Gets the post-check function

   Not collective

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  func - [optional] function evaluation routine, see for the calling sequence SNESNewtonTRDCPostCheck()
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: SNESNewtonTRDCSetPostCheck(), SNESNewtonTRDCPostCheck()
@*/
PetscErrorCode  SNESNewtonTRDCGetPostCheck(SNES snes,PetscErrorCode (**func)(SNES,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void **ctx)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) *func = tr->postcheck;
  if (ctx)  *ctx  = tr->postcheckctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRDCPreCheck - Called before the step has been determined in SNESNEWTONTRDC

   Logically Collective on snes

   Input Parameters:
+  snes - the solver
.  X - The last solution
-  Y - The step direction

   Output Parameters:
.  changed_Y - Indicator that the step direction Y has been changed.

   Level: developer

.seealso: SNESNewtonTRDCSetPreCheck(), SNESNewtonTRDCGetPreCheck()
@*/
static PetscErrorCode SNESNewtonTRDCPreCheck(SNES snes,Vec X,Vec Y,PetscBool *changed_Y)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  if (tr->precheck) {
    ierr = (*tr->precheck)(snes,X,Y,changed_Y,tr->precheckctx);CHKERRQ(ierr);
    PetscValidLogicalCollectiveBool(snes,*changed_Y,4);
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRDCPostCheck - Called after the step has been determined in SNESNEWTONTRDC but before the function evaluation

   Logically Collective on snes

   Input Parameters:
+  snes - the solver.  X - The last solution
.  Y - The full step direction
-  W - The updated solution, W = X - Y

   Output Parameters:
+  changed_Y - indicator if step has been changed
-  changed_W - Indicator if the new candidate solution W has been changed.

   Notes:
     If Y is changed then W is recomputed as X - Y

   Level: developer

.seealso: SNESNewtonTRDCSetPostCheck(), SNESNewtonTRDCGetPostCheck()
@*/
static PetscErrorCode SNESNewtonTRDCPostCheck(SNES snes,Vec X,Vec Y,Vec W,PetscBool *changed_Y,PetscBool *changed_W)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  if (tr->postcheck) {
    ierr = (*tr->postcheck)(snes,X,Y,W,changed_Y,changed_W,tr->postcheckctx);CHKERRQ(ierr);
    PetscValidLogicalCollectiveBool(snes,*changed_Y,5);
    PetscValidLogicalCollectiveBool(snes,*changed_W,6);
  }
  PetscFunctionReturn(0);
}

/*
   SNESSolve_NEWTONTRDC - Implements Newton's Method with trust-region subproblem and adds dogleg Cauchy
   (Steepest Descent direction) step and direction if the trust region is not satisfied for solving system of
   nonlinear equations

*/
static PetscErrorCode SNESSolve_NEWTONTRDC(SNES snes)
{
  SNES_NEWTONTRDC            *neP = (SNES_NEWTONTRDC*)snes->data;
  Vec                        X,F,Y,G,W,GradF,YNtmp;
  Vec                        YCtmp;
  Mat                        jac;
  PetscErrorCode             ierr;
  PetscInt                   maxits,i,j,lits,inner_count,bs;
  PetscReal                  rho,fnorm,gnorm,xnorm=0,delta,ynorm,temp_xnorm,temp_ynorm;  /* TRDC inner iteration */
  PetscReal                  inorms[99]; /* need to make it dynamic eventually, fixed max block size of 99 for now */
  PetscReal                  deltaM,ynnorm,f0,mp,gTy,g,yTHy;  /* rho calculation */
  PetscReal                  auk,gfnorm,ycnorm,c0,c1,c2,tau,tau_pos,tau_neg,gTBg;  /* Cauchy Point */
  KSP                        ksp;
  SNESConvergedReason        reason = SNES_CONVERGED_ITERATING;
  PetscBool                  breakout = PETSC_FALSE;
  SNES_TRDC_KSPConverged_Ctx *ctx;
  PetscErrorCode             (*convtest)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),(*convdestroy)(void*);
  void                       *convctx;

  PetscFunctionBegin;
  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* update vector */
  G      = snes->work[1];               /* updated residual */
  W      = snes->work[2];               /* temporary vector */
  GradF  = snes->work[3];               /* grad f = J^T F */
  YNtmp  = snes->work[4];               /* Newton solution */
  YCtmp  = snes->work[5];               /* Cauchy solution */

  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  ierr = VecGetBlockSize(YNtmp,&bs);CHKERRQ(ierr);

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  /* Set the linear stopping criteria to use the More' trick. From tr.c */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetConvergenceTest(ksp,&convtest,&convctx,&convdestroy);CHKERRQ(ierr);
  if (convtest != SNESTRDC_KSPConverged_Private) {
    ierr                  = PetscNew(&ctx);CHKERRQ(ierr);
    ctx->snes             = snes;
    ierr                  = KSPGetAndClearConvergenceTest(ksp,&ctx->convtest,&ctx->convctx,&ctx->convdestroy);CHKERRQ(ierr);
    ierr                  = KSPSetConvergenceTest(ksp,SNESTRDC_KSPConverged_Private,ctx,SNESTRDC_KSPConverged_Destroy);CHKERRQ(ierr);
    ierr                  = PetscInfo(snes,"Using Krylov convergence test SNESTRDC_KSPConverged_Private\n");CHKERRQ(ierr);
  }

  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);          /* F(X) */
  } else snes->vec_func_init_set = PETSC_FALSE;

  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);             /* fnorm <- || F || */
  SNESCheckFunctionNorm(snes,fnorm);
  ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);             /* xnorm <- || X || */

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  delta      = xnorm ? neP->delta0*xnorm : neP->delta0;  /* initial trust region size scaled by xnorm */
  deltaM     = xnorm ? neP->deltaM*xnorm : neP->deltaM;  /* maximum trust region size scaled by xnorm */
  neP->delta = delta;
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  neP->rho_satisfied = PETSC_FALSE;

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {
    PetscBool changed_y;
    PetscBool changed_w;

    /* dogleg method */
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    SNESCheckJacobianDomainerror(snes);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian);CHKERRQ(ierr);
    ierr = KSPSolve(snes->ksp,F,YNtmp);CHKERRQ(ierr);   /* Quasi Newton Solution */
    SNESCheckKSPSolve(snes);  /* this is necessary but old tr.c did not have it*/
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,&jac,NULL,NULL,NULL);CHKERRQ(ierr);

    /* rescale Jacobian, Newton solution update, and re-calculate delta for multiphase (multivariable)
       for inner iteration and Cauchy direction calculation
    */
    if (bs > 1 && neP->auto_scale_multiphase) {
      ierr = VecStrideNormAll(YNtmp,NORM_INFINITY,inorms);CHKERRQ(ierr);
      for (j=0; j<bs; j++) {
        if (neP->auto_scale_max > 1.0) {
          if (inorms[j] < 1.0/neP->auto_scale_max) {
            inorms[j] = 1.0/neP->auto_scale_max;
          }
        }
        ierr = VecStrideSet(W,j,inorms[j]);CHKERRQ(ierr);
        ierr = VecStrideScale(YNtmp,j,1.0/inorms[j]);
        ierr = VecStrideScale(X,j,1.0/inorms[j]);
      }
      ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);
      if (i == 0) {
        delta = neP->delta0*xnorm;
      } else {
        delta = neP->delta*xnorm;
      }
      deltaM = neP->deltaM*xnorm;
      ierr = MatDiagonalScale(jac,PETSC_NULL,W);CHKERRQ(ierr);
    }

    /* calculating GradF of minimization function */
    ierr = MatMultTranspose(jac,F,GradF);CHKERRQ(ierr);  /* grad f = J^T F */
    ierr = VecNorm(YNtmp,NORM_2,&ynnorm);CHKERRQ(ierr);  /* ynnorm <- || Y_newton || */

    inner_count = 0;
    neP->rho_satisfied = PETSC_FALSE;
    while (1) {
      if (ynnorm <= delta) {  /* see if the Newton solution is within the trust region */
        ierr = VecCopy(YNtmp,Y);CHKERRQ(ierr);
      } else if (neP->use_cauchy) { /* use Cauchy direction if enabled */
        ierr = MatMult(jac,GradF,W);CHKERRQ(ierr);
        ierr = VecDotRealPart(W,W,&gTBg);CHKERRQ(ierr);  /* completes GradF^T J^T J GradF */
        ierr = VecNorm(GradF,NORM_2,&gfnorm);CHKERRQ(ierr);  /* grad f norm <- || grad f || */
        if (gTBg <= 0.0) {
          auk = PETSC_MAX_REAL;
        } else {
          auk = PetscSqr(gfnorm)/gTBg;
        }
        auk  = PetscMin(delta/gfnorm,auk);
        ierr = VecCopy(GradF,YCtmp);CHKERRQ(ierr);  /* this could be improved */
        ierr = VecScale(YCtmp,auk);CHKERRQ(ierr);  /* YCtmp, Cauchy solution*/
        ierr = VecNorm(YCtmp,NORM_2,&ycnorm);CHKERRQ(ierr);  /* ycnorm <- || Y_cauchy || */
        if (ycnorm >= delta) {  /* see if the Cauchy solution meets the criteria */
            ierr = VecCopy(YCtmp,Y);CHKERRQ(ierr);
            ierr = PetscInfo3(snes,"DL evaluated. delta: %8.4e, ynnorm: %8.4e, ycnorm: %8.4e\n",(double)delta,(double)ynnorm,(double)ycnorm);CHKERRQ(ierr);
        } else {  /* take ratio, tau, of Cauchy and Newton direction and step */
          ierr    = VecAXPY(YNtmp,-1.0,YCtmp);CHKERRQ(ierr);  /* YCtmp = A, YNtmp = B */
          ierr    = VecNorm(YNtmp,NORM_2,&c0);CHKERRQ(ierr);  /* this could be improved */
          c0      = PetscSqr(c0);
          ierr    = VecDotRealPart(YCtmp,YNtmp,&c1);CHKERRQ(ierr);
          c1      = 2.0*c1;
          ierr    = VecNorm(YCtmp,NORM_2,&c2);CHKERRQ(ierr);  /* this could be improved */
          c2      = PetscSqr(c2) - PetscSqr(delta);
          tau_pos = (c1 + PetscSqrtReal(PetscSqr(c1) - 4.*c0*c2))/(2.*c0); /* quadratic formula */
          tau_neg = (c1 - PetscSqrtReal(PetscSqr(c1) - 4.*c0*c2))/(2.*c0);
          tau     = PetscMax(tau_pos, tau_neg);  /* can tau_neg > tau_pos? I don't think so, but just in case. */
          ierr    = PetscInfo3(snes,"DL evaluated. tau: %8.4e, ynnorm: %8.4e, ycnorm: %8.4e\n",(double)tau,(double)ynnorm,(double)ycnorm);CHKERRQ(ierr);
          ierr    = VecWAXPY(W,tau,YNtmp,YCtmp);CHKERRQ(ierr);
          ierr    = VecAXPY(W,-tau,YCtmp);CHKERRQ(ierr);
          ierr    = VecCopy(W, Y);CHKERRQ(ierr); /* this could be improved */
        }
      } else {
        /* if Cauchy is disabled, only use Newton direction */
        auk = delta/ynnorm;
        ierr = VecScale(YNtmp,auk);CHKERRQ(ierr);
        ierr = VecCopy(YNtmp,Y);CHKERRQ(ierr); /* this could be improved (many VecCopy, VecNorm)*/
      }

      ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);  /* compute the final ynorm  */
      f0 = 0.5*PetscSqr(fnorm);  /* minimizing function f(X) */
      ierr = MatMult(jac,Y,W);CHKERRQ(ierr);
      ierr = VecDotRealPart(W,W,&yTHy);CHKERRQ(ierr);  /* completes GradY^T J^T J GradY */
      ierr = VecDotRealPart(GradF,Y,&gTy);CHKERRQ(ierr);
      mp = f0 - gTy + 0.5*yTHy;  /* quadratic model to satisfy, -gTy because our update is X-Y*/

      /* scale back solution update */
      if (bs > 1 && neP->auto_scale_multiphase) {
        for (j=0; j<bs; j++) {
          ierr = VecStrideScale(Y,j,inorms[j]);
          if (inner_count == 0) {
            /* TRDC inner algorithm does not need scaled X after calculating delta in the outer iteration */
            /* need to scale back X to match Y and provide proper update to the external code */
            ierr = VecStrideScale(X,j,inorms[j]);
          }
        }
        if (inner_count == 0) {ierr = VecNorm(X,NORM_2,&temp_xnorm);CHKERRQ(ierr);}  /* only in the first iteration */
        ierr = VecNorm(Y,NORM_2,&temp_ynorm);CHKERRQ(ierr);
      } else {
        temp_xnorm = xnorm;
        temp_ynorm = ynorm;
      }
      inner_count++;

      /* Evaluate the solution to meet the improvement ratio criteria */
      ierr = SNESNewtonTRDCPreCheck(snes,X,Y,&changed_y);CHKERRQ(ierr);
      ierr = VecWAXPY(W,-1.0,Y,X);CHKERRQ(ierr);
      ierr = SNESNewtonTRDCPostCheck(snes,X,Y,W,&changed_y,&changed_w);CHKERRQ(ierr);
      if (changed_y) {ierr = VecWAXPY(W,-1.0,Y,X);CHKERRQ(ierr);}
      ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr); /*  F(X-Y) = G */
      ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);      /* gnorm <- || g || */
      SNESCheckFunctionNorm(snes,gnorm);
      g = 0.5*PetscSqr(gnorm); /* minimizing function g(W) */
      if (f0 == mp) rho = 0.0;
      else rho = (f0 - g)/(f0 - mp);  /* actual improvement over predicted improvement */

      if (rho < neP->eta2) {
        delta *= neP->t1;  /* shrink the region */
      } else if (rho > neP->eta3) {
        delta = PetscMin(neP->t2*delta,deltaM); /* expand the region, but not greater than deltaM */
      }

      neP->delta = delta;
      if (rho >= neP->eta1) {
        /* unscale delta and xnorm before going to the next outer iteration */
        if (bs > 1 && neP->auto_scale_multiphase) {
          neP->delta = delta/xnorm;
          xnorm      = temp_xnorm;
          ynorm      = temp_ynorm;
        }
        neP->rho_satisfied = PETSC_TRUE;
        break;  /* the improvement ratio is satisfactory */
      }
      ierr = PetscInfo(snes,"Trying again in smaller region\n");CHKERRQ(ierr);

      /* check to see if progress is hopeless */
      neP->itflag = PETSC_FALSE;
      /* both delta, ynorm, and xnorm are either scaled or unscaled */
      ierr        = SNESTRDC_Converged_Private(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (!reason) {
         /* temp_xnorm, temp_ynorm is always unscaled */
         /* also the inner iteration already calculated the Jacobian and solved the matrix */
         /* therefore, it should be passing iteration number of iter+1 instead of iter+0 in the first iteration and after */
         ierr = (*snes->ops->converged)(snes,snes->iter+1,temp_xnorm,temp_ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      }
      /* if multiphase state changes, break out inner iteration */
      if (reason == SNES_BREAKOUT_INNER_ITER) {
        if (bs > 1 && neP->auto_scale_multiphase) {
          /* unscale delta and xnorm before going to the next outer iteration */
          neP->delta = delta/xnorm;
          xnorm      = temp_xnorm;
          ynorm      = temp_ynorm;
        }
        reason = SNES_CONVERGED_ITERATING;
        break;
      }
      if (reason == SNES_CONVERGED_SNORM_RELATIVE) reason = SNES_DIVERGED_INNER;
      if (reason) {
        if (reason < 0) {
            /* We're not progressing, so return with the current iterate */
            ierr     = SNESMonitor(snes,i+1,fnorm);CHKERRQ(ierr);
            breakout = PETSC_TRUE;
            break;
        } else if (reason > 0) {
            /* We're converged, so return with the current iterate and update solution */
            ierr     = SNESMonitor(snes,i+1,fnorm);CHKERRQ(ierr);
            breakout = PETSC_FALSE;
            break;
        }
      }
      snes->numFailures++;
    }
    if (!breakout) {
      /* Update function and solution vectors */
      fnorm       = gnorm;
      ierr        = VecCopy(G,F);CHKERRQ(ierr);
      ierr        = VecCopy(W,X);CHKERRQ(ierr);
      /* Monitor convergence */
      ierr        = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
      snes->iter  = i+1;
      snes->norm  = fnorm;
      snes->xnorm = xnorm;
      snes->ynorm = ynorm;
      ierr        = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
      ierr        = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
      ierr        = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
      /* Test for convergence, xnorm = || X || */
      neP->itflag = PETSC_TRUE;
      if (snes->ops->converged != SNESConvergedSkip) {ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);}
      ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (reason) break;
    } else break;
  }

  /* ierr         = PetscFree(inorms);CHKERRQ(ierr); */
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %" PetscInt_FMT "\n",maxits);CHKERRQ(ierr);
    if (!reason) reason = SNES_DIVERGED_MAX_IT;
  }
  ierr         = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->reason = reason;
  ierr         = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  if (convtest != SNESTRDC_KSPConverged_Private) {
    ierr       = KSPGetAndClearConvergenceTest(ksp,&ctx->convtest,&ctx->convctx,&ctx->convdestroy);CHKERRQ(ierr);
    ierr       = PetscFree(ctx);CHKERRQ(ierr);
    ierr       = KSPSetConvergenceTest(ksp,convtest,convctx,convdestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode SNESSetUp_NEWTONTRDC(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetWorkVecs(snes,6);CHKERRQ(ierr);
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESReset_NEWTONTRDC(SNES snes)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_NEWTONTRDC(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NEWTONTRDC(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode SNESSetFromOptions_NEWTONTRDC(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NEWTONTRDC  *ctx = (SNES_NEWTONTRDC*)snes->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SNES trust region options for nonlinear equations");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_tol","Trust region tolerance","SNESSetTrustRegionTolerance",snes->deltatol,&snes->deltatol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_eta1","eta1","None",ctx->eta1,&ctx->eta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_eta2","eta2","None",ctx->eta2,&ctx->eta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_eta3","eta3","None",ctx->eta3,&ctx->eta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_t1","t1","None",ctx->t1,&ctx->t1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_t2","t2","None",ctx->t2,&ctx->t2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_deltaM","deltaM","None",ctx->deltaM,&ctx->deltaM,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_delta0","delta0","None",ctx->delta0,&ctx->delta0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trdc_auto_scale_max","auto_scale_max","None",ctx->auto_scale_max,&ctx->auto_scale_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_trdc_use_cauchy","use_cauchy","use Cauchy step and direction",ctx->use_cauchy,&ctx->use_cauchy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_trdc_auto_scale_multiphase","auto_scale_multiphase","Auto scaling for proper cauchy direction",ctx->auto_scale_multiphase,&ctx->auto_scale_multiphase,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_NEWTONTRDC(SNES snes,PetscViewer viewer)
{
  SNES_NEWTONTRDC  *tr = (SNES_NEWTONTRDC*)snes->data;
  PetscErrorCode   ierr;
  PetscBool        iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Trust region tolerance (-snes_trtol)\n",(double)snes->deltatol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  eta1=%g, eta2=%g, eta3=%g\n",(double)tr->eta1,(double)tr->eta2,(double)tr->eta3);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  delta0=%g, t1=%g, t2=%g, deltaM=%g\n",(double)tr->delta0,(double)tr->t1,(double)tr->t2,(double)tr->deltaM);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
/*MC
      SNESNEWTONTRDC - Newton based nonlinear solver that uses trust-region dogleg method with Cauchy direction

   Options Database:
+   -snes_trdc_tol <tol> - trust region tolerance
.   -snes_trdc_eta1 <eta1> - trust region parameter 0.0 <= eta1 <= eta2, rho >= eta1 breaks out of the inner iteration (default: eta1=0.001)
.   -snes_trdc_eta2 <eta2> - trust region parameter 0.0 <= eta1 <= eta2, rho <= eta2 shrinks the trust region (default: eta2=0.25)
.   -snes_trdc_eta3 <eta3> - trust region parameter eta3 > eta2, rho >= eta3 expands the trust region (default: eta3=0.75)
.   -snes_trdc_t1 <t1> - trust region parameter, shrinking factor of trust region (default: 0.25)
.   -snes_trdc_t2 <t2> - trust region parameter, expanding factor of trust region (default: 2.0)
.   -snes_trdc_deltaM <deltaM> - trust region parameter, max size of trust region, deltaM*norm2(x) (default: 0.5)
.   -snes_trdc_delta0 <delta0> - trust region parameter, initial size of trust region, delta0*norm2(x) (default: 0.1)
.   -snes_trdc_auto_scale_max <auto_scale_max> - used with auto_scale_multiphase, caps the maximum auto-scaling factor
.   -snes_trdc_use_cauchy <use_cauchy> - True uses dogleg Cauchy (Steepest Descent direction) step & direction in the trust region algorithm
-   -snes_trdc_auto_scale_multiphase <auto_scale_multiphase> - True turns on auto-scaling for multivariable block matrix for Cauchy and trust region

    Notes:
    The algorithm is taken from "Linear and Nonlinear Solvers for Simulating Multiphase Flow
    within Large-Scale Engineered Subsurface Systems" by Heeho D. Park, Glenn E. Hammond,
    Albert J. Valocchi, Tara LaForce.

   Level: intermediate

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESSetTrustRegionTolerance(), SNESNEWTONTRDC

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONTRDC(SNES snes)
{
  SNES_NEWTONTRDC  *neP;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONTRDC;
  snes->ops->solve          = SNESSolve_NEWTONTRDC;
  snes->ops->destroy        = SNESDestroy_NEWTONTRDC;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONTRDC;
  snes->ops->view           = SNESView_NEWTONTRDC;
  snes->ops->reset          = SNESReset_NEWTONTRDC;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  ierr        = PetscNewLog(snes,&neP);CHKERRQ(ierr);
  snes->data  = (void*)neP;
  neP->delta  = 0.0;
  neP->delta0 = 0.1;
  neP->eta1   = 0.001;
  neP->eta2   = 0.25;
  neP->eta3   = 0.75;
  neP->t1     = 0.25;
  neP->t2     = 2.0;
  neP->deltaM = 0.5;
  neP->sigma  = 0.0001;
  neP->itflag = PETSC_FALSE;
  neP->rnorm0 = 0.0;
  neP->ttol   = 0.0;
  neP->use_cauchy            = PETSC_TRUE;
  neP->auto_scale_multiphase = PETSC_FALSE;
  neP->auto_scale_max        = -1.0;
  neP->rho_satisfied         = PETSC_FALSE;
  snes->deltatol             = 1.e-12;

  /* for multiphase (multivariable) scaling */
  /* may be used for dynamic allocation of inorms, but it fails snes_tutorials-ex3_13
     on test forced DIVERGED_JACOBIAN_DOMAIN test. I will use static array for now.
  ierr = VecGetBlockSize(snes->work[0],&neP->bs);CHKERRQ(ierr);
  ierr = PetscCalloc1(neP->bs,&neP->inorms);CHKERRQ(ierr);
  */

  PetscFunctionReturn(0);
}
