#include <../src/snes/impls/richardson/snesrichardsonimpl.h>

PetscErrorCode SNESReset_NRichardson(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESDestroy_NRichardson(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESReset_NRichardson(snes));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESSetUp_NRichardson(SNES snes)
{
  PetscFunctionBegin;
  PetscCheck(snes->npcside != PC_RIGHT, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "NRichardson only supports left preconditioning");
  if (snes->functype == SNES_FUNCTION_DEFAULT) snes->functype = SNES_FUNCTION_UNPRECONDITIONED;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetFromOptions_NRichardson(SNES snes, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNES Richardson options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESView_NRichardson(SNES snes, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) { }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESSolve_NRichardson(SNES snes)
{
  Vec                  X, Y, F;
  PetscReal            xnorm, fnorm, ynorm;
  PetscInt             maxits, i;
  SNESLineSearchReason lsresult;
  SNESConvergedReason  reason;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;        /* maximum number of iterations */
  X      = snes->vec_sol;        /* X^n */
  Y      = snes->vec_sol_update; /* \tilde X */
  F      = snes->vec_func;       /* residual vector */

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  if (snes->npc && snes->functype == SNES_FUNCTION_PRECONDITIONED) {
    PetscCall(SNESApplyNPC(snes, X, NULL, F));
    PetscCall(SNESGetConvergedReason(snes->npc, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(VecNorm(F, NORM_2, &fnorm));
  } else {
    if (!snes->vec_func_init_set) {
      PetscCall(SNESComputeFunction(snes, X, F));
    } else snes->vec_func_init_set = PETSC_FALSE;

    PetscCall(VecNorm(F, NORM_2, &fnorm));
    SNESCheckFunctionNorm(snes, fnorm);
  }
  if (snes->npc && snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
    PetscCall(SNESApplyNPC(snes, X, F, Y));
    PetscCall(SNESGetConvergedReason(snes->npc, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else {
    PetscCall(VecCopy(F, Y));
  }

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0));
  PetscCall(SNESMonitor(snes, 0, fnorm));

  /* test convergence */
  PetscUseTypeMethod(snes, converged, 0, 0.0, 0.0, fnorm, &snes->reason, snes->cnvP);
  if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Call general purpose update function */
  PetscTryTypeMethod(snes, update, snes->iter);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm * snes->rtol;
  /* test convergence */
  PetscUseTypeMethod(snes, converged, 0, 0.0, 0.0, fnorm, &snes->reason, snes->cnvP);
  if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);

  for (i = 1; i < maxits + 1; i++) {
    PetscCall(SNESLineSearchApply(snes->linesearch, X, F, &fnorm, Y));
    PetscCall(SNESLineSearchGetReason(snes->linesearch, &lsresult));
    PetscCall(SNESLineSearchGetNorms(snes->linesearch, &xnorm, &fnorm, &ynorm));
    if (lsresult) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        break;
      }
    }
    if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }

    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter  = i;
    snes->norm  = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes, snes->norm, 0));
    PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
    /* Test for convergence */
    PetscUseTypeMethod(snes, converged, snes->iter, xnorm, ynorm, fnorm, &snes->reason, snes->cnvP);
    if (snes->reason) break;

    /* Call general purpose update function */
    PetscTryTypeMethod(snes, update, snes->iter);

    if (snes->npc) {
      if (snes->functype == SNES_FUNCTION_PRECONDITIONED) {
        PetscCall(SNESApplyNPC(snes, X, NULL, Y));
        PetscCall(VecNorm(F, NORM_2, &fnorm));
        PetscCall(VecCopy(Y, F));
      } else {
        PetscCall(SNESApplyNPC(snes, X, F, Y));
      }
      PetscCall(SNESGetConvergedReason(snes->npc, &reason));
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    } else {
      PetscCall(VecCopy(F, Y));
    }
  }
  if (i == maxits + 1) {
    PetscCall(PetscInfo(snes, "Maximum number of iterations has been reached: %" PetscInt_FMT "\n", maxits));
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESNRICHARDSON - Richardson nonlinear solver that uses successive substitutions, also sometimes known as Picard iteration.

   Options Database Keys:
+  -snes_linesearch_type <l2,cp,basic> - Line search type.
-  -snes_linesearch_damping<1.0> - Damping for the line search.

   Level: beginner

   Notes:
   If no inner nonlinear preconditioner is provided then solves F(x) - b = 0 using x^{n+1} = x^{n} - lambda
   (F(x^n) - b) where lambda is obtained either `SNESLineSearchSetDamping()`, -snes_damping or a line search.  If
   an inner nonlinear preconditioner is provided (either with -npc_snes_type or `SNESSetNPC()`) then the inner
   solver is called an initial solution x^n and the nonlinear Richardson uses x^{n+1} = x^{n} + lambda d^{n}
   where d^{n} = \hat{x}^{n} - x^{n} where \hat{x}^{n} is the solution returned from the inner solver.

   The update, especially without inner nonlinear preconditioner, may be ill-scaled.  If using the basic
   linesearch, one may have to scale the update with -snes_linesearch_damping

   This uses no derivative information thus will be much slower then Newton's method obtained with -snes_type ls

   Only supports left non-linear preconditioning.

.seealso: `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONLS`, `SNESNEWTONTR`, `SNESNGMRES`, `SNESQN`, `SNESNCG`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NRichardson(SNES snes)
{
  SNES_NRichardson *neP;
  SNESLineSearch    linesearch;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NRichardson;
  snes->ops->setup          = SNESSetUp_NRichardson;
  snes->ops->setfromoptions = SNESSetFromOptions_NRichardson;
  snes->ops->view           = SNESView_NRichardson;
  snes->ops->solve          = SNESSolve_NRichardson;
  snes->ops->reset          = SNESReset_NRichardson;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_TRUE;

  snes->npcside = PC_LEFT;

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHL2));

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNew(&neP));
  snes->data = (void *)neP;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
    snes->stol      = 1e-20;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
