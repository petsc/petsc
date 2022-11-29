
#include <../src/snes/impls/ls/lsimpl.h>

/*
     Checks if J^T F = 0 which implies we've found a local minimum of the norm of the function,
    || F(u) ||_2 but not a zero, F(u) = 0. In the case when one cannot compute J^T F we use the fact that
    0 = (J^T F)^T W = F^T J W iff W not in the null space of J. Thanks for Jorge More
    for this trick. One assumes that the probability that W is in the null space of J is very, very small.
*/
static PetscErrorCode SNESNEWTONLSCheckLocalMin_Private(SNES snes, Mat A, Vec F, PetscReal fnorm, PetscBool *ismin)
{
  PetscReal a1;
  PetscBool hastranspose;
  Vec       W;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  PetscCall(MatHasOperation(A, MATOP_MULT_TRANSPOSE, &hastranspose));
  PetscCall(VecDuplicate(F, &W));
  if (hastranspose) {
    /* Compute || J^T F|| */
    PetscCall(MatMultTranspose(A, F, W));
    PetscCall(VecNorm(W, NORM_2, &a1));
    PetscCall(PetscInfo(snes, "|| J^T F|| %14.12e near zero implies found a local minimum\n", (double)(a1 / fnorm)));
    if (a1 / fnorm < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec         work;
    PetscScalar result;
    PetscReal   wnorm;

    PetscCall(VecSetRandom(W, NULL));
    PetscCall(VecNorm(W, NORM_2, &wnorm));
    PetscCall(VecDuplicate(W, &work));
    PetscCall(MatMult(A, W, work));
    PetscCall(VecDot(F, work, &result));
    PetscCall(VecDestroy(&work));
    a1 = PetscAbsScalar(result) / (fnorm * wnorm);
    PetscCall(PetscInfo(snes, "(F^T J random)/(|| F ||*||random|| %14.12e near zero implies found a local minimum\n", (double)a1));
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  PetscCall(VecDestroy(&W));
  PetscFunctionReturn(0);
}

/*
     Checks if J^T(F - J*X) = 0
*/
static PetscErrorCode SNESNEWTONLSCheckResidual_Private(SNES snes, Mat A, Vec F, Vec X)
{
  PetscReal a1, a2;
  PetscBool hastranspose;

  PetscFunctionBegin;
  PetscCall(MatHasOperation(A, MATOP_MULT_TRANSPOSE, &hastranspose));
  if (hastranspose) {
    Vec W1, W2;

    PetscCall(VecDuplicate(F, &W1));
    PetscCall(VecDuplicate(F, &W2));
    PetscCall(MatMult(A, X, W1));
    PetscCall(VecAXPY(W1, -1.0, F));

    /* Compute || J^T W|| */
    PetscCall(MatMultTranspose(A, W1, W2));
    PetscCall(VecNorm(W1, NORM_2, &a1));
    PetscCall(VecNorm(W2, NORM_2, &a2));
    if (a1 != 0.0) PetscCall(PetscInfo(snes, "||J^T(F-Ax)||/||F-AX|| %14.12e near zero implies inconsistent rhs\n", (double)(a2 / a1)));
    PetscCall(VecDestroy(&W1));
    PetscCall(VecDestroy(&W2));
  }
  PetscFunctionReturn(0);
}

/*
     This file implements a truncated Newton method with a line search,
     for solving a system of nonlinear equations, using the KSP, Vec,
     and Mat interfaces for linear solvers, vectors, and matrices,
     respectively.

     The following basic routines are required for each nonlinear solver:
          SNESCreate_XXX()          - Creates a nonlinear solver context
          SNESSetFromOptions_XXX()  - Sets runtime options
          SNESSolve_XXX()           - Solves the nonlinear system
          SNESDestroy_XXX()         - Destroys the nonlinear solver context
     The suffix "_XXX" denotes a particular implementation, in this case
     we use _NEWTONLS (e.g., SNESCreate_NEWTONLS, SNESSolve_NEWTONLS) for solving
     systems of nonlinear equations with a line search (LS) method.
     These routines are actually called via the common user interface
     routines SNESCreate(), SNESSetFromOptions(), SNESSolve(), and
     SNESDestroy(), so the application code interface remains identical
     for all nonlinear solvers.

     Another key routine is:
          SNESSetUp_XXX()           - Prepares for the use of a nonlinear solver
     by setting data structures and options.   The interface routine SNESSetUp()
     is not usually called directly by the user, but instead is called by
     SNESSolve() if necessary.

     Additional basic routines are:
          SNESView_XXX()            - Prints details of runtime options that
                                      have actually been used.
     These are called by application codes via the interface routines
     SNESView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.

*/
/*
   SNESSolve_NEWTONLS - Solves a nonlinear system with a truncated Newton
   method with a line search.

   Input Parameters:
.  snes - the SNES context

   Output Parameter:
.  outits - number of iterations until termination

   Application Interface Routine: SNESSolve()

*/
PetscErrorCode SNESSolve_NEWTONLS(SNES snes)
{
  PetscInt             maxits, i, lits;
  SNESLineSearchReason lssucceed;
  PetscReal            fnorm, gnorm, xnorm, ynorm;
  Vec                  Y, X, F;
  SNESLineSearch       linesearch;
  SNESConvergedReason  reason;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;        /* maximum number of iterations */
  X      = snes->vec_sol;        /* solution vector */
  F      = snes->vec_func;       /* residual vector */
  Y      = snes->vec_sol_update; /* newton step */

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESGetLineSearch(snes, &linesearch));

  /* compute the preconditioned function first in the case of left preconditioning with preconditioned function */
  if (snes->npc && snes->npcside == PC_LEFT && snes->functype == SNES_FUNCTION_PRECONDITIONED) {
    PetscCall(SNESApplyNPC(snes, X, NULL, F));
    PetscCall(SNESGetConvergedReason(snes->npc, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }

    PetscCall(VecNormBegin(F, NORM_2, &fnorm));
    PetscCall(VecNormEnd(F, NORM_2, &fnorm));
  } else {
    if (!snes->vec_func_init_set) {
      PetscCall(SNESComputeFunction(snes, X, F));
    } else snes->vec_func_init_set = PETSC_FALSE;
  }

  PetscCall(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
  SNESCheckFunctionNorm(snes, fnorm);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0));
  PetscCall(SNESMonitor(snes, 0, fnorm));

  /* test convergence */
  PetscUseTypeMethod(snes, converged, 0, 0.0, 0.0, fnorm, &snes->reason, snes->cnvP);
  if (snes->reason) PetscFunctionReturn(0);

  for (i = 0; i < maxits; i++) {
    /* Call general purpose update function */
    PetscTryTypeMethod(snes, update, snes->iter);

    /* apply the nonlinear preconditioner */
    if (snes->npc) {
      if (snes->npcside == PC_RIGHT) {
        PetscCall(SNESSetInitialFunction(snes->npc, F));
        PetscCall(PetscLogEventBegin(SNES_NPCSolve, snes->npc, X, snes->vec_rhs, 0));
        PetscCall(SNESSolve(snes->npc, snes->vec_rhs, X));
        PetscCall(PetscLogEventEnd(SNES_NPCSolve, snes->npc, X, snes->vec_rhs, 0));
        PetscCall(SNESGetConvergedReason(snes->npc, &reason));
        if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
          snes->reason = SNES_DIVERGED_INNER;
          PetscFunctionReturn(0);
        }
        PetscCall(SNESGetNPCFunction(snes, F, &fnorm));
      } else if (snes->npcside == PC_LEFT && snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
        PetscCall(SNESApplyNPC(snes, X, F, F));
        PetscCall(SNESGetConvergedReason(snes->npc, &reason));
        if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
          snes->reason = SNES_DIVERGED_INNER;
          PetscFunctionReturn(0);
        }
      }
    }

    /* Solve J Y = F, where J is Jacobian matrix */
    PetscCall(SNESComputeJacobian(snes, X, snes->jacobian, snes->jacobian_pre));
    SNESCheckJacobianDomainerror(snes);
    PetscCall(KSPSetOperators(snes->ksp, snes->jacobian, snes->jacobian_pre));
    PetscCall(KSPSolve(snes->ksp, F, Y));
    SNESCheckKSPSolve(snes);
    PetscCall(KSPGetIterationNumber(snes->ksp, &lits));
    PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", linear solve iterations=%" PetscInt_FMT "\n", snes->iter, lits));

    if (PetscLogPrintInfo) PetscCall(SNESNEWTONLSCheckResidual_Private(snes, snes->jacobian, F, Y));

    /* Compute a (scaled) negative update in the line search routine:
         X <- X - lambda*Y
       and evaluate F = function(X) (depends on the line search).
    */
    gnorm = fnorm;
    PetscCall(SNESLineSearchApply(linesearch, X, F, &fnorm, Y));
    PetscCall(SNESLineSearchGetReason(linesearch, &lssucceed));
    PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm));
    PetscCall(PetscInfo(snes, "fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n", (double)gnorm, (double)fnorm, (double)ynorm, (int)lssucceed));
    if (snes->reason) break;
    SNESCheckFunctionNorm(snes, fnorm);
    if (lssucceed) {
      if (snes->stol * xnorm > ynorm) {
        snes->reason = SNES_CONVERGED_SNORM_RELATIVE;
        PetscFunctionReturn(0);
      }
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscCall(SNESNEWTONLSCheckLocalMin_Private(snes, snes->jacobian, F, fnorm, &ismin));
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        if (snes->errorifnotconverged && snes->reason) {
          PetscViewer monitor;
          PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));
          PetscCheck(monitor, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged due to %s. Suggest running with -snes_linesearch_monitor", SNESConvergedReasons[snes->reason]);
          SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged due %s.", SNESConvergedReasons[snes->reason]);
        }
        break;
      }
    }
    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter  = i + 1;
    snes->norm  = fnorm;
    snes->ynorm = ynorm;
    snes->xnorm = xnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes, snes->norm, lits));
    PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
    /* Test for convergence */
    PetscUseTypeMethod(snes, converged, snes->iter, xnorm, ynorm, fnorm, &snes->reason, snes->cnvP);
    if (snes->reason) break;
  }
  if (i == maxits) {
    PetscCall(PetscInfo(snes, "Maximum number of iterations has been reached: %" PetscInt_FMT "\n", maxits));
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/*
   SNESSetUp_NEWTONLS - Sets up the internal data structures for the later use
   of the SNESNEWTONLS nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

 */
PetscErrorCode SNESSetUp_NEWTONLS(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESSetUpMatrices(snes));
  if (snes->npcside == PC_LEFT && snes->functype == SNES_FUNCTION_DEFAULT) snes->functype = SNES_FUNCTION_PRECONDITIONED;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESReset_NEWTONLS(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*
   SNESDestroy_NEWTONLS - Destroys the private SNES_NEWTONLS context that was created
   with SNESCreate_NEWTONLS().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
PetscErrorCode SNESDestroy_NEWTONLS(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESReset_NEWTONLS(snes));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

/*
   SNESView_NEWTONLS - Prints info from the SNESNEWTONLS data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Application Interface Routine: SNESView()
*/
static PetscErrorCode SNESView_NEWTONLS(SNES snes, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) { }
  PetscFunctionReturn(0);
}

/*
   SNESSetFromOptions_NEWTONLS - Sets various parameters for the SNESNEWTONLS method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
static PetscErrorCode SNESSetFromOptions_NEWTONLS(SNES snes, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
      SNESNEWTONLS - Newton based nonlinear solver that uses a line search

   Options Database Keys:
+   -snes_linesearch_type <bt> - bt,basic.  Select line search type
.   -snes_linesearch_order <3> - 2, 3. Selects the order of the line search for bt
.   -snes_linesearch_norms <true> - Turns on/off computation of the norms for basic linesearch (`SNESLineSearchSetComputeNorms()`)
.   -snes_linesearch_alpha <alpha> - Sets alpha used in determining if reduction in function norm is sufficient
.   -snes_linesearch_maxstep <maxstep> - Sets the maximum stepsize the line search will use (if the 2-norm(y) > maxstep then scale y to be y = (maxstep/2-norm(y)) *y)
.   -snes_linesearch_minlambda <minlambda>  - Sets the minimum lambda the line search will tolerate
.   -snes_linesearch_monitor - print information about progress of line searches
-   -snes_linesearch_damping - damping factor used for basic line search

    Note:
    This is the default nonlinear solver in `SNES`

   Level: beginner

.seealso: `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONTR`, `SNESQN`, `SNESLineSearchSetType()`, `SNESLineSearchSetOrder()`
          `SNESLineSearchSetPostCheck()`, `SNESLineSearchSetPreCheck()` `SNESLineSearchSetComputeNorms()`, `SNESGetLineSearch()`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONLS(SNES snes)
{
  SNES_NEWTONLS *neP;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONLS;
  snes->ops->solve          = SNESSolve_NEWTONLS;
  snes->ops->destroy        = SNESDestroy_NEWTONLS;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONLS;
  snes->ops->view           = SNESView_NEWTONLS;
  snes->ops->reset          = SNESReset_NEWTONLS;

  snes->npcside = PC_RIGHT;
  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_TRUE;

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHBT));

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNew(&neP));
  snes->data = (void *)neP;
  PetscFunctionReturn(0);
}
