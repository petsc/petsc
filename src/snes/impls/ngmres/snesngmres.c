#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>
#include <petscdm.h>

const char *const SNESNGMRESRestartTypes[] = {"NONE", "PERIODIC", "DIFFERENCE", "SNESNGMRESRestartType", "SNES_NGMRES_RESTART_", NULL};
const char *const SNESNGMRESSelectTypes[]  = {"NONE", "DIFFERENCE", "LINESEARCH", "SNESNGMRESSelectType", "SNES_NGMRES_SELECT_", NULL};

PetscErrorCode SNESReset_NGMRES(SNES snes)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(ngmres->msize, &ngmres->Fdot));
  PetscCall(VecDestroyVecs(ngmres->msize, &ngmres->Xdot));
  PetscCall(SNESLineSearchDestroy(&ngmres->additive_linesearch));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESDestroy_NGMRES(SNES snes)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  PetscCall(SNESReset_NGMRES(snes));
  PetscCall(PetscFree4(ngmres->h, ngmres->beta, ngmres->xi, ngmres->q));
  PetscCall(PetscFree3(ngmres->xnorms, ngmres->fnorms, ngmres->s));
  PetscCall(PetscFree3(ngmres->window, ngmres->Fwindow, ngmres->Xwindow));
#if PetscDefined(USE_COMPLEX)
  PetscCall(PetscFree(ngmres->rwork));
#endif
  PetscCall(PetscFree(ngmres->work));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetSelectType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetRestartType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetRestartFmRise_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESGetRestartFmRise_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetAdaptive_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESGetAdaptive_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetAdaptiveParameters_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESGetAdaptiveParameters_C", NULL));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESSetUp_NGMRES(SNES snes)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  PetscInt     msize, hsize;
  DM           dm;

  PetscFunctionBegin;
  PetscCheck(!snes->npc || snes->npcside != PC_LEFT || snes->functype != SNES_FUNCTION_UNPRECONDITIONED, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNESNGMRES does not support left preconditioning with unpreconditioned function");
  if (snes->npcside == PC_LEFT && snes->functype == SNES_FUNCTION_DEFAULT) snes->functype = SNES_FUNCTION_PRECONDITIONED;
  PetscCall(SNESSetWorkVecs(snes, 5));

  if (!snes->vec_sol) {
    PetscCall(SNESGetDM(snes, &dm));
    PetscCall(DMCreateGlobalVector(dm, &snes->vec_sol));
  }

  if (!ngmres->Xdot) PetscCall(VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->Xdot));
  if (!ngmres->Fdot) PetscCall(VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->Fdot));
  if (!ngmres->setup_called) {
    msize = ngmres->msize; /* restart size */
    hsize = msize * msize;

    /* explicit least squares minimization solve */
    PetscCall(PetscCalloc4(hsize, &ngmres->h, msize, &ngmres->beta, msize, &ngmres->xi, hsize, &ngmres->q));
    PetscCall(PetscMalloc3(msize, &ngmres->xnorms, msize, &ngmres->fnorms, msize, &ngmres->s));
    PetscCall(PetscMalloc3(msize, &ngmres->window, msize, &ngmres->Fwindow, msize, &ngmres->Xwindow));
    ngmres->nrhs = 1;
    PetscCall(PetscBLASIntCast(msize, &ngmres->lda));
    PetscCall(PetscBLASIntCast(msize, &ngmres->ldb));
    PetscCall(PetscBLASIntCast(12 * msize, &ngmres->lwork));
#if PetscDefined(USE_COMPLEX)
    PetscCall(PetscMalloc1(ngmres->lwork, &ngmres->rwork));
#endif
    PetscCall(PetscMalloc1(ngmres->lwork, &ngmres->work));
  }

  ngmres->setup_called = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetFromOptions_NGMRES(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  PetscBool    debug  = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNES NGMRES options");
  PetscCall(PetscOptionsEnum("-snes_ngmres_select_type", "Select type", "SNESNGMRESSetSelectType", SNESNGMRESSelectTypes, (PetscEnum)ngmres->select_type, (PetscEnum *)&ngmres->select_type, NULL));
  PetscCall(PetscOptionsEnum("-snes_ngmres_restart_type", "Restart type", "SNESNGMRESSetRestartType", SNESNGMRESRestartTypes, (PetscEnum)ngmres->restart_type, (PetscEnum *)&ngmres->restart_type, NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_candidate", "Use candidate storage", "SNES", ngmres->candidate, &ngmres->candidate, NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_approxfunc", "Linearly approximate the function", "SNES", ngmres->approxfunc, &ngmres->approxfunc, NULL));
  PetscCall(PetscOptionsInt("-snes_ngmres_m", "Number of directions", "SNES", ngmres->msize, &ngmres->msize, NULL));
  PetscCall(PetscOptionsInt("-snes_ngmres_restart", "Iterations before forced restart", "SNES", ngmres->restart_periodic, &ngmres->restart_periodic, NULL));
  PetscCall(PetscOptionsInt("-snes_ngmres_restart_it", "Tolerance iterations before restart", "SNES", ngmres->restart_it, &ngmres->restart_it, NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_monitor", "Monitor actions of NGMRES", "SNES", ngmres->monitor ? PETSC_TRUE : PETSC_FALSE, &debug, NULL));
  if (debug) ngmres->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));
  PetscCall(PetscOptionsReal("-snes_ngmres_gammaA", "Residual selection constant", "SNES", ngmres->gammaA, &ngmres->gammaA, NULL));
  PetscCall(PetscOptionsReal("-snes_ngmres_gammaC", "Residual restart constant", "SNES", ngmres->gammaC, &ngmres->gammaC, NULL));
  PetscCall(PetscOptionsReal("-snes_ngmres_epsilonB", "Difference selection constant", "SNES", ngmres->epsilonB, &ngmres->epsilonB, NULL));
  PetscCall(PetscOptionsReal("-snes_ngmres_deltaB", "Difference residual selection constant", "SNES", ngmres->deltaB, &ngmres->deltaB, NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_restart_fm_rise", "Restart on F_M residual rise", "SNESNGMRESSetRestartFmRise", ngmres->restart_fm_rise, &ngmres->restart_fm_rise, NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_adaptive", "Grow the window when theta predicts the residual reduction", "SNESNGMRESSetAdaptive", ngmres->adaptive, &ngmres->adaptive, NULL));
  PetscCall(PetscOptionsInt("-snes_ngmres_adaptive_m", "Initial number of stored solutions in the adaptive window", "SNESNGMRESSetAdaptiveParameters", ngmres->window_size_initial, &ngmres->window_size_initial, NULL));
  PetscCall(PetscOptionsReal("-snes_ngmres_adaptive_tol", "Tolerance on the prediction of the residual reduction that grows the window", "SNESNGMRESSetAdaptiveParameters", ngmres->adaptive_tol, &ngmres->adaptive_tol, NULL));
  PetscOptionsHeadEnd();
  if (ngmres->gammaA > ngmres->gammaC && ngmres->gammaC > 2.) ngmres->gammaC = ngmres->gammaA;
  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    PetscCall(SNESNGMRESGetAdditiveLineSearch_Private(snes, &ngmres->additive_linesearch));
    PetscCall(SNESLineSearchSetFromOptions(ngmres->additive_linesearch));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESView_NGMRES(SNES snes, PetscViewer viewer)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;
  PetscBool    isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Number of stored past updates: %" PetscInt_FMT "\n", ngmres->msize));
    if (ngmres->adaptive) PetscCall(PetscViewerASCIIPrintf(viewer, "  Adaptive window: initial size %" PetscInt_FMT ", tolerance %g\n", ngmres->window_size_initial, (double)ngmres->adaptive_tol));
    if (ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual selection: gammaA=%1.0e, gammaC=%1.0e\n", (double)ngmres->gammaA, (double)ngmres->gammaC));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Difference restart: epsilonB=%1.0e, deltaB=%1.0e\n", (double)ngmres->epsilonB, (double)ngmres->deltaB));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Restart on F_M residual increase: %s\n", PetscBools[ngmres->restart_fm_rise]));
    }
    if (ngmres->additive_linesearch) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Additive line-search details:\n"));
      PetscCall(SNESLineSearchView(ngmres->additive_linesearch, viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSolve_NGMRES(SNES snes)
{
  SNES_NGMRES        *ngmres = (SNES_NGMRES *)snes->data;
  Vec                 X, F, B, D, Y;         /* present solution, residual, and preconditioned residual */
  Vec                 XA, FA, XM, FM;        /* candidate linear combination answers */
  PetscReal           fnorm, fMnorm, fAnorm; /* coefficients and RHS to the minimization problem */
  PetscReal           fnormold, ratio;       /* residual reduction for the adaptive window */
  PetscReal           xnorm, xMnorm, xAnorm;
  PetscReal           ynorm, yMnorm, yAnorm;
  PetscInt            k, k_restart, l, ivec, restart_count = 0;
  PetscReal           objmin, objM, objA, obj; /* support for objective functions minimization */
  PetscBool           selectRestart;           /* solution selection data */
  SNESConvergedReason reason;
  SNESObjectiveFn    *objective;
  /*
      These two variables are initialized to prevent compilers/analyzers from producing false warnings about these variables being passed
      to SNESNGMRESSelect_Private() without being set when SNES_NGMRES_RESTART_DIFFERENCE, the values are not used in the subroutines in that case
      so the code is correct as written.
  */
  PetscReal dnorm = 0.0, dminnorm = 0.0;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  PetscCall(PetscCitationsRegister(SNESCitation, &SNEScite));
  /* variable initialization */
  snes->reason = SNES_CONVERGED_ITERATING;
  X            = snes->vec_sol;
  F            = snes->vec_func;
  B            = snes->vec_rhs;
  XA           = snes->work[2];
  FA           = snes->work[0];
  D            = snes->work[1];

  /* work for the line search */
  Y  = snes->vec_sol_update;
  XM = snes->work[3];
  FM = snes->work[4];

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  /* initialization */

  if (snes->npc && snes->npcside == PC_LEFT) {
    PetscCall(SNESApplyNPC(snes, X, NULL, F));
    PetscCall(SNESGetConvergedReason(snes->npc, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else {
    if (!snes->vec_func_init_set) PetscCall(SNESComputeFunction(snes, X, F));
    else snes->vec_func_init_set = PETSC_FALSE;
  }
  PetscCall(VecNorm(F, NORM_2, &fnorm));
  SNESCheckFunctionDomainError(snes, fnorm);
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  objmin = fnorm;
  if (objective) {
    PetscCall(SNESComputeObjective(snes, X, &objmin));
    SNESCheckObjectiveDomainError(snes, objmin);
  }
  obj = objmin;

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0));
  PetscCall(SNESConverged(snes, 0, 0.0, 0.0, fnorm));
  PetscCall(SNESMonitor(snes, 0, fnorm));
  if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(SNESNGMRESUpdateSubspace_Private(snes, 0, 0, F, fnorm, X));

  k_restart           = 1;
  l                   = 1;
  ivec                = 0;
  ngmres->window_size = ngmres->adaptive ? PetscMin(ngmres->window_size_initial, ngmres->msize) : ngmres->msize;
  for (k = 1; k < snes->max_its + 1; k++) {
    /* Call general purpose update function */
    PetscTryTypeMethod(snes, update, snes->iter);
    fnormold = fnorm;

    /* Computation of x^M */
    if (snes->npc && snes->npcside == PC_RIGHT) {
      PetscCall(VecCopy(X, XM));
      PetscCall(SNESSetInitialFunction(snes->npc, F));

      PetscCall(PetscLogEventBegin(SNES_NPCSolve, snes->npc, XM, B, 0));
      PetscCall(SNESSolve(snes->npc, B, XM));
      PetscCall(PetscLogEventEnd(SNES_NPCSolve, snes->npc, XM, B, 0));

      PetscCall(SNESGetConvergedReason(snes->npc, &reason));
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      PetscCall(SNESGetNPCFunction(snes, FM, &fMnorm));
    } else {
      /* no preconditioner -- just take gradient descent with line search */
      PetscCall(VecCopy(F, Y));
      PetscCall(VecCopy(F, FM));
      PetscCall(VecCopy(X, XM));

      fMnorm = fnorm;

      PetscCall(SNESLineSearchApply(snes->linesearch, XM, FM, &fMnorm, Y));
      if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (objective) {
      PetscCall(SNESComputeObjective(snes, XM, &objM));
      SNESCheckObjectiveDomainError(snes, objM);
    } else objM = fMnorm;
    objmin = PetscMin(objmin, objM);

    PetscCall(SNESNGMRESFormCombinedSolution_Private(snes, ivec, l, XM, FM, fMnorm, X, XA, FA));

    /* differences for selection and restart */
    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE || ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
      PetscCall(SNESNGMRESNorms_Private(snes, l, X, F, XM, FM, XA, FA, D, &dnorm, &dminnorm, &xMnorm, NULL, &yMnorm, &xAnorm, &fAnorm, &yAnorm));
    } else {
      PetscCall(SNESNGMRESNorms_Private(snes, l, X, F, XM, FM, XA, FA, D, NULL, NULL, &xMnorm, NULL, &yMnorm, &xAnorm, &fAnorm, &yAnorm));
    }
    SNESCheckFunctionDomainError(snes, fnorm);
    if (objective) {
      PetscCall(SNESComputeObjective(snes, XA, &objA));
      SNESCheckObjectiveDomainError(snes, objA);
    } else objA = fAnorm;

    /* combination (additive) or selection (multiplicative) of the N-GMRES solution */
    PetscCall(SNESNGMRESSelect_Private(snes, k_restart, XM, FM, xMnorm, fMnorm, yMnorm, objM, XA, FA, xAnorm, fAnorm, yAnorm, objA, dnorm, objmin, dminnorm, X, F, Y, &xnorm, &fnorm, &ynorm));
    SNESCheckFunctionDomainError(snes, fnorm);
    if (objective) {
      PetscCall(SNESComputeObjective(snes, X, &obj));
      SNESCheckObjectiveDomainError(snes, obj);
    } else obj = fnorm;

    // grow the window when theta predicts the residual reduction, which indicates that the higher-order terms are negligible
    if (ngmres->adaptive) {
      ratio         = fnormold > 0.0 ? fnorm / fnormold : 0.0;
      ngmres->theta = fnormold > 0.0 ? ngmres->lsnorm / fnormold : 0.0;
      if (PetscAbsReal(ngmres->theta - ratio) < ngmres->adaptive_tol && ngmres->window_size < ngmres->msize) ngmres->window_size++;
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "theta = %g, gamma = %g, ||F_k+1||/||F_k|| = %g, window size = %" PetscInt_FMT "\n", (double)ngmres->theta, (double)ngmres->gamma, (double)ratio, ngmres->window_size));
    }
    selectRestart = PETSC_FALSE;

    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE) {
      PetscCall(SNESNGMRESSelectRestart_Private(snes, l, obj, objM, objA, dnorm, objmin, dminnorm, &selectRestart));

      /* if the restart conditions persist for more than restart_it iterations, restart. */
      if (selectRestart) restart_count++;
      else restart_count = 0;
    } else if (ngmres->restart_type == SNES_NGMRES_RESTART_PERIODIC) {
      if (k_restart > ngmres->restart_periodic) {
        if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "periodic restart after %" PetscInt_FMT " iterations\n", k_restart));
        restart_count = ngmres->restart_it;
      }
    }

    ivec = k_restart % ngmres->msize; /* replace the last used part of the subspace */

    /* restart after restart conditions have persisted for a fixed number of iterations */
    if (restart_count >= ngmres->restart_it) {
      if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor, "Restarted at iteration %" PetscInt_FMT "\n", k_restart));
      restart_count = 0;
      k_restart     = 1;
      l             = 1;
      ivec          = 0;
      /* q_{00} = nu */
      PetscCall(SNESNGMRESUpdateSubspace_Private(snes, 0, 0, FM, fMnorm, XM));
    } else {
      /* select the current size of the subspace */
      if (l < ngmres->window_size) l++;
      k_restart++;
      /* place the current entry in the list of previous entries */
      if (ngmres->candidate) {
        objmin = PetscMin(objmin, objM);
        PetscCall(SNESNGMRESUpdateSubspace_Private(snes, ivec, l, FM, fMnorm, XM));
      } else {
        objmin = PetscMin(objmin, obj);
        PetscCall(SNESNGMRESUpdateSubspace_Private(snes, ivec, l, F, fnorm, X));
      }
    }

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter  = k;
    snes->norm  = fnorm;
    snes->ynorm = ynorm;
    snes->xnorm = xnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes, snes->norm, snes->iter));
    PetscCall(SNESConverged(snes, snes->iter, 0, 0, fnorm));
    PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
    if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESSetRestartFmRise - Increase the restart count if the step x_M increases the residual F_M inside a `SNESNGMRES` solve

  Input Parameters:
+ snes - the `SNES` context.
- flg  - boolean value deciding whether to use the option or not, default is `PETSC_FALSE`

  Options Database Key:
. -snes_ngmres_restart_fm_rise (true|false) - Increase the restart count if the step x_M increases the residual F_M

  Level: advanced

  Notes:
  If the proposed step x_M increases the residual F_M, it might be trying to get out of a stagnation area.
  To help the solver do that, remove the current stored solutions and residuals whenever F_M increases.

  This option must be used with the `SNESNGMRES` `SNESNGMRESRestartType` of `SNES_NGMRES_RESTART_DIFFERENCE`

.seealso: [](ch_snes), `SNES`, `SNES_NGMRES_RESTART_DIFFERENCE`, `SNESNGMRES`, `SNESNGMRESRestartType`, `SNESNGMRESSetRestartType()`
  @*/
PetscErrorCode SNESNGMRESSetRestartFmRise(SNES snes, PetscBool flg)
{
  PetscErrorCode (*f)(SNES, PetscBool);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes, "SNESNGMRESSetRestartFmRise_C", &f));
  if (f) PetscCall((f)(snes, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESSetRestartFmRise_NGMRES(SNES snes, PetscBool flg)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  ngmres->restart_fm_rise = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESGetRestartFmRise - Get whether `SNESNGMRES` increases the restart count when a step $x_M$ increases the residual $F_M$

  Not Collective

  Input Parameter:
. snes - the `SNES` context

  Output Parameter:
. flg - `PETSC_TRUE` if the option is enabled

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESNGMRES`, `SNESNGMRESSetRestartFmRise()`, `SNESNGMRESSetRestartType()`
@*/
PetscErrorCode SNESNGMRESGetRestartFmRise(SNES snes, PetscBool *flg)
{
  PetscErrorCode (*f)(SNES, PetscBool *);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)snes, "SNESNGMRESGetRestartFmRise_C", &f));
  if (f) PetscCall((f)(snes, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESGetRestartFmRise_NGMRES(SNES snes, PetscBool *flg)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  *flg = ngmres->restart_fm_rise;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESSetRestartType - Sets the restart type for `SNESNGMRES`.

  Logically Collective

  Input Parameters:
+ snes  - the iterative context
- rtype - restart type, see `SNESNGMRESRestartType`

  Options Database Keys:
+ -snes_ngmres_restart_type (difference|periodic|none) - set the restart type
- -snes_ngmres_restart restart                         - sets the number of iterations before restart for periodic

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNES_NGMRES_RESTART_DIFFERENCE`, `SNESNGMRES`, `SNESNGMRESRestartType`, `SNESNGMRESSetRestartFmRise()`,
          `SNESNGMRESSetSelectType()`
@*/
PetscErrorCode SNESNGMRESSetRestartType(SNES snes, SNESNGMRESRestartType rtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscTryMethod(snes, "SNESNGMRESSetRestartType_C", (SNES, SNESNGMRESRestartType), (snes, rtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESSetSelectType - Sets the selection type for `SNESNGMRES`.  This determines how the candidate solution and
  combined solution are used to create the next iterate.

  Logically Collective

  Input Parameters:
+ snes  - the iterative context
- stype - selection type, see `SNESNGMRESSelectType`

  Options Database Key:
. -snes_ngmres_select_type (difference|none|linesearch) - select type

  Level: intermediate

  Note:
  The default line search used is the `SNESLINESEARCHSECANT` line search and it requires two additional function evaluations.

.seealso: [](ch_snes), `SNES`, `SNESNGMRES`, `SNESNGMRESSelectType`, `SNES_NGMRES_SELECT_NONE`, `SNES_NGMRES_SELECT_DIFFERENCE`, `SNES_NGMRES_SELECT_LINESEARCH`,
          `SNESNGMRESSetRestartType()`
@*/
PetscErrorCode SNESNGMRESSetSelectType(SNES snes, SNESNGMRESSelectType stype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscTryMethod(snes, "SNESNGMRESSetSelectType_C", (SNES, SNESNGMRESSelectType), (snes, stype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESSetSelectType_NGMRES(SNES snes, SNESNGMRESSelectType stype)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  ngmres->select_type = stype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESSetRestartType_NGMRES(SNES snes, SNESNGMRESRestartType rtype)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  ngmres->restart_type = rtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESSetAdaptive - Sets whether `SNESNGMRES` grows the window of stored solutions during the solve

  Logically Collective

  Input Parameters:
+ snes - the `SNES` context
- flg  - `PETSC_TRUE` to grow the window adaptively

  Options Database Key:
. -snes_ngmres_adaptive (true|false) - grow the window adaptively

  Level: intermediate

  Notes:
  At each iteration $k$, `SNESNGMRES` combines the candidate solution $x_M$ with the stored solutions $x_j$ in a window, by minimizing
  the norm of the linear combination of their residuals. The ratio $\theta_{k+1}$ of the minimum to the residual norm $\| F(x_k) \|$
  predicts the residual reduction $\| F(x_{k+1}) \| / \| F(x_k) \|$ up to terms of higher order in the residual {cite}`he2026ngmres`.
  With the adaptive strategy, the window starts with a small number of stored solutions and grows by one whenever
  $| \theta_{k+1} - \| F(x_{k+1}) \| / \| F(x_k) \| |$ is below a tolerance, until it reaches the number set with `-snes_ngmres_m`.
  A small window limits the effect of the higher-order terms far from the solution, and a large window improves the rate of convergence
  close to the solution, where the higher-order terms are negligible.

  Use this strategy when a large fixed window is unreliable far from the solution but a larger history is useful near convergence.
  Enable it with `SNESNGMRESSetAdaptive()` or `-snes_ngmres_adaptive`, and choose the initial window size and growth tolerance
  with `SNESNGMRESSetAdaptiveParameters()` or `-snes_ngmres_adaptive_m` and `-snes_ngmres_adaptive_tol`.

  The strategy follows the algorithm of {cite}`he2026ngmres` with `SNES_NGMRES_SELECT_NONE` and `SNES_NGMRES_RESTART_NONE` or
  `SNES_NGMRES_RESTART_PERIODIC`.

.seealso: [](ch_snes), `SNES`, `SNESNGMRES`, `SNESNGMRESSetAdaptiveParameters()`, `SNESNGMRESSetSelectType()`,
          `SNESNGMRESSetRestartType()`
@*/
PetscErrorCode SNESNGMRESSetAdaptive(SNES snes, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveBool(snes, flg, 2);
  PetscTryMethod(snes, "SNESNGMRESSetAdaptive_C", (SNES, PetscBool), (snes, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESSetAdaptive_NGMRES(SNES snes, PetscBool flg)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  ngmres->adaptive = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESGetAdaptive - Gets whether `SNESNGMRES` grows the window of stored solutions during the solve

  Not Collective

  Input Parameter:
. snes - the `SNES` context

  Output Parameter:
. flg - `PETSC_TRUE` if the window grows adaptively

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESNGMRES`, `SNESNGMRESSetAdaptive()`, `SNESNGMRESSetAdaptiveParameters()`
@*/
PetscErrorCode SNESNGMRESGetAdaptive(SNES snes, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  PetscUseMethod(snes, "SNESNGMRESGetAdaptive_C", (SNES, PetscBool *), (snes, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESGetAdaptive_NGMRES(SNES snes, PetscBool *flg)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  *flg = ngmres->adaptive;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESSetAdaptiveParameters - Sets the parameters of the adaptive window of `SNESNGMRES`

  Logically Collective

  Input Parameters:
+ snes - the `SNES` context
. m    - the initial number of stored solutions in the window, or `PETSC_CURRENT` to keep the current value
- tol  - the tolerance on $| \theta_{k+1} - \| F(x_{k+1}) \| / \| F(x_k) \| |$ below which the window grows by one,
         or `PETSC_CURRENT` to keep the current value

  Options Database Keys:
+ -snes_ngmres_adaptive_m m     - the initial number of stored solutions, default 4
- -snes_ngmres_adaptive_tol tol - the tolerance, default 1e-3

  Level: intermediate

  Note:
  The number of stored solutions is one more than the depth $m$ of {cite}`he2026ngmres`, so the default corresponds to the depth 3.

.seealso: [](ch_snes), `SNES`, `SNESNGMRES`, `SNESNGMRESSetAdaptive()`, `SNESNGMRESGetAdaptive()`
@*/
PetscErrorCode SNESNGMRESSetAdaptiveParameters(SNES snes, PetscInt m, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveInt(snes, m, 2);
  PetscValidLogicalCollectiveReal(snes, tol, 3);
  PetscTryMethod(snes, "SNESNGMRESSetAdaptiveParameters_C", (SNES, PetscInt, PetscReal), (snes, m, tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESSetAdaptiveParameters_NGMRES(SNES snes, PetscInt m, PetscReal tol)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  if (m != PETSC_CURRENT) {
    PetscCheck(m > 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_OUTOFRANGE, "Initial window size %" PetscInt_FMT " must be positive", m);
    ngmres->window_size_initial = m;
  }
  if (tol != (PetscReal)PETSC_CURRENT) {
    PetscCheck(tol >= 0.0, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_OUTOFRANGE, "Tolerance %g must be nonnegative", (double)tol);
    ngmres->adaptive_tol = tol;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNGMRESGetAdaptiveParameters - Gets the parameters of the adaptive window of `SNESNGMRES`

  Not Collective

  Input Parameter:
. snes - the `SNES` context

  Output Parameters:
+ m   - the initial number of stored solutions in the window
- tol - the tolerance below which the window grows by one

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESNGMRES`, `SNESNGMRESSetAdaptiveParameters()`, `SNESNGMRESSetAdaptive()`
@*/
PetscErrorCode SNESNGMRESGetAdaptiveParameters(SNES snes, PetscInt *m, PetscReal *tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (m) PetscAssertPointer(m, 2);
  if (tol) PetscAssertPointer(tol, 3);
  PetscUseMethod(snes, "SNESNGMRESGetAdaptiveParameters_C", (SNES, PetscInt *, PetscReal *), (snes, m, tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNGMRESGetAdaptiveParameters_NGMRES(SNES snes, PetscInt *m, PetscReal *tol)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES *)snes->data;

  PetscFunctionBegin;
  if (m) *m = ngmres->window_size_initial;
  if (tol) *tol = ngmres->adaptive_tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  SNESNGMRES - An implementation of the Nonlinear Generalized Minimum Residual method, Nonlinear GMRES, or N-GMRES {cite}`ow1`, {cite}`bruneknepleysmithtu15` for solving
               nonlinear systems with `SNES`.

   Level: beginner

   Options Database Keys:
+  -snes_ngmres_select_type (difference|none|linesearch)             - Choose the select between candidate and combined solution
.  -snes_ngmres_restart_type (difference|none|periodic)              - Choose the restart conditions
.  -snes_ngmres_candidate                                            - Use `SNESNGMRES` variant which combines candidate solutions instead of actual solutions
.  -snes_ngmres_m m                                                  - Number of stored previous solutions and residuals
.  -snes_ngmres_restart_it it                                        - Number of iterations the restart conditions hold before restart
.  -snes_ngmres_gammaA gammaA                                        - Residual tolerance for solution select between the candidate and combination
.  -snes_ngmres_gammaC gammaB                                        - Residual tolerance for restart
.  -snes_ngmres_epsilonB epsilonB                                    - Difference tolerance between subsequent solutions triggering restart
.  -snes_ngmres_deltaB deltaB                                        - Difference tolerance between residuals triggering restart
.  -snes_ngmres_restart_fm_rise (true|false)                         - Restart on residual rise from $x_M$ step
.  -snes_ngmres_adaptive (true|false)                                - Grow the window of stored solutions when $\theta$ predicts the residual reduction, see `SNESNGMRESSetAdaptive()`
.  -snes_ngmres_adaptive_m m                                         - Initial number of stored solutions in the adaptive window
.  -snes_ngmres_adaptive_tol tol                                     - Tolerance on the prediction of the residual reduction that grows the window
.  -snes_ngmres_monitor                                              - Prints relevant information about the nonlinear GNMRES iterations
.  -snes_linesearch_type (none|bt|secant|cp|nleqerr|bisection|shell) - Line search type used for the default smoother, see `SNESLineSearchType`
-  -snes_ngmres_additive_snes_linesearch_type type                   - Line search type used to select between the candidate and combined solution with additive select type, see `SNESLineSearchType`

   Notes:
   The N-GMRES method combines m previous solutions into a minimum-residual solution by solving a small linearized
   optimization problem at each iteration.

   Very similar to the `SNESANDERSON` algorithm.

   Unlike the linear GMRES algorithm this algorithm does not compute a Krylov subspace using the Arnoldi process. Instead it stores a
   collection of previous solutions and the residuals $F(x) - b$ at those solutions.

   This algorithm ignores any Jacobian provided with `SNESSetJacobian()`

   Only supports left non-linear preconditioning.

.seealso: [](ch_snes), `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESType`, `SNESANDERSON`, `SNESNGMRESSetSelectType()`, `SNESNGMRESSetRestartType()`,
          `SNESNGMRESSetRestartFmRise()`, `SNESNGMRESSelectType`, `SNESNGMRESRestartType`
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NGMRES;
  snes->ops->setup          = SNESSetUp_NGMRES;
  snes->ops->setfromoptions = SNESSetFromOptions_NGMRES;
  snes->ops->view           = SNESView_NGMRES;
  snes->ops->solve          = SNESSolve_NGMRES;
  snes->ops->reset          = SNESReset_NGMRES;

  snes->usesnpc = PETSC_TRUE;
  snes->usesksp = PETSC_FALSE;
  snes->npcside = PC_RIGHT;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscObjectParameterSetDefault(snes, max_funcs, 30000);
  PetscObjectParameterSetDefault(snes, max_its, 10000);

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHNONE));

  PetscCall(PetscNew(&ngmres));
  snes->data = (void *)ngmres;

  ngmres->msize            = 30;
  ngmres->restart_it       = 2;
  ngmres->restart_periodic = 30;
  ngmres->gammaA           = 2.0;
  ngmres->gammaC           = 2.0;
  ngmres->deltaB           = 0.9;
  ngmres->epsilonB         = 0.1;
  ngmres->restart_type     = SNES_NGMRES_RESTART_DIFFERENCE;
  ngmres->select_type      = SNES_NGMRES_SELECT_DIFFERENCE;

  ngmres->window_size_initial = 4;
  ngmres->adaptive_tol        = 1.e-3;

  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetSelectType_C", SNESNGMRESSetSelectType_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetRestartType_C", SNESNGMRESSetRestartType_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetRestartFmRise_C", SNESNGMRESSetRestartFmRise_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESGetRestartFmRise_C", SNESNGMRESGetRestartFmRise_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetAdaptive_C", SNESNGMRESSetAdaptive_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESGetAdaptive_C", SNESNGMRESGetAdaptive_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESSetAdaptiveParameters_C", SNESNGMRESSetAdaptiveParameters_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNGMRESGetAdaptiveParameters_C", SNESNGMRESGetAdaptiveParameters_NGMRES));
  PetscFunctionReturn(PETSC_SUCCESS);
}
