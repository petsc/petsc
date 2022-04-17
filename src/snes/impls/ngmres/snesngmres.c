#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>
#include <petscdm.h>

const char *const SNESNGMRESRestartTypes[] = {"NONE","PERIODIC","DIFFERENCE","SNESNGMRESRestartType","SNES_NGMRES_RESTART_",NULL};
const char *const SNESNGMRESSelectTypes[] = {"NONE","DIFFERENCE","LINESEARCH","SNESNGMRESSelectType","SNES_NGMRES_SELECT_",NULL};

PetscErrorCode SNESReset_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;

  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(ngmres->msize,&ngmres->Fdot));
  PetscCall(VecDestroyVecs(ngmres->msize,&ngmres->Xdot));
  PetscCall(SNESLineSearchDestroy(&ngmres->additive_linesearch));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESDestroy_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  PetscCall(SNESReset_NGMRES(snes));
  PetscCall(PetscFree4(ngmres->h,ngmres->beta,ngmres->xi,ngmres->q));
  PetscCall(PetscFree3(ngmres->xnorms,ngmres->fnorms,ngmres->s));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree(ngmres->rwork));
#endif
  PetscCall(PetscFree(ngmres->work));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESSetUp_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  const char     *optionsprefix;
  PetscInt       msize,hsize;
  DM             dm;

  PetscFunctionBegin;
  if (snes->npc && snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
    SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"SNESNGMRES does not support left preconditioning with unpreconditioned function");
  }
  if (snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_DEFAULT) snes->functype = SNES_FUNCTION_PRECONDITIONED;
  PetscCall(SNESSetWorkVecs(snes,5));

  if (!snes->vec_sol) {
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(DMCreateGlobalVector(dm,&snes->vec_sol));
  }

  if (!ngmres->Xdot) PetscCall(VecDuplicateVecs(snes->vec_sol,ngmres->msize,&ngmres->Xdot));
  if (!ngmres->Fdot) PetscCall(VecDuplicateVecs(snes->vec_sol,ngmres->msize,&ngmres->Fdot));
  if (!ngmres->setup_called) {
    msize = ngmres->msize;          /* restart size */
    hsize = msize * msize;

    /* explicit least squares minimization solve */
    PetscCall(PetscCalloc4(hsize,&ngmres->h, msize,&ngmres->beta, msize,&ngmres->xi, hsize,&ngmres->q));
    PetscCall(PetscMalloc3(msize,&ngmres->xnorms,msize,&ngmres->fnorms,msize,&ngmres->s));
    ngmres->nrhs  = 1;
    ngmres->lda   = msize;
    ngmres->ldb   = msize;
    ngmres->lwork = 12*msize;
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscMalloc1(ngmres->lwork,&ngmres->rwork));
#endif
    PetscCall(PetscMalloc1(ngmres->lwork,&ngmres->work));
  }

  /* linesearch setup */
  PetscCall(SNESGetOptionsPrefix(snes,&optionsprefix));

  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    PetscCall(SNESLineSearchCreate(PetscObjectComm((PetscObject)snes),&ngmres->additive_linesearch));
    PetscCall(SNESLineSearchSetSNES(ngmres->additive_linesearch,snes));
    if (!((PetscObject)ngmres->additive_linesearch)->type_name) {
      PetscCall(SNESLineSearchSetType(ngmres->additive_linesearch,SNESLINESEARCHL2));
    }
    PetscCall(SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch,"additive_"));
    PetscCall(SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch,optionsprefix));
    PetscCall(SNESLineSearchSetFromOptions(ngmres->additive_linesearch));
  }

  ngmres->setup_called = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESSetFromOptions_NGMRES(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscBool      debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SNES NGMRES options");
  PetscCall(PetscOptionsEnum("-snes_ngmres_select_type","Select type","SNESNGMRESSetSelectType",SNESNGMRESSelectTypes,(PetscEnum)ngmres->select_type,(PetscEnum*)&ngmres->select_type,NULL));
  PetscCall(PetscOptionsEnum("-snes_ngmres_restart_type","Restart type","SNESNGMRESSetRestartType",SNESNGMRESRestartTypes,(PetscEnum)ngmres->restart_type,(PetscEnum*)&ngmres->restart_type,NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_candidate", "Use candidate storage",              "SNES",ngmres->candidate,&ngmres->candidate,NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_approxfunc","Linearly approximate the function", "SNES",ngmres->approxfunc,&ngmres->approxfunc,NULL));
  PetscCall(PetscOptionsInt("-snes_ngmres_m",          "Number of directions",               "SNES",ngmres->msize,&ngmres->msize,NULL));
  PetscCall(PetscOptionsInt("-snes_ngmres_restart",    "Iterations before forced restart",   "SNES",ngmres->restart_periodic,&ngmres->restart_periodic,NULL));
  PetscCall(PetscOptionsInt("-snes_ngmres_restart_it", "Tolerance iterations before restart","SNES",ngmres->restart_it,&ngmres->restart_it,NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_monitor",   "Monitor actions of NGMRES",          "SNES",ngmres->monitor ? PETSC_TRUE : PETSC_FALSE,&debug,NULL));
  if (debug) {
    ngmres->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));
  }
  PetscCall(PetscOptionsReal("-snes_ngmres_gammaA",    "Residual selection constant",   "SNES",ngmres->gammaA,&ngmres->gammaA,NULL));
  PetscCall(PetscOptionsReal("-snes_ngmres_gammaC",    "Residual restart constant",     "SNES",ngmres->gammaC,&ngmres->gammaC,NULL));
  PetscCall(PetscOptionsReal("-snes_ngmres_epsilonB",  "Difference selection constant", "SNES",ngmres->epsilonB,&ngmres->epsilonB,NULL));
  PetscCall(PetscOptionsReal("-snes_ngmres_deltaB",    "Difference residual selection constant", "SNES",ngmres->deltaB,&ngmres->deltaB,NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_single_reduction", "Aggregate reductions",  "SNES",ngmres->singlereduction,&ngmres->singlereduction,NULL));
  PetscCall(PetscOptionsBool("-snes_ngmres_restart_fm_rise", "Restart on F_M residual rise",  "SNESNGMRESSetRestartFmRise",ngmres->restart_fm_rise,&ngmres->restart_fm_rise,NULL));
  PetscOptionsHeadEnd();
  if ((ngmres->gammaA > ngmres->gammaC) && (ngmres->gammaC > 2.)) ngmres->gammaC = ngmres->gammaA;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESView_NGMRES(SNES snes,PetscViewer viewer)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Number of stored past updates: %" PetscInt_FMT "\n", ngmres->msize));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Residual selection: gammaA=%1.0e, gammaC=%1.0e\n",(double)ngmres->gammaA,(double)ngmres->gammaC));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Difference restart: epsilonB=%1.0e, deltaB=%1.0e\n",(double)ngmres->epsilonB,(double)ngmres->deltaB));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Restart on F_M residual increase: %s\n",PetscBools[ngmres->restart_fm_rise]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESSolve_NGMRES(SNES snes)
{
  SNES_NGMRES          *ngmres = (SNES_NGMRES*) snes->data;
  /* present solution, residual, and preconditioned residual */
  Vec                  X,F,B,D,Y;

  /* candidate linear combination answers */
  Vec                  XA,FA,XM,FM;

  /* coefficients and RHS to the minimization problem */
  PetscReal            fnorm,fMnorm,fAnorm;
  PetscReal            xnorm,xMnorm,xAnorm;
  PetscReal            ynorm,yMnorm,yAnorm;
  PetscInt             k,k_restart,l,ivec,restart_count = 0;

  /* solution selection data */
  PetscBool            selectRestart;
  /*
      These two variables are initialized to prevent compilers/analyzers from producing false warnings about these variables being passed
      to SNESNGMRESSelect_Private() without being set when SNES_NGMRES_RESTART_DIFFERENCE, the values are not used in the subroutines in that case
      so the code is correct as written.
  */
  PetscReal            dnorm = 0.0,dminnorm = 0.0;
  PetscReal            fminnorm;

  SNESConvergedReason  reason;
  SNESLineSearchReason lssucceed;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  PetscCall(PetscCitationsRegister(SNESCitation,&SNEScite));
  /* variable initialization */
  snes->reason = SNES_CONVERGED_ITERATING;
  X            = snes->vec_sol;
  F            = snes->vec_func;
  B            = snes->vec_rhs;
  XA           = snes->vec_sol_update;
  FA           = snes->work[0];
  D            = snes->work[1];

  /* work for the line search */
  Y  = snes->work[2];
  XM = snes->work[3];
  FM = snes->work[4];

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  /* initialization */

  if (snes->npc && snes->npcside== PC_LEFT) {
    PetscCall(SNESApplyNPC(snes,X,NULL,F));
    PetscCall(SNESGetConvergedReason(snes->npc,&reason));
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    PetscCall(VecNorm(F,NORM_2,&fnorm));
  } else {
    if (!snes->vec_func_init_set) {
      PetscCall(SNESComputeFunction(snes,X,F));
    } else snes->vec_func_init_set = PETSC_FALSE;

    PetscCall(VecNorm(F,NORM_2,&fnorm));
    SNESCheckFunctionNorm(snes,fnorm);
  }
  fminnorm = fnorm;

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes,fnorm,0));
  PetscCall(SNESMonitor(snes,0,fnorm));
  PetscCall((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  if (snes->reason) PetscFunctionReturn(0);
  SNESNGMRESUpdateSubspace_Private(snes,0,0,F,fnorm,X);

  k_restart = 1;
  l         = 1;
  ivec      = 0;
  for (k=1; k < snes->max_its+1; k++) {
    /* Computation of x^M */
    if (snes->npc && snes->npcside== PC_RIGHT) {
      PetscCall(VecCopy(X,XM));
      PetscCall(SNESSetInitialFunction(snes->npc,F));

      PetscCall(PetscLogEventBegin(SNES_NPCSolve,snes->npc,XM,B,0));
      PetscCall(SNESSolve(snes->npc,B,XM));
      PetscCall(PetscLogEventEnd(SNES_NPCSolve,snes->npc,XM,B,0));

      PetscCall(SNESGetConvergedReason(snes->npc,&reason));
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      PetscCall(SNESGetNPCFunction(snes,FM,&fMnorm));
    } else {
      /* no preconditioner -- just take gradient descent with line search */
      PetscCall(VecCopy(F,Y));
      PetscCall(VecCopy(F,FM));
      PetscCall(VecCopy(X,XM));

      fMnorm = fnorm;

      PetscCall(SNESLineSearchApply(snes->linesearch,XM,FM,&fMnorm,Y));
      PetscCall(SNESLineSearchGetReason(snes->linesearch,&lssucceed));
      if (lssucceed) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(0);
        }
      }
    }

    PetscCall(SNESNGMRESFormCombinedSolution_Private(snes,ivec,l,XM,FM,fMnorm,X,XA,FA));
    /* r = F(x) */
    if (fminnorm > fMnorm) fminnorm = fMnorm;  /* the minimum norm is now of F^M */

    /* differences for selection and restart */
    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE || ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
      PetscCall(SNESNGMRESNorms_Private(snes,l,X,F,XM,FM,XA,FA,D,&dnorm,&dminnorm,&xMnorm,NULL,&yMnorm,&xAnorm,&fAnorm,&yAnorm));
    } else {
      PetscCall(SNESNGMRESNorms_Private(snes,l,X,F,XM,FM,XA,FA,D,NULL,NULL,&xMnorm,NULL,&yMnorm,&xAnorm,&fAnorm,&yAnorm));
    }
    SNESCheckFunctionNorm(snes,fnorm);

    /* combination (additive) or selection (multiplicative) of the N-GMRES solution */
    PetscCall(SNESNGMRESSelect_Private(snes,k_restart,XM,FM,xMnorm,fMnorm,yMnorm,XA,FA,xAnorm,fAnorm,yAnorm,dnorm,fminnorm,dminnorm,X,F,Y,&xnorm,&fnorm,&ynorm));
    selectRestart = PETSC_FALSE;

    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE) {
      PetscCall(SNESNGMRESSelectRestart_Private(snes,l,fMnorm,fAnorm,dnorm,fminnorm,dminnorm,&selectRestart));

      /* if the restart conditions persist for more than restart_it iterations, restart. */
      if (selectRestart) restart_count++;
      else restart_count = 0;
    } else if (ngmres->restart_type == SNES_NGMRES_RESTART_PERIODIC) {
      if (k_restart > ngmres->restart_periodic) {
        if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"periodic restart after %" PetscInt_FMT " iterations\n",k_restart));
        restart_count = ngmres->restart_it;
      }
    }

    ivec = k_restart % ngmres->msize; /* replace the last used part of the subspace */

    /* restart after restart conditions have persisted for a fixed number of iterations */
    if (restart_count >= ngmres->restart_it) {
      if (ngmres->monitor) {
        PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"Restarted at iteration %" PetscInt_FMT "\n",k_restart));
      }
      restart_count = 0;
      k_restart     = 1;
      l             = 1;
      ivec          = 0;
      /* q_{00} = nu */
      PetscCall(SNESNGMRESUpdateSubspace_Private(snes,0,0,FM,fMnorm,XM));
    } else {
      /* select the current size of the subspace */
      if (l < ngmres->msize) l++;
      k_restart++;
      /* place the current entry in the list of previous entries */
      if (ngmres->candidate) {
        if (fminnorm > fMnorm) fminnorm = fMnorm;
        PetscCall(SNESNGMRESUpdateSubspace_Private(snes,ivec,l,FM,fMnorm,XM));
      } else {
        if (fminnorm > fnorm) fminnorm = fnorm;
        PetscCall(SNESNGMRESUpdateSubspace_Private(snes,ivec,l,F,fnorm,X));
      }
    }

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = k;
    snes->norm = fnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,snes->iter));
    PetscCall(SNESMonitor(snes,snes->iter,snes->norm));
    PetscCall((*snes->ops->converged)(snes,snes->iter,0,0,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) PetscFunctionReturn(0);
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*@
 SNESNGMRESSetRestartFmRise - Increase the restart count if the step x_M increases the residual F_M

  Input Parameters:
  +  snes - the SNES context.
  -  flg  - boolean value deciding whether to use the option or not

  Options Database:
  + -snes_ngmres_restart_fm_rise - Increase the restart count if the step x_M increases the residual F_M

  Level: intermediate

  Notes:
  If the proposed step x_M increases the residual F_M, it might be trying to get out of a stagnation area.
  To help the solver do that, reset the Krylov subspace whenever F_M increases.

  This option must be used with SNES_NGMRES_RESTART_DIFFERENCE

  The default is FALSE.
  .seealso: SNES_NGMRES_RESTART_DIFFERENCE
  @*/
PetscErrorCode SNESNGMRESSetRestartFmRise(SNES snes,PetscBool flg)
{
    PetscErrorCode (*f)(SNES,PetscBool);

    PetscFunctionBegin;
    PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNGMRESSetRestartFmRise_C",&f));
    if (f) PetscCall((f)(snes,flg));
    PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESSetRestartFmRise_NGMRES(SNES snes,PetscBool flg)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  ngmres->restart_fm_rise = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESGetRestartFmRise(SNES snes,PetscBool *flg)
{
    PetscErrorCode (*f)(SNES,PetscBool*);

    PetscFunctionBegin;
    PetscCall(PetscObjectQueryFunction((PetscObject)snes,"SNESNGMRESGetRestartFmRise_C",&f));
    if (f) PetscCall((f)(snes,flg));
    PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESGetRestartFmRise_NGMRES(SNES snes,PetscBool *flg)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  *flg = ngmres->restart_fm_rise;
  PetscFunctionReturn(0);
}

/*@
    SNESNGMRESSetRestartType - Sets the restart type for SNESNGMRES.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   rtype - restart type

    Options Database:
+   -snes_ngmres_restart_type<difference,periodic,none> - set the restart type
-   -snes_ngmres_restart[30] - sets the number of iterations before restart for periodic

    Level: intermediate

    SNESNGMRESRestartTypes:
+   SNES_NGMRES_RESTART_NONE - never restart
.   SNES_NGMRES_RESTART_DIFFERENCE - restart based upon difference criteria
-   SNES_NGMRES_RESTART_PERIODIC - restart after a fixed number of iterations

    Notes:
    The default line search used is the L2 line search and it requires two additional function evaluations.

@*/
PetscErrorCode SNESNGMRESSetRestartType(SNES snes,SNESNGMRESRestartType rtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscTryMethod(snes,"SNESNGMRESSetRestartType_C",(SNES,SNESNGMRESRestartType),(snes,rtype));
  PetscFunctionReturn(0);
}

/*@
    SNESNGMRESSetSelectType - Sets the selection type for SNESNGMRES.  This determines how the candidate solution and
    combined solution are used to create the next iterate.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   stype - selection type

    Options Database:
.   -snes_ngmres_select_type<difference,none,linesearch> - select type

    Level: intermediate

    SNESNGMRESSelectTypes:
+   SNES_NGMRES_SELECT_NONE - choose the combined solution all the time
.   SNES_NGMRES_SELECT_DIFFERENCE - choose based upon the selection criteria
-   SNES_NGMRES_SELECT_LINESEARCH - choose based upon line search combination

    Notes:
    The default line search used is the L2 line search and it requires two additional function evaluations.

@*/
PetscErrorCode SNESNGMRESSetSelectType(SNES snes,SNESNGMRESSelectType stype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscTryMethod(snes,"SNESNGMRESSetSelectType_C",(SNES,SNESNGMRESSelectType),(snes,stype));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESSetSelectType_NGMRES(SNES snes,SNESNGMRESSelectType stype)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  ngmres->select_type = stype;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESSetRestartType_NGMRES(SNES snes,SNESNGMRESRestartType rtype)
{
  SNES_NGMRES *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  ngmres->restart_type = rtype;
  PetscFunctionReturn(0);
}

/*MC
  SNESNGMRES - The Nonlinear Generalized Minimum Residual method.

   Level: beginner

   Options Database:
+  -snes_ngmres_select_type<difference,none,linesearch> - choose the select between candidate and combined solution
.  -snes_ngmres_restart_type<difference,none,periodic> - choose the restart conditions
.  -snes_ngmres_candidate        - Use NGMRES variant which combines candidate solutions instead of actual solutions
.  -snes_ngmres_m                - Number of stored previous solutions and residuals
.  -snes_ngmres_restart_it       - Number of iterations the restart conditions hold before restart
.  -snes_ngmres_gammaA           - Residual tolerance for solution select between the candidate and combination
.  -snes_ngmres_gammaC           - Residual tolerance for restart
.  -snes_ngmres_epsilonB         - Difference tolerance between subsequent solutions triggering restart
.  -snes_ngmres_deltaB           - Difference tolerance between residuals triggering restart
.  -snes_ngmres_restart_fm_rise  - Restart on residual rise from x_M step
.  -snes_ngmres_monitor          - Prints relevant information about the ngmres iteration
.  -snes_linesearch_type <basic,l2,cp> - Line search type used for the default smoother
-  -additive_snes_linesearch_type - linesearch type used to select between the candidate and combined solution with additive select type

   Notes:

   The N-GMRES method combines m previous solutions into a minimum-residual solution by solving a small linearized
   optimization problem at each iteration.

   Very similar to the SNESANDERSON algorithm.

   References:
+  * - C. W. Oosterlee and T. Washio, "Krylov Subspace Acceleration of Nonlinear Multigrid with Application to Recirculating Flows",
   SIAM Journal on Scientific Computing, 21(5), 2000.
-  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NGMRES;
  snes->ops->setup          = SNESSetUp_NGMRES;
  snes->ops->setfromoptions = SNESSetFromOptions_NGMRES;
  snes->ops->view           = SNESView_NGMRES;
  snes->ops->solve          = SNESSolve_NGMRES;
  snes->ops->reset          = SNESReset_NGMRES;

  snes->usesnpc  = PETSC_TRUE;
  snes->usesksp  = PETSC_FALSE;
  snes->npcside  = PC_RIGHT;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNewLog(snes,&ngmres));
  snes->data    = (void*) ngmres;
  ngmres->msize = 30;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  ngmres->candidate = PETSC_FALSE;

  PetscCall(SNESGetLineSearch(snes,&linesearch));
  if (!((PetscObject)linesearch)->type_name) {
    PetscCall(SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC));
  }

  ngmres->additive_linesearch = NULL;
  ngmres->approxfunc          = PETSC_FALSE;
  ngmres->restart_it          = 2;
  ngmres->restart_periodic    = 30;
  ngmres->gammaA              = 2.0;
  ngmres->gammaC              = 2.0;
  ngmres->deltaB              = 0.9;
  ngmres->epsilonB            = 0.1;
  ngmres->restart_fm_rise     = PETSC_FALSE;

  ngmres->restart_type = SNES_NGMRES_RESTART_DIFFERENCE;
  ngmres->select_type  = SNES_NGMRES_SELECT_DIFFERENCE;

  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESSetSelectType_C",SNESNGMRESSetSelectType_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESSetRestartType_C",SNESNGMRESSetRestartType_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESSetRestartFmRise_C",SNESNGMRESSetRestartFmRise_NGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESGetRestartFmRise_C",SNESNGMRESGetRestartFmRise_NGMRES));
  PetscFunctionReturn(0);
}
