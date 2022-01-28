#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>
#include <petscdm.h>

const char *const SNESNGMRESRestartTypes[] = {"NONE","PERIODIC","DIFFERENCE","SNESNGMRESRestartType","SNES_NGMRES_RESTART_",NULL};
const char *const SNESNGMRESSelectTypes[] = {"NONE","DIFFERENCE","LINESEARCH","SNESNGMRESSelectType","SNES_NGMRES_SELECT_",NULL};

PetscErrorCode SNESReset_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ngmres->msize,&ngmres->Fdot);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ngmres->msize,&ngmres->Xdot);CHKERRQ(ierr);
  ierr = SNESLineSearchDestroy(&ngmres->additive_linesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESDestroy_NGMRES(SNES snes)
{
  PetscErrorCode ierr;
  SNES_NGMRES    *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  ierr = SNESReset_NGMRES(snes);CHKERRQ(ierr);
  ierr = PetscFree4(ngmres->h,ngmres->beta,ngmres->xi,ngmres->q);CHKERRQ(ierr);
  ierr = PetscFree3(ngmres->xnorms,ngmres->fnorms,ngmres->s);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(ngmres->rwork);CHKERRQ(ierr);
#endif
  ierr = PetscFree(ngmres->work);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESSetUp_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  const char     *optionsprefix;
  PetscInt       msize,hsize;
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  if (snes->npc && snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
    SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"SNESNGMRES does not support left preconditioning with unpreconditioned function");
  }
  if (snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_DEFAULT) snes->functype = SNES_FUNCTION_PRECONDITIONED;
  ierr = SNESSetWorkVecs(snes,5);CHKERRQ(ierr);

  if (!snes->vec_sol) {
    ierr             = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr             = DMCreateGlobalVector(dm,&snes->vec_sol);CHKERRQ(ierr);
  }

  if (!ngmres->Xdot) {ierr = VecDuplicateVecs(snes->vec_sol,ngmres->msize,&ngmres->Xdot);CHKERRQ(ierr);}
  if (!ngmres->Fdot) {ierr = VecDuplicateVecs(snes->vec_sol,ngmres->msize,&ngmres->Fdot);CHKERRQ(ierr);}
  if (!ngmres->setup_called) {
    msize = ngmres->msize;          /* restart size */
    hsize = msize * msize;

    /* explicit least squares minimization solve */
    ierr = PetscCalloc4(hsize,&ngmres->h, msize,&ngmres->beta, msize,&ngmres->xi, hsize,&ngmres->q);CHKERRQ(ierr);
    ierr = PetscMalloc3(msize,&ngmres->xnorms,msize,&ngmres->fnorms,msize,&ngmres->s);CHKERRQ(ierr);
    ngmres->nrhs  = 1;
    ngmres->lda   = msize;
    ngmres->ldb   = msize;
    ngmres->lwork = 12*msize;
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc1(ngmres->lwork,&ngmres->rwork);CHKERRQ(ierr);
#endif
    ierr = PetscMalloc1(ngmres->lwork,&ngmres->work);CHKERRQ(ierr);
  }

  /* linesearch setup */
  ierr = SNESGetOptionsPrefix(snes,&optionsprefix);CHKERRQ(ierr);

  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    ierr = SNESLineSearchCreate(PetscObjectComm((PetscObject)snes),&ngmres->additive_linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSNES(ngmres->additive_linesearch,snes);CHKERRQ(ierr);
    if (!((PetscObject)ngmres->additive_linesearch)->type_name) {
      ierr = SNESLineSearchSetType(ngmres->additive_linesearch,SNESLINESEARCHL2);CHKERRQ(ierr);
    }
    ierr = SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch,"additive_");CHKERRQ(ierr);
    ierr = SNESLineSearchAppendOptionsPrefix(ngmres->additive_linesearch,optionsprefix);CHKERRQ(ierr);
    ierr = SNESLineSearchSetFromOptions(ngmres->additive_linesearch);CHKERRQ(ierr);
  }

  ngmres->setup_called = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESSetFromOptions_NGMRES(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscErrorCode ierr;
  PetscBool      debug = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SNES NGMRES options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_ngmres_select_type","Select type","SNESNGMRESSetSelectType",SNESNGMRESSelectTypes,
                          (PetscEnum)ngmres->select_type,(PetscEnum*)&ngmres->select_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_ngmres_restart_type","Restart type","SNESNGMRESSetRestartType",SNESNGMRESRestartTypes,
                          (PetscEnum)ngmres->restart_type,(PetscEnum*)&ngmres->restart_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_candidate", "Use candidate storage",              "SNES",ngmres->candidate,&ngmres->candidate,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_approxfunc","Linearly approximate the function", "SNES",ngmres->approxfunc,&ngmres->approxfunc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_m",          "Number of directions",               "SNES",ngmres->msize,&ngmres->msize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_restart",    "Iterations before forced restart",   "SNES",ngmres->restart_periodic,&ngmres->restart_periodic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_restart_it", "Tolerance iterations before restart","SNES",ngmres->restart_it,&ngmres->restart_it,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_monitor",   "Monitor actions of NGMRES",          "SNES",ngmres->monitor ? PETSC_TRUE : PETSC_FALSE,&debug,NULL);CHKERRQ(ierr);
  if (debug) {
    ngmres->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));
  }
  ierr = PetscOptionsReal("-snes_ngmres_gammaA",    "Residual selection constant",   "SNES",ngmres->gammaA,&ngmres->gammaA,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_gammaC",    "Residual restart constant",     "SNES",ngmres->gammaC,&ngmres->gammaC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_epsilonB",  "Difference selection constant", "SNES",ngmres->epsilonB,&ngmres->epsilonB,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_deltaB",    "Difference residual selection constant", "SNES",ngmres->deltaB,&ngmres->deltaB,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_single_reduction", "Aggregate reductions",  "SNES",ngmres->singlereduction,&ngmres->singlereduction,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_restart_fm_rise", "Restart on F_M residual rise",  "SNESNGMRESSetRestartFmRise",ngmres->restart_fm_rise,&ngmres->restart_fm_rise,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ngmres->gammaA > ngmres->gammaC) && (ngmres->gammaC > 2.)) ngmres->gammaC = ngmres->gammaA;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESView_NGMRES(SNES snes,PetscViewer viewer)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Number of stored past updates: %d\n", ngmres->msize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual selection: gammaA=%1.0e, gammaC=%1.0e\n",ngmres->gammaA,ngmres->gammaC);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Difference restart: epsilonB=%1.0e, deltaB=%1.0e\n",ngmres->epsilonB,ngmres->deltaB);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Restart on F_M residual increase: %s\n",ngmres->restart_fm_rise?"TRUE":"FALSE");CHKERRQ(ierr);
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
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  PetscAssertFalse(snes->xl || snes->xu || snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  ierr = PetscCitationsRegister(SNESCitation,&SNEScite);CHKERRQ(ierr);
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

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  /* initialization */

  if (snes->npc && snes->npcside== PC_LEFT) {
    ierr = SNESApplyNPC(snes,X,NULL,F);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->npc,&reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
  } else {
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    } else snes->vec_func_init_set = PETSC_FALSE;

    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    SNESCheckFunctionNorm(snes,fnorm);
  }
  fminnorm = fnorm;

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);
  ierr       = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);
  SNESNGMRESUpdateSubspace_Private(snes,0,0,F,fnorm,X);

  k_restart = 1;
  l         = 1;
  ivec      = 0;
  for (k=1; k < snes->max_its+1; k++) {
    /* Computation of x^M */
    if (snes->npc && snes->npcside== PC_RIGHT) {
      ierr = VecCopy(X,XM);CHKERRQ(ierr);
      ierr = SNESSetInitialFunction(snes->npc,F);CHKERRQ(ierr);

      ierr = PetscLogEventBegin(SNES_NPCSolve,snes->npc,XM,B,0);CHKERRQ(ierr);
      ierr = SNESSolve(snes->npc,B,XM);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(SNES_NPCSolve,snes->npc,XM,B,0);CHKERRQ(ierr);

      ierr = SNESGetConvergedReason(snes->npc,&reason);CHKERRQ(ierr);
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = SNESGetNPCFunction(snes,FM,&fMnorm);CHKERRQ(ierr);
    } else {
      /* no preconditioner -- just take gradient descent with line search */
      ierr = VecCopy(F,Y);CHKERRQ(ierr);
      ierr = VecCopy(F,FM);CHKERRQ(ierr);
      ierr = VecCopy(X,XM);CHKERRQ(ierr);

      fMnorm = fnorm;

      ierr = SNESLineSearchApply(snes->linesearch,XM,FM,&fMnorm,Y);CHKERRQ(ierr);
      ierr = SNESLineSearchGetReason(snes->linesearch,&lssucceed);CHKERRQ(ierr);
      if (lssucceed) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(0);
        }
      }
    }

    ierr = SNESNGMRESFormCombinedSolution_Private(snes,ivec,l,XM,FM,fMnorm,X,XA,FA);CHKERRQ(ierr);
    /* r = F(x) */
    if (fminnorm > fMnorm) fminnorm = fMnorm;  /* the minimum norm is now of F^M */

    /* differences for selection and restart */
    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE || ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
      ierr = SNESNGMRESNorms_Private(snes,l,X,F,XM,FM,XA,FA,D,&dnorm,&dminnorm,&xMnorm,NULL,&yMnorm,&xAnorm,&fAnorm,&yAnorm);CHKERRQ(ierr);
    } else {
      ierr = SNESNGMRESNorms_Private(snes,l,X,F,XM,FM,XA,FA,D,NULL,NULL,&xMnorm,NULL,&yMnorm,&xAnorm,&fAnorm,&yAnorm);CHKERRQ(ierr);
    }
    SNESCheckFunctionNorm(snes,fnorm);

    /* combination (additive) or selection (multiplicative) of the N-GMRES solution */
    ierr          = SNESNGMRESSelect_Private(snes,k_restart,XM,FM,xMnorm,fMnorm,yMnorm,XA,FA,xAnorm,fAnorm,yAnorm,dnorm,fminnorm,dminnorm,X,F,Y,&xnorm,&fnorm,&ynorm);CHKERRQ(ierr);
    selectRestart = PETSC_FALSE;

    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE) {
      ierr = SNESNGMRESSelectRestart_Private(snes,l,fMnorm,fAnorm,dnorm,fminnorm,dminnorm,&selectRestart);CHKERRQ(ierr);

      /* if the restart conditions persist for more than restart_it iterations, restart. */
      if (selectRestart) restart_count++;
      else restart_count = 0;
    } else if (ngmres->restart_type == SNES_NGMRES_RESTART_PERIODIC) {
      if (k_restart > ngmres->restart_periodic) {
        if (ngmres->monitor) {ierr = PetscViewerASCIIPrintf(ngmres->monitor,"periodic restart after %D iterations\n",k_restart);CHKERRQ(ierr);}
        restart_count = ngmres->restart_it;
      }
    }

    ivec = k_restart % ngmres->msize; /* replace the last used part of the subspace */

    /* restart after restart conditions have persisted for a fixed number of iterations */
    if (restart_count >= ngmres->restart_it) {
      if (ngmres->monitor) {
        ierr = PetscViewerASCIIPrintf(ngmres->monitor,"Restarted at iteration %d\n",k_restart);CHKERRQ(ierr);
      }
      restart_count = 0;
      k_restart     = 1;
      l             = 1;
      ivec          = 0;
      /* q_{00} = nu */
      ierr = SNESNGMRESUpdateSubspace_Private(snes,0,0,FM,fMnorm,XM);CHKERRQ(ierr);
    } else {
      /* select the current size of the subspace */
      if (l < ngmres->msize) l++;
      k_restart++;
      /* place the current entry in the list of previous entries */
      if (ngmres->candidate) {
        if (fminnorm > fMnorm) fminnorm = fMnorm;
        ierr = SNESNGMRESUpdateSubspace_Private(snes,ivec,l,FM,fMnorm,XM);CHKERRQ(ierr);
      } else {
        if (fminnorm > fnorm) fminnorm = fnorm;
        ierr = SNESNGMRESUpdateSubspace_Private(snes,ivec,l,F,fnorm,X);CHKERRQ(ierr);
      }
    }

    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = k;
    snes->norm = fnorm;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr = SNESLogConvergenceHistory(snes,snes->norm,snes->iter);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    ierr = (*snes->ops->converged)(snes,snes->iter,0,0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
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
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNGMRESSetRestartFmRise_C",&f);CHKERRQ(ierr);
    if (f) {ierr = (f)(snes,flg);CHKERRQ(ierr);}
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
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNGMRESGetRestartFmRise_C",&f);CHKERRQ(ierr);
    if (f) {ierr = (f)(snes,flg);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESNGMRESSetRestartType_C",(SNES,SNESNGMRESRestartType),(snes,rtype));CHKERRQ(ierr);
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
.   -snes_ngmres_select_type<difference,none,linesearch>

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESNGMRESSetSelectType_C",(SNES,SNESNGMRESSelectType),(snes,stype));CHKERRQ(ierr);
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
+  1. - C. W. Oosterlee and T. Washio, "Krylov Subspace Acceleration of Nonlinear Multigrid with Application to Recirculating Flows",
   SIAM Journal on Scientific Computing, 21(5), 2000.
-  2. - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres;
  PetscErrorCode ierr;
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

  ierr          = PetscNewLog(snes,&ngmres);CHKERRQ(ierr);
  snes->data    = (void*) ngmres;
  ngmres->msize = 30;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  ngmres->candidate = PETSC_FALSE;

  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  if (!((PetscObject)linesearch)->type_name) {
    ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);
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

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESSetSelectType_C",SNESNGMRESSetSelectType_NGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESSetRestartType_C",SNESNGMRESSetRestartType_NGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESSetRestartFmRise_C",SNESNGMRESSetRestartFmRise_NGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESNGMRESGetRestartFmRise_C",SNESNGMRESGetRestartFmRise_NGMRES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
