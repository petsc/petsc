/* Defines the basic SNES object */
#include <../src/snes/impls/fas/fasimpls.h>    /*I  "petscsnes.h"  I*/

const char *const SNESFASTypes[] = {"MULTIPLICATIVE","ADDITIVE","FULL","KASKADE","SNESFASType","SNES_FAS",NULL};

static PetscErrorCode SNESReset_FAS(SNES snes)
{
  SNES_FAS       *fas = (SNES_FAS*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDestroy(&fas->smoothu);CHKERRQ(ierr);
  ierr = SNESDestroy(&fas->smoothd);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->inject);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->restrct);CHKERRQ(ierr);
  ierr = VecDestroy(&fas->rscale);CHKERRQ(ierr);
  ierr = VecDestroy(&fas->Xg);CHKERRQ(ierr);
  ierr = VecDestroy(&fas->Fg);CHKERRQ(ierr);
  if (fas->next) {ierr = SNESReset(fas->next);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_FAS(SNES snes)
{
  SNES_FAS       *fas = (SNES_FAS*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* recursively resets and then destroys */
  ierr = SNESReset_FAS(snes);CHKERRQ(ierr);
  ierr = SNESDestroy(&fas->next);CHKERRQ(ierr);
  ierr = PetscFree(fas);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASSetUpLineSearch_Private(SNES snes, SNES smooth)
{
  SNESLineSearch linesearch;
  SNESLineSearch slinesearch;
  void           *lsprectx,*lspostctx;
  PetscErrorCode (*precheck)(SNESLineSearch,Vec,Vec,PetscBool*,void*);
  PetscErrorCode (*postcheck)(SNESLineSearch,Vec,Vec,Vec,PetscBool*,PetscBool*,void*);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!snes->linesearch) PetscFunctionReturn(0);
  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESGetLineSearch(smooth,&slinesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchGetPreCheck(linesearch,&precheck,&lsprectx);CHKERRQ(ierr);
  ierr = SNESLineSearchGetPostCheck(linesearch,&postcheck,&lspostctx);CHKERRQ(ierr);
  ierr = SNESLineSearchSetPreCheck(slinesearch,precheck,lsprectx);CHKERRQ(ierr);
  ierr = SNESLineSearchSetPostCheck(slinesearch,postcheck,lspostctx);CHKERRQ(ierr);
  ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)linesearch, (PetscObject)slinesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycleSetUpSmoother_Private(SNES snes, SNES smooth)
{
  SNES_FAS       *fas = (SNES_FAS*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)smooth);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(smooth);CHKERRQ(ierr);
  ierr = SNESFASSetUpLineSearch_Private(snes, smooth);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)snes->vec_sol);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)snes->vec_sol_update);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)snes->vec_func);CHKERRQ(ierr);
  smooth->vec_sol        = snes->vec_sol;
  smooth->vec_sol_update = snes->vec_sol_update;
  smooth->vec_func       = snes->vec_func;

  if (fas->eventsmoothsetup) {ierr = PetscLogEventBegin(fas->eventsmoothsetup,smooth,0,0,0);CHKERRQ(ierr);}
  ierr = SNESSetUp(smooth);CHKERRQ(ierr);
  if (fas->eventsmoothsetup) {ierr = PetscLogEventEnd(fas->eventsmoothsetup,smooth,0,0,0);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_FAS(SNES snes)
{
  SNES_FAS       *fas = (SNES_FAS*) snes->data;
  PetscErrorCode ierr;
  PetscInt       dm_levels;
  SNES           next;
  PetscBool      isFine, hasCreateRestriction, hasCreateInjection;

  PetscFunctionBegin;
  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  if (fas->usedmfornumberoflevels && isFine) {
    ierr = DMGetRefineLevel(snes->dm,&dm_levels);CHKERRQ(ierr);
    dm_levels++;
    if (dm_levels > fas->levels) {
      /* reset the number of levels */
      ierr = SNESFASSetLevels(snes,dm_levels,NULL);CHKERRQ(ierr);
      ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    }
  }
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  if (!isFine) snes->gridsequence = 0; /* no grid sequencing inside the multigrid hierarchy! */

  ierr = SNESSetWorkVecs(snes, 2);CHKERRQ(ierr); /* work vectors used for intergrid transfers */

  /* set up the smoothers if they haven't already been set up */
  if (!fas->smoothd) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd);CHKERRQ(ierr);
  }

  if (snes->dm) {
    /* set the smoother DMs properly */
    if (fas->smoothu) {ierr = SNESSetDM(fas->smoothu, snes->dm);CHKERRQ(ierr);}
    ierr = SNESSetDM(fas->smoothd, snes->dm);CHKERRQ(ierr);
    /* construct EVERYTHING from the DM -- including the progressive set of smoothers */
    if (next) {
      /* for now -- assume the DM and the evaluation functions have been set externally */
      if (!next->dm) {
        ierr = DMCoarsen(snes->dm, PetscObjectComm((PetscObject)next), &next->dm);CHKERRQ(ierr);
        ierr = SNESSetDM(next, next->dm);CHKERRQ(ierr);
      }
      /* set the interpolation and restriction from the DM */
      if (!fas->interpolate) {
        ierr = DMCreateInterpolation(next->dm, snes->dm, &fas->interpolate, &fas->rscale);CHKERRQ(ierr);
        if (!fas->restrct) {
          ierr = DMHasCreateRestriction(next->dm, &hasCreateRestriction);CHKERRQ(ierr);
          /* DM can create restrictions, use that */
          if (hasCreateRestriction) {
            ierr = DMCreateRestriction(next->dm, snes->dm, &fas->restrct);CHKERRQ(ierr);
          } else {
            ierr         = PetscObjectReference((PetscObject)fas->interpolate);CHKERRQ(ierr);
            fas->restrct = fas->interpolate;
          }
        }
      }
      /* set the injection from the DM */
      if (!fas->inject) {
        ierr = DMHasCreateInjection(next->dm, &hasCreateInjection);CHKERRQ(ierr);
        if (hasCreateInjection) {
          ierr = DMCreateInjection(next->dm, snes->dm, &fas->inject);CHKERRQ(ierr);
        }
      }
    }
  }

  /*pass the smoother, function, and jacobian up to the next level if it's not user set already */
  if (fas->galerkin) {
    if (next) {
      ierr = SNESSetFunction(next, NULL, SNESFASGalerkinFunctionDefault, next);CHKERRQ(ierr);
    }
    if (fas->smoothd && fas->level != fas->levels - 1) {
      ierr = SNESSetFunction(fas->smoothd, NULL, SNESFASGalerkinFunctionDefault, snes);CHKERRQ(ierr);
    }
    if (fas->smoothu && fas->level != fas->levels - 1) {
      ierr = SNESSetFunction(fas->smoothu, NULL, SNESFASGalerkinFunctionDefault, snes);CHKERRQ(ierr);
    }
  }

  /* sets the down (pre) smoother's default norm and sets it from options */
  if (fas->smoothd) {
    if (fas->level == 0 && fas->levels != 1) {
      ierr = SNESSetNormSchedule(fas->smoothd, SNES_NORM_NONE);CHKERRQ(ierr);
    } else {
      ierr = SNESSetNormSchedule(fas->smoothd, SNES_NORM_FINAL_ONLY);CHKERRQ(ierr);
    }
    ierr = SNESFASCycleSetUpSmoother_Private(snes, fas->smoothd);CHKERRQ(ierr);
  }

  /* sets the up (post) smoother's default norm and sets it from options */
  if (fas->smoothu) {
    if (fas->level != fas->levels - 1) {
      ierr = SNESSetNormSchedule(fas->smoothu, SNES_NORM_NONE);CHKERRQ(ierr);
    } else {
      ierr = SNESSetNormSchedule(fas->smoothu, SNES_NORM_FINAL_ONLY);CHKERRQ(ierr);
    }
    ierr = SNESFASCycleSetUpSmoother_Private(snes, fas->smoothu);CHKERRQ(ierr);
  }

  if (next) {
    /* gotta set up the solution vector for this to work */
    if (!next->vec_sol) {ierr = SNESFASCreateCoarseVec(snes,&next->vec_sol);CHKERRQ(ierr);}
    if (!next->vec_rhs) {ierr = SNESFASCreateCoarseVec(snes,&next->vec_rhs);CHKERRQ(ierr);}
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)next);CHKERRQ(ierr);
    ierr = SNESFASSetUpLineSearch_Private(snes, next);CHKERRQ(ierr);
    ierr = SNESSetUp(next);CHKERRQ(ierr);
  }

  /* setup FAS work vectors */
  if (fas->galerkin) {
    ierr = VecDuplicate(snes->vec_sol, &fas->Xg);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &fas->Fg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_FAS(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_FAS       *fas   = (SNES_FAS*) snes->data;
  PetscInt       levels = 1;
  PetscBool      flg    = PETSC_FALSE, upflg = PETSC_FALSE, downflg = PETSC_FALSE, monflg = PETSC_FALSE, galerkinflg = PETSC_FALSE,continuationflg = PETSC_FALSE;
  PetscErrorCode ierr;
  SNESFASType    fastype;
  const char     *optionsprefix;
  SNESLineSearch linesearch;
  PetscInt       m, n_up, n_down;
  SNES           next;
  PetscBool      isFine;

  PetscFunctionBegin;
  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"SNESFAS Options-----------------------------------");CHKERRQ(ierr);

  /* number of levels -- only process most options on the finest level */
  if (isFine) {
    ierr = PetscOptionsInt("-snes_fas_levels", "Number of Levels", "SNESFASSetLevels", levels, &levels, &flg);CHKERRQ(ierr);
    if (!flg && snes->dm) {
      ierr = DMGetRefineLevel(snes->dm,&levels);CHKERRQ(ierr);
      levels++;
      fas->usedmfornumberoflevels = PETSC_TRUE;
    }
    ierr    = SNESFASSetLevels(snes, levels, NULL);CHKERRQ(ierr);
    fastype = fas->fastype;
    ierr    = PetscOptionsEnum("-snes_fas_type","FAS correction type","SNESFASSetType",SNESFASTypes,(PetscEnum)fastype,(PetscEnum*)&fastype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESFASSetType(snes, fastype);CHKERRQ(ierr);
    }

    ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_fas_cycles","Number of cycles","SNESFASSetCycles",fas->n_cycles,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESFASSetCycles(snes, m);CHKERRQ(ierr);
    }
    ierr = PetscOptionsBool("-snes_fas_continuation","Corrected grid-sequence continuation","SNESFASSetContinuation",fas->continuation,&continuationflg,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESFASSetContinuation(snes,continuationflg);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBool("-snes_fas_galerkin", "Form coarse problems with Galerkin","SNESFASSetGalerkin",fas->galerkin,&galerkinflg,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESFASSetGalerkin(snes, galerkinflg);CHKERRQ(ierr);
    }

    if (fas->fastype == SNES_FAS_FULL) {
      ierr   = PetscOptionsBool("-snes_fas_full_downsweep","Smooth on the initial down sweep for full FAS cycles","SNESFASFullSetDownSweep",fas->full_downsweep,&fas->full_downsweep,&flg);CHKERRQ(ierr);
      if (flg) {ierr = SNESFASFullSetDownSweep(snes,fas->full_downsweep);CHKERRQ(ierr);}
    }

    ierr = PetscOptionsInt("-snes_fas_smoothup","Number of post-smoothing steps","SNESFASSetNumberSmoothUp",fas->max_up_it,&n_up,&upflg);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-snes_fas_smoothdown","Number of pre-smoothing steps","SNESFASSetNumberSmoothDown",fas->max_down_it,&n_down,&downflg);CHKERRQ(ierr);

    {
      PetscViewer viewer;
      PetscViewerFormat format;
      ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject) snes)->options,((PetscObject)snes)->prefix,"-snes_fas_monitor",&viewer,&format,&monflg);CHKERRQ(ierr);
      if (monflg) {
        PetscViewerAndFormat *vf;
        ierr = PetscViewerAndFormatCreate(viewer,format,&vf);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
        ierr = SNESFASSetMonitor(snes,vf,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    flg    = PETSC_FALSE;
    monflg = PETSC_TRUE;
    ierr   = PetscOptionsBool("-snes_fas_log","Log times for each FAS level","SNESFASSetLog",monflg,&monflg,&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESFASSetLog(snes,monflg);CHKERRQ(ierr);}
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  /* setup from the determined types if there is no pointwise procedure or smoother defined */
  if (upflg) {
    ierr = SNESFASSetNumberSmoothUp(snes,n_up);CHKERRQ(ierr);
  }
  if (downflg) {
    ierr = SNESFASSetNumberSmoothDown(snes,n_down);CHKERRQ(ierr);
  }

  /* set up the default line search for coarse grid corrections */
  if (fas->fastype == SNES_FAS_ADDITIVE) {
    if (!snes->linesearch) {
      ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
      ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHL2);CHKERRQ(ierr);
    }
  }

  /* recursive option setting for the smoothers */
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  if (next) {ierr = SNESSetFromOptions(next);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer)
{
  SNES_FAS       *fas = (SNES_FAS*) snes->data;
  PetscBool      isFine,iascii,isdraw;
  PetscInt       i;
  PetscErrorCode ierr;
  SNES           smoothu, smoothd, levelsnes;

  PetscFunctionBegin;
  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  if (isFine) {
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer, "  type is %s, levels=%D, cycles=%D\n",  SNESFASTypes[fas->fastype], fas->levels, fas->n_cycles);CHKERRQ(ierr);
      if (fas->galerkin) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Using Galerkin computed coarse grid function evaluation\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Not using Galerkin computed coarse grid function evaluation\n");CHKERRQ(ierr);
      }
      for (i=0; i<fas->levels; i++) {
        ierr = SNESFASGetCycleSNES(snes, i, &levelsnes);CHKERRQ(ierr);
        ierr = SNESFASCycleGetSmootherUp(levelsnes, &smoothu);CHKERRQ(ierr);
        ierr = SNESFASCycleGetSmootherDown(levelsnes, &smoothd);CHKERRQ(ierr);
        if (!i) {
          ierr = PetscViewerASCIIPrintf(viewer,"  Coarse grid solver -- level %D -------------------------------\n",i);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"  Down solver (pre-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        if (smoothd) {
          ierr = SNESView(smoothd,viewer);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"Not yet available\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        if (i && (smoothd == smoothu)) {
          ierr = PetscViewerASCIIPrintf(viewer,"  Up solver (post-smoother) same as down solver (pre-smoother)\n");CHKERRQ(ierr);
        } else if (i) {
          ierr = PetscViewerASCIIPrintf(viewer,"  Up solver (post-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (smoothu) {
            ierr = SNESView(smoothu,viewer);CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer,"Not yet available\n");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
      }
    } else if (isdraw) {
      PetscDraw draw;
      PetscReal x,w,y,bottom,th,wth;
      SNES_FAS  *curfas = fas;
      ierr   = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
      ierr   = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
      ierr   = PetscDrawStringGetSize(draw,&wth,&th);CHKERRQ(ierr);
      bottom = y - th;
      while (curfas) {
        if (!curfas->smoothu) {
          ierr = PetscDrawPushCurrentPoint(draw,x,bottom);CHKERRQ(ierr);
          if (curfas->smoothd) ierr = SNESView(curfas->smoothd,viewer);CHKERRQ(ierr);
          ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
        } else {
          w    = 0.5*PetscMin(1.0-x,x);
          ierr = PetscDrawPushCurrentPoint(draw,x-w,bottom);CHKERRQ(ierr);
          if (curfas->smoothd) ierr = SNESView(curfas->smoothd,viewer);CHKERRQ(ierr);
          ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
          ierr = PetscDrawPushCurrentPoint(draw,x+w,bottom);CHKERRQ(ierr);
          if (curfas->smoothu) ierr = SNESView(curfas->smoothu,viewer);CHKERRQ(ierr);
          ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
        }
        /* this is totally bogus but we have no way of knowing how low the previous one was draw to */
        bottom -= 5*th;
        if (curfas->next) curfas = (SNES_FAS*)curfas->next->data;
        else curfas = NULL;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
Defines the action of the downsmoother
 */
static PetscErrorCode SNESFASDownSmooth_Private(SNES snes, Vec B, Vec X, Vec F, PetscReal *fnorm)
{
  PetscErrorCode      ierr;
  SNESConvergedReason reason;
  Vec                 FPC;
  SNES                smoothd;
  PetscBool           flg;
  SNES_FAS            *fas = (SNES_FAS*) snes->data;

  PetscFunctionBegin;
  ierr = SNESFASCycleGetSmootherDown(snes, &smoothd);CHKERRQ(ierr);
  ierr = SNESSetInitialFunction(smoothd, F);CHKERRQ(ierr);
  if (fas->eventsmoothsolve) {ierr = PetscLogEventBegin(fas->eventsmoothsolve,smoothd,B,X,0);CHKERRQ(ierr);}
  ierr = SNESSolve(smoothd, B, X);CHKERRQ(ierr);
  if (fas->eventsmoothsolve) {ierr = PetscLogEventEnd(fas->eventsmoothsolve,smoothd,B,X,0);CHKERRQ(ierr);}
  /* check convergence reason for the smoother */
  ierr = SNESGetConvergedReason(smoothd,&reason);CHKERRQ(ierr);
  if (reason < 0 && !(reason == SNES_DIVERGED_MAX_IT || reason == SNES_DIVERGED_LOCAL_MIN || reason == SNES_DIVERGED_LINE_SEARCH)) {
    snes->reason = SNES_DIVERGED_INNER;
    PetscFunctionReturn(0);
  }

  ierr = SNESGetFunction(smoothd, &FPC, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESGetAlwaysComputesFinalResidual(smoothd, &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = SNESComputeFunction(smoothd, X, FPC);CHKERRQ(ierr);
  }
  ierr = VecCopy(FPC, F);CHKERRQ(ierr);
  if (fnorm) {ierr = VecNorm(F,NORM_2,fnorm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


/*
Defines the action of the upsmoother
 */
static PetscErrorCode SNESFASUpSmooth_Private(SNES snes, Vec B, Vec X, Vec F, PetscReal *fnorm)
{
  PetscErrorCode      ierr;
  SNESConvergedReason reason;
  Vec                 FPC;
  SNES                smoothu;
  PetscBool           flg;
  SNES_FAS            *fas = (SNES_FAS*) snes->data;

  PetscFunctionBegin;
  ierr = SNESFASCycleGetSmootherUp(snes, &smoothu);CHKERRQ(ierr);
  if (fas->eventsmoothsolve) {ierr = PetscLogEventBegin(fas->eventsmoothsolve,smoothu,0,0,0);CHKERRQ(ierr);}
  ierr = SNESSolve(smoothu, B, X);CHKERRQ(ierr);
  if (fas->eventsmoothsolve) {ierr = PetscLogEventEnd(fas->eventsmoothsolve,smoothu,0,0,0);CHKERRQ(ierr);}
  /* check convergence reason for the smoother */
  ierr = SNESGetConvergedReason(smoothu,&reason);CHKERRQ(ierr);
  if (reason < 0 && !(reason == SNES_DIVERGED_MAX_IT || reason == SNES_DIVERGED_LOCAL_MIN || reason == SNES_DIVERGED_LINE_SEARCH)) {
    snes->reason = SNES_DIVERGED_INNER;
    PetscFunctionReturn(0);
  }
  ierr = SNESGetFunction(smoothu, &FPC, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESGetAlwaysComputesFinalResidual(smoothu, &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = SNESComputeFunction(smoothu, X, FPC);CHKERRQ(ierr);
  }
  ierr = VecCopy(FPC, F);CHKERRQ(ierr);
  if (fnorm) {ierr = VecNorm(F,NORM_2,fnorm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
   SNESFASCreateCoarseVec - create Vec corresponding to a state vector on one level coarser than current level

   Collective

   Input Arguments:
.  snes - SNESFAS

   Output Arguments:
.  Xcoarse - vector on level one coarser than snes

   Level: developer

.seealso: SNESFASSetRestriction(), SNESFASRestrict()
@*/
PetscErrorCode SNESFASCreateCoarseVec(SNES snes,Vec *Xcoarse)
{
  PetscErrorCode ierr;
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(Xcoarse,2);
  fas = (SNES_FAS*)snes->data;
  if (fas->rscale) {
    ierr = VecDuplicate(fas->rscale,Xcoarse);CHKERRQ(ierr);
  } else if (fas->inject) {
    ierr = MatCreateVecs(fas->inject,Xcoarse,NULL);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Must set restriction or injection");
  PetscFunctionReturn(0);
}

/*@
   SNESFASRestrict - restrict a Vec to the next coarser level

   Collective

   Input Arguments:
+  fine - SNES from which to restrict
-  Xfine - vector to restrict

   Output Arguments:
.  Xcoarse - result of restriction

   Level: developer

.seealso: SNESFASSetRestriction(), SNESFASSetInjection()
@*/
PetscErrorCode SNESFASRestrict(SNES fine,Vec Xfine,Vec Xcoarse)
{
  PetscErrorCode ierr;
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fine,SNES_CLASSID,1,SNESFAS);
  PetscValidHeaderSpecific(Xfine,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Xcoarse,VEC_CLASSID,3);
  fas = (SNES_FAS*)fine->data;
  if (fas->inject) {
    ierr = MatRestrict(fas->inject,Xfine,Xcoarse);CHKERRQ(ierr);
  } else {
    ierr = MatRestrict(fas->restrct,Xfine,Xcoarse);CHKERRQ(ierr);
    ierr = VecPointwiseMult(Xcoarse,fas->rscale,Xcoarse);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*

Performs the FAS coarse correction as:

fine problem:   F(x) = b
coarse problem: F^c(x^c) = b^c

b^c = F^c(Rx) - R(F(x) - b)

 */
PetscErrorCode SNESFASCoarseCorrection(SNES snes, Vec X, Vec F, Vec X_new)
{
  PetscErrorCode      ierr;
  Vec                 X_c, Xo_c, F_c, B_c;
  SNESConvergedReason reason;
  SNES                next;
  Mat                 restrct, interpolate;
  SNES_FAS            *fasc;

  PetscFunctionBegin;
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  if (next) {
    fasc = (SNES_FAS*)next->data;

    ierr = SNESFASCycleGetRestriction(snes, &restrct);CHKERRQ(ierr);
    ierr = SNESFASCycleGetInterpolation(snes, &interpolate);CHKERRQ(ierr);

    X_c  = next->vec_sol;
    Xo_c = next->work[0];
    F_c  = next->vec_func;
    B_c  = next->vec_rhs;

    if (fasc->eventinterprestrict) {ierr = PetscLogEventBegin(fasc->eventinterprestrict,snes,0,0,0);CHKERRQ(ierr);}
    ierr = SNESFASRestrict(snes, X, Xo_c);CHKERRQ(ierr);
    /* restrict the defect: R(F(x) - b) */
    ierr = MatRestrict(restrct, F, B_c);CHKERRQ(ierr);
    if (fasc->eventinterprestrict) {ierr = PetscLogEventEnd(fasc->eventinterprestrict,snes,0,0,0);CHKERRQ(ierr);}

    if (fasc->eventresidual) {ierr = PetscLogEventBegin(fasc->eventresidual,next,0,0,0);CHKERRQ(ierr);}
    /* F_c = F^c(Rx) - R(F(x) - b) since the second term was sitting in next->vec_rhs */
    ierr = SNESComputeFunction(next, Xo_c, F_c);CHKERRQ(ierr);
    if (fasc->eventresidual) {ierr = PetscLogEventEnd(fasc->eventresidual,next,0,0,0);CHKERRQ(ierr);}

    /* solve the coarse problem corresponding to F^c(x^c) = b^c = F^c(Rx) - R(F(x) - b) */
    ierr = VecCopy(B_c, X_c);CHKERRQ(ierr);
    ierr = VecCopy(F_c, B_c);CHKERRQ(ierr);
    ierr = VecCopy(X_c, F_c);CHKERRQ(ierr);
    /* set initial guess of the coarse problem to the projected fine solution */
    ierr = VecCopy(Xo_c, X_c);CHKERRQ(ierr);

    /* recurse to the next level */
    ierr = SNESSetInitialFunction(next, F_c);CHKERRQ(ierr);
    ierr = SNESSolve(next, B_c, X_c);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(next,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    /* correct as x <- x + I(x^c - Rx)*/
    ierr = VecAXPY(X_c, -1.0, Xo_c);CHKERRQ(ierr);

    if (fasc->eventinterprestrict) {ierr = PetscLogEventBegin(fasc->eventinterprestrict,snes,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolateAdd(interpolate, X_c, X, X_new);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X_c, "Coarse correction");CHKERRQ(ierr);
    ierr = VecViewFromOptions(X_c, NULL, "-fas_coarse_solution_view");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X_new, "Updated Fine solution");CHKERRQ(ierr);
    ierr = VecViewFromOptions(X_new, NULL, "-fas_levels_1_solution_view");CHKERRQ(ierr);
    if (fasc->eventinterprestrict) {ierr = PetscLogEventEnd(fasc->eventinterprestrict,snes,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*

The additive cycle looks like:

xhat = x
xhat = dS(x, b)
x = coarsecorrection(xhat, b_d)
x = x + nu*(xhat - x);
(optional) x = uS(x, b)

With the coarse RHS (defect correction) as below.

 */
static PetscErrorCode SNESFASCycle_Additive(SNES snes, Vec X)
{
  Vec                  F, B, Xhat;
  Vec                  X_c, Xo_c, F_c, B_c;
  PetscErrorCode       ierr;
  SNESConvergedReason  reason;
  PetscReal            xnorm, fnorm, ynorm;
  SNESLineSearchReason lsresult;
  SNES                 next;
  Mat                  restrct, interpolate;
  SNES_FAS             *fas = (SNES_FAS*)snes->data,*fasc;

  PetscFunctionBegin;
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  F    = snes->vec_func;
  B    = snes->vec_rhs;
  Xhat = snes->work[1];
  ierr = VecCopy(X, Xhat);CHKERRQ(ierr);
  /* recurse first */
  if (next) {
    fasc = (SNES_FAS*)next->data;
    ierr = SNESFASCycleGetRestriction(snes, &restrct);CHKERRQ(ierr);
    ierr = SNESFASCycleGetInterpolation(snes, &interpolate);CHKERRQ(ierr);
    if (fas->eventresidual) {ierr = PetscLogEventBegin(fas->eventresidual,snes,0,0,0);CHKERRQ(ierr);}
    ierr = SNESComputeFunction(snes, Xhat, F);CHKERRQ(ierr);
    if (fas->eventresidual) {ierr = PetscLogEventEnd(fas->eventresidual,snes,0,0,0);CHKERRQ(ierr);}
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    X_c  = next->vec_sol;
    Xo_c = next->work[0];
    F_c  = next->vec_func;
    B_c  = next->vec_rhs;

    ierr = SNESFASRestrict(snes,Xhat,Xo_c);CHKERRQ(ierr);
    /* restrict the defect */
    ierr = MatRestrict(restrct, F, B_c);CHKERRQ(ierr);

    /* solve the coarse problem corresponding to F^c(x^c) = b^c = Rb + F^c(Rx) - RF(x) */
    if (fasc->eventresidual) {ierr = PetscLogEventBegin(fasc->eventresidual,next,0,0,0);CHKERRQ(ierr);}
    ierr = SNESComputeFunction(next, Xo_c, F_c);CHKERRQ(ierr);
    if (fasc->eventresidual) {ierr = PetscLogEventEnd(fasc->eventresidual,next,0,0,0);CHKERRQ(ierr);}
    ierr = VecCopy(B_c, X_c);CHKERRQ(ierr);
    ierr = VecCopy(F_c, B_c);CHKERRQ(ierr);
    ierr = VecCopy(X_c, F_c);CHKERRQ(ierr);
    /* set initial guess of the coarse problem to the projected fine solution */
    ierr = VecCopy(Xo_c, X_c);CHKERRQ(ierr);

    /* recurse */
    ierr = SNESSetInitialFunction(next, F_c);CHKERRQ(ierr);
    ierr = SNESSolve(next, B_c, X_c);CHKERRQ(ierr);

    /* smooth on this level */
    ierr = SNESFASDownSmooth_Private(snes, B, X, F, &fnorm);CHKERRQ(ierr);

    ierr = SNESGetConvergedReason(next,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }

    /* correct as x <- x + I(x^c - Rx)*/
    ierr = VecAYPX(X_c, -1.0, Xo_c);CHKERRQ(ierr);
    ierr = MatInterpolate(interpolate, X_c, Xhat);CHKERRQ(ierr);

    /* additive correction of the coarse direction*/
    ierr = SNESLineSearchApply(snes->linesearch, X, F, &fnorm, Xhat);CHKERRQ(ierr);
    ierr = SNESLineSearchGetReason(snes->linesearch, &lsresult);CHKERRQ(ierr);
    ierr = SNESLineSearchGetNorms(snes->linesearch, &xnorm, &snes->norm, &ynorm);CHKERRQ(ierr);
    if (lsresult) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
  } else {
    ierr = SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*

Defines the FAS cycle as:

fine problem: F(x) = b
coarse problem: F^c(x) = b^c

b^c = F^c(Rx) - R(F(x) - b)

correction:

x = x + I(x^c - Rx)

 */
static PetscErrorCode SNESFASCycle_Multiplicative(SNES snes, Vec X)
{

  PetscErrorCode ierr;
  Vec            F,B;
  SNES           next;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  /* pre-smooth -- just update using the pre-smoother */
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  ierr = SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm);CHKERRQ(ierr);
  if (next) {
    ierr = SNESFASCoarseCorrection(snes, X, F, X);CHKERRQ(ierr);
    ierr = SNESFASUpSmooth_Private(snes, B, X, F, &snes->norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycleSetupPhase_Full(SNES snes)
{
  SNES           next;
  SNES_FAS       *fas = (SNES_FAS*)snes->data;
  PetscBool      isFine;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* pre-smooth -- just update using the pre-smoother */
  ierr = SNESFASCycleIsFine(snes,&isFine);CHKERRQ(ierr);
  ierr = SNESFASCycleGetCorrection(snes,&next);CHKERRQ(ierr);
  fas->full_stage = 0;
  if (next) {ierr = SNESFASCycleSetupPhase_Full(next);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycle_Full(SNES snes, Vec X)
{
  PetscErrorCode ierr;
  Vec            F,B;
  SNES_FAS       *fas = (SNES_FAS*)snes->data;
  PetscBool      isFine;
  SNES           next;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  ierr = SNESFASCycleIsFine(snes,&isFine);CHKERRQ(ierr);
  ierr = SNESFASCycleGetCorrection(snes,&next);CHKERRQ(ierr);

  if (isFine) {
    ierr = SNESFASCycleSetupPhase_Full(snes);CHKERRQ(ierr);
  }

  if (fas->full_stage == 0) {
    /* downsweep */
    if (next) {
      if (fas->level != 1) next->max_its += 1;
      if (fas->full_downsweep) {ierr = SNESFASDownSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);}
      fas->full_downsweep = PETSC_TRUE;
      ierr = SNESFASCoarseCorrection(snes,X,F,X);CHKERRQ(ierr);
      ierr = SNESFASUpSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);
      if (fas->level != 1) next->max_its -= 1;
    } else {
      /* The smoother on the coarse level is the coarse solver */
      ierr = SNESFASDownSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);
    }
    fas->full_stage = 1;
  } else if (fas->full_stage == 1) {
    if (snes->iter == 0) {ierr = SNESFASDownSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);}
    if (next) {
      ierr = SNESFASCoarseCorrection(snes,X,F,X);CHKERRQ(ierr);
      ierr = SNESFASUpSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);
    }
  }
  /* final v-cycle */
  if (isFine) {
    if (next) {
      ierr = SNESFASCoarseCorrection(snes,X,F,X);CHKERRQ(ierr);
      ierr = SNESFASUpSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycle_Kaskade(SNES snes, Vec X)
{
  PetscErrorCode ierr;
  Vec            F,B;
  SNES           next;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  ierr = SNESFASCycleGetCorrection(snes,&next);CHKERRQ(ierr);
  if (next) {
    ierr = SNESFASCoarseCorrection(snes,X,F,X);CHKERRQ(ierr);
    ierr = SNESFASUpSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);
  } else {
    ierr = SNESFASDownSmooth_Private(snes,B,X,F,&snes->norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscBool SNEScite = PETSC_FALSE;
const char SNESCitation[] = "@techreport{pbmkbsxt2012,\n"
                            "  title = {Composing Scalable Nonlinear Algebraic Solvers},\n"
                            "  author = {Peter Brune and Mathew Knepley and Barry Smith and Xuemin Tu},\n"
                            "  year = 2013,\n"
                            "  type = Preprint,\n"
                            "  number = {ANL/MCS-P2010-0112},\n"
                            "  institution = {Argonne National Laboratory}\n}\n";

static PetscErrorCode SNESSolve_FAS(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            X, F;
  PetscReal      fnorm;
  SNES_FAS       *fas = (SNES_FAS*)snes->data,*ffas;
  DM             dm;
  PetscBool      isFine;

  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  ierr = PetscCitationsRegister(SNESCitation,&SNEScite);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITERATING;
  X            = snes->vec_sol;
  F            = snes->vec_func;

  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  /* norm setup */
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  if (!snes->vec_func_init_set) {
    if (fas->eventresidual) {ierr = PetscLogEventBegin(fas->eventresidual,snes,0,0,0);CHKERRQ(ierr);}
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (fas->eventresidual) {ierr = PetscLogEventEnd(fas->eventresidual,snes,0,0,0);CHKERRQ(ierr);}
  } else snes->vec_func_init_set = PETSC_FALSE;

  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
  SNESCheckFunctionNorm(snes,fnorm);
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,snes->iter,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);


  if (isFine) {
    /* propagate scale-dependent data up the hierarchy */
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    for (ffas=fas; ffas->next; ffas=(SNES_FAS*)ffas->next->data) {
      DM dmcoarse;
      ierr = SNESGetDM(ffas->next,&dmcoarse);CHKERRQ(ierr);
      ierr = DMRestrict(dm,ffas->restrct,ffas->rscale,ffas->inject,dmcoarse);CHKERRQ(ierr);
      dm   = dmcoarse;
    }
  }

  for (i = 0; i < snes->max_its; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    if (fas->fastype == SNES_FAS_MULTIPLICATIVE) {
      ierr = SNESFASCycle_Multiplicative(snes, X);CHKERRQ(ierr);
    } else if (fas->fastype == SNES_FAS_ADDITIVE) {
      ierr = SNESFASCycle_Additive(snes, X);CHKERRQ(ierr);
    } else if (fas->fastype == SNES_FAS_FULL) {
      ierr = SNESFASCycle_Full(snes, X);CHKERRQ(ierr);
    } else if (fas->fastype == SNES_FAS_KASKADE) {
      ierr = SNESFASCycle_Kaskade(snes, X);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Unsupported FAS type");

    /* check for FAS cycle divergence */
    if (snes->reason != SNES_CONVERGED_ITERATING) PetscFunctionReturn(0);

    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (isFine) {
      ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,snes->norm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
      if (snes->reason) break;
    }
  }
  if (i == snes->max_its) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", i);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/*MC

SNESFAS - Full Approximation Scheme nonlinear multigrid solver.

   The nonlinear problem is solved by correction using coarse versions
   of the nonlinear problem.  This problem is perturbed so that a projected
   solution of the fine problem elicits no correction from the coarse problem.

Options Database:
+   -snes_fas_levels -  The number of levels
.   -snes_fas_cycles<1> -  The number of cycles -- 1 for V, 2 for W
.   -snes_fas_type<additive,multiplicative,full,kaskade>  -  Additive or multiplicative cycle
.   -snes_fas_galerkin<PETSC_FALSE> -  Form coarse problems by projection back upon the fine problem
.   -snes_fas_smoothup<1> -  The number of iterations of the post-smoother
.   -snes_fas_smoothdown<1> -  The number of iterations of the pre-smoother
.   -snes_fas_monitor -  Monitor progress of all of the levels
.   -snes_fas_full_downsweep<PETSC_FALSE> - call the downsmooth on the initial downsweep of full FAS
.   -fas_levels_snes_ -  SNES options for all smoothers
.   -fas_levels_cycle_snes_ -  SNES options for all cycles
.   -fas_levels_i_snes_ -  SNES options for the smoothers on level i
.   -fas_levels_i_cycle_snes_ - SNES options for the cycle on level i
-   -fas_coarse_snes_ -  SNES options for the coarsest smoother

Notes:
   The organization of the FAS solver is slightly different from the organization of PCMG
   As each level has smoother SNES instances(down and potentially up) and a cycle SNES instance.
   The cycle SNES instance may be used for monitoring convergence on a particular level.

Level: beginner

   References:
. 1. -  Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso: PCMG, SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_FAS(SNES snes)
{
  SNES_FAS       *fas;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_FAS;
  snes->ops->setup          = SNESSetUp_FAS;
  snes->ops->setfromoptions = SNESSetFromOptions_FAS;
  snes->ops->view           = SNESView_FAS;
  snes->ops->solve          = SNESSolve_FAS;
  snes->ops->reset          = SNESReset_FAS;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  ierr = PetscNewLog(snes,&fas);CHKERRQ(ierr);

  snes->data                  = (void*) fas;
  fas->level                  = 0;
  fas->levels                 = 1;
  fas->n_cycles               = 1;
  fas->max_up_it              = 1;
  fas->max_down_it            = 1;
  fas->smoothu                = NULL;
  fas->smoothd                = NULL;
  fas->next                   = NULL;
  fas->previous               = NULL;
  fas->fine                   = snes;
  fas->interpolate            = NULL;
  fas->restrct                = NULL;
  fas->inject                 = NULL;
  fas->usedmfornumberoflevels = PETSC_FALSE;
  fas->fastype                = SNES_FAS_MULTIPLICATIVE;
  fas->full_downsweep         = PETSC_FALSE;

  fas->eventsmoothsetup    = 0;
  fas->eventsmoothsolve    = 0;
  fas->eventresidual       = 0;
  fas->eventinterprestrict = 0;
  PetscFunctionReturn(0);
}
