/* Defines the basic SNES object */
#include <../src/snes/impls/fas/fasimpls.h>    /*I  "petscsnesfas.h"  I*/

const char *SNESFASTypes[] = {"MULTIPLICATIVE","ADDITIVE","SNESFASType","SNES_FAS",0};

extern PetscErrorCode SNESDestroy_FAS(SNES snes);
extern PetscErrorCode SNESSetUp_FAS(SNES snes);
extern PetscErrorCode SNESSetFromOptions_FAS(SNES snes);
extern PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer);
extern PetscErrorCode SNESSolve_FAS(SNES snes);
extern PetscErrorCode SNESReset_FAS(SNES snes);
extern PetscErrorCode SNESFASGalerkinDefaultFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode SNESFASCycleCreateSmoother_Private(SNES, SNES *);

/*MC

SNESFAS - Full Approximation Scheme nonlinear multigrid solver.

   The nonlinear problem is solved by correction using coarse versions
   of the nonlinear problem.  This problem is perturbed so that a projected
   solution of the fine problem elicits no correction from the coarse problem.

Options Database:
+   -snes_fas_levels -  The number of levels
.   -snes_fas_cycles<1> -  The number of cycles -- 1 for V, 2 for W
.   -snes_fas_type<additive, multiplicative>  -  Additive or multiplicative cycle
.   -snes_fas_galerkin<PETSC_FALSE> -  Form coarse problems by projection back upon the fine problem
.   -snes_fas_smoothup<1> -  The number of iterations of the post-smoother
.   -snes_fas_smoothdown<1> -  The number of iterations of the pre-smoother
.   -snes_fas_monitor -  Monitor progress of all of the levels
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

.seealso: PCMG, SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

#undef __FUNCT__
#define __FUNCT__ "SNESCreate_FAS"
PETSC_EXTERN_C PetscErrorCode SNESCreate_FAS(SNES snes)
{
  SNES_FAS * fas;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_FAS;
  snes->ops->setup          = SNESSetUp_FAS;
  snes->ops->setfromoptions = SNESSetFromOptions_FAS;
  snes->ops->view           = SNESView_FAS;
  snes->ops->solve          = SNESSolve_FAS;
  snes->ops->reset          = SNESReset_FAS;

  snes->usesksp             = PETSC_FALSE;
  snes->usespc              = PETSC_FALSE;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  ierr = PetscNewLog(snes, SNES_FAS, &fas);CHKERRQ(ierr);
  snes->data                  = (void*) fas;
  fas->level                  = 0;
  fas->levels                 = 1;
  fas->n_cycles               = 1;
  fas->max_up_it              = 1;
  fas->max_down_it            = 1;
  fas->smoothu                = PETSC_NULL;
  fas->smoothd                = PETSC_NULL;
  fas->next                   = PETSC_NULL;
  fas->previous               = PETSC_NULL;
  fas->fine                   = snes;
  fas->interpolate            = PETSC_NULL;
  fas->restrct                = PETSC_NULL;
  fas->inject                 = PETSC_NULL;
  fas->monitor                = PETSC_NULL;
  fas->usedmfornumberoflevels = PETSC_FALSE;
  fas->fastype                = SNES_FAS_MULTIPLICATIVE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_FAS"
PetscErrorCode SNESReset_FAS(SNES snes)
{
  PetscErrorCode ierr = 0;
  SNES_FAS * fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  ierr = SNESDestroy(&fas->smoothu);CHKERRQ(ierr);
  ierr = SNESDestroy(&fas->smoothd);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->inject);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->restrct);CHKERRQ(ierr);
  ierr = VecDestroy(&fas->rscale);CHKERRQ(ierr);
  if (fas->next)      ierr = SNESReset(fas->next);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_FAS"
PetscErrorCode SNESDestroy_FAS(SNES snes)
{
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscErrorCode ierr = 0;

  PetscFunctionBegin;
  /* recursively resets and then destroys */
  ierr = SNESReset(snes);CHKERRQ(ierr);
  if (fas->next)         ierr = SNESDestroy(&fas->next);CHKERRQ(ierr);
  ierr = PetscFree(fas);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_FAS"
PetscErrorCode SNESSetUp_FAS(SNES snes)
{
  SNES_FAS                    *fas = (SNES_FAS *) snes->data;
  PetscErrorCode              ierr;
  VecScatter                  injscatter;
  PetscInt                    dm_levels;
  Vec                         vec_sol, vec_func, vec_sol_update, vec_rhs; /* preserve these if they're set through the reset */
  SNES                        next;
  PetscBool                   isFine;
  SNESLineSearch              linesearch;
  SNESLineSearch              slinesearch;
  void                        *lsprectx,*lspostctx;
  SNESLineSearchPreCheckFunc  lsprefunc;
  SNESLineSearchPostCheckFunc lspostfunc;
  PetscFunctionBegin;

  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  if (fas->usedmfornumberoflevels && isFine) {
    ierr = DMGetRefineLevel(snes->dm,&dm_levels);CHKERRQ(ierr);
    dm_levels++;
    if (dm_levels > fas->levels) {
      /* we don't want the solution and func vectors to be destroyed in the SNESReset when it's called in SNESFASSetLevels_FAS*/
      vec_sol = snes->vec_sol;
      vec_func = snes->vec_func;
      vec_sol_update = snes->vec_sol_update;
      vec_rhs = snes->vec_rhs;
      snes->vec_sol = PETSC_NULL;
      snes->vec_func = PETSC_NULL;
      snes->vec_sol_update = PETSC_NULL;
      snes->vec_rhs = PETSC_NULL;

      /* reset the number of levels */
      ierr = SNESFASSetLevels(snes,dm_levels,PETSC_NULL);CHKERRQ(ierr);
      ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

      snes->vec_sol = vec_sol;
      snes->vec_func = vec_func;
      snes->vec_rhs = vec_rhs;
      snes->vec_sol_update = vec_sol_update;
    }
  }
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  if (!isFine) snes->gridsequence = 0; /* no grid sequencing inside the multigrid hierarchy! */

  if (fas->fastype == SNES_FAS_MULTIPLICATIVE) {
    ierr = SNESDefaultGetWork(snes, 2);CHKERRQ(ierr); /* work vectors used for intergrid transfers */
  } else {
    ierr = SNESDefaultGetWork(snes, 2);CHKERRQ(ierr); /* work vectors used for intergrid transfers */
  }

  /* set up the smoothers if they haven't already been set up */
  if (!fas->smoothd) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd);CHKERRQ(ierr);
  }
  if (!fas->smoothu && fas->level != fas->levels - 1) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu);CHKERRQ(ierr);
  }

  if (snes->dm) {
    /* set the smoother DMs properly */
    if (fas->smoothu) ierr = SNESSetDM(fas->smoothu, snes->dm);CHKERRQ(ierr);
    ierr = SNESSetDM(fas->smoothd, snes->dm);CHKERRQ(ierr);
    /* construct EVERYTHING from the DM -- including the progressive set of smoothers */
    if (next) {
      /* for now -- assume the DM and the evaluation functions have been set externally */
      if (!next->dm) {
        ierr = DMCoarsen(snes->dm, ((PetscObject)next)->comm, &next->dm);CHKERRQ(ierr);
        ierr = SNESSetDM(next, next->dm);CHKERRQ(ierr);
      }
      /* set the interpolation and restriction from the DM */
      if (!fas->interpolate) {
        ierr = DMCreateInterpolation(next->dm, snes->dm, &fas->interpolate, &fas->rscale);CHKERRQ(ierr);
        if (!fas->restrct) {
          ierr = PetscObjectReference((PetscObject)fas->interpolate);CHKERRQ(ierr);
          fas->restrct = fas->interpolate;
        }
      }
      /* set the injection from the DM */
      if (!fas->inject) {
        ierr = DMCreateInjection(next->dm, snes->dm, &injscatter);CHKERRQ(ierr);
        ierr = MatCreateScatter(((PetscObject)snes)->comm, injscatter, &fas->inject);CHKERRQ(ierr);
        ierr = VecScatterDestroy(&injscatter);CHKERRQ(ierr);
      }
    }
  }
  /*pass the smoother, function, and jacobian up to the next level if it's not user set already */
  if (fas->galerkin) {
    if (next)
      ierr = SNESSetFunction(next, PETSC_NULL, SNESFASGalerkinDefaultFunction, next);CHKERRQ(ierr);
    if (fas->smoothd && fas->level != fas->levels - 1) ierr = SNESSetFunction(fas->smoothd, PETSC_NULL, SNESFASGalerkinDefaultFunction, snes);CHKERRQ(ierr);
    if (fas->smoothu && fas->level != fas->levels - 1) ierr = SNESSetFunction(fas->smoothu, PETSC_NULL, SNESFASGalerkinDefaultFunction, snes);CHKERRQ(ierr);
  }

  /* sets the down (pre) smoother's default norm and sets it from options */
  if(fas->smoothd){
    if (fas->level == 0) {
      ierr = SNESSetNormType(fas->smoothd, SNES_NORM_NONE);CHKERRQ(ierr);
     } else {
      ierr = SNESSetNormType(fas->smoothd, SNES_NORM_FINAL_ONLY);CHKERRQ(ierr);
    }
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)fas->smoothd);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(fas->smoothd);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(snes,&linesearch);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(fas->smoothd,&slinesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPreCheck(linesearch,&lsprefunc,&lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPostCheck(linesearch,&lspostfunc,&lspostctx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPreCheck(slinesearch,lsprefunc,lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPostCheck(slinesearch,lspostfunc,lspostctx);CHKERRQ(ierr);
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)linesearch, (PetscObject)slinesearch);CHKERRQ(ierr);
  }

  /* sets the up (post) smoother's default norm and sets it from options */
  if(fas->smoothu){
    if (fas->level != fas->levels - 1) {
      ierr = SNESSetNormType(fas->smoothu, SNES_NORM_NONE);CHKERRQ(ierr);
    } else {
      ierr = SNESSetNormType(fas->smoothu, SNES_NORM_FINAL_ONLY);CHKERRQ(ierr);
    }
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)fas->smoothu);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(fas->smoothu);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(snes,&linesearch);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(fas->smoothu,&slinesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPreCheck(linesearch,&lsprefunc,&lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPostCheck(linesearch,&lspostfunc,&lspostctx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPreCheck(slinesearch,lsprefunc,lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPostCheck(slinesearch,lspostfunc,lspostctx);CHKERRQ(ierr);
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)linesearch, (PetscObject)slinesearch);CHKERRQ(ierr);
  }

  if (next) {
    /* gotta set up the solution vector for this to work */
    if (!next->vec_sol) {ierr = SNESFASCreateCoarseVec(snes,&next->vec_sol);CHKERRQ(ierr);}
    if (!next->vec_rhs) {ierr = SNESFASCreateCoarseVec(snes,&next->vec_rhs);CHKERRQ(ierr);}
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)next);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(snes,&linesearch);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(fas->next,&slinesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPreCheck(linesearch,&lsprefunc,&lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPostCheck(linesearch,&lspostfunc,&lspostctx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPreCheck(slinesearch,lsprefunc,lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPostCheck(slinesearch,lspostfunc,lspostctx);CHKERRQ(ierr);
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)linesearch, (PetscObject)slinesearch);CHKERRQ(ierr);
    ierr = SNESSetUp(next);CHKERRQ(ierr);
  }
  /* setup FAS work vectors */
  if (fas->galerkin) {
    ierr = VecDuplicate(snes->vec_sol, &fas->Xg);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &fas->Fg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_FAS"
PetscErrorCode SNESSetFromOptions_FAS(SNES snes)
{
  SNES_FAS       *fas = (SNES_FAS *) snes->data;
  PetscInt       levels = 1;
  PetscBool      flg = PETSC_FALSE, upflg = PETSC_FALSE, downflg = PETSC_FALSE, monflg = PETSC_FALSE, galerkinflg = PETSC_FALSE;
  PetscErrorCode ierr;
  char           monfilename[PETSC_MAX_PATH_LEN];
  SNESFASType    fastype;
  const char     *optionsprefix;
  SNESLineSearch linesearch;
  PetscInt       m, n_up, n_down;
  SNES           next;
  PetscBool      isFine;

  PetscFunctionBegin;
  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  ierr = PetscOptionsHead("SNESFAS Options-----------------------------------");CHKERRQ(ierr);

  /* number of levels -- only process most options on the finest level */
  if (isFine) {
    ierr = PetscOptionsInt("-snes_fas_levels", "Number of Levels", "SNESFASSetLevels", levels, &levels, &flg);CHKERRQ(ierr);
    if (!flg && snes->dm) {
      ierr = DMGetRefineLevel(snes->dm,&levels);CHKERRQ(ierr);
      levels++;
      fas->usedmfornumberoflevels = PETSC_TRUE;
    }
    ierr = SNESFASSetLevels(snes, levels, PETSC_NULL);CHKERRQ(ierr);
    fastype = fas->fastype;
    ierr = PetscOptionsEnum("-snes_fas_type","FAS correction type","SNESFASSetType",SNESFASTypes,(PetscEnum)fastype,(PetscEnum*)&fastype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESFASSetType(snes, fastype);CHKERRQ(ierr);
    }

    ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_fas_cycles","Number of cycles","SNESFASSetCycles",fas->n_cycles,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESFASSetCycles(snes, m);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBool("-snes_fas_galerkin", "Form coarse problems with Galerkin","SNESFASSetGalerkin",fas->galerkin,&galerkinflg,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESFASSetGalerkin(snes, galerkinflg);CHKERRQ(ierr);
    }

    ierr = PetscOptionsInt("-snes_fas_smoothup","Number of post-smoothing steps","SNESFASSetNumberSmoothUp",fas->max_up_it,&n_up,&upflg);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-snes_fas_smoothdown","Number of pre-smoothing steps","SNESFASSetNumberSmoothDown",fas->max_down_it,&n_down,&downflg);CHKERRQ(ierr);

    ierr = PetscOptionsString("-snes_fas_monitor","Monitor FAS progress","SNESFASSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&monflg);CHKERRQ(ierr);
    if (monflg) ierr = SNESFASSetMonitor(snes, PETSC_TRUE);CHKERRQ(ierr);
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
      ierr = SNESGetSNESLineSearch(snes, &linesearch);CHKERRQ(ierr);
      ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHL2);CHKERRQ(ierr);
    }
  }

  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  /* recursive option setting for the smoothers */
  if (next) {ierr = SNESSetFromOptions(next);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_FAS"
PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer)
{
  SNES_FAS       *fas = (SNES_FAS *) snes->data;
  PetscBool      isFine, iascii;
  PetscInt       i;
  PetscErrorCode ierr;
  SNES           smoothu, smoothd, levelsnes;

  PetscFunctionBegin;
  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  if (isFine) {
    ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer, "FAS: type is %s, levels=%D, cycles=%D\n",  SNESFASTypes[fas->fastype], fas->levels, fas->n_cycles);CHKERRQ(ierr);
      if (fas->galerkin) {
        ierr = PetscViewerASCIIPrintf(viewer,"    Using Galerkin computed coarse grid function evaluation\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"    Not using Galerkin computed coarse grid function evaluation\n");CHKERRQ(ierr);
      }
      for (i=0; i<fas->levels; i++) {
        ierr = SNESFASGetCycleSNES(snes, i, &levelsnes);CHKERRQ(ierr);
        ierr = SNESFASCycleGetSmootherUp(levelsnes, &smoothu);CHKERRQ(ierr);
        ierr = SNESFASCycleGetSmootherDown(levelsnes, &smoothd);CHKERRQ(ierr);
        if (!i) {
          ierr = PetscViewerASCIIPrintf(viewer,"Coarse grid solver -- level %D -------------------------------\n",i);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"Down solver (pre-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = SNESView(smoothd,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        if (i && (smoothd == smoothu)) {
          ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) same as down solver (pre-smoother)\n");CHKERRQ(ierr);
        } else if (i) {
          ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = SNESView(smoothu,viewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
      }
    } else {
      SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for SNESFAS",((PetscObject)viewer)->type_name);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASDownSmooth_Private"
/*
Defines the action of the downsmoother
 */
PetscErrorCode SNESFASDownSmooth_Private(SNES snes, Vec B, Vec X, Vec F, PetscReal *fnorm)
{
  PetscErrorCode      ierr = 0;
  SNESConvergedReason reason;
  Vec                 FPC;
  SNES                smoothd;
  PetscFunctionBegin;
  ierr = SNESFASCycleGetSmootherDown(snes, &smoothd);CHKERRQ(ierr);
  ierr = SNESSetInitialFunction(smoothd, F);CHKERRQ(ierr);
  ierr = SNESSetInitialFunctionNorm(smoothd, *fnorm);CHKERRQ(ierr);
  ierr = SNESSolve(smoothd, B, X);CHKERRQ(ierr);
  /* check convergence reason for the smoother */
  ierr = SNESGetConvergedReason(smoothd,&reason);CHKERRQ(ierr);
  if (reason < 0 && !(reason == SNES_DIVERGED_MAX_IT || reason == SNES_DIVERGED_LOCAL_MIN)) {
    snes->reason = SNES_DIVERGED_INNER;
    PetscFunctionReturn(0);
  }
  ierr = SNESGetFunction(smoothd, &FPC, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = VecCopy(FPC, F);CHKERRQ(ierr);
  ierr = SNESGetFunctionNorm(smoothd, fnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASUpSmooth_Private"
/*
Defines the action of the upsmoother
 */
PetscErrorCode SNESFASUpSmooth_Private (SNES snes, Vec B, Vec X, Vec F, PetscReal *fnorm) {
  PetscErrorCode      ierr = 0;
  SNESConvergedReason reason;
  Vec                 FPC;
  SNES                smoothu;
  PetscFunctionBegin;

  ierr = SNESFASCycleGetSmootherUp(snes, &smoothu);CHKERRQ(ierr);
  ierr = SNESSolve(smoothu, B, X);CHKERRQ(ierr);
  /* check convergence reason for the smoother */
  ierr = SNESGetConvergedReason(smoothu,&reason);CHKERRQ(ierr);
  if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
    snes->reason = SNES_DIVERGED_INNER;
    PetscFunctionReturn(0);
  }
  ierr = SNESGetFunction(smoothu, &FPC, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = VecCopy(FPC, F);CHKERRQ(ierr);
  ierr = SNESGetFunctionNorm(smoothu, fnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCreateCoarseVec"
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
  SNES_FAS       *fas = (SNES_FAS*)snes->data;

  PetscFunctionBegin;
  if (fas->rscale) {ierr = VecDuplicate(fas->rscale,Xcoarse);CHKERRQ(ierr);}
  else if (fas->inject) {ierr = MatGetVecs(fas->inject,Xcoarse,PETSC_NULL);CHKERRQ(ierr);}
  else SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set restriction or injection");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASRestrict"
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
  SNES_FAS       *fas = (SNES_FAS*)fine->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fine,SNES_CLASSID,1);
  PetscValidHeaderSpecific(Xfine,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Xcoarse,VEC_CLASSID,3);
  if (fas->inject) {
    ierr = MatRestrict(fas->inject,Xfine,Xcoarse);CHKERRQ(ierr);
  } else {
    ierr = MatRestrict(fas->restrct,Xfine,Xcoarse);CHKERRQ(ierr);
    ierr = VecPointwiseMult(Xcoarse,fas->rscale,Xcoarse);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCoarseCorrection"
/*

Performs the FAS coarse correction as:

fine problem: F(x) = 0
coarse problem: F^c(x) = b^c

b^c = F^c(I^c_fx^f - I^c_fF(x))

 */
PetscErrorCode SNESFASCoarseCorrection(SNES snes, Vec X, Vec F, Vec X_new) {
  PetscErrorCode      ierr;
  Vec                 X_c, Xo_c, F_c, B_c;
  SNESConvergedReason reason;
  SNES                next;
  Mat                 restrct, interpolate;
  PetscFunctionBegin;
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  if (next) {
    ierr = SNESFASCycleGetRestriction(snes, &restrct);CHKERRQ(ierr);
    ierr = SNESFASCycleGetInterpolation(snes, &interpolate);CHKERRQ(ierr);

    X_c  = next->vec_sol;
    Xo_c = next->work[0];
    F_c  = next->vec_func;
    B_c  = next->vec_rhs;

    ierr = SNESFASRestrict(snes,X,Xo_c);CHKERRQ(ierr);
    /* restrict the defect */
    ierr = MatRestrict(restrct, F, B_c);CHKERRQ(ierr);
    /* solve the coarse problem corresponding to F^c(x^c) = b^c = F^c(Rx) - R(F(x) - b) */
    ierr = SNESComputeFunction(next, Xo_c, F_c);CHKERRQ(ierr);
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
    ierr = MatInterpolateAdd(interpolate, X_c, X, X_new);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycle_Additive"
/*

The additive cycle looks like:

xhat = x
xhat = dS(x, b)
x = coarsecorrection(xhat, b_d)
x = x + nu*(xhat - x);
(optional) x = uS(x, b)

With the coarse RHS (defect correction) as below.

 */
PetscErrorCode SNESFASCycle_Additive(SNES snes, Vec X) {
  Vec                 F, B, Xhat;
  Vec                 X_c, Xo_c, F_c, B_c;
  PetscErrorCode      ierr;
  SNESConvergedReason reason;
  PetscReal           xnorm, fnorm, ynorm;
  PetscBool           lssuccess;
  SNES                next;
  Mat                 restrct, interpolate;
  PetscFunctionBegin;
  ierr = SNESFASCycleGetCorrection(snes, &next);CHKERRQ(ierr);
  F = snes->vec_func;
  B = snes->vec_rhs;
  Xhat = snes->work[1];
  ierr = VecCopy(X, Xhat);CHKERRQ(ierr);
  /* recurse first */
  if (next) {
    ierr = SNESFASCycleGetRestriction(snes, &restrct);CHKERRQ(ierr);
    ierr = SNESFASCycleGetInterpolation(snes, &interpolate);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, Xhat, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    X_c  = next->vec_sol;
    Xo_c = next->work[0];
    F_c  = next->vec_func;
    B_c  = next->vec_rhs;

    ierr = SNESFASRestrict(snes,Xhat,Xo_c);CHKERRQ(ierr);
    /* restrict the defect */
    ierr = MatRestrict(restrct, F, B_c);CHKERRQ(ierr);

    /* solve the coarse problem corresponding to F^c(x^c) = b^c = Rb + F^c(Rx) - RF(x) */
    ierr = SNESComputeFunction(next, Xo_c, F_c);CHKERRQ(ierr);
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
    ierr = SNESLineSearchGetSuccess(snes->linesearch, &lssuccess);CHKERRQ(ierr);
    if (!lssuccess) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
    ierr = SNESLineSearchGetNorms(snes->linesearch, &xnorm, &snes->norm, &ynorm);CHKERRQ(ierr);
  } else {
    ierr = SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycle_Multiplicative"
/*

Defines the FAS cycle as:

fine problem: F(x) = 0
coarse problem: F^c(x) = b^c

b^c = F^c(I^c_fx^f - I^c_fF(x))

correction:

x = x + I(x^c - Rx)

 */
PetscErrorCode SNESFASCycle_Multiplicative(SNES snes, Vec X) {

  PetscErrorCode      ierr;
  Vec                 F,B;
  SNES_FAS            *fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  /* pre-smooth -- just update using the pre-smoother */
  ierr = SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm);CHKERRQ(ierr);
  if (fas->level != 0) {
    ierr = SNESFASCoarseCorrection(snes, X, F, X);CHKERRQ(ierr);
    ierr = SNESFASUpSmooth_Private(snes, B, X, F, &snes->norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_FAS"

PetscErrorCode SNESSolve_FAS(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       i, maxits;
  Vec            X, F;
  PetscReal      fnorm;
  SNES_FAS       *fas = (SNES_FAS *)snes->data,*ffas;
  DM             dm;
  PetscBool      isFine;

  PetscFunctionBegin;
  maxits = snes->max_its;            /* maximum number of iterations */
  snes->reason = SNES_CONVERGED_ITERATING;
  X = snes->vec_sol;
  F = snes->vec_func;

  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  /*norm setup */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
  } else {
    snes->vec_func_init_set = PETSC_FALSE;
  }

  if (!snes->norm_init_set) {
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
    if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  } else {
    fnorm = snes->norm_init;
    snes->norm_init_set = PETSC_FALSE;
  }

  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
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
      dm = dmcoarse;
    }
  }

  for (i = 0; i < maxits; i++) {
    /* Call general purpose update function */

    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    if (fas->fastype == SNES_FAS_MULTIPLICATIVE) {
      ierr = SNESFASCycle_Multiplicative(snes, X);CHKERRQ(ierr);
    } else {
      ierr = SNESFASCycle_Additive(snes, X);CHKERRQ(ierr);
    }

    /* check for FAS cycle divergence */
    if (snes->reason != SNES_CONVERGED_ITERATING) {
      PetscFunctionReturn(0);
    }

    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (isFine) {
      ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,snes->norm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
      if (snes->reason) break;
    }
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}
