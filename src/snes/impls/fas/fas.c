/* Defines the basic SNES object */
#include <../src/snes/impls/fas/fasimpls.h>

/*MC
Full Approximation Scheme nonlinear multigrid solver.

The nonlinear problem is solved via the repeated application of nonlinear preconditioners and coarse-grid corrections

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

extern PetscErrorCode SNESDestroy_FAS(SNES snes);
extern PetscErrorCode SNESSetUp_FAS(SNES snes);
extern PetscErrorCode SNESSetFromOptions_FAS(SNES snes);
extern PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer);
extern PetscErrorCode SNESSolve_FAS(SNES snes);
extern PetscErrorCode SNESReset_FAS(SNES snes);

EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "SNESCreate_FAS"
PetscErrorCode SNESCreate_FAS(SNES snes)
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

  ierr = PetscNewLog(snes, SNES_FAS, &fas);CHKERRQ(ierr);
  snes->data = (void*) fas;
  fas->level = 0;
  fas->presmooth  = PETSC_NULL;
  fas->postsmooth = PETSC_NULL;
  fas->next = PETSC_NULL;
  fas->interpolate = PETSC_NULL;
  fas->restrct = PETSC_NULL;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetLevels"
PetscErrorCode SNESFASGetLevels(SNES snes, PetscInt * levels) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscFunctionBegin;
  *levels = fas->level;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetSNES"
PetscErrorCode SNESFASGetSNES(SNES snes, PetscInt level, SNES * lsnes) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscInt levels = fas->level;
  PetscInt i;
  PetscFunctionBegin;
  *lsnes = snes;
  if (fas->level < level) {
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "SNESFASGetSNES should only be called on a finer SNESFAS instance than the level.");
  }
  if (level > levels - 1) {
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Level %d doesn't exist in the SNESFAS.");
  }
  for (i = fas->level; i > level; i--) {
    *lsnes = fas->next;
    fas = (SNES_FAS *)(*lsnes)->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "SNESFASGetSNES didn't return the right level!");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetLevels"
PetscErrorCode SNESFASSetLevels(SNES snes, PetscInt levels, MPI_Comm * comms) {
  PetscErrorCode ierr;
  PetscInt i;
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  MPI_Comm comm;

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  /* user has changed the number of levels; reset */
  ierr = SNESReset(snes);CHKERRQ(ierr);
  /* destroy any coarser levels if necessary */
  if (fas->next) SNESDestroy(&fas->next);CHKERRQ(ierr);
  /* setup the finest level */
  for (i = levels - 1; i >= 0; i--) {
    if (comms) comm = comms[i];
    fas->level = i;
    fas->levels = levels;
    if (i > 0) {
      ierr = SNESCreate(comm, &fas->next);CHKERRQ(ierr);
      ierr = SNESSetType(fas->next, SNESFAS);CHKERRQ(ierr);
      ierr = SNESSetOptionsPrefix(fas->next, "fas_");CHKERRQ(ierr);
      ierr = SNESSetFromOptions(fas->next);CHKERRQ(ierr);
      fas = (SNES_FAS *)fas->next->data;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetInterpolation"
PetscErrorCode SNESFASSetInterpolation(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetInterpolation", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetInterpolation");
  fas->interpolate = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetRestriction"
PetscErrorCode SNESFASSetRestriction(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetRestriction", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetRestriction");
  fas->restrct = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetRScale"
PetscErrorCode SNESFASSetRScale(SNES snes, PetscInt level, Vec rscale) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetRestriction", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetRestriction");
  fas->rscale = rscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_FAS"
PetscErrorCode SNESReset_FAS(SNES snes)
{
  PetscErrorCode ierr = 0;
  SNES_FAS * fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  /* destroy local data created in SNESSetup_FAS */
  if (fas->presmooth)    ierr = SNESDestroy(&fas->presmooth);CHKERRQ(ierr);
  if (fas->postsmooth)   ierr = SNESDestroy(&fas->postsmooth);CHKERRQ(ierr);
  if (fas->interpolate)  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
  if (fas->restrct)      ierr = MatDestroy(&fas->restrct);CHKERRQ(ierr);
  if (fas->rscale)       ierr = VecDestroy(&fas->rscale);CHKERRQ(ierr);

  /* recurse -- reset should NOT destroy the structures -- destroy should destroy the structures recursively */
  if (fas->next) ierr = SNESReset(fas->next);CHKERRQ(ierr);
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_FAS"
PetscErrorCode SNESDestroy_FAS(SNES snes)
{
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* recursively resets and then destroys */
  ierr = SNESReset_FAS(snes);CHKERRQ(ierr);
  if (fas->next) ierr = SNESDestroy(&fas->next);CHKERRQ(ierr);
  ierr = PetscFree(fas);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_FAS"
PetscErrorCode SNESSetUp_FAS(SNES snes)
{
  SNES_FAS   *fas = (SNES_FAS *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes, 4);CHKERRQ(ierr); /* the four work vectors are used to transfer stuff BACK */
  /* gets the solver ready for solution */
  if (snes->dm) {
    /* construct EVERYTHING from the DM -- including the progressive set of smoothers */
    if (fas->next) {
      /* for now -- assume the DM and the evaluation functions have been set externally */
      if (!fas->interpolate) {
        if (!fas->next->dm) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONGSTATE, "All levels must be assigned a DM");
        /* set the interpolation and restriction from the DM */
        ierr = DMGetInterpolation(fas->next->dm, snes->dm, &fas->interpolate, &fas->rscale);CHKERRQ(ierr);
        fas->restrct = fas->interpolate;
      }
      /* TODO LATER: Preconditioner setup goes here */
    }
  }
  if (fas->next) {
    /* gotta set up the solution vector for this to work */
    ierr = VecDuplicate(fas->rscale, &fas->next->vec_sol);CHKERRQ(ierr);
    ierr = SNESSetUp(fas->next);CHKERRQ(ierr);
  }
  /* got to set them all up at once */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_FAS"
PetscErrorCode SNESSetFromOptions_FAS(SNES snes)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* types for the pre and postsmoothers */
  ierr = PetscOptionsHead("SNESFAS Options-----------------------------------");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_FAS"
PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer)
{
  SNES_FAS   *fas = (SNES_FAS *) snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;
  PetscInt levels = fas->levels;
  PetscInt i;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "FAS, levels = %d\n",  fas->levels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);
    for (i = levels - 1; i >= 0; i--) {
      ierr = PetscViewerASCIIPrintf(viewer, "level: %d\n",  fas->level);CHKERRQ(ierr);
      if (fas->presmooth) {
        ierr = PetscViewerASCIIPrintf(viewer, "pre-smoother on level %D\n",  fas->level);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);
        ierr = SNESView(fas->presmooth, viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "no pre-smoother on level %D\n",  fas->level);CHKERRQ(ierr);
      }
      if (fas->postsmooth) {
        ierr = PetscViewerASCIIPrintf(viewer, "post-smoother on level %D\n",  fas->level);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);
        ierr = SNESView(fas->postsmooth, viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "no post-smoother on level %D\n",  fas->level);CHKERRQ(ierr);
      }
      if (fas->next) fas = (SNES_FAS *)fas->next->data;
    }
    ierr = PetscViewerASCIIPopTab(viewer);
  } else {
    SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for SNESFAS",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*

Defines the FAS cycle as:

fine problem: F(x) = 0
coarse problem: F^c(x) = b^c

b^c = F^c(I^c_fx^f - I^c_fF(x))

correction:

x = x + I(x^c - Rx)

 */

#undef __FUNCT__
#define __FUNCT__ "FASCycle_Private"
PetscErrorCode FASCycle_Private(SNES snes, Vec B, Vec X, Vec F) {

  PetscErrorCode ierr;
  Vec X_c, Xo_c, F_c, B_c;
  SNES_FAS * fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  /* pre-smooth -- just update using the pre-smoother */
  if (fas->presmooth) {
    ierr = SNESSolve(fas->presmooth, B, X);CHKERRQ(ierr);
  } else if (snes->pc) {
    ierr = SNESSolve(snes->pc, B, X);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  if (fas->next) {
    X_c  = fas->next->vec_sol;
    Xo_c = fas->next->work[1];
    F_c  = fas->next->vec_func;
    B_c  = fas->next->work[2];
    /* project to coarse */
    ierr = MatRestrict(fas->restrct, X, Xo_c);CHKERRQ(ierr);
    ierr = VecPointwiseMult(Xo_c, fas->rscale, Xo_c);CHKERRQ(ierr);
    ierr = MatRestrict(fas->restrct, F, F_c);CHKERRQ(ierr);
    ierr = VecPointwiseMult(F_c,  fas->rscale, F_c);CHKERRQ(ierr);
    if (B) {
      ierr = MatRestrict(fas->restrct, B, B_c);CHKERRQ(ierr);
      ierr = VecPointwiseMult(B_c,  fas->rscale, B_c);CHKERRQ(ierr);
    } else {
      ierr = VecSet(B_c, 0.0);CHKERRQ(ierr);
    }
    /* solve the coarse problem corresponding to F^c(x^c) = b^c = Rb + F^c(Rx) - RF(x) */
    fas->next->vec_rhs = PETSC_NULL; /*unset the RHS for the next problem so we may evaluate the function rather than the residual */
    ierr = VecAXPY(B_c, -1.0, F_c);CHKERRQ(ierr);                     /* B_c = RB + F(X_c) - F_C */
    ierr = SNESComputeFunction(fas->next, Xo_c, F_c);CHKERRQ(ierr);
    ierr = VecAXPY(B_c, 1.0, F_c);CHKERRQ(ierr);
    ierr = VecCopy(Xo_c, X_c);CHKERRQ(ierr);

    /* test -- initial residuals with and without the RHS set */
    fas->next->vec_rhs = B_c;
    ierr = SNESComputeFunction(fas->next, Xo_c, F_c);CHKERRQ(ierr);
    fas->next->vec_rhs = PETSC_NULL;
    ierr = SNESComputeFunction(fas->next, Xo_c, F_c);CHKERRQ(ierr);

    /* recurse to the next level */
    ierr = FASCycle_Private(fas->next, B_c, X_c, F_c);CHKERRQ(ierr);
    /* correct as x <- x + I(x^c - Rx)*/
    ierr = VecAXPY(X_c, -1.0, Xo_c);CHKERRQ(ierr);
    ierr = MatInterpolate(fas->interpolate, X_c, F);CHKERRQ(ierr);
    ierr = VecAXPY(X, 1.0, F);CHKERRQ(ierr);
  }
  /* post-smooth -- just update using the post-smoother */
  if (fas->level != 0) {
    if (fas->postsmooth) {
      ierr = SNESSolve(fas->postsmooth, B, X);CHKERRQ(ierr);
    } else if (snes->pc) {
      ierr = SNESSolve(snes->pc, B, X);CHKERRQ(ierr);
    }
  }
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FASInitialGuess_Private"


#undef __FUNCT__
#define __FUNCT__ "SNESSolve_FAS"

PetscErrorCode SNESSolve_FAS(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt i, maxits;
  Vec X, F, B;
  PetscReal fnorm;
  PetscFunctionBegin;
  maxits = snes->max_its;	     /* maximum number of iterations */
  snes->reason = SNES_CONVERGED_ITERATING;
  X = snes->vec_sol;
  F = snes->vec_func;
  B = snes->vec_rhs;
  /* initial iteration */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);
  for (i = 0; i < maxits; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    ierr = FASCycle_Private(snes, B, X, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}
