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
      fas = (SNES_FAS *)fas->next->data;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetInterpolation"
PetscErrorCode SNESFASSetInterpolation(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetInterpolation", level);
  /* get to the correct level */
  for (int i = fas->level; i > level; i--) {
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
  PetscInt top_level = fas->level;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetRestriction", level);
  /* get to the correct level */
  for (int i = fas->level; i > level; i--) {
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
  PetscInt top_level = fas->level;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetRestriction", level);
  /* get to the correct level */
  for (int i = fas->level; i > level; i--) {
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
  } else {
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_SUP, "SNESSetup_FAS presently only works with DM Coarsening.  This will be fixed. ");
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_FAS"
PetscErrorCode SNESSetFromOptions_FAS(SNES snes)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* types for the pre and postsmoothers */
  ierr = PetscOptionsHead("SNESFAS Options");CHKERRQ(ierr);
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
 */
#undef __FUNCT__
#define __FUNCT__ "FASCycle_Private"
PetscErrorCode FASCycle_Private(SNES snes) {
  
  PetscFunctionBegin;
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_FAS"

PetscErrorCode SNESSolve_FAS(SNES snes)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
