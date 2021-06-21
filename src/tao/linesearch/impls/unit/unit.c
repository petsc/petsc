
#include <petsc/private/taolinesearchimpl.h>

static PetscErrorCode TaoLineSearchDestroy_Unit(TaoLineSearch ls)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(ls->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchSetFromOptions_Unit(PetscOptionItems *PetscOptionsObject,TaoLineSearch ls)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"No Unit line search options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchView_Unit(TaoLineSearch ls,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr=PetscViewerASCIIPrintf(viewer,"  Line Search: Unit Step.\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchApply_Unit(TaoLineSearch ls,Vec x,PetscReal *f,Vec g,Vec step_direction)
{
  PetscErrorCode ierr;
  PetscReal      ftry;
  PetscReal      startf = *f;

  PetscFunctionBegin;
  /* Take unit step (newx = startx + 1.0*step_direction) */
  ierr = TaoLineSearchMonitor(ls, 0, *f, 0.0);CHKERRQ(ierr);
  ierr = VecAXPY(x,1.0,step_direction);CHKERRQ(ierr);
  ierr = TaoLineSearchComputeObjectiveAndGradient(ls,x,&ftry,g);CHKERRQ(ierr);
  ierr = TaoLineSearchMonitor(ls, 1, *f, 1.0);CHKERRQ(ierr);
  ierr = PetscInfo1(ls,"Tao Apply Unit Step: %4.4e\n",1.0);CHKERRQ(ierr);
  if (startf < ftry) {
    ierr = PetscInfo2(ls,"Tao Apply Unit Step, FINCREASE: F old:= %12.10e, F new: %12.10e\n",(double)startf,(double)ftry);CHKERRQ(ierr);
  }
  *f = ftry;
  ls->step = 1.0;
  ls->reason=TAOLINESEARCH_SUCCESS;
  PetscFunctionReturn(0);
}

/*MC
   TAOLINESEARCHUNIT - Line-search type that disables line search and accepts the unit step length every time

   Level: developer

.seealso: TaoLineSearchCreate(), TaoLineSearchSetType(), TaoLineSearchApply()

.keywords: Tao, linesearch
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_Unit(TaoLineSearch ls)
{
  PetscFunctionBegin;
  ls->ops->setup = NULL;
  ls->ops->reset = NULL;
  ls->ops->apply = TaoLineSearchApply_Unit;
  ls->ops->view = TaoLineSearchView_Unit;
  ls->ops->destroy = TaoLineSearchDestroy_Unit;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_Unit;
  PetscFunctionReturn(0);
}

