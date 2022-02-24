
#include <petsc/private/taolinesearchimpl.h>

static PetscErrorCode TaoLineSearchDestroy_Unit(TaoLineSearch ls)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(ls->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchSetFromOptions_Unit(PetscOptionItems *PetscOptionsObject,TaoLineSearch ls)
{
  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"No Unit line search options"));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchView_Unit(TaoLineSearch ls,PetscViewer viewer)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Line Search: Unit Step.\n"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchApply_Unit(TaoLineSearch ls,Vec x,PetscReal *f,Vec g,Vec step_direction)
{
  PetscReal      ftry;
  PetscReal      startf = *f;

  PetscFunctionBegin;
  /* Take unit step (newx = startx + 1.0*step_direction) */
  CHKERRQ(TaoLineSearchMonitor(ls, 0, *f, 0.0));
  CHKERRQ(VecAXPY(x,1.0,step_direction));
  CHKERRQ(TaoLineSearchComputeObjectiveAndGradient(ls,x,&ftry,g));
  CHKERRQ(TaoLineSearchMonitor(ls, 1, *f, 1.0));
  CHKERRQ(PetscInfo(ls,"Tao Apply Unit Step: %4.4e\n",1.0));
  if (startf < ftry) {
    CHKERRQ(PetscInfo(ls,"Tao Apply Unit Step, FINCREASE: F old:= %12.10e, F new: %12.10e\n",(double)startf,(double)ftry));
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
  ls->ops->monitor = NULL;
  ls->ops->apply = TaoLineSearchApply_Unit;
  ls->ops->view = TaoLineSearchView_Unit;
  ls->ops->destroy = TaoLineSearchDestroy_Unit;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_Unit;
  PetscFunctionReturn(0);
}
