
#include <petsc/private/taolinesearchimpl.h>

static PetscErrorCode TaoLineSearchDestroy_Unit(TaoLineSearch ls)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchSetFromOptions_Unit(PetscOptionItems *PetscOptionsObject,TaoLineSearch ls)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchView_Unit(TaoLineSearch ls,PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer,"  Line Search: Unit Step %g.\n",(double)ls->initstep));
  PetscFunctionReturn(0);
}

/* Take unit step (newx = startx + initstep*step_direction) */
static PetscErrorCode TaoLineSearchApply_Unit(TaoLineSearch ls,Vec x,PetscReal *f,Vec g,Vec step_direction)
{
  PetscFunctionBegin;
  PetscCall(TaoLineSearchMonitor(ls,0,*f,0.0));
  ls->step = ls->initstep;
  PetscCall(VecAXPY(x,ls->step,step_direction));
  PetscCall(TaoLineSearchComputeObjectiveAndGradient(ls,x,f,g));
  PetscCall(TaoLineSearchMonitor(ls,1,*f,ls->step));
  ls->reason = TAOLINESEARCH_SUCCESS;
  PetscFunctionReturn(0);
}

/*MC
   TAOLINESEARCHUNIT - Line-search type that disables line search and accepts the unit step length every time

  Options Database Keys:
. -tao_ls_stepinit <step> - steplength

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
