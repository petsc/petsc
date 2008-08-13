#include "petscvec.h"
#include "taosolver.h"
#include "private/taolinesearch_impl.h"

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchDestroy_Unit"
static PetscErrorCode TaoLineSearchDestroy_Unit(TaoLineSearch ls)
{
  PetscErrorCode info;
  PetscFunctionBegin;
  info = PetscFree(ls->data); CHKERRQ(info);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchSetFromOptions_Unit"
static PetscErrorCode TaoLineSearchSetFromOptions_Unit(TaoLineSearch ls)
{
  PetscErrorCode info;
  PetscFunctionBegin;
  info = PetscOptionsHead("No Unit line search options");CHKERRQ(info);
  info = PetscOptionsTail();CHKERRQ(info);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchView_Unit"
static PetscErrorCode TaoLineSearchView_Unit(TaoLineSearch ls,PetscViewer pv)
{
  
  PetscErrorCode info;
  PetscTruth isascii;

  PetscFunctionBegin;
  
  info = PetscTypeCompare((PetscObject)pv, PETSC_VIEWER_ASCII, &isascii); CHKERRQ(info);
  if (isascii) {
      info=PetscViewerASCIIPrintf(pv,"  Line Search: Unit Step.\n");CHKERRQ(info);
  } else {
      SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for Unit TaoLineSearch.",((PetscObject)pv)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchApply_Unit"
/* @ TaoApply_LineSearch - This routine takes step length of 1.0.

   Input Parameters:
+  tao - TaoSolver context
.  X - current iterate (on output X contains new iterate, X + step*S)
.  f - objective function evaluated at X
.  G - gradient evaluated at X
-  D - search direction


   Info is set to 0.

@ */
static PetscErrorCode TaoLineSearchApply_Unit(TaoLineSearch ls,Vec start_x,PetscScalar start_f,Vec start_g,Vec step_direction)
{
  PetscErrorCode   info;
  PetscScalar ftry;
  //  Vec XL,XU; 

  PetscFunctionBegin;
  
  // Take unit step (newx = startx + 1.0*step_direction)
  info = VecCopy(start_x, ls->new_x);
  info = VecAXPY(ls->new_x,1.0,step_direction);CHKERRQ(info);

  //info = TaoGetVariableBounds(tao,&XL,&XU); CHKERRQ(info);
  //  if (XL && XU){
  //    info = X->Median(XL,X,XU);CHKERRQ(info);
  //  }
  info = TaoLineSearchComputeObjectiveGradient(ls,ls->new_x,&ftry,ls->new_g); CHKERRQ(info);
  info = PetscInfo1(ls,"Tao Apply Unit Step: %4.4e\n",1.0);
         CHKERRQ(info);
  if (start_f < ftry){
    info = PetscInfo2(ls,"Tao Apply Unit Step, FINCREASE: F old:= %12.10e, F new: %12.10e\n",start_f,ftry); CHKERRQ(info);
  }
  ls->new_f=ftry;
  ls->step_length = 1.0;
  
  //  *f_full = fnew;
  //  *info2 = 0;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchCreate_Unit"
/*@C
   TaoCreateUnitLineSearch - Always use step length of 1.0

   Input Parameters:
.  tao - TaoSolver context


   Level: advanced

.keywords: TaoSolver, linesearch
@*/
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate_Unit(TaoLineSearch ls)
{
  PetscErrorCode info;

  PetscFunctionBegin;
  ls->ops->setup = 0;
  ls->ops->apply = TaoLineSearchApply_Unit;
  ls->ops->view = TaoLineSearchView_Unit;
  ls->ops->destroy = TaoLineSearchDestroy_Unit;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_Unit;

  PetscFunctionReturn(0);
}
EXTERN_C_END

