#define PETSCKSP_DLL

/*
     Additive Multigrid V Cycle routine    
*/
#include "../src/ksp/pc/impls/mg/mgimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "PCMGACycle_Private"
PetscErrorCode PCMGACycle_Private(PC pc,PC_MG_Levels **mglevels)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* compute RHS on each level */
  for (i=l-1; i>0; i--) {
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr);
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  /* solve separately on each level */
  for (i=0; i<l; i++) {
    ierr = VecSet(mglevels[i]->x,0.0);CHKERRQ(ierr); 
    if (mg->eventsmoothsolve) {ierr = PetscLogEventBegin(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mglevels[i]->smoothd,mglevels[i]->b,mglevels[i]->x);CHKERRQ(ierr); 
    if (mg->eventsmoothsolve) {ierr = PetscLogEventEnd(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  for (i=1; i<l; i++) {  
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolateAdd(mglevels[i]->interpolate,mglevels[i-1]->x,mglevels[i]->x,mglevels[i]->x);CHKERRQ(ierr);
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}
