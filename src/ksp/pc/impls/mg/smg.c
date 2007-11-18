#define PETSCKSP_DLL

/*
     Additive Multigrid V Cycle routine    
*/
#include "src/ksp/pc/impls/mg/mgimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "PCMGACycle_Private"
PetscErrorCode PCMGACycle_Private(PC_MG **mg)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mg[0]->levels;

  PetscFunctionBegin;
  /* compute RHS on each level */
  for (i=l-1; i>0; i--) {
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mg[i]->restrct,mg[i]->b,mg[i-1]->b);CHKERRQ(ierr);
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  /* solve separately on each level */
  for (i=0; i<l; i++) {
    ierr = VecSet(mg[i]->x,0.0);CHKERRQ(ierr); 
    if (mg[i]->eventsmoothsolve) {ierr = PetscLogEventBegin(mg[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mg[i]->smoothd,mg[i]->b,mg[i]->x);CHKERRQ(ierr); 
    if (mg[i]->eventsmoothsolve) {ierr = PetscLogEventEnd(mg[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  for (i=1; i<l; i++) {  
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolateAdd(mg[i]->interpolate,mg[i-1]->x,mg[i]->x,mg[i]->x);CHKERRQ(ierr);
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}
