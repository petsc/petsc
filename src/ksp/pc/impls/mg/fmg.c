/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include "../src/ksp/pc/impls/mg/mgimpl.h"

EXTERN PetscErrorCode PCMGMCycle_Private(PC,PC_MG_Levels **,PCRichardsonConvergedReason*);

#undef __FUNCT__  
#define __FUNCT__ "PCMGFCycle_Private"
PetscErrorCode PCMGFCycle_Private(PC pc,PC_MG_Levels **mglevels)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--){
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr);
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  
  /* work our way up through the levels */
  ierr = VecSet(mglevels[0]->x,0.0);CHKERRQ(ierr);
  for (i=0; i<l-1; i++) {
    ierr = PCMGMCycle_Private(pc,&mglevels[i],PETSC_NULL);CHKERRQ(ierr);
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->x,mglevels[i+1]->x);CHKERRQ(ierr); 
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  ierr = PCMGMCycle_Private(pc,&mglevels[l-1],PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGKCycle_Private"
PetscErrorCode PCMGKCycle_Private(PC pc,PC_MG_Levels **mglevels)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--){
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr); 
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  
  /* work our way up through the levels */
  ierr = VecSet(mglevels[0]->x,0.0);CHKERRQ(ierr); 
  for (i=0; i<l-1; i++) {
    if (mg->eventsmoothsolve) {ierr = PetscLogEventBegin(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mglevels[i]->smoothd,mglevels[i]->b,mglevels[i]->x);CHKERRQ(ierr);
    if (mg->eventsmoothsolve) {ierr = PetscLogEventEnd(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->x,mglevels[i+1]->x);CHKERRQ(ierr);
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  if (mg->eventsmoothsolve) {ierr = PetscLogEventBegin(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSolve(mglevels[l-1]->smoothd,mglevels[l-1]->b,mglevels[l-1]->x);CHKERRQ(ierr);
  if (mg->eventsmoothsolve) {ierr = PetscLogEventEnd(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}


