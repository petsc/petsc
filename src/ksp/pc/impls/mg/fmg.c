/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include "src/ksp/pc/impls/mg/mgimpl.h"

EXTERN PetscErrorCode PCMGMCycle_Private(PC_MG **,PetscTruth*);

#undef __FUNCT__  
#define __FUNCT__ "PCMGFCycle_Private"
PetscErrorCode PCMGFCycle_Private(PC_MG **mg)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mg[0]->levels;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--){
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mg[i]->restrct,mg[i]->b,mg[i-1]->b);CHKERRQ(ierr);
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  
  /* work our way up through the levels */
  ierr = VecSet(mg[0]->x,0.0);CHKERRQ(ierr);
  for (i=0; i<l-1; i++) {
    ierr = PCMGMCycle_Private(&mg[i],PETSC_NULL);CHKERRQ(ierr);
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolate(mg[i+1]->interpolate,mg[i]->x,mg[i+1]->x);CHKERRQ(ierr); 
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  ierr = PCMGMCycle_Private(&mg[l-1],PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGKCycle_Private"
PetscErrorCode PCMGKCycle_Private(PC_MG **mg)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mg[0]->levels;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--){
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mg[i]->restrct,mg[i]->b,mg[i-1]->b);CHKERRQ(ierr); 
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  
  /* work our way up through the levels */
  ierr = VecSet(mg[0]->x,0.0);CHKERRQ(ierr); 
  for (i=0; i<l-1; i++) {
    if (mg[i]->eventsmoothsolve) {ierr = PetscLogEventBegin(mg[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mg[i]->smoothd,mg[i]->b,mg[i]->x);CHKERRQ(ierr);
    if (mg[i]->eventsmoothsolve) {ierr = PetscLogEventEnd(mg[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolate(mg[i+1]->interpolate,mg[i]->x,mg[i+1]->x);CHKERRQ(ierr);
    if (mg[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mg[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  if (mg[l-1]->eventsmoothsolve) {ierr = PetscLogEventBegin(mg[l-1]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSolve(mg[l-1]->smoothd,mg[l-1]->b,mg[l-1]->x);CHKERRQ(ierr);
  if (mg[l-1]->eventsmoothsolve) {ierr = PetscLogEventEnd(mg[l-1]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}


