#define PETSCKSP_DLL

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
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--){
    ierr = MatRestrict(mg[i]->restrct,mg[i]->b,mg[i-1]->b);CHKERRQ(ierr);
  }
  
  /* work our way up through the levels */
  ierr = VecSet(mg[0]->x,zero);CHKERRQ(ierr);
  for (i=0; i<l-1; i++) {
    ierr = PCMGMCycle_Private(&mg[i],PETSC_NULL);CHKERRQ(ierr);
    ierr = MatInterpolate(mg[i+1]->interpolate,mg[i]->x,mg[i+1]->x);CHKERRQ(ierr); 
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
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--){
    ierr = MatRestrict(mg[i]->restrct,mg[i]->b,mg[i-1]->b);CHKERRQ(ierr); 
  }
  
  /* work our way up through the levels */
  ierr = VecSet(mg[0]->x,zero);CHKERRQ(ierr); 
  for (i=0; i<l-1; i++) {
    if (mg[i]->eventsolve) {ierr = PetscLogEventBegin(mg[i]->eventsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mg[i]->smoothd,mg[i]->b,mg[i]->x);CHKERRQ(ierr);
    if (mg[i]->eventsolve) {ierr = PetscLogEventEnd(mg[i]->eventsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolate(mg[i+1]->interpolate,mg[i]->x,mg[i+1]->x);CHKERRQ(ierr);
  }
  if (mg[l-1]->eventsolve) {ierr = PetscLogEventBegin(mg[l-1]->eventsolve,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSolve(mg[l-1]->smoothd,mg[l-1]->b,mg[l-1]->x);CHKERRQ(ierr);
  if (mg[l-1]->eventsolve) {ierr = PetscLogEventEnd(mg[l-1]->eventsolve,0,0,0,0);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}


