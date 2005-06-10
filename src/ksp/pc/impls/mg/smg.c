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
    ierr = MatRestrict(mg[i]->restrct,mg[i]->b,mg[i-1]->b);CHKERRQ(ierr);
  }
  /* solve seperately on each level */
  for (i=0; i<l; i++) {
    ierr = VecSet(mg[i]->x,0.0);CHKERRQ(ierr); 
    if (mg[i]->eventsolve) {ierr = PetscLogEventBegin(mg[i]->eventsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mg[i]->smoothd,mg[i]->b,mg[i]->x);CHKERRQ(ierr); 
    if (mg[i]->eventsolve) {ierr = PetscLogEventEnd(mg[i]->eventsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  for (i=1; i<l; i++) {  
    ierr = MatInterpolateAdd(mg[i]->interpolate,mg[i-1]->x,mg[i]->x,mg[i]->x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
