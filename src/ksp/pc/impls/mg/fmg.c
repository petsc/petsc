/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGFCycle_Private(PC pc,PC_MG_Levels **mglevels,PetscBool transpose)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  if (!transpose) {
    /* restrict the RHS through all levels to coarsest. */
    for (i=l-1; i>0; i--) {
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr);
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    }

    /* work our way up through the levels */
    ierr = VecSet(mglevels[0]->x,0.0);CHKERRQ(ierr);
    for (i=0; i<l-1; i++) {
      ierr = PCMGMCycle_Private(pc,&mglevels[i],transpose,NULL);CHKERRQ(ierr);
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      ierr = MatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->x,mglevels[i+1]->x);CHKERRQ(ierr);
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    }
    ierr = PCMGMCycle_Private(pc,&mglevels[l-1],transpose,NULL);CHKERRQ(ierr);
  } else {
    ierr = PCMGMCycle_Private(pc,&mglevels[l-1],transpose,NULL);CHKERRQ(ierr);
    for (i=l-2; i>=0; i--) {
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      ierr = MatRestrict(mglevels[i+1]->interpolate,mglevels[i+1]->x,mglevels[i]->x);CHKERRQ(ierr);
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      ierr = PCMGMCycle_Private(pc,&mglevels[i],transpose,NULL);CHKERRQ(ierr);
    }
    for (i=1; i<l; i++) {
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      ierr = MatInterpolate(mglevels[i]->restrct,mglevels[i-1]->b,mglevels[i]->b);CHKERRQ(ierr);
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGKCycle_Private(PC pc,PC_MG_Levels **mglevels,PetscBool transpose)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--) {
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr);
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }

  /* work our way up through the levels */
  ierr = VecSet(mglevels[0]->x,0.0);CHKERRQ(ierr);
  for (i=0; i<l-1; i++) {
    if (mglevels[i]->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mglevels[i]->smoothd,mglevels[i]->b,mglevels[i]->x);CHKERRQ(ierr);
    ierr = KSPCheckSolve(mglevels[i]->smoothd,pc,mglevels[i]->x);CHKERRQ(ierr);
    if (mglevels[i]->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->x,mglevels[i+1]->x);CHKERRQ(ierr);
    if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  if (mglevels[l-1]->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels[l-1]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSolve(mglevels[l-1]->smoothd,mglevels[l-1]->b,mglevels[l-1]->x);CHKERRQ(ierr);
  ierr = KSPCheckSolve(mglevels[l-1]->smoothd,pc,mglevels[l-1]->x);CHKERRQ(ierr);
  if (mglevels[l-1]->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels[l-1]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


