/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGFCycle_Private(PC pc,PC_MG_Levels **mglevels,PetscBool transpose,PetscBool matapp)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  if (!transpose) {
    /* restrict the RHS through all levels to coarsest. */
    for (i=l-1; i>0; i--) {
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      if (matapp) { ierr = MatMatRestrict(mglevels[i]->restrct,mglevels[i]->B,&mglevels[i-1]->B);CHKERRQ(ierr); }
      else { ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr); }
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    }

    /* work our way up through the levels */
    if (matapp) {
      if (!mglevels[0]->X) {
        ierr = MatDuplicate(mglevels[0]->B,MAT_DO_NOT_COPY_VALUES,&mglevels[0]->X);CHKERRQ(ierr);
      } else {
        ierr = MatZeroEntries(mglevels[0]->X);CHKERRQ(ierr);
      }
    } else {
      ierr = VecZeroEntries(mglevels[0]->x);CHKERRQ(ierr);
    }
    for (i=0; i<l-1; i++) {
      ierr = PCMGMCycle_Private(pc,&mglevels[i],transpose,matapp,NULL);CHKERRQ(ierr);
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      if (matapp) { ierr = MatMatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->X,&mglevels[i+1]->X);CHKERRQ(ierr); }
      else { ierr = MatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->x,mglevels[i+1]->x);CHKERRQ(ierr); }
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    }
    ierr = PCMGMCycle_Private(pc,&mglevels[l-1],transpose,matapp,NULL);CHKERRQ(ierr);
  } else {
    ierr = PCMGMCycle_Private(pc,&mglevels[l-1],transpose,matapp,NULL);CHKERRQ(ierr);
    for (i=l-2; i>=0; i--) {
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      if (matapp) { ierr = MatMatRestrict(mglevels[i+1]->interpolate,mglevels[i+1]->X,&mglevels[i]->X);CHKERRQ(ierr); }
      else { ierr = MatRestrict(mglevels[i+1]->interpolate,mglevels[i+1]->x,mglevels[i]->x);CHKERRQ(ierr); }
      if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      ierr = PCMGMCycle_Private(pc,&mglevels[i],transpose,matapp,NULL);CHKERRQ(ierr);
    }
    for (i=1; i<l; i++) {
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
      if (matapp) { ierr = MatMatInterpolate(mglevels[i]->restrct,mglevels[i-1]->B,&mglevels[i]->B);CHKERRQ(ierr); }
      else { ierr = MatInterpolate(mglevels[i]->restrct,mglevels[i-1]->b,mglevels[i]->b);CHKERRQ(ierr); }
      if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGKCycle_Private(PC pc,PC_MG_Levels **mglevels,PetscBool transpose,PetscBool matapp)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i=l-1; i>0; i--) {
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (matapp) { ierr = MatMatRestrict(mglevels[i]->restrct,mglevels[i]->B,&mglevels[i-1]->B);CHKERRQ(ierr); }
    else { ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr); }
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }

  /* work our way up through the levels */
  if (matapp) {
    if (!mglevels[0]->X) {
      ierr = MatDuplicate(mglevels[0]->B,MAT_DO_NOT_COPY_VALUES,&mglevels[0]->X);CHKERRQ(ierr);
    } else {
      ierr = MatZeroEntries(mglevels[0]->X);CHKERRQ(ierr);
    }
  } else {
    ierr = VecZeroEntries(mglevels[0]->x);CHKERRQ(ierr);
  }
  for (i=0; i<l-1; i++) {
    if (mglevels[i]->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    if (matapp) {
      ierr = KSPMatSolve(mglevels[i]->smoothd,mglevels[i]->B,mglevels[i]->X);CHKERRQ(ierr);
      ierr = KSPCheckSolve(mglevels[i]->smoothd,pc,NULL);CHKERRQ(ierr);
    } else {
      ierr = KSPSolve(mglevels[i]->smoothd,mglevels[i]->b,mglevels[i]->x);CHKERRQ(ierr);
      ierr = KSPCheckSolve(mglevels[i]->smoothd,pc,mglevels[i]->x);CHKERRQ(ierr);
    }
    if (mglevels[i]->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (matapp) { ierr = MatMatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->X,&mglevels[i+1]->X);CHKERRQ(ierr); }
    else { ierr = MatInterpolate(mglevels[i+1]->interpolate,mglevels[i]->x,mglevels[i+1]->x);CHKERRQ(ierr); }
    if (mglevels[i+1]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i+1]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  if (mglevels[l-1]->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels[l-1]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  if (matapp) {
    ierr = KSPMatSolve(mglevels[l-1]->smoothd,mglevels[l-1]->B,mglevels[l-1]->X);CHKERRQ(ierr);
    ierr = KSPCheckSolve(mglevels[l-1]->smoothd,pc,NULL);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(mglevels[l-1]->smoothd,mglevels[l-1]->b,mglevels[l-1]->x);CHKERRQ(ierr);
    ierr = KSPCheckSolve(mglevels[l-1]->smoothd,pc,mglevels[l-1]->x);CHKERRQ(ierr);
  }
  if (mglevels[l-1]->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels[l-1]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
