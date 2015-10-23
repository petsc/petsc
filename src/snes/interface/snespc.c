
#include <petsc/private/snesimpl.h>      /*I "petscsnes.h"  I*/
#include <petscdmshell.h>


#undef __FUNCT__
#define __FUNCT__ "SNESApplyNPC"
/*@
   SNESApplyNPC - Calls SNESSolve() on preconditioner for the SNES

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  x - input vector
-  f - optional; the function evaluation on x

   Output Parameter:
.  y - function vector, as set by SNESSetFunction()

   Notes:
   SNESComputeFunction() should be called on x before SNESApplyNPC() is called, as it is
   with SNESComuteJacobian().

   Level: developer

.keywords: SNES, nonlinear, compute, function

.seealso: SNESGetNPC(),SNESSetNPC(),SNESComputeFunction()
@*/
PetscErrorCode  SNESApplyNPC(SNES snes,Vec x,Vec f,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheckSameComm(snes,1,x,2);
  PetscCheckSameComm(snes,1,y,3);
  ierr = VecValidValues(x,2,PETSC_TRUE);CHKERRQ(ierr);
  if (snes->pc) {
    if (f) {
      ierr = SNESSetInitialFunction(snes->pc,f);CHKERRQ(ierr);
    }
    ierr = VecCopy(x,y);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SNES_NPCSolve,snes->pc,x,y,0);CHKERRQ(ierr);
    ierr = SNESSolve(snes->pc,snes->vec_rhs,y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_NPCSolve,snes->pc,x,y,0);CHKERRQ(ierr);
    ierr = VecAYPX(y,-1.0,x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeFunctionDefaultNPC"
PetscErrorCode SNESComputeFunctionDefaultNPC(SNES snes,Vec X,Vec F)
{
/* This is to be used as an argument to SNESMF -- NOT as a "function" */
  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (snes->pc) {
    ierr = SNESApplyNPC(snes,X,NULL,F);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      ierr = SNESSetFunctionDomainError(snes);CHKERRQ(ierr);
    }
  } else {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGetNPCFunction"
/*@
   SNESGetNPCFunction - Gets the function from a preconditioner after SNESSolve() has been called.

   Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Output Parameter:
.  F - function vector
.  fnorm - the norm of F

   Level: developer

.keywords: SNES, nonlinear, function

.seealso: SNESGetNPC(),SNESSetNPC(),SNESComputeFunction(),SNESApplyNPC(),SNESSolve()
@*/
PetscErrorCode SNESGetNPCFunction(SNES snes,Vec F,PetscReal *fnorm)
{
  PetscErrorCode   ierr;
  PCSide           npcside;
  SNESFunctionType functype;
  SNESNormSchedule normschedule;
  Vec              FPC,XPC;

  PetscFunctionBegin;
  if (snes->pc) {
    ierr = SNESGetNPCSide(snes->pc,&npcside);CHKERRQ(ierr);
    ierr = SNESGetFunctionType(snes->pc,&functype);CHKERRQ(ierr);
    ierr = SNESGetNormSchedule(snes->pc,&normschedule);CHKERRQ(ierr);

    /* check if the function is valid based upon how the inner solver is preconditioned */
    if (normschedule != SNES_NORM_NONE && normschedule != SNES_NORM_INITIAL_ONLY && (npcside == PC_RIGHT || functype == SNES_FUNCTION_UNPRECONDITIONED)) {
      ierr = SNESGetFunction(snes->pc,&FPC,NULL,NULL);CHKERRQ(ierr);
      if (FPC) {
        if (fnorm) {ierr = VecNorm(FPC,NORM_2,fnorm);CHKERRQ(ierr);}
        ierr = VecCopy(FPC,F);CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Preconditioner has no function");
    } else {
      ierr = SNESGetSolution(snes->pc,&XPC);CHKERRQ(ierr);
      if (XPC) {
        ierr = SNESComputeFunction(snes->pc,XPC,F);CHKERRQ(ierr);
        if (fnorm) {ierr = VecNorm(F,NORM_2,fnorm);CHKERRQ(ierr);}
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Preconditioner has no solution");
    }
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No preconditioner set");
  PetscFunctionReturn(0);
}
