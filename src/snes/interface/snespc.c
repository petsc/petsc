
#include <petsc-private/snesimpl.h>      /*I "petscsnes.h"  I*/
#include <petscdmshell.h>


#undef __FUNCT__
#define __FUNCT__ "SNESApplyPC"
/*@
   SNESComputeFunction - Calls the function that has been set with SNESSetFunction().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameter:
.  y - function vector, as set by SNESSetFunction()

   Notes:
   SNESComputeFunction() should be called on X before SNESApplyPC() is called, as it is
   with SNESComuteJacobian().

   Level: developer

.keywords: SNES, nonlinear, compute, function

.seealso: SNESGetPC(),SNESSetPC(),SNESComputeFunction()
@*/
PetscErrorCode  SNESApplyPC(SNES snes,Vec x,Vec f,PetscReal *fnorm,Vec y)
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
    if (fnorm) {
      ierr = SNESSetInitialFunctionNorm(snes->pc,*fnorm);CHKERRQ(ierr);
    }
    ierr = VecCopy(x,y);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SNES_NPCSolve,snes->pc,x,y,0);CHKERRQ(ierr);
    ierr = SNESSolve(snes->pc,snes->vec_rhs,y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_NPCSolve,snes->pc,x,y,0);CHKERRQ(ierr);
    ierr = VecAYPX(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecValidValues(y,3,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecValidValues(y,3,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeFunctionDefaultPC"
PetscErrorCode SNESComputeFunctionDefaultPC(SNES snes,Vec X,Vec F,void *ctx) {
/* This is to be used as an argument to SNESMF -- NOT as a "function" */
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESApplyPC(snes,X,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);
  ierr = VecAYPX(F,-1.0,X);CHKERRQ(ierr);CHKERRQ(ierr);

PetscFunctionReturn(0);
}
