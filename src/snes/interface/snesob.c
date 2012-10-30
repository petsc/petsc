#include <petsc-private/snesimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESSetObjective"
/*@C
   SNESSetObjective - Sets the objective function minimized by
   the SNES methods.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - objective evaluation routine
-  ctx - [optional] user-defined context for private data for the
         function evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNES snes,Vec x,PetscReal *obj,void *ctx);

+  snes - the SNES context
.  X - solution
.  F - current function/gradient
.  obj - real to hold the objective value
-  ctx - optional user-defined objective context 

   Level: beginner

.keywords: SNES, nonlinear, set, objective

.seealso: SNESGetObjective(), SNESComputeObjective(), SNESSetFunction(), SNESSetJacobian()
@*/
PetscErrorCode  SNESSetObjective(SNES snes,SNESObjective func,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESSetObjective(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGetObjective"
/*@C
   SNESGetObjective - Returns the objective function.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
+  func - the function (or PETSC_NULL)
-  ctx - the function context (or PETSC_NULL)

   Level: advanced

.keywords: SNES, nonlinear, get, objective

.seealso: SNESSetObjective(), SNESGetSolution()
@*/
PetscErrorCode SNESGetObjective(SNES snes,SNESObjective *func,void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetObjective(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeObjective"
/*@C
   SNESComputeObjective - Computes the objective.

   Collective on SNES

   Input Parameter:
+  snes - the SNES context
-  X    - the state vector

   Output Parameter:
.  ob   - the objective value

   Level: advanced

.keywords: SNES, nonlinear, compute, objective

.seealso: SNESSetObjective(), SNESGetSolution()
@*/
PetscErrorCode SNESComputeObjective(SNES snes,Vec X,PetscReal *ob)
{
  PetscErrorCode ierr;
  DM             dm;
  SNESDM         sdm;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidPointer(ob,3);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (sdm->computeobjective) {
    ierr = (sdm->computeobjective)(snes,X,ob,sdm->objectivectx);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetObjective() before SNESComputeObjective().");
  }
  PetscFunctionReturn(0);
}
