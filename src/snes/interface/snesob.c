#include <petsc-private/snesimpl.h>

/*MC
    SNESObjectiveFunction - functional form used to convey the objective function to the nonlinear solver

     Synopsis:
     #include "petscsnes.h"
       SNESObjectiveFunction(SNES snes,Vec x,PetscReal *obj,void *ctx);

     Input Parameters:
+      snes - the SNES context
.      X - solution
.      F - current function/gradient
.      obj - real to hold the objective value
-      ctx - optional user-defined objective context

   Level: advanced

.seealso:   SNESSetFunction(), SNESGetFunction(), SNESSetObjective(), SNESGetObjective()
M*/


#undef __FUNCT__
#define __FUNCT__ "SNESSetObjective"
/*@C
   SNESSetObjective - Sets the objective function minimized by the SNES methods.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  SNESObjectiveFunction - objective evaluation routine
-  ctx - [optional] user-defined context for private data for the
         function evaluation routine (may be NULL)

   Level: beginner

   Note: If not provided then this defaults to the two norm of the function evaluation (set with SNESSetFunction())

.keywords: SNES, nonlinear, set, objective

.seealso: SNESGetObjective(), SNESComputeObjective(), SNESSetFunction(), SNESSetJacobian(), SNESObjectiveFunction
@*/
PetscErrorCode  SNESSetObjective(SNES snes,PetscErrorCode (*SNESObjectiveFunction)(SNES,Vec,PetscReal*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESSetObjective(dm,SNESObjectiveFunction,ctx);CHKERRQ(ierr);
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
+  SNESObjectiveFunction - objective evaluation routine (or NULL)
-  ctx - the function context (or NULL)

   Level: advanced

.keywords: SNES, nonlinear, get, objective

.seealso: SNESSetObjective(), SNESGetSolution()
@*/
PetscErrorCode SNESGetObjective(SNES snes,PetscErrorCode (**SNESObjectiveFunction)(SNES,Vec,PetscReal*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetObjective(dm,SNESObjectiveFunction,ctx);CHKERRQ(ierr);
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
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidPointer(ob,3);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (sdm->ops->computeobjective) {
    ierr = (sdm->ops->computeobjective)(snes,X,ob,sdm->objectivectx);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetObjective() before SNESComputeObjective().");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESObjectiveComputeFunctionDefaultFD"
/*@C
   SNESObjectiveComputeFunctionDefaultFD - Computes the gradient of a user provided objective

   Collective on SNES

   Input Parameter:
+  snes - the SNES context
.  X    - the state vector
-  ctx  - the (ignored) function context

   Output Parameter:
.  F   - the function value

   Options Database Key:
+  -snes_fd_function_eps - The differencing parameter
-  -snes_fd_function - Compute function from user provided objective with finite difference

   Notes:
   SNESObjectiveComputeFunctionDefaultFD is similar in character to SNESComputeJacobianDefault.
   Therefore, it should be used for debugging purposes only.  Using it in conjunction with
   SNESComputeJacobianDefault is excessively costly and produces a Jacobian that is quite
   noisy.  This is often necessary, but should be done with a grain of salt, even when debugging
   small problems.

   Note that this uses quadratic interpolation of the objective to form each value in the function.

   Level: advanced

.keywords: SNES, objective, debugging, finite differences, function

.seealso: SNESSetFunction(), SNESComputeObjective(), SNESComputeJacobianDefault()
@*/
PetscErrorCode SNESObjectiveComputeFunctionDefaultFD(SNES snes,Vec X,Vec F,void *ctx)
{
  Vec            Xh;
  PetscErrorCode ierr;
  PetscInt       i,N,start,end;
  PetscReal      ob,ob1,ob2,ob3,fob,dx,eps=1e-6;
  PetscScalar    fv,xv;

  PetscFunctionBegin;
  ierr = VecDuplicate(X,&Xh);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_fd_function_eps","Tolerance for nonzero entries in fd function","None",eps,&eps,NULL);CHKERRQ(ierr);
  ierr = VecSet(F,0.);CHKERRQ(ierr);

  ierr = VecNorm(X,NORM_2,&fob);CHKERRQ(ierr);

  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&start,&end);CHKERRQ(ierr);
  ierr = SNESComputeObjective(snes,X,&ob);CHKERRQ(ierr);

  if (fob > 0.) dx =1e-6*fob;
  else dx = 1e-6;

  for (i=0; i<N; i++) {
    /* compute the 1st value */
    ierr = VecCopy(X,Xh);CHKERRQ(ierr);
    if (i>= start && i<end) {
      xv   = dx;
      ierr = VecSetValues(Xh,1,&i,&xv,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(Xh);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Xh);CHKERRQ(ierr);
    ierr = SNESComputeObjective(snes,Xh,&ob1);CHKERRQ(ierr);

    /* compute the 2nd value */
    ierr = VecCopy(X,Xh);CHKERRQ(ierr);
    if (i>= start && i<end) {
      xv   = 2.*dx;
      ierr = VecSetValues(Xh,1,&i,&xv,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(Xh);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Xh);CHKERRQ(ierr);
    ierr = SNESComputeObjective(snes,Xh,&ob2);CHKERRQ(ierr);

    /* compute the 3rd value */
    ierr = VecCopy(X,Xh);CHKERRQ(ierr);
    if (i>= start && i<end) {
      xv   = -dx;
      ierr = VecSetValues(Xh,1,&i,&xv,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(Xh);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Xh);CHKERRQ(ierr);
    ierr = SNESComputeObjective(snes,Xh,&ob3);CHKERRQ(ierr);

    if (i >= start && i<end) {
      /* set this entry to be the gradient of the objective */
      fv = (-ob2 + 6.*ob1 - 3.*ob -2.*ob3) / (6.*dx);
      if (PetscAbsScalar(fv) > eps) {
        ierr = VecSetValues(F,1,&i,&fv,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        fv   = 0.;
        ierr = VecSetValues(F,1,&i,&fv,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = VecDestroy(&Xh);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
