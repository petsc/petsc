#define PETSCSNES_DLL

#include "../src/snes/impls/ls/lsimpl.h"  /*I "petscsnes.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSetParams"
/*@
   SNESLineSearchSetParams - Sets the parameters associated with the line search
   routine in the Newton-based method SNESLS.

   Logically Collective on SNES

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that .5*f_{n+1} . f_{n+1} <= .5*f_n . f_n - alpha |p_n . J . f_n|
.  maxstep - The maximum norm of the update vector
-  minlambda - lambda is not allowed to be smaller than minlambda/( max_i y[i]/x[i]) 

   Level: intermediate

   Note:
   Pass in PETSC_DEFAULT for any parameter you do not wish to change.

   We are finding the zero of f() so the one dimensional minimization problem we are
   solving in the line search is minimize .5*f(x_n + lambda*step_direction) . f(x_n + lambda*step_direction)


.keywords: SNES, nonlinear, set, line search params

.seealso: SNESLineSearchGetParams(), SNESLineSearchSet()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSetParams(SNES snes,PetscReal alpha,PetscReal maxstep,PetscReal minlambda)
{
  SNES_LS *ls = (SNES_LS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveReal(snes,alpha,2);
  PetscValidLogicalCollectiveReal(snes,maxstep,3);
  PetscValidLogicalCollectiveReal(snes,minlambda,4);

  if (alpha   >= 0.0) ls->alpha       = alpha;
  if (maxstep >= 0.0) ls->maxstep     = maxstep;
  if (minlambda >= 0.0) ls->minlambda = minlambda;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchGetParams"
/*@C
   SNESLineSearchGetParams - Gets the parameters associated with the line search
     routine in the Newton-based method SNESLS.

   Not collective, but any processor will return the same values

   Input Parameter:
.  snes    - The nonlinear context obtained from SNESCreate()

   Output Parameters:
+  alpha   - The scalar such that .5*f_{n+1} . f_{n+1} <= .5*f_n . f_n - alpha |p_n . J . f_n|
.  maxstep - The maximum norm of the update vector
-  minlambda - lambda is not allowed to be smaller than minlambda/( max_i y[i]/x[i]) 

   Level: intermediate

   Note:
    To not get a certain parameter, pass in PETSC_NULL

   We are finding the zero of f() so the one dimensional minimization problem we are
   solving in the line search is minimize .5*f(x_n + lambda*step_direction) . f(x_n + lambda*step_direction)

.keywords: SNES, nonlinear, set, line search parameters

.seealso: SNESLineSearchSetParams(), SNESLineSearchSet()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchGetParams(SNES snes,PetscReal *alpha,PetscReal *maxstep,PetscReal *minlambda)
{
  SNES_LS *ls = (SNES_LS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (alpha) {
    PetscValidDoublePointer(alpha,2);
    *alpha   = ls->alpha;
  }
  if (maxstep) {
    PetscValidDoublePointer(maxstep,3);
    *maxstep = ls->maxstep;
  }
  if (minlambda) {
    PetscValidDoublePointer(minlambda,3);
    *minlambda = ls->minlambda;
  }
  PetscFunctionReturn(0);
}

