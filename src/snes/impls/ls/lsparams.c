#define PETSCSNES_DLL

#include "src/snes/impls/ls/ls.h"

#undef __FUNCT__  
#define __FUNCT__ "SNESSetLineSeachParams"
/*@C
   SNESSetLineSearchParams - Sets the parameters associated with the line search
   routine in the Newton-based method SNESLS.

   Collective on SNES

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that .5*f_{n+1} . f_{n+1} <= .5*f_n . f_n - alpha |f_n . J . f_n|
.  maxstep - The maximum norm of the update vector
-  steptol - The minimum norm fraction of the original step after scaling

   Level: intermediate

   Note:
   Pass in PETSC_DEFAULT for any parameter you do not wish to change.

   We are finding the zero of f() so the one dimensional minimization problem we are
   solving in the line search is minimize .5*f(x_n + lambda*step_direction) . f(x_n + lambda*step_direction)

   Contributed by: Mathew Knepley

.keywords: SNES, nonlinear, set, line search params

.seealso: SNESGetLineSearchParams(), SNESSetLineSearch()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetLineSearchParams(SNES snes,PetscReal alpha,PetscReal maxstep,PetscReal steptol)
{
  SNES_LS *ls;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  ls = (SNES_LS*)snes->data;
  if (alpha   >= 0.0) ls->alpha   = alpha;
  if (maxstep >= 0.0) ls->maxstep = maxstep;
  if (steptol >= 0.0) ls->steptol = steptol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLineSeachParams"
/*@C
   SNESGetLineSearchParams - Gets the parameters associated with the line search
     routine in the Newton-based method SNESLS.

   Not collective, but any processor will return the same values

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that .5*f_{n+1} . f_{n+1} <= .5*f_n . f_n - alpha |f_n . J . f_n|
.  maxstep - The maximum norm of the update vector
-  steptol - The minimum norm fraction of the original step after scaling

   Level: intermediate

   Note:
    To not get a certain parameter, pass in PETSC_NULL

   We are finding the zero of f() so the one dimensional minimization problem we are
   solving in the line search is minimize .5*f(x_n + lambda*step_direction) . f(x_n + lambda*step_direction)

   Contributed by: Mathew Knepley

.keywords: SNES, nonlinear, set, line search parameters

.seealso: SNESSetLineSearchParams(), SNESSetLineSearch()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetLineSearchParams(SNES snes,PetscReal *alpha,PetscReal *maxstep,PetscReal *steptol)
{
  SNES_LS *ls;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  ls = (SNES_LS*)snes->data;
  if (alpha) {
    PetscValidDoublePointer(alpha,2);
    *alpha   = ls->alpha;
  }
  if (maxstep) {
    PetscValidDoublePointer(maxstep,3);
    *maxstep = ls->maxstep;
  }
  if (steptol) {
    PetscValidDoublePointer(steptol,4);
    *steptol = ls->steptol;
  }
  PetscFunctionReturn(0);
}

