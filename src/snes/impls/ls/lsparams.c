#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ls.c,v 1.138 1999/06/08 22:54:40 bsmith Exp $";
#endif

#include "src/snes/impls/ls/ls.h"

#undef __FUNC__  
#define __FUNC__ "SNESSetLineSeachParams"
/*@C
   SNESSetLineSearchParams - Sets the parameters associated with the line search
   routine in the Newton-based method SNES_EQ_LS.

   Collective on SNES

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that x_{n+1} . x_{n+1} <= x_n . x_n - alpha |x_n . J . x_n|
.  maxstep - The maximum norm of the update vector
-  steptol - The minimum norm fraction of the original step after scaling

   Level: intermediate

   Note:
   Pass in PETSC_DEFAULT for any parameter you do not wish to change.

   Contributed by: Mathew Knepley

.keywords: SNES, nonlinear, set, line search params

.seealso: SNESGetLineSearchParams(), SNESSetLineSearch()
@*/
int SNESSetLineSearchParams(SNES snes, double alpha, double maxstep, double steptol)
{
  SNES_LS *ls;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_COOKIE);

  ls = (SNES_LS *) snes->data;
  if (alpha   >= 0.0) ls->alpha   = alpha;
  if (maxstep >= 0.0) ls->maxstep = maxstep;
  if (steptol >= 0.0) ls->steptol = steptol;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESGetLineSeachParams"
/*@C
   SNESGetLineSearchParams - Gets the parameters associated with the line search
     routine in the Newton-based method SNES_EQ_LS.

   Not collective, but any processor will return the same values

   Input Parameters:
+  snes    - The nonlinear context obtained from SNESCreate()
.  alpha   - The scalar such that x_{n+1} . x_{n+1} <= x_n . x_n - alpha |x_n . J . x_n|
.  maxstep - The maximum norm of the update vector
-  steptol - The minimum norm fraction of the original step after scaling

   Level: intermediate

   Note:
    To not get a certain parameter, pass in PETSC_NULL

   Contributed by: Mathew Knepley

.keywords: SNES, nonlinear, set, line search parameters

.seealso: SNESSetLineSearchParams(), SNESSetLineSearch()
@*/
int SNESGetLineSearchParams(SNES snes, double *alpha, double *maxstep, double *steptol)
{
  SNES_LS *ls;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_COOKIE);

  ls = (SNES_LS *) snes->data;
  if (alpha) {
    PetscValidPointer(alpha);
    *alpha   = ls->alpha;
  }
  if (maxstep) {
    PetscValidPointer(maxstep);
    *maxstep = ls->maxstep;
  }
  if (steptol) {
    PetscValidPointer(steptol);
    *steptol = ls->steptol;
  }
  PetscFunctionReturn(0);
}

