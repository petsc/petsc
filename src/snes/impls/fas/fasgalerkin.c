#include <../src/snes/impls/fas/fasimpls.h> /*I  "petscsnes.h"  I*/

/*@
   SNESFASGetGalerkin - Gets if the coarse problems are formed by projection to the fine problem

   Not collective but the result would be the same on all MPI ranks

   Input Parameter:
.  snes - the `SNESFAS` nonlinear solver context

   Output parameter:
.  flg - `PETSC_TRUE` if the coarse problem is formed by projection

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetLevels()`, `SNESFASSetGalerkin()`
@*/
PetscErrorCode SNESFASGetGalerkin(SNES snes, PetscBool *flg)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas  = (SNES_FAS *)snes->data;
  *flg = fas->galerkin;
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetGalerkin - Sets coarse problems as formed by projection to the fine problem

   Collective on snes

   Input Parameters:
+  snes - the `SNESFAS` nonlinear solver context
-  flg - `PETSC_TRUE` to use the projection process

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetLevels()`, `SNESFASGetGalerkin()`
@*/
PetscErrorCode SNESFASSetGalerkin(SNES snes, PetscBool flg)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas           = (SNES_FAS *)snes->data;
  fas->galerkin = flg;
  if (fas->next) PetscCall(SNESFASSetGalerkin(fas->next, flg));
  PetscFunctionReturn(0);
}

/*@C
   SNESFASGalerkinFunctionDefault - Computes the Galerkin FAS function

   Collective on snes

   Input Parameters:
+  snes - the `SNESFAS` nonlinear solver context
.  X - input vector
-  ctx - the application context

   Output Parameter:
.  F - output vector

   Note:
   The Galerkin FAS function evalutation is defined as
$  F^l(x^l) = I^l_0 F^0(P^0_l x^l)

   Level: developer

.seealso: `SNESFAS`, `SNESFASGetGalerkin()`, `SNESFASSetGalerkin()`
@*/
PetscErrorCode SNESFASGalerkinFunctionDefault(SNES snes, Vec X, Vec F, void *ctx)
{
  SNES      fassnes;
  SNES_FAS *fas;
  SNES_FAS *prevfas;
  SNES      prevsnes;
  Vec       b_temp;

  PetscFunctionBegin;
  /* prolong to the fine level and evaluate there. */
  fassnes  = (SNES)ctx;
  fas      = (SNES_FAS *)fassnes->data;
  prevsnes = fas->previous;
  prevfas  = (SNES_FAS *)prevsnes->data;
  /* interpolate down the solution */
  PetscCall(MatInterpolate(prevfas->interpolate, X, prevfas->Xg));
  /* the RHS we care about is at the coarsest level */
  b_temp            = prevsnes->vec_rhs;
  prevsnes->vec_rhs = NULL;
  PetscCall(SNESComputeFunction(prevsnes, prevfas->Xg, prevfas->Fg));
  prevsnes->vec_rhs = b_temp;
  /* restrict up the function */
  PetscCall(MatRestrict(prevfas->restrct, prevfas->Fg, F));
  PetscFunctionReturn(0);
}
