#include <../src/snes/impls/fas/fasimpls.h> /*I  "petscsnes.h"  I*/

/*@
   SNESFASGetGalerkin - Gets if the coarse problems are formed by projection to the fine problem

   Input Parameter:
.  snes - the nonlinear solver context

   Output parameter:
.  flg - the status of the galerkin problem

   Level: advanced

.seealso: SNESFASSetLevels(), SNESFASSetGalerkin()
@*/
PetscErrorCode SNESFASGetGalerkin(SNES snes, PetscBool *flg)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  *flg = fas->galerkin;
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetGalerkin - Sets coarse problems as formed by projection to the fine problem

   Input Parameters:
+  snes - the nonlinear solver context
-  flg - the status of the galerkin problem

   Level: advanced

.seealso: SNESFASSetLevels(), SNESFASGetGalerkin()
@*/
PetscErrorCode SNESFASSetGalerkin(SNES snes, PetscBool flg)
{
  SNES_FAS       *fas;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  fas->galerkin = flg;
  if (fas->next) {ierr = SNESFASSetGalerkin(fas->next, flg);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
   SNESFASGalerkinFunctionDefault - Computes the Galerkin FAS function

   Input Parameters:
+  snes - the nonlinear solver context
.  X - input vector
-  ctx - the FAS context

   Output Parameter:
.  F - output vector

   Notes:
   The Galerkin FAS function evalutation is defined as
$  F^l(x^l) = I^l_0 F^0(P^0_l x^l)

   Level: developer

.seealso: SNESFASGetGalerkin(), SNESFASSetGalerkin()
@*/
PetscErrorCode SNESFASGalerkinFunctionDefault(SNES snes, Vec X, Vec F, void *ctx)
{
  SNES           fassnes;
  SNES_FAS       *fas;
  SNES_FAS       *prevfas;
  SNES           prevsnes;
  Vec            b_temp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* prolong to the fine level and evaluate there. */
  fassnes  = (SNES)ctx;
  fas      = (SNES_FAS*)fassnes->data;
  prevsnes = fas->previous;
  prevfas  = (SNES_FAS*)prevsnes->data;
  /* interpolate down the solution */
  ierr = MatInterpolate(prevfas->interpolate, X, prevfas->Xg);CHKERRQ(ierr);
  /* the RHS we care about is at the coarsest level */
  b_temp            = prevsnes->vec_rhs;
  prevsnes->vec_rhs = NULL;
  ierr              = SNESComputeFunction(prevsnes, prevfas->Xg, prevfas->Fg);CHKERRQ(ierr);
  prevsnes->vec_rhs = b_temp;
  /* restrict up the function */
  ierr = MatRestrict(prevfas->restrct, prevfas->Fg, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
