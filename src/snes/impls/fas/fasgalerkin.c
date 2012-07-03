#include "../src/snes/impls/fas/fasimpls.h" /*I  "petscsnesfas.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetGalerkin"
/*@
   SNESFASGetGalerkin - Gets if the coarse problems are formed by projection to the fine problem

   Input Parameter:
.  snes - the nonlinear solver context

   Output parameter:
.  flg - the status of the galerkin problem

   Level: advanced

.keywords: FAS, galerkin

.seealso: SNESFASSetLevels(), SNESFASSetGalerkin()
@*/
PetscErrorCode SNESFASGetGalerkin(SNES snes, PetscBool *flg) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscFunctionBegin;
  *flg = fas->galerkin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetGalerkin"
/*@
   SNESFASSetGalerkin - Sets coarse problems as formed by projection to the fine problem

   Input Parameter:
.  snes - the nonlinear solver context
.  flg - the status of the galerkin problem

   Level: advanced

.keywords: FAS, galerkin

.seealso: SNESFASSetLevels(), SNESFASGetGalerkin()
@*/
PetscErrorCode SNESFASSetGalerkin(SNES snes, PetscBool flg) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fas->galerkin = flg;
  if (fas->next) {ierr = SNESFASSetGalerkin(fas->next, flg);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGalerkinDefaultFunction"
/*
SNESFASGalerkinDefaultFunction

 */
PetscErrorCode SNESFASGalerkinDefaultFunction(SNES snes, Vec X, Vec F, void * ctx) {
  /* the Galerkin FAS function evalutation is defined as
   F^l(x^l) = I^l_0F^0(P^0_lx^l)
   */
  SNES       fassnes;
  SNES_FAS * fas;
  SNES_FAS * prevfas;
  SNES       prevsnes;
  Vec b_temp;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* prolong to the fine level and evaluate there. */
  fassnes = (SNES)ctx;
  fas     = (SNES_FAS *)fassnes->data;
  prevsnes = fas->previous;
  prevfas = (SNES_FAS *)prevsnes->data;
  /* interpolate down the solution */
  ierr = MatInterpolate(prevfas->interpolate, X, prevfas->Xg);CHKERRQ(ierr);
  /* the RHS we care about is at the coarsest level */
  b_temp = prevsnes->vec_rhs;
  prevsnes->vec_rhs = PETSC_NULL;
  ierr = SNESComputeFunction(prevsnes, prevfas->Xg, prevfas->Fg);CHKERRQ(ierr);
  prevsnes->vec_rhs = b_temp;
  /* restrict up the function */
  ierr = MatRestrict(prevfas->restrct, prevfas->Fg, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
