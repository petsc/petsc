
#include <../src/ts/impls/semiexplicit/semiexplicit.h>            /*I  "petscts.h"  */



#undef __FUNCT__
#define __FUNCT__ "TSReset_DAESimple"
PetscErrorCode TSReset_DAESimple(TS ts)
{
  TS_DAESimple *dae=(TS_DAESimple*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dae->destroy)(ts);CHKERRQ(ierr);
  ierr = VecDestroy(&(dae)->U);CHKERRQ(ierr);
  ierr = VecDestroy(&(dae)->V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_DAESimple"
PetscErrorCode TSDestroy_DAESimple(TS ts)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_DAESimple(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_DAESimple"
PetscErrorCode TSSetFromOptions_DAESimple(TS ts)
{
  TS_DAESimple   *dae=(TS_DAESimple*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dae->setfromoptions)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSetRHSFunction"
PetscErrorCode TSDAESimpleSetRHSFunction(TS ts,Vec U,PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec,void*),void *ctx)
{
  TS_DAESimple *tsdae = (TS_DAESimple*)ts->data;
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetDAESimpleRHSFunction(dm,f,ctx);CHKERRQ(ierr);
  tsdae->U    = U;
  ierr        = PetscObjectReference((PetscObject)U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSetIFunction"
PetscErrorCode TSDAESimpleSetIFunction(TS ts,Vec V,PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec,void*),void *ctx)
{
  TS_DAESimple *tsdae = (TS_DAESimple*)ts->data;
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetDAESimpleIFunction(dm,F,ctx);CHKERRQ(ierr);
  tsdae->V    = V;
  ierr        = PetscObjectReference((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSolve_DAESimple"
PetscErrorCode TSSolve_DAESimple(TS ts)
{
  TS_DAESimple   *tsdae=(TS_DAESimple*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*tsdae->solve)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
