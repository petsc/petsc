
#include <../src/ts/impls/semiexplicit/semiexplicit.h>

typedef struct {
  PetscReal t;
  TS        ts;
  SNES      snes;
  Vec       U;
} TS_DAESimple_Reduced;

#undef __FUNCT__
#define __FUNCT__ "TSReset_DAESimple_Reduced"
PetscErrorCode TSReset_DAESimple_Reduced(TS ts)
{
  TS_DAESimple *dae=(TS_DAESimple*)ts->data;
  TS_DAESimple_Reduced *red = (TS_DAESimple_Reduced*)dae->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSDestroy(&red->ts);CHKERRQ(ierr);
  ierr = SNESDestroy(&red->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_DAESimple_Reduced"
PetscErrorCode TSDestroy_DAESimple_Reduced(TS ts)
{
  TS_DAESimple    *dae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Reduced *red = (TS_DAESimple_Reduced*)dae->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_DAESimple_Reduced(ts);CHKERRQ(ierr);
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_DAESimple_Reduced"
PetscErrorCode TSSetFromOptions_DAESimple_Reduced(TS ts)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSolve_DAESimple_Reduced"
PetscErrorCode TSSolve_DAESimple_Reduced(TS ts)
{
  PetscErrorCode      ierr;
  TS_DAESimple         *tsdae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Reduced *red = (TS_DAESimple_Reduced*)tsdae->data;

  PetscFunctionBegin;
  ierr = TSSolve(red->ts,tsdae->U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimple_Reduced_TSFunction"
/*
   Defines the RHS function that is passed to the time-integrator.

   Solves F(U,V) for V and then computes f(U,V)

*/
PetscErrorCode TSDAESimple_Reduced_TSFunction(TS tsinner,PetscReal t,Vec U,Vec F,void *actx)
{
  TS                   ts = (TS)actx;
  TS_DAESimple         *tsdae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Reduced *red = (TS_DAESimple_Reduced*)tsdae->data;
  PetscErrorCode       ierr;
  DM                   dm;
  PetscErrorCode       (*rhsfunction)(PetscReal,Vec,Vec,Vec,void*);
  void                 *rhsfunctionctx;

  PetscFunctionBegin;
  red->t = t;
  red->U = U;
  ierr   = SNESSolve(red->snes,NULL,tsdae->V);CHKERRQ(ierr);
  ierr   = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr   = DMTSGetDAESimpleRHSFunction(dm,&rhsfunction,&rhsfunctionctx);CHKERRQ(ierr);
  ierr   = (*rhsfunction)(t,U,tsdae->V,F,rhsfunctionctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimple_Reduced_SNESFunction"
/*
   Defines the nonlinear function that is passed to the nonlinear solver

*/
PetscErrorCode TSDAESimple_Reduced_SNESFunction(SNES snes,Vec V,Vec F,void *actx)
{
  TS                   ts=(TS)actx;
  TS_DAESimple         *tsdae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Reduced *red = (TS_DAESimple_Reduced*)tsdae->data;
  PetscErrorCode       ierr;
  DM                   dm;
  PetscErrorCode       (*ifunction)(PetscReal,Vec,Vec,Vec,void*);
  void                 *ifunctionctx;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetDAESimpleIFunction(dm,&ifunction,&ifunctionctx);
  ierr = (*ifunction)(red->t,red->U,V,F,ifunctionctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_DAESimple_Reduced"
PetscErrorCode TSSetUp_DAESimple_Reduced(TS ts)
{
  PetscErrorCode       ierr;
  TS_DAESimple         *tsdae=(TS_DAESimple*)ts->data;
  TS_DAESimple_Reduced *red = (TS_DAESimple_Reduced*)tsdae->data;
  Vec                  tsrhs;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,tsdae->U);CHKERRQ(ierr);
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),&red->ts);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(red->ts,"dae_reduced_ode_");CHKERRQ(ierr);
  ierr = TSSetProblemType(red->ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(red->ts,TSEULER);CHKERRQ(ierr);
  ierr = VecDuplicate(tsdae->U,&tsrhs);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(red->ts,tsrhs,TSDAESimple_Reduced_TSFunction,ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(red->ts);CHKERRQ(ierr);
  ierr = VecDestroy(&tsrhs);CHKERRQ(ierr);

  ierr = SNESCreate(PetscObjectComm((PetscObject)ts),&red->snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(red->snes,"dae_reduced_alg_");CHKERRQ(ierr);
  ierr = SNESSetFunction(red->snes,NULL,TSDAESimple_Reduced_SNESFunction,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(red->snes,NULL,NULL,SNESComputeJacobianDefault,ts);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(red->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------ */
/*MC
      TSDAESimple - Semi-explicit DAE solver

   Level: advanced

.seealso:  TSCreate(), TS, TSSetType(), TSCN, TSBEULER, TSThetaSetTheta(), TSThetaSetEndpoint()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_DAESimple_Reduced"
PETSC_EXTERN PetscErrorCode TSCreate_DAESimple_Reduced(TS ts)
{
  TS_DAESimple    *tsdae;
  TS_DAESimple_Reduced *red;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_DAESimple_Reduced;
  ts->ops->destroy        = TSDestroy_DAESimple_Reduced;
  ts->ops->setup          = TSSetUp_DAESimple_Reduced;
  ts->ops->setfromoptions = TSSetFromOptions_DAESimple;
  ts->ops->solve          = TSSolve_DAESimple;

  ierr = PetscNewLog(ts,TS_DAESimple,&tsdae);CHKERRQ(ierr);
  ts->data = (void*)tsdae;

  tsdae->setfromoptions = TSSetFromOptions_DAESimple_Reduced;
  tsdae->solve          = TSSolve_DAESimple_Reduced;
  tsdae->destroy        = TSDestroy_DAESimple_Reduced;

  ierr = PetscMalloc(sizeof(TS_DAESimple_Reduced),&red);CHKERRQ(ierr);
  tsdae->data = red;

  PetscFunctionReturn(0);
}
