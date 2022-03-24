static char help[] = "Solves DAE with integrator only on non-algebraic terms \n";

#include <petscts.h>

/*
        \dot{U} = f(U,V)
        F(U,V)  = 0
*/

/*
   f(U,V) = U + V
*/
PetscErrorCode f(PetscReal t,Vec U,Vec V,Vec F)
{
  PetscFunctionBeginUser;
  CHKERRQ(VecWAXPY(F,1.0,U,V));
  PetscFunctionReturn(0);
}

/*
   F(U,V) = U - V
*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F)
{
  PetscFunctionBeginUser;
  CHKERRQ(VecWAXPY(F,-1.0,V,U));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal      t;
  SNES           snes;
  Vec            U,V;
  PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec);
  PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec);
} AppCtx;

extern PetscErrorCode TSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode SNESFunction(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         ctx;
  TS             ts;
  Vec            tsrhs,U;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSEULER));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,1,&tsrhs));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,1,&U));
  CHKERRQ(TSSetRHSFunction(ts,tsrhs,TSFunction,&ctx));
  CHKERRQ(TSSetMaxTime(ts,1.0));
  ctx.f = f;

  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&ctx.snes));
  CHKERRQ(SNESSetFromOptions(ctx.snes));
  CHKERRQ(SNESSetFunction(ctx.snes,NULL,SNESFunction,&ctx));
  CHKERRQ(SNESSetJacobian(ctx.snes,NULL,NULL,SNESComputeJacobianDefault,&ctx));
  ctx.F = F;
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,1,&ctx.V));

  CHKERRQ(VecSet(U,1.0));
  CHKERRQ(TSSolve(ts,U));

  CHKERRQ(VecDestroy(&ctx.V));
  CHKERRQ(VecDestroy(&tsrhs));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(SNESDestroy(&ctx.snes));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*
   Defines the RHS function that is passed to the time-integrator.

   Solves F(U,V) for V and then computes f(U,V)
*/
PetscErrorCode TSFunction(TS ts,PetscReal t,Vec U,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;

  PetscFunctionBeginUser;
  ctx->t = t;
  ctx->U = U;
  CHKERRQ(SNESSolve(ctx->snes,NULL,ctx->V));
  CHKERRQ((*ctx->f)(t,U,ctx->V,F));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the nonlinear solver
*/
PetscErrorCode SNESFunction(SNES snes,Vec V,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;

  PetscFunctionBeginUser;
  CHKERRQ((*ctx->F)(ctx->t,ctx->U,V,F));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args:  -ts_monitor -ts_view

TEST*/
