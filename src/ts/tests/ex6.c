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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,1.0,U,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   F(U,V) = U - V
*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,-1.0,V,U);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  AppCtx         ctx;
  TS             ts;
  Vec            tsrhs,U;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,1,&tsrhs);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,1,&U);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,tsrhs,TSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ctx.f = f;

  ierr = SNESCreate(PETSC_COMM_WORLD,&ctx.snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(ctx.snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(ctx.snes,NULL,SNESFunction,&ctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(ctx.snes,NULL,NULL,SNESComputeJacobianDefault,&ctx);CHKERRQ(ierr);
  ctx.F = F;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,1,&ctx.V);CHKERRQ(ierr);

  ierr = VecSet(U,1.0);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&ctx.V);CHKERRQ(ierr);
  ierr = VecDestroy(&tsrhs);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = SNESDestroy(&ctx.snes);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*
   Defines the RHS function that is passed to the time-integrator.

   Solves F(U,V) for V and then computes f(U,V)
*/
PetscErrorCode TSFunction(TS ts,PetscReal t,Vec U,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ctx->t = t;
  ctx->U = U;
  ierr   = SNESSolve(ctx->snes,NULL,ctx->V);CHKERRQ(ierr);
  ierr   = (*ctx->f)(t,U,ctx->V,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the nonlinear solver
*/
PetscErrorCode SNESFunction(SNES snes,Vec V,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = (*ctx->F)(ctx->t,ctx->U,V,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args:  -ts_monitor -ts_view

TEST*/
