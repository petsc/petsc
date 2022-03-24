static char help[] = "Solves DAE with integrator only on non-algebraic terms \n";

#include <petscts.h>

/*
        \dot{U} = f(U,V)
        F(U,V)  = 0

    Same as ex6.c and ex7.c except calls the TSROSW integrator on the entire DAE
*/

/*
   f(U,V) = U + V

*/
PetscErrorCode f(PetscReal t,Vec UV,Vec F)
{
  const PetscScalar *u,*v;
  PetscScalar       *f;
  PetscInt          n,i;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetLocalSize(UV,&n));
  n    = n/2;
  CHKERRQ(VecGetArrayRead(UV,&u));
  v    = u + n;
  CHKERRQ(VecGetArrayWrite(F,&f));
  for (i=0; i<n; i++) f[i] = u[i] + v[i];
  CHKERRQ(VecRestoreArrayRead(UV,&u));
  CHKERRQ(VecRestoreArrayWrite(F,&f));
  PetscFunctionReturn(0);
}

/*
   F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t,Vec UV,Vec F)
{
  const PetscScalar *u,*v;
  PetscScalar       *f;
  PetscInt          n,i;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetLocalSize(UV,&n));
  n    = n/2;
  CHKERRQ(VecGetArrayRead(UV,&u));
  v    = u + n;
  CHKERRQ(VecGetArrayWrite(F,&f));
  f    = f + n;
  for (i=0; i<n; i++) f[i] = u[i] - v[i];
  f    = f - n;
  CHKERRQ(VecRestoreArrayRead(UV,&u));
  CHKERRQ(VecRestoreArrayWrite(F,&f));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscErrorCode (*f)(PetscReal,Vec,Vec);
  PetscErrorCode (*F)(PetscReal,Vec,Vec);
} AppCtx;

extern PetscErrorCode TSFunctionRHS(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode TSFunctionI(TS,PetscReal,Vec,Vec,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         ctx;
  TS             ts;
  Vec            tsrhs,UV;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSROSW));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&tsrhs));
  CHKERRQ(VecDuplicate(tsrhs,&UV));
  CHKERRQ(TSSetRHSFunction(ts,tsrhs,TSFunctionRHS,&ctx));
  CHKERRQ(TSSetIFunction(ts,NULL,TSFunctionI,&ctx));
  CHKERRQ(TSSetMaxTime(ts,1.0));
  ctx.f = f;
  ctx.F = F;

  CHKERRQ(VecSet(UV,1.0));
  CHKERRQ(TSSolve(ts,UV));
  CHKERRQ(VecDestroy(&tsrhs));
  CHKERRQ(VecDestroy(&UV));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*
   Defines the RHS function that is passed to the time-integrator.
*/
PetscErrorCode TSFunctionRHS(TS ts,PetscReal t,Vec UV,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(F,0.0));
  CHKERRQ((*ctx->f)(t,UV,F));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the time-integrator
*/
PetscErrorCode TSFunctionI(TS ts,PetscReal t,Vec UV,Vec UVdot,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;

  PetscFunctionBeginUser;
  CHKERRQ(VecCopy(UVdot,F));
  CHKERRQ((*ctx->F)(t,UV,F));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args:  -ts_view

    test:
      suffix: 2
      args: -snes_lag_jacobian 2 -ts_view

TEST*/
