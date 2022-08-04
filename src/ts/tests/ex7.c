static char help[] = "Solves DAE with integrator only on non-algebraic terms \n";

#include <petscts.h>

/*
        \dot{U} = f(U,V)
        F(U,V)  = 0

    Same as ex6.c except the user provided functions take input values as a single vector instead of two vectors
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
  PetscCall(VecGetLocalSize(UV,&n));
  n    = n/2;
  PetscCall(VecGetArrayRead(UV,&u));
  v    = u + n;
  PetscCall(VecGetArrayWrite(F,&f));
  for (i=0; i<n; i++) f[i] = u[i] + v[i];
  PetscCall(VecRestoreArrayRead(UV,&u));
  PetscCall(VecRestoreArrayWrite(F,&f));
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
  PetscCall(VecGetLocalSize(UV,&n));
  n    = n/2;
  PetscCall(VecGetArrayRead(UV,&u));
  v    = u + n;
  PetscCall(VecGetArrayWrite(F,&f));
  for (i=0; i<n; i++) f[i] = u[i] - v[i];
  PetscCall(VecRestoreArrayRead(UV,&u));
  PetscCall(VecRestoreArrayWrite(F,&f));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal      t;
  SNES           snes;
  Vec            UV,V;
  VecScatter     scatterU,scatterV;
  PetscErrorCode (*f)(PetscReal,Vec,Vec);
  PetscErrorCode (*F)(PetscReal,Vec,Vec);
} AppCtx;

extern PetscErrorCode TSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode SNESFunction(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         ctx;
  TS             ts;
  Vec            tsrhs,U;
  IS             is;
  PetscInt       i;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSEULER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&tsrhs));
  PetscCall(VecDuplicate(tsrhs,&U));
  PetscCall(TSSetRHSFunction(ts,tsrhs,TSFunction,&ctx));
  PetscCall(TSSetMaxTime(ts,1.0));
  ctx.f = f;

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&ctx.snes));
  PetscCall(SNESSetFromOptions(ctx.snes));
  PetscCall(SNESSetFunction(ctx.snes,NULL,SNESFunction,&ctx));
  PetscCall(SNESSetJacobian(ctx.snes,NULL,NULL,SNESComputeJacobianDefault,&ctx));
  ctx.F = F;
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.V));

  /* Create scatters to move between separate U and V representation and UV representation of solution */
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&ctx.UV));
  i    = 2*rank;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,1,&i,PETSC_COPY_VALUES,&is));
  PetscCall(VecScatterCreate(U,NULL,ctx.UV,is,&ctx.scatterU));
  PetscCall(ISDestroy(&is));
  i    = 2*rank + 1;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,1,&i,PETSC_COPY_VALUES,&is));
  PetscCall(VecScatterCreate(ctx.V,NULL,ctx.UV,is,&ctx.scatterV));
  PetscCall(ISDestroy(&is));

  PetscCall(VecSet(U,1.0));
  PetscCall(TSSolve(ts,U));

  PetscCall(VecDestroy(&ctx.V));
  PetscCall(VecDestroy(&ctx.UV));
  PetscCall(VecScatterDestroy(&ctx.scatterU));
  PetscCall(VecScatterDestroy(&ctx.scatterV));
  PetscCall(VecDestroy(&tsrhs));
  PetscCall(VecDestroy(&U));
  PetscCall(SNESDestroy(&ctx.snes));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
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
  PetscCall(VecScatterBegin(ctx->scatterU,U,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatterU,U,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(SNESSolve(ctx->snes,NULL,ctx->V));
  PetscCall(VecScatterBegin(ctx->scatterV,ctx->V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatterV,ctx->V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall((*ctx->f)(t,ctx->UV,F));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the nonlinear solver

*/
PetscErrorCode SNESFunction(SNES snes,Vec V,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;

  PetscFunctionBeginUser;
  PetscCall(VecScatterBegin(ctx->scatterV,V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatterV,V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall((*ctx->F)(ctx->t,ctx->UV,F));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_monitor -ts_view

TEST*/
