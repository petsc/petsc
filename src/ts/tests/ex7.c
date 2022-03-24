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
  for (i=0; i<n; i++) f[i] = u[i] - v[i];
  CHKERRQ(VecRestoreArrayRead(UV,&u));
  CHKERRQ(VecRestoreArrayWrite(F,&f));
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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSEULER));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&tsrhs));
  CHKERRQ(VecDuplicate(tsrhs,&U));
  CHKERRQ(TSSetRHSFunction(ts,tsrhs,TSFunction,&ctx));
  CHKERRQ(TSSetMaxTime(ts,1.0));
  ctx.f = f;

  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&ctx.snes));
  CHKERRQ(SNESSetFromOptions(ctx.snes));
  CHKERRQ(SNESSetFunction(ctx.snes,NULL,SNESFunction,&ctx));
  CHKERRQ(SNESSetJacobian(ctx.snes,NULL,NULL,SNESComputeJacobianDefault,&ctx));
  ctx.F = F;
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.V));

  /* Create scatters to move between separate U and V representation and UV representation of solution */
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&ctx.UV));
  i    = 2*rank;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,1,&i,PETSC_COPY_VALUES,&is));
  CHKERRQ(VecScatterCreate(U,NULL,ctx.UV,is,&ctx.scatterU));
  CHKERRQ(ISDestroy(&is));
  i    = 2*rank + 1;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,1,&i,PETSC_COPY_VALUES,&is));
  CHKERRQ(VecScatterCreate(ctx.V,NULL,ctx.UV,is,&ctx.scatterV));
  CHKERRQ(ISDestroy(&is));

  CHKERRQ(VecSet(U,1.0));
  CHKERRQ(TSSolve(ts,U));

  CHKERRQ(VecDestroy(&ctx.V));
  CHKERRQ(VecDestroy(&ctx.UV));
  CHKERRQ(VecScatterDestroy(&ctx.scatterU));
  CHKERRQ(VecScatterDestroy(&ctx.scatterV));
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
  CHKERRQ(VecScatterBegin(ctx->scatterU,U,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx->scatterU,U,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(SNESSolve(ctx->snes,NULL,ctx->V));
  CHKERRQ(VecScatterBegin(ctx->scatterV,ctx->V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx->scatterV,ctx->V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*ctx->f)(t,ctx->UV,F));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the nonlinear solver

*/
PetscErrorCode SNESFunction(SNES snes,Vec V,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;

  PetscFunctionBeginUser;
  CHKERRQ(VecScatterBegin(ctx->scatterV,V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx->scatterV,V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*ctx->F)(ctx->t,ctx->UV,F));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_monitor -ts_view

TEST*/
