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
  PetscErrorCode    ierr;
  const PetscScalar *u,*v;
  PetscScalar       *f;
  PetscInt          n,i;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(UV,&n);CHKERRQ(ierr);
  n    = n/2;
  ierr = VecGetArrayRead(UV,&u);CHKERRQ(ierr);
  v    = u + n;
  ierr = VecGetArrayWrite(F,&f);CHKERRQ(ierr);
  for (i=0; i<n; i++) f[i] = u[i] + v[i];
  ierr = VecRestoreArrayRead(UV,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   F(U,V) = U - V
*/
PetscErrorCode F(PetscReal t,Vec UV,Vec F)
{
  PetscErrorCode    ierr;
  const PetscScalar *u,*v;
  PetscScalar       *f;
  PetscInt          n,i;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(UV,&n);CHKERRQ(ierr);
  n    = n/2;
  ierr = VecGetArrayRead(UV,&u);CHKERRQ(ierr);
  v    = u + n;
  ierr = VecGetArrayWrite(F,&f);CHKERRQ(ierr);
  for (i=0; i<n; i++) f[i] = u[i] - v[i];
  ierr = VecRestoreArrayRead(UV,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(F,&f);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  AppCtx         ctx;
  TS             ts;
  Vec            tsrhs,U;
  IS             is;
  PetscInt       i;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&tsrhs);CHKERRQ(ierr);
  ierr = VecDuplicate(tsrhs,&U);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,tsrhs,TSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ctx.f = f;

  ierr = SNESCreate(PETSC_COMM_WORLD,&ctx.snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(ctx.snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(ctx.snes,NULL,SNESFunction,&ctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(ctx.snes,NULL,NULL,SNESComputeJacobianDefault,&ctx);CHKERRQ(ierr);
  ctx.F = F;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.V);CHKERRQ(ierr);

  /* Create scatters to move between separate U and V representation and UV representation of solution */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&ctx.UV);CHKERRQ(ierr);
  i    = 2*rank;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,&i,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(U,NULL,ctx.UV,is,&ctx.scatterU);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  i    = 2*rank + 1;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,&i,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(ctx.V,NULL,ctx.UV,is,&ctx.scatterV);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = VecSet(U,1.0);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&ctx.V);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.UV);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx.scatterU);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx.scatterV);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(ctx->scatterU,U,ctx->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterU,U,ctx->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = SNESSolve(ctx->snes,NULL,ctx->V);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterV,ctx->V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterV,ctx->V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*ctx->f)(t,ctx->UV,F);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(ctx->scatterV,V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterV,V,ctx->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*ctx->F)(ctx->t,ctx->UV,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_monitor -ts_view

TEST*/

