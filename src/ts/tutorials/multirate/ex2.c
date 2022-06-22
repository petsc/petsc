static char help[] = "Basic problem for multi-rate method.\n";

/*F

\begin{eqnarray}
                 ys' = -sin(a*t)\\
                 yf' = bcos(b*t)ys-sin(b*t)sin(a*t)\\
\end{eqnarray}

F*/

#include <petscts.h>

typedef struct {
  PetscReal a,b,Tf,dt;
} AppCtx;

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));
  f[0] = -PetscSinScalar(ctx->a*t);
  f[1] = ctx->b*PetscCosScalar(ctx->b*t)*u[0]-PetscSinScalar(ctx->b*t)*PetscSinScalar(ctx->a*t);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
 }

static PetscErrorCode RHSFunctionslow(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));
  f[0] = -PetscSinScalar(ctx->a*t);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionfast(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));
  f[0] = ctx->b*PetscCosScalar(ctx->b*t)*u[0]-PetscSinScalar(ctx->b*t)*PetscSinScalar(ctx->a*t);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
  Define the analytic solution for check method easily
*/
static PetscErrorCode sol_true(PetscReal t,Vec U,AppCtx *ctx)
{
  PetscScalar    *u;

  PetscFunctionBegin;
  PetscCall(VecGetArray(U,&u));
  u[0] = PetscCosScalar(ctx->a*t)/ctx->a;
  u[1] = PetscSinScalar(ctx->b*t)*PetscCosScalar(ctx->a*t)/ctx->a;
  PetscCall(VecRestoreArray(U,&u));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Vec            Utrue;
  PetscMPIInt    size;
  AppCtx         ctx;
  PetscScalar    *u;
  IS             iss;
  IS             isf;
  PetscInt       *indicess;
  PetscInt       *indicesf;
  PetscInt       n=2;
  PetscScalar    error,tt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create index for slow part and fast part
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscMalloc1(1,&indicess));
  indicess[0]=0;
  PetscCall(PetscMalloc1(1,&indicesf));
  indicesf[0]=1;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,1,indicess,PETSC_COPY_VALUES,&iss));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,1,indicesf,PETSC_COPY_VALUES,&isf));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&U));
  PetscCall(VecSetSizes(U,n,PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(U));
  PetscCall(VecDuplicate(U,&Utrue));
  PetscCall(VecCopy(U,Utrue));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ODE options","");
  {
    ctx.a  = 1.0;
    ctx.b  = 25.0;
    PetscCall(PetscOptionsScalar("-a","","",ctx.a,&ctx.a,NULL));
    PetscCall(PetscOptionsScalar("-b","","",ctx.b,&ctx.b,NULL));
    ctx.Tf = 5.0;
    ctx.dt = 0.01;
    PetscCall(PetscOptionsScalar("-Tf","","",ctx.Tf,&ctx.Tf,NULL));
    PetscCall(PetscOptionsScalar("-dt","","",ctx.dt,&ctx.dt,NULL));
  }
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize the solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(U,&u));
  u[0] = 1.0/ctx.a;
  u[1] = 0.0;
  PetscCall(VecRestoreArray(U,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSMPRK));

  PetscCall(TSSetRHSFunction(ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  PetscCall(TSRHSSplitSetIS(ts,"slow",iss));
  PetscCall(TSRHSSplitSetIS(ts,"fast",isf));
  PetscCall(TSRHSSplitSetRHSFunction(ts,"slow",NULL,(TSRHSFunction)RHSFunctionslow,&ctx));
  PetscCall(TSRHSSplitSetRHSFunction(ts,"fast",NULL,(TSRHSFunction)RHSFunctionfast,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts,ctx.Tf));
  PetscCall(TSSetTimeStep(ts,ctx.dt));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));
  PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check the error of the Petsc solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSGetTime(ts,&tt));
  PetscCall(sol_true(tt,Utrue,&ctx));
  PetscCall(VecAXPY(Utrue,-1.0,U));
  PetscCall(VecNorm(Utrue,NORM_2,&error));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Print norm2 error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"l2 error norm: %g\n", (double)PetscAbsScalar(error)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&Utrue));
  PetscCall(ISDestroy(&iss));
  PetscCall(ISDestroy(&isf));
  PetscCall(PetscFree(indicess));
  PetscCall(PetscFree(indicesf));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !complex

    test:

TEST*/
