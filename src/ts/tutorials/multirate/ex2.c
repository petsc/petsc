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
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = -PetscSinScalar(ctx->a*t);
  f[1] = ctx->b*PetscCosScalar(ctx->b*t)*u[0]-PetscSinScalar(ctx->b*t)*PetscSinScalar(ctx->a*t);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
 }

static PetscErrorCode RHSFunctionslow(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = -PetscSinScalar(ctx->a*t);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionfast(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = ctx->b*PetscCosScalar(ctx->b*t)*u[0]-PetscSinScalar(ctx->b*t)*PetscSinScalar(ctx->a*t);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
  Define the analytic solution for check method easily
*/
static PetscErrorCode sol_true(PetscReal t,Vec U,AppCtx *ctx)
{
  PetscScalar    *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(U,&u));
  u[0] = PetscCosScalar(ctx->a*t)/ctx->a;
  u[1] = PetscSinScalar(ctx->b*t)*PetscCosScalar(ctx->a*t)/ctx->a;
  CHKERRQ(VecRestoreArray(U,&u));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Vec            Utrue;
  PetscErrorCode ierr;
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
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create index for slow part and fast part
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscMalloc1(1,&indicess));
  indicess[0]=0;
  CHKERRQ(PetscMalloc1(1,&indicesf));
  indicesf[0]=1;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,1,indicess,PETSC_COPY_VALUES,&iss));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,1,indicesf,PETSC_COPY_VALUES,&isf));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necesary vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&U));
  CHKERRQ(VecSetSizes(U,n,PETSC_DETERMINE));
  CHKERRQ(VecSetFromOptions(U));
  CHKERRQ(VecDuplicate(U,&Utrue));
  CHKERRQ(VecCopy(U,Utrue));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ODE options","");CHKERRQ(ierr);
  {
    ctx.a  = 1.0;
    ctx.b  = 25.0;
    CHKERRQ(PetscOptionsScalar("-a","","",ctx.a,&ctx.a,NULL));
    CHKERRQ(PetscOptionsScalar("-b","","",ctx.b,&ctx.b,NULL));
    ctx.Tf = 5.0;
    ctx.dt = 0.01;
    CHKERRQ(PetscOptionsScalar("-Tf","","",ctx.Tf,&ctx.Tf,NULL));
    CHKERRQ(PetscOptionsScalar("-dt","","",ctx.dt,&ctx.dt,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize the solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(U,&u));
  u[0] = 1.0/ctx.a;
  u[1] = 0.0;
  CHKERRQ(VecRestoreArray(U,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSMPRK));

  CHKERRQ(TSSetRHSFunction(ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  CHKERRQ(TSRHSSplitSetIS(ts,"slow",iss));
  CHKERRQ(TSRHSSplitSetIS(ts,"fast",isf));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts,"slow",NULL,(TSRHSFunction)RHSFunctionslow,&ctx));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts,"fast",NULL,(TSRHSFunction)RHSFunctionfast,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ts,ctx.Tf));
  CHKERRQ(TSSetTimeStep(ts,ctx.dt));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));
  CHKERRQ(VecView(U,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check the error of the Petsc solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSGetTime(ts,&tt));
  CHKERRQ(sol_true(tt,Utrue,&ctx));
  CHKERRQ(VecAXPY(Utrue,-1.0,U));
  CHKERRQ(VecNorm(Utrue,NORM_2,&error));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Print norm2 error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"l2 error norm: %g\n", error));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(VecDestroy(&Utrue));
  CHKERRQ(ISDestroy(&iss));
  CHKERRQ(ISDestroy(&isf));
  CHKERRQ(PetscFree(indicess));
  CHKERRQ(PetscFree(indicesf));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !complex

    test:

TEST*/
