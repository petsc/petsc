static char help[] = "Basic problem for multi-rate method.\n";

/*F

\begin{eqnarray}
                 ys' = -2.0*\frac{-1.0+ys^2.0-\cos(t)}{2.0*ys}+0.05*\frac{-2.0+yf^2-\cos(5.0*t)}{2.0*yf}-\frac{\sin(t)}{2.0*ys}\\
                 yf' = 0.05*\frac{-1.0+ys^2-\cos(t)}{2.0*ys}-\frac{-2.0+yf^2-\cos(5.0*t)}{2.0*yf}-5.0*\frac{\sin(5.0*t)}{2.0*yf}\\
\end{eqnarray}

F*/

#include <petscts.h>

typedef struct {
  PetscReal Tf,dt;
} AppCtx;

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = -2.0*(-1.0+u[0]*u[0]-PetscCosScalar(t))/(2.0*u[0])+0.05*(-2.0+u[1]*u[1]-PetscCosScalar(5.0*t))/(2.0*u[1])-PetscSinScalar(t)/(2.0*u[0]);
  f[1] = 0.05*(-1.0+u[0]*u[0]-PetscCosScalar(t))/(2.0*u[0])-(-2.0+u[1]*u[1]-PetscCosScalar(5.0*t))/(2.0*u[1])-5.0*PetscSinScalar(5.0*t)/(2.0*u[1]);
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
  f[0] = -2.0*(-1.0+u[0]*u[0]-PetscCosScalar(t))/(2.0*u[0])+0.05*(-2.0+u[1]*u[1]-PetscCosScalar(5.0*t))/(2.0*u[1])-PetscSinScalar(t)/(2.0*u[0]);
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
  f[0] = 0.05*(-1.0+u[0]*u[0]-PetscCosScalar(t))/(2.0*u[0])-(-2.0+u[1]*u[1]-PetscCosScalar(5.0*t))/(2.0*u[1])-5.0*PetscSinScalar(5.0*t)/(2.0*u[1]);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode sol_true(PetscReal t,Vec U)
{
  PetscScalar    *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(U,&u));
  u[0] = PetscSqrtScalar(1.0+PetscCosScalar(t));
  u[1] = PetscSqrtScalar(2.0+PetscCosScalar(5.0*t));
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
  PetscReal      error,tt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
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
    Set initial condition
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(U,&u));
  u[0] = PetscSqrtScalar(2.0);
  u[1] = PetscSqrtScalar(3.0);
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
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ODE options","");CHKERRQ(ierr);
  {
    ctx.Tf = 0.3;
    ctx.dt = 0.01;
    CHKERRQ(PetscOptionsScalar("-Tf","","",ctx.Tf,&ctx.Tf,NULL));
    CHKERRQ(PetscOptionsScalar("-dt","","",ctx.dt,&ctx.dt,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  CHKERRQ(sol_true(tt,Utrue));
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
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
    build:
      requires: !complex

    test:

TEST*/
