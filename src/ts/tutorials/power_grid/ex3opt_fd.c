
static char help[] = "Finds optimal parameter P_m for the generator system while maintaining generator stability.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

F*/

/*
  Solve the same optimization problem as in ex3opt.c.
  Use finite difference to approximate the gradients.
*/
#include <petsctao.h>
#include <petscts.h>
#include "ex3.h"

PetscErrorCode FormFunction(Tao,Vec,PetscReal*,void*);

PetscErrorCode monitor(Tao tao,AppCtx *ctx)
{
  FILE               *fp;
  PetscInt           iterate;
  PetscReal          f,gnorm,cnorm,xdiff;
  Vec                X,G;
  const PetscScalar  *x,*g;
  TaoConvergedReason reason;

  PetscFunctionBeginUser;
  CHKERRQ(TaoGetSolutionStatus(tao,&iterate,&f,&gnorm,&cnorm,&xdiff,&reason));
  CHKERRQ(TaoGetSolution(tao,&X));
  CHKERRQ(TaoGetGradient(tao,&G,NULL,NULL));
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(G,&g));
  fp = fopen("ex3opt_fd_conv.out","a");
  CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,fp,"%d %g %.12lf %.12lf\n",iterate,gnorm,x[0],g[0]));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(G,&g));
  fclose(fp);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec                p;
  PetscScalar        *x_ptr;
  PetscErrorCode     ierr;
  PetscMPIInt        size;
  AppCtx             ctx;
  Vec                lowerb,upperb;
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  PetscBool          printtofile;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Swing equation options","");CHKERRQ(ierr);
  {
    ctx.beta    = 2;
    ctx.c       = 10000.0;
    ctx.u_s     = 1.0;
    ctx.omega_s = 1.0;
    ctx.omega_b = 120.0*PETSC_PI;
    ctx.H       = 5.0;
    CHKERRQ(PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL));
    ctx.D       = 5.0;
    CHKERRQ(PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL));
    ctx.E       = 1.1378;
    ctx.V       = 1.0;
    ctx.X       = 0.545;
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    ctx.Pmax_ini = ctx.Pmax;
    CHKERRQ(PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL));
    ctx.Pm      = 1.06;
    CHKERRQ(PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL));
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    CHKERRQ(PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL));
    CHKERRQ(PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL));
    printtofile = PETSC_FALSE;
    CHKERRQ(PetscOptionsBool("-printtofile","Print convergence results to file","",printtofile,&printtofile,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOBLMVM));
  if (printtofile) {
    CHKERRQ(TaoSetMonitor(tao,(PetscErrorCode (*)(Tao, void*))monitor,(void *)&ctx,PETSC_NULL));
  }
  CHKERRQ(TaoSetMaximumIterations(tao,30));
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,1,&p));
  CHKERRQ(VecGetArray(p,&x_ptr));
  x_ptr[0]   = ctx.Pm;
  CHKERRQ(VecRestoreArray(p,&x_ptr));

  CHKERRQ(TaoSetSolution(tao,p));
  /* Set routine for function and gradient evaluation */
  CHKERRQ(TaoSetObjective(tao,FormFunction,(void *)&ctx));
  CHKERRQ(TaoSetGradient(tao,NULL,TaoDefaultComputeGradient,(void *)&ctx));

  /* Set bounds for the optimization */
  CHKERRQ(VecDuplicate(p,&lowerb));
  CHKERRQ(VecDuplicate(p,&upperb));
  CHKERRQ(VecGetArray(lowerb,&x_ptr));
  x_ptr[0] = 0.;
  CHKERRQ(VecRestoreArray(lowerb,&x_ptr));
  CHKERRQ(VecGetArray(upperb,&x_ptr));
  x_ptr[0] = 1.1;
  CHKERRQ(VecRestoreArray(upperb,&x_ptr));
  CHKERRQ(TaoSetVariableBounds(tao,lowerb,upperb));

  /* Check for any TAO command line options */
  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoGetKSP(tao,&ksp));
  if (ksp) {
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCNONE));
  }

  /* SOLVE THE APPLICATION */
  CHKERRQ(TaoSolve(tao));

  CHKERRQ(VecView(p,PETSC_VIEWER_STDOUT_WORLD));

  /* Free TAO data structures */
  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(VecDestroy(&p));
  CHKERRQ(VecDestroy(&lowerb));
  CHKERRQ(VecDestroy(&upperb));
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunction - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
*/
PetscErrorCode FormFunction(Tao tao,Vec P,PetscReal *f,void *ctx0)
{
  AppCtx            *ctx = (AppCtx*)ctx0;
  TS                ts,quadts;
  Vec               U;             /* solution will be stored here */
  Mat               A;             /* Jacobian matrix */
  PetscInt          n = 2;
  PetscReal         ftime;
  PetscInt          steps;
  PetscScalar       *u;
  const PetscScalar *x_ptr,*qx_ptr;
  Vec               q;
  PetscInt          direction[2];
  PetscBool         terminate[2];

  CHKERRQ(VecGetArrayRead(P,&x_ptr));
  ctx->Pm = x_ptr[0];
  CHKERRQ(VecRestoreArrayRead(P,&x_ptr));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(A,MATDENSE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&U,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSCN));
  CHKERRQ(TSSetIFunction(ts,NULL,(TSIFunction) IFunction,ctx));
  CHKERRQ(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(U,&u));
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = 1.0;
  CHKERRQ(VecRestoreArray(U,&u));
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ts,1.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTimeStep(ts,0.03125));
  CHKERRQ(TSCreateQuadratureTS(ts,PETSC_TRUE,&quadts));
  CHKERRQ(TSGetSolution(quadts,&q));
  CHKERRQ(VecSet(q,0.0));
  CHKERRQ(TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,ctx));
  CHKERRQ(TSSetFromOptions(ts));

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;

  CHKERRQ(TSSetEventHandler(ts,2,direction,terminate,EventFunction,PostEventFunction,(void*)ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));

  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(VecGetArrayRead(q,&qx_ptr));
  *f   = -ctx->Pm + qx_ptr[0];
  CHKERRQ(VecRestoreArrayRead(q,&qx_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -ts_type cn -pc_type lu -tao_monitor -tao_gatol 1e-3

TEST*/
