
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
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = TaoGetSolutionStatus(tao,&iterate,&f,&gnorm,&cnorm,&xdiff,&reason);CHKERRQ(ierr);
  ierr = TaoGetSolutionVector(tao,&X);CHKERRQ(ierr);
  ierr = TaoGetGradientVector(tao,&G);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(G,&g);CHKERRQ(ierr);
  fp = fopen("ex3opt_fd_conv.out","a");
  ierr = PetscFPrintf(PETSC_COMM_WORLD,fp,"%d %g %.12lf %.12lf\n",iterate,gnorm,x[0],g[0]);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(G,&g);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

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
    ierr        = PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL);CHKERRQ(ierr);
    ctx.D       = 5.0;
    ierr        = PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL);CHKERRQ(ierr);
    ctx.E       = 1.1378;
    ctx.V       = 1.0;
    ctx.X       = 0.545;
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    ctx.Pmax_ini = ctx.Pmax;
    ierr        = PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL);CHKERRQ(ierr);
    ctx.Pm      = 1.06;
    ierr        = PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL);CHKERRQ(ierr);
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    ierr        = PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL);CHKERRQ(ierr);
    printtofile = PETSC_FALSE;
    ierr        = PetscOptionsBool("-printtofile","Print convergence results to file","",printtofile,&printtofile,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
  if (printtofile) {
    ierr = TaoSetMonitor(tao,(PetscErrorCode (*)(Tao, void*))monitor,(void *)&ctx,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = TaoSetMaximumIterations(tao,30);CHKERRQ(ierr);
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,1,&p);CHKERRQ(ierr);
  ierr = VecGetArray(p,&x_ptr);CHKERRQ(ierr);
  x_ptr[0]   = ctx.Pm;
  ierr = VecRestoreArray(p,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,p);CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveRoutine(tao,FormFunction,(void *)&ctx);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(tao,TaoDefaultComputeGradient,(void *)&ctx);CHKERRQ(ierr);

  /* Set bounds for the optimization */
  ierr = VecDuplicate(p,&lowerb);CHKERRQ(ierr);
  ierr = VecDuplicate(p,&upperb);CHKERRQ(ierr);
  ierr = VecGetArray(lowerb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.;
  ierr = VecRestoreArray(lowerb,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(upperb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.1;
  ierr = VecRestoreArray(upperb,&x_ptr);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,lowerb,upperb);CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = VecView(p,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = VecDestroy(&lowerb);CHKERRQ(ierr);
  ierr = VecDestroy(&upperb);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunction - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
*/
PetscErrorCode FormFunction(Tao tao,Vec P,PetscReal *f,void *ctx0)
{
  AppCtx            *ctx = (AppCtx*)ctx0;
  TS                ts,quadts;
  Vec               U;             /* solution will be stored here */
  Mat               A;             /* Jacobian matrix */
  PetscErrorCode    ierr;
  PetscInt          n = 2;
  PetscReal         ftime;
  PetscInt          steps;
  PetscScalar       *u;
  const PetscScalar *x_ptr,*qx_ptr;
  Vec               q;
  PetscInt          direction[2];
  PetscBool         terminate[2];

  ierr = VecGetArrayRead(P,&x_ptr);CHKERRQ(ierr);
  ctx->Pm = x_ptr[0];
  ierr = VecRestoreArrayRead(P,&x_ptr);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = 1.0;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);
  ierr = TSCreateQuadratureTS(ts,PETSC_TRUE,&quadts);CHKERRQ(ierr);
  ierr = TSGetSolution(quadts,&q);CHKERRQ(ierr);
  ierr = VecSet(q,0.0);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,ctx);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;

  ierr = TSSetEventHandler(ts,2,direction,terminate,EventFunction,PostEventFunction,(void*)ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = VecGetArrayRead(q,&qx_ptr);CHKERRQ(ierr);
  *f   = -ctx->Pm + qx_ptr[0];
  ierr = VecRestoreArrayRead(q,&qx_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -ts_type cn -pc_type lu -tao_monitor -tao_gatol 1e-3

TEST*/
