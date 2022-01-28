
static char help[] = "Basic equation for generator stability analysis.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

  Ensemble of initial conditions
   ./ex9 -ensemble -ts_monitor_draw_solution_phase -1,-3,3,3 -ts_adapt_dt_max .01 -ts_monitor -ts_type rk -pc_type lu -ksp_type preonly

  Fault at .1 seconds
   ./ex9 -ts_monitor_draw_solution_phase .42,.95,.6,1.05 -ts_adapt_dt_max .01 -ts_monitor -ts_type rk -pc_type lu -ksp_type preonly

  Initial conditions same as when fault is ended
   ./ex9 -u 0.496792,1.00932 -ts_monitor_draw_solution_phase .42,.95,.6,1.05 -ts_adapt_dt_max .01 -ts_monitor -ts_type rk -pc_type lu -ksp_type preonly

F*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscts.h>

typedef struct {
  PetscScalar H,D,omega_b,omega_s,Pmax,Pm,E,V,X,u_s,c;
  PetscInt    beta;
  PetscReal   tf,tcl;
} AppCtx;

PetscErrorCode PostStepFunction(TS ts)
{
  PetscErrorCode    ierr;
  Vec               U;
  PetscReal         t;
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"delta(%3.2f) = %8.7f\n",(double)t,(double)u[0]);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f,Pmax;
  const PetscScalar *u;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else Pmax = ctx->Pmax;

  f[0] = ctx->omega_b*(u[1] - ctx->omega_s);
  f[1] = (-Pmax*PetscSinScalar(u[0]) - ctx->D*(u[1] - ctx->omega_s) + ctx->Pm)*ctx->omega_s/(2.0*ctx->H);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2],Pmax;
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else Pmax = ctx->Pmax;

  J[0][0] = 0;                                  J[0][1] = ctx->omega_b;
  J[1][1] = -ctx->D*ctx->omega_s/(2.0*ctx->H);  J[1][0] = -Pmax*PetscCosScalar(u[0])*ctx->omega_s/(2.0*ctx->H);

  ierr = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx0)
{
  PetscErrorCode ierr;
  PetscInt       row[] = {0,1},col[]={0};
  PetscScalar    J[2][1];
  AppCtx         *ctx=(AppCtx*)ctx0;

  PetscFunctionBeginUser;
  J[0][0] = 0;
  J[1][0] = ctx->omega_s/(2.0*ctx->H);
  ierr    = MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr    = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CostIntegrand(TS ts,PetscReal t,Vec U,Vec R,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *r;
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);
  r[0] = ctx->c*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDU,Mat B,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       ru[1];
  const PetscScalar *u;
  PetscInt          row[] = {0},col[] = {0};

  PetscFunctionBegin;
  ierr  = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ru[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta-1);
  ierr  = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr  = MatSetValues(DRDU,1,row,1,col,ru,INSERT_VALUES);CHKERRQ(ierr);
  ierr  = MatAssemblyBegin(DRDU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(DRDU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDP,AppCtx *ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(DRDP);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(DRDP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(DRDP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeSensiP(Vec lambda,Vec mu,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       sensip;
  const PetscScalar *x,*y;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(lambda,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(mu,&y);CHKERRQ(ierr);
  sensip = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax*x[0]+y[0];
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameter pm: %.7f \n",(double)sensip);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(lambda,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(mu,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts,quadts;     /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  Mat            Jacp;          /* Jacobian matrix */
  Mat            DRDU,DRDP;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  AppCtx         ctx;
  PetscScalar    *u;
  PetscReal      du[2] = {0.0,0.0};
  PetscBool      ensemble = PETSC_FALSE,flg1,flg2;
  PetscReal      ftime;
  PetscInt       steps;
  PetscScalar    *x_ptr,*y_ptr;
  Vec            lambda[1],q,mu[1];

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(Jacp);CHKERRQ(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&DRDP);CHKERRQ(ierr);
  ierr = MatSetUp(DRDP);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,2,NULL,&DRDU);CHKERRQ(ierr);
  ierr = MatSetUp(DRDU);CHKERRQ(ierr);

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
    ierr        = PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL);CHKERRQ(ierr);
    ctx.Pm      = 1.1;
    ierr        = PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL);CHKERRQ(ierr);
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    ierr        = PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsBool("-ensemble","Run ensemble of different initial conditions","",ensemble,&ensemble,NULL);CHKERRQ(ierr);
    if (ensemble) {
      ctx.tf      = -1;
      ctx.tcl     = -1;
    }

    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    u[0] = PetscAsinScalar(ctx.Pm/ctx.Pmax);
    u[1] = 1.0;
    ierr = PetscOptionsRealArray("-u","Initial solution","",u,&n,&flg1);CHKERRQ(ierr);
    n    = 2;
    ierr = PetscOptionsRealArray("-du","Perturbation in initial solution","",du,&n,&flg2);CHKERRQ(ierr);
    u[0] += du[0];
    u[1] += du[1];
    ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
    if (flg1 || flg2) {
      ctx.tf      = -1;
      ctx.tcl     = -1;
    }
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_ODE_EXPLICIT);CHKERRQ(ierr); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,(TSRHSFunction)RHSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,(TSRHSJacobian)RHSJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSCreateQuadratureTS(ts,PETSC_TRUE,&quadts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(quadts,DRDU,DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(quadts,DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&lambda[0],NULL);CHKERRQ(ierr);
  /*   Set initial conditions for the adjoint integration */
  ierr = VecGetArray(lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 0.0; y_ptr[1] = 0.0;
  ierr = VecRestoreArray(lambda[0],&y_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(Jacp,&mu[0],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(mu[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = -1.0;
  ierr = VecRestoreArray(mu[0],&x_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,lambda,mu);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,10.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.01);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (ensemble) {
    for (du[1] = -2.5; du[1] <= .01; du[1] += .1) {
      ierr = VecGetArray(U,&u);CHKERRQ(ierr);
      u[0] = PetscAsinScalar(ctx.Pm/ctx.Pmax);
      u[1] = ctx.omega_s;
      u[0] += du[0];
      u[1] += du[1];
      ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
      ierr = TSSetTimeStep(ts,.01);CHKERRQ(ierr);
      ierr = TSSolve(ts,U);CHKERRQ(ierr);
    }
  } else {
    ierr = TSSolve(ts,U);CHKERRQ(ierr);
  }
  ierr = VecView(U,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*   Set initial conditions for the adjoint integration */
  ierr = VecGetArray(lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 0.0; y_ptr[1] = 0.0;
  ierr = VecRestoreArray(lambda[0],&y_ptr);CHKERRQ(ierr);

  ierr = VecGetArray(mu[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = -1.0;
  ierr = VecRestoreArray(mu[0],&x_ptr);CHKERRQ(ierr);

  /*   Set RHS JacobianP */
  ierr = TSSetRHSJacobianP(ts,Jacp,RHSJacobianP,&ctx);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[Psi(tf)]/d[phi0]  d[Psi(tf)]/d[omega0]\n");CHKERRQ(ierr);
  ierr = VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(mu[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = VecView(q,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecGetArray(q,&x_ptr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n cost function=%g\n",(double)(x_ptr[0]-ctx.Pm));CHKERRQ(ierr);
  ierr = VecRestoreArray(q,&x_ptr);CHKERRQ(ierr);

  ierr = ComputeSensiP(lambda[0],mu[0],&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Jacp);CHKERRQ(ierr);
  ierr = MatDestroy(&DRDU);CHKERRQ(ierr);
  ierr = MatDestroy(&DRDP);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&mu[0]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -viewer_binary_skip_info -ts_adapt_type none

TEST*/
