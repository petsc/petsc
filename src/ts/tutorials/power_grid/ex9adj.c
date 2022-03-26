
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
  Vec               U;
  PetscReal         t;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetSolution(ts,&U));
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"delta(%3.2f) = %8.7f\n",(double)t,(double)u[0]));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  PetscScalar       *f,Pmax;
  const PetscScalar *u;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else Pmax = ctx->Pmax;

  f[0] = ctx->omega_b*(u[1] - ctx->omega_s);
  f[1] = (-Pmax*PetscSinScalar(u[0]) - ctx->D*(u[1] - ctx->omega_s) + ctx->Pm)*ctx->omega_s/(2.0*ctx->H);

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,AppCtx *ctx)
{
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2],Pmax;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else Pmax = ctx->Pmax;

  J[0][0] = 0;                                  J[0][1] = ctx->omega_b;
  J[1][1] = -ctx->D*ctx->omega_s/(2.0*ctx->H);  J[1][0] = -Pmax*PetscCosScalar(u[0])*ctx->omega_s/(2.0*ctx->H);

  PetscCall(MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U,&u));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx0)
{
  PetscInt       row[] = {0,1},col[]={0};
  PetscScalar    J[2][1];
  AppCtx         *ctx=(AppCtx*)ctx0;

  PetscFunctionBeginUser;
  J[0][0] = 0;
  J[1][0] = ctx->omega_s/(2.0*ctx->H);
  PetscCall(MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode CostIntegrand(TS ts,PetscReal t,Vec U,Vec R,AppCtx *ctx)
{
  PetscScalar       *r;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(R,&r));
  r[0] = ctx->c*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta);
  PetscCall(VecRestoreArray(R,&r));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDU,Mat B,AppCtx *ctx)
{
  PetscScalar       ru[1];
  const PetscScalar *u;
  PetscInt          row[] = {0},col[] = {0};

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  ru[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta-1);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(MatSetValues(DRDU,1,row,1,col,ru,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(DRDU,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(DRDU,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDP,AppCtx *ctx)
{
  PetscFunctionBegin;
  PetscCall(MatZeroEntries(DRDP));
  PetscCall(MatAssemblyBegin(DRDP,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(DRDP,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeSensiP(Vec lambda,Vec mu,AppCtx *ctx)
{
  PetscScalar       sensip;
  const PetscScalar *x,*y;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(lambda,&x));
  PetscCall(VecGetArrayRead(mu,&y));
  sensip = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax*x[0]+y[0];
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameter pm: %.7f \n",(double)sensip));
  PetscCall(VecRestoreArrayRead(lambda,&x));
  PetscCall(VecRestoreArrayRead(mu,&y));
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
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetType(A,MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A,&U,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Jacp));
  PetscCall(MatSetSizes(Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  PetscCall(MatSetFromOptions(Jacp));
  PetscCall(MatSetUp(Jacp));

  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&DRDP));
  PetscCall(MatSetUp(DRDP));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,2,NULL,&DRDU));
  PetscCall(MatSetUp(DRDU));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Swing equation options","");PetscCall(ierr);
  {
    ctx.beta    = 2;
    ctx.c       = 10000.0;
    ctx.u_s     = 1.0;
    ctx.omega_s = 1.0;
    ctx.omega_b = 120.0*PETSC_PI;
    ctx.H       = 5.0;
    PetscCall(PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL));
    ctx.D       = 5.0;
    PetscCall(PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL));
    ctx.E       = 1.1378;
    ctx.V       = 1.0;
    ctx.X       = 0.545;
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    PetscCall(PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL));
    ctx.Pm      = 1.1;
    PetscCall(PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL));
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    PetscCall(PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL));
    PetscCall(PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL));
    PetscCall(PetscOptionsBool("-ensemble","Run ensemble of different initial conditions","",ensemble,&ensemble,NULL));
    if (ensemble) {
      ctx.tf      = -1;
      ctx.tcl     = -1;
    }

    PetscCall(VecGetArray(U,&u));
    u[0] = PetscAsinScalar(ctx.Pm/ctx.Pmax);
    u[1] = 1.0;
    PetscCall(PetscOptionsRealArray("-u","Initial solution","",u,&n,&flg1));
    n    = 2;
    PetscCall(PetscOptionsRealArray("-du","Perturbation in initial solution","",du,&n,&flg2));
    u[0] += du[0];
    u[1] += du[1];
    PetscCall(VecRestoreArray(U,&u));
    if (flg1 || flg2) {
      ctx.tf      = -1;
      ctx.tcl     = -1;
    }
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetEquationType(ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  PetscCall(TSSetType(ts,TSRK));
  PetscCall(TSSetRHSFunction(ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  PetscCall(TSSetRHSJacobian(ts,A,A,(TSRHSJacobian)RHSJacobian,&ctx));
  PetscCall(TSCreateQuadratureTS(ts,PETSC_TRUE,&quadts));
  PetscCall(TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
  PetscCall(TSSetRHSJacobian(quadts,DRDU,DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
  PetscCall(TSSetRHSJacobianP(quadts,DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSaveTrajectory(ts));

  PetscCall(MatCreateVecs(A,&lambda[0],NULL));
  /*   Set initial conditions for the adjoint integration */
  PetscCall(VecGetArray(lambda[0],&y_ptr));
  y_ptr[0] = 0.0; y_ptr[1] = 0.0;
  PetscCall(VecRestoreArray(lambda[0],&y_ptr));

  PetscCall(MatCreateVecs(Jacp,&mu[0],NULL));
  PetscCall(VecGetArray(mu[0],&x_ptr));
  x_ptr[0] = -1.0;
  PetscCall(VecRestoreArray(mu[0],&x_ptr));
  PetscCall(TSSetCostGradients(ts,1,lambda,mu));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts,10.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts,.01));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (ensemble) {
    for (du[1] = -2.5; du[1] <= .01; du[1] += .1) {
      PetscCall(VecGetArray(U,&u));
      u[0] = PetscAsinScalar(ctx.Pm/ctx.Pmax);
      u[1] = ctx.omega_s;
      u[0] += du[0];
      u[1] += du[1];
      PetscCall(VecRestoreArray(U,&u));
      PetscCall(TSSetTimeStep(ts,.01));
      PetscCall(TSSolve(ts,U));
    }
  } else {
    PetscCall(TSSolve(ts,U));
  }
  PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*   Set initial conditions for the adjoint integration */
  PetscCall(VecGetArray(lambda[0],&y_ptr));
  y_ptr[0] = 0.0; y_ptr[1] = 0.0;
  PetscCall(VecRestoreArray(lambda[0],&y_ptr));

  PetscCall(VecGetArray(mu[0],&x_ptr));
  x_ptr[0] = -1.0;
  PetscCall(VecRestoreArray(mu[0],&x_ptr));

  /*   Set RHS JacobianP */
  PetscCall(TSSetRHSJacobianP(ts,Jacp,RHSJacobianP,&ctx));

  PetscCall(TSAdjointSolve(ts));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[Psi(tf)]/d[phi0]  d[Psi(tf)]/d[omega0]\n"));
  PetscCall(VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(mu[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(TSGetCostIntegral(ts,&q));
  PetscCall(VecView(q,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecGetArray(q,&x_ptr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n cost function=%g\n",(double)(x_ptr[0]-ctx.Pm)));
  PetscCall(VecRestoreArray(q,&x_ptr));

  PetscCall(ComputeSensiP(lambda[0],mu[0],&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Jacp));
  PetscCall(MatDestroy(&DRDU));
  PetscCall(MatDestroy(&DRDP));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&lambda[0]));
  PetscCall(VecDestroy(&mu[0]));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -viewer_binary_skip_info -ts_adapt_type none

TEST*/
