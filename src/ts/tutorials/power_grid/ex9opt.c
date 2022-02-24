
static char help[] = "Basic equation for generator stability analysis.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

  Ensemble of initial conditions
   ./ex2 -ensemble -ts_monitor_draw_solution_phase -1,-3,3,3      -ts_adapt_dt_max .01  -ts_monitor -ts_type rosw -pc_type lu -ksp_type preonly

  Fault at .1 seconds
   ./ex2           -ts_monitor_draw_solution_phase .42,.95,.6,1.05 -ts_adapt_dt_max .01  -ts_monitor -ts_type rosw -pc_type lu -ksp_type preonly

  Initial conditions same as when fault is ended
   ./ex2 -u 0.496792,1.00932 -ts_monitor_draw_solution_phase .42,.95,.6,1.05  -ts_adapt_dt_max .01  -ts_monitor -ts_type rosw -pc_type lu -ksp_type preonly

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

#include <petsctao.h>
#include <petscts.h>

typedef struct {
  TS          ts;
  PetscScalar H,D,omega_b,omega_s,Pmax,Pm,E,V,X,u_s,c;
  PetscInt    beta;
  PetscReal   tf,tcl,dt;
} AppCtx;

PetscErrorCode FormFunction(Tao,Vec,PetscReal*,void*);
PetscErrorCode FormGradient(Tao,Vec,Vec,void*);

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  PetscScalar       *f,Pmax;
  const PetscScalar *u;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else Pmax = ctx->Pmax;

  f[0] = ctx->omega_b*(u[1] - ctx->omega_s);
  f[1] = (-Pmax*PetscSinScalar(u[0]) - ctx->D*(u[1] - ctx->omega_s) + ctx->Pm)*ctx->omega_s/(2.0*ctx->H);

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
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
  CHKERRQ(VecGetArrayRead(U,&u));
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else Pmax = ctx->Pmax;

  J[0][0] = 0;                                  J[0][1] = ctx->omega_b;
  J[1][1] = -ctx->D*ctx->omega_s/(2.0*ctx->H);  J[1][0] = -Pmax*PetscCosScalar(u[0])*ctx->omega_s/(2.0*ctx->H);

  CHKERRQ(MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(U,&u));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
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
  CHKERRQ(MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode CostIntegrand(TS ts,PetscReal t,Vec U,Vec R,AppCtx *ctx)
{
  PetscScalar       *r;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(R,&r));
  r[0] = ctx->c*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta);
  CHKERRQ(VecRestoreArray(R,&r));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDU,Mat B,AppCtx *ctx)
{
  PetscScalar       ru[1];
  const PetscScalar *u;
  PetscInt          row[] = {0},col[] = {0};

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  ru[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta-1);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(MatSetValues(DRDU,1,row,1,col,ru,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(DRDU,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(DRDU,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDP,AppCtx *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(DRDP));
  CHKERRQ(MatAssemblyBegin(DRDP,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(DRDP,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeSensiP(Vec lambda,Vec mu,AppCtx *ctx)
{
  PetscScalar       *y,sensip;
  const PetscScalar *x;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(lambda,&x));
  CHKERRQ(VecGetArray(mu,&y));
  sensip = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax*x[0]+y[0];
  y[0] = sensip;
  CHKERRQ(VecRestoreArray(mu,&y));
  CHKERRQ(VecRestoreArrayRead(lambda,&x));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec            p;
  PetscScalar    *x_ptr;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  AppCtx         ctx;
  Vec            lowerb,upperb;
  Tao            tao;
  KSP            ksp;
  PC             pc;
  Vec            U,lambda[1],mu[1];
  Mat            A;             /* Jacobian matrix */
  Mat            Jacp;          /* Jacobian matrix */
  Mat            DRDU,DRDP;
  PetscInt       n = 2;
  TS             quadts;

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
    ctx.c       = PetscRealConstant(10000.0);
    ctx.u_s     = PetscRealConstant(1.0);
    ctx.omega_s = PetscRealConstant(1.0);
    ctx.omega_b = PetscRealConstant(120.0)*PETSC_PI;
    ctx.H       = PetscRealConstant(5.0);
    CHKERRQ(PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL));
    ctx.D       = PetscRealConstant(5.0);
    CHKERRQ(PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL));
    ctx.E       = PetscRealConstant(1.1378);
    ctx.V       = PetscRealConstant(1.0);
    ctx.X       = PetscRealConstant(0.545);
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    CHKERRQ(PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL));
    ctx.Pm      = PetscRealConstant(1.0194);
    CHKERRQ(PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL));
    ctx.tf      = PetscRealConstant(0.1);
    ctx.tcl     = PetscRealConstant(0.2);
    CHKERRQ(PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL));
    CHKERRQ(PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL));

  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(A,MATDENSE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&U,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Jacp));
  CHKERRQ(MatSetSizes(Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  CHKERRQ(MatSetFromOptions(Jacp));
  CHKERRQ(MatSetUp(Jacp));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&DRDP));
  CHKERRQ(MatSetUp(DRDP));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,2,NULL,&DRDU));
  CHKERRQ(MatSetUp(DRDU));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ctx.ts));
  CHKERRQ(TSSetProblemType(ctx.ts,TS_NONLINEAR));
  CHKERRQ(TSSetEquationType(ctx.ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  CHKERRQ(TSSetType(ctx.ts,TSRK));
  CHKERRQ(TSSetRHSFunction(ctx.ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  CHKERRQ(TSSetRHSJacobian(ctx.ts,A,A,(TSRHSJacobian)RHSJacobian,&ctx));
  CHKERRQ(TSSetExactFinalTime(ctx.ts,TS_EXACTFINALTIME_MATCHSTEP));

  CHKERRQ(MatCreateVecs(A,&lambda[0],NULL));
  CHKERRQ(MatCreateVecs(Jacp,&mu[0],NULL));
  CHKERRQ(TSSetCostGradients(ctx.ts,1,lambda,mu));
  CHKERRQ(TSSetRHSJacobianP(ctx.ts,Jacp,RHSJacobianP,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ctx.ts,PetscRealConstant(1.0)));
  CHKERRQ(TSSetTimeStep(ctx.ts,PetscRealConstant(0.01)));
  CHKERRQ(TSSetFromOptions(ctx.ts));

  CHKERRQ(TSGetTimeStep(ctx.ts,&ctx.dt)); /* save the stepsize */

  CHKERRQ(TSCreateQuadratureTS(ctx.ts,PETSC_TRUE,&quadts));
  CHKERRQ(TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
  CHKERRQ(TSSetRHSJacobian(quadts,DRDU,DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
  CHKERRQ(TSSetRHSJacobianP(quadts,DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,&ctx));
  CHKERRQ(TSSetSolution(ctx.ts,U));

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOBLMVM));

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
  CHKERRQ(TaoSetGradient(tao,NULL,FormGradient,(void *)&ctx));

  /* Set bounds for the optimization */
  CHKERRQ(VecDuplicate(p,&lowerb));
  CHKERRQ(VecDuplicate(p,&upperb));
  CHKERRQ(VecGetArray(lowerb,&x_ptr));
  x_ptr[0] = 0.;
  CHKERRQ(VecRestoreArray(lowerb,&x_ptr));
  CHKERRQ(VecGetArray(upperb,&x_ptr));
  x_ptr[0] = PetscRealConstant(1.1);
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

  CHKERRQ(TSDestroy(&ctx.ts));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Jacp));
  CHKERRQ(MatDestroy(&DRDU));
  CHKERRQ(MatDestroy(&DRDP));
  CHKERRQ(VecDestroy(&lambda[0]));
  CHKERRQ(VecDestroy(&mu[0]));
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunction - Evaluates the function

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
*/
PetscErrorCode FormFunction(Tao tao,Vec P,PetscReal *f,void *ctx0)
{
  AppCtx         *ctx = (AppCtx*)ctx0;
  TS             ts = ctx->ts;
  Vec            U;             /* solution will be stored here */
  PetscScalar    *u;
  PetscScalar    *x_ptr;
  Vec            q;

  CHKERRQ(VecGetArrayRead(P,(const PetscScalar**)&x_ptr));
  ctx->Pm = x_ptr[0];
  CHKERRQ(VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr));

  /* reset time */
  CHKERRQ(TSSetTime(ts,0.0));
  /* reset step counter, this is critical for adjoint solver */
  CHKERRQ(TSSetStepNumber(ts,0));
  /* reset step size, the step size becomes negative after TSAdjointSolve */
  CHKERRQ(TSSetTimeStep(ts,ctx->dt));
  /* reinitialize the integral value */
  CHKERRQ(TSGetCostIntegral(ts,&q));
  CHKERRQ(VecSet(q,0.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSGetSolution(ts,&U));
  CHKERRQ(VecGetArray(U,&u));
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = PetscRealConstant(1.0);
  CHKERRQ(VecRestoreArray(U,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));
  CHKERRQ(TSGetCostIntegral(ts,&q));
  CHKERRQ(VecGetArray(q,&x_ptr));
  *f   = -ctx->Pm + x_ptr[0];
  CHKERRQ(VecRestoreArray(q,&x_ptr));
  return 0;
}

PetscErrorCode FormGradient(Tao tao,Vec P,Vec G,void *ctx0)
{
  AppCtx         *ctx = (AppCtx*)ctx0;
  TS             ts = ctx->ts;
  Vec            U;             /* solution will be stored here */
  PetscReal      ftime;
  PetscInt       steps;
  PetscScalar    *u;
  PetscScalar    *x_ptr,*y_ptr;
  Vec            *lambda,q,*mu;

  CHKERRQ(VecGetArrayRead(P,(const PetscScalar**)&x_ptr));
  ctx->Pm = x_ptr[0];
  CHKERRQ(VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr));

  /* reset time */
  CHKERRQ(TSSetTime(ts,0.0));
  /* reset step counter, this is critical for adjoint solver */
  CHKERRQ(TSSetStepNumber(ts,0));
  /* reset step size, the step size becomes negative after TSAdjointSolve */
  CHKERRQ(TSSetTimeStep(ts,ctx->dt));
  /* reinitialize the integral value */
  CHKERRQ(TSGetCostIntegral(ts,&q));
  CHKERRQ(VecSet(q,0.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSGetSolution(ts,&U));
  CHKERRQ(VecGetArray(U,&u));
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = PetscRealConstant(1.0);
  CHKERRQ(VecRestoreArray(U,&u));

  /* Set up to save trajectory before TSSetFromOptions() so that TSTrajectory options can be captured */
  CHKERRQ(TSSetSaveTrajectory(ts));
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));

  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSGetCostGradients(ts,NULL,&lambda,&mu));
  /*   Set initial conditions for the adjoint integration */
  CHKERRQ(VecGetArray(lambda[0],&y_ptr));
  y_ptr[0] = 0.0; y_ptr[1] = 0.0;
  CHKERRQ(VecRestoreArray(lambda[0],&y_ptr));
  CHKERRQ(VecGetArray(mu[0],&x_ptr));
  x_ptr[0] = PetscRealConstant(-1.0);
  CHKERRQ(VecRestoreArray(mu[0],&x_ptr));

  CHKERRQ(TSAdjointSolve(ts));
  CHKERRQ(TSGetCostIntegral(ts,&q));
  CHKERRQ(ComputeSensiP(lambda[0],mu[0],ctx));
  CHKERRQ(VecCopy(mu[0],G));
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -viewer_binary_skip_info -ts_adapt_type none -tao_monitor -tao_gatol 0.0 -tao_grtol 1.e-3 -tao_converged_reason

   test:
      suffix: 2
      args: -viewer_binary_skip_info -ts_adapt_type none -tao_monitor -tao_gatol 0.0 -tao_grtol 1.e-3 -tao_converged_reason -tao_test_gradient

TEST*/
