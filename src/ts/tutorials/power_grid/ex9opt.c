
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
  PetscScalar       *y,sensip;
  const PetscScalar *x;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(lambda,&x));
  PetscCall(VecGetArray(mu,&y));
  sensip = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax*x[0]+y[0];
  y[0] = sensip;
  PetscCall(VecRestoreArray(mu,&y));
  PetscCall(VecRestoreArrayRead(lambda,&x));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec            p;
  PetscScalar    *x_ptr;
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
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Swing equation options","");
  {
    ctx.beta    = 2;
    ctx.c       = PetscRealConstant(10000.0);
    ctx.u_s     = PetscRealConstant(1.0);
    ctx.omega_s = PetscRealConstant(1.0);
    ctx.omega_b = PetscRealConstant(120.0)*PETSC_PI;
    ctx.H       = PetscRealConstant(5.0);
    PetscCall(PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL));
    ctx.D       = PetscRealConstant(5.0);
    PetscCall(PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL));
    ctx.E       = PetscRealConstant(1.1378);
    ctx.V       = PetscRealConstant(1.0);
    ctx.X       = PetscRealConstant(0.545);
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    PetscCall(PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL));
    ctx.Pm      = PetscRealConstant(1.0194);
    PetscCall(PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL));
    ctx.tf      = PetscRealConstant(0.1);
    ctx.tcl     = PetscRealConstant(0.2);
    PetscCall(PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL));
    PetscCall(PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL));

  }
  PetscOptionsEnd();

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
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ctx.ts));
  PetscCall(TSSetProblemType(ctx.ts,TS_NONLINEAR));
  PetscCall(TSSetEquationType(ctx.ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  PetscCall(TSSetType(ctx.ts,TSRK));
  PetscCall(TSSetRHSFunction(ctx.ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  PetscCall(TSSetRHSJacobian(ctx.ts,A,A,(TSRHSJacobian)RHSJacobian,&ctx));
  PetscCall(TSSetExactFinalTime(ctx.ts,TS_EXACTFINALTIME_MATCHSTEP));

  PetscCall(MatCreateVecs(A,&lambda[0],NULL));
  PetscCall(MatCreateVecs(Jacp,&mu[0],NULL));
  PetscCall(TSSetCostGradients(ctx.ts,1,lambda,mu));
  PetscCall(TSSetRHSJacobianP(ctx.ts,Jacp,RHSJacobianP,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ctx.ts,PetscRealConstant(1.0)));
  PetscCall(TSSetTimeStep(ctx.ts,PetscRealConstant(0.01)));
  PetscCall(TSSetFromOptions(ctx.ts));

  PetscCall(TSGetTimeStep(ctx.ts,&ctx.dt)); /* save the stepsize */

  PetscCall(TSCreateQuadratureTS(ctx.ts,PETSC_TRUE,&quadts));
  PetscCall(TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
  PetscCall(TSSetRHSJacobian(quadts,DRDU,DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
  PetscCall(TSSetRHSJacobianP(quadts,DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,&ctx));
  PetscCall(TSSetSolution(ctx.ts,U));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
  PetscCall(TaoSetType(tao,TAOBLMVM));

  /*
     Optimization starts
  */
  /* Set initial solution guess */
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,1,&p));
  PetscCall(VecGetArray(p,&x_ptr));
  x_ptr[0]   = ctx.Pm;
  PetscCall(VecRestoreArray(p,&x_ptr));

  PetscCall(TaoSetSolution(tao,p));
  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjective(tao,FormFunction,(void *)&ctx));
  PetscCall(TaoSetGradient(tao,NULL,FormGradient,(void *)&ctx));

  /* Set bounds for the optimization */
  PetscCall(VecDuplicate(p,&lowerb));
  PetscCall(VecDuplicate(p,&upperb));
  PetscCall(VecGetArray(lowerb,&x_ptr));
  x_ptr[0] = 0.;
  PetscCall(VecRestoreArray(lowerb,&x_ptr));
  PetscCall(VecGetArray(upperb,&x_ptr));
  x_ptr[0] = PetscRealConstant(1.1);
  PetscCall(VecRestoreArray(upperb,&x_ptr));
  PetscCall(TaoSetVariableBounds(tao,lowerb,upperb));

  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoGetKSP(tao,&ksp));
  if (ksp) {
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCNONE));
  }

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  PetscCall(VecView(p,PETSC_VIEWER_STDOUT_WORLD));
  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&lowerb));
  PetscCall(VecDestroy(&upperb));

  PetscCall(TSDestroy(&ctx.ts));
  PetscCall(VecDestroy(&U));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Jacp));
  PetscCall(MatDestroy(&DRDU));
  PetscCall(MatDestroy(&DRDP));
  PetscCall(VecDestroy(&lambda[0]));
  PetscCall(VecDestroy(&mu[0]));
  PetscCall(PetscFinalize());
  return 0;
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

  PetscCall(VecGetArrayRead(P,(const PetscScalar**)&x_ptr));
  ctx->Pm = x_ptr[0];
  PetscCall(VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr));

  /* reset time */
  PetscCall(TSSetTime(ts,0.0));
  /* reset step counter, this is critical for adjoint solver */
  PetscCall(TSSetStepNumber(ts,0));
  /* reset step size, the step size becomes negative after TSAdjointSolve */
  PetscCall(TSSetTimeStep(ts,ctx->dt));
  /* reinitialize the integral value */
  PetscCall(TSGetCostIntegral(ts,&q));
  PetscCall(VecSet(q,0.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSGetSolution(ts,&U));
  PetscCall(VecGetArray(U,&u));
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = PetscRealConstant(1.0);
  PetscCall(VecRestoreArray(U,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));
  PetscCall(TSGetCostIntegral(ts,&q));
  PetscCall(VecGetArray(q,&x_ptr));
  *f   = -ctx->Pm + x_ptr[0];
  PetscCall(VecRestoreArray(q,&x_ptr));
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

  PetscCall(VecGetArrayRead(P,(const PetscScalar**)&x_ptr));
  ctx->Pm = x_ptr[0];
  PetscCall(VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr));

  /* reset time */
  PetscCall(TSSetTime(ts,0.0));
  /* reset step counter, this is critical for adjoint solver */
  PetscCall(TSSetStepNumber(ts,0));
  /* reset step size, the step size becomes negative after TSAdjointSolve */
  PetscCall(TSSetTimeStep(ts,ctx->dt));
  /* reinitialize the integral value */
  PetscCall(TSGetCostIntegral(ts,&q));
  PetscCall(VecSet(q,0.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSGetSolution(ts,&U));
  PetscCall(VecGetArray(U,&u));
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = PetscRealConstant(1.0);
  PetscCall(VecRestoreArray(U,&u));

  /* Set up to save trajectory before TSSetFromOptions() so that TSTrajectory options can be captured */
  PetscCall(TSSetSaveTrajectory(ts));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));

  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSGetCostGradients(ts,NULL,&lambda,&mu));
  /*   Set initial conditions for the adjoint integration */
  PetscCall(VecGetArray(lambda[0],&y_ptr));
  y_ptr[0] = 0.0; y_ptr[1] = 0.0;
  PetscCall(VecRestoreArray(lambda[0],&y_ptr));
  PetscCall(VecGetArray(mu[0],&x_ptr));
  x_ptr[0] = PetscRealConstant(-1.0);
  PetscCall(VecRestoreArray(mu[0],&x_ptr));

  PetscCall(TSAdjointSolve(ts));
  PetscCall(TSGetCostIntegral(ts,&q));
  PetscCall(ComputeSensiP(lambda[0],mu[0],ctx));
  PetscCall(VecCopy(mu[0],G));
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
