
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

  ierr    = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

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
  r[0] = ctx->c*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta);CHKERRQ(ierr);
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
  ru[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta-1);CHKERRQ(ierr);
  ierr  = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr  = MatSetValues(DRDU,1,row,1,col,ru,INSERT_VALUES);CHKERRQ(ierr);
  ierr  = MatAssemblyBegin(DRDU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(DRDU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDP,AppCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(DRDP);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(DRDP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(DRDP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeSensiP(Vec lambda,Vec mu,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *y,sensip;
  const PetscScalar *x;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(lambda,&x);CHKERRQ(ierr);
  ierr = VecGetArray(mu,&y);CHKERRQ(ierr);
  sensip = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax*x[0]+y[0];
  y[0] = sensip;
  ierr = VecRestoreArray(mu,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(lambda,&x);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

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
    ierr        = PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL);CHKERRQ(ierr);
    ctx.D       = PetscRealConstant(5.0);
    ierr        = PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL);CHKERRQ(ierr);
    ctx.E       = PetscRealConstant(1.1378);
    ctx.V       = PetscRealConstant(1.0);
    ctx.X       = PetscRealConstant(0.545);
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    ierr        = PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL);CHKERRQ(ierr);
    ctx.Pm      = PetscRealConstant(1.0194);
    ierr        = PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL);CHKERRQ(ierr);
    ctx.tf      = PetscRealConstant(0.1);
    ctx.tcl     = PetscRealConstant(0.2);
    ierr        = PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL);CHKERRQ(ierr);

  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

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
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ctx.ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ctx.ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetEquationType(ctx.ts,TS_EQ_ODE_EXPLICIT);CHKERRQ(ierr); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  ierr = TSSetType(ctx.ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ctx.ts,NULL,(TSRHSFunction)RHSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ctx.ts,A,A,(TSRHSJacobian)RHSJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(Jacp,&mu[0],NULL);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ctx.ts,1,lambda,mu);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(ctx.ts,Jacp,RHSJacobianP,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ctx.ts,PetscRealConstant(1.0));CHKERRQ(ierr);
  ierr = TSSetTimeStep(ctx.ts,PetscRealConstant(0.01));CHKERRQ(ierr);
  ierr = TSSetFromOptions(ctx.ts);CHKERRQ(ierr);

  ierr = TSGetTimeStep(ctx.ts,&ctx.dt);CHKERRQ(ierr); /* save the stepsize */

  ierr = TSCreateQuadratureTS(ctx.ts,PETSC_TRUE,&quadts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(quadts,DRDU,DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(quadts,DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,&ctx);CHKERRQ(ierr);
  ierr = TSSetSolution(ctx.ts,U);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);

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
  ierr = TaoSetGradientRoutine(tao,FormGradient,(void *)&ctx);CHKERRQ(ierr);

  /* Set bounds for the optimization */
  ierr = VecDuplicate(p,&lowerb);CHKERRQ(ierr);
  ierr = VecDuplicate(p,&upperb);CHKERRQ(ierr);
  ierr = VecGetArray(lowerb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.;
  ierr = VecRestoreArray(lowerb,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(upperb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = PetscRealConstant(1.1);
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

  ierr = TSDestroy(&ctx.ts);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Jacp);CHKERRQ(ierr);
  ierr = MatDestroy(&DRDU);CHKERRQ(ierr);
  ierr = MatDestroy(&DRDP);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&mu[0]);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunction - Evaluates the function

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
*/
PetscErrorCode FormFunction(Tao tao,Vec P,PetscReal *f,void *ctx0)
{
  AppCtx         *ctx = (AppCtx*)ctx0;
  TS             ts = ctx->ts;
  Vec            U;             /* solution will be stored here */
  PetscErrorCode ierr;
  PetscScalar    *u;
  PetscScalar    *x_ptr;
  Vec            q;

  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  ctx->Pm = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);

  /* reset time */
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  /* reset step counter, this is critical for adjoint solver */
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  /* reset step size, the step size becomes negative after TSAdjointSolve */
  ierr = TSSetTimeStep(ts,ctx->dt);CHKERRQ(ierr);
  /* reinitialize the integral value */
  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = VecSet(q,0.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = PetscRealConstant(1.0);
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = VecGetArray(q,&x_ptr);CHKERRQ(ierr);
  *f   = -ctx->Pm + x_ptr[0];
  ierr = VecRestoreArray(q,&x_ptr);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormGradient(Tao tao,Vec P,Vec G,void *ctx0)
{
  AppCtx         *ctx = (AppCtx*)ctx0;
  TS             ts = ctx->ts;
  Vec            U;             /* solution will be stored here */
  PetscErrorCode ierr;
  PetscReal      ftime;
  PetscInt       steps;
  PetscScalar    *u;
  PetscScalar    *x_ptr,*y_ptr;
  Vec            *lambda,q,*mu;

  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  ctx->Pm = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);

  /* reset time */
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  /* reset step counter, this is critical for adjoint solver */
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  /* reset step size, the step size becomes negative after TSAdjointSolve */
  ierr = TSSetTimeStep(ts,ctx->dt);CHKERRQ(ierr);
  /* reinitialize the integral value */
  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = VecSet(q,0.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = PetscRealConstant(1.0);
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

  /* Set up to save trajectory before TSSetFromOptions() so that TSTrajectory options can be captured */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSGetCostGradients(ts,NULL,&lambda,&mu);CHKERRQ(ierr);
  /*   Set initial conditions for the adjoint integration */
  ierr = VecGetArray(lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 0.0; y_ptr[1] = 0.0;
  ierr = VecRestoreArray(lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(mu[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = PetscRealConstant(-1.0);
  ierr = VecRestoreArray(mu[0],&x_ptr);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = ComputeSensiP(lambda[0],mu[0],ctx);CHKERRQ(ierr);
  ierr = VecCopy(mu[0],G);CHKERRQ(ierr);
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
