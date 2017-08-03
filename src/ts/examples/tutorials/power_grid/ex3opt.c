
static char help[] = "Finds optimal parameter P_m for the generator system while maintaining generator stability.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

F*/

/*
  This code demonstrates how to solve a ODE-constrained optimization problem with TAO, TSEvent, TSAdjoint and TS.
  The problem features discontinuities and a cost function in integral form.
  The gradient is computed with the discrete adjoint of an implicit theta method, see ex3adj.c for details.
*/
#include <petsctao.h>
#include <petscts.h>

typedef struct {
  PetscScalar H,D,omega_b,omega_s,Pmax,Pmax_ini,Pm,E,V,X,u_s,c;
  PetscInt    beta;
  PetscReal   tf,tcl;
} AppCtx;

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

/* Event check */
PetscErrorCode EventFunction(TS ts,PetscReal t,Vec X,PetscScalar *fvalue,void *ctx)
{
  AppCtx        *user=(AppCtx*)ctx;

  PetscFunctionBegin;
  /* Event for fault-on time */
  fvalue[0] = t - user->tf;
  /* Event for fault-off time */
  fvalue[1] = t - user->tcl;

  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec X,PetscBool forwardsolve,void* ctx)
{
  AppCtx *user=(AppCtx*)ctx;

  PetscFunctionBegin;

  if (event_list[0] == 0) {
    if (forwardsolve) user->Pmax = 0.0; /* Apply disturbance - this is done by setting Pmax = 0 */
    else user->Pmax = user->Pmax_ini; /* Going backward, reversal of event */
  } else if(event_list[0] == 1) {
    if (forwardsolve) user->Pmax = user->Pmax_ini; /* Remove the fault  - this is done by setting Pmax = Pmax_ini */
    else user->Pmax = 0.0; /* Going backward, reversal of event */
  }
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f,Pmax;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  Pmax = ctx->Pmax;
  f[0] = udot[0] - ctx->omega_b*(u[1] - ctx->omega_s);
  f[1] = 2.0*ctx->H/ctx->omega_s*udot[1] +  Pmax*PetscSinScalar(u[0]) + ctx->D*(u[1] - ctx->omega_s)- ctx->Pm;

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2],Pmax;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  Pmax = ctx->Pmax;
  J[0][0] = a;                       J[0][1] = -ctx->omega_b;
  J[1][1] = 2.0*ctx->H/ctx->omega_s*a + ctx->D;   J[1][0] = Pmax*PetscCosScalar(u[0]);

  ierr    = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);

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

  PetscFunctionBeginUser;
  J[0][0] = 0;
  J[1][0] = 1.;
  ierr  = MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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

static PetscErrorCode DRDYFunction(TS ts,PetscReal t,Vec U,Vec *drdy,AppCtx *ctx)
{
  PetscErrorCode     ierr;
  PetscScalar        *ry;
  const PetscScalar  *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(drdy[0],&ry);CHKERRQ(ierr);
  ry[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta-1);CHKERRQ(ierr);
  ierr = VecRestoreArray(drdy[0],&ry);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPFunction(TS ts,PetscReal t,Vec U,Vec *drdp,AppCtx *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *rp;

  PetscFunctionBegin;
  ierr = VecGetArray(drdp[0],&rp);CHKERRQ(ierr);
  rp[0] = 0.;
  ierr  = VecRestoreArray(drdp[0],&rp);CHKERRQ(ierr);
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
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameter pm: %g \n",(double)sensip);CHKERRQ(ierr); */
  y[0] = sensip;
  ierr = VecRestoreArray(mu,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(lambda,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode monitor(Tao tao,AppCtx *ctx)
{
  FILE               *fp;
  PetscInt           iterate;
  PetscReal          f,gnorm,cnorm,xdiff;
  TaoConvergedReason reason;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = TaoGetSolutionStatus(tao,&iterate,&f,&gnorm,&cnorm,&xdiff,&reason);CHKERRQ(ierr);

  fp = fopen("ex3opt_conv.out","a");
  ierr = PetscFPrintf(PETSC_COMM_WORLD,fp,"%D %g\n",iterate,(double)gnorm);CHKERRQ(ierr);
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
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  Vec                lowerb,upperb;
  PetscBool          printtofile;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

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
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;;
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
  if(printtofile) {
    ierr = TaoSetMonitor(tao,(PetscErrorCode (*)(Tao, void*))monitor,(void *)&ctx,PETSC_NULL);CHKERRQ(ierr);
  }
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,1,&p);CHKERRQ(ierr);
  ierr = VecGetArray(p,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = ctx.Pm;
  ierr = VecRestoreArray(p,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,p);CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&ctx);CHKERRQ(ierr);

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
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = VecDestroy(&lowerb);CHKERRQ(ierr);
  ierr = VecDestroy(&upperb);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx0)
{
  AppCtx         *ctx = (AppCtx*)ctx0;
  TS             ts;
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  Mat            Jacp;          /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscInt       n = 2;
  PetscReal      ftime;
  PetscInt       steps;
  PetscScalar    *u;
  PetscScalar    *x_ptr,*y_ptr;
  Vec            lambda[1],q,mu[1];
  PetscInt       direction[2];
  PetscBool      terminate[2];

  ierr = VecGetArray(P,&x_ptr);CHKERRQ(ierr);
  ctx->Pm = x_ptr[0];
  ierr = VecRestoreArray(P,&x_ptr);CHKERRQ(ierr);

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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,ctx);CHKERRQ(ierr);

  ierr = TSSetCostIntegrand(ts,1,NULL,(PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))CostIntegrand,
                                        (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDYFunction,
                                        (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDPFunction,PETSC_TRUE,ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = 1.0;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);
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
  /* ierr = VecView(U,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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

  /*   Set RHS JacobianP */
  ierr = TSAdjointSetRHSJacobian(ts,Jacp,RHSJacobianP,ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     One can set up the integral to be evaluated during the forward run
     instead by calling this function before TSSolve and specifying
     PETSC_TRUE for the second last argument
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetCostIntegrand(ts,1,NULL,(PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))CostIntegrand,
                                        (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDYFunction,
                                        (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDPFunction,PETSC_FALSE,ctx);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = ComputeSensiP(lambda[0],mu[0],ctx);CHKERRQ(ierr);
  ierr = VecCopy(mu[0],G);CHKERRQ(ierr);

  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = VecGetArray(q,&x_ptr);CHKERRQ(ierr);
  *f   = -ctx->Pm + x_ptr[0];
  ierr = VecRestoreArray(q,&x_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&mu[0]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  return 0;
}
