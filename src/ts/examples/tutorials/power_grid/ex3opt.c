
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

typedef enum {SA_ADJ, SA_TLM} SAMethod;
static const char *const SAMethods[] = {"ADJ","TLM","SAMethod","SA_",0};

typedef struct {
  PetscScalar H,D,omega_b,omega_s,Pmax,Pmax_ini,Pm,E,V,X,u_s,c;
  PetscInt    beta;
  PetscReal   tf,tcl;
  /* Solver context */
  TS          ts;
  Vec         U;    /* solution will be stored here */
  Mat         Jac;  /* Jacobian matrix */
  Mat         Jacp; /* Jacobianp matrix */
  SAMethod    sa;
} AppCtx;

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

/* Event check */
PetscErrorCode EventFunction(TS ts,PetscReal t,Vec X,PetscScalar *fvalue,void *ctx)
{
  AppCtx *user=(AppCtx*)ctx;

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
  r[0] = ctx->c*PetscPowScalarInt(PetscMax(0.,u[0]-ctx->u_s),ctx->beta);CHKERRQ(ierr);
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
  ry[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0.,u[0]-ctx->u_s),ctx->beta-1);CHKERRQ(ierr);
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
  Vec                lambda[1],mu[1],lowerb,upperb;
  PetscBool          printtofile;
  PetscInt           direction[2];
  PetscBool          terminate[2];
  Vec                qgrad[1];      /* Forward sesivitiy */
  Mat                sp;            /* Forward sensitivity matrix */

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
    ctx.sa      = SA_ADJ;
    ierr        = PetscOptionsEnum("-sa_method","Sensitivity analysis method (adj or tlm)","",SAMethods,(PetscEnum)ctx.sa,(PetscEnum*)&ctx.sa,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&ctx.Jac);CHKERRQ(ierr);
  ierr = MatSetSizes(ctx.Jac,2,2,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(ctx.Jac,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx.Jac);CHKERRQ(ierr);
  ierr = MatSetUp(ctx.Jac);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&ctx.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(ctx.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(ctx.Jacp);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx.Jac,&ctx.U,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ctx.ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ctx.ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ctx.ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ctx.ts,NULL,(TSIFunction) IFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ctx.ts,ctx.Jac,ctx.Jac,(TSIJacobian)IJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(ctx.ts,ctx.Jacp,RHSJacobianP,&ctx);CHKERRQ(ierr);

  if (ctx.sa == SA_ADJ) {
    ierr = MatCreateVecs(ctx.Jac,&lambda[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(ctx.Jacp,&mu[0],NULL);CHKERRQ(ierr);
    ierr = TSSetSaveTrajectory(ctx.ts);CHKERRQ(ierr);
    ierr = TSSetCostGradients(ctx.ts,1,lambda,mu);CHKERRQ(ierr);
    ierr = TSSetCostIntegrand(ctx.ts,1,NULL,(PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))CostIntegrand,
                              (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDYFunction,
                              (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDPFunction,PETSC_TRUE,&ctx);CHKERRQ(ierr);
  }
  if (ctx.sa == SA_TLM) {
    ierr = VecCreate(PETSC_COMM_WORLD,&qgrad[0]);CHKERRQ(ierr);
    ierr = VecSetSizes(qgrad[0],PETSC_DECIDE,1);CHKERRQ(ierr);
    ierr = VecSetFromOptions(qgrad[0]);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&sp);CHKERRQ(ierr);
    ierr = TSForwardSetSensitivities(ctx.ts,1,sp);CHKERRQ(ierr);
    ierr = TSForwardSetIntegralGradients(ctx.ts,1,qgrad);CHKERRQ(ierr);
    ierr = TSSetCostIntegrand(ctx.ts,1,NULL,(PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))CostIntegrand,
                              (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDYFunction,
                              (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDPFunction,PETSC_TRUE,&ctx);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ctx.ts,1.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ctx.ts,0.03125);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ctx.ts);CHKERRQ(ierr);

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;
  ierr = TSSetEventHandler(ctx.ts,2,direction,terminate,EventFunction,PostEventFunction,&ctx);CHKERRQ(ierr);

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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&ctx.Jac);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx.Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.U);CHKERRQ(ierr);
  if (ctx.sa == SA_ADJ) {
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&mu[0]);CHKERRQ(ierr);
  }
  if (ctx.sa == SA_TLM) {
    ierr = VecDestroy(&qgrad[0]);CHKERRQ(ierr);
    ierr = MatDestroy(&sp);CHKERRQ(ierr);
  }
  ierr = TSDestroy(&ctx.ts);CHKERRQ(ierr);
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
  PetscInt       nadj;
  PetscReal      ftime;
  PetscInt       steps;
  PetscScalar    *u;
  PetscScalar    *x_ptr,*y_ptr;
  Vec            q,*qgrad;
  PetscErrorCode ierr;

  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  ctx->Pm = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);

  /* reinitialize the solution vector */
  ierr = VecGetArray(ctx->U,&u);CHKERRQ(ierr);
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = 1.0;
  ierr = VecRestoreArray(ctx->U,&u);CHKERRQ(ierr);
  ierr = TSSetSolution(ctx->ts,ctx->U);CHKERRQ(ierr);

  /* reset time */
  ierr = TSSetTime(ctx->ts,0.0);CHKERRQ(ierr);

  /* reset step counter, this is critical for adjoint solver */
  ierr = TSSetStepNumber(ctx->ts,0);CHKERRQ(ierr);

  /* reset step size, the step size becomes negative after TSAdjointSolve */
  ierr = TSSetTimeStep(ctx->ts,0.03125);CHKERRQ(ierr);

  /* reinitialize the integral value */
  ierr = TSGetCostIntegral(ctx->ts,&q);CHKERRQ(ierr);
  ierr = VecSet(q,0.0);CHKERRQ(ierr);

  if (ctx->sa == SA_TLM) { /* reset the forward sensitivities */
    Mat            sp;
    PetscScalar    val[2];
    const PetscInt row[]={0,1},col[]={0};

    ierr = TSForwardGetIntegralGradients(ctx->ts,NULL,&qgrad);CHKERRQ(ierr);
    ierr = VecSet(qgrad[0],0.0);CHKERRQ(ierr);
    ierr = TSForwardGetSensitivities(ctx->ts,NULL,&sp);CHKERRQ(ierr);
    val[0] = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax;
    val[1] = 0.0;
    ierr = MatSetValues(sp,2,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(sp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* sovle the ODE */
  ierr = TSSolve(ctx->ts,ctx->U);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ctx->ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ctx->ts,&steps);CHKERRQ(ierr);

  if (ctx->sa == SA_ADJ) {
    Vec *lambda,*mu;
    /* reset the terminal condition for adjoint */
    ierr = TSGetCostGradients(ctx->ts,&nadj,&lambda,&mu);CHKERRQ(ierr);
    ierr = VecGetArray(lambda[0],&y_ptr);CHKERRQ(ierr);
    y_ptr[0] = 0.0; y_ptr[1] = 0.0;
    ierr = VecRestoreArray(lambda[0],&y_ptr);CHKERRQ(ierr);
    ierr = VecGetArray(mu[0],&x_ptr);CHKERRQ(ierr);
    x_ptr[0] = -1.0;
    ierr = VecRestoreArray(mu[0],&x_ptr);CHKERRQ(ierr);

    /* solve the adjont */
    ierr = TSAdjointSolve(ctx->ts);CHKERRQ(ierr);

    ierr = ComputeSensiP(lambda[0],mu[0],ctx);CHKERRQ(ierr);
    ierr = VecCopy(mu[0],G);CHKERRQ(ierr);
  }

  if (ctx->sa == SA_TLM) {
    ierr = VecGetArray(G,&x_ptr);CHKERRQ(ierr);
    ierr = VecGetArray(qgrad[0],&y_ptr);CHKERRQ(ierr);
    x_ptr[0] = y_ptr[0]-1.;
    ierr = VecRestoreArray(qgrad[0],&y_ptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&x_ptr);CHKERRQ(ierr);
  }

  ierr = TSGetCostIntegral(ctx->ts,&q);CHKERRQ(ierr);
  ierr = VecGetArray(q,&x_ptr);CHKERRQ(ierr);
  *f   = -ctx->Pm + x_ptr[0];
  ierr = VecRestoreArray(q,&x_ptr);CHKERRQ(ierr);
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -viewer_binary_skip_info -ts_type cn -pc_type lu -tao_monitor

   test:
      suffix: 2
      output_file: output/ex3opt_1.out
      args: -sa_method tlm -ts_type cn -pc_type lu -tao_monitor
TEST*/
