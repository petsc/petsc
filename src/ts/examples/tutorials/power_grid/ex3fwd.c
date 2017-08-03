
static char help[] = "Forward sensitivity analysis of the basic equation for generator stability analysis.\n";

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
#include <petscts.h>

typedef struct {
  PetscScalar H,D,omega_b,omega_s,Pmax,Pmax_ini,Pm,E,V,X,u_s,c;
  PetscInt    beta;
  PetscReal   tf,tcl;
} AppCtx;

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

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Vec *V,void *ctx0)
{
  PetscErrorCode ierr;
  PetscScalar    *v;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(V[0]);CHKERRQ(ierr);
  ierr = VecZeroEntries(V[1]);CHKERRQ(ierr);
  ierr = VecGetArray(V[2],&v);CHKERRQ(ierr);
  v[0] = 0; v[1] = 1;
  ierr = VecRestoreArray(V[2],&v);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscScalar       *ry;
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(drdy[0],&ry);CHKERRQ(ierr);
  ry[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0., u[0]-ctx->u_s),ctx->beta-1);CHKERRQ(ierr);
  ry[1] = 0;
  ierr  = VecRestoreArray(drdy[0],&ry);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPFunction(TS ts,PetscReal t,Vec U,Vec *drdp,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *rp;
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(drdp[0],&rp);CHKERRQ(ierr);
  rp[0] = 0;
  rp[1] = 0;
  rp[2] = 0;
  ierr = VecRestoreArray(drdp[0],&rp);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
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
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  Vec            Jacp[3];          /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  AppCtx         ctx;
  PetscScalar    *u;
  PetscReal      du[2] = {0.0,0.0};
  PetscBool      ensemble = PETSC_FALSE,flg1,flg2;
  PetscScalar    *s_ptr;
  Vec            q,s[3],qgrad[1];
  PetscInt       direction[2];
  PetscBool      terminate[2];

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&Jacp[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&Jacp[1],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&Jacp[2],NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&qgrad[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(qgrad[0],PETSC_DECIDE,3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(qgrad[0]);CHKERRQ(ierr);

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
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /*   Set RHS JacobianP */
  ierr = TSForwardSetRHSJacobianP(ts,Jacp,RHSJacobianP,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&s[0],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(s[0],&s_ptr);CHKERRQ(ierr);
  s_ptr[0] = 1.0; s_ptr[1] = 0.0;
  ierr = VecRestoreArray(s[0],&s_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&s[1],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(s[1],&s_ptr);CHKERRQ(ierr);
  s_ptr[0] = 0.0; s_ptr[1] = 1.0;
  ierr = VecRestoreArray(s[1],&s_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&s[2],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(s[2],&s_ptr);CHKERRQ(ierr);
  s_ptr[0] = 1./PetscSqrtScalar(1.-(ctx.Pm/ctx.Pmax)*(ctx.Pm/ctx.Pmax))/ctx.Pmax;
  s_ptr[1] = 0.0;
  ierr = VecRestoreArray(s[2],&s_ptr);CHKERRQ(ierr);
  ierr = TSForwardSetSensitivities(ts,3,s,0,NULL);CHKERRQ(ierr);
  ierr = TSForwardSetIntegralGradients(ts,1,qgrad,NULL);CHKERRQ(ierr);
  ierr = TSSetCostIntegrand(ts,1,NULL,(PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))CostIntegrand,
                                        (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDYFunction,
                                        (PetscErrorCode (*)(TS,PetscReal,Vec,Vec*,void*))DRDPFunction,PETSC_TRUE,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;

  ierr = TSSetEventHandler(ts,2,direction,terminate,EventFunction,PostEventFunction,(void*)&ctx);CHKERRQ(ierr);

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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[phi(tf)]/d[phi0]  d[omega(tf)]/d[phi0]\n");CHKERRQ(ierr);
  ierr = VecView(s[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[phi(tf)]/d[omega0]  d[omega(tf)]/d[omega0]\n");CHKERRQ(ierr);
  ierr = VecView(s[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameters: d[phi(tf)]/d[pm]  d[omega(tf)]/d[pm]\n");CHKERRQ(ierr);
  ierr = VecView(s[2],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(ts,&q);CHKERRQ(ierr);
  ierr = VecGetArray(q,&s_ptr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n cost function=%g\n",(double)(s_ptr[0]-ctx.Pm));CHKERRQ(ierr);
  ierr = VecRestoreArray(q,&s_ptr);CHKERRQ(ierr);

  ierr = VecGetArray(qgrad[0],&s_ptr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameter pm: %.7f \n",(double)(s_ptr[2]-1.0));CHKERRQ(ierr);
  ierr = VecRestoreArray(q,&s_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&Jacp[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&Jacp[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&Jacp[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&qgrad[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&s[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&s[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&s[2]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
