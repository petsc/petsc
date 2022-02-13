static char help[] = "Demonstrates automatic Jacobian generation using ADOL-C for a nonlinear reaction problem from chemistry.\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: Automatic differentiation using ADOL-C
   Processors: 1
*/
/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/
/* ------------------------------------------------------------------------
  See ../advection-diffusion-reaction/ex1 for a description of the problem
  ------------------------------------------------------------------------- */
#include <petscts.h>
#include "adolc-utils/drivers.cxx"
#include <adolc/adolc.h>

typedef struct {
  PetscScalar k;
  Vec         initialsolution;
  AdolcCtx    *adctx; /* Automatic differentiation support */
} AppCtx;

PetscErrorCode IFunctionView(AppCtx *ctx,PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryWrite(v,&ctx->k,1,PETSC_SCALAR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IFunctionLoad(AppCtx **ctx,PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(v,&(*ctx)->k,1,NULL,PETSC_SCALAR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Defines the ODE passed to the ODE solver
*/
PetscErrorCode IFunctionPassive(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = udot[0] + ctx->k*u[0]*u[1];
  f[1] = udot[1] + ctx->k*u[0]*u[1];
  f[2] = udot[2] - ctx->k*u[0]*u[1];
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  'Active' ADOL-C annotated version, marking dependence upon u.
*/
PetscErrorCode IFunctionActive1(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  adouble           f_a[3]; /* 'active' double for dependent variables */
  adouble           u_a[3]; /* 'active' double for independent variables */

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /* Start of active section */
  trace_on(1);
  u_a[0] <<= u[0]; u_a[1] <<= u[1]; u_a[2] <<= u[2]; /* Mark independence */
  f_a[0] = udot[0] + ctx->k*u_a[0]*u_a[1];
  f_a[1] = udot[1] + ctx->k*u_a[0]*u_a[1];
  f_a[2] = udot[2] - ctx->k*u_a[0]*u_a[1];
  f_a[0] >>= f[0]; f_a[1] >>= f[1]; f_a[2] >>= f[2]; /* Mark dependence */
  trace_off();
  /* End of active section */

  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  'Active' ADOL-C annotated version, marking dependence upon udot.
*/
PetscErrorCode IFunctionActive2(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  adouble           f_a[3];    /* 'active' double for dependent variables */
  adouble           udot_a[3]; /* 'active' double for independent variables */

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /* Start of active section */
  trace_on(2);
  udot_a[0] <<= udot[0]; udot_a[1] <<= udot[1]; udot_a[2] <<= udot[2]; /* Mark independence */
  f_a[0] = udot_a[0] + ctx->k*u[0]*u[1];
  f_a[1] = udot_a[1] + ctx->k*u[0]*u[1];
  f_a[2] = udot_a[2] - ctx->k*u[0]*u[1];
  f_a[0] >>= f[0]; f_a[1] >>= f[1]; f_a[2] >>= f[2];                   /* Mark dependence */
  trace_off();
  /* End of active section */

  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Defines the Jacobian of the ODE passed to the ODE solver, using the PETSc-ADOL-C driver for
 implicit TS.
*/
PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  AppCtx            *appctx = (AppCtx*)ctx;
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = PetscAdolcComputeIJacobian(1,2,A,u,a,appctx->adctx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the exact (analytic) solution to the ODE
*/
static PetscErrorCode Solution(TS ts,PetscReal t,Vec U,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *uinit;
  PetscScalar       *u,d0,q;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->initialsolution,&uinit);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  d0   = uinit[0] - uinit[1];
  if (d0 == 0.0) q = ctx->k*t;
  else q = (1.0 - PetscExpScalar(-ctx->k*t*d0))/d0;
  u[0] = uinit[0]/(1.0 + uinit[1]*q);
  u[1] = u[0] - d0;
  u[2] = uinit[1] + uinit[2] - u[1];
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->initialsolution,&uinit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U,Udot,R;      /* solution, derivative, residual */
  Mat            A;             /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 3;
  AppCtx         ctx;
  AdolcCtx       *adctx;
  PetscScalar    *u;
  const char     * const names[] = {"U1","U2","U3",NULL};

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");
  ierr = PetscNew(&adctx);CHKERRQ(ierr);
  adctx->m = n;adctx->n = n;adctx->p = n;
  ctx.adctx = adctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ctx.k = .9;
  ierr  = PetscOptionsGetScalar(NULL,NULL,"-k",&ctx.k,NULL);CHKERRQ(ierr);
  ierr  = VecDuplicate(U,&ctx.initialsolution);CHKERRQ(ierr);
  ierr  = VecGetArray(ctx.initialsolution,&u);CHKERRQ(ierr);
  u[0]  = 1;
  u[1]  = .7;
  u[2]  = 0;
  ierr  = VecRestoreArray(ctx.initialsolution,&u);CHKERRQ(ierr);
  ierr  = PetscOptionsGetVec(NULL,NULL,"-initial",ctx.initialsolution,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunctionPassive,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = Solution(ts,0,U,&ctx);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace just once for each tape
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDuplicate(U,&Udot);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&R);CHKERRQ(ierr);
  ierr = IFunctionActive1(ts,0.,U,Udot,R,&ctx);CHKERRQ(ierr);
  ierr = IFunctionActive2(ts,0.,U,Udot,R,&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = VecDestroy(&Udot);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetSolutionFunction(ts,(TSSolutionFunction)Solution,&ctx);CHKERRQ(ierr);

  {
    DM   dm;
    void *ptr;
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = PetscDLSym(NULL,"IFunctionView",&ptr);CHKERRQ(ierr);
    ierr = PetscDLSym(NULL,"IFunctionLoad",&ptr);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionSerialize(dm,(PetscErrorCode (*)(void*,PetscViewer))IFunctionView,(PetscErrorCode (*)(void**,PetscViewer))IFunctionLoad);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianSerialize(dm,(PetscErrorCode (*)(void*,PetscViewer))IFunctionView,(PetscErrorCode (*)(void**,PetscViewer))IFunctionLoad);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,1000);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,20.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSMonitorLGSetVariableNames(ts,names);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = TSView(ts,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&ctx.initialsolution);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree(adctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: double !complex adolc

  test:
    suffix: 1
    args: -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor
    output_file: output/adr_ex1_1.out

  test:
    suffix: 2
    args: -ts_max_steps 1 -snes_test_jacobian
    output_file: output/adr_ex1_2.out

TEST*/
