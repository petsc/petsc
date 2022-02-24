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
  PetscFunctionBegin;
  CHKERRQ(PetscViewerBinaryWrite(v,&ctx->k,1,PETSC_SCALAR));
  PetscFunctionReturn(0);
}

PetscErrorCode IFunctionLoad(AppCtx **ctx,PetscViewer v)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNew(ctx));
  CHKERRQ(PetscViewerBinaryRead(v,&(*ctx)->k,1,NULL,PETSC_SCALAR));
  PetscFunctionReturn(0);
}

/*
  Defines the ODE passed to the ODE solver
*/
PetscErrorCode IFunctionPassive(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = udot[0] + ctx->k*u[0]*u[1];
  f[1] = udot[1] + ctx->k*u[0]*u[1];
  f[2] = udot[2] - ctx->k*u[0]*u[1];
  CHKERRQ(VecRestoreArray(F,&f));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/*
  'Active' ADOL-C annotated version, marking dependence upon u.
*/
PetscErrorCode IFunctionActive1(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  adouble           f_a[3]; /* 'active' double for dependent variables */
  adouble           u_a[3]; /* 'active' double for independent variables */

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(1);
  u_a[0] <<= u[0]; u_a[1] <<= u[1]; u_a[2] <<= u[2]; /* Mark independence */
  f_a[0] = udot[0] + ctx->k*u_a[0]*u_a[1];
  f_a[1] = udot[1] + ctx->k*u_a[0]*u_a[1];
  f_a[2] = udot[2] - ctx->k*u_a[0]*u_a[1];
  f_a[0] >>= f[0]; f_a[1] >>= f[1]; f_a[2] >>= f[2]; /* Mark dependence */
  trace_off();
  /* End of active section */

  CHKERRQ(VecRestoreArray(F,&f));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/*
  'Active' ADOL-C annotated version, marking dependence upon udot.
*/
PetscErrorCode IFunctionActive2(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  adouble           f_a[3];    /* 'active' double for dependent variables */
  adouble           udot_a[3]; /* 'active' double for independent variables */

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(2);
  udot_a[0] <<= udot[0]; udot_a[1] <<= udot[1]; udot_a[2] <<= udot[2]; /* Mark independence */
  f_a[0] = udot_a[0] + ctx->k*u[0]*u[1];
  f_a[1] = udot_a[1] + ctx->k*u[0]*u[1];
  f_a[2] = udot_a[2] - ctx->k*u[0]*u[1];
  f_a[0] >>= f[0]; f_a[1] >>= f[1]; f_a[2] >>= f[2];                   /* Mark dependence */
  trace_off();
  /* End of active section */

  CHKERRQ(VecRestoreArray(F,&f));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/*
 Defines the Jacobian of the ODE passed to the ODE solver, using the PETSc-ADOL-C driver for
 implicit TS.
*/
PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,AppCtx *ctx)
{
  AppCtx            *appctx = (AppCtx*)ctx;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(PetscAdolcComputeIJacobian(1,2,A,u,a,appctx->adctx));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/*
     Defines the exact (analytic) solution to the ODE
*/
static PetscErrorCode Solution(TS ts,PetscReal t,Vec U,AppCtx *ctx)
{
  const PetscScalar *uinit;
  PetscScalar       *u,d0,q;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(ctx->initialsolution,&uinit));
  CHKERRQ(VecGetArray(U,&u));
  d0   = uinit[0] - uinit[1];
  if (d0 == 0.0) q = ctx->k*t;
  else q = (1.0 - PetscExpScalar(-ctx->k*t*d0))/d0;
  u[0] = uinit[0]/(1.0 + uinit[1]*q);
  u[1] = u[0] - d0;
  u[2] = uinit[1] + uinit[2] - u[1];
  CHKERRQ(VecRestoreArray(U,&u));
  CHKERRQ(VecRestoreArrayRead(ctx->initialsolution,&uinit));
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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");
  CHKERRQ(PetscNew(&adctx));
  adctx->m = n;adctx->n = n;adctx->p = n;
  ctx.adctx = adctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&U,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ctx.k = .9;
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-k",&ctx.k,NULL));
  CHKERRQ(VecDuplicate(U,&ctx.initialsolution));
  CHKERRQ(VecGetArray(ctx.initialsolution,&u));
  u[0]  = 1;
  u[1]  = .7;
  u[2]  = 0;
  CHKERRQ(VecRestoreArray(ctx.initialsolution,&u));
  CHKERRQ(PetscOptionsGetVec(NULL,NULL,"-initial",ctx.initialsolution,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSROSW));
  CHKERRQ(TSSetIFunction(ts,NULL,(TSIFunction) IFunctionPassive,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(Solution(ts,0,U,&ctx));
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace just once for each tape
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDuplicate(U,&Udot));
  CHKERRQ(VecDuplicate(U,&R));
  CHKERRQ(IFunctionActive1(ts,0.,U,Udot,R,&ctx));
  CHKERRQ(IFunctionActive2(ts,0.,U,Udot,R,&ctx));
  CHKERRQ(VecDestroy(&R));
  CHKERRQ(VecDestroy(&Udot));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx));
  CHKERRQ(TSSetSolutionFunction(ts,(TSSolutionFunction)Solution,&ctx));

  {
    DM   dm;
    void *ptr;
    CHKERRQ(TSGetDM(ts,&dm));
    CHKERRQ(PetscDLSym(NULL,"IFunctionView",&ptr));
    CHKERRQ(PetscDLSym(NULL,"IFunctionLoad",&ptr));
    CHKERRQ(DMTSSetIFunctionSerialize(dm,(PetscErrorCode (*)(void*,PetscViewer))IFunctionView,(PetscErrorCode (*)(void**,PetscViewer))IFunctionLoad));
    CHKERRQ(DMTSSetIJacobianSerialize(dm,(PetscErrorCode (*)(void*,PetscViewer))IFunctionView,(PetscErrorCode (*)(void**,PetscViewer))IFunctionLoad));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetTimeStep(ts,.001));
  CHKERRQ(TSSetMaxSteps(ts,1000));
  CHKERRQ(TSSetMaxTime(ts,20.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSMonitorLGSetVariableNames(ts,names));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));

  CHKERRQ(TSView(ts,PETSC_VIEWER_BINARY_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&ctx.initialsolution));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFree(adctx));
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
