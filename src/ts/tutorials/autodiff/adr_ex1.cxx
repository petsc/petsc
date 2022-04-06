static char help[] = "Demonstrates automatic Jacobian generation using ADOL-C for a nonlinear reaction problem from chemistry.\n";

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
  PetscCall(PetscViewerBinaryWrite(v,&ctx->k,1,PETSC_SCALAR));
  PetscFunctionReturn(0);
}

PetscErrorCode IFunctionLoad(AppCtx **ctx,PetscViewer v)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(ctx));
  PetscCall(PetscViewerBinaryRead(v,&(*ctx)->k,1,NULL,PETSC_SCALAR));
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
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));
  f[0] = udot[0] + ctx->k*u[0]*u[1];
  f[1] = udot[1] + ctx->k*u[0]*u[1];
  f[2] = udot[2] - ctx->k*u[0]*u[1];
  PetscCall(VecRestoreArray(F,&f));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArrayRead(U,&u));
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
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(1);
  u_a[0] <<= u[0]; u_a[1] <<= u[1]; u_a[2] <<= u[2]; /* Mark independence */
  f_a[0] = udot[0] + ctx->k*u_a[0]*u_a[1];
  f_a[1] = udot[1] + ctx->k*u_a[0]*u_a[1];
  f_a[2] = udot[2] - ctx->k*u_a[0]*u_a[1];
  f_a[0] >>= f[0]; f_a[1] >>= f[1]; f_a[2] >>= f[2]; /* Mark dependence */
  trace_off();
  /* End of active section */

  PetscCall(VecRestoreArray(F,&f));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArrayRead(U,&u));
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
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(2);
  udot_a[0] <<= udot[0]; udot_a[1] <<= udot[1]; udot_a[2] <<= udot[2]; /* Mark independence */
  f_a[0] = udot_a[0] + ctx->k*u[0]*u[1];
  f_a[1] = udot_a[1] + ctx->k*u[0]*u[1];
  f_a[2] = udot_a[2] - ctx->k*u[0]*u[1];
  f_a[0] >>= f[0]; f_a[1] >>= f[1]; f_a[2] >>= f[2];                   /* Mark dependence */
  trace_off();
  /* End of active section */

  PetscCall(VecRestoreArray(F,&f));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArrayRead(U,&u));
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
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(PetscAdolcComputeIJacobian(1,2,A,u,a,appctx->adctx));
  PetscCall(VecRestoreArrayRead(U,&u));
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
  PetscCall(VecGetArrayRead(ctx->initialsolution,&uinit));
  PetscCall(VecGetArray(U,&u));
  d0   = uinit[0] - uinit[1];
  if (d0 == 0.0) q = ctx->k*t;
  else q = (1.0 - PetscExpScalar(-ctx->k*t*d0))/d0;
  u[0] = uinit[0]/(1.0 + uinit[1]*q);
  u[1] = u[0] - d0;
  u[2] = uinit[1] + uinit[2] - u[1];
  PetscCall(VecRestoreArray(U,&u));
  PetscCall(VecRestoreArrayRead(ctx->initialsolution,&uinit));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U,Udot,R;      /* solution, derivative, residual */
  Mat            A;             /* Jacobian matrix */
  PetscMPIInt    size;
  PetscInt       n = 3;
  AppCtx         ctx;
  AdolcCtx       *adctx;
  PetscScalar    *u;
  const char     * const names[] = {"U1","U2","U3",NULL};

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size <= 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");
  PetscCall(PetscNew(&adctx));
  adctx->m = n;adctx->n = n;adctx->p = n;
  ctx.adctx = adctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A,&U,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ctx.k = .9;
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-k",&ctx.k,NULL));
  PetscCall(VecDuplicate(U,&ctx.initialsolution));
  PetscCall(VecGetArray(ctx.initialsolution,&u));
  u[0]  = 1;
  u[1]  = .7;
  u[2]  = 0;
  PetscCall(VecRestoreArray(ctx.initialsolution,&u));
  PetscCall(PetscOptionsGetVec(NULL,NULL,"-initial",ctx.initialsolution,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSROSW));
  PetscCall(TSSetIFunction(ts,NULL,(TSIFunction) IFunctionPassive,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(Solution(ts,0,U,&ctx));
  PetscCall(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace just once for each tape
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDuplicate(U,&Udot));
  PetscCall(VecDuplicate(U,&R));
  PetscCall(IFunctionActive1(ts,0.,U,Udot,R,&ctx));
  PetscCall(IFunctionActive2(ts,0.,U,Udot,R,&ctx));
  PetscCall(VecDestroy(&R));
  PetscCall(VecDestroy(&Udot));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx));
  PetscCall(TSSetSolutionFunction(ts,(TSSolutionFunction)Solution,&ctx));

  {
    DM   dm;
    void *ptr;
    PetscCall(TSGetDM(ts,&dm));
    PetscCall(PetscDLSym(NULL,"IFunctionView",&ptr));
    PetscCall(PetscDLSym(NULL,"IFunctionLoad",&ptr));
    PetscCall(DMTSSetIFunctionSerialize(dm,(PetscErrorCode (*)(void*,PetscViewer))IFunctionView,(PetscErrorCode (*)(void**,PetscViewer))IFunctionLoad));
    PetscCall(DMTSSetIJacobianSerialize(dm,(PetscErrorCode (*)(void*,PetscViewer))IFunctionView,(PetscErrorCode (*)(void**,PetscViewer))IFunctionLoad));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetTimeStep(ts,.001));
  PetscCall(TSSetMaxSteps(ts,1000));
  PetscCall(TSSetMaxTime(ts,20.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSMonitorLGSetVariableNames(ts,names));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));

  PetscCall(TSView(ts,PETSC_VIEWER_BINARY_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&ctx.initialsolution));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFree(adctx));
  PetscCall(PetscFinalize());
  return 0;
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
