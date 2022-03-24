static char help[] = "Pseudotransient continuation to solve a many-variable system that comes from the 2 variable Rosenbrock function + trivial.\n\n";

#include <petscts.h>

static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode MonitorObjective(TS,PetscInt,PetscReal,Vec,void*);

typedef struct {
  PetscInt  n;
  PetscBool monitor_short;
} Ctx;

int main(int argc,char **argv)
{
  TS             ts;            /* time integration context */
  Vec            X;             /* solution, residual vectors */
  Mat            J;             /* Jacobian matrix */
  PetscScalar    *x;
  PetscReal      ftime;
  PetscInt       i,steps,nits,lits;
  PetscBool      view_final;
  Ctx            ctx;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  ctx.n = 3;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&ctx.n,NULL));
  PetscCheck(ctx.n >= 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The dimension specified with -n must be at least 2");

  view_final = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_final",&view_final,NULL));

  ctx.monitor_short = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor_short",&ctx.monitor_short,NULL));

  /*
     Create Jacobian matrix data structure and state vector
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&J));
  CHKERRQ(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,ctx.n,ctx.n));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(MatSetUp(J));
  CHKERRQ(MatCreateVecs(J,&X,NULL));

  /* Create time integration context */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSPSEUDO));
  CHKERRQ(TSSetIFunction(ts,NULL,FormIFunction,&ctx));
  CHKERRQ(TSSetIJacobian(ts,J,J,FormIJacobian,&ctx));
  CHKERRQ(TSSetMaxSteps(ts,1000));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetTimeStep(ts,1e-3));
  CHKERRQ(TSMonitorSet(ts,MonitorObjective,&ctx,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize time integrator; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecSet(X,0.0));
  CHKERRQ(VecGetArray(X,&x));
#if 1
  x[0] = 5.;
  x[1] = -5.;
  for (i=2; i<ctx.n; i++) x[i] = 5.;
#else
  x[0] = 1.0;
  x[1] = 15.0;
  for (i=2; i<ctx.n; i++) x[i] = 10.0;
#endif
  CHKERRQ(VecRestoreArray(X,&x));

  CHKERRQ(TSSolve(ts,X));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(TSGetSNESIterations(ts,&nits));
  CHKERRQ(TSGetKSPIterations(ts,&lits));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Time integrator took (%D,%D,%D) iterations to reach final time %g\n",steps,nits,lits,(double)ftime));
  if (view_final) {
    CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(VecDestroy(&X));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFinalize());
  return 0;
}

static PetscErrorCode MonitorObjective(TS ts,PetscInt step,PetscReal t,Vec X,void *ictx)
{
  Ctx               *ctx = (Ctx*)ictx;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       f;
  PetscReal         dt,gnorm;
  PetscInt          i,snesit,linit;
  SNES              snes;
  Vec               Xdot,F;

  PetscFunctionBeginUser;
  /* Compute objective functional */
  CHKERRQ(VecGetArrayRead(X,&x));
  f    = 0;
  for (i=0; i<ctx->n-1; i++) f += PetscSqr(1. - x[i]) + 100. * PetscSqr(x[i+1] - PetscSqr(x[i]));
  CHKERRQ(VecRestoreArrayRead(X,&x));

  /* Compute norm of gradient */
  CHKERRQ(VecDuplicate(X,&Xdot));
  CHKERRQ(VecDuplicate(X,&F));
  CHKERRQ(VecZeroEntries(Xdot));
  CHKERRQ(FormIFunction(ts,t,X,Xdot,F,ictx));
  CHKERRQ(VecNorm(F,NORM_2,&gnorm));
  CHKERRQ(VecDestroy(&Xdot));
  CHKERRQ(VecDestroy(&F));

  CHKERRQ(TSGetTimeStep(ts,&dt));
  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(SNESGetIterationNumber(snes,&snesit));
  CHKERRQ(SNESGetLinearSolveIterations(snes,&linit));
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     (ctx->monitor_short
                      ? "%3D t=%10.1e  dt=%10.1e  f=%10.1e  df=%10.1e  it=(%2D,%3D)\n"
                      : "%3D t=%10.4e  dt=%10.4e  f=%10.4e  df=%10.4e  it=(%2D,%3D)\n"),
                     step,(double)t,(double)dt,(double)PetscRealPart(f),(double)gnorm,snesit,linit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormIFunction - Evaluates nonlinear function, F(X,Xdot) = Xdot + grad(objective(X))

   Input Parameters:
+  ts   - the TS context
.  t - time
.  X    - input vector
.  Xdot - time derivative
-  ctx  - optional user-defined context

   Output Parameter:
.  F - function vector
 */
static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ictx)
{
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt          i;
  Ctx               *ctx = (Ctx*)ictx;

  PetscFunctionBeginUser;
  /*
    Get pointers to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
    the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
    the array.
  */
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecZeroEntries(F));
  CHKERRQ(VecGetArray(F,&f));

  /* Compute gradient of objective */
  for (i=0; i<ctx->n-1; i++) {
    PetscScalar a,a0,a1;
    a       = x[i+1] - PetscSqr(x[i]);
    a0      = -2.*x[i];
    a1      = 1.;
    f[i]   += -2.*(1. - x[i]) + 200.*a*a0;
    f[i+1] += 200.*a*a1;
  }
  /* Restore vectors */
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
  CHKERRQ(VecAXPY(F,1.0,Xdot));
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
   FormIJacobian - Evaluates Jacobian matrix.

   Input Parameters:
+  ts - the TS context
.  t - pseudo-time
.  X - input vector
.  Xdot - time derivative
.  shift - multiplier for mass matrix
.  dummy - user-defined context

   Output Parameters:
.  J - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
static PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift,Mat J,Mat B,void *ictx)
{
  const PetscScalar *x;
  PetscInt          i;
  Ctx               *ctx = (Ctx*)ictx;

  PetscFunctionBeginUser;
  CHKERRQ(MatZeroEntries(B));
  /*
     Get pointer to vector data
  */
  CHKERRQ(VecGetArrayRead(X,&x));

  /*
     Compute Jacobian entries and insert into matrix.
  */
  for (i=0; i<ctx->n-1; i++) {
    PetscInt    rowcol[2];
    PetscScalar v[2][2],a,a0,a1,a00,a01,a10,a11;
    rowcol[0] = i;
    rowcol[1] = i+1;
    a         = x[i+1] - PetscSqr(x[i]);
    a0        = -2.*x[i];
    a00       = -2.;
    a01       = 0.;
    a1        = 1.;
    a10       = 0.;
    a11       = 0.;
    v[0][0]   = 2. + 200.*(a*a00 + a0*a0);
    v[0][1]   = 200.*(a*a01 + a1*a0);
    v[1][0]   = 200.*(a*a10 + a0*a1);
    v[1][1]   = 200.*(a*a11 + a1*a1);
    CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&v[0][0],ADD_VALUES));
  }
  for (i=0; i<ctx->n; i++) {
    CHKERRQ(MatSetValue(B,i,i,(PetscScalar)shift,ADD_VALUES));
  }

  CHKERRQ(VecRestoreArrayRead(X,&x));

  /*
     Assemble matrix
  */
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (J != B) {
    CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*TEST

    test:
      requires: !single

    test:
      args: -pc_type lu -ts_dt 1e-5 -ts_max_time 1e5 -n 50 -monitor_short -snes_max_it 5 -snes_type newtonls -ts_max_snes_failures -1
      requires: !single
      suffix: 2

TEST*/
