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
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscReal      ftime;
  PetscInt       i,steps,nits,lits;
  PetscBool      view_final;
  Ctx            ctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ctx.n = 3;
  ierr  = PetscOptionsGetInt(NULL,NULL,"-n",&ctx.n,NULL);CHKERRQ(ierr);
  PetscCheckFalse(ctx.n < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The dimension specified with -n must be at least 2");

  view_final = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_final",&view_final,NULL);CHKERRQ(ierr);

  ctx.monitor_short = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor_short",&ctx.monitor_short,NULL);CHKERRQ(ierr);

  /*
     Create Jacobian matrix data structure and state vector
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,ctx.n,ctx.n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = MatCreateVecs(J,&X,NULL);CHKERRQ(ierr);

  /* Create time integration context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSPSEUDO);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,1000);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-3);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,MonitorObjective,&ctx,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize time integrator; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
#if 1
  x[0] = 5.;
  x[1] = -5.;
  for (i=2; i<ctx.n; i++) x[i] = 5.;
#else
  x[0] = 1.0;
  x[1] = 15.0;
  for (i=2; i<ctx.n; i++) x[i] = 10.0;
#endif
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetSNESIterations(ts,&nits);CHKERRQ(ierr);
  ierr = TSGetKSPIterations(ts,&lits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Time integrator took (%D,%D,%D) iterations to reach final time %g\n",steps,nits,lits,(double)ftime);CHKERRQ(ierr);
  if (view_final) {
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  f    = 0;
  for (i=0; i<ctx->n-1; i++) f += PetscSqr(1. - x[i]) + 100. * PetscSqr(x[i+1] - PetscSqr(x[i]));
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  /* Compute norm of gradient */
  ierr = VecDuplicate(X,&Xdot);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = VecZeroEntries(Xdot);CHKERRQ(ierr);
  ierr = FormIFunction(ts,t,X,Xdot,F,ictx);CHKERRQ(ierr);
  ierr = VecNorm(F,NORM_2,&gnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&Xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&snesit);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(snes,&linit);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
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
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

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
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecAXPY(F,1.0,Xdot);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt          i;
  Ctx               *ctx = (Ctx*)ictx;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  /*
     Get pointer to vector data
  */
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

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
    ierr      = MatSetValues(B,2,rowcol,2,rowcol,&v[0][0],ADD_VALUES);CHKERRQ(ierr);
  }
  for (i=0; i<ctx->n; i++) {
    ierr = MatSetValue(B,i,i,(PetscScalar)shift,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != B) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
