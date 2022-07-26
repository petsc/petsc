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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  ctx.n = 3;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&ctx.n,NULL));
  PetscCheck(ctx.n >= 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The dimension specified with -n must be at least 2");

  view_final = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_final",&view_final,NULL));

  ctx.monitor_short = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor_short",&ctx.monitor_short,NULL));

  /*
     Create Jacobian matrix data structure and state vector
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,ctx.n,ctx.n));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));
  PetscCall(MatCreateVecs(J,&X,NULL));

  /* Create time integration context */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSPSEUDO));
  PetscCall(TSSetIFunction(ts,NULL,FormIFunction,&ctx));
  PetscCall(TSSetIJacobian(ts,J,J,FormIJacobian,&ctx));
  PetscCall(TSSetMaxSteps(ts,1000));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts,1e-3));
  PetscCall(TSMonitorSet(ts,MonitorObjective,&ctx,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize time integrator; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecSet(X,0.0));
  PetscCall(VecGetArray(X,&x));
#if 1
  x[0] = 5.;
  x[1] = -5.;
  for (i=2; i<ctx.n; i++) x[i] = 5.;
#else
  x[0] = 1.0;
  x[1] = 15.0;
  for (i=2; i<ctx.n; i++) x[i] = 10.0;
#endif
  PetscCall(VecRestoreArray(X,&x));

  PetscCall(TSSolve(ts,X));
  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));
  PetscCall(TSGetSNESIterations(ts,&nits));
  PetscCall(TSGetKSPIterations(ts,&lits));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Time integrator took (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ") iterations to reach final time %g\n",steps,nits,lits,(double)ftime));
  if (view_final) PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&X));
  PetscCall(MatDestroy(&J));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode MonitorObjective(TS ts,PetscInt step,PetscReal t,Vec X,void *ictx)
{
  Ctx               *ctx = (Ctx*)ictx;
  const PetscScalar *x;
  PetscScalar       f;
  PetscReal         dt,gnorm;
  PetscInt          i,snesit,linit;
  SNES              snes;
  Vec               Xdot,F;

  PetscFunctionBeginUser;
  /* Compute objective functional */
  PetscCall(VecGetArrayRead(X,&x));
  f    = 0;
  for (i=0; i<ctx->n-1; i++) f += PetscSqr(1. - x[i]) + 100. * PetscSqr(x[i+1] - PetscSqr(x[i]));
  PetscCall(VecRestoreArrayRead(X,&x));

  /* Compute norm of gradient */
  PetscCall(VecDuplicate(X,&Xdot));
  PetscCall(VecDuplicate(X,&F));
  PetscCall(VecZeroEntries(Xdot));
  PetscCall(FormIFunction(ts,t,X,Xdot,F,ictx));
  PetscCall(VecNorm(F,NORM_2,&gnorm));
  PetscCall(VecDestroy(&Xdot));
  PetscCall(VecDestroy(&F));

  PetscCall(TSGetTimeStep(ts,&dt));
  PetscCall(TSGetSNES(ts,&snes));
  PetscCall(SNESGetIterationNumber(snes,&snesit));
  PetscCall(SNESGetLinearSolveIterations(snes,&linit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,(ctx->monitor_short ? "%3" PetscInt_FMT " t=%10.1e  dt=%10.1e  f=%10.1e  df=%10.1e  it=(%2" PetscInt_FMT ",%3" PetscInt_FMT ")\n"
                                                             : "%3" PetscInt_FMT " t=%10.4e  dt=%10.4e  f=%10.4e  df=%10.4e  it=(%2" PetscInt_FMT ",%3" PetscInt_FMT ")\n"),
                        step,(double)t,(double)dt,(double)PetscRealPart(f),(double)gnorm,snesit,linit));
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
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecZeroEntries(F));
  PetscCall(VecGetArray(F,&f));

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
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscCall(VecAXPY(F,1.0,Xdot));
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
  PetscCall(MatZeroEntries(B));
  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(X,&x));

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
    PetscCall(MatSetValues(B,2,rowcol,2,rowcol,&v[0][0],ADD_VALUES));
  }
  for (i=0; i<ctx->n; i++) {
    PetscCall(MatSetValue(B,i,i,(PetscScalar)shift,ADD_VALUES));
  }

  PetscCall(VecRestoreArrayRead(X,&x));

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (J != B) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
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
