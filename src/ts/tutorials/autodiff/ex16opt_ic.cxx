static char help[] = "Demonstrates automatic Jacobian generation using ADOL-C for an ODE-constrained optimization problem.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Concepts: Optimization using adjoint sensitivities
   Concepts: Automatic differentation using ADOL-C
   Processors: 1
*/
/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/
/* ------------------------------------------------------------------------
  See ex16opt_ic for a description of the problem being solved.
  ------------------------------------------------------------------------- */
#include <petsctao.h>
#include <petscts.h>
#include <petscmat.h>
#include "adolc-utils/drivers.cxx"
#include <adolc/adolc.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
  PetscInt  steps;

  /* Sensitivity analysis support */
  PetscReal ftime,x_ob[2];
  Mat       A;             /* Jacobian matrix */
  Vec       x,lambda[2];   /* adjoint variables */

  /* Automatic differentiation support */
  AdolcCtx  *adctx;
};

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

/*
  'Passive' RHS function, used in residual evaluations during the time integration.
*/
static PetscErrorCode RHSFunctionPassive(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));
  f[0] = x[1];
  f[1] = user->mu*(1.-x[0]*x[0])*x[1]-x[0];
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
  Trace RHS to mark on tape 1 the dependence of f upon x. This tape is used in generating the
  Jacobian transform.
*/
static PetscErrorCode RHSFunctionActive(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  User              user = (User)ctx;
  PetscReal         mu   = user->mu;
  PetscScalar       *f;
  const PetscScalar *x;

  adouble           f_a[2];                     /* adouble for dependent variables */
  adouble           x_a[2];                     /* adouble for independent variables */

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));

  trace_on(1);                                  /* Start of active section */
  x_a[0] <<= x[0]; x_a[1] <<= x[1];             /* Mark as independent */
  f_a[0] = x_a[1];
  f_a[1] = mu*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0]; f_a[1] >>= f[1];             /* Mark as dependent */
  trace_off(1);                                 /* End of active section */

  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
  Compute the Jacobian w.r.t. x using PETSc-ADOL-C driver.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  User              user=(User)ctx;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(PetscAdolcComputeRHSJacobian(1,A,x,user->adctx));
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscFunctionReturn(0);
}

/*
  Monitor timesteps and use interpolation to output at integer multiples of 0.1
*/
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  const PetscScalar *x;
  PetscReal         tfinal, dt, tprev;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetTimeStep(ts,&dt));
  PetscCall(TSGetMaxTime(ts,&tfinal));
  PetscCall(TSGetPrevTime(ts,&tprev));
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",(double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1])));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"t %.6f (tprev = %.6f) \n",(double)t,(double)tprev));
  PetscCall(VecGetArrayRead(X,&x));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS                 ts = NULL;          /* nonlinear solver */
  Vec                ic,r;
  PetscBool          monitor = PETSC_FALSE;
  PetscScalar        *x_ptr;
  PetscMPIInt        size;
  struct _n_User     user;
  AdolcCtx           *adctx;
  Tao                tao;
  KSP                ksp;
  PC                 pc;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options and create AdolcCtx
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscNew(&adctx));
  user.mu          = 1.0;
  user.next_output = 0.0;
  user.steps       = 0;
  user.ftime       = 0.5;
  adctx->m = 2;adctx->n = 2;adctx->p = 2;
  user.adctx = adctx;

  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.A));
  PetscCall(MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(user.A));
  PetscCall(MatSetUp(user.A));
  PetscCall(MatCreateVecs(user.A,&user.x,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(user.x,&x_ptr));
  x_ptr[0] = 2.0;   x_ptr[1] = 0.66666654321;
  PetscCall(VecRestoreArray(user.x,&x_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace just once on each tape and put zeros on Jacobian diagonal
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDuplicate(user.x,&r));
  PetscCall(RHSFunctionActive(ts,0.,user.x,r,&user));
  PetscCall(VecSet(r,0));
  PetscCall(MatDiagonalSet(user.A,r,INSERT_VALUES));
  PetscCall(VecDestroy(&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSRK));
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunctionPassive,&user));
  PetscCall(TSSetRHSJacobian(ts,user.A,user.A,RHSJacobian,&user));
  PetscCall(TSSetMaxTime(ts,user.ftime));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  if (monitor) {
    PetscCall(TSMonitorSet(ts,Monitor,&user,NULL));
  }

  PetscCall(TSSetTime(ts,0.0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)(user.ftime)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,user.x));
  PetscCall(TSGetSolveTime(ts,&(user.ftime)));
  PetscCall(TSGetStepNumber(ts,&user.steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)user.ftime));

  PetscCall(VecGetArray(user.x,&x_ptr));
  user.x_ob[0] = x_ptr[0];
  user.x_ob[1] = x_ptr[1];
  PetscCall(VecRestoreArray(user.x,&x_ptr));

  PetscCall(MatCreateVecs(user.A,&user.lambda[0],NULL));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
  PetscCall(TaoSetType(tao,TAOCG));

  /* Set initial solution guess */
  PetscCall(MatCreateVecs(user.A,&ic,NULL));
  PetscCall(VecGetArray(ic,&x_ptr));
  x_ptr[0]  = 2.1;
  x_ptr[1]  = 0.7;
  PetscCall(VecRestoreArray(ic,&x_ptr));

  PetscCall(TaoSetSolution(tao,ic));

  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user));

  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoGetKSP(tao,&ksp));
  if (ksp) {
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCNONE));
  }

  PetscCall(TaoSetTolerances(tao,1e-10,PETSC_DEFAULT,PETSC_DEFAULT));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&user.A));
  PetscCall(VecDestroy(&user.x));
  PetscCall(VecDestroy(&user.lambda[0]));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&ic));
  PetscCall(PetscFree(adctx));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  User              user = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr,*y_ptr;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(IC,user->x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSRK));
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunctionPassive,user));
  /*   Set RHS Jacobian  for the adjoint integration */
  PetscCall(TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetTimeStep(ts,.001));
  PetscCall(TSSetMaxTime(ts,0.5));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));

  PetscCall(TSSetTolerances(ts,1e-7,NULL,1e-7,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts,user->x));
  PetscCall(TSGetSolveTime(ts,&user->ftime));
  PetscCall(TSGetStepNumber(ts,&user->steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"mu %.6f, steps %D, ftime %g\n",(double)user->mu,user->steps,(double)user->ftime));

  PetscCall(VecGetArray(user->x,&x_ptr));
  *f   = (x_ptr[0]-user->x_ob[0])*(x_ptr[0]-user->x_ob[0])+(x_ptr[1]-user->x_ob[1])*(x_ptr[1]-user->x_ob[1]);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Observed value y_ob=[%f; %f], ODE solution y=[%f;%f], Cost function f=%f\n",(double)user->x_ob[0],(double)user->x_ob[1],(double)x_ptr[0],(double)x_ptr[1],(double)(*f)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*   Redet initial conditions for the adjoint integration */
  PetscCall(VecGetArray(user->lambda[0],&y_ptr));
  y_ptr[0] = 2.*(x_ptr[0]-user->x_ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user->x_ob[1]);
  PetscCall(VecRestoreArray(user->lambda[0],&y_ptr));
  PetscCall(VecRestoreArray(user->x,&x_ptr));
  PetscCall(TSSetCostGradients(ts,1,user->lambda,NULL));

  PetscCall(TSAdjointSolve(ts));

  PetscCall(VecCopy(user->lambda[0],G));

  PetscCall(TSDestroy(&ts));
  PetscFunctionReturn(0);
}

/*TEST

  build:
    requires: double !complex adolc

  test:
    suffix: 1
    args: -ts_rhs_jacobian_test_mult_transpose FALSE -tao_max_it 2 -ts_rhs_jacobian_test_mult FALSE
    output_file: output/ex16opt_ic_1.out

TEST*/
