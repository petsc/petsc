static char help[] = "Demonstrates automatic Jacobian generation using ADOL-C for an adjoint sensitivity analysis of the van der Pol equation.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Concepts: TS^adjoint sensitivity analysis
   Concepts: Automatic differentation using ADOL-C
   Concepts: Automatic differentation w.r.t. a parameter using ADOL-C
   Processors: 1
*/
/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/
/* ------------------------------------------------------------------------
   See ex16adj for a description of the problem being solved.
  ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscmat.h>
#include "adolc-utils/drivers.cxx"
#include <adolc/adolc.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
  PetscReal tprev;

  /* Automatic differentiation support */
  AdolcCtx  *adctx;
};

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
  PetscScalar       *f;
  const PetscScalar *x;

  adouble           f_a[2]; /* 'active' double for dependent variables */
  adouble           x_a[2]; /* 'active' double for independent variables */

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(1);
  x_a[0] <<= x[0];x_a[1] <<= x[1]; /* Mark independence */
  f_a[0] = x_a[1];
  f_a[1] = user->mu*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0];f_a[1] >>= f[1]; /* Mark dependence */
  trace_off();
  /* End of active section */

  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
  Trace RHS again to mark on tape 2 the dependence of f upon the parameter mu. This tape is used in
  generating JacobianP.
*/
static PetscErrorCode RHSFunctionActiveP(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *x;

  adouble           f_a[2];      /* 'active' double for dependent variables */
  adouble           x_a[2],mu_a; /* 'active' double for independent variables */

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(3);
  x_a[0] <<= x[0];x_a[1] <<= x[1];mu_a <<= user->mu; /* Mark independence */
  f_a[0] = x_a[1];
  f_a[1] = mu_a*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0];f_a[1] >>= f[1];                   /* Mark dependence */
  trace_off();
  /* End of active section */

  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
  Compute the Jacobian w.r.t. x using PETSc-ADOL-C driver for explicit TS.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(PetscAdolcComputeRHSJacobian(1,A,x,user->adctx));
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscFunctionReturn(0);
}

/*
  Compute the Jacobian w.r.t. mu using PETSc-ADOL-C driver for explicit TS.
*/
static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(PetscAdolcComputeRHSJacobianP(3,A,x,&user->mu,user->adctx));
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
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            x;             /* solution, residual vectors */
  Mat            A;             /* Jacobian matrix */
  Mat            Jacp;          /* JacobianP matrix */
  PetscInt       steps;
  PetscReal      ftime   = 0.5;
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  AdolcCtx       *adctx;
  Vec            lambda[2],mu[2],r;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options and create AdolcCtx
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscNew(&adctx));
  user.mu          = 1;
  user.next_output = 0.0;
  adctx->m = 2;adctx->n = 2;adctx->p = 2;adctx->num_params = 1;
  user.adctx = adctx;

  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatCreateVecs(A,&x,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Jacp));
  PetscCall(MatSetSizes(Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  PetscCall(MatSetFromOptions(Jacp));
  PetscCall(MatSetUp(Jacp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSRK));
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunctionPassive,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(x,&x_ptr));
  x_ptr[0] = 2;   x_ptr[1] = 0.66666654321;
  PetscCall(VecRestoreArray(x,&x_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace just once on each tape and put zeros on Jacobian diagonal
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDuplicate(x,&r));
  PetscCall(RHSFunctionActive(ts,0.,x,r,&user));
  PetscCall(RHSFunctionActiveP(ts,0.,x,r,&user));
  PetscCall(VecSet(r,0));
  PetscCall(MatDiagonalSet(A,r,INSERT_VALUES));
  PetscCall(VecDestroy(&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set RHS Jacobian for the adjoint integration
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetRHSJacobian(ts,A,A,RHSJacobian,&user));
  PetscCall(TSSetMaxTime(ts,ftime));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  if (monitor) {
    PetscCall(TSMonitorSet(ts,Monitor,&user,NULL));
  }
  PetscCall(TSSetTimeStep(ts,.001));

  /*
    Have the TS save its trajectory so that TSAdjointSolve() may be used
  */
  PetscCall(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,x));
  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,steps,(double)ftime));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Start the Adjoint model
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreateVecs(A,&lambda[0],NULL));
  PetscCall(MatCreateVecs(A,&lambda[1],NULL));
  /*   Reset initial conditions for the adjoint integration */
  PetscCall(VecGetArray(lambda[0],&x_ptr));
  x_ptr[0] = 1.0;   x_ptr[1] = 0.0;
  PetscCall(VecRestoreArray(lambda[0],&x_ptr));
  PetscCall(VecGetArray(lambda[1],&x_ptr));
  x_ptr[0] = 0.0;   x_ptr[1] = 1.0;
  PetscCall(VecRestoreArray(lambda[1],&x_ptr));

  PetscCall(MatCreateVecs(Jacp,&mu[0],NULL));
  PetscCall(MatCreateVecs(Jacp,&mu[1],NULL));
  PetscCall(VecGetArray(mu[0],&x_ptr));
  x_ptr[0] = 0.0;
  PetscCall(VecRestoreArray(mu[0],&x_ptr));
  PetscCall(VecGetArray(mu[1],&x_ptr));
  x_ptr[0] = 0.0;
  PetscCall(VecRestoreArray(mu[1],&x_ptr));
  PetscCall(TSSetCostGradients(ts,2,lambda,mu));

  /*   Set RHS JacobianP */
  PetscCall(TSSetRHSJacobianP(ts,Jacp,RHSJacobianP,&user));

  PetscCall(TSAdjointSolve(ts));

  PetscCall(VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(lambda[1],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(mu[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(mu[1],PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Jacp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&lambda[0]));
  PetscCall(VecDestroy(&lambda[1]));
  PetscCall(VecDestroy(&mu[0]));
  PetscCall(VecDestroy(&mu[1]));
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
    output_file: output/ex16adj_1.out

  test:
    suffix: 2
    args: -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -mu 5
    output_file: output/ex16adj_2.out

  test:
    suffix: 3
    args: -ts_max_steps 10 -monitor
    output_file: output/ex16adj_3.out

  test:
    suffix: 4
    args: -ts_max_steps 10 -monitor -mu 5
    output_file: output/ex16adj_4.out

TEST*/
