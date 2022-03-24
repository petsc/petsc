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
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = x[1];
  f[1] = user->mu*(1.-x[0]*x[0])*x[1]-x[0];
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
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
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(1);
  x_a[0] <<= x[0];x_a[1] <<= x[1]; /* Mark independence */
  f_a[0] = x_a[1];
  f_a[1] = user->mu*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0];f_a[1] >>= f[1]; /* Mark dependence */
  trace_off();
  /* End of active section */

  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
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
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));

  /* Start of active section */
  trace_on(3);
  x_a[0] <<= x[0];x_a[1] <<= x[1];mu_a <<= user->mu; /* Mark independence */
  f_a[0] = x_a[1];
  f_a[1] = mu_a*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0];f_a[1] >>= f[1];                   /* Mark dependence */
  trace_off();
  /* End of active section */

  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
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
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(PetscAdolcComputeRHSJacobian(1,A,x,user->adctx));
  CHKERRQ(VecRestoreArrayRead(X,&x));
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
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(PetscAdolcComputeRHSJacobianP(3,A,x,&user->mu,user->adctx));
  CHKERRQ(VecRestoreArrayRead(X,&x));
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
  CHKERRQ(TSGetTimeStep(ts,&dt));
  CHKERRQ(TSGetMaxTime(ts,&tfinal));
  CHKERRQ(TSGetPrevTime(ts,&tprev));
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",(double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1])));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"t %.6f (tprev = %.6f) \n",(double)t,(double)tprev));
  CHKERRQ(VecRestoreArrayRead(X,&x));
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
  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options and create AdolcCtx
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscNew(&adctx));
  user.mu          = 1;
  user.next_output = 0.0;
  adctx->m = 2;adctx->n = 2;adctx->p = 2;adctx->num_params = 1;
  user.adctx = adctx;

  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatCreateVecs(A,&x,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Jacp));
  CHKERRQ(MatSetSizes(Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  CHKERRQ(MatSetFromOptions(Jacp));
  CHKERRQ(MatSetUp(Jacp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSRK));
  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunctionPassive,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(x,&x_ptr));
  x_ptr[0] = 2;   x_ptr[1] = 0.66666654321;
  CHKERRQ(VecRestoreArray(x,&x_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace just once on each tape and put zeros on Jacobian diagonal
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDuplicate(x,&r));
  CHKERRQ(RHSFunctionActive(ts,0.,x,r,&user));
  CHKERRQ(RHSFunctionActiveP(ts,0.,x,r,&user));
  CHKERRQ(VecSet(r,0));
  CHKERRQ(MatDiagonalSet(A,r,INSERT_VALUES));
  CHKERRQ(VecDestroy(&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set RHS Jacobian for the adjoint integration
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetRHSJacobian(ts,A,A,RHSJacobian,&user));
  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  if (monitor) {
    CHKERRQ(TSMonitorSet(ts,Monitor,&user,NULL));
  }
  CHKERRQ(TSSetTimeStep(ts,.001));

  /*
    Have the TS save its trajectory so that TSAdjointSolve() may be used
  */
  CHKERRQ(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,steps,(double)ftime));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Start the Adjoint model
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreateVecs(A,&lambda[0],NULL));
  CHKERRQ(MatCreateVecs(A,&lambda[1],NULL));
  /*   Reset initial conditions for the adjoint integration */
  CHKERRQ(VecGetArray(lambda[0],&x_ptr));
  x_ptr[0] = 1.0;   x_ptr[1] = 0.0;
  CHKERRQ(VecRestoreArray(lambda[0],&x_ptr));
  CHKERRQ(VecGetArray(lambda[1],&x_ptr));
  x_ptr[0] = 0.0;   x_ptr[1] = 1.0;
  CHKERRQ(VecRestoreArray(lambda[1],&x_ptr));

  CHKERRQ(MatCreateVecs(Jacp,&mu[0],NULL));
  CHKERRQ(MatCreateVecs(Jacp,&mu[1],NULL));
  CHKERRQ(VecGetArray(mu[0],&x_ptr));
  x_ptr[0] = 0.0;
  CHKERRQ(VecRestoreArray(mu[0],&x_ptr));
  CHKERRQ(VecGetArray(mu[1],&x_ptr));
  x_ptr[0] = 0.0;
  CHKERRQ(VecRestoreArray(mu[1],&x_ptr));
  CHKERRQ(TSSetCostGradients(ts,2,lambda,mu));

  /*   Set RHS JacobianP */
  CHKERRQ(TSSetRHSJacobianP(ts,Jacp,RHSJacobianP,&user));

  CHKERRQ(TSAdjointSolve(ts));

  CHKERRQ(VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(lambda[1],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(mu[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(mu[1],PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Jacp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&lambda[0]));
  CHKERRQ(VecDestroy(&lambda[1]));
  CHKERRQ(VecDestroy(&mu[0]));
  CHKERRQ(VecDestroy(&mu[1]));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFree(adctx));
  CHKERRQ(PetscFinalize());
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
