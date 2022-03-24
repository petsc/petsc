static char help[] = "Demonstrates tapeless automatic Jacobian generation using ADOL-C for an adjoint sensitivity analysis of the van der Pol equation.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Concepts: TS^adjoint sensitivity analysis
   Concepts: Automatic differentation using ADOL-C
   Concepts: Tapeless automatic differentiation using ADOL-C
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

#define ADOLC_TAPELESS
#define NUMBER_DIRECTIONS 3
#include "adolc-utils/drivers.cxx"
#include <adolc/adtl.h>
using namespace adtl;

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
  PetscReal tprev;

  /* Automatic differentiation support */
  AdolcCtx  *adctx;
  Vec       F;
};

/*
  Residual evaluation templated, so as to allow for PetscScalar or adouble
  arguments.
*/
template <class T> PetscErrorCode EvaluateResidual(const T *x,T mu,T *f)
{
  PetscFunctionBegin;
  f[0] = x[1];
  f[1] = mu*(1.-x[0]*x[0])*x[1]-x[0];
  PetscFunctionReturn(0);
}

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
  CHKERRQ(EvaluateResidual(x,user->mu,f));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
  Compute the Jacobian w.r.t. x using tapeless mode of ADOL-C.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  User              user = (User)ctx;
  PetscScalar       **J;
  const PetscScalar *x;
  adouble           f_a[2];      /* 'active' double for dependent variables */
  adouble           x_a[2],mu_a; /* 'active' doubles for independent variables */
  PetscInt          i,j;

  PetscFunctionBeginUser;
  /* Set values for independent variables and parameters */
  CHKERRQ(VecGetArrayRead(X,&x));
  x_a[0].setValue(x[0]);
  x_a[1].setValue(x[1]);
  mu_a.setValue(user->mu);
  CHKERRQ(VecRestoreArrayRead(X,&x));

  /* Set seed matrix as 3x3 identity matrix */
  x_a[0].setADValue(0,1.);x_a[0].setADValue(1,0.);x_a[0].setADValue(2,0.);
  x_a[1].setADValue(0,0.);x_a[1].setADValue(1,1.);x_a[1].setADValue(2,0.);
  mu_a.setADValue(0,0.);mu_a.setADValue(1,0.);mu_a.setADValue(2,1.);

  /* Evaluate residual (on active variables) */
  CHKERRQ(EvaluateResidual(x_a,mu_a,f_a));

  /* Extract derivatives */
  CHKERRQ(PetscMalloc1(user->adctx->n,&J));
  J[0] = (PetscScalar*) f_a[0].getADValue();
  J[1] = (PetscScalar*) f_a[1].getADValue();

  /* Set matrix values */
  for (i=0; i<user->adctx->m; i++) {
    for (j=0; j<user->adctx->n; j++) {
      CHKERRQ(MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
    }
  }
  CHKERRQ(PetscFree(J));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*
  Compute the Jacobian w.r.t. mu using tapeless mode of ADOL-C.
*/
static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  User           user = (User)ctx;
  PetscScalar    **J;
  PetscScalar    *x;
  adouble        f_a[2];      /* 'active' double for dependent variables */
  adouble        x_a[2],mu_a; /* 'active' doubles for independent variables */
  PetscInt       i,j = 0;

  PetscFunctionBeginUser;

  /* Set values for independent variables and parameters */
  CHKERRQ(VecGetArray(X,&x));
  x_a[0].setValue(x[0]);
  x_a[1].setValue(x[1]);
  mu_a.setValue(user->mu);
  CHKERRQ(VecRestoreArray(X,&x));

  /* Set seed matrix as 3x3 identity matrix */
  x_a[0].setADValue(0,1.);x_a[0].setADValue(1,0.);x_a[0].setADValue(2,0.);
  x_a[1].setADValue(0,0.);x_a[1].setADValue(1,1.);x_a[1].setADValue(2,0.);
  mu_a.setADValue(0,0.);mu_a.setADValue(1,0.);mu_a.setADValue(2,1.);

  /* Evaluate residual (on active variables) */
  CHKERRQ(EvaluateResidual(x_a,mu_a,f_a));

  /* Extract derivatives */
  CHKERRQ(PetscMalloc1(2,&J));
  J[0] = (PetscScalar*) f_a[0].getADValue();
  J[1] = (PetscScalar*) f_a[1].getADValue();

  /* Set matrix values */
  for (i=0; i<user->adctx->m; i++) {
    CHKERRQ(MatSetValues(A,1,&i,1,&j,&J[i][user->adctx->n],INSERT_VALUES));
  }
  CHKERRQ(PetscFree(J));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
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
  Vec            lambda[2],mu[2];

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
  adctx->m = 2;adctx->n = 2;adctx->p = 2;
  user.adctx = adctx;
  adtl::setNumDir(adctx->n+1); /* #indep. variables, plus parameters */

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
    output_file: output/ex16adj_tl_1.out

  test:
    suffix: 2
    args: -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -mu 5
    output_file: output/ex16adj_tl_2.out

  test:
    suffix: 3
    args: -ts_max_steps 10 -monitor
    output_file: output/ex16adj_tl_3.out

  test:
    suffix: 4
    args: -ts_max_steps 10 -monitor -mu 5
    output_file: output/ex16adj_tl_4.out

TEST*/
