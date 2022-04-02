
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/* ------------------------------------------------------------------------

   This program solves the van der Pol DAE ODE equivalent
       y' = z                 (1)
       z' = \mu ((1-y^2)z-y)
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = - 2/3 +10/(81*\mu) - 292/(2187*\mu^2),
   and
       \mu = 10^6 ( y'(0) ~ -0.6666665432100101).
   This is a nonlinear equation. The well prepared initial condition gives errors that are not dominated by the first few steps of the method when \mu is large.

   Notes:
   This code demonstrates the TS solver interface to an ODE -- RHSFunction for explicit form and IFunction for implicit form.

  ------------------------------------------------------------------------- */

#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
};

/*
   User-defined routines
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
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

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArrayRead(Xdot,&xdot));
  PetscCall(VecGetArray(F,&f));
  f[0] = xdot[0] - x[1];
  f[1] = xdot[1] - user->mu*((1.0-x[0]*x[0])*x[1] - x[0]);
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArrayRead(Xdot,&xdot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  const PetscScalar *x;
  PetscScalar       J[2][2];

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  J[0][0] = a;     J[0][1] = -1.0;
  J[1][0] = user->mu*(2.0*x[0]*x[1] + 1.0);   J[1][1] = a - user->mu*(1.0-x[0]*x[0]);
  PetscCall(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X,&x));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  const PetscScalar *x;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedX;

  PetscFunctionBeginUser;
  PetscCall(TSGetTimeStep(ts,&dt));
  PetscCall(TSGetMaxTime(ts,&tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    PetscCall(VecDuplicate(X,&interpolatedX));
    PetscCall(TSInterpolate(ts,user->next_output,interpolatedX));
    PetscCall(VecGetArrayRead(interpolatedX,&x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",
                          user->next_output,step,t,dt,(double)PetscRealPart(x[0]),
                          (double)PetscRealPart(x[1])));
    PetscCall(VecRestoreArrayRead(interpolatedX,&x));
    PetscCall(VecDestroy(&interpolatedX));
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            x;             /* solution, residual vectors */
  Mat            A;             /* Jacobian matrix */
  PetscInt       steps;
  PetscReal      ftime = 0.5;
  PetscBool      monitor = PETSC_FALSE,implicitform = PETSC_TRUE;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e3;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Physical parameters",NULL);
  PetscCall(PetscOptionsReal("-mu","Stiffness parameter","<1.0e6>",user.mu,&user.mu,NULL));
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A,&x,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  if (implicitform) {
    PetscCall(TSSetIFunction(ts,NULL,IFunction,&user));
    PetscCall(TSSetIJacobian(ts,A,A,IJacobian,&user));
    PetscCall(TSSetType(ts,TSBEULER));
  } else {
    PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,&user));
    PetscCall(TSSetType(ts,TSRK));
  }
  PetscCall(TSSetMaxTime(ts,ftime));
  PetscCall(TSSetTimeStep(ts,0.001));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  if (monitor) {
    PetscCall(TSMonitorSet(ts,Monitor,&user,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(x,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user.mu) - 292.0/(2187.0*user.mu*user.mu);
  PetscCall(VecRestoreArray(x,&x_ptr));

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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"steps %D, ftime %g\n",steps,(double)ftime));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
  return(0);
}

/*TEST

    test:
      requires: !single
      args: -mu 1e6

    test:
      requires: !single
      suffix: 2
      args: -implicitform false -ts_type rk -ts_rk_type 5dp -ts_adapt_type dsp

    test:
      requires: !single
      suffix: 3
      args: -implicitform false -ts_type rk -ts_rk_type 5dp -ts_adapt_type dsp -ts_adapt_dsp_filter H0312

TEST*/
