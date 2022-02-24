#define c11 1.0
#define c12 0
#define c21 2.0
#define c22 1.0
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/*
   Concepts: TS^forward sensitivity analysis for time-dependent nonlinear problems
   Concepts: TS^van der Pol equation DAE equivalent
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This code demonstrates how to compute trajectory sensitivties w.r.t. the stiffness parameter mu.
   1) Use two vectors s and sp for sensitivities w.r.t. initial values and paraeters respectively. This case requires the original Jacobian matrix and a JacobianP matrix for the only parameter mu.
   2) Consider the initial values to be parameters as well. Then there are three parameters in total. The JacobianP matrix will be combined matrix of the Jacobian matrix and JacobianP matrix in the previous case. This choice can be selected by using command line option '-combined'

  ------------------------------------------------------------------------- */
#include <petscts.h>
#include <petsctao.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
  PetscBool combined;
  /* Sensitivity analysis support */
  PetscInt  steps;
  PetscReal ftime;
  Mat       Jac;                    /* Jacobian matrix */
  Mat       Jacp;                   /* JacobianP matrix */
  Vec       x;
  Mat       sp;                     /* forward sensitivity variables */
};

/*
   User-defined routines
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = xdot[0] - x[1];
  f[1] = c21*(xdot[0]-x[1]) + xdot[1] - user->mu*((1.0-x[0]*x[0])*x[1] - x[0]) ;
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = c21*a + user->mu*(1.0 + 2.0*x[0]*x[1]);   J[1][1] = -c21 + a - user->mu*(1.0-x[0]*x[0]);
  CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(X,&x));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  User              user = (User)ctx;
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  if (user->combined) col[0] = 2;
  CHKERRQ(VecGetArrayRead(X,&x));
  J[0][0] = 0;
  J[1][0] = (1.-x[0]*x[0])*x[1]-x[0];
  CHKERRQ(MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(X,&x));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedX;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetTimeStep(ts,&dt));
  CHKERRQ(TSGetMaxTime(ts,&tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    CHKERRQ(VecDuplicate(X,&interpolatedX));
    CHKERRQ(TSInterpolate(ts,user->next_output,interpolatedX));
    CHKERRQ(VecGetArrayRead(interpolatedX,&x));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",
                       user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    CHKERRQ(VecRestoreArrayRead(interpolatedX,&x));
    CHKERRQ(VecDestroy(&interpolatedX));
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscInt       rows,cols;
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e6;
  user.steps       = 0;
  user.ftime       = 0.5;
  user.combined    = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-combined",&user.combined,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  rows = 2;
  cols = user.combined ? 3 : 1;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.Jac));
  CHKERRQ(MatSetSizes(user.Jac,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(user.Jac));
  CHKERRQ(MatSetUp(user.Jac));
  CHKERRQ(MatCreateVecs(user.Jac,&user.x,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSBEULER));
  CHKERRQ(TSSetIFunction(ts,NULL,IFunction,&user));
  CHKERRQ(TSSetIJacobian(ts,user.Jac,user.Jac,IJacobian,&user));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetMaxTime(ts,user.ftime));
  if (monitor) {
    CHKERRQ(TSMonitorSet(ts,Monitor,&user,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(user.x,&x_ptr));
  x_ptr[0] = 2.0;   x_ptr[1] = -0.66666654321;
  CHKERRQ(VecRestoreArray(user.x,&x_ptr));
  CHKERRQ(TSSetTimeStep(ts,1.0/1024.0));

  /* Set up forward sensitivity */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.Jacp));
  CHKERRQ(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,rows,cols));
  CHKERRQ(MatSetFromOptions(user.Jacp));
  CHKERRQ(MatSetUp(user.Jacp));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,rows,cols,NULL,&user.sp));
  if (user.combined) {
    CHKERRQ(MatZeroEntries(user.sp));
    CHKERRQ(MatShift(user.sp,1.0));
  } else {
    CHKERRQ(MatZeroEntries(user.sp));
  }
  CHKERRQ(TSForwardSetSensitivities(ts,cols,user.sp));
  CHKERRQ(TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSolve(ts,user.x));
  CHKERRQ(TSGetSolveTime(ts,&user.ftime));
  CHKERRQ(TSGetStepNumber(ts,&user.steps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)user.ftime));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n ode solution \n"));
  CHKERRQ(VecView(user.x,PETSC_VIEWER_STDOUT_WORLD));

  if (user.combined) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n forward sensitivity: d[y(tf) z(tf)]/d[y0 z0 mu]\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n forward sensitivity: d[y(tf) z(tf)]/d[mu]\n"));
  }
  CHKERRQ(MatView(user.sp,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&user.Jac));
  CHKERRQ(MatDestroy(&user.sp));
  CHKERRQ(MatDestroy(&user.Jacp));
  CHKERRQ(VecDestroy(&user.x));
  CHKERRQ(TSDestroy(&ts));

  ierr = PetscFinalize();
  return(ierr);
}

/*TEST

    test:
      args: -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -combined
      requires:  !complex !single

    test:
      suffix: 2
      requires: !complex !single
      args: -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5

TEST*/
