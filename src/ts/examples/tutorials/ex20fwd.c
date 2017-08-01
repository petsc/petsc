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

   This code demonstrates two ways of computing trajectory sensitivties w.r.t. intial values and the stiffness parameter mu.
   1) Use two vectors s and sp for sensitivities w.r.t. intial values and paraeters respectively. This case requires the original Jacobian matrix and a JacobianP matrix for the only parameter mu.
   2) Consider the intial values to be parameters as well. Then there are three parameters in total. The JacobianP matrix will be combined matrix of the Jacobian matrix and JacobianP matrix in the previous case. This choice can be selected by using command line option '-combined'

  ------------------------------------------------------------------------- */
#include <petscts.h>
#include <petsctao.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;

  /* Sensitivity analysis support */
  PetscInt  steps;
  PetscReal ftime;
  Mat       A;                      /* Jacobian matrix */
  Vec       jacp_combined[3];       /* JacobianP matrix (p includes initial values)*/
  Vec       jacp[1];                /* JacobianP matrix */
  Vec       x;
  Vec       s_combined[3];          /* forward sensitivity variables */
  Vec       s[2],sp[1];
};

/*
*  User-defined routines
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] - x[1];
  f[1] = c21*(xdot[0]-x[1]) + xdot[1] - user->mu*((1.0-x[0]*x[0])*x[1] - x[0]) ;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = c21*a + user->mu*(1.0 + 2.0*x[0]*x[1]);   J[1][1] = -c21 + a - user->mu*(1.0-x[0]*x[0]);

  ierr    = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP_Combined(TS ts,PetscReal t,Vec X,Vec *J,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *j;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecZeroEntries(J[0]);CHKERRQ(ierr);
  ierr = VecZeroEntries(J[1]);CHKERRQ(ierr);

  ierr = VecGetArray(J[2],&j);CHKERRQ(ierr);
  j[0] = 0;
  j[1] = (1.-x[0]*x[0])*x[1]-x[0];
  ierr = VecRestoreArray(J[2],&j);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Vec *J,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *j;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(J[0],&j);CHKERRQ(ierr);
  j[0] = 0;
  j[1] = (1.-x[0]*x[0])*x[1]-x[0];
  ierr = VecRestoreArray(J[0],&j);CHKERRQ(ierr);
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
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",
                       user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  PetscBool      monitor = PETSC_FALSE,combined = PETSC_FALSE;
  PetscScalar    *x_ptr,*y_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e6;
  user.steps       = 0;
  user.ftime       = 0.5;
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-combined",&combined,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.x,NULL);CHKERRQ(ierr);

  if (combined) {
    ierr = MatCreateVecs(user.A,&user.jacp_combined[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(user.A,&user.jacp_combined[1],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(user.A,&user.jacp_combined[2],NULL);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(user.A,&user.jacp[0],NULL);CHKERRQ(ierr);
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user.A,user.A,IJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1.0/1024.0);CHKERRQ(ierr);

  if (combined) {
    ierr = MatCreateVecs(user.A,&user.s_combined[0],NULL);CHKERRQ(ierr);
    /*   Set initial conditions for the adjoint integration */
    ierr = VecGetArray(user.s_combined[0],&y_ptr);CHKERRQ(ierr);
    y_ptr[0] = 1.0; y_ptr[1] = 0.0;
    ierr = VecRestoreArray(user.s_combined[0],&y_ptr);CHKERRQ(ierr);
    ierr = MatCreateVecs(user.A,&user.s_combined[1],NULL);CHKERRQ(ierr);
    ierr = VecGetArray(user.s_combined[1],&y_ptr);CHKERRQ(ierr);
    y_ptr[0] = 0.0; y_ptr[1] = 1.0;
    ierr = VecRestoreArray(user.s_combined[1],&y_ptr);CHKERRQ(ierr);
    ierr = MatCreateVecs(user.A,&user.s_combined[2],NULL);CHKERRQ(ierr);
    ierr = VecZeroEntries(user.s_combined[2]);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(user.A,&user.s[0],NULL);CHKERRQ(ierr);
    /*   Set initial conditions for the adjoint integration */
    ierr = VecGetArray(user.s[0],&y_ptr);CHKERRQ(ierr);
    y_ptr[0] = 1.0; y_ptr[1] = 0.0;
    ierr = VecRestoreArray(user.s[0],&y_ptr);CHKERRQ(ierr);
    ierr = MatCreateVecs(user.A,&user.s[1],NULL);CHKERRQ(ierr);
    ierr = VecGetArray(user.s[1],&y_ptr);CHKERRQ(ierr);
    y_ptr[0] = 0.0; y_ptr[1] = 1.0;
    ierr = VecRestoreArray(user.s[1],&y_ptr);CHKERRQ(ierr);
    ierr = MatCreateVecs(user.A,&user.sp[0],NULL);CHKERRQ(ierr);
    ierr = VecZeroEntries(user.sp[0]);CHKERRQ(ierr);
  }

  if (combined) {
    ierr = TSForwardSetSensitivities(ts,3,user.s_combined,0,NULL);CHKERRQ(ierr);
    /*   Set RHS JacobianP */
    ierr = TSForwardSetRHSJacobianP(ts,user.jacp_combined,RHSJacobianP_Combined,&user);CHKERRQ(ierr);
  } else {
    ierr = TSForwardSetSensitivities(ts,1,user.sp,2,user.s);CHKERRQ(ierr);
    /*   Set RHS JacobianP */
    ierr = TSForwardSetRHSJacobianP(ts,user.jacp,RHSJacobianP,&user);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&user.ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&user.steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)user.ftime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n ode solution \n");CHKERRQ(ierr);
  ierr = VecView(user.x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[y(tf)]/d[y0]  d[z(tf)]/d[y0]\n");CHKERRQ(ierr);
  if (combined) {
    ierr = VecView(user.s_combined[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  } else {
    ierr = VecView(user.s[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[y(tf)]/d[z0]  d[z(tf)]/d[z0]\n");CHKERRQ(ierr);
  if (combined) {
    ierr = VecView(user.s_combined[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }else {
    ierr = VecView(user.s[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameters: d[y(tf)]/d[mu] d[z(tf)]/d[mu]\n");CHKERRQ(ierr);
  if (combined) {
    ierr = VecView(user.s_combined[2],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  } else {
    ierr = VecView(user.sp[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  if (combined) {
    ierr = VecDestroy(&user.jacp_combined[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.jacp_combined[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.jacp_combined[2]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.s_combined[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.s_combined[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.s_combined[2]);CHKERRQ(ierr);
  } else {
    ierr = VecDestroy(&user.jacp[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.s[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.s[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&user.sp[0]);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}
