static char help[] = "Solves an ODE-constrained optimization problem -- finding the optimal initial conditions for the van der Pol equation.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Concepts: Optimization using adjoint sensitivities
   Processors: 1
*/
/* ------------------------------------------------------------------------

   Notes:
   This code demonstrates how to solve an ODE-constrained optimization problem with TAO, TSAdjoint and TS.
   The objective is to minimize the difference between observation and model prediction by finding optimal values for initial conditions.
   The gradient is computed with the discrete adjoint of an explicit Runge-Kutta method, see ex16adj.c for details.
  ------------------------------------------------------------------------- */
#include <petsctao.h>
#include <petscts.h>
#include <petscmat.h>
typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;

  PetscInt  steps;
  PetscReal ftime,x_ob[2];
  Mat       A;             /* Jacobian matrix */
  Vec       x,lambda[2];   /* adjoint variables */
};

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

/*
*  User-defined routines
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = x[1];
  f[1] = user->mu*(1.-x[0]*x[0])*x[1]-x[0];
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscReal         mu   = user->mu;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = 0;
  J[1][0] = -2.*mu*x[1]*x[0]-1;
  J[0][1] = 1.0;
  J[1][1] = mu*(1.0-x[0]*x[0]);
  ierr    = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt, tprev;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",(double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"t %.6f (tprev = %.6f) \n",(double)t,(double)tprev);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS                 ts;          /* nonlinear solver */
  Vec                ic;
  PetscBool          monitor = PETSC_FALSE;
  PetscScalar        *x_ptr;
  PetscMPIInt        size;
  struct _n_User     user;
  PetscErrorCode     ierr;
  Tao                tao;
  KSP                ksp;
  PC                 pc;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.mu          = 1.0;
  user.next_output = 0.0;
  user.steps       = 0;
  user.ftime       = 0.5;

  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.x,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)(user.ftime));CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&(user.ftime));CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&user.steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)user.ftime);CHKERRQ(ierr);

  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  user.x_ob[0] = x_ptr[0];
  user.x_ob[1] = x_ptr[1];

  ierr = MatCreateVecs(user.A,&user.lambda[0],NULL);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

  /* Set initial solution guess */
  ierr = MatCreateVecs(user.A,&ic,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(ic,&x_ptr);CHKERRQ(ierr);
  x_ptr[0]  = 2.1;
  x_ptr[1]  = 0.7;
  ierr = VecRestoreArray(ic,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,ic);CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  ierr = TaoSetTolerances(tao,1e-10,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[0]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = VecDestroy(&ic);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  User              user = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr,*y_ptr;
  PetscErrorCode    ierr;
  PetscScalar       *ic_ptr;

  ierr = VecCopy(IC,user->x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSetTolerances(ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,user->x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&user->ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&user->steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %.6f, steps %D, ftime %g\n",(double)user->mu,user->steps,(double)user->ftime);CHKERRQ(ierr);

  ierr = VecGetArray(IC,&ic_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(user->x,&x_ptr);CHKERRQ(ierr);
  *f   = (x_ptr[0]-user->x_ob[0])*(x_ptr[0]-user->x_ob[0])+(x_ptr[1]-user->x_ob[1])*(x_ptr[1]-user->x_ob[1]);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Observed value y_ob=[%f; %f], ODE solution y=[%f;%f], Cost function f=%f\n",(double)user->x_ob[0],(double)user->x_ob[1],(double)x_ptr[0],(double)x_ptr[1],(double)(*f));CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*   Redet initial conditions for the adjoint integration */
  ierr = VecGetArray(user->lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*(x_ptr[0]-user->x_ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user->x_ob[1]);
  ierr = VecRestoreArray(user->lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,user->lambda,NULL);CHKERRQ(ierr);

  /*   Set RHS Jacobian  for the adjoint integration */
  ierr = TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = VecCopy(user->lambda[0],G);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
