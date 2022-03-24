
static char help[] = "Solves the van der Pol equation and demonstrate IMEX.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This program solves the van der Pol equation
       y'' - \mu ((1-y^2)*y' - y) = 0        (1)
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = - 2/3 +10/(81*\mu) - 292/(2187*\mu^2),
   This is a nonlinear equation. The well prepared initial condition gives errors that are not dominated by the first few steps of the method when \mu is large.

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely turning (1) into a system of
   first order differential equations,

   [ y' ] = [          z            ]
   [ z' ]   [ \mu ((1 - y^2) z - y) ]

   which then we can write as a vector equation

   [ u_1' ] = [             u_2           ]  (2)
   [ u_2' ]   [ \mu (1 - u_1^2) u_2 - u_1 ]

   which is now in the desired form of u_t = f(u,t). One way that we
   can split f(u,t) in (2) is to split by component,

   [ u_1' ] = [ u_2 ] + [            0                ]
   [ u_2' ]   [  0  ]   [ \mu ((1 - u_1^2) u_2 - u_1) ]

   where

   [ G(u,t) ] = [ u_2 ]
                [  0  ]

   and

   [ F(u',u,t) ] = [ u_1' ] - [            0                ]
                   [ u_2' ]   [ \mu ((1 - u_1^2) u_2 - u_1) ]

   Using the definition of the Jacobian of F (from the PETSc user manual),
   in the equation F(u',u,t) = G(u,t),

              dF   dF
   J(F) = a * -- - --
              du'  du

   where d is the partial derivative. In this example,

   dF   [ 1 ; 0 ]
   -- = [       ]
   du'  [ 0 ; 1 ]

   dF   [       0             ;         0        ]
   -- = [                                        ]
   du   [ -\mu (2*u_1*u_2 + 1);  \mu (1 - u_1^2) ]

   Hence,

          [      a             ;          0          ]
   J(F) = [                                          ]
          [ \mu (2*u_1*u_2 + 1); a - \mu (1 - u_1^2) ]

  ------------------------------------------------------------------------- */

#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscBool imex;
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
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = (user->imex ? x[1] : 0);
  f[1] = 0.0;
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = xdot[0] + (user->imex ? 0 : x[1]);
  f[1] = xdot[1] - user->mu*((1. - x[0]*x[0])*x[1] - x[0]);
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user     = (User)ctx;
  PetscReal         mu       = user->mu;
  PetscInt          rowcol[] = {0,1};
  const PetscScalar *x;
  PetscScalar       J[2][2];

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  J[0][0] = a;                    J[0][1] = (user->imex ? 0 : 1.);
  J[1][0] = mu*(2.*x[0]*x[1]+1.);   J[1][1] = a - mu*(1. - x[0]*x[0]);
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

static PetscErrorCode RegisterMyARK2(void)
{
  PetscFunctionBeginUser;
  {
    const PetscReal
      A[3][3] = {{0,0,0},
                 {0.41421356237309504880,0,0},
                 {0.75,0.25,0}},
      At[3][3] = {{0,0,0},
                  {0.12132034355964257320,0.29289321881345247560,0},
                  {0.20710678118654752440,0.50000000000000000000,0.29289321881345247560}},
      *bembedt = NULL,*bembed = NULL;
    CHKERRQ(TSARKIMEXRegister("myark2",2,3,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembed,0,NULL,NULL));
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
  CHKERRQ(TSGetTimeStep(ts,&dt));
  CHKERRQ(TSGetMaxTime(ts,&tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    CHKERRQ(VecDuplicate(X,&interpolatedX));
    CHKERRQ(TSInterpolate(ts,user->next_output,interpolatedX));
    CHKERRQ(VecGetArrayRead(interpolatedX,&x));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",user->next_output,step,t,dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1])));
    CHKERRQ(VecRestoreArrayRead(interpolatedX,&x));
    CHKERRQ(VecDestroy(&interpolatedX));

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
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  CHKERRQ(RegisterMyARK2());

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.mu          = 1000.0;
  user.imex        = PETSC_TRUE;
  user.next_output = 0.0;

  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-imex",&user.imex,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatCreateVecs(A,&x,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSBEULER));
  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,&user));
  CHKERRQ(TSSetIFunction(ts,NULL,IFunction,&user));
  CHKERRQ(TSSetIJacobian(ts,A,A,IJacobian,&user));
  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  if (monitor) {
    CHKERRQ(TSMonitorSet(ts,Monitor,&user,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(x,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user.mu) - 292.0/(2187.0*user.mu*user.mu);
  CHKERRQ(VecRestoreArray(x,&x_ptr));
  CHKERRQ(TSSetTimeStep(ts,0.01));

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
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -ts_type arkimex -ts_arkimex_type myark2 -ts_adapt_type none
      requires: !single

TEST*/
