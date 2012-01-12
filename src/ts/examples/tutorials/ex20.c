
/* Program usage:  ./ex16 [-help] [all PETSc options] */

static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation DAE equivalent
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This program solves the van der Pol DAE ODE equivalent
       y' = -z            (1)
       z' = -z/(z^2 - 1)
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = -2, y'(0) = −2.355301397608119909925287735864250951918,
   This is a nonlinear equation.

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely turning (1) into a system of
   first order differential equations,

   [ y' ] = [          -z          ]
   [ z' ]   [     -z/(z^2 - 1)     ]

   which then we can write as a vector equation

   [ u_1' ] = [      -u_2       ]  (2)
   [ u_2' ]   [ -u_2/(u_2^2 - 1 ]

   which is now in the desired form of u_t = f(u,t). One way that we
   can split f(u,t) in (2) is to split by component,

   [ u_1' ] = [ -u_2 ] + [       0         ]
   [ u_2' ]   [  0   ]   [ -u_2/(u_2^2 - 1 ]

   where

   [ F(u,t) ] = [ -u_2 ]
                [  0  ]

   and

   [ G(u',u,t) ] = [ u_1' ] - [            0              ]
                   [ u_2' ]   [ -u_2/(u_2^2 - 1 ]

   Using the definition of the Jacobian of G (from the PETSc user manual),
   in the equation G(u',u,t) = F(u,t),

              dG   dG
   J(G) = a * -- - --
              du'  du

   where d is the partial derivative. In this example,

   dG   [ 1 ; 0 ]
   -- = [       ]
   du'  [ 0 ; 1 ]

   dG   [ 0 ;                   0                  ]
   -- = [                                          ]
   du   [ 0 ; 1/(u_2^2 - 1) - 2*u_2^2/(u_2^2-1)^2  ]

   Hence,

          [      a       ;               0                ]
   J(G) = [                                               ]
          [ 0 ; a - 1/(u_2^2 - 1) + 2*u_2^2/(u_2^2 - 1)^2 ]

Notes:
     RHSFunction has an imex split with -u_2/(u_2^2-1) as part of Ifunc (as
     described above, and RHSFunction2 has the split with that as part of 
     the RHS, choosing RHS or RHS2 also chooses the corresponding IJacobian
     or IJacobian2.

  ------------------------------------------------------------------------- */

#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscBool imex;
  PetscReal next_output;
};

/*
*  User-defined routines
*/
#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User user = (User)ctx;
  PetscScalar *x,*f;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = (user->imex ? -1.*x[1] : 0.0);
  f[1] = 0.0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction2"
static PetscErrorCode RHSFunction2(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User user = (User)ctx;
  PetscScalar *x,*f;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = (user->imex ? -1.*x[1] : 0.0);
  f[1] = (user->imex ? -1.*x[1]/(x[1]*x[1]-1) : 0.0);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IFunction"
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User user = (User)ctx;
  PetscScalar *x,*xdot,*f;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] + (user->imex ? 0 : x[1]);
  f[1] = xdot[1] + x[1]/(x[1]*x[1]-1);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IFunction2"
static PetscErrorCode IFunction2(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User user = (User)ctx;
  PetscScalar *x,*xdot,*f;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] + (user->imex ? 0 : x[1]);
  f[1] = xdot[1] + (user->imex ? 0 : x[1]/(x[1]*x[1]-1));
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IJacobian"
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  User user = (User)ctx;
  PetscInt rowcol[] = {0,1};
  PetscScalar *x,J[2][2];

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  J[0][0] = a;     J[0][1] = (user->imex ? 0 : -1.);
  J[1][0] = 0.0;   J[1][1] = a - 1./(x[1]*x[1]-1)+x[1]*x[1]/(x[1]*x[1]-1)/(x[1]*x[1]-1)*2;
  ierr = MatSetValues(*B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IJacobian2"
static PetscErrorCode IJacobian2(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  User user = (User)ctx;
  PetscInt rowcol[] = {0,1};
  PetscScalar *x,J[2][2];

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  J[0][0] = a;    J[0][1] = (user->imex ? 0 : -1.);
  J[1][0] = 0.0;  J[1][1] = a - (user->imex ? 0 : 1./(x[1]*x[1]-1) + 2*x[1]*x[1]/(x[1]*x[1]-1)/(x[1]*x[1]-1));
  ierr = MatSetValues(*B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterMyARK2"
static PetscErrorCode RegisterMyARK2(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    const PetscReal
      A[3][3] = {{0,0,0},
                 {0.41421356237309504880,0,0},
                 {0.75,0.25,0}},
      At[3][3] = {{0,0,0},
                  {0.12132034355964257320,0.29289321881345247560,0},
                  {0.20710678118654752440,0.50000000000000000000,0.29289321881345247560}};
      ierr = TSARKIMEXRegister("myark2",2,3,&At[0][0],PETSC_NULL,PETSC_NULL,&A[0][0],PETSC_NULL,PETSC_NULL,0,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "Monitor"
/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode ierr;
  const PetscScalar *x;
  PetscReal tfinal, dt;
  User user = (User)ctx;
  Vec interpolatedX;

  PetscFunctionBegin;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetDuration(ts,PETSC_NULL,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",user->next_output,step,t,dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS              ts;           /* nonlinear solver */
  Vec             x;            /* solution, residual vectors */
  Mat             A;            /* Jacobian matrix */
  PetscInt        steps;
  PetscReal       ftime=0.5;
  PetscBool       monitor = PETSC_FALSE,rhs2 = PETSC_FALSE;
  PetscScalar     *x_ptr;
  PetscMPIInt     size;
  struct _n_User  user;
  PetscErrorCode  ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  ierr = RegisterMyARK2();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.imex = PETSC_TRUE;
  user.next_output = 0.0;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-imex",&user.imex,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-monitor",&monitor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-rhs2",&rhs2,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatGetVecs(A,&x,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  if(rhs2 == PETSC_FALSE){
    ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunction,&user);CHKERRQ(ierr);
    ierr = TSSetIFunction(ts,PETSC_NULL,IFunction,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,A,A,IJacobian,&user);CHKERRQ(ierr);
  }else{
    ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunction2,&user);CHKERRQ(ierr);
    ierr = TSSetIFunction(ts,PETSC_NULL,IFunction2,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,A,A,IJacobian2,&user);CHKERRQ(ierr);
  }

  ierr = TSSetDuration(ts,PETSC_DEFAULT,ftime);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,PETSC_NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = -2;   x_ptr[1] = -2.355301397608119909925287735864250951918;
  ierr = VecRestoreArray(x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"steps %D, ftime %G\n",steps,ftime);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
