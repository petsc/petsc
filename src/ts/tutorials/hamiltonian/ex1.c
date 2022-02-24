
static char help[] = "Solves the motion of spring.\n\
Input parameters include:\n";

/*
   Concepts: TS^Separable Hamiltonian problems
   Concepts: TS^Symplectic intergrators
   Processors: 1
*/
/* ------------------------------------------------------------------------

  This program solves the motion of spring by Hooke's law
  x' = f(t,v) = v
  v' = g(t,x) = -omega^2*x
  on the interval 0 <= t <= 0.1, with the initial conditions
    x(0) = 0.2, x'(0) = v(0) = 0,
  and
    omega = 64.
  The exact solution is
    x(t) = A*sin(t*omega) + B*cos(t*omega)
  where A and B are constants that can be determined from the initial conditions.
  In this case, B=0.2, A=0.

  Notes:
  This code demonstrates the TS solver interface to solve a separable Hamiltonian
  system, which can be split into two subsystems involving two coupling components,
  named generailized momentum and generailized position respectively.
  Using a symplectic intergrator can preserve energy
  E = (v^2+omega^2*x^2-omega^2*h*v*x)/2
  ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscvec.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal omega;
  PetscInt  nts; /* print the energy at each nts time steps */
};

/*
  User-defined routines.
  The first RHS function provides f(t,x), the residual for the generalized momentum,
  and the second one provides g(t,v), the residual for the generalized position.
*/
static PetscErrorCode RHSFunction2(TS ts,PetscReal t,Vec X,Vec Vres,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x;
  PetscScalar       *vres;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(Vres,&vres));
  vres[0] = -user->omega*user->omega*x[0];
  CHKERRQ(VecRestoreArray(Vres,&vres));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction1(TS ts,PetscReal t,Vec V,Vec Xres,void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArray(Xres,&xres));
  CHKERRQ(VecGetArrayRead(V,&v));
  xres[0] = v[0];
  CHKERRQ(VecRestoreArrayRead(V,&v));
  CHKERRQ(VecRestoreArray(Xres,&xres));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec R,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *u;
  PetscScalar       *r;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(R,&r));
  r[0] = u[1];
  r[1] = -user->omega*user->omega*u[0];
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(R,&r));
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  const PetscScalar *u;
  PetscReal         dt;
  PetscScalar       energy,menergy;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  if (step%user->nts == 0) {
    CHKERRQ(TSGetTimeStep(ts,&dt));
    CHKERRQ(VecGetArrayRead(U,&u));
    menergy = (u[1]*u[1]+user->omega*user->omega*u[0]*u[0]-user->omega*user->omega*dt*u[0]*u[1])/2.;
    energy = (u[1]*u[1]+user->omega*user->omega*u[0]*u[0])/2.;
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"At time %.6lf, Energy = %8g, Modified Energy = %8g\n",t,(double)energy,(double)menergy));
    CHKERRQ(VecRestoreArrayRead(U,&u));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            U;             /* solution, residual vectors */
  IS             is1,is2;
  PetscInt       nindices[1];
  PetscReal      ftime   = 0.1;
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *u_ptr;
  PetscMPIInt    size;
  struct _n_User user;
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
  user.omega = 64.;
  user.nts = 100;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Physical parameters",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsReal("-omega","parameter","<64>",user.omega,&user.omega,PETSC_NULL));
  CHKERRQ(PetscOptionsInt("-next_output","time steps for next output point","<100>",user.nts,&user.nts,PETSC_NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,2,&U));
  nindices[0] = 0;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,1,nindices,PETSC_COPY_VALUES,&is1));
  nindices[0] = 1;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,1,nindices,PETSC_COPY_VALUES,&is2));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSBASICSYMPLECTIC));
  CHKERRQ(TSRHSSplitSetIS(ts,"position",is1));
  CHKERRQ(TSRHSSplitSetIS(ts,"momentum",is2));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user));
  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,&user));

  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetTimeStep(ts,0.0001));
  CHKERRQ(TSSetMaxSteps(ts,1000));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  if (monitor) {
    CHKERRQ(TSMonitorSet(ts,Monitor,&user,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(U,&u_ptr));
  u_ptr[0] = 0.2;
  u_ptr[1] = 0.0;
  CHKERRQ(VecRestoreArray(U,&u_ptr));

  CHKERRQ(TSSetTime(ts,0.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(VecView(U,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The exact solution at time %.6lf is [%g %g]\n",(double)ftime,(double)0.2*PetscCosReal(user.omega*ftime),(double)-0.2*user.omega*PetscSinReal(user.omega*ftime)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   build:
     requires: !single !complex

   test:
     args: -ts_basicsymplectic_type 1 -monitor

   test:
     suffix: 2
     args: -ts_basicsymplectic_type 2 -monitor

TEST*/
