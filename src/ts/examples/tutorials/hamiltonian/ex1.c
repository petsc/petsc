
static char help[] = "Solves the motion of spring.\n\
Input parameters include:\n";

/*
   Concepts: TS^Separable Hamiltonian problems
   Concepts: TS^Symplectic intergrators
   Processors: 1
*/
/* ------------------------------------------------------------------------

  This program solves the motion of spring by Hooke's law
  v' = f(t,x) = -omega^2*x
  x' = g(t,v) = v
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
static PetscErrorCode RHSFunction1(TS ts,PetscReal t,Vec X,Vec Vres,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Vres,&vres);CHKERRQ(ierr);
  vres[0] = -user->omega*user->omega*x[0];
  ierr = VecRestoreArray(Vres,&vres);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction2(TS ts,PetscReal t,Vec V,Vec Xres,void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Xres,&xres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  xres[0] = v[0];
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xres,&xres);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec R,void *ctx)
{
  User              user = (User)ctx;
  Vec               V,X,Rv,Rx;
  const PetscScalar *v,*x;
  PetscScalar       *rv,*rx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecNestGetSubVec(U,0,&V);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(U,1,&X);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(R,0,&Rv);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(R,1,&Rx);CHKERRQ(ierr);

  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Rv,&rv);CHKERRQ(ierr);
  ierr = VecGetArray(Rx,&rx);CHKERRQ(ierr);
  rv[0] = -user->omega*user->omega*x[0];
  rx[0] = v[0];
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Rv,&rv);CHKERRQ(ierr);
  ierr = VecRestoreArray(Rx,&rx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  PetscErrorCode    ierr;
  Vec               X,V;
  const PetscScalar *x,*v;
  PetscReal         energy, dt;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  if (step%user->nts == 0) {
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = VecNestGetSubVec(U,0,&V);CHKERRQ(ierr);
    ierr = VecNestGetSubVec(U,1,&X);CHKERRQ(ierr);
    ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    energy = (v[0]*v[0]+user->omega*user->omega*x[0]*x[0]-user->omega*user->omega*dt*v[0]*x[0])/2.;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Energy = %6e at time  %6f\n",(double)energy,t);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            U,X,V;         /* solution, residual vectors */
  Vec            hx[2];
  PetscReal      ftime   = 0.1;
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr,*v_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.omega = 64.;
  user.nts = 100;
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Physical parameters",NULL);
  ierr = PetscOptionsReal("-omega","parameter","<64>",user.omega,&user.omega,PETSC_NULL);
  ierr = PetscOptionsInt("-next_output","time steps for next output point","<100>",user.nts,&user.nts,PETSC_NULL);
  ierr = PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&V);CHKERRQ(ierr);
  hx[0] = V;
  hx[1] = X;
  ierr = VecCreateNest(PETSC_COMM_WORLD,2,NULL,hx,&U);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(U);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBSI);CHKERRQ(ierr);
  ierr = TSSetRHSFunctionSplit2w(ts,NULL,RHSFunction1,RHSFunction2,&user);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);

  ierr = TSSetDuration(ts,PETSC_DEFAULT,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(X,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.2;
  ierr = VecRestoreArray(X,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(V,&v_ptr);CHKERRQ(ierr);
  v_ptr[0] = 0.0;
  ierr = VecRestoreArray(V,&v_ptr);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);

  ierr = TSSetInitialTimeStep(ts,0.0,.0001);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = VecView(U,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"The exact solution is [%6g %6g]\n",(double)-0.2*user.omega*PetscSinReal(user.omega*ftime),(double)0.2*PetscCosReal(user.omega*ftime));CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
