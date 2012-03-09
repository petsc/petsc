static const char help[] = "Time-dependent advection-reaction PDE in 1d, demonstrates IMEX methods.\n";
/*
   u_t + a1*u_x = -k1*u + k2*v + s1
   v_t + a2*v_x = k1*u - k2*v + s2
   0 < x < 1;
   a1 = 1, k1 = 10^6, s1 = 0,
   a2 = 0, k2 = 2*k1, s2 = 1

   Initial conditions:
   u(x,0) = 1 + s2*x
   v(x,0) = k0/k1*u(x,0) + s1/k1

   Upstream boundary conditions:
   u(0,t) = 1-sin(12*t)^4
*/

#include <petscdmda.h>
#include <petscts.h>

typedef PetscScalar Field[2];

typedef struct _User *User;
struct _User {
  PetscReal      a[2];         /* Advection speeds */
  PetscReal      k[2];         /* Reaction coefficients */
  PetscReal      s[2];         /* Source terms */
};

static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                ts;         /* nonlinear solver */
  Vec               X;          /* solution, residual vectors */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,maxsteps,mx;
  PetscErrorCode    ierr;
  DM                da;
  PetscReal         ftime,dt;
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-11,2,2,PETSC_NULL,&da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);

  /* Initialize user application context */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Advection-reaction options",""); {
    ierr = PetscOptionsReal("-a0","Advection rate 0","",user.a[0]=1,&user.a[0],PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-a1","Advection rate 1","",user.a[1]=0,&user.a[1],PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-k0","Reaction rate 0","",user.k[0]=1e6,&user.k[0],PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-k1","Reaction rate 1","",user.k[1]=2*user.k[0],&user.k[1],PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-s0","Source 0","",user.s[0]=0,&user.s[0],PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-s1","Source 1","",user.s[1]=1,&user.s[1],PETSC_NULL);CHKERRQ(ierr);
  } ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,PETSC_NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,PETSC_NULL,FormIFunction,&user);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,MATAIJ,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,&user);CHKERRQ(ierr);

  ftime = 1.0;
  maxsteps = 10000;
  ierr = TSSetDuration(ts,maxsteps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&mx);CHKERRQ(ierr);
  dt   = .1 * PetscMax(user.a[0],user.a[1]) / mx; /* Advective CFL, I don't know why it needs so much safety factor. */
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormIFunction"
static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  User           user = (User)ptr;
  DM             da;
  DMDALocalInfo  info;
  PetscInt       i;
  Field          *x,*xdot,*f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    f[i][0] = xdot[i][0] + user->k[0]*x[i][0] - user->k[1]*x[i][1] - user->s[0];
    f[i][1] = xdot[i][1] - user->k[0]*x[i][0] + user->k[1]*x[i][1] - user->s[1];
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User           user = (User)ptr;
  DM             da;
  Vec            Xloc;
  DMDALocalInfo  info;
  PetscInt       i,j;
  PetscReal      hx;
  Field          *x,*f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)info.mx;

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    const PetscReal *a = user->a;
    PetscReal u0t[2] = {1. - PetscPowScalar(sin(12*t),4.),0};
    for (j=0; j<2; j++) {
      if (i == 0) {
        f[i][j] = a[j]/hx*(1./3*u0t[j] + 0.5*x[i][j] - x[i+1][j] + 1./6*x[i+2][j]);
      } else if (i == 1) {
        f[i][j] = a[j]/hx*(-1./12*u0t[j] + 2./3*x[i-1][j] - 2./3*x[i+1][j] + 1./12*x[i+2][j]);
      } else if (i == info.mx-2) {
        f[i][j] = a[j]/hx*(-1./6*x[i-2][j] + x[i-1][j] - 0.5*x[i][j] - 1./3*x[i+1][j]);
      } else if (i == info.mx-1) {
        f[i][j] = a[j]/hx*(-x[i][j] + x[i-1][j]);
      } else {
        f[i][j] = a[j]/hx*(-1./12*x[i-2][j] + 2./3*x[i-1][j] - 2./3*x[i+1][j] + 1./12*x[i+2][j]);
      }
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *J,Mat *Jpre,MatStructure *str,void *ptr)
{
  User           user = (User)ptr;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i;
  DM             da;
  Field          *x,*xdot;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xdot,&xdot);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    const PetscReal *k = user->k;
    PetscScalar v[2][2] = {{a + k[0], -k[1]},
                           {-k[0], a+k[1]}};
    ierr = MatSetValuesBlocked(*Jpre,1,&i,1,&i,&v[0][0],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *Jpre) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  User user = (User)ctx;
  DM             da;
  PetscInt       i;
  DMDALocalInfo  info;
  Field          *x;
  PetscReal      hx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)info.mx;

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal r = (i+1)*hx,ik = user->k[1] != 0.0 ? 1.0/user->k[1] : 1.0;
    x[i][0] = 1 + user->s[1]*r;
    x[i][1] = user->k[0]*ik*x[i][0] + user->s[1]*ik;
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
