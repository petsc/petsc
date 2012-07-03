static const char help[] = "Time-dependent Brusselator reaction-diffusion PDE in 1d. Demonstrates IMEX methods.\n";
/*
   u_t - alpha u_xx = A + u^2 v - (B+1) u
   v_t - alpha v_xx = B u - u^2 v
   0 < x < 1;
   A = 1, B = 3, alpha = 1/50

   Initial conditions:
   u(x,0) = 1 + sin(2 pi x)
   v(x,0) = 3

   Boundary conditions:
   u(0,t) = u(1,t) = 1
   v(0,t) = v(1,t) = 3
*/

#include <petscdmda.h>
#include <petscts.h>

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct _User *User;
struct _User {
  PetscReal A,B;                /* Reaction coefficients */
  PetscReal alpha;              /* Diffusion coefficient */
  PetscReal uleft,uright;       /* Dirichlet boundary conditions */
  PetscReal vleft,vright;       /* Dirichlet boundary conditions */
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
    user.A      = 1;
    user.B      = 3;
    user.alpha  = 0.02;
    user.uleft  = 1;
    user.uright = 1;
    user.vleft  = 3;
    user.vright = 3;
    ierr = PetscOptionsReal("-A","Reaction rate","",user.A,&user.A,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-B","Reaction rate","",user.B,&user.B,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","Diffusion coefficient","",user.alpha,&user.alpha,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-uleft","Dirichlet boundary condition","",user.uleft,&user.uleft,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-uright","Dirichlet boundary condition","",user.uright,&user.uright,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vleft","Dirichlet boundary condition","",user.vleft,&user.vleft,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vright","Dirichlet boundary condition","",user.vright,&user.vright,PETSC_NULL);CHKERRQ(ierr);
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

  ftime = 10.0;
  maxsteps = 10000;
  ierr = TSSetDuration(ts,maxsteps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&mx);CHKERRQ(ierr);
  dt   = 0.4 * user.alpha / PetscSqr(mx); /* Diffusive CFL */
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
  PetscReal      hx;
  Vec            Xloc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(info.mx-1);

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
  ierr = DMDAVecGetArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    if (i == 0) {
      f[i].u = hx * (x[i].u - user->uleft);
      f[i].v = hx * (x[i].v - user->vleft);
    } else if (i == info.mx-1) {
      f[i].u = hx * (x[i].u - user->uright);
      f[i].v = hx * (x[i].v - user->vright);
    } else {
      f[i].u = hx * xdot[i].u - user->alpha * (x[i-1].u - 2.*x[i].u + x[i+1].u) / hx;
      f[i].v = hx * xdot[i].v - user->alpha * (x[i-1].v - 2.*x[i].v + x[i+1].v) / hx;
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User           user = (User)ptr;
  DM             da;
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      hx;
  Field          *x,*f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscScalar u = x[i].u,v = x[i].v;
    f[i].u = hx*(user->A + u*u*v - (user->B+1)*u);
    f[i].v = hx*(user->B*u - u*u*v);
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
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
  PetscReal      hx;
  DM             da;
  Field          *x,*xdot;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xdot,&xdot);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    if (i == 0 || i == info.mx-1) {
      const PetscInt row = i,col = i;
      const PetscScalar vals[2][2] = {{hx,0},{0,hx}};
      ierr = MatSetValuesBlocked(*Jpre,1,&row,1,&col,&vals[0][0],INSERT_VALUES);CHKERRQ(ierr);
    } else {
      const PetscInt row = i,col[] = {i-1,i,i+1};
      const PetscScalar dxxL = -user->alpha/hx,dxx0 = 2.*user->alpha/hx,dxxR = -user->alpha/hx;
      const PetscScalar vals[2][3][2] = {{{dxxL,0},{a*hx+dxx0,0},{dxxR,0}},
                                         {{0,dxxL},{0,a*hx+dxx0},{0,dxxR}}};
      ierr = MatSetValuesBlocked(*Jpre,1,&row,3,col,&vals[0][0][0],INSERT_VALUES);CHKERRQ(ierr);
    }
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
  hx = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal xi = i*hx;
    x[i].u = user->uleft*(1.-xi) + user->uright*xi + sin(2.*PETSC_PI*xi);
    x[i].v = user->vleft*(1.-xi) + user->vright*xi;
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
