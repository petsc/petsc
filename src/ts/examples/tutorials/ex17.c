static const char help[] = "Time-dependent PDE in 1d. Simplified from ex15.c for illustrating how to solve DAEs. \n";
/*
   u_t = uxx
   0 < x < 1;
   At t=0: u(x) = exp(c*r*r*r), if r=PetscSqrtReal((x-.5)*(x-.5)) < .125
           u(x) = 0.0           if r >= .125


   Boundary conditions:
   Dirichlet BC:
   At x=0, x=1, u = 0.0

   Neumann BC:
   At x=0, x=1: du(x,t)/dx = 0

   mpiexec -n 2 ./ex17 -da_grid_x 40 -ts_max_steps 2 -snes_monitor -ksp_monitor
         ./ex17 -da_grid_x 40 -monitor_solution
         ./ex17 -da_grid_x 100  -ts_type theta -ts_theta_theta 0.5 # Midpoint is not L-stable
         ./ex17 -jac_type fd_coloring  -da_grid_x 500 -boundary 1
         ./ex17 -da_grid_x 100  -ts_type gl -ts_adapt_type none -ts_max_steps 2
*/

#include <petscdmda.h>
#include <petscts.h>

typedef enum {JACOBIAN_ANALYTIC,JACOBIAN_FD_COLORING,JACOBIAN_FD_FULL} JacobianType;
static const char *const JacobianTypes[] = {"analytic","fd_coloring","fd_full","JacobianType","fd_",0};

/*
   User-defined data structures and routines
*/
typedef struct {
  PetscReal      c;
  PetscInt       boundary;       /* Type of boundary condition */
  PetscBool      viewJacobian;
} AppCtx;

static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;                   /* nonlinear solver */
  Vec            u;                    /* solution, residual vectors */
  Mat            J;                    /* Jacobian matrix */
  PetscInt       steps,maxsteps = 1000;     /* iterations for convergence */
  PetscErrorCode ierr;
  DM             da;
  PetscReal      ftime,dt;
  AppCtx         user;              /* user-defined work context */
  JacobianType   jacType;

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-11,1,1,PETSC_NULL,&da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);

  /* Initialize user application context */
  user.c             = -30.0;
  user.boundary      = 0; /* 0: Dirichlet BC; 1: Neumann BC */
  user.viewJacobian  = PETSC_FALSE;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-boundary",&user.boundary,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-viewJacobian",&user.viewJacobian);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts,1.0);CHKERRQ(ierr); /* Make the Theta method behave like backward Euler */
  ierr = TSSetIFunction(ts,PETSC_NULL,FormIFunction,&user);CHKERRQ(ierr);

  ierr = DMCreateMatrix(da,MATAIJ,&J);CHKERRQ(ierr);
  jacType = JACOBIAN_ANALYTIC; /* use user-provide Jacobian */
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,&user);CHKERRQ(ierr);

  ierr = TSSetDM(ts,da);CHKERRQ(ierr); /* Use TSGetDM() to access. Setting here allows easy use of geometric multigrid. */

  ftime = 1.0;
  ierr = TSSetDuration(ts,maxsteps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,u,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  dt   = .01;
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Use slow fd Jacobian or fast fd Jacobian with colorings.
     Note: this requirs snes which is not created until TSSetUp()/TSSetFromOptions() is called */
  ierr = PetscOptionsBegin(((PetscObject)da)->comm,PETSC_NULL,"Options for Jacobian evaluation",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-jac_type","Type of Jacobian","",JacobianTypes,(PetscEnum)jacType,(PetscEnum*)&jacType,0);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (jacType == JACOBIAN_FD_COLORING) {
    SNES       snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianColor,0);CHKERRQ(ierr);
  } else if (jacType == JACOBIAN_FD_FULL){
    SNES       snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobian,&user);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,u,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormIFunction"
static PetscErrorCode FormIFunction(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,Mx,xs,xm;
  PetscReal      hx,sx;
  PetscScalar    *u,*udot,*f;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    if (user->boundary == 0) { /* Dirichlet BC */
      if (i == 0 || i == Mx-1) {
        f[i] = u[i]; /* F = U */
      } else {
        f[i] = udot[i] + (2.*u[i] - u[i-1] - u[i+1])*sx;
      }
    } else { /* Neumann BC */
      if (i == 0) {
        f[i] = u[0] - u[1];
      } else if (i == Mx-1) {
        f[i] = u[i] - u[i-1];
      } else {
        f[i] = udot[i] + (2.*u[i] - u[i-1] - u[i+1])*sx;
      }
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat *J,Mat *Jpre,MatStructure *str,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend,Mx;
  PetscReal      hx,sx;
  AppCtx         *user = (AppCtx*)ctx;
  DM             da;
  MatStencil     col[3],row;
  PetscInt       nc;
  PetscScalar    vals[3];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*Jpre,&rstart,&rend);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  for (i=rstart; i<rend; i++) {
    nc    = 0;
    row.i = i;
    if (user->boundary == 0 && (i == 0 || i == Mx-1)) {
      col[nc].i = i; vals[nc++] = 1.0;
    } else if (user->boundary > 0 && i == 0) { /* Left Neumann */
      col[nc].i = i;   vals[nc++] = 1.0;
      col[nc].i = i+1; vals[nc++] = -1.0;
    } else if (user->boundary > 0 && i == Mx-1) { /* Right Neumann */
      col[nc].i = i-1; vals[nc++] = -1.0;
      col[nc].i = i;   vals[nc++] = 1.0;
    } else {                    /* Interior */
      col[nc].i = i-1; vals[nc++] = -1.0*sx;
      col[nc].i = i;   vals[nc++] = 2.0*sx + a;
      col[nc].i = i+1; vals[nc++] = -1.0*sx;
    }
    ierr = MatSetValuesStencil(*Jpre,1,&row,nc,col,vals,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *Jpre) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  if (user->viewJacobian){
    ierr = PetscPrintf(((PetscObject)*Jpre)->comm,"Jpre:\n");CHKERRQ(ierr);
    ierr = MatView(*Jpre,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(TS ts,Vec U,void* ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  PetscReal      c=user->c;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,xs,xm,Mx;
  PetscScalar    *u;
  PetscReal      hx,x,r;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)(Mx-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    r = PetscSqrtScalar((x-.5)*(x-.5));
    if (r < .125) {
      u[i] = PetscExpScalar(c*r*r*r);
    } else {
      u[i] = 0.0;
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



