
static char help[] = "Time-dependent PDE in 1d. Simplified from ex15.c for illustrating how to solve DAEs. \n";
/* 
   u_t = uxx 
   0 < x < 1; 
   At t=0: u(x) = exp(c*r*r*r), if r=sqrt((x-.5)*(x-.5)) < .125
           u(x) = 0.0           if r >= .125


   Boundary conditions:   
   Drichlet BC:
   At x=0, x=1, u = 0.0

   Neumann BC:
   At x=0, x=1: du(x,t)/dx = 0

Program usage:  
   mpiexec -n <procs> ./ex16 [-help] [all PETSc options] 
   e.g., mpiexec -n 2 ./ex16 -da_grid_x 40 -ts_max_steps 2 -use_coloring -snes_monitor -ksp_monitor 
         ./ex16 -da_grid_x 40 -use_coloring -drawcontours 
         ./ex16 -use_coloring -drawcontours -draw_pause .1
         ./ex16 -use_coloring -ts_type theta -ts_theta_theta 0.5 
         ./ex16 -use_coloring -boundary 1 
*/

#include "petscdm.h"
#include "petscts.h"

/* 
   User-defined data structures and routines
*/
typedef struct {
   PetscBool drawcontours;   
} MonitorCtx;

typedef struct {
  DM             da;
  PetscReal      c;   
  PetscBool      coloring;
  MatFDColoring  matfdcoloring;
  PetscInt       boundary;       /* Type of boundary condition */
} AppCtx;

extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*); 
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormInitialSolution(Vec,void*);
extern PetscErrorCode MyTSMonitor(TS,PetscInt,PetscReal,Vec,void*);

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
  ISColoring     iscoloring;
  PetscReal      ftime,dt;
  MonitorCtx     usermonitor;       /* user-defined monitor context */
  AppCtx         user;              /* user-defined work context */

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_NONPERIODIC,-8,
                    1,1,PETSC_NULL,&da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);

  /* Initialize user application context */
  user.da            = da;
  user.c             = -30.0;
  user.coloring      = PETSC_FALSE;
  user.matfdcoloring = PETSC_NULL;
  user.boundary      = 0; /* 0: Drichlet BC; 1: Neumann BC */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-boundary",&user.boundary,PETSC_NULL);CHKERRQ(ierr);

  usermonitor.drawcontours = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-drawcontours",&usermonitor.drawcontours);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr); /* General Linear method, TSTHETA can also solve DAE */
  ierr = TSThetaSetTheta(ts,1.0);CHKERRQ(ierr); /* incorrect solution when theta=0.5? */
  ierr = TSSetRHSFunction(ts,RHSFunction,&user);CHKERRQ(ierr); /* needed by RHSJacobian()! */
  ierr = TSSetIFunction(ts,IFunction,&user);CHKERRQ(ierr);

  ierr = DMGetMatrix(da,MATAIJ,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,IJacobian,&user);CHKERRQ(ierr);

  /* Use coloring to compute rhs Jacobian efficiently */
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_coloring",&user.coloring,PETSC_NULL);CHKERRQ(ierr);
  if (user.coloring){
    ierr = DMGetColoring(da,IS_COLORING_GLOBAL,MATAIJ,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(J,iscoloring,&user.matfdcoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(user.matfdcoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
    
    ierr = MatFDColoringSetFunction(user.matfdcoloring,(PetscErrorCode (*)(void))RHSFunction,&user);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,J,J,TSDefaultComputeJacobianColor,&user);CHKERRQ(ierr);
  }

  ftime = 1.0;
  ierr = TSSetDuration(ts,maxsteps,ftime);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,MyTSMonitor,&usermonitor,PETSC_NULL);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(u,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  dt   = .01;
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSStep(ts,&steps,&ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(J);CHKERRQ(ierr);
  if (user.coloring){
    ierr = MatFDColoringDestroy(user.matfdcoloring);CHKERRQ(ierr);
  }
  ierr = VecDestroy(u);CHKERRQ(ierr);     
  ierr = TSDestroy(ts);CHKERRQ(ierr);
  ierr = DMDestroy(da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
/* 
   RHSFunction - Evaluates nonlinear function, F(u).

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  ptr - optional user-defined context, as set by TSSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  DM             da = (DM)user->da;
  PetscErrorCode ierr;
  PetscInt       i,Mx,xs,xm;
  PetscReal      two = 2.0,hx,sx;
  PetscScalar    u,uxx,*uarray,*f;
  Vec            localU;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

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
  ierr = DMDAVecGetArray(da,localU,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    /* Boundary conditions */
    if (i == 0 || i == Mx-1) {
      if (user->boundary == 0){ /* Drichlet BC */
        f[i] = uarray[i]; /* F = U */
      } else {                  /* Neumann BC */
        if (i == 0){
          f[i] = uarray[1] - uarray[0];
        } else if (i == Mx-1){
          f[i] = uarray[Mx-2] - uarray[Mx-1];
        } 
      }
    } else {
      u    = uarray[i];
      uxx  = (-two*u + uarray[i-1] + uarray[i+1])*sx;
      f[i] = uxx;                        
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,localU,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

/* --------------------------------------------------------------------- */
/*
  IFunction = Udot - RHSFunction
*/
#undef __FUNCT__
#define __FUNCT__ "IFunction"
PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;
  DM             da = (DM)user->da;
  PetscInt       i,Mx,xs,xm;
  PetscReal      two = 2.0,hx,sx;
  PetscScalar    u,uxx,*uarray,*f,*udot;
  Vec            localU;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

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
  ierr = DMDAVecGetArray(da,localU,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Udot,&udot);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    /* Boundary conditions */
    if (i == 0 || i == Mx-1) {
      if (user->boundary == 0){ /* Drichlet BC */
        f[i] = uarray[i]; /* F = U */
      } else {                  /* Neumann BC */
        if (i == 0){
          f[i] = uarray[1] - uarray[0];
        } else if (i == Mx-1){
          f[i] = uarray[Mx-2] - uarray[Mx-1];
        } 
      }
    } else {
      u    = uarray[i];
      uxx  = (-two*u + uarray[i-1] + uarray[i+1])*sx;
      f[i] = uxx ;  
      f[i] = udot[i] - f[i]; /* F = Udot - Frhs */
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,localU,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
/*
   RHSJacobian - User-provided routine to compute the Jacobian of
   the nonlinear right-hand-side function of the ODE.

   Input Parameters:
   ts - the TS context
   t - current time
   U - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   J - Jacobian matrix
   Jpre - optionally different preconditioning matrix
   str - flag indicating matrix structure
*/
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat *J,Mat *Jpre,MatStructure *str,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ctx;
  MatFDColoring  matfdcoloring=user->matfdcoloring;

  PetscFunctionBegin;
  
  if (user->coloring){
    ierr = TSDefaultComputeJacobianColor(ts,t,U,J,Jpre,str,matfdcoloring);CHKERRQ(ierr);
  } else {
    ierr = TSDefaultComputeJacobian(ts,t,U,J,Jpre,str,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot = -dFrhs/dU + a dF/dUdot
*/
#undef __FUNCT__
#define __FUNCT__ "IJacobian"
PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat *J,Mat *Jpre,MatStructure *str,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend,M;
  PetscScalar    aa=(PetscScalar)a;

  PetscFunctionBegin;
  /* Compute *J = dFrhs/dU */
  ierr = RHSJacobian(ts,t,U,J,Jpre,str,(AppCtx*)ctx);CHKERRQ(ierr);
  //printf("RHS Jpre:\n");
  //ierr = MatView(*Jpre,PETSC_VIEWER_STDOUT_WORLD);

  /* Compute *J = -dFrhs/dU + aI */
  ierr = MatScale(*Jpre,-1.0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*Jpre,&rstart,&rend);CHKERRQ(ierr);
  
  /* Do not add shift at boundary point? */
  ierr = MatGetSize(*Jpre,&M,PETSC_NULL);CHKERRQ(ierr);
  if (rstart == 0) rstart++;
  if (rend == M) rend--;

  for (i=rstart; i<rend; i++){
    ierr = MatSetValues(*Jpre,1,&i,1,&i,&aa,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *Jpre) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  //printf("Jpre:\n");
  //ierr = MatView(*Jpre,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(Vec U,void* ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  DM             da=user->da;
  PetscReal      c=user->c;
  PetscErrorCode ierr;
  PetscInt       i,xs,xm,Mx;
  PetscScalar    *u;
  PetscReal      hx,x,r;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx     = 1.0/(PetscReal)(Mx-1);

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

#undef __FUNCT__  
#define __FUNCT__ "MyTSMonitor"
PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ptr)
{
  PetscErrorCode ierr;
  PetscReal      norm,vmax,vmin;
  MPI_Comm       comm;
  MonitorCtx     *user = (MonitorCtx*)ptr;

  PetscFunctionBegin;
  ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecMax(v,PETSC_NULL,&vmax);CHKERRQ(ierr);
  ierr = VecMin(v,PETSC_NULL,&vmin);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"timestep %D: time %G, solution norm %G, max %G, min %G\n",step,ptime,norm,vmax,vmin);CHKERRQ(ierr);
  if (user->drawcontours){
    ierr = VecView(v,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

