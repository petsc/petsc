
static char help[] = "Time-dependent PDE in 2d. Modified from ex13.c for illustrating how to solve DAEs. \n";
/* 
   u_t = uxx + uyy
   0 < x < 1, 0 < y < 1; 
   At t=0: u(x,y) = exp(c*r*r*r), if r=sqrt((x-.5)*(x-.5) + (y-.5)*(y-.5)) < .125
           u(x,y) = 0.0           if r >= .125

Program usage:  
   mpiexec -n <procs> ./ex13 [-help] [all PETSc options] 
   e.g., mpiexec -n 2 ./ex13 -da_grid_x 40 -da_grid_y 40 -ts_max_steps 2 -use_coloring -snes_monitor -ksp_monitor 
         ./ex13 -da_grid_x 40 -da_grid_y 40 -use_coloring -drawcontours 
         ./ex13 -use_coloring -drawcontours -draw_pause -1
         mpiexec -n 2 ./ex13 -drawcontours -ts_type sundials -ts_sundials_monitor_steps
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
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_NONPERIODIC,DMDA_STENCIL_STAR,-8,-8,PETSC_DECIDE,PETSC_DECIDE,
                    1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);

  /* Initialize user application context */
  user.da            = da;
  user.c             = -30.0;
  user.coloring      = PETSC_FALSE;
  user.matfdcoloring = PETSC_NULL;

  usermonitor.drawcontours = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-drawcontours",&usermonitor.drawcontours);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSGL);CHKERRQ(ierr); /* General Linear method, TSTHETA can also solve DAE */
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
    ierr = TSSetRHSJacobian(ts,J,J,TSDefaultComputeJacobianColor,user.matfdcoloring);CHKERRQ(ierr);
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
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      two = 2.0,hx,hy,hxdhy,hydhx,sx,sy;
  PetscScalar    u,uxx,uyy,**uarray,**f;
  Vec            localU;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx     = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  hy     = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);
  hxdhy  = hx/hy; 
  hydhx  = hy/hx;

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
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        f[j][i] = uarray[j][i];
        continue;
      }
      u       = uarray[j][i];
      uxx     = (-two*u + uarray[j][i-1] + uarray[j][i+1])*sx;
      uyy     = (-two*u + uarray[j-1][i] + uarray[j+1][i])*sy;
      f[j][i] = uxx + uyy;                        
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,localU,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*ym*xm);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "IFunction"
PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = RHSFunction(ts,t,U,F,ctx);CHKERRQ(ierr);
  ierr = VecAYPX(F,-1.0,Udot);CHKERRQ(ierr);      /* F = Udot - F */
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
  PetscInt       i,rstart,rend;
  PetscScalar    aa=(PetscScalar)a;

  PetscFunctionBegin;
  /* Compute *J = dFrhs/dU */
  ierr = RHSJacobian(ts,t,U,J,Jpre,str,(AppCtx*)ctx);CHKERRQ(ierr);

  /* Compute *J = -dFrhs/dU + aI */
  ierr = MatScale(*J,-1.0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*J,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++){
    ierr = MatSetValues(*J,1,&i,1,&i,&aa,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *Jpre) {
    ierr = MatAssemblyBegin(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
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
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscScalar    **u;
  PetscReal      hx,hy,x,y,r;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      r = PetscSqrtScalar((x-.5)*(x-.5) + (y-.5)*(y-.5));
      if (r < .125) {
        u[j][i] = PetscExpScalar(c*r*r*r);
      } else {
        u[j][i] = 0.0;
      }
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
  PetscReal      norm;
  MPI_Comm       comm;
  MonitorCtx     *user = (MonitorCtx*)ptr;

  PetscFunctionBegin;
  ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"timestep %D: time %G, solution norm %G\n",step,ptime,norm);CHKERRQ(ierr);
  if (user->drawcontours){
    ierr = VecView(v,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

