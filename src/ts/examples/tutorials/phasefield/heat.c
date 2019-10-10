
static char help[] = "Solves heat equation in 1d.\n";

/*
  Solves the equation

    u_t = kappa  \Delta u
       Periodic boundary conditions

Evolve the  heat equation:
---------------
./heat -ts_monitor -snes_monitor  -pc_type lu  -draw_pause .1 -snes_converged_reason  -ts_type cn  -da_refine 5 -mymonitor

Evolve the  Allen-Cahn equation:
---------------
./heat -ts_monitor -snes_monitor  -pc_type lu  -draw_pause .1 -snes_converged_reason  -ts_type cn  -da_refine 5   -allen-cahn -kappa .001 -ts_max_time 5  -mymonitor

Evolve the  Allen-Cahn equation: zoom in on part of the domain
---------------
./heat -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason     -ts_type cn  -da_refine 5   -allen-cahn -kappa .001 -ts_max_time 5  -zoom .25,.45  -mymonitor


The option -square_initial indicates it should use a square wave initial condition otherwise it loads the file InitialSolution.heat as the initial solution. You should run with
./heat -square_initial -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason    -ts_type cn  -da_refine 9 -ts_max_time 1.e-4 -ts_dt .125e-6 -snes_atol 1.e-25 -snes_rtol 1.e-25  -ts_max_steps 15
to generate InitialSolution.heat

*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,void*),FormInitialSolution(DM,Vec),MyMonitor(TS,PetscInt,PetscReal,Vec,void*),MyDestroy(void**);
typedef struct {PetscReal kappa;PetscBool allencahn;PetscDrawViewPorts *ports;} UserCtx;

int main(int argc,char **argv)
{
  TS             ts;                           /* time integrator */
  Vec            x,r;                          /* solution, residual vectors */
  PetscInt       steps,Mx;
  PetscErrorCode ierr;
  DM             da;
  PetscReal      dt;
  UserCtx        ctx;
  PetscBool      mymonitor;
  PetscViewer    viewer;
  PetscBool      flg;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ctx.kappa     = 1.0;
  ierr          = PetscOptionsGetReal(NULL,NULL,"-kappa",&ctx.kappa,NULL);CHKERRQ(ierr);
  ctx.allencahn = PETSC_FALSE;
  ierr          = PetscOptionsHasName(NULL,NULL,"-allen-cahn",&ctx.allencahn);CHKERRQ(ierr);
  ierr          = PetscOptionsHasName(NULL,NULL,"-mymonitor",&mymonitor);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 10,1,2,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"Heat equation: u");CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dt   = 1.0/(ctx.kappa*Mx*Mx);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormFunction,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(da,x);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,.02);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_INTERPOLATE);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);


  if (mymonitor) {
    ctx.ports = NULL;
    ierr      = TSMonitorSet(ts,MyMonitor,&ctx,MyDestroy);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-square_initial",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr  = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"InitialSolution.heat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr  = VecView(x,viewer);CHKERRQ(ierr);
    ierr  = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(TS ts,PetscReal ftime,Vec X,Vec F,void *ptr)
{
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,Mx,xs,xm;
  PetscReal      hx,sx;
  PetscScalar    *x,*f;
  Vec            localX;
  UserCtx        *ctx = (UserCtx*)ptr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    f[i] = ctx->kappa*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
    if (ctx->allencahn) f[i] += (x[i] - x[i]*x[i]*x[i]);
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U)
{
  PetscErrorCode    ierr;
  PetscInt          i,xs,xm,Mx,scale=1,N;
  PetscScalar       *u;
  const PetscScalar *f;
  PetscReal         hx,x,r;
  Vec               finesolution;
  PetscViewer       viewer;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)Mx;

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*  InitialSolution is obtained with
      ./heat -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason    -ts_type cn  -da_refine 9 -ts_max_time 1.e-4 -ts_dt .125e-6 -snes_atol 1.e-25 -snes_rtol 1.e-25  -ts_max_steps 15
  */
  ierr = PetscOptionsHasName(NULL,NULL,"-square_initial",&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr  = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"InitialSolution.heat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr  = VecCreate(PETSC_COMM_WORLD,&finesolution);CHKERRQ(ierr);
    ierr  = VecLoad(finesolution,viewer);CHKERRQ(ierr);
    ierr  = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr  = VecGetSize(finesolution,&N);CHKERRQ(ierr);
    scale = N/Mx;
    ierr  = VecGetArrayRead(finesolution,&f);CHKERRQ(ierr);
  }

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    r = PetscSqrtScalar((x-.5)*(x-.5));
    if (r < .125) u[i] = 1.0;
    else u[i] = -.5;

    /* With the initial condition above the method is first order in space */
    /* this is a smooth initial condition so the method becomes second order in space */
    /*u[i] = PetscSinScalar(2*PETSC_PI*x); */
    /*  u[i] = f[scale*i];*/
    if (!flg) u[i] = f[scale*i];
  }
  if (!flg) {
    ierr = VecRestoreArrayRead(finesolution,&f);CHKERRQ(ierr);
    ierr = VecDestroy(&finesolution);CHKERRQ(ierr);
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    This routine is not parallel
*/
PetscErrorCode  MyMonitor(TS ts,PetscInt step,PetscReal time,Vec U,void *ptr)
{
  UserCtx            *ctx = (UserCtx*)ptr;
  PetscDrawLG        lg;
  PetscErrorCode     ierr;
  PetscScalar        *u;
  PetscInt           Mx,i,xs,xm,cnt;
  PetscReal          x,y,hx,pause,sx,len,max,xx[2],yy[2];
  PetscDraw          draw;
  Vec                localU;
  DM                 da;
  int                colors[] = {PETSC_DRAW_YELLOW,PETSC_DRAW_RED,PETSC_DRAW_BLUE};
  const char*const   legend[] = {"-kappa (\\grad u,\\grad u)","(1 - u^2)^2"};
  PetscDrawAxis      axis;
  PetscDrawViewPorts *ports;
  PetscReal          vbounds[] = {-1.1,1.1};

  PetscFunctionBegin;
  ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,vbounds);CHKERRQ(ierr);
  ierr = PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1200,800);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx);
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);

  ierr = PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  if (!ctx->ports) {
    ierr = PetscDrawViewPortsCreateRect(draw,1,3,&ctx->ports);CHKERRQ(ierr);
  }
  ports = ctx->ports;
  ierr  = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr  = PetscDrawLGReset(lg);CHKERRQ(ierr);

  xx[0] = 0.0; xx[1] = 1.0; cnt = 2;
  ierr  = PetscOptionsGetRealArray(NULL,NULL,"-zoom",xx,&cnt,NULL);CHKERRQ(ierr);
  xs    = xx[0]/hx; xm = (xx[1] - xx[0])/hx;

  /*
      Plot the  energies
  */
  ierr = PetscDrawLGSetDimension(lg,1 + (ctx->allencahn ? 1 : 0));CHKERRQ(ierr);
  ierr = PetscDrawLGSetColors(lg,colors+1);CHKERRQ(ierr);
  ierr = PetscDrawViewPortsSet(ports,2);CHKERRQ(ierr);
  x    = hx*xs;
  for (i=xs; i<xs+xm; i++) {
    xx[0] = xx[1] = x;
    yy[0] = PetscRealPart(.25*ctx->kappa*(u[i-1] - u[i+1])*(u[i-1] - u[i+1])*sx);
    if (ctx->allencahn) yy[1] = .25*PetscRealPart((1. - u[i]*u[i])*(1. - u[i]*u[i]));
    ierr = PetscDrawLGAddPoint(lg,xx,yy);CHKERRQ(ierr);
    x   += hx;
  }
  ierr = PetscDrawGetPause(draw,&pause);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,0.0);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Energy","","");CHKERRQ(ierr);
  ierr = PetscDrawLGSetLegend(lg,legend);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

  /*
      Plot the  forces
  */
  ierr = PetscDrawViewPortsSet(ports,1);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
  x    = xs*hx;
  max  = 0.;
  for (i=xs; i<xs+xm; i++) {
    xx[0] = xx[1] = x;
    yy[0] = PetscRealPart(ctx->kappa*(u[i-1] + u[i+1] - 2.0*u[i])*sx);
    max   = PetscMax(max,PetscAbs(yy[0]));
    if (ctx->allencahn) {
      yy[1] = PetscRealPart(u[i] - u[i]*u[i]*u[i]);
      max   = PetscMax(max,PetscAbs(yy[1]));
    }
    ierr = PetscDrawLGAddPoint(lg,xx,yy);CHKERRQ(ierr);
    x   += hx;
  }
  ierr = PetscDrawAxisSetLabels(axis,"Right hand side","","");CHKERRQ(ierr);
  ierr = PetscDrawLGSetLegend(lg,NULL);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

  /*
        Plot the solution
  */
  ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
  ierr = PetscDrawViewPortsSet(ports,0);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
  x    = hx*xs;
  ierr = PetscDrawLGSetLimits(lg,x,x+(xm-1)*hx,-1.1,1.1);CHKERRQ(ierr);
  ierr = PetscDrawLGSetColors(lg,colors);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    xx[0] = x;
    yy[0] = PetscRealPart(u[i]);
    ierr  = PetscDrawLGAddPoint(lg,xx,yy);CHKERRQ(ierr);
    x    += hx;
  }
  ierr = PetscDrawAxisSetLabels(axis,"Solution","","");CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

  /*
      Print the  forces as arrows on the solution
  */
  x   = hx*xs;
  cnt = xm/60;
  cnt = (!cnt) ? 1 : cnt;

  for (i=xs; i<xs+xm; i += cnt) {
    y    = PetscRealPart(u[i]);
    len  = .5*PetscRealPart(ctx->kappa*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max;
    ierr = PetscDrawArrow(draw,x,y,x,y+len,PETSC_DRAW_RED);CHKERRQ(ierr);
    if (ctx->allencahn) {
      len  = .5*PetscRealPart(u[i] - u[i]*u[i]*u[i])/max;
      ierr = PetscDrawArrow(draw,x,y,x,y+len,PETSC_DRAW_BLUE);CHKERRQ(ierr);
    }
    x += cnt*hx;
  }
  ierr = DMDAVecRestoreArrayRead(da,localU,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = PetscDrawStringSetSize(draw,.2,.2);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,pause);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MyDestroy(void **ptr)
{
  UserCtx        *ctx = *(UserCtx**)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawViewPortsDestroy(ctx->ports);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
     args: -ts_monitor -snes_monitor  -pc_type lu  -snes_converged_reason   -ts_type cn  -da_refine 2 -square_initial

   test:
     suffix: 2
     args: -ts_monitor -snes_monitor  -pc_type lu  -snes_converged_reason   -ts_type cn  -da_refine 5 -mymonitor -square_initial -allen-cahn -kappa .001
     requires: x

TEST*/
