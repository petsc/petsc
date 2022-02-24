
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
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-kappa",&ctx.kappa,NULL));
  ctx.allencahn = PETSC_FALSE;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-allen-cahn",&ctx.allencahn));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mymonitor",&mymonitor));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 10,1,2,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetFieldName(da,0,"Heat equation: u"));
  CHKERRQ(DMDAGetInfo(da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0));
  dt   = 1.0/(ctx.kappa*Mx*Mx);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(VecDuplicate(x,&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetDM(ts,da));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetRHSFunction(ts,NULL,FormFunction,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetType(ts,TSCN));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(da,x));
  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSSetMaxTime(ts,.02));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_INTERPOLATE));
  CHKERRQ(TSSetSolution(ts,x));

  if (mymonitor) {
    ctx.ports = NULL;
    CHKERRQ(TSMonitorSet(ts,MyMonitor,&ctx,MyDestroy));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-square_initial",&flg));
  if (flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"InitialSolution.heat",FILE_MODE_WRITE,&viewer));
    CHKERRQ(VecView(x,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&da));

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
  PetscInt       i,Mx,xs,xm;
  PetscReal      hx,sx;
  PetscScalar    *x,*f;
  Vec            localX;
  UserCtx        *ctx = (UserCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMGetLocalVector(da,&localX));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  CHKERRQ(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMDAVecGetArrayRead(da,localX,&x));
  CHKERRQ(DMDAVecGetArray(da,F,&f));

  /*
     Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

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
  CHKERRQ(DMDAVecRestoreArrayRead(da,localX,&x));
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  CHKERRQ(DMRestoreLocalVector(da,&localX));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U)
{
  PetscInt          i,xs,xm,Mx,scale=1,N;
  PetscScalar       *u;
  const PetscScalar *f;
  PetscReal         hx,x,r;
  Vec               finesolution;
  PetscViewer       viewer;
  PetscBool         flg;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)Mx;

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMDAVecGetArray(da,U,&u));

  /*
     Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /*  InitialSolution is obtained with
      ./heat -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason    -ts_type cn  -da_refine 9 -ts_max_time 1.e-4 -ts_dt .125e-6 -snes_atol 1.e-25 -snes_rtol 1.e-25  -ts_max_steps 15
  */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-square_initial",&flg));
  if (!flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"InitialSolution.heat",FILE_MODE_READ,&viewer));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&finesolution));
    CHKERRQ(VecLoad(finesolution,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(VecGetSize(finesolution,&N));
    scale = N/Mx;
    CHKERRQ(VecGetArrayRead(finesolution,&f));
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
    CHKERRQ(VecRestoreArrayRead(finesolution,&f));
    CHKERRQ(VecDestroy(&finesolution));
  }

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

/*
    This routine is not parallel
*/
PetscErrorCode  MyMonitor(TS ts,PetscInt step,PetscReal time,Vec U,void *ptr)
{
  UserCtx            *ctx = (UserCtx*)ptr;
  PetscDrawLG        lg;
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
  CHKERRQ(PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,vbounds));
  CHKERRQ(PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1200,800));
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMGetLocalVector(da,&localU));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));
  hx   = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx);
  CHKERRQ(DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU));
  CHKERRQ(DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU));
  CHKERRQ(DMDAVecGetArrayRead(da,localU,&u));

  CHKERRQ(PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,&lg));
  CHKERRQ(PetscDrawLGGetDraw(lg,&draw));
  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  if (!ctx->ports) {
    CHKERRQ(PetscDrawViewPortsCreateRect(draw,1,3,&ctx->ports));
  }
  ports = ctx->ports;
  CHKERRQ(PetscDrawLGGetAxis(lg,&axis));
  CHKERRQ(PetscDrawLGReset(lg));

  xx[0] = 0.0; xx[1] = 1.0; cnt = 2;
  CHKERRQ(PetscOptionsGetRealArray(NULL,NULL,"-zoom",xx,&cnt,NULL));
  xs    = xx[0]/hx; xm = (xx[1] - xx[0])/hx;

  /*
      Plot the  energies
  */
  CHKERRQ(PetscDrawLGSetDimension(lg,1 + (ctx->allencahn ? 1 : 0)));
  CHKERRQ(PetscDrawLGSetColors(lg,colors+1));
  CHKERRQ(PetscDrawViewPortsSet(ports,2));
  x    = hx*xs;
  for (i=xs; i<xs+xm; i++) {
    xx[0] = xx[1] = x;
    yy[0] = PetscRealPart(.25*ctx->kappa*(u[i-1] - u[i+1])*(u[i-1] - u[i+1])*sx);
    if (ctx->allencahn) yy[1] = .25*PetscRealPart((1. - u[i]*u[i])*(1. - u[i]*u[i]));
    CHKERRQ(PetscDrawLGAddPoint(lg,xx,yy));
    x   += hx;
  }
  CHKERRQ(PetscDrawGetPause(draw,&pause));
  CHKERRQ(PetscDrawSetPause(draw,0.0));
  CHKERRQ(PetscDrawAxisSetLabels(axis,"Energy","",""));
  CHKERRQ(PetscDrawLGSetLegend(lg,legend));
  CHKERRQ(PetscDrawLGDraw(lg));

  /*
      Plot the  forces
  */
  CHKERRQ(PetscDrawViewPortsSet(ports,1));
  CHKERRQ(PetscDrawLGReset(lg));
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
    CHKERRQ(PetscDrawLGAddPoint(lg,xx,yy));
    x   += hx;
  }
  CHKERRQ(PetscDrawAxisSetLabels(axis,"Right hand side","",""));
  CHKERRQ(PetscDrawLGSetLegend(lg,NULL));
  CHKERRQ(PetscDrawLGDraw(lg));

  /*
        Plot the solution
  */
  CHKERRQ(PetscDrawLGSetDimension(lg,1));
  CHKERRQ(PetscDrawViewPortsSet(ports,0));
  CHKERRQ(PetscDrawLGReset(lg));
  x    = hx*xs;
  CHKERRQ(PetscDrawLGSetLimits(lg,x,x+(xm-1)*hx,-1.1,1.1));
  CHKERRQ(PetscDrawLGSetColors(lg,colors));
  for (i=xs; i<xs+xm; i++) {
    xx[0] = x;
    yy[0] = PetscRealPart(u[i]);
    CHKERRQ(PetscDrawLGAddPoint(lg,xx,yy));
    x    += hx;
  }
  CHKERRQ(PetscDrawAxisSetLabels(axis,"Solution","",""));
  CHKERRQ(PetscDrawLGDraw(lg));

  /*
      Print the  forces as arrows on the solution
  */
  x   = hx*xs;
  cnt = xm/60;
  cnt = (!cnt) ? 1 : cnt;

  for (i=xs; i<xs+xm; i += cnt) {
    y    = PetscRealPart(u[i]);
    len  = .5*PetscRealPart(ctx->kappa*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max;
    CHKERRQ(PetscDrawArrow(draw,x,y,x,y+len,PETSC_DRAW_RED));
    if (ctx->allencahn) {
      len  = .5*PetscRealPart(u[i] - u[i]*u[i]*u[i])/max;
      CHKERRQ(PetscDrawArrow(draw,x,y,x,y+len,PETSC_DRAW_BLUE));
    }
    x += cnt*hx;
  }
  CHKERRQ(DMDAVecRestoreArrayRead(da,localU,&x));
  CHKERRQ(DMRestoreLocalVector(da,&localU));
  CHKERRQ(PetscDrawStringSetSize(draw,.2,.2));
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawSetPause(draw,pause));
  CHKERRQ(PetscDrawPause(draw));
  PetscFunctionReturn(0);
}

PetscErrorCode  MyDestroy(void **ptr)
{
  UserCtx        *ctx = *(UserCtx**)ptr;

  PetscFunctionBegin;
  CHKERRQ(PetscDrawViewPortsDestroy(ctx->ports));
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
