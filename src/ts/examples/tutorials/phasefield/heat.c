
static char help[] = "Solves heat equation in 1d.\n";

/*
  Solves the equation

    u_t = kappa  \Delta u 
       Periodic boundary conditions

Evolve the  heat equation: 
---------------
./heat -ts_monitor -snes_monitor -ts_monitor_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason -ts_monitor_solution_initial -wait   -ts_type beuler  -da_refine 5

Evolve the  Allen-Cahn equation: 
---------------
./heat -ts_monitor -snes_monitor -ts_monitor_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason -ts_monitor_solution_initial -wait   -ts_type beuler  -da_refine 5   -allen-cahn -kappa .001 -ts_max_time 5


*/
#include <petscdmda.h>
#include <petscts.h>


/* 
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,void*),FormInitialSolution(DM,Vec),MyMonitor(TS,PetscInt,PetscReal,Vec,void*);
typedef struct {PetscDrawLG lg;PetscReal kappa;PetscBool allencahn;} UserCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                     ts;                 /* nonlinear solver */
  Vec                    x,r;                  /* solution, residual vectors */
  Mat                    J;                    /* Jacobian matrix */
  PetscInt               steps,Mx, maxsteps = 1000000;
  PetscErrorCode         ierr;
  DM                     da;
  MatFDColoring          matfdcoloring;
  ISColoring             iscoloring;
  PetscReal              ftime,dt;
  PetscReal              vbounds[] = {-1.1,1.1};
  PetscBool              wait;
  UserCtx                ctx;
  int                    colors[] = {PETSC_DRAW_YELLOW,PETSC_DRAW_BLACK,PETSC_DRAW_GREEN};

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char *)0,help);
  ctx.kappa = 1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-kappa",&ctx.kappa,PETSC_NULL);CHKERRQ(ierr);
  ctx.allencahn = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-allen-cahn",&ctx.allencahn);CHKERRQ(ierr);

  ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,vbounds);CHKERRQ(ierr); 
  ierr = PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1200,1000);CHKERRQ(ierr); 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_PERIODIC, -9 ,1,1,PETSC_NULL,&da);CHKERRQ(ierr);
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
  ierr = TSSetRHSFunction(ts,PETSC_NULL,FormFunction,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner) 
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMGetColoring(da,IS_COLORING_GLOBAL,MATAIJ,&iscoloring);CHKERRQ(ierr);
  ierr = DMGetMatrix(da,MATAIJ,&J);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormFunction,&ctx);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,TSDefaultComputeJacobianColor,matfdcoloring);CHKERRQ(ierr);

  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(da,x);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,.02);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  ierr = PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,&ctx.lg);CHKERRQ(ierr);
  if (ctx.allencahn) {
    ierr = PetscDrawLGSetDimension(ctx.lg,3);CHKERRQ(ierr);
  } else {
    ierr = PetscDrawLGSetDimension(ctx.lg,2);CHKERRQ(ierr);
  }
  ierr = PetscDrawLGSetColors(ctx.lg,colors);CHKERRQ(ierr);

  ierr = TSMonitorSet(ts,MyMonitor,&ctx,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x,&ftime);CHKERRQ(ierr);
  wait  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-wait",&wait,PETSC_NULL);CHKERRQ(ierr);
  if (wait) {
    ierr = PetscSleep(-1);CHKERRQ(ierr);
  }
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&matfdcoloring);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
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
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx     = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx);

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
  ierr = DMDAVecGetArray(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    f[i] = ctx->kappa*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
    if (ctx->allencahn) {
      f[i] += (x[i] - x[i]*x[i]*x[i]);
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,xs,xm,Mx;
  PetscScalar    *u,r;
  PetscReal      hx,x;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx     = 1.0/(PetscReal)Mx;

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    r = PetscSqrtScalar((x-.5)*(x-.5));
    if (r < .125) {
      u[i] = 1.0;
    } else {
      u[i] = -.5;
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "MyMonitor"
/*
    This routine is not parallel
*/
PetscErrorCode  MyMonitor(TS ts,PetscInt step,PetscReal time,Vec U,void *ptr)
{
  UserCtx        *ctx = (UserCtx*)ptr;
  PetscDrawLG    lg = (PetscDrawLG)ctx->lg;
  PetscErrorCode ierr;
  PetscScalar    *u;
  PetscInt       Mx,i,xs,xm,cnt;
  PetscReal      x,y,hx,pause,sx,len,max,maxe,xx[3],yy[3];
  PetscDraw      draw;
  Vec            localU;
  DM             da;
  int            cl;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                        PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  hx     = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx); cnt = Mx/60;
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localU,&u);CHKERRQ(ierr);

  /*
      Plot the energies
  */
  maxe = 0;
  for (i=0; i<Mx; i++) {
    len   = PetscRealPart(.25*ctx->kappa*(u[i-1] - u[i+1])*(u[i-1] - u[i+1])*sx);
    maxe  = PetscMax(maxe,len);
    if (ctx->allencahn) {
      len   = .25*PetscRealPart((1. - u[i]*u[i])*(1. - u[i]*u[i]));
      maxe  = PetscMax(maxe,len);
    }
    x   += hx;
  }

  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);
  ierr = PetscDrawLGSetLimits(lg,0.0,1,-1.,1.6);CHKERRQ(ierr);
  x = 0.;
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = x;
    yy[0] = PetscRealPart(u[i]);
    yy[1] = .5*PetscRealPart(.25*ctx->kappa*(u[i-1] - u[i+1])*(u[i-1] - u[i+1])*sx/maxe) + 1.1;
    if (ctx->allencahn) {
      yy[2] = .5*.25*PetscRealPart((1. - u[i]*u[i])*(1. - u[i]*u[i])/maxe) + 1.1;
    } 

    ierr = PetscDrawLGAddPoint(lg,xx,yy);CHKERRQ(ierr);
    x   += hx;
  }
  ierr = PetscDrawGetPause(draw,&pause);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,0.0);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

  /*
      Plot the forces
  */
  x   = 0.;
  max = 0;
  for (i=0; i<Mx; i += cnt) {
    len  = PetscAbs(PetscRealPart(ctx->kappa*(u[i-1] + u[i+1] - 2.0*u[i])*sx));
    max  = PetscMax(max,len);
    if (ctx->allencahn) {
      len   = PetscAbs(PetscRealPart(u[i] - u[i]*u[i]*u[i]));
      max   = PetscMax(max,len);
    }
    x   += cnt*hx;
  }
  x = 0.;
  for (i=0; i<Mx; i += cnt) {
    y    = PetscRealPart(u[i]);
    len  = .5*PetscRealPart(ctx->kappa*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max;
    cl   = len < 0. ? PETSC_DRAW_RED : PETSC_DRAW_MAGENTA;
    ierr = PetscDrawArrow(draw,x,y,x,y+len,cl);CHKERRQ(ierr);
    if (ctx->allencahn) {
      len   = .5*PetscRealPart(u[i] - u[i]*u[i]*u[i])/max;
      cl   = len < 0. ? PETSC_DRAW_GREEN : PETSC_DRAW_BLUE;
      ierr = PetscDrawArrow(draw,x,y,x,y+len,cl);CHKERRQ(ierr);
    }
    x   += cnt*hx; 
  }
  ierr = DMDAVecRestoreArray(da,localU,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = PetscDrawStringSetSize(draw,.2,.2);CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.75,1.5,PETSC_DRAW_BLACK,"Relative Energy");
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,pause);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
