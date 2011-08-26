
static char help[] = "Solves biharmonic equation in 1d.\n";

/*
  Solves the equation

    u_t = - kappa  \Delta \Delta u 
    Periodic boundary conditions

Evolve the biharmonic heat equation: 
---------------
./biharmonic -ts_monitor -snes_monitor -ts_monitor_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason -ts_monitor_solution_initial -wait   -ts_type beuler  -da_refine 5

Evolve with the restriction that -1 <= u <= 1; i.e. as a variational inequality
---------------
./biharmonic -ts_monitor -snes_monitor -ts_monitor_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason -ts_monitor_solution_initial -wait   -ts_type beuler   -da_refine 5 -vi 

Add a simple term to the right hand side so that the bulge grows/expands with time
---------------
./biharmonic -ts_monitor -snes_monitor -ts_monitor_solution -pc_type lu  -draw_pause .1  -snes_converged_reason  -ts_monitor_solution_initial -wait   -ts_type beuler   -da_refine 5 -vi -growth

   u_t =  kappa \Delta \Delta  6.*u*(u_x)^2 + (3*u^2 - 12) \Delta u
    -1 <= u <= 1 
    Periodic boundary conditions

Evolve the Cahn-Hillard equations:
---------------
./biharmonic -ts_monitor -snes_monitor -ts_monitor_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason  -ts_monitor_solution_initial -wait   -ts_type beuler    -da_refine 6 -vi  -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard


*/
#include <petscdmda.h>
#include <petscts.h>

extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,void*),FormInitialSolution(DM,Vec),MyMonitor(TS,PetscInt,PetscReal,Vec,void*);
                                                                                          typedef struct {PetscDrawLG lg;PetscBool growth;PetscBool cahnhillard;PetscReal kappa;PetscInt energy;PetscReal tol;PetscReal theta;PetscReal theta_c} UserCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                     ts;                 /* nonlinear solver */
  Vec                    x,r;                  /* solution, residual vectors */
  Mat                    J;                    /* Jacobian matrix */
  PetscInt               steps,Mx,maxsteps = 10000000;
  PetscErrorCode         ierr;
  DM                     da;
  MatFDColoring          matfdcoloring;
  ISColoring             iscoloring;
  PetscReal              ftime,dt;
  PetscReal              vbounds[] = {-1.1,1.1};
  PetscBool              wait,vi = PETSC_FALSE;
  Vec                    ul,uh;
  UserCtx                ctx;
  int                    colors[] = {PETSC_DRAW_YELLOW,PETSC_DRAW_RED,PETSC_DRAW_RED,PETSC_DRAW_BLUE,PETSC_DRAW_BLUE};  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char *)0,help);
  ctx.kappa = 1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-kappa",&ctx.kappa,PETSC_NULL);CHKERRQ(ierr);
  ctx.growth = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-growth",&ctx.growth,PETSC_NULL);CHKERRQ(ierr);
  ctx.cahnhillard = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-cahn-hillard",&ctx.cahnhillard,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-vi",&vi,PETSC_NULL);CHKERRQ(ierr);
  ctx.energy = 1;
  ierr = PetscOptionsInt("-energy","type of energy (1=double well, 2=double obstacle, 3=logarithmic)","",ctx.energy,&ctx.energy,PETSC_NULL);CHKERRQ(ierr);
  ctx.tol = 1.0e-8;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-tol",&ctx.tol,PETSC_NULL);CHKERRQ(ierr);
  ctx.theta = .001;
  ctx.theta_c = 1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-theta",&ctx.theta,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-theta_c",&ctx.theta_c,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,vbounds);CHKERRQ(ierr); 
  ierr = PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1200,1000);CHKERRQ(ierr); 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_PERIODIC, -10 ,1,2,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"Biharmonic heat equation: u");CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dt   = 1.0/(10.*ctx.kappa*Mx*Mx*Mx*Mx);

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
  ierr = TSSetDuration(ts,maxsteps,.02);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,PETSC_TRUE);CHKERRQ(ierr);

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

  if (vi) {
    ierr = VecDuplicate(x,&ul);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&uh);CHKERRQ(ierr);
    ierr = VecSet(ul,-1.0);CHKERRQ(ierr);
    ierr = VecSet(uh,1.0);CHKERRQ(ierr);
    ierr = TSVISetVariableBounds(ts,ul,uh);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(da,x);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  ierr = PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,&ctx.lg);CHKERRQ(ierr);
  if (ctx.cahnhillard) {
    ierr = PetscDrawLGSetDimension(ctx.lg,5);CHKERRQ(ierr);
  } else {
    ierr = PetscDrawLGSetDimension(ctx.lg,3);CHKERRQ(ierr);
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
  if (vi) {
    ierr = VecDestroy(&ul);CHKERRQ(ierr);
    ierr = VecDestroy(&uh);CHKERRQ(ierr);
  }
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
  PetscScalar    *x,*f,c,r,l;
  Vec            localX;
  UserCtx        *ctx = (UserCtx*)ptr;
  PetscReal      tol=ctx->tol,theta=ctx->theta,theta_c=ctx->theta_c;

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
    c = (x[i-1] + x[i+1] - 2.0*x[i])*sx;
    r = (x[i] + x[i+2] - 2.0*x[i+1])*sx;
    l = (x[i-2] + x[i] - 2.0*x[i-1])*sx;
    f[i] = -ctx->kappa*(l + r - 2.0*c)*sx; 
    if (ctx->growth) {
      f[i] += 100000*((x[i] < .2) ?  0: sx*x[i]);
    }
    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: // double well
      f[i] += 6.*.25*x[i]*(x[i+1] - x[i-1])*(x[i+1] - x[i-1])*sx + (3.*x[i]*x[i] - 1.)*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
      break;
      case 2: // double obstacle
        f[i] += -(x[i-1] + x[i+1] - 2.0*x[i])*sx;
          break;
      case 3: // logarithmic
        if (x[i] < -1.0 + 2.0*ctx->tol) {
          f[i] += 2.0*theta*(2.0*ctx->tol-1.0)/(16.0*(ctx->tol-ctx->tol*ctx->tol)*(ctx->tol-ctx->tol*ctx->tol))*.25*(x[i+1] - x[i-1])*(x[i+1] - x[i-1])*sx + ( .25*theta/(ctx->tol-ctx->tol*ctx->tol) - theta_c)*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
        } else if (x[i] > 1.0 - 2.0*ctx->tol) {
          f[i] += 2.0*theta*(-2.0*ctx->tol+1.0)/(16.0*(ctx->tol-ctx->tol*ctx->tol)*(ctx->tol-ctx->tol*ctx->tol))*.25*(x[i+1] - x[i-1])*(x[i+1] - x[i-1])*sx + ( .25*theta/(ctx->tol-ctx->tol*ctx->tol) - theta_c)*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
        } else {
          f[i] += 2.0*theta*x[i]/((1.0-x[i]*x[i])*(1.0-x[i]*x[i]))*.25*(x[i+1] - x[i-1])*(x[i+1] - x[i-1])*sx + (theta/(1.0-x[i]*x[i]) - theta_c)*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
        }
        break;
      }
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
  PetscScalar    *u,c,r,l;
  PetscInt       Mx,i,xs,xm,cnt;
  PetscReal      x,y,y_pm[2],hx,pause,sx,len,max,maxe,xx[5],yy[5];
  PetscDraw      draw;
  Vec            localU;
  DM             da;
  int            cl;
  PetscReal      tol=ctx->tol,theta=ctx->theta,theta_c=ctx->theta_c;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                        PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  hx     = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx); cnt = Mx/200;
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localU,&u);CHKERRQ(ierr);
 
  /*
      Plot the solution, energies and forces
  */
  maxe = 0;
  max = 0;
  for (i=0; i<Mx; i++) {
    len   = PetscRealPart(.25*ctx->kappa*(u[i-1] - u[i+1])*(u[i-1] - u[i+1])*sx);
    maxe  = PetscMax(maxe,len);
    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: // double well
        len   = .25*PetscRealPart((1. - u[i]*u[i])*(1. - u[i]*u[i]));
        break;
      case 2: // double obstacle 
        len = .5*PetscRealPart(1. - u[i]*u[i]);
        break;
      case 3:
        if (u[i] < -1.0 + 2.0*tol) {
          len = .5*theta*(2.0*tol*log(tol) + (1.0-u[i])*log((1-u[i])/2.0)) + .5*theta_c*(1.0-u[i]*u[i]);
        }
        else if (u[i] > 1.0 - 2.0*tol) {
          len = .5*theta*((1.0+u[i])*log((1.0+u[i])/2.0) + 2.0*tol*log(tol) ) + .5*theta_c*(1.0-u[i]*u[i]);
        } else {
          len = .5*theta*((1.0+u[i])*log((1.0+u[i])/2.0) + (1.0-u[i])*log((1.0-u[i])/2.0)) + .5*theta_c*(1.0-u[i]*u[i]);
        }
        break;
      }
      maxe  = PetscMax(maxe,len);
    }
    c = (u[i-1] + u[i+1] - 2.0*u[i])*sx;
    r = (u[i] + u[i+2] - 2.0*u[i+1])*sx;
    l = (u[i-2] + u[i] - 2.0*u[i-1])*sx;
    len = PetscAbs(-ctx->kappa*(l + r - 2.0*c)*sx); 
    max  = PetscMax(max,len);
    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: // double well
        len   = PetscAbs( 6.*.25*u[i]*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (3.*u[i]*u[i] - 1.)*(u[i-1] + u[i+1] - 2.0*u[i])*sx);
      break;
      case 2: // double obstacle
        len   = PetscAbs(-(u[i-1] + u[i+1] - 2.0*u[i])*sx);
          break;
      case 3: // logarithmic
        if (u[i] < -1.0 + 2.0*tol) {
          len = PetscAbs(2.0*theta*(2.0*tol-1.0)/(16.0*(tol-tol*tol)*(tol-tol*tol))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + ( .25*theta/(tol-tol*tol) - theta_c)*(u[i-1] + u[i+1] - 2.0*u[i])*sx);
        } else if (u[i] > 1.0 - 2.0*tol) {
          len = PetscAbs(2.0*theta*(-2.0*tol+1.0)/(16.0*(tol-tol*tol)*(tol-tol*tol))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + ( .25*theta/(tol-tol*tol) - theta_c)*(u[i-1] + u[i+1] - 2.0*u[i])*sx);
        } else {
          len = PetscAbs(- theta_c*(u[i-1] + u[i+1] - 2.0*u[i])*sx);
          max = PetscMax(max,len);
          len = PetscAbs(2.0*theta*u[i]/((1.0-u[i]*u[i])*(1.0-u[i]*u[i]))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (theta/(1.0-u[i]*u[i]))*(u[i-1] + u[i+1] - 2.0*u[i])*sx);
        }
        break;
      }
      max   = PetscMax(max,len);
    }
    x   += hx;
    //if (max > 7200150000.0)
    //printf("max very big when i = %d\n",i);
  }

  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);
  ierr = PetscDrawLGSetLimits(lg,0.0,1,-2.0,3.2);CHKERRQ(ierr);
  x = 0.;
  /*
   yy[0] : phase variable u
   yy[1] : pointwise value of the integrand of the energy functional,
           penalizes large gradients
   yy[3] : pointwise value of the integrand of the energy functional,
           homogeneous free energy
   yy[2] : right-hand side of u_t, corresponding to y[1]
   yy[4] : right-hand side of u_t, corresponding to y[3]
           
           
   */
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = xx[3] = xx[4] = x;
    yy[0] = PetscRealPart(u[i]);
    yy[1] = .5*PetscRealPart(.25*ctx->kappa*(u[i-1] - u[i+1])*(u[i-1] - u[i+1])*sx/maxe) + 1.6;
    c = (u[i-1] + u[i+1] - 2.0*u[i])*sx;
    r = (u[i] + u[i+2] - 2.0*u[i+1])*sx;
    l = (u[i-2] + u[i] - 2.0*u[i-1])*sx;
    yy[2] = -.5*ctx->kappa*(l + r - 2.0*c)*sx/max + 2.6;
    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: // double well
        yy[3] = .5*.25*PetscRealPart((1. - u[i]*u[i])*(1. - u[i]*u[i])/maxe) + 1.6;
        yy[4] = .5*(6.*.25*u[i]*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (3.*u[i]*u[i] - 1.)*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max + 2.6;
        break;
      case 2: // double obstacle
        yy[3] = .5*.5*PetscRealPart((1. - u[i]*u[i])/maxe) + 1.6;
        yy[4] = .5*(-(u[i-1] + u[i+1] - 2.0*u[i])*sx/max) + 2.6;
        break;
      case 3: // logarithmic
        if (u[i] < -1.0 + 2.0*tol) {
          yy[3] = .5*PetscRealPart(.5*theta*(2.0*tol*log(tol) + (1.0-u[i])*log((1-u[i])/2.0)) + .5*theta_c*(1.0-u[i]*u[i]))/maxe + 1.6;
          yy[4] = .5*(2.0*theta*(2.0*tol-1.0)/(16.0*(tol-tol*tol)*(tol-tol*tol))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + ( .25*theta/(tol-tol*tol) - theta_c)*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max + 2.6;
        } else if (u[i] > 1.0 - 2.0*tol) {
          yy[3] = .5*PetscRealPart(.5*theta*((1.0+u[i])*log((1.0+u[i])/2.0) + 2.0*tol*log(tol) ) + .5*theta_c*(1.0-u[i]*u[i]))/maxe + 1.6;
          yy[4] = .5*(2.0*theta*(-2.0*tol+1.0)/(16.0*(tol-tol*tol)*(tol-tol*tol))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + ( .25*theta/(tol-tol*tol) - theta_c)*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max + 2.6;
        } else {
          yy[3] = .5*PetscRealPart(.5*theta*((1.0+u[i])*log((1.0+u[i])/2.0) + (1.0-u[i])*log((1.0-u[i])/2.0)) + .5*theta_c*(1.0-u[i]*u[i]))/maxe + 1.6;
          yy[4] = .5*(2.0*theta*u[i]/((1.0-u[i]*u[i])*(1.0-u[i]*u[i]))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (theta/(1.0-u[i]*u[i]) - theta_c)*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max + 2.6;
        }
        break;
      }
    }
    ierr = PetscDrawLGAddPoint(lg,xx,yy);CHKERRQ(ierr);
    x   += hx;
  }
  ierr = PetscDrawGetPause(draw,&pause);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,0.0);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

  /*
        Draw arrows for the forces
  */
  x = 0.;
  /*
   y_pm[0] : y-coordinate of the initial point for the next force in the positive direction
   y_pm[1] : y-coordinate of the initial point for the next force in the negative direction
   */

  for (i=0; i<Mx; i += cnt) {
    y_pm[0]    = PetscRealPart(u[i]);
    y_pm[1]    = PetscRealPart(u[i]);
    y          = y_pm[0];
    c = (u[i-1] + u[i+1] - 2.0*u[i])*sx;
    r = (u[i] + u[i+2] - 2.0*u[i+1])*sx;
    l = (u[i-2] + u[i] - 2.0*u[i-1])*sx;
    len = -.5*ctx->kappa*(l + r - 2.0*c)*sx/max; 
    cl   = len < 0. ? PETSC_DRAW_RED : PETSC_DRAW_MAGENTA;
    ierr = PetscDrawArrow(draw,x,y,x,y+len,cl);CHKERRQ(ierr);
    if (ctx->cahnhillard) {
      if (len < 0.) {
        y_pm[1] += len;
      } else {
        y_pm[0] += len;
      }
      switch (ctx->energy) {
      case 1: // double well
        len   = .5*(6.*.25*u[i]*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (3.*u[i]*u[i] - 1.)*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max;
        cl   = len < 0. ? PETSC_DRAW_GREEN : PETSC_DRAW_BLUE;
        y    = len < 0. ? y_pm[1] : y_pm[0];
        ierr = PetscDrawArrow(draw,x,y,x,y+len,cl);CHKERRQ(ierr);
        break;
      case 2: // double obstacle
        len   = .5*(-(u[i-1] + u[i+1] - 2.0*u[i])*sx/max);
        cl   = len < 0. ? PETSC_DRAW_GREEN : PETSC_DRAW_BLUE;
        y    = len < 0. ? y_pm[1] : y_pm[0];
        ierr = PetscDrawArrow(draw,x,y,x,y+len,cl);CHKERRQ(ierr);
        break;
      case 3: // logarithmic
        len   = .5*theta_c*(-(u[i-1] + u[i+1] - 2.0*u[i])*sx/max);
        cl   = len < 0. ? PETSC_DRAW_GREEN : PETSC_DRAW_BLUE;
        y    = len < 0. ? y_pm[1] : y_pm[0];
        ierr = PetscDrawArrow(draw,x,y,x,y+len,cl);CHKERRQ(ierr);
        if (len < 0.) {
          y_pm[1] += len;
        } else {
          y_pm[0] += len;
        }
        /* Finally, the logarithmic component of the force */
        if (u[i] < -1.0 + 2.0*tol) {
          len   = .5*(2.0*theta*( 2.0*tol-1.0)/(16.0*(tol-tol*tol)*(tol-tol*tol))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (.25*theta/(tol-tol*tol))*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max;
        } else if (u[i] > 1.0 - 2.0*tol) {
          len   = .5*(2.0*theta*(-2.0*tol+1.0)/(16.0*(tol-tol*tol)*(tol-tol*tol))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (.25*theta/(tol-tol*tol))*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max;
        } else {
          len   = .5*(2.0*theta*u[i]/((1.0-u[i]*u[i])*(1.0-u[i]*u[i]))*.25*(u[i+1] - u[i-1])*(u[i+1] - u[i-1])*sx + (theta/(1.0-u[i]*u[i]))*(u[i-1] + u[i+1] - 2.0*u[i])*sx)/max;
        }
        cl   = len < 0. ? PETSC_DRAW_BLACK : PETSC_DRAW_PLUM;
        y    = len < 0. ? y_pm[1] : y_pm[0];
        ierr = PetscDrawArrow(draw,x,y,x,y+len,cl);CHKERRQ(ierr);
        break ;
      }
    }
    x   += cnt*hx; 
  }
  ierr = DMDAVecRestoreArray(da,localU,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = PetscDrawStringSetSize(draw,.2,.2);CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.75,2.3,PETSC_DRAW_BLACK,"Relative Forcing");
  ierr = PetscDrawString(draw,.75,1.5,PETSC_DRAW_BLACK,"Relative Energy");
  ierr = PetscDrawString(draw,.75,.8,PETSC_DRAW_BLACK,"Solution");
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,pause);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
