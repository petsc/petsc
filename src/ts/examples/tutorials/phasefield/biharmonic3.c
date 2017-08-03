
static char help[] = "Solves biharmonic equation in 1d.\n";

/*
  Solves the equation biharmonic equation in split form

    w = -kappa \Delta u
    u_t =  \Delta w
    -1  <= u <= 1
    Periodic boundary conditions

Evolve the biharmonic heat equation with bounds:  (same as biharmonic)
---------------
./biharmonic2 -ts_monitor -snes_monitor -ts_monitor_draw_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason --wait   -ts_type beuler  -da_refine 5 -draw_fields 1 -ts_dt 9.53674e-9

    w = -kappa \Delta u  + u^3  - u
    u_t =  \Delta w
    -1  <= u <= 1
    Periodic boundary conditions

Evolve the Cahn-Hillard equations:
---------------
./biharmonic2 -ts_monitor -snes_monitor -ts_monitor_draw_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason  --wait   -ts_type beuler    -da_refine 6 -vi  -draw_fields 1  -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard


*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,Vec,void*),FormInitialSolution(DM,Vec,PetscReal);
typedef struct {PetscBool cahnhillard;PetscReal kappa;PetscInt energy;PetscReal tol;PetscReal theta;PetscReal theta_c;} UserCtx;

int main(int argc,char **argv)
{
  TS             ts;                           /* nonlinear solver */
  Vec            x,r;                          /* solution, residual vectors */
  Mat            J;                            /* Jacobian matrix */
  PetscInt       steps,Mx,maxsteps = 10000000;
  PetscErrorCode ierr;
  DM             da;
  MatFDColoring  matfdcoloring;
  ISColoring     iscoloring;
  PetscReal      dt;
  PetscReal      vbounds[] = {-100000,100000,-1.1,1.1};
  PetscBool      wait;
  Vec            ul,uh;
  SNES           snes;
  UserCtx        ctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ctx.kappa       = 1.0;
  ierr            = PetscOptionsGetReal(NULL,"-kappa",&ctx.kappa,NULL);CHKERRQ(ierr);
  ctx.cahnhillard = PETSC_FALSE;
  ierr            = PetscOptionsGetBool(NULL,NULL,"-cahn-hillard",&ctx.cahnhillard,NULL);CHKERRQ(ierr);
  ierr            = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),2,vbounds);CHKERRQ(ierr);
  ierr            = PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),600,600);CHKERRQ(ierr);
  ctx.energy      = 1;
  /* ierr = PetscOptionsGetInt(NULL,NULL,"-energy",&ctx.energy,NULL);CHKERRQ(ierr); */
  ierr        = PetscOptionsGetInt(NULL,NULL,"-energy",&ctx.energy,NULL);CHKERRQ(ierr);
  ctx.tol     = 1.0e-8;
  ierr        = PetscOptionsGetReal(NULL,"-tol",&ctx.tol,NULL);CHKERRQ(ierr);
  ctx.theta   = .001;
  ctx.theta_c = 1.0;
  ierr        = PetscOptionsGetReal(NULL,"-theta",&ctx.theta,NULL);CHKERRQ(ierr);
  ierr        = PetscOptionsGetReal(NULL,"-theta_c",&ctx.theta_c,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, -10,2,2,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"Biharmonic heat equation: w = -kappa*u_xx");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"Biharmonic heat equation: u");CHKERRQ(ierr);
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
  ierr = TSSetIFunction(ts,NULL,FormFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,.02);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

<     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = DMCreateColoring(da,IS_COLORING_GLOBAL,&iscoloring);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))SNESTSFormFunction,ts);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(J,iscoloring,matfdcoloring);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,matfdcoloring);CHKERRQ(ierr);

  {
    ierr = VecDuplicate(x,&ul);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&uh);CHKERRQ(ierr);
    ierr = VecStrideSet(ul,0,PETSC_NINFINITY);CHKERRQ(ierr);
    ierr = VecStrideSet(ul,1,-1.0);CHKERRQ(ierr);
    ierr = VecStrideSet(uh,0,PETSC_INFINITY);CHKERRQ(ierr);
    ierr = VecStrideSet(uh,1,1.0);CHKERRQ(ierr);
    ierr = TSVISetVariableBounds(ts,ul,uh);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(da,x,ctx.kappa);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  wait = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-wait",&wait,NULL);CHKERRQ(ierr);
  if (wait) {
    ierr = PetscSleep(-1);CHKERRQ(ierr);
  }
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  {
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

typedef struct {PetscScalar w,u;} Field;
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
PetscErrorCode FormFunction(TS ts,PetscReal ftime,Vec X,Vec Xdot,Vec F,void *ptr)
{
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,Mx,xs,xm;
  PetscReal      hx,sx;
  PetscScalar    r,l;
  Field          *x,*xdot,*f;
  Vec            localX,localXdot;
  UserCtx        *ctx = (UserCtx*)ptr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localXdot);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,Xdot,INSERT_VALUES,localXdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,Xdot,INSERT_VALUES,localXdot);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localXdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    f[i].w =  x[i].w + ctx->kappa*(x[i-1].u + x[i+1].u - 2.0*x[i].u)*sx;
    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: /* double well */
        f[i].w += -x[i].u*x[i].u*x[i].u + x[i].u;
        break;
      case 2: /* double obstacle */
        f[i].w += x[i].u;
        break;
      case 3: /* logarithmic */
        if (x[i].u < -1.0 + 2.0*ctx->tol)      f[i].w += .5*ctx->theta*(-log(ctx->tol) + log((1.0-x[i].u)/2.0)) + ctx->theta_c*x[i].u;
        else if (x[i].u > 1.0 - 2.0*ctx->tol)  f[i].w += .5*ctx->theta*(-log((1.0+x[i].u)/2.0) + log(ctx->tol)) + ctx->theta_c*x[i].u;
        else                                   f[i].w += .5*ctx->theta*(-log((1.0+x[i].u)/2.0) + log((1.0-x[i].u)/2.0)) + ctx->theta_c*x[i].u;
        break;
      case 4:
        break;
      }
    }
    f[i].u = xdot[i].u - (x[i-1].w + x[i+1].w - 2.0*x[i].w)*sx;
    if (ctx->energy==4) {
      f[i].u = xdot[i].u;
      /* approximation of \grad (M(u) \grad w), where M(u) = (1-u^2) */
      r       = (1.0 - x[i+1].u*x[i+1].u)*(x[i+2].w-x[i].w)*.5/hx;
      l       = (1.0 - x[i-1].u*x[i-1].u)*(x[i].w-x[i-2].w)*.5/hx;
      f[i].u -= (r - l)*.5/hx;
      f[i].u += 2.0*ctx->theta_c*x[i].u*(x[i+1].u-x[i-1].u)*(x[i+1].u-x[i-1].u)*.25*sx - (ctx->theta - ctx->theta_c*(1-x[i].u*x[i].u))*(x[i+1].u + x[i-1].u - 2.0*x[i].u)*sx;
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localXdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localXdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec X,PetscReal kappa)
{
  PetscErrorCode ierr;
  PetscInt       i,xs,xm,Mx;
  Field          *x;
  PetscReal      hx,xx,r,sx;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)Mx;
  sx = 1.0/(hx*hx);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    xx = i*hx;
    r  = PetscSqrtScalar((xx-.5)*(xx-.5));
    if (r < .125) x[i].u = 1.0;
    else          x[i].u = -.50;
    /*  u[i] = PetscPowScalar(x - .5,4.0); */
  }
  for (i=xs; i<xs+xm; i++) x[i].w = -kappa*(x[i-1].u + x[i+1].u - 2.0*x[i].u)*sx;

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

