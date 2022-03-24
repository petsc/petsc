
static char help[] = "Solves biharmonic equation in 1d.\n";

/*
  Solves the equation biharmonic equation in split form

    w = -kappa \Delta u
    u_t =  \Delta w
    -1  <= u <= 1
    Periodic boundary conditions

Evolve the biharmonic heat equation with bounds:  (same as biharmonic)
---------------
./biharmonic3 -ts_monitor -snes_monitor -ts_monitor_draw_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason -ts_type beuler  -da_refine 5 -draw_fields 1 -ts_dt 9.53674e-9

    w = -kappa \Delta u  + u^3  - u
    u_t =  \Delta w
    -1  <= u <= 1
    Periodic boundary conditions

Evolve the Cahn-Hillard equations:
---------------
./biharmonic3 -ts_monitor -snes_monitor -ts_monitor_draw_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason  -ts_type beuler    -da_refine 6  -draw_fields 1  -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard

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
  PetscInt       steps,Mx;
  DM             da;
  MatFDColoring  matfdcoloring;
  ISColoring     iscoloring;
  PetscReal      dt;
  PetscReal      vbounds[] = {-100000,100000,-1.1,1.1};
  SNES           snes;
  UserCtx        ctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  ctx.kappa       = 1.0;
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-kappa",&ctx.kappa,NULL));
  ctx.cahnhillard = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-cahn-hillard",&ctx.cahnhillard,NULL));
  CHKERRQ(PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),2,vbounds));
  CHKERRQ(PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),600,600));
  ctx.energy      = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-energy",&ctx.energy,NULL));
  ctx.tol     = 1.0e-8;
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tol",&ctx.tol,NULL));
  ctx.theta   = .001;
  ctx.theta_c = 1.0;
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-theta",&ctx.theta,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-theta_c",&ctx.theta_c,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 10,2,2,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetFieldName(da,0,"Biharmonic heat equation: w = -kappa*u_xx"));
  CHKERRQ(DMDASetFieldName(da,1,"Biharmonic heat equation: u"));
  CHKERRQ(DMDAGetInfo(da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0));
  dt   = 1.0/(10.*ctx.kappa*Mx*Mx*Mx*Mx);

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
  CHKERRQ(TSSetIFunction(ts,NULL,FormFunction,&ctx));
  CHKERRQ(TSSetMaxTime(ts,.02));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

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
  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(SNESSetType(snes,SNESVINEWTONRSLS));
  CHKERRQ(DMCreateColoring(da,IS_COLORING_GLOBAL,&iscoloring));
  CHKERRQ(DMSetMatType(da,MATAIJ));
  CHKERRQ(DMCreateMatrix(da,&J));
  CHKERRQ(MatFDColoringCreate(J,iscoloring,&matfdcoloring));
  CHKERRQ(MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))SNESTSFormFunction,ts));
  CHKERRQ(MatFDColoringSetFromOptions(matfdcoloring));
  CHKERRQ(MatFDColoringSetUp(J,iscoloring,matfdcoloring));
  CHKERRQ(ISColoringDestroy(&iscoloring));
  CHKERRQ(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,matfdcoloring));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetType(ts,TSBEULER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(da,x,ctx.kappa));
  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSSetSolution(ts,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));
  CHKERRQ(TSGetStepNumber(ts,&steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(MatFDColoringDestroy(&matfdcoloring));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&da));

  CHKERRQ(PetscFinalize());
  return 0;
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
  PetscInt       i,Mx,xs,xm;
  PetscReal      hx,sx;
  PetscScalar    r,l;
  Field          *x,*xdot,*f;
  Vec            localX,localXdot;
  UserCtx        *ctx = (UserCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMGetLocalVector(da,&localX));
  CHKERRQ(DMGetLocalVector(da,&localXdot));
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
  CHKERRQ(DMGlobalToLocalBegin(da,Xdot,INSERT_VALUES,localXdot));
  CHKERRQ(DMGlobalToLocalEnd(da,Xdot,INSERT_VALUES,localXdot));

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMDAVecGetArrayRead(da,localX,&x));
  CHKERRQ(DMDAVecGetArrayRead(da,localXdot,&xdot));
  CHKERRQ(DMDAVecGetArray(da,F,&f));

  /*
     Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

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
        if (x[i].u < -1.0 + 2.0*ctx->tol)      f[i].w += .5*ctx->theta*(-PetscLogScalar(ctx->tol) + PetscLogScalar((1.0-x[i].u)/2.0)) + ctx->theta_c*x[i].u;
        else if (x[i].u > 1.0 - 2.0*ctx->tol)  f[i].w += .5*ctx->theta*(-PetscLogScalar((1.0+x[i].u)/2.0) + PetscLogScalar(ctx->tol)) + ctx->theta_c*x[i].u;
        else                                   f[i].w += .5*ctx->theta*(-PetscLogScalar((1.0+x[i].u)/2.0) + PetscLogScalar((1.0-x[i].u)/2.0)) + ctx->theta_c*x[i].u;
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
  CHKERRQ(DMDAVecRestoreArrayRead(da,localXdot,&xdot));
  CHKERRQ(DMDAVecRestoreArrayRead(da,localX,&x));
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  CHKERRQ(DMRestoreLocalVector(da,&localX));
  CHKERRQ(DMRestoreLocalVector(da,&localXdot));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec X,PetscReal kappa)
{
  PetscInt       i,xs,xm,Mx,xgs,xgm;
  Field          *x;
  PetscReal      hx,xx,r,sx;
  Vec            Xg;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)Mx;
  sx = 1.0/(hx*hx);

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMCreateLocalVector(da,&Xg));
  CHKERRQ(DMDAVecGetArray(da,Xg,&x));

  /*
     Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));
  CHKERRQ(DMDAGetGhostCorners(da,&xgs,NULL,NULL,&xgm,NULL,NULL));

  /*
     Compute u function over the locally owned part of the grid including ghost points
  */
  for (i=xgs; i<xgs+xgm; i++) {
    xx = i*hx;
    r = PetscSqrtReal((xx-.5)*(xx-.5));
    if (r < .125) x[i].u = 1.0;
    else          x[i].u = -.50;
    /* fill in x[i].w so that valgrind doesn't detect use of uninitialized memory */
    x[i].w = 0;
  }
  for (i=xs; i<xs+xm; i++) x[i].w = -kappa*(x[i-1].u + x[i+1].u - 2.0*x[i].u)*sx;

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArray(da,Xg,&x));

  /* Grab only the global part of the vector */
  CHKERRQ(VecSet(X,0));
  CHKERRQ(DMLocalToGlobalBegin(da,Xg,ADD_VALUES,X));
  CHKERRQ(DMLocalToGlobalEnd(da,Xg,ADD_VALUES,X));
  CHKERRQ(VecDestroy(&Xg));
  PetscFunctionReturn(0);
}

/*TEST

   build:
     requires: !complex !single

   test:
     args: -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason  -ts_type beuler  -da_refine 5 -ts_dt 9.53674e-9 -ts_max_steps 50
     requires: x

TEST*/
