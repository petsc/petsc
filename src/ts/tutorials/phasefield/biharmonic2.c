
static char help[] = "Solves biharmonic equation in 1d.\n";

/*
  Solves the equation biharmonic equation in split form

    w = -kappa \Delta u
    u_t =  \Delta w
    -1  <= u <= 1
    Periodic boundary conditions

Evolve the biharmonic heat equation with bounds:  (same as biharmonic)
---------------
./biharmonic2 -ts_monitor -snes_monitor -ts_monitor_draw_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason  -ts_type beuler  -da_refine 5 -draw_fields 1 -ts_dt 9.53674e-9

    w = -kappa \Delta u  + u^3  - u
    u_t =  \Delta w
    -1  <= u <= 1
    Periodic boundary conditions

Evolve the Cahn-Hillard equations: (this fails after a few timesteps 12/17/2017)
---------------
./biharmonic2 -ts_monitor -snes_monitor -ts_monitor_draw_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason   -ts_type beuler    -da_refine 6  -draw_fields 1  -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard

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
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  ctx.kappa = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-kappa",&ctx.kappa,NULL));
  ctx.cahnhillard = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-cahn-hillard",&ctx.cahnhillard,NULL));
  PetscCall(PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),2,vbounds));
  PetscCall(PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),600,600));
  ctx.energy = 1;
  /*PetscCall(PetscOptionsGetInt(NULL,NULL,"-energy",&ctx.energy,NULL));*/
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-energy",&ctx.energy,NULL));
  ctx.tol     = 1.0e-8;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-tol",&ctx.tol,NULL));
  ctx.theta   = .001;
  ctx.theta_c = 1.0;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-theta",&ctx.theta,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-theta_c",&ctx.theta_c,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 10,2,2,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da,0,"Biharmonic heat equation: w = -kappa*u_xx"));
  PetscCall(DMDASetFieldName(da,1,"Biharmonic heat equation: u"));
  PetscCall(DMDAGetInfo(da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0));
  dt   = 1.0/(10.*ctx.kappa*Mx*Mx*Mx*Mx);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(VecDuplicate(x,&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetIFunction(ts,NULL,FormFunction,&ctx));
  PetscCall(TSSetMaxTime(ts,.02));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_INTERPOLATE));

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
  PetscCall(TSGetSNES(ts,&snes));
  PetscCall(DMCreateColoring(da,IS_COLORING_GLOBAL,&iscoloring));
  PetscCall(DMSetMatType(da,MATAIJ));
  PetscCall(DMCreateMatrix(da,&J));
  PetscCall(MatFDColoringCreate(J,iscoloring,&matfdcoloring));
  PetscCall(MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))SNESTSFormFunction,ts));
  PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
  PetscCall(MatFDColoringSetUp(J,iscoloring,matfdcoloring));
  PetscCall(ISColoringDestroy(&iscoloring));
  PetscCall(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,matfdcoloring));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetType(ts,TSBEULER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(da,x,ctx.kappa));
  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetSolution(ts,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,x));
  PetscCall(TSGetStepNumber(ts,&steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&J));
  PetscCall(MatFDColoringDestroy(&matfdcoloring));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
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
  Field          *x,*xdot,*f;
  Vec            localX,localXdot;
  UserCtx        *ctx = (UserCtx*)ptr;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGetLocalVector(da,&localX));
  PetscCall(DMGetLocalVector(da,&localXdot));
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)Mx; sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalBegin(da,Xdot,INSERT_VALUES,localXdot));
  PetscCall(DMGlobalToLocalEnd(da,Xdot,INSERT_VALUES,localXdot));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da,localX,&x));
  PetscCall(DMDAVecGetArrayRead(da,localXdot,&xdot));
  PetscCall(DMDAVecGetArray(da,F,&f));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

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
        if (PetscRealPart(x[i].u) < -1.0 + 2.0*ctx->tol)     f[i].w += .5*ctx->theta*(-PetscLogReal(ctx->tol) + PetscLogScalar((1.0-x[i].u)/2.0)) + ctx->theta_c*x[i].u;
        else if (PetscRealPart(x[i].u) > 1.0 - 2.0*ctx->tol) f[i].w += .5*ctx->theta*(-PetscLogScalar((1.0+x[i].u)/2.0) + PetscLogReal(ctx->tol)) + ctx->theta_c*x[i].u;
        else                                                 f[i].w += .5*ctx->theta*(-PetscLogScalar((1.0+x[i].u)/2.0) + PetscLogScalar((1.0-x[i].u)/2.0)) + ctx->theta_c*x[i].u;
        break;
      }
    }
    f[i].u = xdot[i].u - (x[i-1].w + x[i+1].w - 2.0*x[i].w)*sx;
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayRead(da,localXdot,&xdot));
  PetscCall(DMDAVecRestoreArrayRead(da,localX,&x));
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMRestoreLocalVector(da,&localX));
  PetscCall(DMRestoreLocalVector(da,&localXdot));
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
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)Mx;
  sx = 1.0/(hx*hx);

  /*
     Get pointers to vector data
  */
  PetscCall(DMCreateLocalVector(da,&Xg));
  PetscCall(DMDAVecGetArray(da,Xg,&x));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));
  PetscCall(DMDAGetGhostCorners(da,&xgs,NULL,NULL,&xgm,NULL,NULL));

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
  PetscCall(DMDAVecRestoreArray(da,Xg,&x));

  /* Grab only the global part of the vector */
  PetscCall(VecSet(X,0));
  PetscCall(DMLocalToGlobalBegin(da,Xg,ADD_VALUES,X));
  PetscCall(DMLocalToGlobalEnd(da,Xg,ADD_VALUES,X));
  PetscCall(VecDestroy(&Xg));
  PetscFunctionReturn(0);
}

/*TEST

   build:
     requires: !complex !single

   test:
     args: -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason  -ts_type beuler  -da_refine 5 -ts_dt 9.53674e-9 -ts_max_steps 50
     requires: x

TEST*/
