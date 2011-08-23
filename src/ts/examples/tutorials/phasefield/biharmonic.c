
static char help[] = "Solves biharmonic equation in 1d.\n";

/*
  Solves the equation

    u_t = - kappa  \Delta \Delta u 
    Periodic boundary conditions

Evolve the biharmonic heat equation: 
---------------
./biharmonic -ts_monitor -snes_monitor -ts_monitor_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason -ts_monitor_solution_initial -wait   -ts_type beuler  -da_refine 5

Evolve with the restriction that 0 <= u <= 1; i.e. as a variational inequality
---------------
./biharmonic -ts_monitor -snes_monitor -ts_monitor_solution  -pc_type lu  -draw_pause .1 -snes_converged_reason -ts_monitor_solution_initial -wait   -ts_type beuler   -da_refine 5 -vi 

Add a term to the right hand side so that the bulge grows/expands with time
---------------
./biharmonic -ts_monitor -snes_monitor -ts_monitor_solution -pc_type lu  -draw_pause .1  -snes_converged_reason  -ts_monitor_solution_initial -wait   -ts_type beuler   -da_refine 5 -vi -growth


*/
#include <petscdmda.h>
#include <petscts.h>

extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,void*),FormInitialSolution(DM,Vec);
typedef struct {PetscBool growth;PetscBool cahnhillard;PetscReal kappa;} UserCtx;

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
  PetscReal              vbounds[] = {-.125,1.25};
  PetscBool              wait,vi = PETSC_FALSE;
  Vec                    ul,uh;
  UserCtx                ctx;
  
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
  ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1,vbounds);CHKERRQ(ierr); 
  ierr = PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),600,600);CHKERRQ(ierr); 

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
    ierr = VecSet(ul,0.0);CHKERRQ(ierr);
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
      f[i] += 12.*(2.*x[i]-1.)*(x[i+1] - x[i-1])*(x[i+1] - x[i-1])/(4.*hx*hx) + (12.*x[i]*x[i] - 12.*x[i]+2.)*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
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
      u[i] = 0.25; //01;
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

