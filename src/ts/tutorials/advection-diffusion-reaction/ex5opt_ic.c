static char help[] = "Demonstrates adjoint sensitivity analysis for Reaction-Diffusion Equations.\n";

/*
   See ex5.c for details on the equation.
   This code demonestrates the TSAdjoint/TAO interface to solve an inverse initial value problem built on a system of
   time-dependent partial differential equations.
   In this problem, the initial value for the PDE is unknown, but the output (the final solution of the PDE) is known.
   We want to determine the initial value that can produce the given output.
   We formulate the problem as a nonlinear optimization problem that minimizes the discrepency between the simulated
   result and given reference solution, calculate the gradient of the objective function with the discrete adjoint
   solver, and solve the optimization problem with TAO.

   Runtime options:
     -forwardonly  - run only the forward simulation
     -implicitform - provide IFunction and IJacobian to TS, if not set, RHSFunction and RHSJacobian will be used
 */

#include "reaction_diffusion.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petsctao.h>

/* User-defined routines */
extern PetscErrorCode FormFunctionAndGradient(Tao,Vec,PetscReal*,Vec,void*);

/*
   Set terminal condition for the adjoint variable
 */
PetscErrorCode InitializeLambda(DM da,Vec lambda,Vec U,AppCtx *appctx)
{
  char           filename[PETSC_MAX_PATH_LEN]="";
  PetscViewer    viewer;
  Vec            Uob;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(U,&Uob));
  CHKERRQ(PetscSNPrintf(filename,sizeof filename,"ex5opt.ob"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(VecLoad(Uob,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecAYPX(Uob,-1.,U));
  CHKERRQ(VecScale(Uob,2.0));
  CHKERRQ(VecAXPY(lambda,1.,Uob));
  CHKERRQ(VecDestroy(&Uob));
  PetscFunctionReturn(0);
}

/*
   Set up a viewer that dumps data into a binary file
 */
PetscErrorCode OutputBIN(DM da, const char *filename, PetscViewer *viewer)
{
  PetscFunctionBegin;
  CHKERRQ(PetscViewerCreate(PetscObjectComm((PetscObject)da),viewer));
  CHKERRQ(PetscViewerSetType(*viewer,PETSCVIEWERBINARY));
  CHKERRQ(PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE));
  CHKERRQ(PetscViewerFileSetName(*viewer,filename));
  PetscFunctionReturn(0);
}

/*
   Generate a reference solution and save it to a binary file
 */
PetscErrorCode GenerateOBs(TS ts,Vec U,AppCtx *appctx)
{
  char           filename[PETSC_MAX_PATH_LEN] = "";
  PetscViewer    viewer;
  DM             da;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(TSSolve(ts,U));
  CHKERRQ(PetscSNPrintf(filename,sizeof filename,"ex5opt.ob"));
  CHKERRQ(OutputBIN(da,filename,&viewer));
  CHKERRQ(VecView(U,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  CHKERRQ(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if ((1.0 <= x) && (x <= 1.5) && (1.0 <= y) && (y <= 1.5)) u[j][i].v = .25*PetscPowReal(PetscSinReal(4.0*PETSC_PI*x),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*y),2.0);
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbedInitialConditions(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  CHKERRQ(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if ((1.0 <= x) && (x <= 1.5) && (1.0 <= y) && (y <= 1.5)) u[j][i].v = .25*PetscPowReal(PetscSinReal(4.0*PETSC_PI*0.5*x),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*0.5*y),2.0);
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbedInitialConditions2(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  CHKERRQ(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if (PetscApproximateGTE(x,1.0) && PetscApproximateLTE(x,1.5) && PetscApproximateGTE(y,1.0) && PetscApproximateLTE(y,1.5)) u[j][i].v = PetscPowReal(PetscSinReal(4.0*PETSC_PI*(x-0.5)),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*y),2.0)/4.0;
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbedInitialConditions3(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  CHKERRQ(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if ((0.5 <= x) && (x <= 2.0) && (0.5 <= y) && (y <= 2.0)) u[j][i].v = .25*PetscPowReal(PetscSinReal(4.0*PETSC_PI*x),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*y),2.0);
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  PetscBool      forwardonly = PETSC_FALSE,implicitform = PETSC_FALSE;
  PetscInt       perturbic = 1;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-perturbic",&perturbic,NULL));

  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  appctx.aijpc = PETSC_FALSE;

  /* Create distributed array (DMDA) to manage parallel grid and vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,64,64,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetFieldName(da,0,"u"));
  CHKERRQ(DMDASetFieldName(da,1,"v"));

  /* Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types */
  CHKERRQ(DMCreateGlobalVector(da,&appctx.U));

  /* Create timestepping solver context */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&appctx.ts));
  CHKERRQ(TSSetType(appctx.ts,TSCN));
  CHKERRQ(TSSetDM(appctx.ts,da));
  CHKERRQ(TSSetProblemType(appctx.ts,TS_NONLINEAR));
  CHKERRQ(TSSetEquationType(appctx.ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (!implicitform) {
    CHKERRQ(TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx));
    CHKERRQ(TSSetRHSJacobian(appctx.ts,NULL,NULL,RHSJacobian,&appctx));
  } else {
    CHKERRQ(TSSetIFunction(appctx.ts,NULL,IFunction,&appctx));
    CHKERRQ(TSSetIJacobian(appctx.ts,NULL,NULL,IJacobian,&appctx));
  }

  /* Set initial conditions */
  CHKERRQ(InitialConditions(da,appctx.U));
  CHKERRQ(TSSetSolution(appctx.ts,appctx.U));

  /* Set solver options */
  CHKERRQ(TSSetMaxTime(appctx.ts,2000.0));
  CHKERRQ(TSSetTimeStep(appctx.ts,0.5));
  CHKERRQ(TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetFromOptions(appctx.ts));

  CHKERRQ(GenerateOBs(appctx.ts,appctx.U,&appctx));

  if (!forwardonly) {
    Tao           tao;
    Vec           P;
    Vec           lambda[1];
#if defined(PETSC_USE_LOG)
    PetscLogStage opt_stage;
#endif

    CHKERRQ(PetscLogStageRegister("Optimization",&opt_stage));
    CHKERRQ(PetscLogStagePush(opt_stage));
    if (perturbic == 1) {
      CHKERRQ(PerturbedInitialConditions(da,appctx.U));
    } else if (perturbic == 2) {
      CHKERRQ(PerturbedInitialConditions2(da,appctx.U));
    } else if (perturbic == 3) {
      CHKERRQ(PerturbedInitialConditions3(da,appctx.U));
    }

    CHKERRQ(VecDuplicate(appctx.U,&lambda[0]));
    CHKERRQ(TSSetCostGradients(appctx.ts,1,lambda,NULL));

    /* Have the TS save its trajectory needed by TSAdjointSolve() */
    CHKERRQ(TSSetSaveTrajectory(appctx.ts));

    /* Create TAO solver and set desired solution method */
    CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
    CHKERRQ(TaoSetType(tao,TAOBLMVM));

    /* Set initial guess for TAO */
    CHKERRQ(VecDuplicate(appctx.U,&P));
    CHKERRQ(VecCopy(appctx.U,P));
    CHKERRQ(TaoSetSolution(tao,P));

    /* Set routine for function and gradient evaluation */
    CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionAndGradient,&appctx));

    /* Check for any TAO command line options */
    CHKERRQ(TaoSetFromOptions(tao));

    CHKERRQ(TaoSolve(tao));
    CHKERRQ(TaoDestroy(&tao));
    CHKERRQ(VecDestroy(&lambda[0]));
    CHKERRQ(VecDestroy(&P));
    CHKERRQ(PetscLogStagePop());
  }

  /* Free work space.  All PETSc objects should be destroyed when they
     are no longer needed. */
  CHKERRQ(VecDestroy(&appctx.U));
  CHKERRQ(TSDestroy(&appctx.ts));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------ TAO callbacks ---------------------------- */

/*
   FormFunctionAndGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   P   - the input vector
   ctx - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionAndGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscReal      soberr,timestep;
  Vec            *lambda;
  Vec            SDiff;
  DM             da;
  char           filename[PETSC_MAX_PATH_LEN]="";
  PetscViewer    viewer;

  PetscFunctionBeginUser;
  CHKERRQ(TSSetTime(appctx->ts,0.0));
  CHKERRQ(TSGetTimeStep(appctx->ts,&timestep));
  if (timestep<0) {
    CHKERRQ(TSSetTimeStep(appctx->ts,-timestep));
  }
  CHKERRQ(TSSetStepNumber(appctx->ts,0));
  CHKERRQ(TSSetFromOptions(appctx->ts));

  CHKERRQ(VecDuplicate(P,&SDiff));
  CHKERRQ(VecCopy(P,appctx->U));
  CHKERRQ(TSGetDM(appctx->ts,&da));
  *f = 0;

  CHKERRQ(TSSolve(appctx->ts,appctx->U));
  CHKERRQ(PetscSNPrintf(filename,sizeof filename,"ex5opt.ob"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(VecLoad(SDiff,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecAYPX(SDiff,-1.,appctx->U));
  CHKERRQ(VecDot(SDiff,SDiff,&soberr));
  *f += soberr;

  CHKERRQ(TSGetCostGradients(appctx->ts,NULL,&lambda,NULL));
  CHKERRQ(VecSet(lambda[0],0.0));
  CHKERRQ(InitializeLambda(da,lambda[0],appctx->U,appctx));
  CHKERRQ(TSAdjointSolve(appctx->ts));

  CHKERRQ(VecCopy(lambda[0],G));

  CHKERRQ(VecDestroy(&SDiff));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      depends: reaction_diffusion.c
      requires: !complex !single

   test:
      args: -ts_max_steps 5 -ts_type rk -ts_rk_type 3 -ts_trajectory_type memory -tao_monitor -tao_view -tao_gatol 1e-6
      output_file: output/ex5opt_ic_1.out

TEST*/
