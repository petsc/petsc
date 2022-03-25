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
  PetscCall(VecDuplicate(U,&Uob));
  PetscCall(PetscSNPrintf(filename,sizeof filename,"ex5opt.ob"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  PetscCall(VecLoad(Uob,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecAYPX(Uob,-1.,U));
  PetscCall(VecScale(Uob,2.0));
  PetscCall(VecAXPY(lambda,1.,Uob));
  PetscCall(VecDestroy(&Uob));
  PetscFunctionReturn(0);
}

/*
   Set up a viewer that dumps data into a binary file
 */
PetscErrorCode OutputBIN(DM da, const char *filename, PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)da),viewer));
  PetscCall(PetscViewerSetType(*viewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(*viewer,filename));
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
  PetscCall(TSGetDM(ts,&da));
  PetscCall(TSSolve(ts,U));
  PetscCall(PetscSNPrintf(filename,sizeof filename,"ex5opt.ob"));
  PetscCall(OutputBIN(da,filename,&viewer));
  PetscCall(VecView(U,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  PetscCall(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

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

  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbedInitialConditions(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  PetscCall(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

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

  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbedInitialConditions2(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  PetscCall(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

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

  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbedInitialConditions3(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  PetscCall(DMDAVecGetArray(da,U,&u));
  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

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

  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DM             da;
  AppCtx         appctx;
  PetscBool      forwardonly = PETSC_FALSE,implicitform = PETSC_FALSE;
  PetscInt       perturbic = 1;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-perturbic",&perturbic,NULL));

  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  appctx.aijpc = PETSC_FALSE;

  /* Create distributed array (DMDA) to manage parallel grid and vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,64,64,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da,0,"u"));
  PetscCall(DMDASetFieldName(da,1,"v"));

  /* Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types */
  PetscCall(DMCreateGlobalVector(da,&appctx.U));

  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&appctx.ts));
  PetscCall(TSSetType(appctx.ts,TSCN));
  PetscCall(TSSetDM(appctx.ts,da));
  PetscCall(TSSetProblemType(appctx.ts,TS_NONLINEAR));
  PetscCall(TSSetEquationType(appctx.ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (!implicitform) {
    PetscCall(TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx));
    PetscCall(TSSetRHSJacobian(appctx.ts,NULL,NULL,RHSJacobian,&appctx));
  } else {
    PetscCall(TSSetIFunction(appctx.ts,NULL,IFunction,&appctx));
    PetscCall(TSSetIJacobian(appctx.ts,NULL,NULL,IJacobian,&appctx));
  }

  /* Set initial conditions */
  PetscCall(InitialConditions(da,appctx.U));
  PetscCall(TSSetSolution(appctx.ts,appctx.U));

  /* Set solver options */
  PetscCall(TSSetMaxTime(appctx.ts,2000.0));
  PetscCall(TSSetTimeStep(appctx.ts,0.5));
  PetscCall(TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(appctx.ts));

  PetscCall(GenerateOBs(appctx.ts,appctx.U,&appctx));

  if (!forwardonly) {
    Tao           tao;
    Vec           P;
    Vec           lambda[1];
#if defined(PETSC_USE_LOG)
    PetscLogStage opt_stage;
#endif

    PetscCall(PetscLogStageRegister("Optimization",&opt_stage));
    PetscCall(PetscLogStagePush(opt_stage));
    if (perturbic == 1) {
      PetscCall(PerturbedInitialConditions(da,appctx.U));
    } else if (perturbic == 2) {
      PetscCall(PerturbedInitialConditions2(da,appctx.U));
    } else if (perturbic == 3) {
      PetscCall(PerturbedInitialConditions3(da,appctx.U));
    }

    PetscCall(VecDuplicate(appctx.U,&lambda[0]));
    PetscCall(TSSetCostGradients(appctx.ts,1,lambda,NULL));

    /* Have the TS save its trajectory needed by TSAdjointSolve() */
    PetscCall(TSSetSaveTrajectory(appctx.ts));

    /* Create TAO solver and set desired solution method */
    PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
    PetscCall(TaoSetType(tao,TAOBLMVM));

    /* Set initial guess for TAO */
    PetscCall(VecDuplicate(appctx.U,&P));
    PetscCall(VecCopy(appctx.U,P));
    PetscCall(TaoSetSolution(tao,P));

    /* Set routine for function and gradient evaluation */
    PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionAndGradient,&appctx));

    /* Check for any TAO command line options */
    PetscCall(TaoSetFromOptions(tao));

    PetscCall(TaoSolve(tao));
    PetscCall(TaoDestroy(&tao));
    PetscCall(VecDestroy(&lambda[0]));
    PetscCall(VecDestroy(&P));
    PetscCall(PetscLogStagePop());
  }

  /* Free work space.  All PETSc objects should be destroyed when they
     are no longer needed. */
  PetscCall(VecDestroy(&appctx.U));
  PetscCall(TSDestroy(&appctx.ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
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
  PetscCall(TSSetTime(appctx->ts,0.0));
  PetscCall(TSGetTimeStep(appctx->ts,&timestep));
  if (timestep<0) {
    PetscCall(TSSetTimeStep(appctx->ts,-timestep));
  }
  PetscCall(TSSetStepNumber(appctx->ts,0));
  PetscCall(TSSetFromOptions(appctx->ts));

  PetscCall(VecDuplicate(P,&SDiff));
  PetscCall(VecCopy(P,appctx->U));
  PetscCall(TSGetDM(appctx->ts,&da));
  *f = 0;

  PetscCall(TSSolve(appctx->ts,appctx->U));
  PetscCall(PetscSNPrintf(filename,sizeof filename,"ex5opt.ob"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  PetscCall(VecLoad(SDiff,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecAYPX(SDiff,-1.,appctx->U));
  PetscCall(VecDot(SDiff,SDiff,&soberr));
  *f += soberr;

  PetscCall(TSGetCostGradients(appctx->ts,NULL,&lambda,NULL));
  PetscCall(VecSet(lambda[0],0.0));
  PetscCall(InitializeLambda(da,lambda[0],appctx->U,appctx));
  PetscCall(TSAdjointSolve(appctx->ts));

  PetscCall(VecCopy(lambda[0],G));

  PetscCall(VecDestroy(&SDiff));
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
