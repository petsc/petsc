
static char help[] = "Chemo-taxis Problems from Mathematical Biology.\n";

/*
     Page 18, Chemo-taxis Problems from Mathematical Biology

        rho_t =
        c_t   =

     Further discusson on Page 134 and in 2d on Page 409
*/

/*

   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscts.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef struct {
  PetscScalar rho,c;
} Field;

typedef struct {
  PetscScalar epsilon,delta,alpha,beta,gamma,kappa,lambda,mu,cstar;
  PetscBool   upwind;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*),InitialConditions(DM,Vec);

int main(int argc,char **argv)
{
  TS             ts;                  /* nonlinear solver */
  Vec            U;                   /* solution, residual vectors */
  DM             da;
  AppCtx         appctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  appctx.epsilon = 1.0e-3;
  appctx.delta   = 1.0;
  appctx.alpha   = 10.0;
  appctx.beta    = 4.0;
  appctx.gamma   = 1.0;
  appctx.kappa   = .75;
  appctx.lambda  = 1.0;
  appctx.mu      = 100.;
  appctx.cstar   = .2;
  appctx.upwind  = PETSC_TRUE;

  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-delta",&appctx.delta,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-upwind",&appctx.upwind,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,8,2,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da,0,"rho"));
  PetscCall(DMDASetFieldName(da,1,"c"));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da,&U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSROSW));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetIFunction(ts,NULL,IFunction,&appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(InitialConditions(da,U));
  PetscCall(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetTimeStep(ts,.0001));
  PetscCall(TSSetMaxTime(ts,1.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   IFunction - Evaluates nonlinear function, F(U).

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode IFunction(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscInt       i,Mx,xs,xm;
  PetscReal      hx,sx;
  PetscScalar    rho,c,rhoxx,cxx,cx,rhox,kcxrhox;
  Field          *u,*f,*udot;
  Vec            localU;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGetLocalVector(da,&localU));
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU));
  PetscCall(DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da,localU,&u));
  PetscCall(DMDAVecGetArrayRead(da,Udot,&udot));
  PetscCall(DMDAVecGetArrayWrite(da,F,&f));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  if (!xs) {
    f[0].rho = udot[0].rho; /* u[0].rho - 0.0; */
    f[0].c   = udot[0].c; /* u[0].c   - 1.0; */
    xs++;
    xm--;
  }
  if (xs+xm == Mx) {
    f[Mx-1].rho = udot[Mx-1].rho; /* u[Mx-1].rho - 1.0; */
    f[Mx-1].c   = udot[Mx-1].c;  /* u[Mx-1].c   - 0.0;  */
    xm--;
  }

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    rho   = u[i].rho;
    rhoxx = (-2.0*rho + u[i-1].rho + u[i+1].rho)*sx;
    c     = u[i].c;
    cxx   = (-2.0*c + u[i-1].c + u[i+1].c)*sx;

    if (!appctx->upwind) {
      rhox    = .5*(u[i+1].rho - u[i-1].rho)/hx;
      cx      = .5*(u[i+1].c - u[i-1].c)/hx;
      kcxrhox = appctx->kappa*(cxx*rho + cx*rhox);
    } else {
      kcxrhox = appctx->kappa*((u[i+1].c - u[i].c)*u[i+1].rho - (u[i].c - u[i-1].c)*u[i].rho)*sx;
    }

    f[i].rho = udot[i].rho - appctx->epsilon*rhoxx + kcxrhox  - appctx->mu*PetscAbsScalar(rho)*(1.0 - rho)*PetscMax(0,PetscRealPart(c - appctx->cstar)) + appctx->beta*rho;
    f[i].c   = udot[i].c - appctx->delta*cxx + appctx->lambda*c + appctx->alpha*rho*c/(appctx->gamma + c);
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayRead(da,localU,&u));
  PetscCall(DMDAVecRestoreArrayRead(da,Udot,&udot));
  PetscCall(DMDAVecRestoreArrayWrite(da,F,&f));
  PetscCall(DMRestoreLocalVector(da,&localU));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscInt       i,xs,xm,Mx;
  Field          *u;
  PetscReal      hx,x;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1);

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayWrite(da,U,&u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    if (i < Mx-1) u[i].rho = 0.0;
    else          u[i].rho = 1.0;
    u[i].c = PetscCosReal(.5*PETSC_PI*x);
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayWrite(da,U,&u));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -pc_type lu -da_refine 2  -ts_view  -ts_monitor -ts_max_time 1
      requires: double

   test:
     suffix: 2
     args:  -pc_type lu -da_refine 2  -ts_view  -ts_monitor_draw_solution -ts_monitor -ts_max_time 1
     requires: x double
     output_file: output/ex4_1.out

TEST*/
