static char help[] = "Demonstrates adjoint sensitivity analysis for Reaction-Diffusion Equations.\n";

/*
  See ex5.c for details on the equation.
  This code demonestrates the TSAdjoint interface to a system of time-dependent partial differential equations.
  It computes the sensitivity of a component in the final solution, which locates in the center of the 2D domain, w.r.t. the initial conditions.
  The user does not need to provide any additional functions. The required functions in the original simulation are reused in the adjoint run.

  Runtime options:
    -forwardonly  - run the forward simulation without adjoint
    -implicitform - provide IFunction and IJacobian to TS, if not set, RHSFunction and RHSJacobian will be used
    -aijpc        - set the preconditioner matrix to be aij (the Jacobian matrix can be of a different type such as ELL)
*/

#include "reaction_diffusion.h"
#include <petscdm.h>
#include <petscdmda.h>

/* Matrix free support */
typedef struct {
  PetscReal time;
  Vec       U;
  Vec       Udot;
  PetscReal shift;
  AppCtx*   appctx;
  TS        ts;
} MCtx;

/*
   User-defined routines
*/
PetscErrorCode InitialConditions(DM,Vec);
PetscErrorCode RHSJacobianShell(TS,PetscReal,Vec,Mat,Mat,void*);
PetscErrorCode IJacobianShell(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y)
{
   PetscInt i,j,Mx,My,xs,ys,xm,ym;
   PetscErrorCode ierr;
   Field **l;
   PetscFunctionBegin;

   ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
   /* locate the global i index for x and j index for y */
   i = (PetscInt)(x*(Mx-1));
   j = (PetscInt)(y*(My-1));
   ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

   if (xs <= i && i < xs+xm && ys <= j && j < ys+ym) {
     /* the i,j vertex is on this process */
     ierr = DMDAVecGetArray(da,lambda,&l);CHKERRQ(ierr);
     l[j][i].u = 1.0;
     l[j][i].v = 1.0;
     ierr = DMDAVecRestoreArray(da,lambda,&l);CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode MyRHSMatMultTranspose(Mat A_shell,Vec X,Vec Y)
{
  MCtx           *mctx;
  AppCtx         *appctx;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy,ucb,vcb;
  Field          **u,**x,**y;
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,&mctx);CHKERRQ(ierr);
  appctx = mctx->appctx;
  ierr = TSGetDM(mctx->ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)My; sy = 1.0/(hy*hy);

  /* Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition. */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,mctx->U,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Y,&y);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      ucb       = x[j][i].u;
      uxx       = (-2.0*ucb + x[j][i-1].u + x[j][i+1].u)*sx;
      uyy       = (-2.0*ucb + x[j-1][i].u + x[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vcb       = x[j][i].v;
      vxx       = (-2.0*vcb + x[j][i-1].v + x[j][i+1].v)*sx;
      vyy       = (-2.0*vcb + x[j-1][i].v + x[j+1][i].v)*sy;
      y[j][i].u = appctx->D1*(uxx + uyy) - ucb*(vc*vc+appctx->gamma) + vc*vc*vcb;
      y[j][i].v = appctx->D2*(vxx + vyy) - 2.0*uc*vc*ucb + (2.0*uc*vc-appctx->gamma - appctx->kappa)*vcb;
    }
  }
  ierr = DMDAVecRestoreArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,mctx->U,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Y,&y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyIMatMultTranspose(Mat A_shell,Vec X,Vec Y)
{
  MCtx           *mctx;
  AppCtx         *appctx;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy,ucb,vcb;
  Field          **u,**x,**y;
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,&mctx);CHKERRQ(ierr);
  appctx = mctx->appctx;
  ierr = TSGetDM(mctx->ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)My; sy = 1.0/(hy*hy);

  /* Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition. */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,mctx->U,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Y,&y);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      ucb       = x[j][i].u;
      uxx       = (-2.0*ucb + x[j][i-1].u + x[j][i+1].u)*sx;
      uyy       = (-2.0*ucb + x[j-1][i].u + x[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vcb       = x[j][i].v;
      vxx       = (-2.0*vcb + x[j][i-1].v + x[j][i+1].v)*sx;
      vyy       = (-2.0*vcb + x[j-1][i].v + x[j+1][i].v)*sy;
      y[j][i].u = appctx->D1*(uxx + uyy) - ucb*(vc*vc+appctx->gamma) + vc*vc*vcb;
      y[j][i].v = appctx->D2*(vxx + vyy) - 2.0*uc*vc*ucb + (2.0*uc*vc-appctx->gamma - appctx->kappa)*vcb;
      y[j][i].u = mctx->shift*ucb - y[j][i].u;
      y[j][i].v = mctx->shift*vcb - y[j][i].v;
    }
  }
  ierr = DMDAVecRestoreArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,mctx->U,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Y,&y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyIMatMult(Mat A_shell,Vec X,Vec Y)
{
  MCtx           *mctx;
  AppCtx         *appctx;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy,ucb,vcb;
  Field          **u,**x,**y;
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,&mctx);CHKERRQ(ierr);
  appctx = mctx->appctx;
  ierr = TSGetDM(mctx->ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)My; sy = 1.0/(hy*hy);

  /* Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition. */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,mctx->U,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Y,&y);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      ucb       = x[j][i].u;
      uxx       = (-2.0*ucb + x[j][i-1].u + x[j][i+1].u)*sx;
      uyy       = (-2.0*ucb + x[j-1][i].u + x[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vcb       = x[j][i].v;
      vxx       = (-2.0*vcb + x[j][i-1].v + x[j][i+1].v)*sx;
      vyy       = (-2.0*vcb + x[j-1][i].v + x[j+1][i].v)*sy;
      y[j][i].u = appctx->D1*(uxx + uyy) - (vc*vc+appctx->gamma)*ucb - 2.0*uc*vc*vcb;
      y[j][i].v = appctx->D2*(vxx + vyy) + vc*vc*ucb + (2.0*uc*vc-appctx->gamma-appctx->kappa)*vcb;
      y[j][i].u = mctx->shift*ucb - y[j][i].u;
      y[j][i].v = mctx->shift*vcb - y[j][i].v;
    }
  }
  ierr = DMDAVecRestoreArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,mctx->U,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Y,&y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x;                   /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  MCtx           mctx;
  Vec            lambda[1];
  PetscBool      forwardonly=PETSC_FALSE,implicitform=PETSC_TRUE,mf = PETSC_FALSE;
  PetscLogDouble v1,v2;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL);CHKERRQ(ierr);
  appctx.aijpc = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-aijpc",&appctx.aijpc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-mf",&mf,NULL);CHKERRQ(ierr);

  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;

  PetscLogStageRegister("MyAdjoint", &stage);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,64,64,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_ODE_EXPLICIT);CHKERRQ(ierr); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (!implicitform) {
    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,&appctx);CHKERRQ(ierr);
  } else {
    ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
    ierr = TSSetIFunction(ts,NULL,IFunction,&appctx);CHKERRQ(ierr);
    if (appctx.aijpc) {
      Mat                    A,B;

      ierr = DMSetMatType(da,MATSELL);CHKERRQ(ierr);
      ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
      ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      /* FIXME do we need to change viewer to display matrix in natural ordering as DMCreateMatrix_DA does? */
      ierr = TSSetIJacobian(ts,A,B,IJacobian,&appctx);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
    } else {
      ierr = TSSetIJacobian(ts,NULL,NULL,IJacobian,&appctx);CHKERRQ(ierr);
    }
  }

  if (mf) {
    PetscInt xm,ym,Mx,My,dof;
    mctx.ts = ts;
    mctx.appctx = &appctx;
    ierr = VecDuplicate(x,&mctx.U);CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Create matrix free context
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&dof,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(da,NULL,NULL,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,dof*xm*ym,PETSC_DETERMINE,dof*Mx*My,dof*Mx*My,&mctx,&appctx.A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(appctx.A,MATOP_MULT_TRANSPOSE,(void (*)(void))MyRHSMatMultTranspose);CHKERRQ(ierr);
    if (!implicitform) { /* for explicit methods only */
      ierr = TSSetRHSJacobian(ts,appctx.A,appctx.A,RHSJacobianShell,&appctx);CHKERRQ(ierr);
    } else {
      /* ierr = VecDuplicate(appctx.U,&mctx.Udot);CHKERRQ(ierr); */
      ierr = MatShellSetOperation(appctx.A,MATOP_MULT,(void (*)(void))MyIMatMult);CHKERRQ(ierr);
      ierr = MatShellSetOperation(appctx.A,MATOP_MULT_TRANSPOSE,(void (*)(void))MyIMatMultTranspose);CHKERRQ(ierr);
      ierr = TSSetIJacobian(ts,appctx.A,appctx.A,IJacobianShell,&appctx);CHKERRQ(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /*
    Have the TS save its trajectory so that TSAdjointSolve() may be used
  */
  if (!forwardonly) { ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr); }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,200.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = PetscTime(&v1);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  if (!forwardonly) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Start the Adjoint model
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = VecDuplicate(x,&lambda[0]);CHKERRQ(ierr);
    /*   Reset initial conditions for the adjoint integration */
    ierr = InitializeLambda(da,lambda[0],0.5,0.5);CHKERRQ(ierr);
    ierr = TSSetCostGradients(ts,1,lambda,NULL);CHKERRQ(ierr);
    PetscLogStagePush(stage);
    ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
    PetscLogStagePop();
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  }
  ierr = PetscTime(&v2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," %.3lf ",v2-v1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  if (mf) {
    ierr = VecDestroy(&mctx.U);CHKERRQ(ierr);
    /* ierr = VecDestroy(&mctx.Udot);CHKERRQ(ierr);*/
    ierr = MatDestroy(&appctx.A);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */
PetscErrorCode RHSJacobianShell(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ctx)
{
  MCtx           *mctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&mctx);CHKERRQ(ierr);
  ierr = VecCopy(U,mctx->U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobianShell(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat BB,void *ctx)
{
  MCtx           *mctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&mctx);CHKERRQ(ierr);
  ierr = VecCopy(U,mctx->U);CHKERRQ(ierr);
  /* ierr = VecCopy(Udot,mctx->Udot);CHKERRQ(ierr); */
  mctx->shift = a;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if ((1.0 <= x) && (x <= 1.5) && (1.0 <= y) && (y <= 1.5)) u[j][i].v = .25*PetscPowReal(PetscSinReal(4.0*PETSC_PI*x),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*y),2.0);
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      depends: reaction_diffusion.c
      requires: !complex !single

   test:
      requires: cams
      args: -mf -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -ts_trajectory_type memory -ts_trajectory_solution_only 0 -ts_trajectory_max_units_ram 6 -ts_trajectory_memory_type cams
TEST*/
