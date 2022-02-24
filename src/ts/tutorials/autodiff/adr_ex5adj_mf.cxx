static char help[] = "Demonstrates automatic, matrix-free Jacobian generation using ADOL-C for a time-dependent PDE in 2d, solved using implicit timestepping.\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: Automatic differentiation using ADOL-C
   Concepts: Matrix-free methods
*/
/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/
/* ------------------------------------------------------------------------
  See ../advection-diffusion-reaction/ex5 for a description of the problem
  ------------------------------------------------------------------------- */

#include <petscdmda.h>
#include <petscts.h>
#include "adolc-utils/init.cxx"
#include "adolc-utils/matfree.cxx"
#include <adolc/adolc.h>

/* (Passive) field for the two variables */
typedef struct {
  PetscScalar u,v;
} Field;

/* Active field for the two variables */
typedef struct {
  adouble u,v;
} AField;

/* Application context */
typedef struct {
  PetscReal D1,D2,gamma,kappa;
  AField    **u_a,**f_a;
  AdolcCtx  *adctx; /* Automatic differentation support */
} AppCtx;

extern PetscErrorCode InitialConditions(DM da,Vec U);
extern PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y);
extern PetscErrorCode IFunctionLocalPassive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr);
extern PetscErrorCode IFunctionActive(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr);
extern PetscErrorCode IJacobianMatFree(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A_shell,Mat B,void *ctx);

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x,r;                 /* solution, residual */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;              /* Application context */
  AdolcMatCtx    matctx;              /* Matrix (free) context */
  Vec            lambda[1];
  PetscBool      forwardonly=PETSC_FALSE;
  Mat            A;                   /* (Matrix free) Jacobian matrix */
  PetscInt       gxm,gym;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL));
  PetscFunctionBeginUser;
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  CHKERRQ(PetscLogEventRegister("df/dx forward",MAT_CLASSID,&matctx.event1));
  CHKERRQ(PetscLogEventRegister("df/d(xdot) forward",MAT_CLASSID,&matctx.event2));
  CHKERRQ(PetscLogEventRegister("df/dx reverse",MAT_CLASSID,&matctx.event3));
  CHKERRQ(PetscLogEventRegister("df/d(xdot) reverse",MAT_CLASSID,&matctx.event4));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,65,65,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetFieldName(da,0,"u"));
  CHKERRQ(DMDASetFieldName(da,1,"v"));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(VecDuplicate(x,&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create matrix free context and specify usage of PETSc-ADOL-C drivers
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMSetMatType(da,MATSHELL));
  CHKERRQ(DMCreateMatrix(da,&A));
  CHKERRQ(MatShellSetContext(A,&matctx));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT,(void (*)(void))PetscAdolcIJacobianVectorProductIDMass));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void (*)(void))PetscAdolcIJacobianTransposeVectorProductIDMass));
  CHKERRQ(VecDuplicate(x,&matctx.X));
  CHKERRQ(VecDuplicate(x,&matctx.Xdot));
  CHKERRQ(DMGetLocalVector(da,&matctx.localX0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSCN));
  CHKERRQ(TSSetDM(ts,da));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalPassive,&appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Some data required for matrix-free context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDAGetGhostCorners(da,NULL,NULL,NULL,&gxm,&gym,NULL));
  matctx.m = 2*gxm*gym;matctx.n = 2*gxm*gym; /* Number of dependent and independent variables */
  matctx.flg = PETSC_FALSE;                  /* Flag for reverse mode */
  matctx.tag1 = 1;                           /* Tape identifier */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace function just once
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscNew(&appctx.adctx));
  CHKERRQ(IFunctionActive(ts,1.,x,matctx.Xdot,r,&appctx));
  CHKERRQ(PetscFree(appctx.adctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian. In this case, IJacobian simply acts to pass context
     information to the matrix-free Jacobian vector product.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetIJacobian(ts,A,A,IJacobianMatFree,&appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(InitialConditions(da,x));
  CHKERRQ(TSSetSolution(ts,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Have the TS save its trajectory so that TSAdjointSolve() may be used
    and set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!forwardonly) {
    CHKERRQ(TSSetSaveTrajectory(ts));
    CHKERRQ(TSSetMaxTime(ts,200.0));
    CHKERRQ(TSSetTimeStep(ts,0.5));
  } else {
    CHKERRQ(TSSetMaxTime(ts,2000.0));
    CHKERRQ(TSSetTimeStep(ts,10));
  }
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));
  if (!forwardonly) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Start the Adjoint model
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    CHKERRQ(VecDuplicate(x,&lambda[0]));
    /*   Reset initial conditions for the adjoint integration */
    CHKERRQ(InitializeLambda(da,lambda[0],0.5,0.5));
    CHKERRQ(TSSetCostGradients(ts,1,lambda,NULL));
    CHKERRQ(TSAdjointSolve(ts));
    CHKERRQ(VecDestroy(&lambda[0]));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMRestoreLocalVector(da,&matctx.localX0));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&matctx.X));
  CHKERRQ(VecDestroy(&matctx.Xdot));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&da));

  ierr = PetscFinalize();
  return ierr;
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

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMDAVecGetArray(da,U,&u));

  /*
     Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if (PetscApproximateGTE(x,1.0) && PetscApproximateLTE(x,1.5) && PetscApproximateGTE(y,1.0) && PetscApproximateLTE(y,1.5)) u[j][i].v = PetscPowReal(PetscSinReal(4.0*PETSC_PI*x),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*y),2.0)/4.0;
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y)
{
   PetscInt i,j,Mx,My,xs,ys,xm,ym;
   Field **l;

   PetscFunctionBegin;
   CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
   /* locate the global i index for x and j index for y */
   i = (PetscInt)(x*(Mx-1));
   j = (PetscInt)(y*(My-1));
   CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

   if (xs <= i && i < xs+xm && ys <= j && j < ys+ym) {
     /* the i,j vertex is on this process */
     CHKERRQ(DMDAVecGetArray(da,lambda,&l));
     l[j][i].u = 1.0;
     l[j][i].v = 1.0;
     CHKERRQ(DMDAVecRestoreArray(da,lambda,&l));
   }
   PetscFunctionReturn(0);
}

PetscErrorCode IFunctionLocalPassive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); sy = 1.0/(hy*hy);

  /* Get local grid boundaries */
  xs = info->xs; xm = info->xm; ys = info->ys; ym = info->ym;

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
      f[j][i].u = udot[j][i].u - appctx->D1*(uxx + uyy) + uc*vc*vc - appctx->gamma*(1.0 - uc);
      f[j][i].v = udot[j][i].v - appctx->D2*(vxx + vyy) - uc*vc*vc + (appctx->gamma + appctx->kappa)*vc;
    }
  }
  CHKERRQ(PetscLogFlops(16.0*xm*ym));
  PetscFunctionReturn(0);
}

PetscErrorCode IFunctionActive(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  DMDALocalInfo  info;
  Field          **u,**f,**udot;
  Vec            localU;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      hx,hy,sx,sy;
  adouble        uc,uxx,uyy,vc,vxx,vyy;
  AField         **f_a,*f_c,**u_a,*u_c;
  PetscScalar    dummy;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  CHKERRQ(DMGetLocalVector(da,&localU));
  hx = 2.50/(PetscReal)(info.mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info.my); sy = 1.0/(hy*hy);
  xs = info.xs; xm = info.xm; gxs = info.gxs; gxm = info.gxm;
  ys = info.ys; ym = info.ym; gys = info.gys; gym = info.gym;

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  CHKERRQ(DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU));
  CHKERRQ(DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU));

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMDAVecGetArrayRead(da,localU,&u));
  CHKERRQ(DMDAVecGetArray(da,F,&f));
  CHKERRQ(DMDAVecGetArrayRead(da,Udot,&udot));

  /*
    Create contiguous 1-arrays of AFields

    NOTE: Memory for ADOL-C active variables (such as adouble and AField)
          cannot be allocated using PetscMalloc, as this does not call the
          relevant class constructor. Instead, we use the C++ keyword `new`.
  */
  u_c = new AField[info.gxm*info.gym];
  f_c = new AField[info.gxm*info.gym];

  /* Create corresponding 2-arrays of AFields */
  u_a = new AField*[info.gym];
  f_a = new AField*[info.gym];

  /* Align indices between array types to endow 2d array with ghost points */
  CHKERRQ(GiveGhostPoints(da,u_c,&u_a));
  CHKERRQ(GiveGhostPoints(da,f_c,&f_a));

  trace_on(1);  /* Start of active section on tape 1 */

  /*
    Mark independence

    NOTE: Ghost points are marked as independent, in place of the points they represent on
          other processors / on other boundaries.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_a[j][i].u <<= u[j][i].u;
      u_a[j][i].v <<= u[j][i].v;
    }
  }

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u_a[j][i].u;
      uxx       = (-2.0*uc + u_a[j][i-1].u + u_a[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u_a[j-1][i].u + u_a[j+1][i].u)*sy;
      vc        = u_a[j][i].v;
      vxx       = (-2.0*vc + u_a[j][i-1].v + u_a[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u_a[j-1][i].v + u_a[j+1][i].v)*sy;
      f_a[j][i].u = udot[j][i].u - appctx->D1*(uxx + uyy) + uc*vc*vc - appctx->gamma*(1.0 - uc);
      f_a[j][i].v = udot[j][i].v - appctx->D2*(vxx + vyy) - uc*vc*vc + (appctx->gamma + appctx->kappa)*vc;
    }
  }

  /*
    Mark dependence

    NOTE: Marking dependence of dummy variables makes the index notation much simpler when forming
          the Jacobian later.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      if ((i < xs) || (i >= xs+xm) || (j < ys) || (j >= ys+ym)) {
        f_a[j][i].u >>= dummy;
        f_a[j][i].v >>= dummy;
      } else {
        f_a[j][i].u >>= f[j][i].u;
        f_a[j][i].v >>= f[j][i].v;
      }
    }
  }
  trace_off();  /* End of active section */
  CHKERRQ(PetscLogFlops(16.0*xm*ym));

  /* Restore vectors */
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  CHKERRQ(DMDAVecRestoreArrayRead(da,localU,&u));
  CHKERRQ(DMDAVecRestoreArrayRead(da,Udot,&udot));

  CHKERRQ(DMRestoreLocalVector(da,&localU));

  /* Destroy AFields appropriately */
  f_a += info.gys;
  u_a += info.gys;
  delete[] f_a;
  delete[] u_a;
  delete[] f_c;
  delete[] u_c;
  PetscFunctionReturn(0);
}

/*
  Simply acts to pass TS information to the AdolcMatCtx
*/
PetscErrorCode IJacobianMatFree(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A_shell,Mat B,void *ctx)
{
  AdolcMatCtx       *mctx;
  DM                da;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A_shell,&mctx));

  mctx->time  = t;
  mctx->shift = a;
  if (mctx->ts != ts) mctx->ts = ts;
  CHKERRQ(VecCopy(X,mctx->X));
  CHKERRQ(VecCopy(Xdot,mctx->Xdot));
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMGlobalToLocalBegin(da,mctx->X,INSERT_VALUES,mctx->localX0));
  CHKERRQ(DMGlobalToLocalEnd(da,mctx->X,INSERT_VALUES,mctx->localX0));
  PetscFunctionReturn(0);
}

/*TEST

  build:
    requires: double !complex adolc

  test:
    suffix: 1
    args: -ts_max_steps 1 -da_grid_x 12 -da_grid_y 12 -snes_test_jacobian
    output_file: output/adr_ex5adj_mf_1.out

  test:
    suffix: 2
    nsize: 4
    args: -ts_max_steps 10 -da_grid_x 12 -da_grid_y 12 -ts_monitor -ts_adjoint_monitor
    output_file: output/adr_ex5adj_mf_2.out

TEST*/
