static char help[] = "Demonstrates automatic, matrix-free Jacobian generation using ADOL-C for a time-dependent PDE in 2d, solved using implicit timestepping.\n";

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
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL));
  PetscFunctionBeginUser;
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  PetscCall(PetscLogEventRegister("df/dx forward",MAT_CLASSID,&matctx.event1));
  PetscCall(PetscLogEventRegister("df/d(xdot) forward",MAT_CLASSID,&matctx.event2));
  PetscCall(PetscLogEventRegister("df/dx reverse",MAT_CLASSID,&matctx.event3));
  PetscCall(PetscLogEventRegister("df/d(xdot) reverse",MAT_CLASSID,&matctx.event4));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,65,65,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da,0,"u"));
  PetscCall(DMDASetFieldName(da,1,"v"));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(VecDuplicate(x,&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create matrix free context and specify usage of PETSc-ADOL-C drivers
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSetMatType(da,MATSHELL));
  PetscCall(DMCreateMatrix(da,&A));
  PetscCall(MatShellSetContext(A,&matctx));
  PetscCall(MatShellSetOperation(A,MATOP_MULT,(void (*)(void))PetscAdolcIJacobianVectorProductIDMass));
  PetscCall(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void (*)(void))PetscAdolcIJacobianTransposeVectorProductIDMass));
  PetscCall(VecDuplicate(x,&matctx.X));
  PetscCall(VecDuplicate(x,&matctx.Xdot));
  PetscCall(DMGetLocalVector(da,&matctx.localX0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSCN));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalPassive,&appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Some data required for matrix-free context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDAGetGhostCorners(da,NULL,NULL,NULL,&gxm,&gym,NULL));
  matctx.m = 2*gxm*gym;matctx.n = 2*gxm*gym; /* Number of dependent and independent variables */
  matctx.flg = PETSC_FALSE;                  /* Flag for reverse mode */
  matctx.tag1 = 1;                           /* Tape identifier */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace function just once
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscNew(&appctx.adctx));
  PetscCall(IFunctionActive(ts,1.,x,matctx.Xdot,r,&appctx));
  PetscCall(PetscFree(appctx.adctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian. In this case, IJacobian simply acts to pass context
     information to the matrix-free Jacobian vector product.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetIJacobian(ts,A,A,IJacobianMatFree,&appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(InitialConditions(da,x));
  PetscCall(TSSetSolution(ts,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Have the TS save its trajectory so that TSAdjointSolve() may be used
    and set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!forwardonly) {
    PetscCall(TSSetSaveTrajectory(ts));
    PetscCall(TSSetMaxTime(ts,200.0));
    PetscCall(TSSetTimeStep(ts,0.5));
  } else {
    PetscCall(TSSetMaxTime(ts,2000.0));
    PetscCall(TSSetTimeStep(ts,10));
  }
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,x));
  if (!forwardonly) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Start the Adjoint model
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(VecDuplicate(x,&lambda[0]));
    /*   Reset initial conditions for the adjoint integration */
    PetscCall(InitializeLambda(da,lambda[0],0.5,0.5));
    PetscCall(TSSetCostGradients(ts,1,lambda,NULL));
    PetscCall(TSAdjointSolve(ts));
    PetscCall(VecDestroy(&lambda[0]));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMRestoreLocalVector(da,&matctx.localX0));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&matctx.X));
  PetscCall(VecDestroy(&matctx.Xdot));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
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

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArray(da,U,&u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

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
  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y)
{
   PetscInt i,j,Mx,My,xs,ys,xm,ym;
   Field **l;

   PetscFunctionBegin;
   PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
   /* locate the global i index for x and j index for y */
   i = (PetscInt)(x*(Mx-1));
   j = (PetscInt)(y*(My-1));
   PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

   if (xs <= i && i < xs+xm && ys <= j && j < ys+ym) {
     /* the i,j vertex is on this process */
     PetscCall(DMDAVecGetArray(da,lambda,&l));
     l[j][i].u = 1.0;
     l[j][i].v = 1.0;
     PetscCall(DMDAVecRestoreArray(da,lambda,&l));
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
  PetscCall(PetscLogFlops(16.0*xm*ym));
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
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMGetLocalVector(da,&localU));
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
  PetscCall(DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU));
  PetscCall(DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da,localU,&u));
  PetscCall(DMDAVecGetArray(da,F,&f));
  PetscCall(DMDAVecGetArrayRead(da,Udot,&udot));

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
  PetscCall(GiveGhostPoints(da,u_c,&u_a));
  PetscCall(GiveGhostPoints(da,f_c,&f_a));

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
  PetscCall(PetscLogFlops(16.0*xm*ym));

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMDAVecRestoreArrayRead(da,localU,&u));
  PetscCall(DMDAVecRestoreArrayRead(da,Udot,&udot));

  PetscCall(DMRestoreLocalVector(da,&localU));

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
  PetscCall(MatShellGetContext(A_shell,&mctx));

  mctx->time  = t;
  mctx->shift = a;
  if (mctx->ts != ts) mctx->ts = ts;
  PetscCall(VecCopy(X,mctx->X));
  PetscCall(VecCopy(Xdot,mctx->Xdot));
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGlobalToLocalBegin(da,mctx->X,INSERT_VALUES,mctx->localX0));
  PetscCall(DMGlobalToLocalEnd(da,mctx->X,INSERT_VALUES,mctx->localX0));
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
