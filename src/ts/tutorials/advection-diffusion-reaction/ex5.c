
static char help[] = "Demonstrates Pattern Formation with Reaction-Diffusion Equations.\n";

/*F
     This example is taken from the book, Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations by
      W. Hundsdorf and J.G. Verwer,  Page 21, Pattern Formation with Reaction-Diffusion Equations
\begin{eqnarray*}
        u_t = D_1 (u_{xx} + u_{yy})  - u*v^2 + \gamma(1 -u)           \\
        v_t = D_2 (v_{xx} + v_{yy})  + u*v^2 - (\gamma + \kappa)v
\end{eqnarray*}
    Unlike in the book this uses periodic boundary conditions instead of Neumann
    (since they are easier for finite differences).
F*/

/*
      Helpful runtime monitor options:
           -ts_monitor_draw_solution
           -draw_save -draw_save_movie

      Helpful runtime linear solver options:
           -pc_type mg -pc_mg_galerkin pmat -da_refine 1 -snes_monitor -ksp_monitor -ts_view  (note that these Jacobians are so well-conditioned multigrid may not be the best solver)

      Point your browser to localhost:8080 to monitor the simulation
           ./ex5  -ts_view_pre saws  -stack_view saws -draw_save -draw_save_single_file -x_virtual -ts_monitor_draw_solution -saws_root .

*/

/*

   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscts.h" so that we can use SNES numerical (ODE) integrators.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h - matrices                    petscis.h   - index sets  
     petscksp.h - Krylov subspace methods     petscpc.h   - preconditioners
     petscviewer.h - viewers                  petscsnes.h - nonlinear solvers
*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*  Simple C struct that allows us to access the two velocity (x and y directions) values easily in the code */
typedef struct {
  PetscScalar u,v;
} Field;

/*  Data structure to store the model parameters */
typedef struct {
  PetscReal D1,D2,gamma,kappa;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*),InitialConditions(DM,Vec);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x;                   /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,65,65,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create global vector from DMDA; this will be used to store the solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,&appctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,2000.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.0001);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
/*
   RHSFunction - Evaluates nonlinear function, that defines the right 
     hand side of the ODE

   Input Parameters:
.  ts - the TS context
.  time - current time
.  U - input vector
.  ptr - optional user-defined context, as set by TSSetRHSFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode RHSFunction(TS ts,PetscReal time,Vec U,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;
  Field          **u,**f;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  /* Get local (ghosted) work vector */
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  /* Get information about mesh needed for discretization */
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My); sy = 1.0/(hy*hy);

  /*
     Scatter ghost points to local vector, using the 2-step process
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to actual vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries; this is the region that this process owns and must operate on
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid with standard finite differences
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
      f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;
    }
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);

  /*
     Restore access to vectors and return no longer needed work vector
  */
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
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

  hx = 2.5/(PetscReal)(Mx);
  hy = 2.5/(PetscReal)(My);

  /*
     Get pointers to actual vector data
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
     Restore access to vector
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   RHSJacobian - Evaluates the Jacobian of the right hand side
     function of the ODE.

   Input Parameters:
.  ts - the TS context
.  time - current time
.  U - input vector
.  ptr - optional user-defined context, as set by TSSetRHSJacobian()

   Output Parameter:
.  A - the Jacobian
.  BB - optional additional matrix where an approximation to the Jacobian
        may be stored from which the preconditioner is constructed
 */
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,vc;
  Field          **u;
  Vec            localU;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My); sy = 1.0/(hy*hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {

    stencil[0].j = j-1;
    stencil[1].j = j+1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0; rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      uc = u[j][i].u;
      vc = u[j][i].v;

      /*      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;

      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
       f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);*/

      stencil[0].i = i; stencil[0].c = 0; entries[0] = appctx->D1*sy;
      stencil[1].i = i; stencil[1].c = 0; entries[1] = appctx->D1*sy;
      stencil[2].i = i-1; stencil[2].c = 0; entries[2] = appctx->D1*sx;
      stencil[3].i = i+1; stencil[3].c = 0; entries[3] = appctx->D1*sx;
      stencil[4].i = i; stencil[4].c = 0; entries[4] = -2.0*appctx->D1*(sx + sy) - vc*vc - appctx->gamma;
      stencil[5].i = i; stencil[5].c = 1; entries[5] = -2.0*uc*vc;
      rowstencil.i = i; rowstencil.c = 0;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);

      stencil[0].c = 1; entries[0] = appctx->D2*sy;
      stencil[1].c = 1; entries[1] = appctx->D2*sy;
      stencil[2].c = 1; entries[2] = appctx->D2*sx;
      stencil[3].c = 1; entries[3] = appctx->D2*sx;
      stencil[4].c = 1; entries[4] = -2.0*appctx->D2*(sx + sy) + 2.0*uc*vc - appctx->gamma - appctx->kappa;
      stencil[5].c = 0; entries[5] = vc*vc;
      rowstencil.c = 1;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      /* f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc; */
    }
  }

  /*
     Restore vectors
  */
  ierr = PetscLogFlops(19*xm*ym);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

   test:
      args: -ts_view  -ts_monitor -ts_max_time 500
      requires: double
      timeoutfactor: 3

   test:
      suffix: 2
      args: -ts_view  -ts_monitor -ts_max_time 500 -ts_monitor_draw_solution
      requires: x double
      output_file: output/ex5_1.out
      timeoutfactor: 3

TEST*/
