
static char help[] = "Time-dependent PDE in 2d. Modified from ex13.c for illustrating how to solve DAEs. \n";
/*
   u_t = uxx + uyy
   0 < x < 1, 0 < y < 1;
   At t=0: u(x,y) = exp(c*r*r*r), if r=PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5)) < .125
           u(x,y) = 0.0           if r >= .125

   Boundary conditions:
   Drichlet BC:
   At x=0, x=1, y=0, y=1: u = 0.0

   Neumann BC:
   At x=0, x=1: du(x,y,t)/dx = 0
   At y=0, y=1: du(x,y,t)/dy = 0

   mpiexec -n 2 ./ex15 -da_grid_x 40 -da_grid_y 40 -ts_max_steps 2 -snes_monitor -ksp_monitor
         ./ex15 -da_grid_x 40 -da_grid_y 40  -draw_pause .1 -boundary 1 -ts_monitor_draw_solution
         ./ex15 -da_grid_x 40 -da_grid_y 40  -draw_pause .1 -boundary 1 -Jtype 2 -nstencilpts 9

*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*
   User-defined data structures and routines
*/

/* AppCtx: used by FormIFunction() and FormIJacobian() */
typedef struct {
  DM        da;
  PetscInt  nstencilpts;         /* number of stencil points: 5 or 9 */
  PetscReal c;
  PetscInt  boundary;            /* Type of boundary condition */
  PetscBool viewJacobian;
} AppCtx;

extern PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode FormInitialSolution(Vec,void*);

int main(int argc,char **argv)
{
  TS             ts;                   /* nonlinear solver */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            J,Jmf = NULL;   /* Jacobian matrices */
  DM             da;
  PetscReal      dt;
  AppCtx         user;              /* user-defined work context */
  SNES           snes;
  PetscInt       Jtype; /* Jacobian type
                            0: user provide Jacobian;
                            1: slow finite difference;
                            2: fd with coloring; */

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Initialize user application context */
  user.da           = NULL;
  user.nstencilpts  = 5;
  user.c            = -30.0;
  user.boundary     = 0;  /* 0: Drichlet BC; 1: Neumann BC */
  user.viewJacobian = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nstencilpts",&user.nstencilpts,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-boundary",&user.boundary,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-viewJacobian",&user.viewJacobian));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (user.nstencilpts == 5) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  } else if (user.nstencilpts == 9) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"nstencilpts %d is not supported",user.nstencilpts);
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  user.da = da;

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da,&u));
  PetscCall(VecDuplicate(u,&r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSBEULER));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetIFunction(ts,r,FormIFunction,&user));
  PetscCall(TSSetMaxTime(ts,1.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(u,&user));
  PetscCall(TSSetSolution(ts,u));
  dt   = .01;
  PetscCall(TSSetTimeStep(ts,dt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set Jacobian evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSetMatType(da,MATAIJ));
  PetscCall(DMCreateMatrix(da,&J));
  Jtype = 0;
  PetscCall(PetscOptionsGetInt(NULL,NULL, "-Jtype",&Jtype,NULL));
  if (Jtype == 0) { /* use user provided Jacobian evaluation routine */
    PetscCheck(user.nstencilpts == 5,PETSC_COMM_WORLD,PETSC_ERR_SUP,"user Jacobian routine FormIJacobian() does not support nstencilpts=%D",user.nstencilpts);
    PetscCall(TSSetIJacobian(ts,J,J,FormIJacobian,&user));
  } else { /* use finite difference Jacobian J as preconditioner and '-snes_mf_operator' for Mat*vec */
    PetscCall(TSGetSNES(ts,&snes));
    PetscCall(MatCreateSNESMF(snes,&Jmf));
    if (Jtype == 1) { /* slow finite difference J; */
      PetscCall(SNESSetJacobian(snes,Jmf,J,SNESComputeJacobianDefault,NULL));
    } else if (Jtype == 2) { /* Use coloring to compute  finite difference J efficiently */
      PetscCall(SNESSetJacobian(snes,Jmf,J,SNESComputeJacobianDefaultColor,0));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Jtype is not supported");
  }

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Sets various TS parameters from user options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&Jmf));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/* --------------------------------------------------------------------- */
/*
  FormIFunction = Udot - RHSFunction
*/
PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  AppCtx         *user=(AppCtx*)ctx;
  DM             da   = (DM)user->da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    u,uxx,uyy,**uarray,**f,**udot;
  Vec            localU;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(da,&localU));
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  hy = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);
  PetscCheck(user->nstencilpts != 9 || hx == hy,PETSC_COMM_WORLD,PETSC_ERR_SUP,"hx must equal hy when nstencilpts = 9 for this example");

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU));
  PetscCall(DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da,localU,&uarray));
  PetscCall(DMDAVecGetArray(da,F,&f));
  PetscCall(DMDAVecGetArray(da,Udot,&udot));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      /* Boundary conditions */
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        if (user->boundary == 0) { /* Drichlet BC */
          f[j][i] = uarray[j][i]; /* F = U */
        } else {                  /* Neumann BC */
          if (i == 0 && j == 0) {              /* SW corner */
            f[j][i] = uarray[j][i] - uarray[j+1][i+1];
          } else if (i == Mx-1 && j == 0) {    /* SE corner */
            f[j][i] = uarray[j][i] - uarray[j+1][i-1];
          } else if (i == 0 && j == My-1) {    /* NW corner */
            f[j][i] = uarray[j][i] - uarray[j-1][i+1];
          } else if (i == Mx-1 && j == My-1) { /* NE corner */
            f[j][i] = uarray[j][i] - uarray[j-1][i-1];
          } else if (i == 0) {                  /* Left */
            f[j][i] = uarray[j][i] - uarray[j][i+1];
          } else if (i == Mx-1) {               /* Right */
            f[j][i] = uarray[j][i] - uarray[j][i-1];
          } else if (j == 0) {                 /* Bottom */
            f[j][i] = uarray[j][i] - uarray[j+1][i];
          } else if (j == My-1) {               /* Top */
            f[j][i] = uarray[j][i] - uarray[j-1][i];
          }
        }
      } else { /* Interior */
        u = uarray[j][i];
        /* 5-point stencil */
        uxx = (-2.0*u + uarray[j][i-1] + uarray[j][i+1]);
        uyy = (-2.0*u + uarray[j-1][i] + uarray[j+1][i]);
        if (user->nstencilpts == 9) {
          /* 9-point stencil: assume hx=hy */
          uxx = 2.0*uxx/3.0 + (0.5*(uarray[j-1][i-1]+uarray[j-1][i+1]+uarray[j+1][i-1]+uarray[j+1][i+1]) - 2.0*u)/6.0;
          uyy = 2.0*uyy/3.0 + (0.5*(uarray[j-1][i-1]+uarray[j-1][i+1]+uarray[j+1][i-1]+uarray[j+1][i+1]) - 2.0*u)/6.0;
        }
        f[j][i] = udot[j][i] - (uxx*sx + uyy*sy);
      }
    }
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArrayRead(da,localU,&uarray));
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMDAVecRestoreArray(da,Udot,&udot));
  PetscCall(DMRestoreLocalVector(da,&localU));
  PetscCall(PetscLogFlops(11.0*ym*xm));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  FormIJacobian() - Compute IJacobian = dF/dU + a dF/dUdot
  This routine is not used with option '-use_coloring'
*/
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,nc;
  AppCtx         *user = (AppCtx*)ctx;
  DM             da    = (DM)user->da;
  MatStencil     col[5],row;
  PetscScalar    vals[5],hx,hy,sx,sy;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  hy = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      nc    = 0;
      row.j = j; row.i = i;
      if (user->boundary == 0 && (i == 0 || i == Mx-1 || j == 0 || j == My-1)) {
        col[nc].j = j; col[nc].i = i; vals[nc++] = 1.0;

      } else if (user->boundary > 0 && i == 0) {  /* Left Neumann */
        col[nc].j = j; col[nc].i = i;   vals[nc++] = 1.0;
        col[nc].j = j; col[nc].i = i+1; vals[nc++] = -1.0;
      } else if (user->boundary > 0 && i == Mx-1) { /* Right Neumann */
        col[nc].j = j; col[nc].i = i;   vals[nc++] = 1.0;
        col[nc].j = j; col[nc].i = i-1; vals[nc++] = -1.0;
      } else if (user->boundary > 0 && j == 0) {  /* Bottom Neumann */
        col[nc].j = j;   col[nc].i = i; vals[nc++] = 1.0;
        col[nc].j = j+1; col[nc].i = i; vals[nc++] = -1.0;
      } else if (user->boundary > 0 && j == My-1) { /* Top Neumann */
        col[nc].j = j;   col[nc].i = i;  vals[nc++] = 1.0;
        col[nc].j = j-1; col[nc].i = i;  vals[nc++] = -1.0;
      } else {   /* Interior */
        col[nc].j = j-1; col[nc].i = i;   vals[nc++] = -sy;
        col[nc].j = j;   col[nc].i = i-1; vals[nc++] = -sx;
        col[nc].j = j;   col[nc].i = i;   vals[nc++] = 2.0*(sx + sy) + a;
        col[nc].j = j;   col[nc].i = i+1; vals[nc++] = -sx;
        col[nc].j = j+1; col[nc].i = i;   vals[nc++] = -sy;
      }
      PetscCall(MatSetValuesStencil(Jpre,1,&row,nc,col,vals,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }

  if (user->viewJacobian) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)Jpre),"Jpre:\n"));
    PetscCall(MatView(Jpre,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(Vec U,void *ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  DM             da   =user->da;
  PetscReal      c    =user->c;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscScalar    **u;
  PetscReal      hx,hy,x,y,r;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1);
  hy = 1.0/(PetscReal)(My-1);

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArray(da,U,&u));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
      if (r < .125) u[j][i] = PetscExpReal(c*r*r*r);
      else u[j][i] = 0.0;
    }
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -ts_monitor

    test:
      suffix: 2
      args: -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -Jtype 2 -ts_monitor

    test:
      suffix: 3
      requires: !single
      args: -da_grid_x 20 -da_grid_y 20 -boundary 1 -ts_max_steps 10 -ts_monitor

    test:
      suffix: 4
      requires: !single
      nsize: 2
      args: -da_grid_x 20 -da_grid_y 20 -boundary 1 -ts_max_steps 10 -ts_monitor

    test:
      suffix: 5
      nsize: 1
      args: -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -Jtype 1 -ts_monitor

TEST*/
