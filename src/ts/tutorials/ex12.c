static char help[] = "Nonlinear, time-dependent PDE in 2d.\n";
/*
  Solves the equation

    u_tt - \Delta u = 0

  which we split into two first-order systems

    u_t -     v    = 0    so that     F(u,v).u = v
    v_t - \Delta u = 0    so that     F(u,v).v = Delta u

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

/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,void*),FormInitialSolution(DM,Vec);
extern PetscErrorCode MyTSMonitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode MySNESMonitor(SNES,PetscInt,PetscReal,PetscViewerAndFormat*);

int main(int argc,char **argv)
{
  TS                   ts;                         /* nonlinear solver */
  Vec                  x,r;                        /* solution, residual vectors */
  PetscInt             steps;                      /* iterations for convergence */
  PetscErrorCode       ierr;
  DM                   da;
  PetscReal            ftime;
  SNES                 ts_snes;
  PetscBool            usemonitor = PETSC_TRUE;
  PetscViewerAndFormat *vf;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-usemonitor",&usemonitor,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,8,8,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
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
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetDM(ts,da));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetRHSFunction(ts,NULL,FormFunction,da));

  CHKERRQ(TSSetMaxTime(ts,1.0));
  if (usemonitor) {
    CHKERRQ(TSMonitorSet(ts,MyTSMonitor,0,0));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetType(ts,TSBEULER));
  CHKERRQ(TSGetSNES(ts,&ts_snes));
  if (usemonitor) {
    CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
    CHKERRQ(SNESMonitorSet(ts_snes,(PetscErrorCode (*)(SNES,PetscInt,PetscReal,void *))MySNESMonitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(da,x));
  CHKERRQ(TSSetTimeStep(ts,.0001));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetSolution(ts,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(VecViewFromOptions(x,NULL,"-final_sol"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&da));

  ierr = PetscFinalize();
  return ierr;
}
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
PetscErrorCode FormFunction(TS ts,PetscReal ftime,Vec X,Vec F,void *ptr)
{
  DM             da = (DM)ptr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,/*hxdhy,hydhx,*/ sx,sy;
  PetscScalar    u,uxx,uyy,v,***x,***f;
  Vec            localX;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetLocalVector(da,&localX));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  hy = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);
  /*hxdhy  = hx/hy;*/
  /*hydhx  = hy/hx;*/

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  CHKERRQ(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMDAVecGetArrayDOF(da,localX,&x));
  CHKERRQ(DMDAVecGetArrayDOF(da,F,&f));

  /*
     Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        f[j][i][0] = x[j][i][0];
        f[j][i][1] = x[j][i][1];
        continue;
      }
      u          = x[j][i][0];
      v          = x[j][i][1];
      uxx        = (-2.0*u + x[j][i-1][0] + x[j][i+1][0])*sx;
      uyy        = (-2.0*u + x[j-1][i][0] + x[j+1][i][0])*sy;
      f[j][i][0] = v;
      f[j][i][1] = uxx + uyy;
    }
  }

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArrayDOF(da,localX,&x));
  CHKERRQ(DMDAVecRestoreArrayDOF(da,F,&f));
  CHKERRQ(DMRestoreLocalVector(da,&localX));
  CHKERRQ(PetscLogFlops(11.0*ym*xm));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscScalar    ***u;
  PetscReal      hx,hy,x,y,r;

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1);
  hy = 1.0/(PetscReal)(My-1);

  /*
     Get pointers to vector data
  */
  CHKERRQ(DMDAVecGetArrayDOF(da,U,&u));

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
      r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
      if (r < .125) {
        u[j][i][0] = PetscExpReal(-30.0*r*r*r);
        u[j][i][1] = 0.0;
      } else {
        u[j][i][0] = 0.0;
        u[j][i][1] = 0.0;
      }
    }
  }

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArrayDOF(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscReal      norm;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  CHKERRQ(VecNorm(v,NORM_2,&norm));
  CHKERRQ(PetscObjectGetComm((PetscObject)ts,&comm));
  if (step > -1) { /* -1 is used to indicate an interpolated value */
    CHKERRQ(PetscPrintf(comm,"timestep %D time %g norm %g\n",step,(double)ptime,(double)norm));
  }
  PetscFunctionReturn(0);
}

/*
   MySNESMonitor - illustrate how to set user-defined monitoring routine for SNES.
   Input Parameters:
     snes - the SNES context
     its - iteration number
     fnorm - 2-norm function value (may be estimated)
     ctx - optional user-defined context for private data for the
         monitor routine, as set by SNESMonitorSet()
 */
PetscErrorCode MySNESMonitor(SNES snes,PetscInt its,PetscReal fnorm,PetscViewerAndFormat *vf)
{
  PetscFunctionBeginUser;
  CHKERRQ(SNESMonitorDefaultShort(snes,its,fnorm,vf));
  PetscFunctionReturn(0);
}
/*TEST

    test:
      args: -da_grid_x 20 -ts_max_time 3 -ts_dt 1e-1 -ts_theta_initial_guess_extrapolate 0 -ts_monitor -ksp_monitor_short
      requires: !single

    test:
      suffix: 2
      args: -da_grid_x 20 -ts_max_time 0.11 -ts_dt 1e-1 -ts_type glle -ts_monitor -ksp_monitor_short
      requires: !single

    test:
      suffix: glvis_da_2d_vect
      args: -usemonitor 0 -da_grid_x 20 -ts_max_time 0.3 -ts_dt 1e-1 -ts_type glle -final_sol glvis: -viewer_glvis_dm_da_bs 2,0
      requires: !single

    test:
      suffix: glvis_da_2d_vect_ll
      args: -usemonitor 0 -da_grid_x 20 -ts_max_time 0.3 -ts_dt 1e-1 -ts_type glle -final_sol glvis: -viewer_glvis_dm_da_bs 2,0 -viewer_glvis_dm_da_ll
      requires: !single

TEST*/
