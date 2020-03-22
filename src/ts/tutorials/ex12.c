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
  ierr = PetscOptionsGetBool(NULL,NULL,"-usemonitor",&usemonitor,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,8,8,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

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
  ierr = TSSetRHSFunction(ts,NULL,FormFunction,da);CHKERRQ(ierr);

  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  if (usemonitor) {
    ierr = TSMonitorSet(ts,MyTSMonitor,0,0);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&ts_snes);CHKERRQ(ierr);
  if (usemonitor) {
    ierr = PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf);CHKERRQ(ierr);
    ierr = SNESMonitorSet(ts_snes,(PetscErrorCode (*)(SNES,PetscInt,PetscReal,void *))MySNESMonitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(da,x);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.0001);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = VecViewFromOptions(x,NULL,"-final_sol");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,/*hxdhy,hydhx,*/ sx,sy;
  PetscScalar    u,uxx,uyy,v,***x,***f;
  Vec            localX;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

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
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayDOF(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

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
  ierr = DMDAVecRestoreArrayDOF(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*ym*xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscScalar    ***u;
  PetscReal      hx,hy,x,y,r;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)(Mx-1);
  hy = 1.0/(PetscReal)(My-1);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayDOF(da,U,&u);CHKERRQ(ierr);

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
  ierr = DMDAVecRestoreArrayDOF(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      norm;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  if (step > -1) { /* -1 is used to indicate an interpolated value */
    ierr = PetscPrintf(comm,"timestep %D time %g norm %g\n",step,(double)ptime,(double)norm);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESMonitorDefaultShort(snes,its,fnorm,vf);CHKERRQ(ierr);
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
