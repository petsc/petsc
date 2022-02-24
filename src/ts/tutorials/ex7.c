
static char help[] = "Nonlinear, time-dependent PDE in 2d.\n";

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

/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,void*),FormInitialSolution(DM,Vec);
extern PetscErrorCode MyTSMonitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode MySNESMonitor(SNES,PetscInt,PetscReal,PetscViewerAndFormat*);

int main(int argc,char **argv)
{
  TS                   ts;                         /* time integrator */
  SNES                 snes;
  Vec                  x,r;                        /* solution, residual vectors */
  PetscErrorCode       ierr;
  DM                   da;
  PetscViewerAndFormat *vf;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,8,8,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

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
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetRHSFunction(ts,NULL,FormFunction,da));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(TSSetMaxTime(ts,1.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSMonitorSet(ts,MyTSMonitor,PETSC_VIEWER_STDOUT_WORLD,NULL));
  CHKERRQ(TSSetDM(ts,da));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetType(ts,TSBEULER));
  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
  CHKERRQ(SNESMonitorSet(snes,(PetscErrorCode (*)(SNES,PetscInt,PetscReal,void*))MySNESMonitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(da,x));
  CHKERRQ(TSSetTimeStep(ts,.0001));
  CHKERRQ(TSSetSolution(ts,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));

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
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      two = 2.0,hx,hy,sx,sy;
  PetscScalar    u,uxx,uyy,**x,**f;
  Vec            localX;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMGetLocalVector(da,&localX));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  hy = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);

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
  CHKERRQ(DMDAVecGetArrayRead(da,localX,&x));
  CHKERRQ(DMDAVecGetArray(da,F,&f));

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
        f[j][i] = x[j][i];
        continue;
      }
      u   = x[j][i];
      uxx = (two*u - x[j][i-1] - x[j][i+1])*sx;
      uyy = (two*u - x[j-1][i] - x[j+1][i])*sy;
      /*      f[j][i] = -(uxx + uyy); */
      f[j][i] = -u*(uxx + uyy) - (4.0 - 1.0)*((x[j][i+1] - x[j][i-1])*(x[j][i+1] - x[j][i-1])*.25*sx +
                                              (x[j+1][i] - x[j-1][i])*(x[j+1][i] - x[j-1][i])*.25*sy);
    }
  }

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArrayRead(da,localX,&x));
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  CHKERRQ(DMRestoreLocalVector(da,&localX));
  CHKERRQ(PetscLogFlops(11.0*ym*xm));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U)
{
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscScalar    **u;
  PetscReal      hx,hy,x,y,r;

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1);
  hy = 1.0/(PetscReal)(My-1);

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
      r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
      if (r < .125) u[j][i] = PetscExpReal(-30.0*r*r*r);
      else          u[j][i] = 0.0;
    }
  }

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscReal      norm;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(0); /* step of -1 indicates an interpolated solution */
  CHKERRQ(VecNorm(v,NORM_2,&norm));
  CHKERRQ(PetscObjectGetComm((PetscObject)ts,&comm));
  CHKERRQ(PetscPrintf(comm,"timestep %D time %g norm %g\n",step,(double)ptime,(double)norm));
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
      args: -ts_max_steps 5

    test:
      suffix: 2
      args: -ts_max_steps 5  -snes_mf_operator

    test:
      suffix: 3
      args: -ts_max_steps 5  -snes_mf -pc_type none

TEST*/
