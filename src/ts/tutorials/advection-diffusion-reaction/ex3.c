
static char help[] ="Model Equations for Advection-Diffusion\n";

/*
    Page 9, Section 1.2 Model Equations for Advection-Diffusion

          u_t = a u_x + d u_xx

   The initial conditions used here different then in the book.

*/

/*
     Helpful runtime linear solver options:
           -pc_type mg -da_refine 2 -snes_monitor -ksp_monitor -ts_view   (geometric multigrid with three levels)

*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  PetscScalar a,d;   /* advection and diffusion strength */
  PetscBool   upwind;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(TS,Vec,AppCtx*);
extern PetscErrorCode RHSMatrixHeat(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode Solution(TS,PetscReal,Vec,AppCtx*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Vec            U;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscReal      dt;
  DM             da;
  PetscInt       M;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr          = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  appctx.a      = 1.0;
  appctx.d      = 0.0;
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-a",&appctx.a,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-d",&appctx.d,NULL));
  appctx.upwind = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-upwind",&appctx.upwind,NULL));

  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC, 60, 1, 1,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vector data structures for approximate and exact solutions
  */
  CHKERRQ(DMCreateGlobalVector(da,&U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetDM(ts,da));

  /*
      For linear problems with a time-dependent f(U,t) in the equation
     u_t = f(u,t), the user provides the discretized right-hand-side
      as a time-dependent matrix.
  */
  CHKERRQ(TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx));
  CHKERRQ(TSSetRHSJacobian(ts,NULL,NULL,RHSMatrixHeat,&appctx));
  CHKERRQ(TSSetSolutionFunction(ts,(PetscErrorCode (*)(TS,PetscReal,Vec,void*))Solution,&appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_max_time <maxtime>
     to override the defaults set by TSSetMaxSteps()/TSSetMaxTime().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&M,0,0,0,0,0,0,0,0,0,0,0));
  dt   = .48/(M*M);
  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSSetMaxSteps(ts,1000));
  CHKERRQ(TSSetMaxTime(ts,100.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetType(ts,TSARKIMEX));
  CHKERRQ(TSSetFromOptions(ts));

  /*
     Evaluate initial conditions
  */
  CHKERRQ(InitialConditions(ts,U,&appctx));

  /*
     Run the timestepping solver
  */
  CHKERRQ(TSSolve(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(DMDestroy(&da));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}
/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(TS ts,Vec U,AppCtx *appctx)
{
  PetscScalar    *u,h;
  PetscInt       i,mstart,mend,xm,M;
  DM             da;

  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetCorners(da,&mstart,0,0,&xm,0,0));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&M,0,0,0,0,0,0,0,0,0,0,0));
  h    = 1.0/M;
  mend = mstart + xm;
  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  CHKERRQ(DMDAVecGetArray(da,U,&u));

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=mstart; i<mend; i++) u[i] = PetscSinScalar(PETSC_PI*i*6.*h) + 3.*PetscSinScalar(PETSC_PI*i*2.*h);

  /*
     Restore vector
  */
  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   Solution - Computes the exact solution at a given time.

   Input Parameters:
   t - current time
   solution - vector in which exact solution will be computed
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
PetscErrorCode Solution(TS ts,PetscReal t,Vec U,AppCtx *appctx)
{
  PetscScalar    *u,ex1,ex2,sc1,sc2,h;
  PetscInt       i,mstart,mend,xm,M;
  DM             da;

  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetCorners(da,&mstart,0,0,&xm,0,0));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&M,0,0,0,0,0,0,0,0,0,0,0));
  h    = 1.0/M;
  mend = mstart + xm;
  /*
     Get a pointer to vector data.
  */
  CHKERRQ(DMDAVecGetArray(da,U,&u));

  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  ex1 = PetscExpScalar(-36.*PETSC_PI*PETSC_PI*appctx->d*t);
  ex2 = PetscExpScalar(-4.*PETSC_PI*PETSC_PI*appctx->d*t);
  sc1 = PETSC_PI*6.*h;                 sc2 = PETSC_PI*2.*h;
  for (i=mstart; i<mend; i++) u[i] = PetscSinScalar(sc1*(PetscReal)i + appctx->a*PETSC_PI*6.*t)*ex1 + 3.*PetscSinScalar(sc2*(PetscReal)i + appctx->a*PETSC_PI*2.*t)*ex2;

  /*
     Restore vector
  */
  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  return 0;
}

/* --------------------------------------------------------------------- */
/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

   Notes:
   Recall that MatSetValues() uses 0-based row and column numbers
   in Fortran as well as in C.
*/
PetscErrorCode RHSMatrixHeat(TS ts,PetscReal t,Vec U,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;                /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscInt       mstart, mend;
  PetscInt       i,idx[3],M,xm;
  PetscScalar    v[3],h;
  DM             da;

  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(DMDAGetCorners(da,&mstart,0,0,&xm,0,0));
  h    = 1.0/M;
  mend = mstart + xm;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set matrix rows corresponding to boundary data
  */

  /* diffusion */
  v[0] = appctx->d/(h*h);
  v[1] = -2.0*appctx->d/(h*h);
  v[2] = appctx->d/(h*h);
  if (!mstart) {
    idx[0] = M-1; idx[1] = 0; idx[2] = 1;
    CHKERRQ(MatSetValues(A,1,&mstart,3,idx,v,INSERT_VALUES));
    mstart++;
  }

  if (mend == M) {
    mend--;
    idx[0] = M-2; idx[1] = M-1; idx[2] = 0;
    CHKERRQ(MatSetValues(A,1,&mend,3,idx,v,INSERT_VALUES));
  }

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
  for (i=mstart; i<mend; i++) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY));

  CHKERRQ(DMDAGetCorners(da,&mstart,0,0,&xm,0,0));
  mend = mstart + xm;
  if (!appctx->upwind) {
    /* advection -- centered differencing */
    v[0] = -.5*appctx->a/(h);
    v[1] = .5*appctx->a/(h);
    if (!mstart) {
      idx[0] = M-1; idx[1] = 1;
      CHKERRQ(MatSetValues(A,1,&mstart,2,idx,v,ADD_VALUES));
      mstart++;
    }

    if (mend == M) {
      mend--;
      idx[0] = M-2; idx[1] = 0;
      CHKERRQ(MatSetValues(A,1,&mend,2,idx,v,ADD_VALUES));
    }

    for (i=mstart; i<mend; i++) {
      idx[0] = i-1; idx[1] = i+1;
      CHKERRQ(MatSetValues(A,1,&i,2,idx,v,ADD_VALUES));
    }
  } else {
    /* advection -- upwinding */
    v[0] = -appctx->a/(h);
    v[1] = appctx->a/(h);
    if (!mstart) {
      idx[0] = 0; idx[1] = 1;
      CHKERRQ(MatSetValues(A,1,&mstart,2,idx,v,ADD_VALUES));
      mstart++;
    }

    if (mend == M) {
      mend--;
      idx[0] = M-1; idx[1] = 0;
      CHKERRQ(MatSetValues(A,1,&mend,2,idx,v,ADD_VALUES));
    }

    for (i=mstart; i<mend; i++) {
      idx[0] = i; idx[1] = i+1;
      CHKERRQ(MatSetValues(A,1,&i,2,idx,v,ADD_VALUES));
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Complete the matrix assembly process and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  return 0;
}

/*TEST

   test:
      args: -pc_type mg -da_refine 2  -ts_view  -ts_monitor -ts_max_time .3 -mg_levels_ksp_max_it 3
      requires: double
      filter: grep -v "total number of"

   test:
      suffix: 2
      args:  -pc_type mg -da_refine 2  -ts_view  -ts_monitor_draw_solution -ts_monitor -ts_max_time .3 -mg_levels_ksp_max_it 3
      requires: x
      output_file: output/ex3_1.out
      requires: double
      filter: grep -v "total number of"

TEST*/
