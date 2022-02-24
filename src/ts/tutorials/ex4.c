
static char help[] ="Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -time_dependent_rhs : Treat the problem as having a time-dependent right-hand side\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program solves the one-dimensional heat equation (also called the
   diffusion equation),
       u_t = u_xx,
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u(t,1) = 0,
   and the initial condition
       u(0,x) = sin(6*pi*x) + 3*sin(2*pi*x).
   This is a linear, second-order, parabolic equation.

   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)
   We then demonstrate time evolution using the various TS methods by
   running the program via
       mpiexec -n <procs> ex3 -ts_type <timestepping solver>

   We compare the approximate solution with the exact solution, given by
       u_exact(x,t) = exp(-36*pi*pi*t) * sin(6*pi*x) +
                      3*exp(-4*pi*pi*t) * sin(2*pi*x)

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely
     - time-dependent f:   f(u,t) is a function of t
     - time-independent f: f(u,t) is simply f(u)

    The uniprocessor version of this code is ts/tutorials/ex3.c

  ------------------------------------------------------------------------- */

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs) to manage
   the parallel grid.  Include "petscts.h" so that we can use TS solvers.
   Note that this file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  MPI_Comm    comm;              /* communicator */
  DM          da;                /* distributed array data structure */
  Vec         localwork;         /* local ghosted work vector */
  Vec         u_local;           /* local ghosted approximate solution vector */
  Vec         solution;          /* global exact solution vector */
  PetscInt    m;                 /* total number of grid points */
  PetscReal   h;                 /* mesh width h = 1/(m-1) */
  PetscBool   debug;             /* flag (1 indicates activation of debugging printouts) */
  PetscViewer viewer1,viewer2;  /* viewers for the solution and error */
  PetscReal   norm_2,norm_max;  /* error norms */
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixHeat(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSFunctionHeat(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      time_total_max = 1.0;   /* default max total time */
  PetscInt       time_steps_max = 100;   /* default max timesteps */
  PetscDraw      draw;                   /* drawing context */
  PetscErrorCode ierr;
  PetscInt       steps,m;
  PetscMPIInt    size;
  PetscReal      dt,ftime;
  PetscBool      flg;
  TSProblemType  tsproblem = TS_LINEAR;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr        = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  appctx.comm = PETSC_COMM_WORLD;

  m               = 60;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug));
  appctx.m        = m;
  appctx.h        = 1.0/(m-1.0);
  appctx.norm_2   = 0.0;
  appctx.norm_max = 0.0;

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solving a linear TS problem, number of processors = %d\n",size));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are M
     total grid values spread equally among all the processors.
  */

  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,m,1,1,NULL,&appctx.da));
  CHKERRQ(DMSetFromOptions(appctx.da));
  CHKERRQ(DMSetUp(appctx.da));

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */
  CHKERRQ(DMCreateGlobalVector(appctx.da,&u));
  CHKERRQ(DMCreateLocalVector(appctx.da,&appctx.u_local));

  /*
     Create local work vector for use in evaluating right-hand-side function;
     create global work vector for storing exact solution.
  */
  CHKERRQ(VecDuplicate(appctx.u_local,&appctx.localwork));
  CHKERRQ(VecDuplicate(u,&appctx.solution));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up displays to show graphs of the solution and error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,380,400,160,&appctx.viewer1));
  CHKERRQ(PetscViewerDrawGetDraw(appctx.viewer1,0,&draw));
  CHKERRQ(PetscDrawSetDoubleBuffer(draw));
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,0,400,160,&appctx.viewer2));
  CHKERRQ(PetscViewerDrawGetDraw(appctx.viewer2,0,&draw));
  CHKERRQ(PetscDrawSetDoubleBuffer(draw));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-nonlinear",&flg,NULL));
  CHKERRQ(TSSetProblemType(ts,flg ? TS_NONLINEAR : TS_LINEAR));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSMonitorSet(ts,Monitor,&appctx,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-time_dependent_rhs",&flg,NULL));
  if (flg) {
    /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
    CHKERRQ(TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx));
    CHKERRQ(TSSetRHSJacobian(ts,A,A,RHSMatrixHeat,&appctx));
  } else {
    /*
       For linear problems with a time-independent f(u) in the equation
       u_t = f(u), the user provides the discretized right-hand-side
       as a matrix only once, and then sets a null matrix evaluation
       routine.
    */
    CHKERRQ(RHSMatrixHeat(ts,0.0,u,A,A,&appctx));
    CHKERRQ(TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx));
    CHKERRQ(TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx));
  }

  if (tsproblem == TS_NONLINEAR) {
    SNES snes;
    CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunctionHeat,&appctx));
    CHKERRQ(TSGetSNES(ts,&snes));
    CHKERRQ(SNESSetJacobian(snes,NULL,NULL,SNESComputeJacobianDefault,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  dt   = appctx.h*appctx.h/2.0;
  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSSetSolution(ts,u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_max_time <maxtime>
     to override the defaults set by TSSetMaxSteps()/TSSetMaxTime().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(TSSetMaxSteps(ts,time_steps_max));
  CHKERRQ(TSSetMaxTime(ts,time_total_max));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Evaluate initial conditions
  */
  CHKERRQ(InitialConditions(u,&appctx));

  /*
     Run the timestepping solver
  */
  CHKERRQ(TSSolve(ts,u));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Total timesteps %D, Final time %g\n",steps,(double)ftime));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Avg. error (2 norm) = %g Avg. error (max norm) = %g\n",(double)(appctx.norm_2/steps),(double)(appctx.norm_max/steps)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(PetscViewerDestroy(&appctx.viewer1));
  CHKERRQ(PetscViewerDestroy(&appctx.viewer2));
  CHKERRQ(VecDestroy(&appctx.localwork));
  CHKERRQ(VecDestroy(&appctx.solution));
  CHKERRQ(VecDestroy(&appctx.u_local));
  CHKERRQ(DMDestroy(&appctx.da));

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
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar    *u_localptr,h = appctx->h;
  PetscInt       i,mybase,myend;

  /*
     Determine starting point of each processor's range of
     grid values.
  */
  CHKERRQ(VecGetOwnershipRange(u,&mybase,&myend));

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  CHKERRQ(VecGetArray(u,&u_localptr));

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=mybase; i<myend; i++) u_localptr[i-mybase] = PetscSinScalar(PETSC_PI*i*6.*h) + 3.*PetscSinScalar(PETSC_PI*i*2.*h);

  /*
     Restore vector
  */
  CHKERRQ(VecRestoreArray(u,&u_localptr));

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    CHKERRQ(PetscPrintf(appctx->comm,"initial guess vector\n"));
    CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));
  }

  return 0;
}
/* --------------------------------------------------------------------- */
/*
   ExactSolution - Computes the exact solution at a given time.

   Input Parameters:
   t - current time
   solution - vector in which exact solution will be computed
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
PetscErrorCode ExactSolution(PetscReal t,Vec solution,AppCtx *appctx)
{
  PetscScalar    *s_localptr,h = appctx->h,ex1,ex2,sc1,sc2;
  PetscInt       i,mybase,myend;

  /*
     Determine starting and ending points of each processor's
     range of grid values
  */
  CHKERRQ(VecGetOwnershipRange(solution,&mybase,&myend));

  /*
     Get a pointer to vector data.
  */
  CHKERRQ(VecGetArray(solution,&s_localptr));

  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  ex1 = PetscExpReal(-36.*PETSC_PI*PETSC_PI*t); ex2 = PetscExpReal(-4.*PETSC_PI*PETSC_PI*t);
  sc1 = PETSC_PI*6.*h;                 sc2 = PETSC_PI*2.*h;
  for (i=mybase; i<myend; i++) s_localptr[i-mybase] = PetscSinScalar(sc1*(PetscReal)i)*ex1 + 3.*PetscSinScalar(sc2*(PetscReal)i)*ex2;

  /*
     Restore vector
  */
  CHKERRQ(VecRestoreArray(solution,&s_localptr));
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   Monitor - User-provided routine to monitor the solution computed at
   each timestep.  This example plots the solution and computes the
   error in two different norms.

   Input Parameters:
   ts     - the timestep context
   step   - the count of the current step (with 0 meaning the
             initial condition)
   time   - the current time
   u      - the solution at this timestep
   ctx    - the user-provided context for this monitoring routine.
            In this case we use the application context which contains
            information about the problem size, workspace and the exact
            solution.
*/
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
  PetscReal      norm_2,norm_max;

  /*
     View a graph of the current iterate
  */
  CHKERRQ(VecView(u,appctx->viewer2));

  /*
     Compute the exact solution
  */
  CHKERRQ(ExactSolution(time,appctx->solution,appctx));

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    CHKERRQ(PetscPrintf(appctx->comm,"Computed solution vector\n"));
    CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscPrintf(appctx->comm,"Exact solution vector\n"));
    CHKERRQ(VecView(appctx->solution,PETSC_VIEWER_STDOUT_WORLD));
  }

  /*
     Compute the 2-norm and max-norm of the error
  */
  CHKERRQ(VecAXPY(appctx->solution,-1.0,u));
  CHKERRQ(VecNorm(appctx->solution,NORM_2,&norm_2));
  norm_2 = PetscSqrtReal(appctx->h)*norm_2;
  CHKERRQ(VecNorm(appctx->solution,NORM_MAX,&norm_max));
  if (norm_2   < 1e-14) norm_2   = 0;
  if (norm_max < 1e-14) norm_max = 0;

  /*
     PetscPrintf() causes only the first processor in this
     communicator to print the timestep information.
  */
  CHKERRQ(PetscPrintf(appctx->comm,"Timestep %D: time = %g 2-norm error = %g max norm error = %g\n",step,(double)time,(double)norm_2,(double)norm_max));
  appctx->norm_2   += norm_2;
  appctx->norm_max += norm_max;

  /*
     View a graph of the error
  */
  CHKERRQ(VecView(appctx->solution,appctx->viewer1));

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    CHKERRQ(PetscPrintf(appctx->comm,"Error vector\n"));
    CHKERRQ(VecView(appctx->solution,PETSC_VIEWER_STDOUT_WORLD));
  }

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
  RHSMatrixHeat computes entries for the locally owned part of the system.
   - Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.
   - Each processor needs to insert only elements that it owns
     locally (but any non-local elements will be sent to the
     appropriate processor during matrix assembly).
   - Always specify global row and columns of matrix entries when
     using MatSetValues(); we could alternatively use MatSetValuesLocal().
   - Here, we set all entries for a particular row at once.
   - Note that MatSetValues() uses 0-based row and column numbers
     in Fortran as well as in C.
*/
PetscErrorCode RHSMatrixHeat(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;              /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscInt       i,mstart,mend,idx[3];
  PetscScalar    v[3],stwo = -2./(appctx->h*appctx->h),sone = -.5*stwo;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatGetOwnershipRange(A,&mstart,&mend));

  /*
     Set matrix rows corresponding to boundary data
  */

  if (mstart == 0) {  /* first processor only */
    v[0] = 1.0;
    CHKERRQ(MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES));
    mstart++;
  }

  if (mend == appctx->m) { /* last processor only */
    mend--;
    v[0] = 1.0;
    CHKERRQ(MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES));
  }

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
  v[0] = sone; v[1] = stwo; v[2] = sone;
  for (i=mstart; i<mend; i++) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES));
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

PetscErrorCode RHSFunctionHeat(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  Mat            A;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetRHSJacobian(ts,&A,NULL,NULL,&ctx));
  CHKERRQ(RHSMatrixHeat(ts,t,globalin,A,NULL,ctx));
  /* CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD)); */
  CHKERRQ(MatMult(A,globalin,globalout));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_view -nox

    test:
      suffix: 2
      args: -ts_view -nox
      nsize: 3

    test:
      suffix: 3
      args: -ts_view -nox -nonlinear

    test:
      suffix: 4
      args: -ts_view -nox -nonlinear
      nsize: 3
      timeoutfactor: 3

    test:
      suffix: sundials
      requires: sundials2
      args: -nox -ts_type sundials -ts_max_steps 5 -nonlinear
      nsize: 4

    test:
      suffix: sundials_dense
      requires: sundials2
      args: -nox -ts_type sundials -ts_sundials_use_dense -ts_max_steps 5 -nonlinear
      nsize: 1

TEST*/
