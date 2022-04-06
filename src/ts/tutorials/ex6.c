
static char help[] ="Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -time_dependent_rhs : Treat the problem as having a time-dependent right-hand side\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

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
       ex3 -ts_type <timestepping solver>

   We compare the approximate solution with the exact solution, given by
       u_exact(x,t) = exp(-36*pi*pi*t) * sin(6*pi*x) +
                      3*exp(-4*pi*pi*t) * sin(2*pi*x)

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely
     - time-dependent f:   f(u,t) is a function of t
     - time-independent f: f(u,t) is simply f(u)

    The parallel version of this code is ts/tutorials/ex4.c

  ------------------------------------------------------------------------- */

/*
   Include "ts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h  - base PETSc routines   vec.h  - vectors
     sys.h    - system routines       mat.h  - matrices
     is.h     - index sets            ksp.h  - Krylov subspace methods
     viewer.h - viewers               pc.h   - preconditioners
     snes.h - nonlinear solvers
*/

#include <petscts.h>
#include <petscdraw.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Vec         solution;          /* global exact solution vector */
  PetscInt    m;                 /* total number of grid points */
  PetscReal   h;                 /* mesh width h = 1/(m-1) */
  PetscBool   debug;             /* flag (1 indicates activation of debugging printouts) */
  PetscViewer viewer1, viewer2;  /* viewers for the solution and error */
  PetscReal   norm_2, norm_max;  /* error norms */
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixHeat(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);
extern PetscErrorCode MyBCRoutine(TS,PetscReal,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      time_total_max = 100.0; /* default max total time */
  PetscInt       time_steps_max = 100;   /* default max timesteps */
  PetscDraw      draw;                   /* drawing context */
  PetscInt       steps, m;
  PetscMPIInt    size;
  PetscReal      dt;
  PetscReal      ftime;
  PetscBool      flg;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  m    = 60;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug));

  appctx.m        = m;
  appctx.h        = 1.0/(m-1.0);
  appctx.norm_2   = 0.0;
  appctx.norm_max = 0.0;

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vector data structures for approximate and exact solutions
  */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&u));
  PetscCall(VecDuplicate(u,&appctx.solution));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up displays to show graphs of the solution and error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscViewerDrawOpen(PETSC_COMM_SELF,0,"",80,380,400,160,&appctx.viewer1));
  PetscCall(PetscViewerDrawGetDraw(appctx.viewer1,0,&draw));
  PetscCall(PetscDrawSetDoubleBuffer(draw));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_SELF,0,"",80,0,400,160,&appctx.viewer2));
  PetscCall(PetscViewerDrawGetDraw(appctx.viewer2,0,&draw));
  PetscCall(PetscDrawSetDoubleBuffer(draw));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(TSCreate(PETSC_COMM_SELF,&ts));
  PetscCall(TSSetProblemType(ts,TS_LINEAR));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(TSMonitorSet(ts,Monitor,&appctx,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-time_dependent_rhs",&flg));
  if (flg) {
    /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
    PetscCall(TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx));
    PetscCall(TSSetRHSJacobian(ts,A,A,RHSMatrixHeat,&appctx));
  } else {
    /*
       For linear problems with a time-independent f(u) in the equation
       u_t = f(u), the user provides the discretized right-hand-side
       as a matrix only once, and then sets a null matrix evaluation
       routine.
    */
    PetscCall(RHSMatrixHeat(ts,0.0,u,A,A,&appctx));
    PetscCall(TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx));
    PetscCall(TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  dt   = appctx.h*appctx.h/2.0;
  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetSolution(ts,u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_max_time <maxtime>
     to override the defaults set by TSSetMaxSteps()/TSSetMaxTime().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(TSSetMaxSteps(ts,time_steps_max));
  PetscCall(TSSetMaxTime(ts,time_total_max));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Evaluate initial conditions
  */
  PetscCall(InitialConditions(u,&appctx));

  /*
     Run the timestepping solver
  */
  PetscCall(TSSolve(ts,u));
  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"avg. error (2 norm) = %g, avg. error (max norm) = %g\n",(double)(appctx.norm_2/steps),(double)(appctx.norm_max/steps)));
  PetscCall(TSView(ts,PETSC_VIEWER_STDOUT_SELF));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(TSDestroy(&ts));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&u));
  PetscCall(PetscViewerDestroy(&appctx.viewer1));
  PetscCall(PetscViewerDestroy(&appctx.viewer2));
  PetscCall(VecDestroy(&appctx.solution));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
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
  PetscScalar    *u_localptr;
  PetscInt       i;

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  PetscCall(VecGetArray(u,&u_localptr));

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=0; i<appctx->m; i++) u_localptr[i] = PetscSinReal(PETSC_PI*i*6.*appctx->h) + 3.*PetscSinReal(PETSC_PI*i*2.*appctx->h);

  /*
     Restore vector
  */
  PetscCall(VecRestoreArray(u,&u_localptr));

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
     PetscCall(VecView(u,PETSC_VIEWER_STDOUT_SELF));
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
  PetscScalar    *s_localptr, h = appctx->h, ex1, ex2, sc1, sc2;
  PetscInt       i;

  /*
     Get a pointer to vector data.
  */
  PetscCall(VecGetArray(solution,&s_localptr));

  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  ex1 = PetscExpReal(-36.*PETSC_PI*PETSC_PI*t); ex2 = PetscExpReal(-4.*PETSC_PI*PETSC_PI*t);
  sc1 = PETSC_PI*6.*h;                 sc2 = PETSC_PI*2.*h;
  for (i=0; i<appctx->m; i++) s_localptr[i] = PetscSinReal(PetscRealPart(sc1)*(PetscReal)i)*ex1 + 3.*PetscSinReal(PetscRealPart(sc2)*(PetscReal)i)*ex2;

  /*
     Restore vector
  */
  PetscCall(VecRestoreArray(solution,&s_localptr));
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   Monitor - User-provided routine to monitor the solution computed at
   each timestep.  This example plots the solution and computes the
   error in two different norms.

   This example also demonstrates changing the timestep via TSSetTimeStep().

   Input Parameters:
   ts     - the timestep context
   step   - the count of the current step (with 0 meaning the
             initial condition)
   crtime  - the current time
   u      - the solution at this timestep
   ctx    - the user-provided context for this monitoring routine.
            In this case we use the application context which contains
            information about the problem size, workspace and the exact
            solution.
*/
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal crtime,Vec u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
  PetscReal      norm_2, norm_max, dt, dttol;
  PetscBool      flg;

  /*
     View a graph of the current iterate
  */
  PetscCall(VecView(u,appctx->viewer2));

  /*
     Compute the exact solution
  */
  PetscCall(ExactSolution(crtime,appctx->solution,appctx));

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Computed solution vector\n"));
    PetscCall(VecView(u,PETSC_VIEWER_STDOUT_SELF));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Exact solution vector\n"));
    PetscCall(VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF));
  }

  /*
     Compute the 2-norm and max-norm of the error
  */
  PetscCall(VecAXPY(appctx->solution,-1.0,u));
  PetscCall(VecNorm(appctx->solution,NORM_2,&norm_2));
  norm_2 = PetscSqrtReal(appctx->h)*norm_2;
  PetscCall(VecNorm(appctx->solution,NORM_MAX,&norm_max));

  PetscCall(TSGetTimeStep(ts,&dt));
  if (norm_2 > 1.e-2) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Timestep %D: step size = %g, time = %g, 2-norm error = %g, max norm error = %g\n",step,(double)dt,(double)crtime,(double)norm_2,(double)norm_max));
  }
  appctx->norm_2   += norm_2;
  appctx->norm_max += norm_max;

  dttol = .0001;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dttol",&dttol,&flg));
  if (dt < dttol) {
    dt  *= .999;
    PetscCall(TSSetTimeStep(ts,dt));
  }

  /*
     View a graph of the error
  */
  PetscCall(VecView(appctx->solution,appctx->viewer1));

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error vector\n"));
    PetscCall(VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF));
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
   Recall that MatSetValues() uses 0-based row and column numbers
   in Fortran as well as in C.
*/
PetscErrorCode RHSMatrixHeat(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;                /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*) ctx;      /* user-defined application context */
  PetscInt       mstart  = 0;
  PetscInt       mend    = appctx->m;
  PetscInt       i, idx[3];
  PetscScalar    v[3], stwo = -2./(appctx->h*appctx->h), sone = -.5*stwo;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set matrix rows corresponding to boundary data
  */

  mstart = 0;
  v[0]   = 1.0;
  PetscCall(MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES));
  mstart++;

  mend--;
  v[0] = 1.0;
  PetscCall(MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES));

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
  v[0] = sone; v[1] = stwo; v[2] = sone;
  for (i=mstart; i<mend; i++) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    PetscCall(MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES));
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
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));

  return 0;
}
/* --------------------------------------------------------------------- */
/*
   Input Parameters:
   ts - the TS context
   t - current time
   f - function
   ctx - optional user-defined context, as set by TSetBCFunction()
 */
PetscErrorCode MyBCRoutine(TS ts,PetscReal t,Vec f,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;      /* user-defined application context */
  PetscInt       m = appctx->m;
  PetscScalar    *fa;

  PetscCall(VecGetArray(f,&fa));
  fa[0]   = 0.0;
  fa[m-1] = 1.0;
  PetscCall(VecRestoreArray(f,&fa));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"t=%g\n",(double)t));

  return 0;
}

/*TEST

    test:
      args: -nox -ts_max_steps 4

TEST*/
