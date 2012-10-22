
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
   Routines: TSCreate(); TSSetSolution(); TSSetRHSJacobian(), TSSetIJacobian();
   Routines: TSSetInitialTimeStep(); TSSetDuration(); TSMonitorSet();
   Routines: TSSetFromOptions(); TSStep(); TSDestroy();
   Routines: TSSetTimeStep(); TSGetTimeStep();
   Processors: 1
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
       ex3 -ts_type <timestepping solver>

   We compare the approximate solution with the exact solution, given by
       u_exact(x,t) = exp(-36*pi*pi*t) * sin(6*pi*x) +
                      3*exp(-4*pi*pi*t) * sin(2*pi*x)

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely
     - time-dependent f:   f(u,t) is a function of t
     - time-independent f: f(u,t) is simply f(u)

    The parallel version of this code is ts/examples/tutorials/ex4.c

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
extern PetscErrorCode RHSMatrixHeat(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);
extern PetscErrorCode MyBCRoutine(TS,PetscReal,Vec,void*);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      time_total_max = 100.0; /* default max total time */
  PetscInt       time_steps_max = 100;   /* default max timesteps */
  PetscDraw      draw;                   /* drawing context */
  PetscErrorCode ierr;
  PetscInt       steps, m;
  PetscMPIInt    size;
  PetscReal      dt;
  PetscReal      ftime;
  PetscBool      flg;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  m    = 60;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-debug",&appctx.debug);CHKERRQ(ierr);
  appctx.m        = m;
  appctx.h        = 1.0/(m-1.0);
  appctx.norm_2   = 0.0;
  appctx.norm_max = 0.0;
  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vector data structures for approximate and exact solutions
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.solution);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up displays to show graphs of the solution and error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"",80,380,400,160,&appctx.viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(appctx.viewer1,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"",80,0,400,160,&appctx.viewer2);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(appctx.viewer2,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSMonitorSet(ts,Monitor,&appctx,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-time_dependent_rhs",&flg);CHKERRQ(ierr);
  if (flg) {
    /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
    ierr = TSSetRHSFunction(ts,PETSC_NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,A,A,RHSMatrixHeat,&appctx);CHKERRQ(ierr);
  } else {
    /*
       For linear problems with a time-independent f(u) in the equation
       u_t = f(u), the user provides the discretized right-hand-side
       as a matrix only once, and then sets a null matrix evaluation
       routine.
    */
    MatStructure A_structure;
    ierr = RHSMatrixHeat(ts,0.0,u,&A,&A,&A_structure,&appctx);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,PETSC_NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  dt = appctx.h*appctx.h/2.0;
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_final_time <maxtime>
     to override the defaults set by TSSetDuration().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetDuration(ts,time_steps_max,time_total_max);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Evaluate initial conditions
  */
  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);

  /*
     Run the timestepping solver
  */
  ierr = TSSolve(ts,u,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscPrintf(PETSC_COMM_SELF,"avg. error (2 norm) = %G, avg. error (max norm) = %G\n",
              appctx.norm_2/steps,appctx.norm_max/steps);
  ierr = TSView(ts,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.viewer2);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.solution);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
  PetscFinalize();
  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "InitialConditions"
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
  PetscErrorCode ierr;

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  ierr = VecGetArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=0; i<appctx->m; i++) {
    u_localptr[i] = sin(PETSC_PI*i*6.*appctx->h) + 3.*sin(PETSC_PI*i*2.*appctx->h);
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
     printf("initial guess vector\n");
     ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "ExactSolution"
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
  PetscErrorCode ierr;

  /*
     Get a pointer to vector data.
  */
  ierr = VecGetArray(solution,&s_localptr);CHKERRQ(ierr);

  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  ex1 = exp(-36.*PETSC_PI*PETSC_PI*t); ex2 = exp(-4.*PETSC_PI*PETSC_PI*t);
  sc1 = PETSC_PI*6.*h;                 sc2 = PETSC_PI*2.*h;
  for (i=0; i<appctx->m; i++) {
    s_localptr[i] = sin(PetscRealPart(sc1)*(PetscReal)i)*ex1 + 3.*sin(PetscRealPart(sc2)*(PetscReal)i)*ex2;
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(solution,&s_localptr);CHKERRQ(ierr);
  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "Monitor"
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
  PetscErrorCode ierr;
  PetscReal      norm_2, norm_max, dt, dttol;
  PetscBool      flg;

  /*
     View a graph of the current iterate
  */
  ierr = VecView(u,appctx->viewer2);CHKERRQ(ierr);

  /*
     Compute the exact solution
  */
  ierr = ExactSolution(crtime,appctx->solution,appctx);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
     printf("Computed solution vector\n");
     ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
     printf("Exact solution vector\n");
     ierr = VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  /*
     Compute the 2-norm and max-norm of the error
  */
  ierr = VecAXPY(appctx->solution,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(appctx->solution,NORM_2,&norm_2);CHKERRQ(ierr);
  norm_2 = PetscSqrtReal(appctx->h)*norm_2;
  ierr = VecNorm(appctx->solution,NORM_MAX,&norm_max);CHKERRQ(ierr);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (norm_2 > 1.e-2){
    printf("Timestep %d: step size = %G, time = %G, 2-norm error = %G, max norm error = %G\n",
         (int)step,dt,crtime,norm_2,norm_max);
  }
  appctx->norm_2   += norm_2;
  appctx->norm_max += norm_max;

  dttol = .0001;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-dttol",&dttol,&flg);CHKERRQ(ierr);
  if (dt < dttol) {
    dt *= .999;
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  }

  /*
     View a graph of the error
  */
  ierr = VecView(appctx->solution,appctx->viewer1);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
     printf("Error vector\n");
     ierr = VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "RHSMatrixHeat"
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
PetscErrorCode RHSMatrixHeat(TS ts,PetscReal t,Vec X,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  Mat            A = *AA;                      /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx *) ctx;     /* user-defined application context */
  PetscInt       mstart = 0;
  PetscInt       mend = appctx->m;
  PetscErrorCode ierr;
  PetscInt       i, idx[3];
  PetscScalar    v[3], stwo = -2./(appctx->h*appctx->h), sone = -.5*stwo;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set matrix rows corresponding to boundary data
  */

  mstart = 0;
  v[0] = 1.0;
  ierr = MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES);CHKERRQ(ierr);
  mstart++;

  mend--;
  v[0] = 1.0;
  ierr = MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES);CHKERRQ(ierr);

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
  v[0] = sone; v[1] = stwo; v[2] = sone;
  for ( i=mstart; i<mend; i++ ) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);
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
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Set flag to indicate that the Jacobian matrix retains an identical
     nonzero structure throughout all timestepping iterations (although the
     values of the entries change). Thus, we can save some work in setting
     up the preconditioner (e.g., no need to redo symbolic factorization for
     ILU/ICC preconditioners).
      - If the nonzero structure of the matrix is different during
        successive linear solves, then the flag DIFFERENT_NONZERO_PATTERN
        must be used instead.  If you are unsure whether the matrix
        structure has changed or not, use the flag DIFFERENT_NONZERO_PATTERN.
      - Caution:  If you specify SAME_NONZERO_PATTERN, PETSc
        believes your assertion and does not check the structure
        of the matrix.  If you erroneously claim that the structure
        is the same when it actually is not, the new preconditioner
        will not function correctly.  Thus, use this optimization
        feature with caution!
  */
  *str = SAME_NONZERO_PATTERN;

  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "MYBCRoutine"
/*
   Input Parameters:
   ts - the TS context
   t - current time
   f - function
   ctx - optional user-defined context, as set by TSetBCFunction()
 */
PetscErrorCode MyBCRoutine(TS ts,PetscReal t,Vec f,void *ctx)
{
  AppCtx         *appctx = (AppCtx *) ctx;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       m = appctx->m;
  PetscScalar    *fa;

  ierr = VecGetArray(f,&fa);CHKERRQ(ierr);
  fa[0] = 0.0;
  fa[m-1] = 1.0;
  ierr = VecRestoreArray(f,&fa);CHKERRQ(ierr);
  printf("t=%g\n",t);

  return 0;
}
