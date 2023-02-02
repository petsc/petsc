
static char help[] = "Tests PetscObjectSetOptions() for TS object\n\n";

/* ------------------------------------------------------------------------

   This program solves the PDE

               u * u_xx
         u_t = ---------
               2*(t+1)^2

    on the domain 0 <= x <= 1, with boundary conditions
         u(t,0) = t + 1,  u(t,1) = 2*t + 2,
    and initial condition
         u(0,x) = 1 + x*x.

    The exact solution is:
         u(t,x) = (1 + x*x) * (1 + t)

    Note that since the solution is linear in time and quadratic in x,
    the finite difference scheme actually computes the "exact" solution.

    We use by default the backward Euler method.

  ------------------------------------------------------------------------- */

/*
   Include "petscts.h" to use the PETSc timestepping routines. Note that
   this file automatically includes "petscsys.h" and other lower-level
   PETSc include files.

   Include the "petscdmda.h" to allow us to use the distributed array data
   structures to manage the parallel grid.
*/
#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

/*
   User-defined application context - contains data needed by the
   application-provided callback routines.
*/
typedef struct {
  MPI_Comm  comm;      /* communicator */
  DM        da;        /* distributed array data structure */
  Vec       localwork; /* local ghosted work vector */
  Vec       u_local;   /* local ghosted approximate solution vector */
  Vec       solution;  /* global exact solution vector */
  PetscInt  m;         /* total number of grid points */
  PetscReal h;         /* mesh width: h = 1/(m-1) */
} AppCtx;

/*
   User-defined routines, provided below.
*/
extern PetscErrorCode InitialConditions(Vec, AppCtx *);
extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode ExactSolution(PetscReal, Vec, AppCtx *);

int main(int argc, char **argv)
{
  AppCtx       appctx;               /* user-defined application context */
  TS           ts;                   /* timestepping context */
  Mat          A;                    /* Jacobian matrix data structure */
  Vec          u;                    /* approximate solution vector */
  PetscInt     time_steps_max = 100; /* default max timesteps */
  PetscReal    dt;
  PetscReal    time_total_max = 100.0; /* default max total time */
  PetscOptions options, optionscopy;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(PetscOptionsCreate(&options));
  PetscCall(PetscOptionsSetValue(options, "-ts_monitor", "ascii"));
  PetscCall(PetscOptionsSetValue(options, "-snes_monitor", "ascii"));
  PetscCall(PetscOptionsSetValue(options, "-ksp_monitor", "ascii"));

  appctx.comm = PETSC_COMM_WORLD;
  appctx.m    = 60;

  PetscCall(PetscOptionsGetInt(options, NULL, "-M", &appctx.m, NULL));

  appctx.h = 1.0 / (appctx.m - 1.0);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are M
     total grid values spread equally among all the processors.
  */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, appctx.m, 1, 1, NULL, &appctx.da));
  PetscCall(PetscObjectSetOptions((PetscObject)appctx.da, options));
  PetscCall(DMSetFromOptions(appctx.da));
  PetscCall(DMSetUp(appctx.da));

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */
  PetscCall(DMCreateGlobalVector(appctx.da, &u));
  PetscCall(DMCreateLocalVector(appctx.da, &appctx.u_local));

  /*
     Create local work vector for use in evaluating right-hand-side function;
     create global work vector for storing exact solution.
  */
  PetscCall(VecDuplicate(appctx.u_local, &appctx.localwork));
  PetscCall(VecDuplicate(u, &appctx.solution));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context; set callback routine for
     right-hand-side function evaluation.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(PetscObjectSetOptions((PetscObject)ts, options));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     For nonlinear problems, the user can provide a Jacobian evaluation
     routine (or use a finite differencing approximation).

     Create matrix data structure; set Jacobian evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(PetscObjectSetOptions((PetscObject)A, options));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, appctx.m, appctx.m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(TSSetRHSJacobian(ts, A, A, RHSJacobian, &appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  dt = appctx.h / 2.0;
  PetscCall(TSSetTimeStep(ts, dt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_max_time <maxtime>
     to override the defaults set by TSSetMaxSteps()/TSSetMaxTime().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxSteps(ts, time_steps_max));
  PetscCall(TSSetMaxTime(ts, time_total_max));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Evaluate initial conditions
  */
  PetscCall(InitialConditions(u, &appctx));

  /*
     Run the timestepping solver
  */
  PetscCall(TSSolve(ts, u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscObjectGetOptions((PetscObject)ts, &optionscopy));
  PetscCheck(options == optionscopy, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscObjectGetOptions() failed");

  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&appctx.da));
  PetscCall(VecDestroy(&appctx.localwork));
  PetscCall(VecDestroy(&appctx.solution));
  PetscCall(VecDestroy(&appctx.u_local));
  PetscCall(PetscOptionsDestroy(&options));

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

   Input Parameters:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u, AppCtx *appctx)
{
  PetscScalar *u_localptr, h = appctx->h, x;
  PetscInt     i, mybase, myend;

  PetscFunctionBeginUser;
  /*
     Determine starting point of each processor's range of
     grid values.
  */
  PetscCall(VecGetOwnershipRange(u, &mybase, &myend));

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  PetscCall(VecGetArray(u, &u_localptr));

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i = mybase; i < myend; i++) {
    x                      = h * (PetscReal)i; /* current location in global grid */
    u_localptr[i - mybase] = 1.0 + x * x;
  }

  /*
     Restore vector
  */
  PetscCall(VecRestoreArray(u, &u_localptr));

  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode ExactSolution(PetscReal t, Vec solution, AppCtx *appctx)
{
  PetscScalar *s_localptr, h = appctx->h, x;
  PetscInt     i, mybase, myend;

  PetscFunctionBeginUser;
  /*
     Determine starting and ending points of each processor's
     range of grid values
  */
  PetscCall(VecGetOwnershipRange(solution, &mybase, &myend));

  /*
     Get a pointer to vector data.
  */
  PetscCall(VecGetArray(solution, &s_localptr));

  /*
     Simply write the solution directly into the array locations.
     Alternatively, we could use VecSetValues() or VecSetValuesLocal().
  */
  for (i = mybase; i < myend; i++) {
    x                      = h * (PetscReal)i;
    s_localptr[i - mybase] = (t + 1.0) * (1.0 + x * x);
  }

  /*
     Restore vector
  */
  PetscCall(VecRestoreArray(solution, &s_localptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* --------------------------------------------------------------------- */
/*
   RHSFunction - User-provided routine that evalues the right-hand-side
   function of the ODE.  This routine is set in the main program by
   calling TSSetRHSFunction().  We compute:
          global_out = F(global_in)

   Input Parameters:
   ts         - timesteping context
   t          - current time
   global_in  - vector containing the current iterate
   ctx        - (optional) user-provided context for function evaluation.
                In this case we use the appctx defined above.

   Output Parameter:
   global_out - vector containing the newly evaluated function
*/
PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec global_in, Vec global_out, void *ctx)
{
  AppCtx            *appctx    = (AppCtx *)ctx;     /* user-defined application context */
  DM                 da        = appctx->da;        /* distributed array */
  Vec                local_in  = appctx->u_local;   /* local ghosted input vector */
  Vec                localwork = appctx->localwork; /* local ghosted work vector */
  PetscInt           i, localsize;
  PetscMPIInt        rank, size;
  PetscScalar       *copyptr, sc;
  const PetscScalar *localptr;

  PetscFunctionBeginUser;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get ready for local function computations
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, global_in, INSERT_VALUES, local_in));
  PetscCall(DMGlobalToLocalEnd(da, global_in, INSERT_VALUES, local_in));

  /*
      Access directly the values in our local INPUT work array
  */
  PetscCall(VecGetArrayRead(local_in, &localptr));

  /*
      Access directly the values in our local OUTPUT work array
  */
  PetscCall(VecGetArray(localwork, &copyptr));

  sc = 1.0 / (appctx->h * appctx->h * 2.0 * (1.0 + t) * (1.0 + t));

  /*
      Evaluate our function on the nodes owned by this processor
  */
  PetscCall(VecGetLocalSize(local_in, &localsize));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Handle boundary conditions: This is done by using the boundary condition
        u(t,boundary) = g(t,boundary)
     for some function g. Now take the derivative with respect to t to obtain
        u_{t}(t,boundary) = g_{t}(t,boundary)

     In our case, u(t,0) = t + 1, so that u_{t}(t,0) = 1
             and  u(t,1) = 2t+ 2, so that u_{t}(t,1) = 2
  */
  PetscCallMPI(MPI_Comm_rank(appctx->comm, &rank));
  PetscCallMPI(MPI_Comm_size(appctx->comm, &size));
  if (rank == 0) copyptr[0] = 1.0;
  if (rank == size - 1) copyptr[localsize - 1] = 2.0;

  /*
     Handle the interior nodes where the PDE is replace by finite
     difference operators.
  */
  for (i = 1; i < localsize - 1; i++) copyptr[i] = localptr[i] * sc * (localptr[i + 1] + localptr[i - 1] - 2.0 * localptr[i]);

  /*
     Restore vectors
  */
  PetscCall(VecRestoreArrayRead(local_in, &localptr));
  PetscCall(VecRestoreArray(localwork, &copyptr));

  /*
     Insert values from the local OUTPUT vector into the global
     output vector
  */
  PetscCall(DMLocalToGlobalBegin(da, localwork, INSERT_VALUES, global_out));
  PetscCall(DMLocalToGlobalEnd(da, localwork, INSERT_VALUES, global_out));

  PetscFunctionReturn(PETSC_SUCCESS);
}
/* --------------------------------------------------------------------- */
/*
   RHSJacobian - User-provided routine to compute the Jacobian of
   the nonlinear right-hand-side function of the ODE.

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
  RHSJacobian computes entries for the locally owned part of the Jacobian.
   - Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.
   - Each processor needs to insert only elements that it owns
     locally (but any non-local elements will be sent to the
     appropriate processor during matrix assembly).
   - Always specify global row and columns of matrix entries when
     using MatSetValues().
   - Here, we set all entries for a particular row at once.
   - Note that MatSetValues() uses 0-based row and column numbers
     in Fortran as well as in C.
*/
PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec global_in, Mat AA, Mat BB, void *ctx)
{
  AppCtx            *appctx   = (AppCtx *)ctx;   /* user-defined application context */
  Vec                local_in = appctx->u_local; /* local ghosted input vector */
  DM                 da       = appctx->da;      /* distributed array */
  PetscScalar        v[3], sc;
  const PetscScalar *localptr;
  PetscInt           i, mstart, mend, mstarts, mends, idx[3], is;

  PetscFunctionBeginUser;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get ready for local Jacobian computations
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, global_in, INSERT_VALUES, local_in));
  PetscCall(DMGlobalToLocalEnd(da, global_in, INSERT_VALUES, local_in));

  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(local_in, &localptr));

  /*
     Get starting and ending locally owned rows of the matrix
  */
  PetscCall(MatGetOwnershipRange(BB, &mstarts, &mends));
  mstart = mstarts;
  mend   = mends;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set matrix rows corresponding to boundary data
  */
  if (mstart == 0) {
    v[0] = 0.0;
    PetscCall(MatSetValues(BB, 1, &mstart, 1, &mstart, v, INSERT_VALUES));
    mstart++;
  }
  if (mend == appctx->m) {
    mend--;
    v[0] = 0.0;
    PetscCall(MatSetValues(BB, 1, &mend, 1, &mend, v, INSERT_VALUES));
  }

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
  sc = 1.0 / (appctx->h * appctx->h * 2.0 * (1.0 + t) * (1.0 + t));
  for (i = mstart; i < mend; i++) {
    idx[0] = i - 1;
    idx[1] = i;
    idx[2] = i + 1;
    is     = i - mstart + 1;
    v[0]   = sc * localptr[is];
    v[1]   = sc * (localptr[is + 1] + localptr[is - 1] - 4.0 * localptr[is]);
    v[2]   = sc * localptr[is];
    PetscCall(MatSetValues(BB, 1, &i, 3, idx, v, INSERT_VALUES));
  }

  /*
     Restore vector
  */
  PetscCall(VecRestoreArrayRead(local_in, &localptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Complete the matrix assembly process and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(MatAssemblyBegin(BB, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(BB, MAT_FINAL_ASSEMBLY));
  if (BB != AA) {
    PetscCall(MatAssemblyBegin(AA, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(AA, MAT_FINAL_ASSEMBLY));
  }

  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  PetscCall(MatSetOption(BB, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      requires: !single

TEST*/
