#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.11 1997/08/07 14:41:33 bsmith Exp bsmith $";
#endif
static char help[] ="Solves a simple time-dependent nonlinear PDE using implicit timestepping";

/*
   Concepts: TS^time-dependent nonlinear problems
   Routines: TSCreate(); TSSetSolution(); TSSetRHSFunction(); TSSetRHSJacobian();
   Routines: TSSetType(); TSSetInitialTimeStep(); TSSetDuration();
   Routines: TSSetFromOptions(); TSStep(); TSDestroy(); TSSetMonitor();
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program solves:

                       U * U_xx 
                 U_t = ---------
                       2*(t+1)^2 

            U(0,x) = 1 + x*x; U(t,0) = t + 1; U(t,1) = 2*t + 2

    The exact solution is U(t,x) = (1 + x*x) * (1 + t)

    Note that since the solution is linear in time and quadratic in x,
    the finite difference scheme actually computes the "exact" solution.

    We use the backward Euler method.

  ------------------------------------------------------------------------- */

/*
   Include "ts.h" to use the PETSc timestepping routines. Note that
   this file automatically includes "petsc.h" and other lower-level
   PETSc include files.

   Include the "da.h" to allow us to use the distributed array data 
   structures to manage the parallel "grid".
*/
#include "ts.h"
#include "da.h"
#include <math.h>

typedef struct {
  MPI_Comm comm;
  Vec      localwork,solution;    /* location for local work (with ghost points) vector */
  DA       da;                    /* manages ghost point communication */
  int      M;                     /* total number of grid points */
  double   h;                     /* mesh width: h = 1/(M-1) */
  int      debug;                 /* flag (1 indicates debugging printouts) */
} AppCtx;

/* 
   User-defined routines, provided below.
*/
int Monitor(TS, int, double , Vec, void *);
int RHSFunction(TS,double,Vec,Vec,void*);
int InitialConditions(Vec, void*);
int RHSJacobian(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);

/*
   Utility routine for finite difference Jacobian approximation
*/
extern int RHSJacobianFD(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);

int main(int argc,char **argv)
{
  int           ierr,  time_steps = 1000, steps, flg;
  AppCtx        appctx;
  Vec           local, global;
  double        dt,ftime;
  TS            ts;
  Mat           A;
 
  PetscInitialize(&argc,&argv,(char*)0,help);

  appctx.comm = PETSC_COMM_WORLD;
  appctx.M    = 60;
  ierr = OptionsGetInt(PETSC_NULL,"-M",&appctx.M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-time",&time_steps,&flg);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-debug",&appctx.debug); CHKERRA(ierr);
  appctx.h = 1.0/(appctx.M-1.0);
  dt       = appctx.h/2.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DA) to manage parallel grid and vectors
     Set up the ghost point communication pattern.  There are M total
     grid values spread equally among all the processors.
  */ 
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,appctx.M,1,1,PETSC_NULL,
                    &appctx.da); CHKERRA(ierr);

  /*
     Extract global and local vectors from DA; then duplicate for remaining
     vectors that are the same types.
  */ 
  ierr = DAGetDistributedVector(appctx.da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(appctx.da,&local); CHKERRA(ierr);

  /* Make local work vector for evaluating right-hand-side function */
  ierr = VecDuplicate(local,&appctx.localwork); CHKERRA(ierr);

  /* Make global work vector for storing exact solution */
  ierr = VecDuplicate(global,&appctx.solution); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = InitialConditions(global,&appctx); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context; set various callback routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_WORLD,TS_NONLINEAR,&ts); CHKERRA(ierr);
  ierr = TSSetMonitor(ts,Monitor,&appctx); CHKERRA(ierr);
  ierr = TSSetRHSFunction(ts,RHSFunction,&appctx); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine;
     Set the solution method to be backward Euler.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,appctx.M,appctx.M,&A); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-fdjac",&flg); CHKERRA(ierr);
  if (flg) {
    ierr = TSSetRHSJacobian(ts,A,A,RHSJacobianFD,&appctx); CHKERRA(ierr);
  } else {
    ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,&appctx); CHKERRA(ierr);
  }
  ierr = TSSetType(ts,TS_BEULER); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver; set runtime options.
     We can override the defaults set by TSSetDuration() with
          -ts_max_steps <maxsteps> -ts_max_time <maxtime>
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetDuration(ts,time_steps,100.); CHKERRA(ierr);
  ierr = TSSetFromOptions(ts); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector, initial timestep, and total duration.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetInitialTimeStep(ts,0.0,dt); CHKERRA(ierr);
  ierr = TSSetSolution(ts,global); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up and run the timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetUp(ts); CHKERRA(ierr);
  ierr = TSStep(ts,&steps,&ftime); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSDestroy(ts); CHKERRA(ierr);
  ierr = VecDestroy(appctx.localwork); CHKERRA(ierr);
  ierr = VecDestroy(appctx.solution); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(appctx.da); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  PetscFinalize();
  return 0;
}

/* -------------------------------------------------------------------*/
/*
   InitialConditions - Computes the solution at the initial time. 

   Input Parameter:
   global - uninitialized solution vector

   Output Parameter:
   global - vector with solution at initial time
*/ 
int InitialConditions(Vec global, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr, h = appctx->h, x;
  int    i, mybase, myend, ierr;

  /* 
     Determine starting point of each processor's range of
     grid values.
  */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRQ(ierr);

  /* 
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  ierr = VecGetArray(global,&localptr); CHKERRQ(ierr);

  /* 
     We initialize the solution array by simply writing the solution
     directly into the array locations.
  */
  for (i=mybase; i<myend; i++) {
    x = h*i; /* current location in global grid */
    localptr[i-mybase] = 1.0 + x*x;
  }

  /* 
     Restore vector
  */
  ierr = VecRestoreArray(global,&localptr); CHKERRQ(ierr);

  /* Print debugging information if desired */
  if (appctx->debug) {
     PetscPrintf(appctx->comm,"initial guess vector");
     ierr = VecView(global,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }

  return 0;
}
/* -------------------------------------------------------------------*/
/*
   ExactSolution - Computes the exact solution at any time.

   Input Parameters:
   t - time
   solution - vector in which exact solution will be computed
   ctx - user-defined context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
int ExactSolution(double t,Vec solution, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr,h = appctx->h,x;
  int    i,mybase,myend,ierr;

  /* 
     Determine starting point of each processor's range of grid values
  */
  ierr = VecGetOwnershipRange(solution,&mybase,&myend); CHKERRQ(ierr);

  /* 
     Get local work array
  */
  ierr = VecGetArray(solution,&localptr); CHKERRQ(ierr);

  /* 
     Simply write the solution directly into the array locations
  */
  for (i=mybase; i<myend; i++) {
    x = i*h;
    localptr[i-mybase] = (t + 1.0)*(1.0 + x*x);
  }

  /* 
     Restore vector
   */
  ierr = VecRestoreArray(solution,&localptr); CHKERRQ(ierr);

  return 0;
}
/* -------------------------------------------------------------------*/
/*
   Monitor - A user provided routine to monitor the solution computed at 
   each time-step. This example plots the solution and computes the
   error in two different norms.

   Input Parameters:
   ts     - the time-step context
   step   - the count of the current step (with 0 meaning the
             initial condition)
   time   - the current time
   ctx    - the user-provided context for this monitoring routine.
            In this case we use the application context which contains 
            information about the problem size, workspace and the exact 
            solution.

   Output Parameter:
   global - the solution at this timestep
*/
int Monitor(TS ts,int step,double time,Vec global, void *ctx)
{
  AppCtx   *appctx = (AppCtx*) ctx;
  int      ierr;
  double   en2, en2s, enmax;
  Scalar   mone = -1.0;
  Draw     draw;

  /*
     We use the default X windows viewer
             VIEWER_DRAWX_(appctx->comm)
     that is associated with the current communicator. This saves
     the effort of calling ViewerDrawOpenX() to create the window.
     Note that if we wished to plot several items in separate windows we
     would create each viewer with ViewerDrawOpenX() and store them in
     the application context, appctx.

     Double buffering makes graphics look better.
  */
  ierr = ViewerDrawGetDraw(VIEWER_DRAWX_(appctx->comm),&draw); CHKERRQ(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRQ(ierr);
  ierr = VecView(global,VIEWER_DRAWX_(appctx->comm)); CHKERRQ(ierr);

  /*
     Compute the exact solution at this timestep
  */
  ierr = ExactSolution(time,appctx->solution, ctx); CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
     PetscPrintf(appctx->comm,"Computed solution vector");
     ierr = VecView(global,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
     PetscPrintf(appctx->comm,"Exact solution vector");
     ierr = VecView(appctx->solution,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }

  /*
     Compute the 2-norm and max-norm of the error
  */
  ierr = VecAXPY(&mone,global,appctx->solution); CHKERRQ(ierr);
  ierr = VecNorm(appctx->solution,NORM_2,&en2); CHKERRQ(ierr);
  en2s  = sqrt(appctx->h)*en2; /* scale the 2-norm by the grid spacing */
  ierr = VecNorm(appctx->solution,NORM_MAX,&enmax); CHKERRQ(ierr);

  /*
      PetscPrintf() causes only the first processor in this 
     communicator to print the time-step information.
  */
  PetscPrintf(appctx->comm,"Timestep %d time %g norm of error -2- %g -max- %g\n",
              step,time,en2s,enmax);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
     PetscPrintf(appctx->comm,"Error vector");
     ierr = VecView(appctx->solution,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }

  return 0;
}
/* -------------------------------------------------------------------*/
/*
   RHSFunction - User-provided routine that evalues the RHS function
   in the ODE.  This routine is set in the main program by calling
   TSSetRHSFunction().  We compute:
          globalout = F(globalin)

   Input Parameters:
   ts        - timestep context
   t         - current time
   globalin  - input vector to function
   ctx       - (optional) user-provided context for function.  In our
                case we use the appctx defined above.

   Output Parameter:
   globalout - value of function
*/
int RHSFunction(TS ts, double t,Vec globalin, Vec globalout, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DA     da = appctx->da;
  Vec    local, localwork = appctx->localwork;
  int    ierr,i,localsize,rank,size; 
  Scalar *copyptr, *localptr,sc;

  /*
       The vector 'local' will be a workspace that contains the ghost region
  */
  ierr = DAGetLocalVector(da,&local); CHKERRQ(ierr);
  
  /*
      Copy the input vector into local and up-date the ghost points
  */
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);

  /*
      Access directly the values in our local INPUT work array
  */
  ierr = VecGetArray(local,&localptr); CHKERRQ(ierr);

  /*
      Access directly the values in our local OUTPUT work array
  */
  ierr = VecGetArray(localwork,&copyptr); CHKERRQ(ierr);

  sc = 1.0/(appctx->h*appctx->h*2.0*(1.0+t)*(1.0+t));

  /*
      Evaluate our function on the nodes owned by this processor
  */
  ierr = VecGetLocalSize(local,&localsize); CHKERRQ(ierr);

  /*
     Handle boundary conditions: This is done by using the boundary condition 
        U(t,boundary) = g(t,boundary) 
     for some function g. Now take the derivative with respect to t to obtain
        U_{t}(t,boundary) = g_{t}(t,boundary)

     In our case, U(t,0) = t + 1; so U_{t}(t,0) = 1 
             and  U(t,1) = 2t+ 1; so U_{t}(t,1) = 2
  */
  MPI_Comm_rank(appctx->comm,&rank);
  MPI_Comm_size(appctx->comm,&size);
  if (rank == 0)      copyptr[0]           = 1.0;
  if (rank == size-1) copyptr[localsize-1] = 2.0;

  /*
      Handle the interior nodes where the PDE is replace by finite 
     difference operators.
  */
  for (i=1; i<localsize-1; i++) {
    copyptr[i] =  localptr[i] * sc * (localptr[i+1] + localptr[i-1] - 2.0*localptr[i]);
  }
  ierr = VecRestoreArray(localwork,&copyptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(localwork,&copyptr); CHKERRQ(ierr);

  /*
      Return the values from our local OUTPUT array into our global 
    output array
  */
  ierr = DALocalToGlobal(da,localwork,INSERT_VALUES,globalout); CHKERRQ(ierr);

  /* Print debugging information if desired */
  if (appctx->debug) {
     PetscPrintf(appctx->comm,"RHS function vector");
     ierr = VecView(globalout,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }

  return 0;
}
/* -------------------------------------------------------------------*/
/*
   RHSJacobian - User provided routine to compute the Jacobian of
   the right-hand-side function.

   Input Parameters:
     ts - the TS context
     t - current time
     globalin - global input vector
     dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
     AA - Jacobian matrix
     BB - optionally different preconditioning matrix
     str - flag indicating matrix structure

  Notes:
  RHSJacobian computes entries for the locally owned part of the Jacobian.
   - Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors. The "grow"
     parameter computed below specifies the global row number 
     corresponding to each local grid point.
   - Each processor needs to insert only elements that it owns
    locally (but any non-local elements will be sent to the
     appropriate processor during matrix assembly). 
   - Always specify global row and columns of matrix entries.
   - Here, we set all entries for a particular row at once.
   - Note that MatSetValues() uses 0-based row and column numbers
     in Fortran as well as in C.
*/
int RHSJacobian(TS ts,double t,Vec globalin,Mat *AA,Mat *BB, MatStructure *str,void *ctx)
{
  Mat    A = *AA;
  AppCtx *appctx = (AppCtx*) ctx;
  int    ierr, i, mstart, mend, mstarts, mends, idx[3], is;
  Scalar v[3];
  DA     da = appctx->da;
  Vec    local;
  Scalar *localptr, sc;

  /* Extract local array */ 
  ierr = DAGetLocalVector(da,&local); CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);

  /* Get a pointer to vector data */
  ierr = VecGetArray(local,&localptr); CHKERRQ(ierr);

  /* Set matrix sparsity info */
  *str = SAME_NONZERO_PATTERN;

  /* Get starting and ending locally owned rows of the matrix */
  ierr = MatGetOwnershipRange(A,&mstarts,&mends); CHKERRQ(ierr);
  mstart = mstarts; mend = mends;

  /* Set matrix rows corresponding to boundary data */
  if (mstart == 0) {
    v[0] = 0.0;
    ierr = MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES); CHKERRQ(ierr);
    mstart++;
  }
  if (mend == appctx->M) {
    mend--;
    v[0] = 0.0;
    ierr = MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES); CHKERRQ(ierr);
  }

  /*
     Set matrix rows corresponding to interior data.
     We construct matrix one row at a time
  */
  sc = 1.0/(appctx->h*appctx->h*2.0*(1.0+t)*(1.0+t));
  for ( i=mstart; i<mend; i++ ) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    is     = i - mstart + 1;
    v[0]   = sc*localptr[is];
    v[1]   = sc*(localptr[is+1] + localptr[is-1] - 4.0*localptr[is]);
    v[2]   = sc*localptr[is];
    ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);
  }
  /* 
     Restore vector
  */
  ierr = VecRestoreArray(local,&localptr); CHKERRQ(ierr);

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}





