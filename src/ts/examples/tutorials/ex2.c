#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.2 1997/04/11 22:01:52 bsmith Exp bsmith $";
#endif
static char help[] ="Solves a simple time PDE using implicit timestepping";

/*
   Concepts: TS^timestepping^nonlinear problems
   Routines: TSCreate(); TSSetSolution(); TSSetRHSFunction(); TSSetRHSJacobian();
   Routines: TSSetType(); TSSetInitialTimeStep(); TSSetDuration();
   Routines: TSSetFromOptions(); TSStep(); TSDestroy(); TSSetMonitor();
   Routines: PetscPrintF();
   Processors: n

*/

/* ------------------------------------------------------------------------
          Solves U_t = U * U_xx 
                       -------
                       2*(t+1)^2 

             U(0,x) = 1 + x*x; U(t,0) = t + 1; U(t,1) = 2*t + 2

      Exact solution is u(t,x) = (1 + x*x) * (1 + t)

      Note that since the solution is linear in time and quadratic in x, the 
    finite difference scheme actually computes the "exact" solution.

    Using backward Eulers method
*/

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
  double   h;                     /* mesh width h = 1/(M-1) */
} AppCtx;

/* 
   User-defined routines, provided below.
*/
int Monitor(TS, int, double , Vec, void *);
int RHSFunction(TS,double,Vec,Vec,void*);
int InitialConditions(Vec, void*);
int RHSJacobian(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);

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
    
  /* 
      Set up the ghost point communication pattern 
    There are appctx.M total grid values spread equally among all the processors.
  */ 
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,appctx.M,1,1,PETSC_NULL,&appctx.da);CHKERRA(ierr);
  ierr = DAGetDistributedVector(appctx.da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(appctx.da,&local); CHKERRA(ierr);

  /* make local work array for evaluating right hand side function */
  ierr = VecDuplicate(local,&appctx.localwork); CHKERRA(ierr);

  /* make global work array for storing exact solution */
  ierr = VecDuplicate(global,&appctx.solution); CHKERRA(ierr);

  appctx.h = 1.0/(appctx.M-1.0);

  /* set initial conditions */
  ierr = InitialConditions(global,&appctx); CHKERRA(ierr);
    
  /* make timestep context */
  ierr = TSCreate(PETSC_COMM_WORLD,TS_NONLINEAR,&ts); CHKERRA(ierr);
  ierr = TSSetMonitor(ts,Monitor,&appctx); CHKERRA(ierr);

  ierr = TSSetRHSFunction(ts,RHSFunction,&appctx); CHKERRA(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,appctx.M,appctx.M,&A); CHKERRA(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,&appctx); CHKERRA(ierr);  

  ierr = TSSetType(ts,TS_BEULER); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRA(ierr);

  dt = appctx.h/2.0;
  ierr = TSSetInitialTimeStep(ts,0.0,dt); CHKERRA(ierr);
  ierr = TSSetDuration(ts,time_steps,100.); CHKERRA(ierr);
  ierr = TSSetSolution(ts,global); CHKERRA(ierr);

  ierr = TSSetUp(ts); CHKERRA(ierr);
  ierr = TSStep(ts,&steps,&ftime); CHKERRA(ierr);

  ierr = TSDestroy(ts); CHKERRA(ierr);
  ierr = VecDestroy(appctx.localwork); CHKERRA(ierr);
  ierr = VecDestroy(appctx.solution); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(appctx.da); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

/* -------------------------------------------------------------------*/
/*
     Computes the solution at the initial time. 
*/ 
int InitialConditions(Vec global, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr,h = appctx->h,x;
  int    i,mybase,myend,ierr;

  /* 
     Determine starting point of each processors range of
    grid values 
  */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRQ(ierr);

  /* 
      Initialize the solution array, by simply writing the solution
    directly into the array locations.
  */
  ierr = VecGetArray(global,&localptr); CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    x = h*i; /* current location in global grid */
    localptr[i-mybase] = 1.0 + x*x;
  }
  ierr = VecRestoreArray(global,&localptr); CHKERRQ(ierr);
  return 0;
}

/*
     Compute the exact solution at any time
*/
int Solution(double t,Vec solution, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr,h = appctx->h,x;
  int    i,mybase,myend,ierr;

  /* 
     Determine starting point of each processors range of
    grid values 
  */
  ierr = VecGetOwnershipRange(solution,&mybase,&myend); CHKERRQ(ierr);

  /* 
     Simply write the solution directly into the array locations.
  */
  ierr = VecGetArray(solution,&localptr); CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    x = i*h;
    localptr[i-mybase] = (t + 1.0)*(1.0 + x*x);
  }
  ierr = VecRestoreArray(solution,&localptr); CHKERRQ(ierr);
  return 0;
}

/*
      A user provided routine to monitor the solution computed at 
  each time-step. This example plots the solution and computes the
  error in two different norms.

    Arguments are: 
        ts     - the time-step context
        step   - the count of the current step; with 0 meaning initial condition
        time   - the current time
        global - the solution at this time-step
        ctx    - the user provided context for this their monitor routine,
                 in this case we use the application context which contains 
                 information about the problem size, workspace and the exact 
                 solution
*/
int Monitor(TS ts, int step, double time,Vec global, void *ctx)
{
  AppCtx   *appctx = (AppCtx*) ctx;
  int      ierr;
  double   norm_2,norm_max;
  Scalar   mone = -1.0;
  Draw     draw;

  /*
       Use the default X windows viewer; VIEWER_DRAWX_(appctx->comm); associated
     with the current communicator. This saves the effort of calling 
     ViewerDrawOpenX() to create the window. Note if we wished to plot several
     items on seperate windows we would create each viewer with ViewerDrawOpenX()
     and store them in the application context, appctx.

     Double buffering makes graphics look better.
  */
  ierr = ViewerDrawGetDraw(VIEWER_DRAWX_(appctx->comm),&draw); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRA(ierr);
  ierr = VecView(global,VIEWER_DRAWX_(appctx->comm)); CHKERRQ(ierr);

  /*
      Compute the exact solution at this time-step, then compute the 
    2-norm and max-norm of the error.
  */
  ierr = Solution(time,appctx->solution, ctx); CHKERRQ(ierr);
  ierr = VecAXPY(&mone,global,appctx->solution); CHKERRQ(ierr);
  ierr = VecNorm(appctx->solution,NORM_2,&norm_2); CHKERRQ(ierr);
  norm_2 = sqrt(appctx->h)*norm_2; /* scale the 2-norm by the grid spacing */
  ierr = VecNorm(appctx->solution,NORM_MAX,&norm_max); CHKERRQ(ierr);

  /*
      PetscPrintf() causes only the first processor in this 
     communicator to print the time-step information.
  */
  PetscPrintf(appctx->comm,"Timestep %d time %g norm of error -2- %g -max- %g\n",step,time,norm_2,norm_max);

  return 0;
}

/*
       User provided routine that evalues the RHS function in the ODE.

                     globalout = F(globalin)
     Parameters are:
         ts        - time-step context
         t         - current time
         globalin  - input vector to function
         globalout - value of function
         ctx       - user provided context for function, in our case we use the appctx
                     defined above.
*/
int RHSFunction(TS ts, double t,Vec globalin, Vec globalout, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DA     da = appctx->da;
  Vec    local, localwork = appctx->localwork;
  int    ierr,i,localsize,rank,size; 
  Scalar *copyptr, *localptr,sc;

  /*
        Local will be a workspace for us that contains the ghost region
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
     Handle the boundary conditions: this is done by using the boundary condition
     U(t,boundary) = g(t,boundary) for some function g. Now take the derivative with
     respect to t to obtain

             U_{t}(t,boundary) = g_{t}(t,boundary)

     In our case U(t,0) = t + 1; so U_{t}(t,0) = 1 and 
                 U(t,1) = 2t+ 1; so U_{t}(t,1) = 2
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
  return 0;
}

/*
        User provided routine to compute the Jacobian of the RHS function
*/
int RHSJacobian(TS ts,double t,Vec globalin,Mat *AA,Mat *BB, MatStructure *str,void *ctx)
{
  Mat    A = *AA;
  AppCtx *appctx = (AppCtx*) ctx;
  int    ierr,i,mstart,mend,mstarts,mends, idx[3],is;
  Scalar v[3];
  DA     da = appctx->da;
  Vec    local;
  Scalar *localptr,sc;

  /*Extract local array */ 
  ierr = DAGetLocalVector(da,&local); CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRQ(ierr);

  *str = SAME_NONZERO_PATTERN;

  ierr = MatGetOwnershipRange(A,&mstarts,&mends); CHKERRQ(ierr);
  mstart = mstarts; mend = mends;
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
     Construct matrix one row at a time
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
  ierr = VecRestoreArray(local,&localptr); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}






