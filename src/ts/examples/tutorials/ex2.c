#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.1 1997/04/10 19:04:58 bsmith Exp bsmith $";
#endif
static char help[] ="Solves a simple time PDE using implicit timestepping";

/*
   Concepts: TS^timestepping^nonlinear problems
   Routines: TSCreate(); TSSetSolution(); TSSetRHSFunction(); TSSetRHSJacobian();
   Routines: TSSetType(); TSSetInitialTimeStep(); TSSetDuration();
   Routines: TSSetFromOptions(); TSStep(); TSDestroy();
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
  Viewer   viewer1,viewer2;
  int      M;                     /* total number of grid points */
  double   h;                     /* mesh width h = 1/(M-1) */
  int      nox;                   /* indicates problem is to be run without graphics */ 
} AppCtx;

/* 
   User-defined routines
*/
int Monitor(TS, int, double , Vec, void *);
int RHSFunction(TS,double,Vec,Vec,void*);
int Initial(Vec, void*);
int RHSJacobian(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);

int main(int argc,char **argv)
{
  int           ierr,  time_steps = 1000, steps, flg;
  AppCtx        appctx;
  Vec           local, global;
  double        dt,ftime;
  TS            ts;
  Mat           A;
  Draw          draw;
 
  PetscInitialize(&argc,&argv,(char*)0,help);

  appctx.comm = PETSC_COMM_WORLD;
  appctx.M    = 60;
  ierr = OptionsGetInt(PETSC_NULL,"-M",&appctx.M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-time",&time_steps,&flg);CHKERRA(ierr);
    
  ierr = OptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRA(ierr); 
  if (flg) appctx.nox = 1; else appctx.nox = 0;

  /* Set up the ghost point communication pattern */ 
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,appctx.M,1,1,PETSC_NULL,&appctx.da);CHKERRA(ierr);
  ierr = DAGetDistributedVector(appctx.da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(appctx.da,&local); CHKERRA(ierr);

  /* Set up display to show solution */

  ierr = ViewerDrawOpenX(PETSC_COMM_WORLD,0,"",80,380,400,160,&appctx.viewer1);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(appctx.viewer1,&draw); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRA(ierr);   
  ierr = ViewerDrawOpenX(PETSC_COMM_WORLD,0,"",80,0,400,160,&appctx.viewer2); CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(appctx.viewer2,&draw); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRA(ierr);   

  /* make local work array for evaluating right hand side function */
  ierr = VecDuplicate(local,&appctx.localwork); CHKERRA(ierr);

  /* make global work array for storing exact solution */
  ierr = VecDuplicate(global,&appctx.solution); CHKERRA(ierr);

  appctx.h = 1.0/(appctx.M-1.0);

  /* set initial conditions */
  ierr = Initial(global,&appctx); CHKERRA(ierr);
    
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
  ierr = ViewerDestroy(appctx.viewer1); CHKERRA(ierr);
  ierr = ViewerDestroy(appctx.viewer2); CHKERRA(ierr);
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
 
int Initial(Vec global, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr,h = appctx->h,x;
  int    i,mybase,myend,ierr;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRQ(ierr);

  /* Initialize the array */
  ierr = VecGetArray(global,&localptr); CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    x = h*i;
    localptr[i-mybase] = 1.0 + x*x;
  }
  ierr = VecRestoreArray(global,&localptr); CHKERRQ(ierr);
  return 0;
}

/*
       Exact solution 
*/
int Solution(double t,Vec solution, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr,h = appctx->h,x;
  int    i,mybase,myend,ierr;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(solution,&mybase,&myend); CHKERRQ(ierr);

  ierr = VecGetArray(solution,&localptr); CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    x = i*h;
    localptr[i-mybase] = (t + 1.0)*(1.0 + x*x);
  }
  ierr = VecRestoreArray(solution,&localptr); CHKERRQ(ierr);
  return 0;
}

int Monitor(TS ts, int step, double time,Vec global, void *ctx)
{
  AppCtx   *appctx = (AppCtx*) ctx;
  int      ierr;
  double   norm_2,norm_max;
  Scalar   mone = -1.0;
  MPI_Comm comm;

  ierr = PetscObjectGetComm((PetscObject)ts,&comm); CHKERRQ(ierr);

  ierr = VecView(global,appctx->viewer2); CHKERRQ(ierr);

  ierr = Solution(time,appctx->solution, ctx); CHKERRQ(ierr);
  ierr = VecAXPY(&mone,global,appctx->solution); CHKERRQ(ierr);
  ierr = VecNorm(appctx->solution,NORM_2,&norm_2); CHKERRQ(ierr);
  norm_2 = sqrt(appctx->h)*norm_2;
  ierr = VecNorm(appctx->solution,NORM_MAX,&norm_max); CHKERRQ(ierr);

  PetscPrintf(comm,"timestep %d time %g norm of error %g %g\n",step,time,norm_2,norm_max);

  ierr = VecView(appctx->solution,appctx->viewer1); CHKERRQ(ierr);

  return 0;
}


int RHSFunction(TS ts, double t,Vec globalin, Vec globalout, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DA     da = appctx->da;
  Vec    local, localwork = appctx->localwork;
  int    ierr,i,localsize,rank,size; 
  Scalar *copyptr, *localptr,sc;

  /*Extract local array */ 
  ierr = DAGetLocalVector(da,&local); CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local); CHKERRQ(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRQ(ierr);

  /* Extract work vector */
  ierr = VecGetArray(localwork,&copyptr); CHKERRQ(ierr);

  /* Update Locally - Make array of new values */
  sc = 1.0/(appctx->h*appctx->h*2.0*(1.0+t)*(1.0+t));
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

  ierr = DALocalToGlobal(da,localwork,INSERT_VALUES,globalout); CHKERRQ(ierr);
  return 0;
}

/* ---------------------------------------------------------------------*/
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






