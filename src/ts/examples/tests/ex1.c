#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.1 1996/01/05 21:14:54 bsmith Exp bsmith $";
#endif

/*
        Solves U_t - U_x = 0 with periodic boundary conditions.

    1) with a basic F(t,u) = (u_i+1 - u_i-1)/h  (unstable for Euler)
    2) with a stabilized F(t,u)

        Solves U_t = U_xx 

    1) F(t,u) = (u_i+1 - 2u_i + u_i-1)/h^2

*/

static char help[] = "Solves 1D wave or heat equation.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include "ts.h"
#include "sysio.h"


#define PI 3.14159265
typedef struct {
  Vec    localwork;
  DA     da;
  Draw   win;
  int    M;
  double h;
} AppCtx;

int Monitor(TS, int, double , Vec, void *);
int RHSFunctionWave(TS,double,Vec,Vec,void*);
int RHSFunctionHeat(TS,double,Vec,Vec,void*);
int RHSFunctionWaveStabilized(TS,double,Vec,Vec,void*);
int Initial(Vec, void*);
int RHSMatrixHeat(TS,double,Mat *,Mat *, MatStructure *,void *);

int main(int argc,char **argv)
{
  int          rank, size, M = 60, ierr,  time_steps = 100,steps,flg1,flg2,flg;
  AppCtx       appctx;
  Vec          local, global;
  double       h, dt,ftime;
  TS           ts;
  TSType       type;
  Mat          A,B;
  MatStructure A_structure;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);

  OptionsGetInt(PETSC_NULL,"-M",&M,&flg1); appctx.M = M;
  OptionsGetInt(PETSC_NULL,"-time",&time_steps,&flg1);
    
  /* Set up the array */ 
  ierr = DACreate1d(MPI_COMM_WORLD,DA_XPERIODIC,M,1,1,&appctx.da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(appctx.da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(appctx.da,&local); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size); 

  /* Set up display to show wave graph */
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",80,380,400,160,&appctx.win); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(appctx.win); CHKERRA(ierr);

  /* make work array for evaluating right hand side function */
  ierr = VecDuplicate(local,&appctx.localwork); CHKERRA(ierr);

  /* set initial conditions */
  ierr = Initial(global,&appctx); CHKERRA(ierr);

  /* make time step context */
  ierr = TSCreate(MPI_COMM_WORLD,&ts); CHKERRA(ierr);
  ierr = TSSetFromOptions(ts);
  ierr = TSGetType(ts,&type,PETSC_NULL); CHKERRA(ierr);

  h  = 1.0/M; appctx.h = h;
  OptionsHasName(PETSC_NULL,"-matrix_based",&flg);
  OptionsHasName(PETSC_NULL,"-heat",&flg1);
  OptionsHasName(PETSC_NULL,"-wave_unstable",&flg2);
  if (!flg) {
    /*
         In this version the user provides the RHS as a matrix 
    */
    if (flg1){
      ierr = TSSetRHSFunction(ts,RHSFunctionHeat,&appctx); CHKERRA(ierr);
      dt = h*h/2.01;
    }
    else if (flg2 || type == TS_BEULER){
      ierr = TSSetRHSFunction(ts,RHSFunctionWave,&appctx); CHKERRA(ierr);
      dt = h;
    }
    else {
      ierr = TSSetRHSFunction(ts,RHSFunctionWaveStabilized,&appctx); CHKERRA(ierr);
      dt = h;
    }
  } else { 
    /*
        In this version the user provides the RHS as a matrix
    */
    if (flg1) {
      ierr = RHSMatrixHeat(ts,0.0,&A,&B,&A_structure,&appctx);  CHKERRA(ierr);
      dt = h*h/2.01;
    }
    else if (flg2 || type == TS_BEULER) {
    }
    else {
    }
    ierr = TSSetRHSMatrix(ts,A,B,PETSC_NULL,&appctx); CHKERRA(ierr);
  }


  ierr = TSSetInitialTimeStep(ts,0.0,dt); CHKERRA(ierr);
  ierr = TSSetDuration(ts,time_steps,100.); CHKERRQ(ierr);
  ierr = TSSetSolution(ts,global); CHKERRA(ierr);
  ierr = TSSetMonitor(ts,Monitor,&appctx); CHKERRA(ierr);

  ierr = TSSetUp(ts); CHKERRA(ierr);

  ierr = TSStep(ts,&steps,&ftime); CHKERRA(ierr);

  ierr = TSDestroy(ts); CHKERRA(ierr);
  ierr = DADestroy(appctx.da); CHKERRA(ierr);
  ierr = ViewerDestroy((Viewer)appctx.win); CHKERRA(ierr);
  ierr = VecDestroy(appctx.localwork); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
int Initial(Vec global, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Vec    local;
  double *localptr;
  int    i,localsize,mybase,myend,ierr,j, M = appctx->M;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRA(ierr);

  /* Initialize the array */
  ierr = DAGetLocalVector(appctx->da,&local); CHKERRQ(ierr);
  ierr = VecGetLocalSize(local,&localsize); CHKERRA(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRA(ierr);
  localptr[0] = 0.0;
  localptr[localsize-1] = 0.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin((PI*j*6)/((double)M) + 1.2*sin((PI*j*2)/((double)M)))*2;
  }
  ierr = VecRestoreArray(local,&localptr); CHKERRA(ierr);
  ierr = DALocalToGlobal(appctx->da,local,INSERT_VALUES,global); CHKERRA(ierr);
  return 0;
}

int Monitor(TS ts, int step, double time,Vec global, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  int    ierr;

  ierr = VecView(global,(Viewer) appctx->win); CHKERRA(ierr);
  return 0;
}

/* ------------------------------------------------------------------------*/

int RHSFunctionWave(TS ts, double t,Vec globalin, Vec globalout, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DA     da = appctx->da;
  Vec    local, localwork = appctx->localwork;
  int    ierr,i,localsize; 
  double *copyptr, *localptr,sc,dt;

  /*Extract local array */ 
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRA(ierr);

  /* Extract work vector */
  ierr = VecGetArray(localwork,&copyptr); CHKERRA(ierr);

  /* Update Locally - Make array of new values */
  /* Note: I don't do anything for the first and last entry */
  sc = .5/appctx->h;
  ierr = VecGetLocalSize(local,&localsize); CHKERRA(ierr);
  ierr = TSGetTimeStep(ts,&dt);
  for (i=1; i< localsize-1; i++) {
    copyptr[i] = sc * (localptr[i+1] - localptr[i-1]);
  }
  ierr = VecRestoreArray(localwork,&copyptr); CHKERRA(ierr);

  /* Local to Global */
  ierr = DALocalToGlobal(da,localwork,INSERT_VALUES,globalout); CHKERRA(ierr);
  return 0;
}

int RHSFunctionWaveStabilized(TS ts, double t,Vec globalin, Vec globalout, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DA     da = appctx->da;
  Vec    local, localwork = appctx->localwork;
  int    ierr,i,localsize; 
  double *copyptr, *localptr,sc,dt;

  /*Extract local array */ 
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRA(ierr);

  /* Extract work vector */
  ierr = VecGetArray(localwork,&copyptr); CHKERRA(ierr);

  /* Update Locally - Make array of new values */
  /* Note: I don't do anything for the first and last entry */
  sc = .5/appctx->h;
  ierr = VecGetLocalSize(local,&localsize); CHKERRA(ierr);
  ierr = TSGetTimeStep(ts,&dt);
  for (i=1; i< localsize-1; i++) {
    copyptr[i] = sc * (localptr[i+1] - localptr[i-1]);
    /* add stabilizing term for Euler's method */
    copyptr[i] += (.5*(localptr[i+1] + localptr[i-1]) - localptr[i])/dt;
  }
  ierr = VecRestoreArray(localwork,&copyptr); CHKERRA(ierr);

  /* Local to Global */
  ierr = DALocalToGlobal(da,localwork,INSERT_VALUES,globalout); CHKERRA(ierr);
  return 0;
}

int RHSFunctionHeat(TS ts, double t,Vec globalin, Vec globalout, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DA     da = appctx->da;
  Vec    local, localwork = appctx->localwork;
  int    ierr,i,localsize; 
  double *copyptr, *localptr,sc;

  /*Extract local array */ 
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRA(ierr);

  /* Extract work vector */
  ierr = VecGetArray(localwork,&copyptr); CHKERRA(ierr);

  /* Update Locally - Make array of new values */
  /* Note: I don't do anything for the first and last entry */
  sc = 1.0/(appctx->h*appctx->h);
  ierr = VecGetLocalSize(local,&localsize); CHKERRA(ierr);
  for (i=1; i<localsize-1; i++) {
    copyptr[i] = sc * (localptr[i+1] + localptr[i-1] - 2.0*localptr[i]);
  }
  ierr = VecRestoreArray(localwork,&copyptr); CHKERRA(ierr);

  /* Local to Global */
  ierr = DALocalToGlobal(da,localwork,INSERT_VALUES,globalout); CHKERRA(ierr);
  return 0;
}

/* ---------------------------------------------------------------------*/
int RHSMatrixHeat(TS ts,double t,Mat *AA,Mat *BB, MatStructure *str,void *ctx)
{
  Mat    A;
  AppCtx *appctx = (AppCtx*) ctx;
  int    ierr,m = appctx->M,i,mstart,mend,rank,size, idx[3];
  Scalar v[3],stwo = -2./(appctx->h*appctx->h), sone = -.5*stwo;

  *str = ALLMAT_SAME_NONZERO_PATTERN;

  ierr = MatCreate(MPI_COMM_WORLD,m,m,&A); CHKERRQ(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  ierr = MatGetOwnershipRange(A,&mstart,&mend); CHKERRQ(ierr);
  mstart++; mend--;

  /*
     Construct matrice one row at a time
  */
  v[0] = sone; v[1] = stwo; v[2] = sone;  
  for ( i=mstart; i<mend; i++ ) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  *AA = *BB = A;
  return 0;
}

int RHSMatrixWave(TS ts,double t,Mat *AA,Mat *BB, MatStructure *str,void *ctx)
{
  Mat    A;
  AppCtx *appctx = (AppCtx*) ctx;
  int    ierr,m = appctx->M,i,mstart,mend,rank,size, idx[3];
  Scalar v[3],stwo = -2./(appctx->h*appctx->h), sone = -.5*stwo;

  *str = ALLMAT_SAME_NONZERO_PATTERN;

  ierr = MatCreate(MPI_COMM_WORLD,m,m,&A); CHKERRQ(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  ierr = MatGetOwnershipRange(A,&mstart,&mend); CHKERRQ(ierr);
  if (!rank) {
    /*
       First processor does periodic conditions on left
    */
    idx[0] = 0; idx[1] = 1; idx[2] = m-1;
    v[0] = stwo; v[1] = sone; v[2] = sone;
    ierr = MatSetValues(A,1,&mstart,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);
    mstart++;
  }
  if (rank == size-1) {
    /*
       Last processor does periodic conditions on right
    */
    mend--;
    idx[0] = 0; idx[1] = m-2; idx[2] = m-1;
    v[0] = sone; v[1] = sone; v[2] = stwo; 
    ierr = MatSetValues(A,1,&mend,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);
  }

  /*
     Construct matrice one row at a time
  */
  v[0] = sone; v[1] = stwo; v[2] = sone;  
  for ( i=mstart; i<mend; i++ ) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  *AA = *BB = A;
  return 0;
}
