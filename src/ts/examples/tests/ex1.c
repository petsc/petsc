/*$Id: ex1.c,v 1.35 1999/11/05 14:47:37 bsmith Exp bsmith $*/
/*
       Formatted test for TS routines.

          Solves U_t = U_xx 
     F(t,u) = (u_i+1 - 2u_i + u_i-1)/h^2
       using several different schemes. 
*/

static char help[] = "Solves 1D heat equation.\n\n";

#include "da.h"
#include "sys.h"
#include "ts.h"

#define PETSC_NEAR(a,b,c) (!(PetscAbsDouble((a)-(b)) > (c)*PetscMax(PetscAbsDouble(a),PetscAbsDouble(b))))

typedef struct {
  Vec    global,local,localwork,solution;    /* location for local work (with ghost points) vector */
  DA     da;                    /* manages ghost point communication */
  Viewer viewer1,viewer2;
  int    M;                     /* total number of grid points */
  double h;                     /* mesh width h = 1/(M-1) */
  double norm_2,norm_max;
  int    nox;                   /* indicates problem is to be run without graphics */ 
} AppCtx;

extern int Monitor(TS,int,double,Vec,void *);
extern int RHSFunctionHeat(TS,double,Vec,Vec,void*);
extern int RHSMatrixFree(Mat,Vec,Vec);
extern int Initial(Vec,void*);
extern int RHSMatrixHeat(TS,double,Mat *,Mat *,MatStructure *,void *);
extern int RHSJacobianHeat(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);

#define linear_no_matrix       0
#define linear_no_time         1
#define linear                 2
#define nonlinear_no_jacobian  3
#define nonlinear              4

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           ierr,time_steps = 100,steps,size,m;
  int           problem = linear_no_matrix;
  PetscTruth    flg;
  AppCtx        appctx;
  double        dt,ftime;
  TS            ts;
  Mat           A = 0;
  MatStructure  A_structure;
  TSProblemType tsproblem = TS_LINEAR;
  Draw          draw;
  Viewer        viewer;
  char          tsinfo[120];
 
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  appctx.M = 60;
  ierr = OptionsGetInt(PETSC_NULL,"-M",&appctx.M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-time",&time_steps,PETSC_NULL);CHKERRA(ierr);
    
  ierr = OptionsHasName(PETSC_NULL,"-nox",&flg);CHKERRA(ierr); 
  if (flg) appctx.nox = 1; else appctx.nox = 0;
  appctx.norm_2 = 0.0; appctx.norm_max = 0.0;

  /* Set up the ghost point communication pattern */ 
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,appctx.M,1,1,PETSC_NULL,&appctx.da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(appctx.da,&appctx.global);CHKERRA(ierr);
  ierr = VecGetLocalSize(appctx.global,&m);CHKERRA(ierr);
  ierr = DACreateLocalVector(appctx.da,&appctx.local);CHKERRA(ierr);

  /* Set up display to show wave graph */

  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,380,400,160,&appctx.viewer1);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(appctx.viewer1,0,&draw);CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw);CHKERRA(ierr);   
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,0,400,160,&appctx.viewer2);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(appctx.viewer2,0,&draw);CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw);CHKERRA(ierr);   


  /* make work array for evaluating right hand side function */
  ierr = VecDuplicate(appctx.local,&appctx.localwork);CHKERRA(ierr);

  /* make work array for storing exact solution */
  ierr = VecDuplicate(appctx.global,&appctx.solution);CHKERRA(ierr);

  appctx.h = 1.0/(appctx.M-1.0);

  /* set initial conditions */
  ierr = Initial(appctx.global,&appctx);CHKERRA(ierr);
 
  /*
     This example is written to allow one to easily test parts 
    of TS, we do not expect users to generally need to use more
    then a single TSProblemType
  */
  ierr = OptionsHasName(PETSC_NULL,"-linear_no_matrix",&flg);CHKERRA(ierr);
  if (flg) {
    tsproblem = TS_LINEAR;
    problem   = linear_no_matrix;
  }
  ierr = OptionsHasName(PETSC_NULL,"-linear_constant_matrix",&flg);CHKERRA(ierr);
  if (flg) {
    tsproblem = TS_LINEAR;
    problem   = linear_no_time;
  }
  ierr = OptionsHasName(PETSC_NULL,"-linear_variable_matrix",&flg);CHKERRA(ierr);
  if (flg) {
    tsproblem = TS_LINEAR;
    problem   = linear;
  }
  ierr = OptionsHasName(PETSC_NULL,"-nonlinear_no_jacobian",&flg);CHKERRA(ierr);
  if (flg) {
    tsproblem = TS_NONLINEAR;
    problem   = nonlinear_no_jacobian;
  }
  ierr = OptionsHasName(PETSC_NULL,"-nonlinear_jacobian",&flg);CHKERRA(ierr);
  if (flg) {
    tsproblem = TS_NONLINEAR;
    problem   = nonlinear;
  }
    
  /* make timestep context */
  ierr = TSCreate(PETSC_COMM_WORLD,tsproblem,&ts);CHKERRA(ierr);
  ierr = TSSetMonitor(ts,Monitor,&appctx,PETSC_NULL);CHKERRA(ierr);

  dt = appctx.h*appctx.h/2.01;

  if (problem == linear_no_matrix) {
    /*
         The user provides the RHS as a Shell matrix.
    */
    ierr = MatCreateShell(PETSC_COMM_WORLD,m,appctx.M,appctx.M,appctx.M,&appctx,&A);CHKERRA(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void*)RHSMatrixFree);CHKERRA(ierr);
    ierr = TSSetRHSMatrix(ts,A,A,PETSC_NULL,&appctx);CHKERRA(ierr);
  } else if (problem == linear_no_time) {
    /*
         The user provides the RHS as a matrix
    */
    ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,appctx.M,appctx.M,&A);CHKERRA(ierr);
    ierr = RHSMatrixHeat(ts,0.0,&A,&A,&A_structure,&appctx);CHKERRA(ierr);
    ierr = TSSetRHSMatrix(ts,A,A,PETSC_NULL,&appctx);CHKERRA(ierr);
  } else if (problem == linear) {
    /*
         The user provides the RHS as a time dependent matrix
    */
    ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,appctx.M,appctx.M,&A);CHKERRA(ierr);
    ierr = RHSMatrixHeat(ts,0.0,&A,&A,&A_structure,&appctx);CHKERRA(ierr);
    ierr = TSSetRHSMatrix(ts,A,A,RHSMatrixHeat,&appctx);CHKERRA(ierr);
  } else if (problem == nonlinear_no_jacobian) {
    /*
         The user provides the RHS and a Shell Jacobian
    */
    ierr = TSSetRHSFunction(ts,RHSFunctionHeat,&appctx);CHKERRA(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,m,appctx.M,appctx.M,appctx.M,&appctx,&A);CHKERRA(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void*)RHSMatrixFree);CHKERRA(ierr);
    ierr = TSSetRHSJacobian(ts,A,A,PETSC_NULL,&appctx);CHKERRA(ierr);  
  } else if (problem == nonlinear) {
    /*
         The user provides the RHS and Jacobian
    */
    ierr = TSSetRHSFunction(ts,RHSFunctionHeat,&appctx);CHKERRA(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,appctx.M,appctx.M,&A);CHKERRA(ierr);
    ierr = RHSMatrixHeat(ts,0.0,&A,&A,&A_structure,&appctx);CHKERRA(ierr);
    ierr = TSSetRHSJacobian(ts,A,A,RHSJacobianHeat,&appctx);CHKERRA(ierr);  
  }

  ierr = TSSetFromOptions(ts);CHKERRA(ierr);

  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRA(ierr);
  ierr = TSSetDuration(ts,time_steps,100.);CHKERRA(ierr);
  ierr = TSSetSolution(ts,appctx.global);CHKERRA(ierr);


  ierr = TSSetUp(ts);CHKERRA(ierr);
  ierr = TSStep(ts,&steps,&ftime);CHKERRA(ierr);
  ierr = ViewerStringOpen(PETSC_COMM_WORLD,tsinfo,120,&viewer);CHKERRA(ierr);
  ierr = TSView(ts,viewer);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-test",&flg);CHKERRA(ierr);
  if (flg) {
    PetscTruth iseuler;
    ierr = PetscTypeCompare((PetscObject)ts,"euler",&iseuler);CHKERRA(ierr);
    if (iseuler) {
      if (!PETSC_NEAR(appctx.norm_2/steps,0.00257244,1.e-4)) {
        fprintf(stderr,"Error in Euler method: 2-norm %g expecting: 0.00257244\n",appctx.norm_2/steps);
      }
    } else {
      if (!PETSC_NEAR(appctx.norm_2/steps,0.00506174,1.e-4)) {
        fprintf(stderr,"Error in %s method: 2-norm %g expecting: 0.00506174\n",tsinfo,appctx.norm_2/steps);
      }
    }
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d Procs Avg. error 2 norm %g max norm %g %s\n",
                size,appctx.norm_2/steps,appctx.norm_max/steps,tsinfo);CHKERRA(ierr);
  }

  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = TSDestroy(ts);CHKERRA(ierr);
  ierr = ViewerDestroy(appctx.viewer1);CHKERRA(ierr);
  ierr = ViewerDestroy(appctx.viewer2);CHKERRA(ierr);
  ierr = VecDestroy(appctx.localwork);CHKERRA(ierr);
  ierr = VecDestroy(appctx.solution);CHKERRA(ierr);
  ierr = VecDestroy(appctx.local);CHKERRA(ierr);
  ierr = VecDestroy(appctx.global);CHKERRA(ierr);
  ierr = DADestroy(appctx.da);CHKERRA(ierr);
  if (A) {ierr= MatDestroy(A);CHKERRA(ierr);}

  PetscFinalize();
  return 0;
}

/* -------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "Initial" 
int Initial(Vec global,void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr,h = appctx->h;
  int    i,mybase,myend,ierr;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRQ(ierr);

  /* Initialize the array */
  ierr = VecGetArray(global,&localptr);CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    localptr[i-mybase] = PetscSinScalar(PETSC_PI*i*6.*h) + 3.*PetscSinScalar(PETSC_PI*i*2.*h);
  }
  ierr = VecRestoreArray(global,&localptr);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "Solution"
/*
       Exact solution 
*/
int Solution(double t,Vec solution,void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  Scalar *localptr,h = appctx->h,ex1,ex2,sc1,sc2;
  int    i,mybase,myend,ierr;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(solution,&mybase,&myend);CHKERRQ(ierr);

  ex1 = PetscExpScalar(-36.*PETSC_PI*PETSC_PI*t); 
  ex2 = PetscExpScalar(-4.*PETSC_PI*PETSC_PI*t);
  sc1 = PETSC_PI*6.*h;                 sc2 = PETSC_PI*2.*h;
  ierr = VecGetArray(solution,&localptr);CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    localptr[i-mybase] = PetscSinScalar(sc1*(double)i)*ex1 + 3.*PetscSinScalar(sc2*(double)i)*ex2;
  }
  ierr = VecRestoreArray(solution,&localptr);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "Monitor"
int Monitor(TS ts,int step,double time,Vec global,void *ctx)
{
  AppCtx   *appctx = (AppCtx*) ctx;
  int      ierr;
  double   norm_2,norm_max;
  Scalar   mone = -1.0;
  MPI_Comm comm;

  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);

  ierr = VecView(global,appctx->viewer2);CHKERRQ(ierr);

  ierr = Solution(time,appctx->solution,ctx);CHKERRQ(ierr);
  ierr = VecAXPY(&mone,global,appctx->solution);CHKERRQ(ierr);
  ierr = VecNorm(appctx->solution,NORM_2,&norm_2);CHKERRQ(ierr);
  norm_2 = sqrt(appctx->h)*norm_2;
  ierr = VecNorm(appctx->solution,NORM_MAX,&norm_max);CHKERRQ(ierr);

  if (!appctx->nox) {
    ierr = PetscPrintf(comm,"timestep %d time %g norm of error %g %g\n",step,time,norm_2,norm_max);CHKERRQ(ierr);
  }

  appctx->norm_2   += norm_2;
  appctx->norm_max += norm_max;

  ierr = VecView(appctx->solution,appctx->viewer1);CHKERRQ(ierr);

  return 0;
}

/* -----------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "RHSMatrixFree"
int RHSMatrixFree(Mat mat,Vec x,Vec y)
{
  int  ierr;
  void *ctx;

  MatShellGetContext(mat,(void **)&ctx);
  ierr = RHSFunctionHeat(0,0.0,x,y,ctx);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "RHSFunctionHeat"
int RHSFunctionHeat(TS ts,double t,Vec globalin,Vec globalout,void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DA     da = appctx->da;
  Vec    local = appctx->local,localwork = appctx->localwork;
  int    ierr,i,localsize; 
  Scalar *copyptr,*localptr,sc;

  /*Extract local array */ 
  ierr = DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = VecGetArray(local,&localptr);CHKERRQ(ierr);

  /* Extract work vector */
  ierr = VecGetArray(localwork,&copyptr);CHKERRQ(ierr);

  /* Update Locally - Make array of new values */
  /* Note: For the first and last entry I copy the value */
  /* if this is an interior node it is irrelevant */
  sc = 1.0/(appctx->h*appctx->h);
  ierr = VecGetLocalSize(local,&localsize);CHKERRQ(ierr);
  copyptr[0] = localptr[0];
  for (i=1; i<localsize-1; i++) {
    copyptr[i] = sc * (localptr[i+1] + localptr[i-1] - 2.0*localptr[i]);
  }
  copyptr[localsize-1] = localptr[localsize-1];
  ierr = VecRestoreArray(local,&localptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localwork,&copyptr);CHKERRQ(ierr);

  /* Local to Global */
  ierr = DALocalToGlobal(da,localwork,INSERT_VALUES,globalout);CHKERRQ(ierr);
  return 0;
}

/* ---------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "RHSMatrixHeat"
int RHSMatrixHeat(TS ts,double t,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  Mat    A = *AA;
  AppCtx *appctx = (AppCtx*) ctx;
  int    ierr,i,mstart,mend,rank,size,idx[3];
  Scalar v[3],stwo = -2./(appctx->h*appctx->h),sone = -.5*stwo;

  *str = SAME_NONZERO_PATTERN;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&mstart,&mend);CHKERRQ(ierr);
  if (mstart == 0) {
    v[0] = 1.0;
    ierr = MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES);CHKERRQ(ierr);
    mstart++;
  }
  if (mend == appctx->M) {
    mend--;
    v[0] = 1.0;
    ierr = MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Construct matrice one row at a time
  */
  v[0] = sone; v[1] = stwo; v[2] = sone;  
  for (i=mstart; i<mend; i++) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "RHSJacobianHeat"
int RHSJacobianHeat(TS ts,double t,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  return RHSMatrixHeat(ts,t,AA,BB,str,ctx);
}





