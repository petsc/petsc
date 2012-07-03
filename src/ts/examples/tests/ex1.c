/*
       Formatted test for TS routines.

          Solves U_t = U_xx 
     F(t,u) = (u_i+1 - 2u_i + u_i-1)/h^2
       using several different schemes. 
*/

/* Usage: 
   ./ex1 -nox -ts_type beuler -ts_view 
   ./ex1 -nox -linear_constant_matrix -ts_type beuler -pc_type lu
   ./ex1 -nox -linear_variable_matrix -ts_type beuler
*/

static char help[] = "Solves 1D heat equation.\n\n";

#include <petscdmda.h>
#include <petscts.h>

#define PETSC_NEAR(a,b,c) (!(PetscAbsReal((a)-(b)) > (c)*PetscMax(PetscAbsReal(a),PetscAbsReal(b))))

typedef struct {
  Vec         global,local,localwork,solution;    /* location for local work (with ghost points) vector */
  DM          da;                    /* manages ghost point communication */
  PetscViewer viewer1,viewer2;
  PetscInt    M;                     /* total number of grid points */
  PetscReal   h;                     /* mesh width h = 1/(M-1) */
  PetscReal   norm_2,norm_max;
  PetscBool   nox;                   /* indicates problem is to be run without graphics */ 
} AppCtx;

extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void *);
extern PetscErrorCode RHSFunctionHeat(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSMatrixFree(Mat,Vec,Vec);
extern PetscErrorCode Initial(Vec,void*);
extern PetscErrorCode RHSMatrixHeat(TS,PetscReal,Mat *,Mat *,MatStructure *,void *);
extern PetscErrorCode LHSMatrixHeat(TS,PetscReal,Mat *,Mat *,MatStructure *,void *);
extern PetscErrorCode RHSJacobianHeat(TS,PetscReal,Vec,Mat*,Mat*,MatStructure *,void*);

#define linear_no_matrix        0
#define linear_constant_matrix  1
#define linear_variable_matrix  2
#define nonlinear_no_jacobian   3
#define nonlinear_jacobian      4

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       maxsteps = 100,steps,m;
  PetscMPIInt    size;
  PetscInt       problem = linear_no_matrix;
  PetscBool      flg;
  AppCtx         appctx;
  PetscReal      dt,ftime,maxtime=100.;
  TS             ts;
  Mat            A=0,Alhs=0;
  MatStructure   A_structure;
  TSProblemType  tsproblem = TS_LINEAR;
  PetscDraw      draw;
  char           tsinfo[120];
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  appctx.M = 60;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&appctx.M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-time",&maxsteps,PETSC_NULL);CHKERRQ(ierr);
    
  ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&appctx.nox);CHKERRQ(ierr); 
  appctx.norm_2 = 0.0; appctx.norm_max = 0.0;

  /* Set up the ghost point communication pattern */ 
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,appctx.M,1,1,PETSC_NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(appctx.da,&appctx.global);CHKERRQ(ierr);
  ierr = VecGetLocalSize(appctx.global,&m);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(appctx.da,&appctx.local);CHKERRQ(ierr);

  /* Set up display to show wave graph */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,380,400,160,&appctx.viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(appctx.viewer1,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);   
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,0,400,160,&appctx.viewer2);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(appctx.viewer2,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);   

  /* make work array for evaluating right hand side function */
  ierr = VecDuplicate(appctx.local,&appctx.localwork);CHKERRQ(ierr);

  /* make work array for storing exact solution */
  ierr = VecDuplicate(appctx.global,&appctx.solution);CHKERRQ(ierr);

  appctx.h = 1.0/(appctx.M-1.0);

  /* set initial conditions */
  ierr = Initial(appctx.global,&appctx);CHKERRQ(ierr);
 
  /*
     This example is written to allow one to easily test parts 
    of TS, we do not expect users to generally need to use more
    then a single TSProblemType
  */
  tsproblem = TS_NONLINEAR;
  problem   = nonlinear_no_jacobian;
  ierr = PetscOptionsHasName(PETSC_NULL,"-linear_no_matrix",&flg);CHKERRQ(ierr);
  if (flg) {
    tsproblem = TS_LINEAR;
    problem   = linear_no_matrix;
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-linear_constant_matrix",&flg);CHKERRQ(ierr);
  if (flg) {
    tsproblem = TS_LINEAR;
    problem   = linear_constant_matrix;
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-linear_variable_matrix",&flg);CHKERRQ(ierr);
  if (flg) {
    tsproblem = TS_LINEAR;
    problem   = linear_variable_matrix;
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-nonlinear_no_jacobian",&flg);CHKERRQ(ierr);
  if (flg) {
    tsproblem = TS_NONLINEAR;
    problem   = nonlinear_jacobian;
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-nonlinear_jacobian",&flg);CHKERRQ(ierr);
  if (flg) {
    tsproblem = TS_NONLINEAR;
    problem   = nonlinear_jacobian;
  }
    
  /* make timestep context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,tsproblem);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-monitor",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TSMonitorSet(ts,Monitor,&appctx,PETSC_NULL);CHKERRQ(ierr);
  }

  dt = appctx.h*appctx.h/2.01;

  if (problem == linear_no_matrix) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This case needs rewritten");
    /*
         The user provides the RHS as a Shell matrix.
    */
    /*
    ierr = MatCreateShell(PETSC_COMM_WORLD,m,m,PETSC_DECIDE,PETSC_DECIDE,&appctx,&A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)(void))RHSMatrixFree);CHKERRQ(ierr);
    ierr = TSSetMatrices(ts,A,PETSC_NULL,PETSC_NULL,PETSC_NULL,DIFFERENT_NONZERO_PATTERN,&appctx);CHKERRQ(ierr);
     */
  } else if (problem == linear_constant_matrix) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This case needs rewritten");
    /*
         The user provides the RHS as a constant matrix
    */
    /*
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = RHSMatrixHeat(ts,0.0,&A,&A,&A_structure,&appctx);CHKERRQ(ierr); 

    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Alhs);CHKERRQ(ierr); 
    ierr = MatZeroEntries(Alhs);CHKERRQ(ierr);
    ierr = MatShift(Alhs,1.0);CHKERRQ(ierr);
    ierr = TSSetMatrices(ts,A,PETSC_NULL,Alhs,PETSC_NULL,DIFFERENT_NONZERO_PATTERN,&appctx);CHKERRQ(ierr);
     */
  } else if (problem == linear_variable_matrix) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This case needs rewritten");
    /*
         The user provides the RHS as a time dependent matrix
    */
    /*
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = RHSMatrixHeat(ts,0.0,&A,&A,&A_structure,&appctx);CHKERRQ(ierr);

    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Alhs);CHKERRQ(ierr);    
    ierr = MatZeroEntries(Alhs);CHKERRQ(ierr);
    ierr = MatShift(Alhs,1.0);CHKERRQ(ierr);
    ierr = LHSMatrixHeat(ts,0.0,&Alhs,&Alhs,&A_structure,&appctx);CHKERRQ(ierr);
    ierr = TSSetMatrices(ts,A,RHSMatrixHeat,Alhs,LHSMatrixHeat,DIFFERENT_NONZERO_PATTERN,&appctx);CHKERRQ(ierr);
     */
  } else if (problem == nonlinear_no_jacobian) {
    /*
         The user provides the RHS and a Shell Jacobian
    */
    ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunctionHeat,&appctx);CHKERRQ(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,m,m,PETSC_DECIDE,PETSC_DECIDE,&appctx,&A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)(void))RHSMatrixFree);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,A,A,PETSC_NULL,&appctx);CHKERRQ(ierr);  
  } else if (problem == nonlinear_jacobian) {
    /*
         The user provides the RHS and Jacobian
    */
    ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunctionHeat,&appctx);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = RHSMatrixHeat(ts,0.0,&A,&A,&A_structure,&appctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,A,A,RHSJacobianHeat,&appctx);CHKERRQ(ierr);  
  }

  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,maxtime);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,appctx.global);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,appctx.global,&ftime);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-testinfo",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscBool  iseuler;
    ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

    ierr = PetscObjectTypeCompare((PetscObject)ts,"euler",&iseuler);CHKERRQ(ierr);
    if (iseuler) {
      if (!PETSC_NEAR(appctx.norm_2/steps,0.00257244,1.e-4)) {
        fprintf(stdout,"Error in Euler method: 2-norm %G expecting: 0.00257244\n",appctx.norm_2/steps);
      }
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%D Procs; Avg. error 2-norm %G; max-norm %G; %s\n",
                size,appctx.norm_2/steps,appctx.norm_max/steps,tsinfo);CHKERRQ(ierr);
    }
  }

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.viewer2);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.localwork);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.solution);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.local);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.global);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  if (A) {ierr= MatDestroy(&A);CHKERRQ(ierr);}
  if (Alhs) {ierr= MatDestroy(&Alhs);CHKERRQ(ierr);}

  ierr = PetscFinalize();
  return 0;
}

/* -------------------------------------------------------------------*/
/*
  Set initial condition: u(t=0) = sin(6*pi*x) + 3*sin(2*pi*x)
*/
#undef __FUNCT__
#define __FUNCT__ "Initial" 
PetscErrorCode Initial(Vec global,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;
  PetscScalar    *localptr,h = appctx->h;
  PetscInt       i,mybase,myend;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRQ(ierr);

  /* Initialize the array */
  ierr = VecGetArray(global,&localptr);CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    localptr[i-mybase] = PetscSinScalar(PETSC_PI*i*6.*h) + 3.*PetscSinScalar(PETSC_PI*i*2.*h);
  }
  ierr = VecRestoreArray(global,&localptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Solution"
/*
   Exact solution: 
     u = sin(6*pi*x)*exp(-36*pi*pi*t) + 3*sin(2*pi*x)*exp(-4*pi*pi*t)
*/
PetscErrorCode Solution(PetscReal t,Vec solution,void *ctx)
{
  AppCtx *       appctx = (AppCtx*) ctx;
  PetscScalar    *localptr,h = appctx->h,ex1,ex2,sc1,sc2;
  PetscInt       i,mybase,myend;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(solution,&mybase,&myend);CHKERRQ(ierr);

  ex1 = exp(-36.*PETSC_PI*PETSC_PI*t); 
  ex2 = exp(-4.*PETSC_PI*PETSC_PI*t);
  sc1 = PETSC_PI*6.*h;                 
  sc2 = PETSC_PI*2.*h;
  ierr = VecGetArray(solution,&localptr);CHKERRQ(ierr);
  for (i=mybase; i<myend; i++) {
    localptr[i-mybase] = PetscSinScalar(sc1*(PetscReal)i)*ex1 + 3.*PetscSinScalar(sc2*(PetscReal)i)*ex2;
  }
  ierr = VecRestoreArray(solution,&localptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  step   - iteration number
  ltime  - current time
  global - current iterate
 */
#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal ltime,Vec global,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;
  PetscErrorCode ierr;
  PetscReal      norm_2,norm_max;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  if (!appctx->nox) {
    ierr = VecView(global,appctx->viewer2);CHKERRQ(ierr); /* show wave graph */
  }
  ierr = Solution(ltime,appctx->solution,ctx);CHKERRQ(ierr); /* get true solution at current time */
  ierr = VecAXPY(appctx->solution,-1.0,global);CHKERRQ(ierr);
  ierr = VecNorm(appctx->solution,NORM_2,&norm_2);CHKERRQ(ierr);
  norm_2 = PetscSqrtReal(appctx->h)*norm_2;
  ierr = VecNorm(appctx->solution,NORM_MAX,&norm_max);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"timestep %D time %G norm of error %.5f %.5f\n",step,ltime,norm_2,norm_max);CHKERRQ(ierr);

  appctx->norm_2   += norm_2;
  appctx->norm_max += norm_max;
  if (!appctx->nox) {
    ierr = VecView(appctx->solution,appctx->viewer1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "RHSMatrixFree"
PetscErrorCode RHSMatrixFree(Mat mat,Vec x,Vec y)
{
  PetscErrorCode  ierr;
  void            *ctx;

  PetscFunctionBegin;
  MatShellGetContext(mat,(void **)&ctx);
  ierr = RHSFunctionHeat(0,0.0,x,y,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunctionHeat"
PetscErrorCode RHSFunctionHeat(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;
  DM             da = appctx->da;
  Vec            local = appctx->local,localwork = appctx->localwork;
  PetscErrorCode ierr;
  PetscInt       i,localsize; 
  PetscScalar    *copyptr,*localptr,sc;

  PetscFunctionBegin;
  /*Extract local array */ 
  ierr = DMGlobalToLocalBegin(da,globalin,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,globalin,INSERT_VALUES,local);CHKERRQ(ierr);
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
  ierr = DMLocalToGlobalBegin(da,localwork,INSERT_VALUES,globalout);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localwork,INSERT_VALUES,globalout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "RHSMatrixHeat"
PetscErrorCode RHSMatrixHeat(TS ts,PetscReal t,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  Mat            A = *AA;
  AppCtx         *appctx = (AppCtx*) ctx;
  PetscErrorCode ierr;
  PetscInt       i,mstart,mend,idx[3];
  PetscMPIInt    size,rank;
  PetscScalar    v[3],stwo = -2./(appctx->h*appctx->h),sone = -.5*stwo;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobianHeat"
PetscErrorCode RHSJacobianHeat(TS ts,PetscReal t,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  PetscFunctionBegin;
  RHSMatrixHeat(ts,t,AA,BB,str,ctx);
  PetscFunctionReturn(0);
}

/* A = indentity matrix */
#undef __FUNCT__
#define __FUNCT__ "LHSMatrixHeat"
PetscErrorCode LHSMatrixHeat(TS ts,PetscReal t,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  PetscErrorCode ierr;
  Mat            A = *AA;

  PetscFunctionBegin;
  *str = SAME_NONZERO_PATTERN;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatShift(A,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  



