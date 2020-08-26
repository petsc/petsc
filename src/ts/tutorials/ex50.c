
static char help[] ="Solves one dimensional Burger's equation compares with exact solution\n\n";

/*

    Not yet tested in parallel

*/
/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^Burger's equation
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program uses the one-dimensional Burger's equation
       u_t = mu*u_xx - u u_x,
   on the domain 0 <= x <= 1, with periodic boundary conditions

   The operators are discretized with the spectral element method

   See the paper PDE-CONSTRAINED OPTIMIZATION WITH SPECTRAL ELEMENTS USING PETSC AND TAO
   by OANA MARIN, EMIL CONSTANTINESCU, AND BARRY SMITH for details on the exact solution
   used

   See src/tao/unconstrained/tutorials/burgers_spectral.c

  ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscdt.h>
#include <petscdraw.h>
#include <petscdmda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt  n;                /* number of nodes */
  PetscReal *nodes;           /* GLL nodes */
  PetscReal *weights;         /* GLL weights */
} PetscGLL;

typedef struct {
  PetscInt    N;             /* grid points per elements*/
  PetscInt    E;              /* number of elements */
  PetscReal   tol_L2,tol_max; /* error norms */
  PetscInt    steps;          /* number of timesteps */
  PetscReal   Tend;           /* endtime */
  PetscReal   mu;             /* viscosity */
  PetscReal   L;              /* total length of domain */
  PetscReal   Le;
  PetscReal   Tadj;
} PetscParam;

typedef struct {
  Vec         grid;              /* total grid */
  Vec         curr_sol;
} PetscData;

typedef struct {
  Vec         grid;              /* total grid */
  Vec         mass;              /* mass matrix for total integration */
  Mat         stiff;             /* stifness matrix */
  Mat         keptstiff;
  Mat         grad;
  PetscGLL    gll;
} PetscSEMOperators;

typedef struct {
  DM                da;                /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  TS                ts;
  PetscReal         initial_dt;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode RHSMatrixLaplaciangllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSMatrixAdvectiongllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode TrueSolution(TS,PetscReal,Vec,AppCtx*);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       i, xs, xm, ind, j, lenglob;
  PetscReal      x, *wrk_ptr1, *wrk_ptr2;
  MatNullSpace   nsp;
  PetscMPIInt    size;

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*initialize parameters */
  appctx.param.N    = 10;  /* order of the spectral element */
  appctx.param.E    = 10;  /* number of elements */
  appctx.param.L    = 4.0;  /* length of the domain */
  appctx.param.mu   = 0.01; /* diffusion coefficient */
  appctx.initial_dt = 5e-3;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 4;

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-E",&appctx.param.E,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-Tend",&appctx.param.Tend,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&appctx.param.mu,NULL);CHKERRQ(ierr);
  appctx.param.Le = appctx.param.L/appctx.param.E;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (appctx.param.E % size) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of elements must be divisible by number of processes");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscMalloc2(appctx.param.N,&appctx.SEMop.gll.nodes,appctx.param.N,&appctx.SEMop.gll.weights);CHKERRQ(ierr);
  ierr = PetscDTGaussLobattoLegendreQuadrature(appctx.param.N,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights);CHKERRQ(ierr);
  appctx.SEMop.gll.n = appctx.param.N;
  lenglob  = appctx.param.E*(appctx.param.N-1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,lenglob,1,1,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da,&appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.curr_sol,&appctx.SEMop.grid);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.curr_sol,&appctx.SEMop.mass);CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx.da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */

    xs=xs/(appctx.param.N-1);
    xm=xm/(appctx.param.N-1);

  /*
     Build total grid and mass over entire mesh (multi-elemental)
  */

  for (i=xs; i<xs+xm; i++) {
    for (j=0; j<appctx.param.N-1; j++) {
      x = (appctx.param.Le/2.0)*(appctx.SEMop.gll.nodes[j]+1.0)+appctx.param.Le*i;
      ind=i*(appctx.param.N-1)+j;
      wrk_ptr1[ind]=x;
      wrk_ptr2[ind]=.5*appctx.param.Le*appctx.SEMop.gll.weights[j];
      if (j==0) wrk_ptr2[ind]+=.5*appctx.param.Le*appctx.SEMop.gll.weights[j];
    }
  }
  ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.stiff);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.grad);CHKERRQ(ierr);
  /*
   For linear problems with a time-dependent f(u,t) in the equation
   u_t = f(u,t), the user provides the discretized right-hand-side
   as a time-dependent matrix.
   */
  ierr = RHSMatrixLaplaciangllDM(appctx.ts,0.0,appctx.dat.curr_sol,appctx.SEMop.stiff,appctx.SEMop.stiff,&appctx);CHKERRQ(ierr);
  ierr = RHSMatrixAdvectiongllDM(appctx.ts,0.0,appctx.dat.curr_sol,appctx.SEMop.grad,appctx.SEMop.grad,&appctx);CHKERRQ(ierr);
   /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */

  ierr = MatDuplicate(appctx.SEMop.stiff,MAT_COPY_VALUES,&appctx.SEMop.keptstiff);CHKERRQ(ierr);

  /* attach the null space to the matrix, this probably is not needed but does no harm */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(appctx.SEMop.stiff,nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(appctx.SEMop.keptstiff,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,appctx.SEMop.stiff,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  /* attach the null space to the matrix, this probably is not needed but does no harm */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(appctx.SEMop.grad,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,appctx.SEMop.grad,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);

  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  ierr = TSCreate(PETSC_COMM_WORLD,&appctx.ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(appctx.ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(appctx.ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetDM(appctx.ts,appctx.da);CHKERRQ(ierr);
  ierr = TSSetTime(appctx.ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx.ts,appctx.initial_dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(appctx.ts,appctx.param.steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(appctx.ts,appctx.param.Tend);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTolerances(appctx.ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(appctx.ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(appctx.ts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(appctx.ts,appctx.SEMop.stiff,appctx.SEMop.stiff,RHSJacobian,&appctx);CHKERRQ(ierr);

  /* Set Initial conditions for the problem  */
  ierr = TrueSolution(appctx.ts,0,appctx.dat.curr_sol,&appctx);CHKERRQ(ierr);

  ierr = TSSetSolutionFunction(appctx.ts,(PetscErrorCode (*)(TS,PetscReal,Vec,void *))TrueSolution,&appctx);CHKERRQ(ierr);
  ierr = TSSetTime(appctx.ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(appctx.ts,0);CHKERRQ(ierr);

  ierr = TSSolve(appctx.ts,appctx.dat.curr_sol);CHKERRQ(ierr);

  ierr = MatDestroy(&appctx.SEMop.stiff);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.SEMop.keptstiff);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.SEMop.grad);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.grid);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = PetscFree2(appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = TSDestroy(&appctx.ts);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
    ierr = PetscFinalize();
    return ierr;
}

/*
   TrueSolution() computes the true solution for the PDE

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode TrueSolution(TS ts, PetscReal t, Vec u,AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscErrorCode    ierr;
  PetscInt          i,xs,xn;

  ierr = DMDAVecGetArray(appctx->da,u,&s);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
  for (i=xs; i<xs+xn; i++) {
    s[i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*xg[i])*PetscExpReal(-appctx->param.mu*PETSC_PI*PETSC_PI*t)/(2.0+PetscCosScalar(PETSC_PI*xg[i])*PetscExpReal(-appctx->param.mu*PETSC_PI*PETSC_PI*t));
  }
  ierr = DMDAVecRestoreArray(appctx->da,u,&s);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx          *appctx = (AppCtx*)ctx;

  PetscFunctionBegin;
  ierr = MatMult(appctx->SEMop.grad,globalin,globalout);CHKERRQ(ierr); /* grad u */
  ierr = VecPointwiseMult(globalout,globalin,globalout);CHKERRQ(ierr); /* u grad u */
  ierr = VecScale(globalout, -1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(appctx->SEMop.keptstiff,globalin,globalout,globalout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

      K is the discretiziation of the Laplacian
      G is the discretization of the gradient

      Computes Jacobian of      K u + diag(u) G u   which is given by
              K   + diag(u)G + diag(Gu)
*/
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec globalin,Mat A, Mat B,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ctx;
  Vec            Gglobalin;

  PetscFunctionBegin;
  /*    A = diag(u) G */

  ierr = MatCopy(appctx->SEMop.grad,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,globalin,NULL);CHKERRQ(ierr);

  /*    A  = A + diag(Gu) */
  ierr = VecDuplicate(globalin,&Gglobalin);CHKERRQ(ierr);
  ierr = MatMult(appctx->SEMop.grad,globalin,Gglobalin);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A,Gglobalin,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecDestroy(&Gglobalin);CHKERRQ(ierr);

  /*   A  = K - A    */
  ierr = MatScale(A,-1.0);CHKERRQ(ierr);
  ierr = MatAXPY(A,0.0,appctx->SEMop.keptstiff,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

#include "petscblaslapack.h"
/*
     Matrix free operation of 1d Laplacian and Grad for GLL spectral elements
*/
PetscErrorCode MatMult_Laplacian(Mat A,Vec x,Vec y)
{
  AppCtx            *appctx;
  PetscErrorCode    ierr;
  PetscReal         **temp,vv;
  PetscInt          i,j,xs,xn;
  Vec               xlocal,ylocal;
  const PetscScalar *xl;
  PetscScalar       *yl;
  PetscBLASInt      _One = 1,n;
  PetscScalar       _DOne = 1;

  ierr = MatShellGetContext(A,&appctx);CHKERRQ(ierr);
  ierr = DMGetLocalVector(appctx->da,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(appctx->da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(appctx->da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(appctx->da,&ylocal);CHKERRQ(ierr);
  ierr = VecSet(ylocal,0.0);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
  for (i=0; i<appctx->param.N; i++) {
    vv =-appctx->param.mu*2.0/appctx->param.Le;
    for (j=0; j<appctx->param.N; j++) temp[i][j]=temp[i][j]*vv;
  }
  ierr = DMDAVecGetArrayRead(appctx->da,xlocal,(void*)&xl);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,ylocal,&yl);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(appctx->param.N,&n);CHKERRQ(ierr);
  for (j=xs; j<xs+xn; j += appctx->param.N-1) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&_DOne,&temp[0][0],&n,&xl[j],&_One,&_DOne,&yl[j],&_One));
  }
  ierr = DMDAVecRestoreArrayRead(appctx->da,xlocal,(void*)&xl);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da,ylocal,&yl);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da,ylocal,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da,ylocal,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(appctx->da,&xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(appctx->da,&ylocal);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(y,y,appctx->SEMop.mass);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode MatMult_Advection(Mat A,Vec x,Vec y)
{
  AppCtx            *appctx;
  PetscErrorCode    ierr;
  PetscReal         **temp;
  PetscInt          j,xs,xn;
  Vec               xlocal,ylocal;
  const PetscScalar *xl;
  PetscScalar       *yl;
  PetscBLASInt      _One = 1,n;
  PetscScalar       _DOne = 1;

  ierr = MatShellGetContext(A,&appctx);CHKERRQ(ierr);
  ierr = DMGetLocalVector(appctx->da,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(appctx->da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(appctx->da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(appctx->da,&ylocal);CHKERRQ(ierr);
  ierr = VecSet(ylocal,0.0);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(appctx->da,xlocal,(void*)&xl);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,ylocal,&yl);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(appctx->param.N,&n);CHKERRQ(ierr);
  for (j=xs; j<xs+xn; j += appctx->param.N-1) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&_DOne,&temp[0][0],&n,&xl[j],&_One,&_DOne,&yl[j],&_One));
  }
  ierr = DMDAVecRestoreArrayRead(appctx->da,xlocal,(void*)&xl);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da,ylocal,&yl);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da,ylocal,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da,ylocal,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(appctx->da,&xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(appctx->da,&ylocal);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(y,y,appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = VecScale(y,-1.0);CHKERRQ(ierr);
  return 0;
}

/*
   RHSMatrixLaplacian - User-provided routine to compute the right-hand-side
   matrix for the Laplacian operator

   Input Parameters:
   ts - the TS context
   t - current time  (ignored)
   X - current solution (ignored)
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different matrix from which the preconditioner is built
   str - flag indicating matrix structure

*/
PetscErrorCode RHSMatrixLaplaciangllDM(TS ts,PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  PetscReal      **temp;
  PetscReal      vv;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       i,xs,xn,l,j;
  PetscInt       *rowsDM;
  PetscBool      flg = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL,NULL,"-gll_mf",&flg,NULL);CHKERRQ(ierr);

  if (!flg) {
    /*
     Creates the element stiffness matrix for the given gll
     */
    ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
    /* workarround for clang analyzer warning: Division by zero */
    if (appctx->param.N <= 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Spectral element order should be > 1");

    /* scale by the size of the element */
    for (i=0; i<appctx->param.N; i++) {
      vv=-appctx->param.mu*2.0/appctx->param.Le;
      for (j=0; j<appctx->param.N; j++) temp[i][j]=temp[i][j]*vv;
    }

    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);

    xs   = xs/(appctx->param.N-1);
    xn   = xn/(appctx->param.N-1);

    ierr = PetscMalloc1(appctx->param.N,&rowsDM);CHKERRQ(ierr);
    /*
     loop over local elements
     */
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<appctx->param.N; l++) {
        rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
      }
      ierr = MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(rowsDM);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);
    ierr = MatDiagonalScale(A,appctx->SEMop.mass,0);CHKERRQ(ierr);
    ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);

    ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatShellSetContext(A,appctx);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MatMult_Laplacian);CHKERRQ(ierr);
  }
  return 0;
}

/*
   RHSMatrixAdvection - User-provided routine to compute the right-hand-side
   matrix for the Advection (gradient) operator.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

*/
PetscErrorCode RHSMatrixAdvectiongllDM(TS ts,PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  PetscReal      **temp;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       xs,xn,l,j;
  PetscInt       *rowsDM;
  PetscBool      flg = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL,NULL,"-gll_mf",&flg,NULL);CHKERRQ(ierr);

  if (!flg) {
    /*
     Creates the advection matrix for the given gll
     */
    ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
    xs   = xs/(appctx->param.N-1);
    xn   = xn/(appctx->param.N-1);

    ierr = PetscMalloc1(appctx->param.N,&rowsDM);CHKERRQ(ierr);
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<appctx->param.N; l++) {
        rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
      }
      ierr = MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(rowsDM);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);
    ierr = MatDiagonalScale(A,appctx->SEMop.mass,0);CHKERRQ(ierr);
    ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);

    ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatShellSetContext(A,appctx);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MatMult_Advection);CHKERRQ(ierr);
  }
  return 0;
}

/*TEST

    build:
      requires: !complex

    test:
      suffix: 1
      requires: !single

    test:
      suffix: 2
      nsize: 5
      requires: !single

    test:
      suffix: 3
      requires: !single
      args: -ts_view  -ts_type beuler -gll_mf -pc_type none -ts_max_steps 5 -ts_monitor_error

    test:
      suffix: 4
      requires: !single
      args: -ts_view  -ts_type beuler  -pc_type none -ts_max_steps 5 -ts_monitor_error

TEST*/
