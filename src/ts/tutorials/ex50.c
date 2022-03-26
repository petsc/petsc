
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
  PetscInt       i, xs, xm, ind, j, lenglob;
  PetscReal      x, *wrk_ptr1, *wrk_ptr2;
  MatNullSpace   nsp;
  PetscMPIInt    size;

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /*initialize parameters */
  appctx.param.N    = 10;  /* order of the spectral element */
  appctx.param.E    = 10;  /* number of elements */
  appctx.param.L    = 4.0;  /* length of the domain */
  appctx.param.mu   = 0.01; /* diffusion coefficient */
  appctx.initial_dt = 5e-3;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 4;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-E",&appctx.param.E,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-Tend",&appctx.param.Tend,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&appctx.param.mu,NULL));
  appctx.param.Le = appctx.param.L/appctx.param.E;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck((appctx.param.E % size) == 0,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of elements must be divisible by number of processes");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscMalloc2(appctx.param.N,&appctx.SEMop.gll.nodes,appctx.param.N,&appctx.SEMop.gll.weights));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(appctx.param.N,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights));
  appctx.SEMop.gll.n = appctx.param.N;
  lenglob  = appctx.param.E*(appctx.param.N-1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,lenglob,1,1,NULL,&appctx.da));
  PetscCall(DMSetFromOptions(appctx.da));
  PetscCall(DMSetUp(appctx.da));

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  PetscCall(DMCreateGlobalVector(appctx.da,&appctx.dat.curr_sol));
  PetscCall(VecDuplicate(appctx.dat.curr_sol,&appctx.SEMop.grid));
  PetscCall(VecDuplicate(appctx.dat.curr_sol,&appctx.SEMop.mass));

  PetscCall(DMDAGetCorners(appctx.da,&xs,NULL,NULL,&xm,NULL,NULL));
  PetscCall(DMDAVecGetArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1));
  PetscCall(DMDAVecGetArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2));

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
  PetscCall(DMDAVecRestoreArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1));
  PetscCall(DMDAVecRestoreArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE));
  PetscCall(DMCreateMatrix(appctx.da,&appctx.SEMop.stiff));
  PetscCall(DMCreateMatrix(appctx.da,&appctx.SEMop.grad));
  /*
   For linear problems with a time-dependent f(u,t) in the equation
   u_t = f(u,t), the user provides the discretized right-hand-side
   as a time-dependent matrix.
   */
  PetscCall(RHSMatrixLaplaciangllDM(appctx.ts,0.0,appctx.dat.curr_sol,appctx.SEMop.stiff,appctx.SEMop.stiff,&appctx));
  PetscCall(RHSMatrixAdvectiongllDM(appctx.ts,0.0,appctx.dat.curr_sol,appctx.SEMop.grad,appctx.SEMop.grad,&appctx));
   /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */

  PetscCall(MatDuplicate(appctx.SEMop.stiff,MAT_COPY_VALUES,&appctx.SEMop.keptstiff));

  /* attach the null space to the matrix, this probably is not needed but does no harm */
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp));
  PetscCall(MatSetNullSpace(appctx.SEMop.stiff,nsp));
  PetscCall(MatSetNullSpace(appctx.SEMop.keptstiff,nsp));
  PetscCall(MatNullSpaceTest(nsp,appctx.SEMop.stiff,NULL));
  PetscCall(MatNullSpaceDestroy(&nsp));
  /* attach the null space to the matrix, this probably is not needed but does no harm */
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp));
  PetscCall(MatSetNullSpace(appctx.SEMop.grad,nsp));
  PetscCall(MatNullSpaceTest(nsp,appctx.SEMop.grad,NULL));
  PetscCall(MatNullSpaceDestroy(&nsp));

  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&appctx.ts));
  PetscCall(TSSetProblemType(appctx.ts,TS_NONLINEAR));
  PetscCall(TSSetType(appctx.ts,TSRK));
  PetscCall(TSSetDM(appctx.ts,appctx.da));
  PetscCall(TSSetTime(appctx.ts,0.0));
  PetscCall(TSSetTimeStep(appctx.ts,appctx.initial_dt));
  PetscCall(TSSetMaxSteps(appctx.ts,appctx.param.steps));
  PetscCall(TSSetMaxTime(appctx.ts,appctx.param.Tend));
  PetscCall(TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTolerances(appctx.ts,1e-7,NULL,1e-7,NULL));
  PetscCall(TSSetSaveTrajectory(appctx.ts));
  PetscCall(TSSetFromOptions(appctx.ts));
  PetscCall(TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx));
  PetscCall(TSSetRHSJacobian(appctx.ts,appctx.SEMop.stiff,appctx.SEMop.stiff,RHSJacobian,&appctx));

  /* Set Initial conditions for the problem  */
  PetscCall(TrueSolution(appctx.ts,0,appctx.dat.curr_sol,&appctx));

  PetscCall(TSSetSolutionFunction(appctx.ts,(PetscErrorCode (*)(TS,PetscReal,Vec,void *))TrueSolution,&appctx));
  PetscCall(TSSetTime(appctx.ts,0.0));
  PetscCall(TSSetStepNumber(appctx.ts,0));

  PetscCall(TSSolve(appctx.ts,appctx.dat.curr_sol));

  PetscCall(MatDestroy(&appctx.SEMop.stiff));
  PetscCall(MatDestroy(&appctx.SEMop.keptstiff));
  PetscCall(MatDestroy(&appctx.SEMop.grad));
  PetscCall(VecDestroy(&appctx.SEMop.grid));
  PetscCall(VecDestroy(&appctx.SEMop.mass));
  PetscCall(VecDestroy(&appctx.dat.curr_sol));
  PetscCall(PetscFree2(appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights));
  PetscCall(DMDestroy(&appctx.da));
  PetscCall(TSDestroy(&appctx.ts));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
    PetscCall(PetscFinalize());
    return 0;
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
  PetscInt          i,xs,xn;

  PetscCall(DMDAVecGetArray(appctx->da,u,&s));
  PetscCall(DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  PetscCall(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));
  for (i=xs; i<xs+xn; i++) {
    s[i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*xg[i])*PetscExpReal(-appctx->param.mu*PETSC_PI*PETSC_PI*t)/(2.0+PetscCosScalar(PETSC_PI*xg[i])*PetscExpReal(-appctx->param.mu*PETSC_PI*PETSC_PI*t));
  }
  PetscCall(DMDAVecRestoreArray(appctx->da,u,&s));
  PetscCall(DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  return 0;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  AppCtx          *appctx = (AppCtx*)ctx;

  PetscFunctionBegin;
  PetscCall(MatMult(appctx->SEMop.grad,globalin,globalout)); /* grad u */
  PetscCall(VecPointwiseMult(globalout,globalin,globalout)); /* u grad u */
  PetscCall(VecScale(globalout, -1.0));
  PetscCall(MatMultAdd(appctx->SEMop.keptstiff,globalin,globalout,globalout));
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
  AppCtx         *appctx = (AppCtx*)ctx;
  Vec            Gglobalin;

  PetscFunctionBegin;
  /*    A = diag(u) G */

  PetscCall(MatCopy(appctx->SEMop.grad,A,SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(A,globalin,NULL));

  /*    A  = A + diag(Gu) */
  PetscCall(VecDuplicate(globalin,&Gglobalin));
  PetscCall(MatMult(appctx->SEMop.grad,globalin,Gglobalin));
  PetscCall(MatDiagonalSet(A,Gglobalin,ADD_VALUES));
  PetscCall(VecDestroy(&Gglobalin));

  /*   A  = K - A    */
  PetscCall(MatScale(A,-1.0));
  PetscCall(MatAXPY(A,0.0,appctx->SEMop.keptstiff,SAME_NONZERO_PATTERN));
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
  PetscReal         **temp,vv;
  PetscInt          i,j,xs,xn;
  Vec               xlocal,ylocal;
  const PetscScalar *xl;
  PetscScalar       *yl;
  PetscBLASInt      _One = 1,n;
  PetscScalar       _DOne = 1;

  PetscCall(MatShellGetContext(A,&appctx));
  PetscCall(DMGetLocalVector(appctx->da,&xlocal));
  PetscCall(DMGlobalToLocalBegin(appctx->da,x,INSERT_VALUES,xlocal));
  PetscCall(DMGlobalToLocalEnd(appctx->da,x,INSERT_VALUES,xlocal));
  PetscCall(DMGetLocalVector(appctx->da,&ylocal));
  PetscCall(VecSet(ylocal,0.0));
  PetscCall(PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  for (i=0; i<appctx->param.N; i++) {
    vv =-appctx->param.mu*2.0/appctx->param.Le;
    for (j=0; j<appctx->param.N; j++) temp[i][j]=temp[i][j]*vv;
  }
  PetscCall(DMDAVecGetArrayRead(appctx->da,xlocal,(void*)&xl));
  PetscCall(DMDAVecGetArray(appctx->da,ylocal,&yl));
  PetscCall(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));
  PetscCall(PetscBLASIntCast(appctx->param.N,&n));
  for (j=xs; j<xs+xn; j += appctx->param.N-1) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&_DOne,&temp[0][0],&n,&xl[j],&_One,&_DOne,&yl[j],&_One));
  }
  PetscCall(DMDAVecRestoreArrayRead(appctx->da,xlocal,(void*)&xl));
  PetscCall(DMDAVecRestoreArray(appctx->da,ylocal,&yl));
  PetscCall(PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  PetscCall(VecSet(y,0.0));
  PetscCall(DMLocalToGlobalBegin(appctx->da,ylocal,ADD_VALUES,y));
  PetscCall(DMLocalToGlobalEnd(appctx->da,ylocal,ADD_VALUES,y));
  PetscCall(DMRestoreLocalVector(appctx->da,&xlocal));
  PetscCall(DMRestoreLocalVector(appctx->da,&ylocal));
  PetscCall(VecPointwiseDivide(y,y,appctx->SEMop.mass));
  return 0;
}

PetscErrorCode MatMult_Advection(Mat A,Vec x,Vec y)
{
  AppCtx            *appctx;
  PetscReal         **temp;
  PetscInt          j,xs,xn;
  Vec               xlocal,ylocal;
  const PetscScalar *xl;
  PetscScalar       *yl;
  PetscBLASInt      _One = 1,n;
  PetscScalar       _DOne = 1;

  PetscCall(MatShellGetContext(A,&appctx));
  PetscCall(DMGetLocalVector(appctx->da,&xlocal));
  PetscCall(DMGlobalToLocalBegin(appctx->da,x,INSERT_VALUES,xlocal));
  PetscCall(DMGlobalToLocalEnd(appctx->da,x,INSERT_VALUES,xlocal));
  PetscCall(DMGetLocalVector(appctx->da,&ylocal));
  PetscCall(VecSet(ylocal,0.0));
  PetscCall(PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  PetscCall(DMDAVecGetArrayRead(appctx->da,xlocal,(void*)&xl));
  PetscCall(DMDAVecGetArray(appctx->da,ylocal,&yl));
  PetscCall(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));
  PetscCall(PetscBLASIntCast(appctx->param.N,&n));
  for (j=xs; j<xs+xn; j += appctx->param.N-1) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&_DOne,&temp[0][0],&n,&xl[j],&_One,&_DOne,&yl[j],&_One));
  }
  PetscCall(DMDAVecRestoreArrayRead(appctx->da,xlocal,(void*)&xl));
  PetscCall(DMDAVecRestoreArray(appctx->da,ylocal,&yl));
  PetscCall(PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  PetscCall(VecSet(y,0.0));
  PetscCall(DMLocalToGlobalBegin(appctx->da,ylocal,ADD_VALUES,y));
  PetscCall(DMLocalToGlobalEnd(appctx->da,ylocal,ADD_VALUES,y));
  PetscCall(DMRestoreLocalVector(appctx->da,&xlocal));
  PetscCall(DMRestoreLocalVector(appctx->da,&ylocal));
  PetscCall(VecPointwiseDivide(y,y,appctx->SEMop.mass));
  PetscCall(VecScale(y,-1.0));
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
  PetscInt       i,xs,xn,l,j;
  PetscInt       *rowsDM;
  PetscBool      flg = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-gll_mf",&flg,NULL));

  if (!flg) {
    /*
     Creates the element stiffness matrix for the given gll
     */
    PetscCall(PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
    /* workaround for clang analyzer warning: Division by zero */
    PetscCheck(appctx->param.N > 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Spectral element order should be > 1");

    /* scale by the size of the element */
    for (i=0; i<appctx->param.N; i++) {
      vv=-appctx->param.mu*2.0/appctx->param.Le;
      for (j=0; j<appctx->param.N; j++) temp[i][j]=temp[i][j]*vv;
    }

    PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    PetscCall(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));

    xs   = xs/(appctx->param.N-1);
    xn   = xn/(appctx->param.N-1);

    PetscCall(PetscMalloc1(appctx->param.N,&rowsDM));
    /*
     loop over local elements
     */
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<appctx->param.N; l++) {
        rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
      }
      PetscCall(MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES));
    }
    PetscCall(PetscFree(rowsDM));
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(VecReciprocal(appctx->SEMop.mass));
    PetscCall(MatDiagonalScale(A,appctx->SEMop.mass,0));
    PetscCall(VecReciprocal(appctx->SEMop.mass));

    PetscCall(PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  } else {
    PetscCall(MatSetType(A,MATSHELL));
    PetscCall(MatSetUp(A));
    PetscCall(MatShellSetContext(A,appctx));
    PetscCall(MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MatMult_Laplacian));
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
  PetscInt       xs,xn,l,j;
  PetscInt       *rowsDM;
  PetscBool      flg = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-gll_mf",&flg,NULL));

  if (!flg) {
    /*
     Creates the advection matrix for the given gll
     */
    PetscCall(PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
    PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    PetscCall(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));
    xs   = xs/(appctx->param.N-1);
    xn   = xn/(appctx->param.N-1);

    PetscCall(PetscMalloc1(appctx->param.N,&rowsDM));
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<appctx->param.N; l++) {
        rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
      }
      PetscCall(MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES));
    }
    PetscCall(PetscFree(rowsDM));
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    PetscCall(VecReciprocal(appctx->SEMop.mass));
    PetscCall(MatDiagonalScale(A,appctx->SEMop.mass,0));
    PetscCall(VecReciprocal(appctx->SEMop.mass));

    PetscCall(PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  } else {
    PetscCall(MatSetType(A,MATSHELL));
    PetscCall(MatSetUp(A));
    PetscCall(MatShellSetContext(A,appctx));
    PetscCall(MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MatMult_Advection));
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
