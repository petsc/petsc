
static char help[] ="Solves a simple data assimilation problem with one dimensional Burger's equation using TSAdjoint\n\n";

/*

    Not yet tested in parallel

*/
/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^Burger's equation
   Concepts: adjoints
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program uses the one-dimensional Burger's equation
       u_t = mu*u_xx - u u_x,
   on the domain 0 <= x <= 1, with periodic boundary conditions

   to demonstrate solving a data assimilation problem of finding the initial conditions
   to produce a given solution at a fixed time.

   The operators are discretized with the spectral element method

   See the paper PDE-CONSTRAINED OPTIMIZATION WITH SPECTRAL ELEMENTS USING PETSC AND TAO
   by OANA MARIN, EMIL CONSTANTINESCU, AND BARRY SMITH for details on the exact solution
   used

  ------------------------------------------------------------------------- */

#include <petsctao.h>
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
  PetscInt    N;              /* grid points per elements*/
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
  Vec         obj;               /* desired end state */
  Vec         grid;              /* total grid */
  Vec         grad;
  Vec         ic;
  Vec         curr_sol;
  Vec         true_solution;     /* actual initial conditions for the final solution */
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
extern PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
extern PetscErrorCode RHSMatrixLaplaciangllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSMatrixAdvectiongllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode TrueSolution(Vec,AppCtx*);
extern PetscErrorCode ComputeObjective(PetscReal,Vec,AppCtx*);
extern PetscErrorCode MonitorError(Tao,void*);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  Tao            tao;
  Vec            u;                      /* approximate solution vector */
  PetscInt       i, xs, xm, ind, j, lenglob;
  PetscReal      x, *wrk_ptr1, *wrk_ptr2;
  MatNullSpace   nsp;
  PetscMPIInt    size;

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  /*initialize parameters */
  appctx.param.N    = 10;  /* order of the spectral element */
  appctx.param.E    = 10;  /* number of elements */
  appctx.param.L    = 4.0;  /* length of the domain */
  appctx.param.mu   = 0.01; /* diffusion coefficient */
  appctx.initial_dt = 5e-3;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 4;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-E",&appctx.param.E,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-Tend",&appctx.param.Tend,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&appctx.param.mu,NULL));
  appctx.param.Le = appctx.param.L/appctx.param.E;

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck((appctx.param.E % size) == 0,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of elements must be divisible by number of processes");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscMalloc2(appctx.param.N,&appctx.SEMop.gll.nodes,appctx.param.N,&appctx.SEMop.gll.weights));
  CHKERRQ(PetscDTGaussLobattoLegendreQuadrature(appctx.param.N,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights));
  appctx.SEMop.gll.n = appctx.param.N;
  lenglob  = appctx.param.E*(appctx.param.N-1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,lenglob,1,1,NULL,&appctx.da));
  CHKERRQ(DMSetFromOptions(appctx.da));
  CHKERRQ(DMSetUp(appctx.da));

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  CHKERRQ(DMCreateGlobalVector(appctx.da,&u));
  CHKERRQ(VecDuplicate(u,&appctx.dat.ic));
  CHKERRQ(VecDuplicate(u,&appctx.dat.true_solution));
  CHKERRQ(VecDuplicate(u,&appctx.dat.obj));
  CHKERRQ(VecDuplicate(u,&appctx.SEMop.grid));
  CHKERRQ(VecDuplicate(u,&appctx.SEMop.mass));
  CHKERRQ(VecDuplicate(u,&appctx.dat.curr_sol));

  CHKERRQ(DMDAGetCorners(appctx.da,&xs,NULL,NULL,&xm,NULL,NULL));
  CHKERRQ(DMDAVecGetArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1));
  CHKERRQ(DMDAVecGetArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2));

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
  CHKERRQ(DMDAVecRestoreArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1));
  CHKERRQ(DMDAVecRestoreArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE));
  CHKERRQ(DMCreateMatrix(appctx.da,&appctx.SEMop.stiff));
  CHKERRQ(DMCreateMatrix(appctx.da,&appctx.SEMop.grad));
  /*
   For linear problems with a time-dependent f(u,t) in the equation
   u_t = f(u,t), the user provides the discretized right-hand-side
   as a time-dependent matrix.
   */
  CHKERRQ(RHSMatrixLaplaciangllDM(appctx.ts,0.0,u,appctx.SEMop.stiff,appctx.SEMop.stiff,&appctx));
  CHKERRQ(RHSMatrixAdvectiongllDM(appctx.ts,0.0,u,appctx.SEMop.grad,appctx.SEMop.grad,&appctx));
   /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */

  CHKERRQ(MatDuplicate(appctx.SEMop.stiff,MAT_COPY_VALUES,&appctx.SEMop.keptstiff));

  /* attach the null space to the matrix, this probably is not needed but does no harm */
  CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp));
  CHKERRQ(MatSetNullSpace(appctx.SEMop.stiff,nsp));
  CHKERRQ(MatSetNullSpace(appctx.SEMop.keptstiff,nsp));
  CHKERRQ(MatNullSpaceTest(nsp,appctx.SEMop.stiff,NULL));
  CHKERRQ(MatNullSpaceDestroy(&nsp));
  /* attach the null space to the matrix, this probably is not needed but does no harm */
  CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp));
  CHKERRQ(MatSetNullSpace(appctx.SEMop.grad,nsp));
  CHKERRQ(MatNullSpaceTest(nsp,appctx.SEMop.grad,NULL));
  CHKERRQ(MatNullSpaceDestroy(&nsp));

  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&appctx.ts));
  CHKERRQ(TSSetProblemType(appctx.ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(appctx.ts,TSRK));
  CHKERRQ(TSSetDM(appctx.ts,appctx.da));
  CHKERRQ(TSSetTime(appctx.ts,0.0));
  CHKERRQ(TSSetTimeStep(appctx.ts,appctx.initial_dt));
  CHKERRQ(TSSetMaxSteps(appctx.ts,appctx.param.steps));
  CHKERRQ(TSSetMaxTime(appctx.ts,appctx.param.Tend));
  CHKERRQ(TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTolerances(appctx.ts,1e-7,NULL,1e-7,NULL));
  CHKERRQ(TSSetFromOptions(appctx.ts));
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  CHKERRQ(TSGetTimeStep(appctx.ts,&appctx.initial_dt));
  CHKERRQ(TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx));
  CHKERRQ(TSSetRHSJacobian(appctx.ts,appctx.SEMop.stiff,appctx.SEMop.stiff,RHSJacobian,&appctx));

  /* Set Objective and Initial conditions for the problem and compute Objective function (evolution of true_solution to final time */
  CHKERRQ(InitialConditions(appctx.dat.ic,&appctx));
  CHKERRQ(TrueSolution(appctx.dat.true_solution,&appctx));
  CHKERRQ(ComputeObjective(appctx.param.Tend,appctx.dat.obj,&appctx));

  CHKERRQ(TSSetSaveTrajectory(appctx.ts));
  CHKERRQ(TSSetFromOptions(appctx.ts));

  /* Create TAO solver and set desired solution method  */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetMonitor(tao,MonitorError,&appctx,NULL));
  CHKERRQ(TaoSetType(tao,TAOBQNLS));
  CHKERRQ(TaoSetSolution(tao,appctx.dat.ic));
  /* Set routine for function and gradient evaluation  */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&appctx));
  /* Check for any TAO command line options  */
  CHKERRQ(TaoSetTolerances(tao,1e-8,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoSolve(tao));

  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(MatDestroy(&appctx.SEMop.stiff));
  CHKERRQ(MatDestroy(&appctx.SEMop.keptstiff));
  CHKERRQ(MatDestroy(&appctx.SEMop.grad));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&appctx.dat.ic));
  CHKERRQ(VecDestroy(&appctx.dat.true_solution));
  CHKERRQ(VecDestroy(&appctx.dat.obj));
  CHKERRQ(VecDestroy(&appctx.SEMop.grid));
  CHKERRQ(VecDestroy(&appctx.SEMop.mass));
  CHKERRQ(VecDestroy(&appctx.dat.curr_sol));
  CHKERRQ(PetscFree2(appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights));
  CHKERRQ(DMDestroy(&appctx.da));
  CHKERRQ(TSDestroy(&appctx.ts));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
  CHKERRQ(PetscFinalize());
  return 0;
}

/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the initial conditions for the Tao optimization solve (these are also initial conditions for the first TSSolve()

                       The routine TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscInt          i,xs,xn;

  PetscFunctionBegin;
  CHKERRQ(DMDAVecGetArray(appctx->da,u,&s));
  CHKERRQ(DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  CHKERRQ(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));
  for (i=xs; i<xs+xn; i++) {
    s[i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*xg[i])/(2.0+PetscCosScalar(PETSC_PI*xg[i]))+0.25*PetscExpReal(-4.0*PetscPowReal(xg[i]-2.0,2.0));
  }
  CHKERRQ(DMDAVecRestoreArray(appctx->da,u,&s));
  CHKERRQ(DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  PetscFunctionReturn(0);
}

/*
   TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function.

             InitialConditions() computes the initial conditions for the beginning of the Tao iterations

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode TrueSolution(Vec u,AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscInt          i,xs,xn;

  PetscFunctionBegin;
  CHKERRQ(DMDAVecGetArray(appctx->da,u,&s));
  CHKERRQ(DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  CHKERRQ(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));
  for (i=xs; i<xs+xn; i++) {
    s[i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*xg[i])/(2.0+PetscCosScalar(PETSC_PI*xg[i]));
  }
  CHKERRQ(DMDAVecRestoreArray(appctx->da,u,&s));
  CHKERRQ(DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
/*
   Sets the desired profile for the final end time

   Input Parameters:
   t - final time
   obj - vector storing the desired profile
   appctx - user-defined application context

*/
PetscErrorCode ComputeObjective(PetscReal t,Vec obj,AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscInt          i, xs,xn;

  PetscFunctionBegin;
  CHKERRQ(DMDAVecGetArray(appctx->da,obj,&s));
  CHKERRQ(DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  CHKERRQ(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));
  for (i=xs; i<xs+xn; i++) {
    s[i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*xg[i])*PetscExpScalar(-PETSC_PI*PETSC_PI*t*appctx->param.mu)\
              /(2.0+PetscExpScalar(-PETSC_PI*PETSC_PI*t*appctx->param.mu)*PetscCosScalar(PETSC_PI*xg[i]));
  }
  CHKERRQ(DMDAVecRestoreArray(appctx->da,obj,&s));
  CHKERRQ(DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,(void*)&xg));
  PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  AppCtx          *appctx = (AppCtx*)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatMult(appctx->SEMop.grad,globalin,globalout)); /* grad u */
  CHKERRQ(VecPointwiseMult(globalout,globalin,globalout)); /* u grad u */
  CHKERRQ(VecScale(globalout, -1.0));
  CHKERRQ(MatMultAdd(appctx->SEMop.keptstiff,globalin,globalout,globalout));
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

  CHKERRQ(MatCopy(appctx->SEMop.grad,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDiagonalScale(A,globalin,NULL));

  /*    A  = A + diag(Gu) */
  CHKERRQ(VecDuplicate(globalin,&Gglobalin));
  CHKERRQ(MatMult(appctx->SEMop.grad,globalin,Gglobalin));
  CHKERRQ(MatDiagonalSet(A,Gglobalin,ADD_VALUES));
  CHKERRQ(VecDestroy(&Gglobalin));

  /*   A  = K - A    */
  CHKERRQ(MatScale(A,-1.0));
  CHKERRQ(MatAXPY(A,1.0,appctx->SEMop.keptstiff,SAME_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

/*
   RHSMatrixLaplacian - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

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

  PetscFunctionBegin;
  /*
   Creates the element stiffness matrix for the given gll
   */
  CHKERRQ(PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  /* workaround for clang analyzer warning: Division by zero */
  PetscCheck(appctx->param.N > 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Spectral element order should be > 1");

  /* scale by the size of the element */
  for (i=0; i<appctx->param.N; i++) {
    vv=-appctx->param.mu*2.0/appctx->param.Le;
    for (j=0; j<appctx->param.N; j++) temp[i][j]=temp[i][j]*vv;
  }

  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));

  xs   = xs/(appctx->param.N-1);
  xn   = xn/(appctx->param.N-1);

  CHKERRQ(PetscMalloc1(appctx->param.N,&rowsDM));
  /*
   loop over local elements
   */
  for (j=xs; j<xs+xn; j++) {
    for (l=0; l<appctx->param.N; l++) {
      rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
    }
    CHKERRQ(MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES));
  }
  CHKERRQ(PetscFree(rowsDM));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecReciprocal(appctx->SEMop.mass));
  CHKERRQ(MatDiagonalScale(A,appctx->SEMop.mass,0));
  CHKERRQ(VecReciprocal(appctx->SEMop.mass));

  CHKERRQ(PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  PetscFunctionReturn(0);
}

/*
   RHSMatrixAdvection - User-provided routine to compute the right-hand-side
   matrix for the Advection equation.

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

  PetscFunctionBegin;
  /*
   Creates the advection matrix for the given gll
   */
  CHKERRQ(PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  CHKERRQ(DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL));

  xs   = xs/(appctx->param.N-1);
  xn   = xn/(appctx->param.N-1);

  CHKERRQ(PetscMalloc1(appctx->param.N,&rowsDM));
  for (j=xs; j<xs+xn; j++) {
    for (l=0; l<appctx->param.N; l++) {
      rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
    }
    CHKERRQ(MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES));
  }
  CHKERRQ(PetscFree(rowsDM));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecReciprocal(appctx->SEMop.mass));
  CHKERRQ(MatDiagonalScale(A,appctx->SEMop.mass,0));
  CHKERRQ(VecReciprocal(appctx->SEMop.mass));
  CHKERRQ(PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&temp));
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   IC   - the input vector
   ctx - optional user-defined context, as set when calling TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient

   Notes:

          The forward equation is
              M u_t = F(U)
          which is converted to
                u_t = M^{-1} F(u)
          in the user code since TS has no direct way of providing a mass matrix. The Jacobian of this is
                 M^{-1} J
          where J is the Jacobian of F. Now the adjoint equation is
                M v_t = J^T v
          but TSAdjoint does not solve this since it can only solve the transposed system for the
          Jacobian the user provided. Hence TSAdjoint solves
                 w_t = J^T M^{-1} w  (where w = M v)
          since there is no way to indicate the mass matrix as a separate entity to TS. Thus one
          must be careful in initializing the "adjoint equation" and using the result. This is
          why
              G = -2 M(u(T) - u_d)
          below (instead of -2(u(T) - u_d) and why the result is
              G = G/appctx->SEMop.mass (that is G = M^{-1}w)
          below (instead of just the result of the "adjoint solve").

*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  AppCtx             *appctx = (AppCtx*)ctx;     /* user-defined application context */
  Vec                temp;
  PetscInt           its;
  PetscReal          ff, gnorm, cnorm, xdiff,errex;
  TaoConvergedReason reason;

  PetscFunctionBegin;
  CHKERRQ(TSSetTime(appctx->ts,0.0));
  CHKERRQ(TSSetStepNumber(appctx->ts,0));
  CHKERRQ(TSSetTimeStep(appctx->ts,appctx->initial_dt));
  CHKERRQ(VecCopy(IC,appctx->dat.curr_sol));

  CHKERRQ(TSSolve(appctx->ts,appctx->dat.curr_sol));

  CHKERRQ(VecWAXPY(G,-1.0,appctx->dat.curr_sol,appctx->dat.obj));

  /*
     Compute the L2-norm of the objective function, cost function is f
  */
  CHKERRQ(VecDuplicate(G,&temp));
  CHKERRQ(VecPointwiseMult(temp,G,G));
  CHKERRQ(VecDot(temp,appctx->SEMop.mass,f));

  /* local error evaluation   */
  CHKERRQ(VecWAXPY(temp,-1.0,appctx->dat.ic,appctx->dat.true_solution));
  CHKERRQ(VecPointwiseMult(temp,temp,temp));
  /* for error evaluation */
  CHKERRQ(VecDot(temp,appctx->SEMop.mass,&errex));
  CHKERRQ(VecDestroy(&temp));
  errex  = PetscSqrtReal(errex);

  /*
     Compute initial conditions for the adjoint integration. See Notes above
  */

  CHKERRQ(VecScale(G, -2.0));
  CHKERRQ(VecPointwiseMult(G,G,appctx->SEMop.mass));
  CHKERRQ(TSSetCostGradients(appctx->ts,1,&G,NULL));
  CHKERRQ(TSAdjointSolve(appctx->ts));
  CHKERRQ(VecPointwiseDivide(G,G,appctx->SEMop.mass));

  CHKERRQ(TaoGetSolutionStatus(tao, &its, &ff, &gnorm, &cnorm, &xdiff, &reason));
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorError(Tao tao,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  Vec            temp;
  PetscReal      nrm;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(appctx->dat.ic,&temp));
  CHKERRQ(VecWAXPY(temp,-1.0,appctx->dat.ic,appctx->dat.true_solution));
  CHKERRQ(VecPointwiseMult(temp,temp,temp));
  CHKERRQ(VecDot(temp,appctx->SEMop.mass,&nrm));
  CHKERRQ(VecDestroy(&temp));
  nrm  = PetscSqrtReal(nrm);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error for initial conditions %g\n",(double)nrm));
  PetscFunctionReturn(0);
}

/*TEST

    build:
      requires: !complex

    test:
      args: -tao_max_it 5 -tao_gatol 1.e-4
      requires: !single

    test:
      suffix: 2
      nsize: 2
      args: -tao_max_it 5 -tao_gatol 1.e-4
      requires: !single

TEST*/
