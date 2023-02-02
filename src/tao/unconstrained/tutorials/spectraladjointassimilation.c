
static char help[] = "Solves a simple data assimilation problem with one dimensional advection diffusion equation using TSAdjoint\n\n";

/*

    Not yet tested in parallel

*/

/* ------------------------------------------------------------------------

   This program uses the one-dimensional advection-diffusion equation),
       u_t = mu*u_xx - a u_x,
   on the domain 0 <= x <= 1, with periodic boundary conditions

   to demonstrate solving a data assimilation problem of finding the initial conditions
   to produce a given solution at a fixed time.

   The operators are discretized with the spectral element method

  ------------------------------------------------------------------------- */

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

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
  PetscInt   n;       /* number of nodes */
  PetscReal *nodes;   /* GLL nodes */
  PetscReal *weights; /* GLL weights */
} PetscGLL;

typedef struct {
  PetscInt  N;               /* grid points per elements*/
  PetscInt  E;               /* number of elements */
  PetscReal tol_L2, tol_max; /* error norms */
  PetscInt  steps;           /* number of timesteps */
  PetscReal Tend;            /* endtime */
  PetscReal mu;              /* viscosity */
  PetscReal a;               /* advection speed */
  PetscReal L;               /* total length of domain */
  PetscReal Le;
  PetscReal Tadj;
} PetscParam;

typedef struct {
  Vec reference; /* desired end state */
  Vec grid;      /* total grid */
  Vec grad;
  Vec ic;
  Vec curr_sol;
  Vec joe;
  Vec true_solution; /* actual initial conditions for the final solution */
} PetscData;

typedef struct {
  Vec      grid;  /* total grid */
  Vec      mass;  /* mass matrix for total integration */
  Mat      stiff; /* stifness matrix */
  Mat      advec;
  Mat      keptstiff;
  PetscGLL gll;
} PetscSEMOperators;

typedef struct {
  DM                da; /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  TS                ts;
  PetscReal         initial_dt;
  PetscReal        *solutioncoefficients;
  PetscInt          ncoeff;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
extern PetscErrorCode RHSLaplacian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode RHSAdvection(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode InitialConditions(Vec, AppCtx *);
extern PetscErrorCode ComputeReference(TS, PetscReal, Vec, AppCtx *);
extern PetscErrorCode MonitorError(Tao, void *);
extern PetscErrorCode MonitorDestroy(void **);
extern PetscErrorCode ComputeSolutionCoefficients(AppCtx *);
extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  AppCtx       appctx; /* user-defined application context */
  Tao          tao;
  Vec          u; /* approximate solution vector */
  PetscInt     i, xs, xm, ind, j, lenglob;
  PetscReal    x, *wrk_ptr1, *wrk_ptr2;
  MatNullSpace nsp;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /*initialize parameters */
  appctx.param.N     = 10;      /* order of the spectral element */
  appctx.param.E     = 8;       /* number of elements */
  appctx.param.L     = 1.0;     /* length of the domain */
  appctx.param.mu    = 0.00001; /* diffusion coefficient */
  appctx.param.a     = 0.0;     /* advection speed */
  appctx.initial_dt  = 1e-4;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 0.01;
  appctx.ncoeff      = 2;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &appctx.param.N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-E", &appctx.param.E, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ncoeff", &appctx.ncoeff, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Tend", &appctx.param.Tend, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &appctx.param.mu, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-a", &appctx.param.a, NULL));
  appctx.param.Le = appctx.param.L / appctx.param.E;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscMalloc2(appctx.param.N, &appctx.SEMop.gll.nodes, appctx.param.N, &appctx.SEMop.gll.weights));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(appctx.param.N, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights));
  appctx.SEMop.gll.n = appctx.param.N;
  lenglob            = appctx.param.E * (appctx.param.N - 1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, lenglob, 1, 1, NULL, &appctx.da));
  PetscCall(DMSetFromOptions(appctx.da));
  PetscCall(DMSetUp(appctx.da));

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  PetscCall(DMCreateGlobalVector(appctx.da, &u));
  PetscCall(VecDuplicate(u, &appctx.dat.ic));
  PetscCall(VecDuplicate(u, &appctx.dat.true_solution));
  PetscCall(VecDuplicate(u, &appctx.dat.reference));
  PetscCall(VecDuplicate(u, &appctx.SEMop.grid));
  PetscCall(VecDuplicate(u, &appctx.SEMop.mass));
  PetscCall(VecDuplicate(u, &appctx.dat.curr_sol));
  PetscCall(VecDuplicate(u, &appctx.dat.joe));

  PetscCall(DMDAGetCorners(appctx.da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMDAVecGetArray(appctx.da, appctx.SEMop.grid, &wrk_ptr1));
  PetscCall(DMDAVecGetArray(appctx.da, appctx.SEMop.mass, &wrk_ptr2));

  /* Compute function over the locally owned part of the grid */

  xs = xs / (appctx.param.N - 1);
  xm = xm / (appctx.param.N - 1);

  /*
     Build total grid and mass over entire mesh (multi-elemental)
  */

  for (i = xs; i < xs + xm; i++) {
    for (j = 0; j < appctx.param.N - 1; j++) {
      x             = (appctx.param.Le / 2.0) * (appctx.SEMop.gll.nodes[j] + 1.0) + appctx.param.Le * i;
      ind           = i * (appctx.param.N - 1) + j;
      wrk_ptr1[ind] = x;
      wrk_ptr2[ind] = .5 * appctx.param.Le * appctx.SEMop.gll.weights[j];
      if (j == 0) wrk_ptr2[ind] += .5 * appctx.param.Le * appctx.SEMop.gll.weights[j];
    }
  }
  PetscCall(DMDAVecRestoreArray(appctx.da, appctx.SEMop.grid, &wrk_ptr1));
  PetscCall(DMDAVecRestoreArray(appctx.da, appctx.SEMop.mass, &wrk_ptr2));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE));
  PetscCall(DMCreateMatrix(appctx.da, &appctx.SEMop.stiff));
  PetscCall(DMCreateMatrix(appctx.da, &appctx.SEMop.advec));

  /*
   For linear problems with a time-dependent f(u,t) in the equation
   u_t = f(u,t), the user provides the discretized right-hand-side
   as a time-dependent matrix.
   */
  PetscCall(RHSLaplacian(appctx.ts, 0.0, u, appctx.SEMop.stiff, appctx.SEMop.stiff, &appctx));
  PetscCall(RHSAdvection(appctx.ts, 0.0, u, appctx.SEMop.advec, appctx.SEMop.advec, &appctx));
  PetscCall(MatAXPY(appctx.SEMop.stiff, -1.0, appctx.SEMop.advec, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatDuplicate(appctx.SEMop.stiff, MAT_COPY_VALUES, &appctx.SEMop.keptstiff));

  /* attach the null space to the matrix, this probably is not needed but does no harm */
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nsp));
  PetscCall(MatSetNullSpace(appctx.SEMop.stiff, nsp));
  PetscCall(MatNullSpaceTest(nsp, appctx.SEMop.stiff, NULL));
  PetscCall(MatNullSpaceDestroy(&nsp));

  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &appctx.ts));
  PetscCall(TSSetSolutionFunction(appctx.ts, (PetscErrorCode(*)(TS, PetscReal, Vec, void *))ComputeReference, &appctx));
  PetscCall(TSSetProblemType(appctx.ts, TS_LINEAR));
  PetscCall(TSSetType(appctx.ts, TSRK));
  PetscCall(TSSetDM(appctx.ts, appctx.da));
  PetscCall(TSSetTime(appctx.ts, 0.0));
  PetscCall(TSSetTimeStep(appctx.ts, appctx.initial_dt));
  PetscCall(TSSetMaxSteps(appctx.ts, appctx.param.steps));
  PetscCall(TSSetMaxTime(appctx.ts, appctx.param.Tend));
  PetscCall(TSSetExactFinalTime(appctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTolerances(appctx.ts, 1e-7, NULL, 1e-7, NULL));
  PetscCall(TSSetFromOptions(appctx.ts));
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  PetscCall(TSGetTimeStep(appctx.ts, &appctx.initial_dt));
  PetscCall(TSSetRHSFunction(appctx.ts, NULL, TSComputeRHSFunctionLinear, &appctx));
  PetscCall(TSSetRHSJacobian(appctx.ts, appctx.SEMop.stiff, appctx.SEMop.stiff, TSComputeRHSJacobianConstant, &appctx));
  /*  PetscCall(TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx));
      PetscCall(TSSetRHSJacobian(appctx.ts,appctx.SEMop.stiff,appctx.SEMop.stiff,RHSJacobian,&appctx)); */

  /* Set random initial conditions as initial guess, compute analytic reference solution and analytic (true) initial conditions */
  PetscCall(ComputeSolutionCoefficients(&appctx));
  PetscCall(InitialConditions(appctx.dat.ic, &appctx));
  PetscCall(ComputeReference(appctx.ts, appctx.param.Tend, appctx.dat.reference, &appctx));
  PetscCall(ComputeReference(appctx.ts, 0.0, appctx.dat.true_solution, &appctx));

  /* Set up to save trajectory before TSSetFromOptions() so that TSTrajectory options can be captured */
  PetscCall(TSSetSaveTrajectory(appctx.ts));
  PetscCall(TSSetFromOptions(appctx.ts));

  /* Create TAO solver and set desired solution method  */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetMonitor(tao, MonitorError, &appctx, MonitorDestroy));
  PetscCall(TaoSetType(tao, TAOBQNLS));
  PetscCall(TaoSetSolution(tao, appctx.dat.ic));
  /* Set routine for function and gradient evaluation  */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&appctx));
  /* Check for any TAO command line options  */
  PetscCall(TaoSetTolerances(tao, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFree(appctx.solutioncoefficients));
  PetscCall(MatDestroy(&appctx.SEMop.advec));
  PetscCall(MatDestroy(&appctx.SEMop.stiff));
  PetscCall(MatDestroy(&appctx.SEMop.keptstiff));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&appctx.dat.ic));
  PetscCall(VecDestroy(&appctx.dat.joe));
  PetscCall(VecDestroy(&appctx.dat.true_solution));
  PetscCall(VecDestroy(&appctx.dat.reference));
  PetscCall(VecDestroy(&appctx.SEMop.grid));
  PetscCall(VecDestroy(&appctx.SEMop.mass));
  PetscCall(VecDestroy(&appctx.dat.curr_sol));
  PetscCall(PetscFree2(appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights));
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
    Computes the coefficients for the analytic solution to the PDE
*/
PetscErrorCode ComputeSolutionCoefficients(AppCtx *appctx)
{
  PetscRandom rand;
  PetscInt    i;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(appctx->ncoeff, &appctx->solutioncoefficients));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetInterval(rand, .9, 1.0));
  for (i = 0; i < appctx->ncoeff; i++) PetscCall(PetscRandomGetValue(rand, &appctx->solutioncoefficients[i]));
  PetscCall(PetscRandomDestroy(&rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the (random) initial conditions for the Tao optimization solve (these are also initial conditions for the first TSSolve()

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u, AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscInt           i, j, lenglob;
  PetscReal          sum, val;
  PetscRandom        rand;

  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetInterval(rand, .9, 1.0));
  PetscCall(DMDAVecGetArray(appctx->da, u, &s));
  PetscCall(DMDAVecGetArrayRead(appctx->da, appctx->SEMop.grid, (void *)&xg));
  lenglob = appctx->param.E * (appctx->param.N - 1);
  for (i = 0; i < lenglob; i++) {
    s[i] = 0;
    for (j = 0; j < appctx->ncoeff; j++) {
      PetscCall(PetscRandomGetValue(rand, &val));
      s[i] += val * PetscSinScalar(2 * (j + 1) * PETSC_PI * xg[i]);
    }
  }
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(DMDAVecRestoreArray(appctx->da, u, &s));
  PetscCall(DMDAVecRestoreArrayRead(appctx->da, appctx->SEMop.grid, (void *)&xg));
  /* make sure initial conditions do not contain the constant functions, since with periodic boundary conditions the constant functions introduce a null space */
  PetscCall(VecSum(u, &sum));
  PetscCall(VecShift(u, -sum / lenglob));
  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode TrueSolution(Vec u, AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscInt           i, j, lenglob;
  PetscReal          sum;

  PetscFunctionBegin;
  PetscCall(DMDAVecGetArray(appctx->da, u, &s));
  PetscCall(DMDAVecGetArrayRead(appctx->da, appctx->SEMop.grid, (void *)&xg));
  lenglob = appctx->param.E * (appctx->param.N - 1);
  for (i = 0; i < lenglob; i++) {
    s[i] = 0;
    for (j = 0; j < appctx->ncoeff; j++) s[i] += appctx->solutioncoefficients[j] * PetscSinScalar(2 * (j + 1) * PETSC_PI * xg[i]);
  }
  PetscCall(DMDAVecRestoreArray(appctx->da, u, &s));
  PetscCall(DMDAVecRestoreArrayRead(appctx->da, appctx->SEMop.grid, (void *)&xg));
  /* make sure initial conditions do not contain the constant functions, since with periodic boundary conditions the constant functions introduce a null space */
  PetscCall(VecSum(u, &sum));
  PetscCall(VecShift(u, -sum / lenglob));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* --------------------------------------------------------------------- */
/*
   Sets the desired profile for the final end time

   Input Parameters:
   t - final time
   obj - vector storing the desired profile
   appctx - user-defined application context

*/
PetscErrorCode ComputeReference(TS ts, PetscReal t, Vec obj, AppCtx *appctx)
{
  PetscScalar       *s, tc;
  const PetscScalar *xg;
  PetscInt           i, j, lenglob;

  PetscFunctionBegin;
  PetscCall(DMDAVecGetArray(appctx->da, obj, &s));
  PetscCall(DMDAVecGetArrayRead(appctx->da, appctx->SEMop.grid, (void *)&xg));
  lenglob = appctx->param.E * (appctx->param.N - 1);
  for (i = 0; i < lenglob; i++) {
    s[i] = 0;
    for (j = 0; j < appctx->ncoeff; j++) {
      tc = -appctx->param.mu * (j + 1) * (j + 1) * 4.0 * PETSC_PI * PETSC_PI * t;
      s[i] += appctx->solutioncoefficients[j] * PetscSinScalar(2 * (j + 1) * PETSC_PI * (xg[i] + appctx->param.a * t)) * PetscExpReal(tc);
    }
  }
  PetscCall(DMDAVecRestoreArray(appctx->da, obj, &s));
  PetscCall(DMDAVecRestoreArrayRead(appctx->da, appctx->SEMop.grid, (void *)&xg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec globalin, Vec globalout, void *ctx)
{
  AppCtx *appctx = (AppCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(MatMult(appctx->SEMop.keptstiff, globalin, globalout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec globalin, Mat A, Mat B, void *ctx)
{
  AppCtx *appctx = (AppCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(MatCopy(appctx->SEMop.keptstiff, A, DIFFERENT_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------------------------------------------- */

/*
   RHSLaplacian -   matrix for diffusion

   Input Parameters:
   ts - the TS context
   t - current time  (ignored)
   X - current solution (ignored)
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different matrix from which the preconditioner is built
   str - flag indicating matrix structure

   Scales by the inverse of the mass matrix (perhaps that should be pulled out)

*/
PetscErrorCode RHSLaplacian(TS ts, PetscReal t, Vec X, Mat A, Mat BB, void *ctx)
{
  PetscReal **temp;
  PetscReal   vv;
  AppCtx     *appctx = (AppCtx *)ctx; /* user-defined application context */
  PetscInt    i, xs, xn, l, j;
  PetscInt   *rowsDM;

  PetscFunctionBegin;
  /*
   Creates the element stiffness matrix for the given gll
   */
  PetscCall(PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &temp));

  /* scale by the size of the element */
  for (i = 0; i < appctx->param.N; i++) {
    vv = -appctx->param.mu * 2.0 / appctx->param.Le;
    for (j = 0; j < appctx->param.N; j++) temp[i][j] = temp[i][j] * vv;
  }

  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(DMDAGetCorners(appctx->da, &xs, NULL, NULL, &xn, NULL, NULL));

  PetscCheck(appctx->param.N - 1 >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Polynomial order must be at least 2");
  xs = xs / (appctx->param.N - 1);
  xn = xn / (appctx->param.N - 1);

  PetscCall(PetscMalloc1(appctx->param.N, &rowsDM));
  /*
   loop over local elements
   */
  for (j = xs; j < xs + xn; j++) {
    for (l = 0; l < appctx->param.N; l++) rowsDM[l] = 1 + (j - xs) * (appctx->param.N - 1) + l;
    PetscCall(MatSetValuesLocal(A, appctx->param.N, rowsDM, appctx->param.N, rowsDM, &temp[0][0], ADD_VALUES));
  }
  PetscCall(PetscFree(rowsDM));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecReciprocal(appctx->SEMop.mass));
  PetscCall(MatDiagonalScale(A, appctx->SEMop.mass, 0));
  PetscCall(VecReciprocal(appctx->SEMop.mass));

  PetscCall(PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Almost identical to Laplacian

    Note that the element matrix is NOT scaled by the size of element like the Laplacian term.
 */
PetscErrorCode RHSAdvection(TS ts, PetscReal t, Vec X, Mat A, Mat BB, void *ctx)
{
  PetscReal **temp;
  PetscReal   vv;
  AppCtx     *appctx = (AppCtx *)ctx; /* user-defined application context */
  PetscInt    i, xs, xn, l, j;
  PetscInt   *rowsDM;

  PetscFunctionBegin;
  /*
   Creates the element stiffness matrix for the given gll
   */
  PetscCall(PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &temp));

  /* scale by the size of the element */
  for (i = 0; i < appctx->param.N; i++) {
    vv = -appctx->param.a;
    for (j = 0; j < appctx->param.N; j++) temp[i][j] = temp[i][j] * vv;
  }

  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(DMDAGetCorners(appctx->da, &xs, NULL, NULL, &xn, NULL, NULL));

  PetscCheck(appctx->param.N - 1 >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Polynomial order must be at least 2");
  xs = xs / (appctx->param.N - 1);
  xn = xn / (appctx->param.N - 1);

  PetscCall(PetscMalloc1(appctx->param.N, &rowsDM));
  /*
   loop over local elements
   */
  for (j = xs; j < xs + xn; j++) {
    for (l = 0; l < appctx->param.N; l++) rowsDM[l] = 1 + (j - xs) * (appctx->param.N - 1) + l;
    PetscCall(MatSetValuesLocal(A, appctx->param.N, rowsDM, appctx->param.N, rowsDM, &temp[0][0], ADD_VALUES));
  }
  PetscCall(PetscFree(rowsDM));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecReciprocal(appctx->SEMop.mass));
  PetscCall(MatDiagonalScale(A, appctx->SEMop.mass, 0));
  PetscCall(VecReciprocal(appctx->SEMop.mass));

  PetscCall(PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   ic   - the input vector
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
          below (instead of -2(u(T) - u_d)

*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec ic, PetscReal *f, Vec G, void *ctx)
{
  AppCtx *appctx = (AppCtx *)ctx; /* user-defined application context */
  Vec     temp;

  PetscFunctionBegin;
  PetscCall(TSSetTime(appctx->ts, 0.0));
  PetscCall(TSSetStepNumber(appctx->ts, 0));
  PetscCall(TSSetTimeStep(appctx->ts, appctx->initial_dt));
  PetscCall(VecCopy(ic, appctx->dat.curr_sol));

  PetscCall(TSSolve(appctx->ts, appctx->dat.curr_sol));
  PetscCall(VecCopy(appctx->dat.curr_sol, appctx->dat.joe));

  /*     Compute the difference between the current ODE solution and target ODE solution */
  PetscCall(VecWAXPY(G, -1.0, appctx->dat.curr_sol, appctx->dat.reference));

  /*     Compute the objective/cost function   */
  PetscCall(VecDuplicate(G, &temp));
  PetscCall(VecPointwiseMult(temp, G, G));
  PetscCall(VecDot(temp, appctx->SEMop.mass, f));
  PetscCall(VecDestroy(&temp));

  /*     Compute initial conditions for the adjoint integration. See Notes above  */
  PetscCall(VecScale(G, -2.0));
  PetscCall(VecPointwiseMult(G, G, appctx->SEMop.mass));
  PetscCall(TSSetCostGradients(appctx->ts, 1, &G, NULL));

  PetscCall(TSAdjointSolve(appctx->ts));
  /* PetscCall(VecPointwiseDivide(G,G,appctx->SEMop.mass));*/
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorError(Tao tao, void *ctx)
{
  AppCtx   *appctx = (AppCtx *)ctx;
  Vec       temp, grad;
  PetscReal nrm;
  PetscInt  its;
  PetscReal fct, gnorm;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(appctx->dat.ic, &temp));
  PetscCall(VecWAXPY(temp, -1.0, appctx->dat.ic, appctx->dat.true_solution));
  PetscCall(VecPointwiseMult(temp, temp, temp));
  PetscCall(VecDot(temp, appctx->SEMop.mass, &nrm));
  nrm = PetscSqrtReal(nrm);
  PetscCall(TaoGetGradient(tao, &grad, NULL, NULL));
  PetscCall(VecPointwiseMult(temp, temp, temp));
  PetscCall(VecDot(temp, appctx->SEMop.mass, &gnorm));
  gnorm = PetscSqrtReal(gnorm);
  PetscCall(VecDestroy(&temp));
  PetscCall(TaoGetIterationNumber(tao, &its));
  PetscCall(TaoGetSolutionStatus(tao, NULL, &fct, NULL, NULL, NULL, NULL));
  if (!its) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%% Iteration Error Objective Gradient-norm\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "history = [\n"));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%3" PetscInt_FMT " %g %g %g\n", its, (double)nrm, (double)fct, (double)gnorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorDestroy(void **ctx)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "];\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     args:  -ts_adapt_dt_max 3.e-3 -E 10 -N 8 -ncoeff 5 -tao_bqnls_mat_lmvm_scale_type none

   test:
     suffix: cn
     requires: !single
     args:  -ts_type cn -ts_dt .003 -pc_type lu -E 10 -N 8 -ncoeff 5 -tao_bqnls_mat_lmvm_scale_type none

   test:
     suffix: 2
     requires: !single
     args:  -ts_adapt_dt_max 3.e-3 -E 10 -N 8 -ncoeff 5  -a .1 -tao_bqnls_mat_lmvm_scale_type none

TEST*/
