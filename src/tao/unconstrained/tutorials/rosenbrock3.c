/* Program usage: mpiexec -n 1 rosenbrock2 [-help] [all TAO options] */

/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>

static char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} (alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2) \n\
or the chained Rosenbrock function:\n\
   sum_{i=0}^{n-1} alpha*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\n";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/
typedef struct {
  PetscInt  n;     /* dimension */
  PetscReal alpha; /* condition parameter */
  PetscBool chained;
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  PetscReal          zero = 0.0;
  Vec                x; /* solution vector */
  Mat                H;
  Tao                tao; /* Tao solver context */
  PetscBool          flg, test_lmvm = PETSC_FALSE;
  PetscMPIInt        size; /* number of processes running */
  AppCtx             user; /* user-defined application context */
  TaoConvergedReason reason;
  PetscInt           its, recycled_its = 0, oneshot_its = 0;

  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  /* Initialize problem parameters */
  user.n       = 2;
  user.alpha   = 99.0;
  user.chained = PETSC_FALSE;
  /* Check for command line arguments to override defaults */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &user.alpha, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-chained", &user.chained, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_lmvm", &test_lmvm, &flg));

  /* Allocate vectors for the solution and gradient */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF, 2, user.n, user.n, 1, NULL, &H));

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOBQNLS));

  /* Set solution vec and an initial guess */
  PetscCall(VecSet(x, zero));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, &user));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, &user));

  /* Check for TAO command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* Solve the problem */
  PetscCall(TaoSetTolerances(tao, 1.e-5, 0.0, 0.0));
  PetscCall(TaoSetMaximumIterations(tao, 5));
  PetscCall(TaoSetRecycleHistory(tao, PETSC_TRUE));
  reason = TAO_CONTINUE_ITERATING;
  flg    = PETSC_FALSE;
  PetscCall(TaoGetRecycleHistory(tao, &flg));
  if (flg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Recycle: enabled\n"));
  while (reason != TAO_CONVERGED_GATOL) {
    PetscCall(TaoSolve(tao));
    PetscCall(TaoGetConvergedReason(tao, &reason));
    PetscCall(TaoGetIterationNumber(tao, &its));
    recycled_its += its;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "-----------------------\n"));
  }

  /* Disable recycling and solve again! */
  PetscCall(TaoSetMaximumIterations(tao, 100));
  PetscCall(TaoSetRecycleHistory(tao, PETSC_FALSE));
  PetscCall(VecSet(x, zero));
  PetscCall(TaoGetRecycleHistory(tao, &flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Recycle: disabled\n"));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetConvergedReason(tao, &reason));
  PetscCheck(reason == TAO_CONVERGED_GATOL, PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED, "Solution failed to converge!");
  PetscCall(TaoGetIterationNumber(tao, &oneshot_its));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "-----------------------\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "recycled its: %" PetscInt_FMT " | oneshot its: %" PetscInt_FMT "\n", recycled_its, oneshot_its));
  PetscCheck(recycled_its == oneshot_its, PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED, "Recycled solution does not match oneshot solution!");

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&H));

  PetscCall(PetscFinalize());
  return 0;
}

/* -------------------------------------------------------------------- */
/*
    FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

    Input Parameters:
.   tao  - the Tao context
.   X    - input vector
.   ptr  - optional user-defined context, as set by TaoSetFunctionGradient()

    Output Parameters:
.   G - vector containing the newly evaluated gradient
.   f - function value

    Note:
    Some optimization methods ask for the function and the gradient evaluation
    at the same time.  Evaluating both at once may be more efficient than
    evaluating each separately.
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx            *user = (AppCtx *)ptr;
  PetscInt           i, nn = user->n / 2;
  PetscReal          ff = 0, t1, t2, alpha = user->alpha;
  PetscScalar       *g;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  /* Get pointers to vector data */
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayWrite(G, &g));

  /* Compute G(X) */
  if (user->chained) {
    g[0] = 0;
    for (i = 0; i < user->n - 1; i++) {
      t1 = x[i + 1] - x[i] * x[i];
      ff += PetscSqr(1 - x[i]) + alpha * t1 * t1;
      g[i] += -2 * (1 - x[i]) + 2 * alpha * t1 * (-2 * x[i]);
      g[i + 1] = 2 * alpha * t1;
    }
  } else {
    for (i = 0; i < nn; i++) {
      t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      t2 = 1 - x[2 * i];
      ff += alpha * t1 * t1 + t2 * t2;
      g[2 * i]     = -4 * alpha * t1 * x[2 * i] - 2.0 * t2;
      g[2 * i + 1] = 2 * alpha * t1;
    }
  }

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayWrite(G, &g));
  *f = ff;

  PetscCall(PetscLogFlops(15.0 * nn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  tao   - the Tao context
.  x     - input vector
.  ptr   - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  H     - Hessian matrix

   Note:  Providing the Hessian may not be necessary.  Only some solvers
   require this matrix.
*/
PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  AppCtx            *user = (AppCtx *)ptr;
  PetscInt           i, ind[2];
  PetscReal          alpha = user->alpha;
  PetscReal          v[2][2];
  const PetscScalar *x;
  PetscBool          assembled;

  PetscFunctionBeginUser;
  /* Zero existing matrix entries */
  PetscCall(MatAssembled(H, &assembled));
  if (assembled || user->chained) PetscCall(MatZeroEntries(H));

  /* Get a pointer to vector data */
  PetscCall(VecGetArrayRead(X, &x));

  /* Compute H(X) entries */
  if (user->chained) {
    for (i = 0; i < user->n - 1; i++) {
      PetscScalar t1 = x[i + 1] - x[i] * x[i];
      v[0][0]        = 2 + 2 * alpha * (t1 * (-2) - 2 * x[i]);
      v[0][1]        = 2 * alpha * (-2 * x[i]);
      v[1][0]        = 2 * alpha * (-2 * x[i]);
      v[1][1]        = 2 * alpha * t1;
      ind[0]         = i;
      ind[1]         = i + 1;
      PetscCall(MatSetValues(H, 2, ind, 2, ind, v[0], ADD_VALUES));
    }
  } else {
    for (i = 0; i < user->n / 2; i++) {
      v[1][1] = 2 * alpha;
      v[0][0] = -4 * alpha * (x[2 * i + 1] - 3 * x[2 * i] * x[2 * i]) + 2;
      v[1][0] = v[0][1] = -4.0 * alpha * x[2 * i];
      ind[0]            = 2 * i;
      ind[1]            = 2 * i + 1;
      PetscCall(MatSetValues(H, 2, ind, 2, ind, v[0], INSERT_VALUES));
    }
  }
  PetscCall(VecRestoreArrayRead(X, &x));

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogFlops(9.0 * user->n / 2.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_type bqnls -tao_monitor
      requires: !single

TEST*/
