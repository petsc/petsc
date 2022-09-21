
static char help[] = "Nonlinear DAE benchmark problems.\n";

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscts.h>

typedef struct _Problem *Problem;
struct _Problem {
  PetscErrorCode (*destroy)(Problem);
  TSIFunction function;
  TSIJacobian jacobian;
  PetscErrorCode (*solution)(PetscReal, Vec, void *);
  MPI_Comm  comm;
  PetscReal final_time;
  PetscInt  n;
  PetscBool hasexact;
  void     *data;
};

/*
      Stiff 3-variable system from chemical reactions, due to Robertson (1966), problem ROBER in Hairer&Wanner, ODE 2, 1996
*/
static PetscErrorCode RoberFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  PetscScalar       *f;
  const PetscScalar *x, *xdot;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArray(F, &f));
  f[0] = xdot[0] + 0.04 * x[0] - 1e4 * x[1] * x[2];
  f[1] = xdot[1] - 0.04 * x[0] + 1e4 * x[1] * x[2] + 3e7 * PetscSqr(x[1]);
  f[2] = xdot[2] - 3e7 * PetscSqr(x[1]);
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat A, Mat B, void *ctx)
{
  PetscInt           rowcol[] = {0, 1, 2};
  PetscScalar        J[3][3];
  const PetscScalar *x, *xdot;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  J[0][0] = a + 0.04;
  J[0][1] = -1e4 * x[2];
  J[0][2] = -1e4 * x[1];
  J[1][0] = -0.04;
  J[1][1] = a + 1e4 * x[2] + 3e7 * 2 * x[1];
  J[1][2] = 1e4 * x[1];
  J[2][0] = 0;
  J[2][1] = -3e7 * 2 * x[1];
  J[2][2] = a;
  PetscCall(MatSetValues(B, 3, rowcol, 3, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberSolution(PetscReal t, Vec X, void *ctx)
{
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCheck(t == 0, PETSC_COMM_WORLD, PETSC_ERR_SUP, "not implemented");
  PetscCall(VecGetArray(X, &x));
  x[0] = 1;
  x[1] = 0;
  x[2] = 0;
  PetscCall(VecRestoreArray(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberCreate(Problem p)
{
  PetscFunctionBeginUser;
  p->destroy    = 0;
  p->function   = &RoberFunction;
  p->jacobian   = &RoberJacobian;
  p->solution   = &RoberSolution;
  p->final_time = 1e11;
  p->n          = 3;
  PetscFunctionReturn(0);
}

/*
     Stiff scalar valued problem
*/

typedef struct {
  PetscReal lambda;
} CECtx;

static PetscErrorCode CEDestroy(Problem p)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(p->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode CEFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  PetscReal          l = ((CECtx *)ctx)->lambda;
  PetscScalar       *f;
  const PetscScalar *x, *xdot;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArray(F, &f));
  f[0] = xdot[0] + l * (x[0] - PetscCosReal(t));
#if 0
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," f(t=%g,x=%g,xdot=%g) = %g\n",(double)t,(double)x[0],(double)xdot[0],(double)f[0]));
#endif
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode CEJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat A, Mat B, void *ctx)
{
  PetscReal          l        = ((CECtx *)ctx)->lambda;
  PetscInt           rowcol[] = {0};
  PetscScalar        J[1][1];
  const PetscScalar *x, *xdot;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  J[0][0] = a + l;
  PetscCall(MatSetValues(B, 1, rowcol, 1, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CESolution(PetscReal t, Vec X, void *ctx)
{
  PetscReal    l = ((CECtx *)ctx)->lambda;
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(X, &x));
  x[0] = l / (l * l + 1) * (l * PetscCosReal(t) + PetscSinReal(t)) - l * l / (l * l + 1) * PetscExpReal(-l * t);
  PetscCall(VecRestoreArray(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode CECreate(Problem p)
{
  CECtx *ce;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc(sizeof(CECtx), &ce));
  p->data = (void *)ce;

  p->destroy    = &CEDestroy;
  p->function   = &CEFunction;
  p->jacobian   = &CEJacobian;
  p->solution   = &CESolution;
  p->final_time = 10;
  p->n          = 1;
  p->hasexact   = PETSC_TRUE;

  ce->lambda = 10;
  PetscOptionsBegin(p->comm, NULL, "CE options", "");
  {
    PetscCall(PetscOptionsReal("-problem_ce_lambda", "Parameter controlling stiffness: xdot + lambda*(x - cos(t))", "", ce->lambda, &ce->lambda, NULL));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*
   Stiff 3-variable oscillatory system from chemical reactions. problem OREGO in Hairer&Wanner
*/
static PetscErrorCode OregoFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  PetscScalar       *f;
  const PetscScalar *x, *xdot;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArray(F, &f));
  f[0] = xdot[0] - 77.27 * (x[1] + x[0] * (1. - 8.375e-6 * x[0] - x[1]));
  f[1] = xdot[1] - 1 / 77.27 * (x[2] - (1. + x[0]) * x[1]);
  f[2] = xdot[2] - 0.161 * (x[0] - x[2]);
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat A, Mat B, void *ctx)
{
  PetscInt           rowcol[] = {0, 1, 2};
  PetscScalar        J[3][3];
  const PetscScalar *x, *xdot;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  J[0][0] = a - 77.27 * ((1. - 8.375e-6 * x[0] - x[1]) - 8.375e-6 * x[0]);
  J[0][1] = -77.27 * (1. - x[0]);
  J[0][2] = 0;
  J[1][0] = 1. / 77.27 * x[1];
  J[1][1] = a + 1. / 77.27 * (1. + x[0]);
  J[1][2] = -1. / 77.27;
  J[2][0] = -0.161;
  J[2][1] = 0;
  J[2][2] = a + 0.161;
  PetscCall(MatSetValues(B, 3, rowcol, 3, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoSolution(PetscReal t, Vec X, void *ctx)
{
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCheck(t == 0, PETSC_COMM_WORLD, PETSC_ERR_SUP, "not implemented");
  PetscCall(VecGetArray(X, &x));
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  PetscCall(VecRestoreArray(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoCreate(Problem p)
{
  PetscFunctionBeginUser;
  p->destroy    = 0;
  p->function   = &OregoFunction;
  p->jacobian   = &OregoJacobian;
  p->solution   = &OregoSolution;
  p->final_time = 360;
  p->n          = 3;
  PetscFunctionReturn(0);
}

/*
   User-defined monitor for comparing to exact solutions when possible
*/
typedef struct {
  MPI_Comm comm;
  Problem  problem;
  Vec      x;
} MonitorCtx;

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal t, Vec x, void *ctx)
{
  MonitorCtx *mon = (MonitorCtx *)ctx;
  PetscReal   h, nrm_x, nrm_exact, nrm_diff;

  PetscFunctionBeginUser;
  if (!mon->problem->solution) PetscFunctionReturn(0);
  PetscCall((*mon->problem->solution)(t, mon->x, mon->problem->data));
  PetscCall(VecNorm(x, NORM_2, &nrm_x));
  PetscCall(VecNorm(mon->x, NORM_2, &nrm_exact));
  PetscCall(VecAYPX(mon->x, -1, x));
  PetscCall(VecNorm(mon->x, NORM_2, &nrm_diff));
  PetscCall(TSGetTimeStep(ts, &h));
  if (step < 0) PetscCall(PetscPrintf(mon->comm, "Interpolated final solution "));
  PetscCall(PetscPrintf(mon->comm, "step %4" PetscInt_FMT " t=%12.8e h=% 8.2e  |x|=%9.2e  |x_e|=%9.2e  |x-x_e|=%9.2e\n", step, (double)t, (double)h, (double)nrm_x, (double)nrm_exact, (double)nrm_diff));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscFunctionList plist = NULL;
  char              pname[256];
  TS                ts;   /* nonlinear solver */
  Vec               x, r; /* solution, residual vectors */
  Mat               A;    /* Jacobian matrix */
  Problem           problem;
  PetscBool         use_monitor = PETSC_FALSE;
  PetscBool         use_result  = PETSC_FALSE;
  PetscInt          steps, nonlinits, linits, snesfails, rejects;
  PetscReal         ftime;
  MonitorCtx        mon;
  PetscMPIInt       size;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  /* Register the available problems */
  PetscCall(PetscFunctionListAdd(&plist, "rober", &RoberCreate));
  PetscCall(PetscFunctionListAdd(&plist, "ce", &CECreate));
  PetscCall(PetscFunctionListAdd(&plist, "orego", &OregoCreate));
  PetscCall(PetscStrcpy(pname, "ce"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Timestepping benchmark options", "");
  {
    PetscCall(PetscOptionsFList("-problem_type", "Name of problem to run", "", plist, pname, pname, sizeof(pname), NULL));
    use_monitor = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-monitor_error", "Display errors relative to exact solutions", "", use_monitor, &use_monitor, NULL));
    PetscCall(PetscOptionsBool("-monitor_result", "Display result", "", use_result, &use_result, NULL));
  }
  PetscOptionsEnd();

  /* Create the new problem */
  PetscCall(PetscNew(&problem));
  problem->comm = MPI_COMM_WORLD;
  {
    PetscErrorCode (*pcreate)(Problem);

    PetscCall(PetscFunctionListFind(plist, pname, &pcreate));
    PetscCheck(pcreate, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No problem '%s'", pname);
    PetscCall((*pcreate)(problem));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, problem->n, problem->n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(VecDuplicate(x, &r));

  mon.comm    = PETSC_COMM_WORLD;
  mon.problem = problem;
  PetscCall(VecDuplicate(x, &mon.x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSROSW)); /* Rosenbrock-W */
  PetscCall(TSSetIFunction(ts, NULL, problem->function, problem->data));
  PetscCall(TSSetIJacobian(ts, A, A, problem->jacobian, problem->data));
  PetscCall(TSSetMaxTime(ts, problem->final_time));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetMaxStepRejections(ts, 10));
  PetscCall(TSSetMaxSNESFailures(ts, -1)); /* unlimited */
  if (use_monitor) PetscCall(TSMonitorSet(ts, &MonitorError, &mon, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall((*problem->solution)(0, x, problem->data));
  PetscCall(TSSetTimeStep(ts, .001));
  PetscCall(TSSetSolution(ts, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, x));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetSNESFailures(ts, &snesfails));
  PetscCall(TSGetStepRejections(ts, &rejects));
  PetscCall(TSGetSNESIterations(ts, &nonlinits));
  PetscCall(TSGetKSPIterations(ts, &linits));
  if (use_result) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "steps %" PetscInt_FMT " (%" PetscInt_FMT " rejected, %" PetscInt_FMT " SNES fails), ftime %g, nonlinits %" PetscInt_FMT ", linits %" PetscInt_FMT "\n", steps, rejects, snesfails, (double)ftime, nonlinits, linits));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&mon.x));
  PetscCall(TSDestroy(&ts));
  if (problem->destroy) PetscCall((*problem->destroy)(problem));
  PetscCall(PetscFree(problem));
  PetscCall(PetscFunctionListDestroy(&plist));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      requires: !complex
      args:  -monitor_result -monitor_error -ts_atol 1e-2 -ts_rtol 1e-2 -ts_exact_final_time interpolate -ts_type arkimex

    test:
      suffix: 2
      requires: !single !complex
      args: -monitor_result -ts_atol 1e-2 -ts_rtol 1e-2 -ts_max_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 0 -ts_adapt_time_step_increase_delay 4

    test:
      suffix: 3
      requires: !single !complex
      args: -monitor_result -ts_atol 1e-2 -ts_rtol 1e-2 -ts_max_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 1

    test:
      suffix: 4

    test:
      suffix: 5
      args: -snes_lag_jacobian 20 -snes_lag_jacobian_persists

TEST*/
