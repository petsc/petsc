static char help[] = "Solves the ODE du/dt = poly(t), u(0) = 0. Tests TSResize for varying size.\n\n";

#include <petscts.h>

PetscScalar poly(PetscInt p, PetscReal t)
{
  return p ? t * poly(p - 1, t) : 1.0;
}

PetscScalar dpoly(PetscInt p, PetscReal t)
{
  return p > 0 ? (PetscReal)p * poly(p - 1, t) : 0.0;
}

PetscErrorCode CreateVec(PetscInt lsize, Vec *out)
{
  Vec x;

  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, lsize, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));
  *out = x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMat(PetscInt lsize, Mat *out)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, lsize, lsize, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  /* ensure that the Jacobian matrix has diagonal entries since that is required by TS */
  PetscCall(MatShift(A, (PetscReal)1));
  PetscCall(MatShift(A, (PetscReal)-1));
  *out = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec x, Vec f, void *ctx)
{
  PetscInt *order = (PetscInt *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecSet(f, dpoly(*order, t)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec x, Mat A, Mat B, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(B));
  if (B != A) PetscCall(MatZeroEntries(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Transfer(TS ts, PetscInt nv, Vec vecsin[], Vec vecsout[], void *ctx)
{
  PetscInt n, nnew;

  PetscFunctionBeginUser;
  PetscAssert(nv > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Zero vectors");
  PetscCall(VecGetLocalSize(vecsin[0], &n));
  nnew = n == 2 ? 1 : 2;
  for (PetscInt i = 0; i < nv; i++) {
    const PetscScalar *vals;

    PetscCall(CreateVec(nnew, &vecsout[i]));
    PetscCall(VecGetArrayRead(vecsin[i], &vals));
    PetscCall(VecSet(vecsout[i], vals[0]));
    PetscCall(VecRestoreArrayRead(vecsin[i], &vals));
  }
  Mat A;
  PetscCall(CreateMat(nnew, &A));
  PetscCall(TSSetRHSJacobian(ts, A, A, RHSJacobian, NULL));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TransferSetUp(TS ts, PetscInt step, PetscReal time, Vec sol, PetscBool *resize, void *ctx)
{
  PetscFunctionBeginUser;
  *resize = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Monitor(TS ts, PetscInt n, PetscReal t, Vec x, void *ctx)
{
  const PetscScalar *a;
  PetscScalar       *store = (PetscScalar *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(x, &a));
  if (n < 10) store[n] = a[0];
  PetscCall(VecRestoreArrayRead(x, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS          ts;
  Vec         x;
  Mat         A;
  PetscInt    order = 2;
  PetscScalar results[2][10];
  /* I would like to use 0 here, but linux-gcc-complex-opt-32bit  errors with arkimex with 1.e-18 errors, macOS clang requires an even larger tolerance */
  PetscReal tol = 10 * PETSC_MACHINE_EPSILON;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-order", &order, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));

  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetProblemType(ts, TS_LINEAR));

    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &order));

    PetscCall(CreateMat(1, &A));
    PetscCall(TSSetRHSJacobian(ts, A, A, RHSJacobian, NULL));
    PetscCall(MatDestroy(&A));

    PetscCall(CreateVec(1, &x));
    PetscCall(TSSetSolution(ts, x));
    PetscCall(VecDestroy(&x));

    for (PetscInt j = 0; j < 10; j++) results[i][j] = 0;
    PetscCall(TSMonitorSet(ts, Monitor, results[i], NULL));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    if (i) PetscCall(TSSetResize(ts, TransferSetUp, Transfer, NULL));
    PetscCall(TSSetTime(ts, 0));
    PetscCall(TSSetTimeStep(ts, 1. / 4.));
    PetscCall(TSSetMaxSteps(ts, 10));
    PetscCall(TSSetFromOptions(ts));

    PetscCall(TSSolve(ts, NULL));

    PetscCall(TSDestroy(&ts));
  }

  /* Dump errors if any */
  PetscBool flg = PETSC_FALSE;
  for (PetscInt i = 0; i < 10; i++) {
    PetscReal err = PetscAbsScalar(results[0][i] - results[1][i]);
    if (err > tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error step %" PetscInt_FMT ": %g\n", i, (double)err));
      flg = PETSC_TRUE;
    }
  }
  if (flg) {
    PetscCall(PetscScalarView(10, results[0], PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscScalarView(10, results[1], PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: bdf
      args: -ts_adapt_wnormtype infinity -ts_type bdf -ts_bdf_order {{2 3 4 5 6}} -order 6 -ts_adapt_type {{none basic dsp}} -ksp_type preonly -pc_type lu
      output_file: output/ex17.out

    test:
      suffix: expl
      args: -ts_adapt_wnormtype infinity -ts_type {{euler rk ssp}} -order 6 -ts_adapt_type {{none basic dsp}}
      output_file: output/ex17.out

    test:
      suffix: impl
      args: -ts_adapt_wnormtype infinity -ts_type {{rosw beuler cn alpha theta arkimex}} -order 6 -ts_adapt_type {{none basic dsp}} -ksp_type preonly -pc_type lu
      output_file: output/ex17.out

TEST*/
