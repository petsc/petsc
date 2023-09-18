static char help[] = "Solves the trivial ODE 2 du/dt = 1, u(0) = 0. \n\n";

#include <petscts.h>
#include <petscpc.h>

PetscErrorCode IFunction(TS, PetscReal, Vec, Vec, Vec, void *);
PetscErrorCode IJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);

int main(int argc, char **argv)
{
  TS        ts;
  Vec       x;
  Mat       A;
  PetscBool flg = PETSC_FALSE, usingimex;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-set_implicit", &flg, NULL));
  if (flg) PetscCall(TSSetEquationType(ts, TS_EQ_ODE_IMPLICIT));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, NULL));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &usingimex));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(TSSetIJacobian(ts, A, A, IJacobian, NULL));
  PetscCall(MatDestroy(&A));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, 1, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));
  PetscCall(TSSetSolution(ts, x));
  PetscCall(VecDestroy(&x));
  PetscCall(TSSetFromOptions(ts));

  /* Need to know if we are using an IMEX scheme to decide on the form
     of the RHS function */
  PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSARKIMEX, &usingimex));
  if (usingimex) {
    PetscCall(TSARKIMEXGetFullyImplicit(ts, &flg));
    if (flg) usingimex = PETSC_FALSE;
  }
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTimeStep(ts, 1));
  PetscCall(TSSetTime(ts, 0));
  PetscCall(TSSetMaxTime(ts, PETSC_MAX_REAL));
  PetscCall(TSSetMaxSteps(ts, 3));

  PetscCall(TSSolve(ts, NULL));

  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec x, Vec f, void *ctx)
{
  PetscBool usingimex = *(PetscBool *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecSet(f, usingimex ? 0.5 : 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IFunction(TS ts, PetscReal t, Vec x, Vec xdot, Vec f, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(xdot, f));
  PetscCall(VecScale(f, 2.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec x, Vec xdot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  PetscScalar j;

  PetscFunctionBeginUser;
  j = shift * 2.0;
  PetscCall(MatSetValue(B, 0, 0, j, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      suffix: arkimex_explicit_stage
      requires: !defined(PETSCTEST_VALGRIND) defined(PETSC_USE_DEBUG)
      args: -ts_type arkimex -petsc_ci_portable_error_output -error_output_stdout -set_implicit
      filter: grep -E -v "(memory block|leaked context|not freed before MPI_Finalize|Could be the program crashed)"

    test:
      suffix: arkimex_implicit_stage
      args: -ts_type arkimex -ts_arkimex_type {{3 l2}} -ts_monitor_solution -ts_monitor

TEST*/
