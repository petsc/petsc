static char help[] = "Solves the trivial ODE 2 du/dt = 1, u(0) = 0. \n\n";

#include <petscts.h>
#include <petscpc.h>

PetscErrorCode IFunction(TS, PetscReal, Vec, Vec, Vec, void *);
PetscErrorCode IJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);

int main(int argc, char **argv)
{
  TS  ts;
  Vec x;
  Vec f;
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetEquationType(ts, TS_EQ_ODE_IMPLICIT));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &f));
  PetscCall(VecSetSizes(f, 1, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(f));
  PetscCall(VecSetUp(f));
  PetscCall(TSSetIFunction(ts, f, IFunction, NULL));
  PetscCall(VecDestroy(&f));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  /* ensure that the Jacobian matrix has diagonal entries since that is required by TS */
  PetscCall(MatShift(A, (PetscReal)1));
  PetscCall(MatShift(A, (PetscReal)-1));
  PetscCall(TSSetIJacobian(ts, A, A, IJacobian, NULL));
  PetscCall(MatDestroy(&A));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, 1, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));
  PetscCall(TSSetSolution(ts, x));
  PetscCall(VecDestroy(&x));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTimeStep(ts, 1));
  PetscCall(TSSetTime(ts, 0));
  PetscCall(TSSetMaxTime(ts, PETSC_MAX_REAL));
  PetscCall(TSSetMaxSteps(ts, 3));

  /*
      When an ARKIMEX scheme with an explicit stage is used this will error with a message informing the user it is not possible to use
      a non-trivial mass matrix with ARKIMEX schemes with explicit stages.
  */
  PetscCall(TSSolve(ts, NULL));

  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode IFunction(TS ts, PetscReal t, Vec x, Vec xdot, Vec f, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(xdot, f));
  PetscCall(VecScale(f, 2.0));
  PetscCall(VecShift(f, -1.0));
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec x, Vec xdot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  PetscScalar j;

  PetscFunctionBeginUser;
  j = shift * 2.0;
  PetscCall(MatSetValue(B, 0, 0, j, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      suffix: arkimex_explicit_stage
      requires: !defined(PETSCTEST_VALGRIND) defined(PETSC_USE_DEBUG)
      args: -ts_type arkimex -petsc_ci_portable_error_output -error_output_stdout
      filter: grep -E -v "(options_left|memory block|leaked context|not freed before MPI_Finalize|Could be the program crashed)"

    test:
      suffix: arkimex_implicit_stage
      args: -ts_type arkimex -ts_arkimex_type l2 -ts_monitor_solution -ts_monitor

TEST*/
