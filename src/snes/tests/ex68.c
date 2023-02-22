static char help[] = "Test problems for Schur complement solvers.\n\n\n";

#include <petscsnes.h>

/*
Test 1:
  I u = b

  solution: u = b

Test 2:
  / I 0 I \  / u_1 \   / b_1 \
  | 0 I 0 | |  u_2 | = | b_2 |
  \ I 0 0 /  \ u_3 /   \ b_3 /

  solution: u_1 = b_3, u_2 = b_2, u_3 = b_1 - b_3
*/

PetscErrorCode ComputeFunctionLinear(SNES snes, Vec x, Vec f, void *ctx)
{
  Mat A = (Mat)ctx;

  PetscFunctionBeginUser;
  PetscCall(MatMult(A, x, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeJacobianLinear(SNES snes, Vec x, Mat A, Mat J, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ConstructProblem1(Mat A, Vec b)
{
  PetscInt rStart, rEnd, row;

  PetscFunctionBeginUser;
  PetscCall(VecSet(b, -3.0));
  PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
  for (row = rStart; row < rEnd; ++row) {
    PetscScalar val = 1.0;

    PetscCall(MatSetValues(A, 1, &row, 1, &row, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckProblem1(Mat A, Vec b, Vec u)
{
  Vec       errorVec;
  PetscReal norm, error;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(b, &errorVec));
  PetscCall(VecWAXPY(errorVec, -1.0, b, u));
  PetscCall(VecNorm(errorVec, NORM_2, &error));
  PetscCall(VecNorm(b, NORM_2, &norm));
  PetscCheck(error / norm <= 1000. * PETSC_MACHINE_EPSILON, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", (double)(error / norm));
  PetscCall(VecDestroy(&errorVec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ConstructProblem2(Mat A, Vec b)
{
  PetscInt N = 10, constraintSize = 4;
  PetscInt row;

  PetscFunctionBeginUser;
  PetscCall(VecSet(b, -3.0));
  for (row = 0; row < constraintSize; ++row) {
    PetscScalar vals[2] = {1.0, 1.0};
    PetscInt    cols[2];

    cols[0] = row;
    cols[1] = row + N - constraintSize;
    PetscCall(MatSetValues(A, 1, &row, 2, cols, vals, INSERT_VALUES));
  }
  for (row = constraintSize; row < N - constraintSize; ++row) {
    PetscScalar val = 1.0;

    PetscCall(MatSetValues(A, 1, &row, 1, &row, &val, INSERT_VALUES));
  }
  for (row = N - constraintSize; row < N; ++row) {
    PetscInt    col = row - (N - constraintSize);
    PetscScalar val = 1.0;

    PetscCall(MatSetValues(A, 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckProblem2(Mat A, Vec b, Vec u)
{
  PetscInt           N = 10, constraintSize = 4, r;
  PetscReal          norm, error;
  const PetscScalar *uArray, *bArray;

  PetscFunctionBeginUser;
  PetscCall(VecNorm(b, NORM_2, &norm));
  PetscCall(VecGetArrayRead(u, &uArray));
  PetscCall(VecGetArrayRead(b, &bArray));
  error = 0.0;
  for (r = 0; r < constraintSize; ++r) error += PetscRealPart(PetscSqr(uArray[r] - bArray[r + N - constraintSize]));

  PetscCheck(error / norm <= 10000 * PETSC_MACHINE_EPSILON, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", (double)(error / norm));
  error = 0.0;
  for (r = constraintSize; r < N - constraintSize; ++r) error += PetscRealPart(PetscSqr(uArray[r] - bArray[r]));

  PetscCheck(error / norm <= 10000 * PETSC_MACHINE_EPSILON, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", (double)(error / norm));
  error = 0.0;
  for (r = N - constraintSize; r < N; ++r) error += PetscRealPart(PetscSqr(uArray[r] - (bArray[r - (N - constraintSize)] - bArray[r])));

  PetscCheck(error / norm <= 10000 * PETSC_MACHINE_EPSILON, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", (double)(error / norm));
  PetscCall(VecRestoreArrayRead(u, &uArray));
  PetscCall(VecRestoreArrayRead(b, &bArray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm comm;
  SNES     snes;    /* nonlinear solver */
  Vec      u, r, b; /* solution, residual, and rhs vectors */
  Mat      A, J;    /* Jacobian matrix */
  PetscInt problem = 1, N = 10;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-problem", &problem, NULL));
  PetscCall(VecCreate(comm, &u));
  PetscCall(VecSetSizes(u, PETSC_DETERMINE, N));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &r));
  PetscCall(VecDuplicate(u, &b));

  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DETERMINE, PETSC_DETERMINE, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
  J = A;

  switch (problem) {
  case 1:
    PetscCall(ConstructProblem1(A, b));
    break;
  case 2:
    PetscCall(ConstructProblem2(A, b));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid problem number %" PetscInt_FMT, problem);
  }

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetJacobian(snes, A, J, ComputeJacobianLinear, NULL));
  PetscCall(SNESSetFunction(snes, r, ComputeFunctionLinear, A));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes, b, u));
  PetscCall(VecView(u, NULL));

  switch (problem) {
  case 1:
    PetscCall(CheckProblem1(A, b, u));
    break;
  case 2:
    PetscCall(CheckProblem2(A, b, u));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid problem number %" PetscInt_FMT, problem);
  }

  if (A != J) PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&b));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -snes_monitor

   test:
     suffix: 2
     args: -problem 2 -pc_type jacobi -snes_monitor

TEST*/
