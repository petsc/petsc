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
  Mat            A = (Mat) ctx;

  PetscFunctionBeginUser;
  CHKERRQ(MatMult(A, x, f));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeJacobianLinear(SNES snes, Vec x, Mat A, Mat J, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructProblem1(Mat A, Vec b)
{
  PetscInt       rStart, rEnd, row;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(b, -3.0));
  CHKERRQ(MatGetOwnershipRange(A, &rStart, &rEnd));
  for (row = rStart; row < rEnd; ++row) {
    PetscScalar val = 1.0;

    CHKERRQ(MatSetValues(A, 1, &row, 1, &row, &val, INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckProblem1(Mat A, Vec b, Vec u)
{
  Vec            errorVec;
  PetscReal      norm, error;

  PetscFunctionBeginUser;
  CHKERRQ(VecDuplicate(b, &errorVec));
  CHKERRQ(VecWAXPY(errorVec, -1.0, b, u));
  CHKERRQ(VecNorm(errorVec, NORM_2, &error));
  CHKERRQ(VecNorm(b, NORM_2, &norm));
  PetscCheckFalse(error/norm > 1000.*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  CHKERRQ(VecDestroy(&errorVec));
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructProblem2(Mat A, Vec b)
{
  PetscInt       N = 10, constraintSize = 4;
  PetscInt       row;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(b, -3.0));
  for (row = 0; row < constraintSize; ++row) {
    PetscScalar vals[2] = {1.0, 1.0};
    PetscInt    cols[2];

    cols[0] = row; cols[1] = row + N - constraintSize;
    CHKERRQ(MatSetValues(A, 1, &row, 2, cols, vals, INSERT_VALUES));
  }
  for (row = constraintSize; row < N - constraintSize; ++row) {
    PetscScalar val = 1.0;

    CHKERRQ(MatSetValues(A, 1, &row, 1, &row, &val, INSERT_VALUES));
  }
  for (row = N - constraintSize; row < N; ++row) {
    PetscInt    col = row - (N - constraintSize);
    PetscScalar val = 1.0;

    CHKERRQ(MatSetValues(A, 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckProblem2(Mat A, Vec b, Vec u)
{
  PetscInt          N = 10, constraintSize = 4, r;
  PetscReal         norm, error;
  const PetscScalar *uArray, *bArray;

  PetscFunctionBeginUser;
  CHKERRQ(VecNorm(b, NORM_2, &norm));
  CHKERRQ(VecGetArrayRead(u, &uArray));
  CHKERRQ(VecGetArrayRead(b, &bArray));
  error = 0.0;
  for (r = 0; r < constraintSize; ++r) error += PetscRealPart(PetscSqr(uArray[r] - bArray[r + N-constraintSize]));

  PetscCheckFalse(error/norm > 10000*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  error = 0.0;
  for (r = constraintSize; r < N - constraintSize; ++r) error += PetscRealPart(PetscSqr(uArray[r] - bArray[r]));

  PetscCheckFalse(error/norm > 10000*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  error = 0.0;
  for (r = N - constraintSize; r < N; ++r) error += PetscRealPart(PetscSqr(uArray[r] - (bArray[r - (N-constraintSize)] - bArray[r])));

  PetscCheckFalse(error/norm > 10000*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  CHKERRQ(VecRestoreArrayRead(u, &uArray));
  CHKERRQ(VecRestoreArrayRead(b, &bArray));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  SNES           snes;                 /* nonlinear solver */
  Vec            u,r,b;                /* solution, residual, and rhs vectors */
  Mat            A,J;                  /* Jacobian matrix */
  PetscInt       problem = 1, N = 10;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-problem", &problem, NULL));
  CHKERRQ(VecCreate(comm, &u));
  CHKERRQ(VecSetSizes(u, PETSC_DETERMINE, N));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u, &r));
  CHKERRQ(VecDuplicate(u, &b));

  CHKERRQ(MatCreate(comm, &A));
  CHKERRQ(MatSetSizes(A, PETSC_DETERMINE, PETSC_DETERMINE, N, N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqAIJSetPreallocation(A, 5, NULL));
  J    = A;

  switch (problem) {
  case 1:
    CHKERRQ(ConstructProblem1(A, b));
    break;
  case 2:
    CHKERRQ(ConstructProblem2(A, b));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid problem number %d", problem);
  }

  CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
  CHKERRQ(SNESSetJacobian(snes, A, J, ComputeJacobianLinear, NULL));
  CHKERRQ(SNESSetFunction(snes, r, ComputeFunctionLinear, A));
  CHKERRQ(SNESSetFromOptions(snes));

  CHKERRQ(SNESSolve(snes, b, u));
  CHKERRQ(VecView(u, NULL));

  switch (problem) {
  case 1:
    CHKERRQ(CheckProblem1(A, b, u));
    break;
  case 2:
    CHKERRQ(CheckProblem2(A, b, u));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid problem number %d", problem);
  }

  if (A != J) {
    CHKERRQ(MatDestroy(&A));
  }
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -snes_monitor

   test:
     suffix: 2
     args: -problem 2 -pc_type jacobi -snes_monitor

TEST*/
