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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatMult(A, x, f);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(b, -3.0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
  for (row = rStart; row < rEnd; ++row) {
    PetscScalar val = 1.0;

    ierr = MatSetValues(A, 1, &row, 1, &row, &val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CheckProblem1(Mat A, Vec b, Vec u)
{
  Vec            errorVec;
  PetscReal      norm, error;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(b, &errorVec);CHKERRQ(ierr);
  ierr = VecWAXPY(errorVec, -1.0, b, u);CHKERRQ(ierr);
  ierr = VecNorm(errorVec, NORM_2, &error);CHKERRQ(ierr);
  ierr = VecNorm(b, NORM_2, &norm);CHKERRQ(ierr);
  if (error/norm > 1000.*PETSC_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  ierr = VecDestroy(&errorVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructProblem2(Mat A, Vec b)
{
  PetscInt       N = 10, constraintSize = 4;
  PetscInt       row;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(b, -3.0);CHKERRQ(ierr);
  for (row = 0; row < constraintSize; ++row) {
    PetscScalar vals[2] = {1.0, 1.0};
    PetscInt    cols[2];

    cols[0] = row; cols[1] = row + N - constraintSize;
    ierr    = MatSetValues(A, 1, &row, 2, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
  }
  for (row = constraintSize; row < N - constraintSize; ++row) {
    PetscScalar val = 1.0;

    ierr = MatSetValues(A, 1, &row, 1, &row, &val, INSERT_VALUES);CHKERRQ(ierr);
  }
  for (row = N - constraintSize; row < N; ++row) {
    PetscInt    col = row - (N - constraintSize);
    PetscScalar val = 1.0;

    ierr = MatSetValues(A, 1, &row, 1, &col, &val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CheckProblem2(Mat A, Vec b, Vec u)
{
  PetscInt          N = 10, constraintSize = 4, r;
  PetscReal         norm, error;
  const PetscScalar *uArray, *bArray;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr  = VecNorm(b, NORM_2, &norm);CHKERRQ(ierr);
  ierr  = VecGetArrayRead(u, &uArray);CHKERRQ(ierr);
  ierr  = VecGetArrayRead(b, &bArray);CHKERRQ(ierr);
  error = 0.0;
  for (r = 0; r < constraintSize; ++r) error += PetscRealPart(PetscSqr(uArray[r] - bArray[r + N-constraintSize]));

  if (error/norm > 10000*PETSC_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  error = 0.0;
  for (r = constraintSize; r < N - constraintSize; ++r) error += PetscRealPart(PetscSqr(uArray[r] - bArray[r]));

  if (error/norm > 10000*PETSC_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  error = 0.0;
  for (r = N - constraintSize; r < N; ++r) error += PetscRealPart(PetscSqr(uArray[r] - (bArray[r - (N-constraintSize)] - bArray[r])));

  if (error/norm > 10000*PETSC_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Relative error %g is too large", error/norm);
  ierr = VecRestoreArrayRead(u, &uArray);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b, &bArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  SNES           snes;                 /* nonlinear solver */
  Vec            u,r,b;                /* solution, residual, and rhs vectors */
  Mat            A,J;                  /* Jacobian matrix */
  PetscInt       problem = 1, N = 10;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsGetInt(NULL,NULL, "-problem", &problem, NULL);CHKERRQ(ierr);
  ierr = VecCreate(comm, &u);CHKERRQ(ierr);
  ierr = VecSetSizes(u, PETSC_DETERMINE, N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &b);CHKERRQ(ierr);

  ierr = MatCreate(comm, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DETERMINE, PETSC_DETERMINE, N, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 5, NULL);CHKERRQ(ierr);
  J    = A;

  switch (problem) {
  case 1:
    ierr = ConstructProblem1(A, b);CHKERRQ(ierr);
    break;
  case 2:
    ierr = ConstructProblem2(A, b);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid problem number %d", problem);
  }

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, ComputeJacobianLinear, NULL);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, ComputeFunctionLinear, A);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes, b, u);CHKERRQ(ierr);
  ierr = VecView(u, NULL);CHKERRQ(ierr);

  switch (problem) {
  case 1:
    ierr = CheckProblem1(A, b, u);CHKERRQ(ierr);
    break;
  case 2:
    ierr = CheckProblem2(A, b, u);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid problem number %d", problem);
  }

  if (A != J) {
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     args: -snes_monitor

   test:
     suffix: 2
     args: -problem 2 -pc_type jacobi -snes_monitor

TEST*/
