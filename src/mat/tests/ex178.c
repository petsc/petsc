
static char help[] = "Tests MatInvertVariableBlockEnvelope()\n\n";

#include <petscmat.h>
extern PetscErrorCode MatIsDiagonal(Mat);
extern PetscErrorCode BuildMatrix(const PetscInt *, PetscInt, const PetscInt *, Mat *);

int main(int argc, char **argv)
{
  Mat         A, C, D, F;
  PetscInt    i, j, rows[2], *parts, cnt, N = 21, nblocks, *blocksizes;
  PetscScalar values[2][2];
  PetscReal   rand;
  PetscRandom rctx;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_DENSE));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, 6, 18));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  values[0][0] = 2;
  values[0][1] = 1;
  values[1][0] = 1;
  values[1][1] = 2;
  for (i = 0; i < 3; i++) {
    rows[0] = 2 * i;
    rows[1] = 2 * i + 1;
    PetscCall(MatSetValues(C, 2, rows, 2, rows, (PetscScalar *)values, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatMatTransposeMult(C, C, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatInvertVariableBlockEnvelope(A, MAT_INITIAL_MATRIX, &D));
  PetscCall(MatView(D, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatMatMult(A, D, MAT_INITIAL_MATRIX, 1.0, &F));
  PetscCall(MatView(F, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatIsDiagonal(F));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&F));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscMalloc1(size, &parts));

  for (j = 0; j < 128; j++) {
    cnt = 0;
    for (i = 0; i < size - 1; i++) {
      PetscCall(PetscRandomGetValueReal(rctx, &rand));
      parts[i] = (PetscInt)N * rand;
      parts[i] = PetscMin(parts[i], N - cnt);
      cnt += parts[i];
    }
    parts[size - 1] = N - cnt;

    PetscCall(PetscRandomGetValueReal(rctx, &rand));
    nblocks = rand * 10;
    nblocks = PetscMax(nblocks, 2);
    cnt     = 0;
    PetscCall(PetscMalloc1(nblocks, &blocksizes));
    for (i = 0; i < nblocks - 1; i++) {
      PetscCall(PetscRandomGetValueReal(rctx, &rand));
      blocksizes[i] = PetscMax(1, (PetscInt)N * rand);
      blocksizes[i] = PetscMin(blocksizes[i], N - cnt);
      cnt += blocksizes[i];
      if (cnt == N) {
        nblocks = i + 1;
        break;
      }
    }
    if (cnt < N) blocksizes[nblocks - 1] = N - cnt;

    PetscCall(BuildMatrix(parts, nblocks, blocksizes, &A));
    PetscCall(PetscFree(blocksizes));

    PetscCall(MatInvertVariableBlockEnvelope(A, MAT_INITIAL_MATRIX, &D));

    PetscCall(MatMatMult(A, D, MAT_INITIAL_MATRIX, 1.0, &F));
    PetscCall(MatIsDiagonal(F));

    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&F));
  }
  PetscCall(PetscFree(parts));
  PetscCall(PetscRandomDestroy(&rctx));

  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode MatIsDiagonal(Mat A)
{
  PetscInt           ncols, i, j, rstart, rend;
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscBool          founddiag;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    founddiag = PETSC_FALSE;
    PetscCall(MatGetRow(A, i, &ncols, &cols, &vals));
    for (j = 0; j < ncols; j++) {
      if (cols[j] == i) {
        PetscCheck(PetscAbsScalar(vals[j] - 1) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Row %" PetscInt_FMT " does not have 1 on the diagonal, it has %g", i, (double)PetscAbsScalar(vals[j]));
        founddiag = PETSC_TRUE;
      } else {
        PetscCheck(PetscAbsScalar(vals[j]) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Row %" PetscInt_FMT " has off-diagonal value %g at %" PetscInt_FMT "", i, (double)PetscAbsScalar(vals[j]), cols[j]);
      }
    }
    PetscCall(MatRestoreRow(A, i, &ncols, &cols, &vals));
    PetscCheck(founddiag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Row %" PetscInt_FMT " does not have diagonal entrie", i);
  }
  PetscFunctionReturn(0);
}

/*
    All processes receive all the block information
*/
PetscErrorCode BuildMatrix(const PetscInt *parts, PetscInt nblocks, const PetscInt *blocksizes, Mat *A)
{
  PetscInt    i, cnt = 0;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, parts[rank], parts[rank], PETSC_DETERMINE, PETSC_DETERMINE, 0, NULL, 0, NULL, A));
  PetscCall(MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  if (rank == 0) {
    for (i = 0; i < nblocks; i++) {
      PetscCall(MatSetValue(*A, cnt, cnt + blocksizes[i] - 1, 1.0, INSERT_VALUES));
      PetscCall(MatSetValue(*A, cnt + blocksizes[i] - 1, cnt, 1.0, INSERT_VALUES));
      cnt += blocksizes[i];
    }
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(*A, 10));
  PetscFunctionReturn(0);
}

/*TEST

   test:

   test:
     suffix: 2
     nsize: 2

   test:
     suffix: 5
     nsize: 5

TEST*/
