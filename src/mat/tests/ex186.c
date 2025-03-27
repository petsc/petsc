#include <petscmat.h>

static char help[] = "Example of MatMat ops with MatDense in PETSc.\n";

int main(int argc, char **args)
{
  Mat         A, P, PtAP, RARt, ABC, Pt;
  PetscInt    n = 4, m = 2; // Example dimensions
  PetscMPIInt size;

  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example is for sequential runs only.");

  // Create dense matrix P (n x m)
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, m, NULL, &P));
  PetscScalar P_values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  PetscInt    rows[]     = {0, 1, 2, 3};
  PetscInt    cols[]     = {0, 1};
  PetscCall(MatSetValues(P, n, rows, m, cols, P_values, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatTranspose(P, MAT_INITIAL_MATRIX, &Pt));

  // Create dense matrix A (n x n)
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, n, NULL, &A));
  PetscCall(MatSetBlockSize(A, n)); // Set block size for A
  PetscScalar A_values[] = {4.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 2.0, 0.0, 5.0, 2.0, 0.0, 1.0, 2.0, 6.0};
  PetscInt    indices[]  = {0, 1, 2, 3};
  PetscCall(MatSetValues(A, n, indices, n, indices, A_values, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PtAP));
  PetscCall(MatRARt(A, Pt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RARt));
  PetscCall(MatMatMatMult(Pt, A, P, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &ABC));

  // View matrices
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Matrix P:\n"));
  PetscCall(MatView(P, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Matrix A:\n"));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Matrix PtAP:\n"));
  PetscCall(MatView(PtAP, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Matrix RARt:\n"));
  PetscCall(MatView(RARt, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Matrix ABC:\n"));
  PetscCall(MatView(ABC, PETSC_VIEWER_STDOUT_SELF));

  // Clean up
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&Pt));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&PtAP));
  PetscCall(MatDestroy(&RARt));
  PetscCall(MatDestroy(&ABC));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    diff_args: -j
    suffix: 1

TEST*/
