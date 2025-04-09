static char help[] = "Test matmat products with matdiagonal on gpus \n\n";

// Contributed by: Steven Dargaville

#include <petscmat.h>

int main(int argc, char **args)
{
  const PetscInt inds[]  = {0, 1};
  PetscScalar    avals[] = {2, 3, 5, 7};
  Mat            A, B_diag, B_aij_diag, result, result_diag;
  Vec            diag;
  PetscBool      equal = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  // Create matrix to start
  PetscCall(MatCreateFromOptions(PETSC_COMM_WORLD, NULL, 1, 2, 2, 2, 2, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValues(A, 2, inds, 2, inds, avals, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Create a matdiagonal matrix
  // Will be the matching vec type as A
  PetscCall(MatCreateVecs(A, &diag, NULL));
  PetscCall(VecSet(diag, 2.0));
  PetscCall(MatCreateDiagonal(diag, &B_diag));

  // Create the same matrix as the matdiagonal but in aij format
  PetscCall(MatCreateFromOptions(PETSC_COMM_WORLD, NULL, 1, 2, 2, 2, 2, &B_aij_diag));
  PetscCall(MatSetUp(B_aij_diag));
  PetscCall(MatDiagonalSet(B_aij_diag, diag, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B_aij_diag, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B_aij_diag, MAT_FINAL_ASSEMBLY));
  PetscCall(VecDestroy(&diag));

  // ~~~~~~~~~~~~~
  // Do an initial matmatmult
  // A * B_aij_diag
  // and then
  // A * B_diag but just using MatDiagonalScale
  // ~~~~~~~~~~~~~

  // aij * aij
  PetscCall(MatMatMult(A, B_aij_diag, MAT_INITIAL_MATRIX, 1.5, &result));
  // PetscCall(MatView(result, PETSC_VIEWER_STDOUT_WORLD));

  // aij * diagonal
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &result_diag));
  PetscCall(MatDiagonalGetDiagonal(B_diag, &diag));
  PetscCall(MatDiagonalScale(result_diag, NULL, diag));
  PetscCall(MatDiagonalRestoreDiagonal(B_diag, &diag));
  // PetscCall(MatView(result_diag, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatEqual(result, result_diag, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMatMult and MatDiagonalScale do not give the same result");

  // ~~~~~~~~~~~~~
  // Now let's modify the diagonal and do it again with "reuse"
  // ~~~~~~~~~~~~~
  PetscCall(MatDiagonalGetDiagonal(B_diag, &diag));
  PetscCall(VecSet(diag, 3.0));
  PetscCall(MatDiagonalSet(B_aij_diag, diag, INSERT_VALUES));
  PetscCall(MatDiagonalRestoreDiagonal(B_diag, &diag));

  // aij * aij
  PetscCall(MatMatMult(A, B_aij_diag, MAT_REUSE_MATRIX, 1.5, &result));

  // aij * diagonal
  PetscCall(MatCopy(A, result_diag, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalGetDiagonal(B_diag, &diag));
  PetscCall(MatDiagonalScale(result_diag, NULL, diag));
  PetscCall(MatDiagonalRestoreDiagonal(B_diag, &diag));

  PetscCall(MatEqual(result, result_diag, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMatMult and MatDiagonalScale do not give the same result");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B_diag));
  PetscCall(MatDestroy(&B_aij_diag));
  PetscCall(MatDestroy(&result));
  PetscCall(MatDestroy(&result_diag));
  PetscCall(VecDestroy(&diag));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    requires: kokkos_kernels
    args: -mat_type aijkokkos
    output_file: output/empty.out
TEST*/
