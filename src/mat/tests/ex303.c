static char help[] = "Testing MatCreateMPIAIJWithSeqAIJ().\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat             A, B;
  PetscInt        i, j, column, M, N, m, n;
  PetscInt       *oi, *oj, nd;
  const PetscInt *garray;
  PetscInt       *garray_h;
  PetscScalar     value;
  PetscScalar    *oa;
  PetscRandom     rctx;
  PetscBool       equal, done;
  Mat             AA, AB;
  PetscMPIInt     size, rank;
  MatType         mat_type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size > 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 2 or more processes");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create a mpiaij matrix for checking */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 0, NULL, 0, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetUp(A));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));

  for (i = 5 * rank; i < 5 * rank + 5; i++) {
    for (j = 0; j < 5 * size; j++) {
      PetscCall(PetscRandomGetValue(rctx, &value));
      column = (PetscInt)(5 * size * PetscRealPart(value));
      PetscCall(PetscRandomGetValue(rctx, &value));
      PetscCall(MatSetValues(A, 1, &i, 1, &column, &value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));

  PetscCall(MatMPIAIJGetSeqAIJ(A, &AA, &AB, &garray));

  Mat output_mat_local, output_mat_nonlocal, output_mat_local_copy, output_mat_nonlocal_copy;

  PetscCall(MatConvert(AA, MATSAME, MAT_INITIAL_MATRIX, &output_mat_local));
  PetscCall(MatConvert(AB, MATSAME, MAT_INITIAL_MATRIX, &output_mat_nonlocal));
  PetscCall(MatConvert(AA, MATSAME, MAT_INITIAL_MATRIX, &output_mat_local_copy));

  // The garray passed in has to be on the host, but it can be created
  // on device and copied to the host
  // We're just going to copy the existing host values here
  PetscInt nonlocalCols;
  PetscCall(MatGetLocalSize(AB, NULL, &nonlocalCols));
  PetscCall(PetscMalloc1(nonlocalCols, &garray_h));
  for (int i = 0; i < nonlocalCols; i++) { garray_h[i] = garray[i]; }

  // Build our MPI matrix
  // If we provide garray and output_mat_nonlocal with local indices and the compactified size
  // it doesn't compactify
  PetscCall(MatCreateMPIAIJWithSeqAIJ(PETSC_COMM_WORLD, M, N, output_mat_local, output_mat_nonlocal, garray_h, &B));

  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateMPIAIJWithSeqAIJ()");
  PetscCall(MatDestroy(&B));

  // ~~~~~~~~~~~~~~~~~
  // Test MatCreateMPIAIJWithSeqAIJ with compactification
  // This is just for testing - would be silly to do this in practice
  // ~~~~~~~~~~~~~~~~~
  garray_h = NULL;
  PetscCall(MatGetRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&oi, (const PetscInt **)&oj, &done));
  PetscCall(MatSeqAIJGetArray(AB, &oa));

  // Create a version of AB of size N with global indices
  PetscCall(MatGetType(AB, &mat_type));
  PetscCall(MatCreate(PETSC_COMM_SELF, &output_mat_nonlocal_copy));
  PetscCall(MatSetSizes(output_mat_nonlocal_copy, m, N, m, N));
  PetscCall(MatSetType(output_mat_nonlocal_copy, mat_type));
  PetscCall(MatSeqAIJSetPreallocation(output_mat_nonlocal_copy, oi[5], NULL));

  // Fill the matrix
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < oi[i + 1] - oi[i]; j++) { PetscCall(MatSetValue(output_mat_nonlocal_copy, i, garray[oj[oi[i] + j]], oa[oi[i] + j], INSERT_VALUES)); }
  }
  PetscCall(MatAssemblyBegin(output_mat_nonlocal_copy, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(output_mat_nonlocal_copy, MAT_FINAL_ASSEMBLY));

  PetscCall(MatRestoreRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&oi, (const PetscInt **)&oj, &done));
  PetscCall(MatSeqAIJRestoreArray(AB, &oa));

  // Build our MPI matrix
  // If we don't provide garray and output_mat_local_copy with global indices and size N
  // it will do compactification
  PetscCall(MatCreateMPIAIJWithSeqAIJ(PETSC_COMM_WORLD, M, N, output_mat_local_copy, output_mat_nonlocal_copy, garray_h, &B));

  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateMPIAIJWithSeqAIJ()");
  PetscCall(MatDestroy(&B));

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /* Free spaces */
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    args: -mat_type aij
    output_file: output/empty.out

TEST*/
