static char help[] = "Testing MatCreateMPIAIJWithSeqAIJ().\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat                mat, mat2;
  Mat                A, B;   // diag, offdiag of mat
  Mat                A2, B2; // diag, offdiag of mat2
  PetscInt           M, N, m = 5, n = 6, d_nz = 3, o_nz = 4;
  PetscInt          *bi, *bj, nd;
  const PetscScalar *ba;
  const PetscInt    *garray;
  PetscInt          *garray_h;
  PetscBool          equal, done;
  PetscMPIInt        size;
  char               mat_type[PETSC_MAX_PATH_LEN];
  PetscBool          flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size > 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 2 or more processes");

  // Create a MATMPIAIJ matrix for checking against
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, m, n, PETSC_DECIDE, PETSC_DECIDE, d_nz, NULL, o_nz, NULL, &mat));
  PetscCall(MatSetRandom(mat, NULL));
  PetscCall(MatGetSize(mat, &M, &N));
  PetscCall(MatGetLocalSize(mat, &m, &n));
  PetscCall(MatMPIAIJGetSeqAIJ(mat, &A, &B, &garray));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_type", mat_type, sizeof(mat_type), &flg));
  if (!flg) PetscCall(PetscStrncpy(mat_type, MATSEQAIJ, sizeof(mat_type))); // Default to MATAIJ
  PetscCall(MatConvert(A, mat_type, MAT_INITIAL_MATRIX, &A2));              // Copy A, B to A2, B2 but in the given mat_type
  PetscCall(MatConvert(B, mat_type, MAT_INITIAL_MATRIX, &B2));

  // The garray passed in has to be on the host, but it can be created on device and copied to the host.
  // We're just going to copy the existing host values here.
  PetscInt nonlocalCols;
  PetscCall(MatGetLocalSize(B, NULL, &nonlocalCols));
  PetscCall(PetscMalloc1(nonlocalCols, &garray_h));
  for (int i = 0; i < nonlocalCols; i++) garray_h[i] = garray[i];

  // 1) Test MatCreateMPIAIJWithSeqAIJ with compressed off-diag.
  PetscCall(MatCreateMPIAIJWithSeqAIJ(PETSC_COMM_WORLD, M, N, A2, B2, garray_h, &mat2));
  PetscCall(MatEqual(mat, mat2, &equal));                   // might directly compare value arrays
  if (equal) PetscCall(MatMultEqual(mat, mat2, 2, &equal)); // compare via MatMult()
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateMPIAIJWithSeqAIJ()");
  PetscCall(MatDestroy(&mat2)); // A2 and B2 are also destroyed

  // 2) Test MatCreateMPIAIJWithSeqAIJ with non-compressed off-diag.
  PetscCall(MatConvert(A, mat_type, MAT_INITIAL_MATRIX, &A2));

  // Create a version of B2 of size N with global indices
  PetscCall(MatGetRowIJ(B, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&bi, (const PetscInt **)&bj, &done));
  PetscCall(MatSeqAIJGetArrayRead(B, &ba));
  PetscCall(MatCreate(PETSC_COMM_SELF, &B2));
  PetscCall(MatSetSizes(B2, m, N, m, N));
  PetscCall(MatSetType(B2, mat_type));
  PetscCall(MatSeqAIJSetPreallocation(B2, o_nz, NULL));
  for (int i = 0; i < m; i++) { // Fill B2 with values from B
    for (int j = 0; j < bi[i + 1] - bi[i]; j++) PetscCall(MatSetValue(B2, i, garray[bj[bi[i] + j]], ba[bi[i] + j], INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(B2, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B2, MAT_FINAL_ASSEMBLY));

  PetscCall(MatRestoreRowIJ(B, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&bi, (const PetscInt **)&bj, &done));
  PetscCall(MatSeqAIJRestoreArrayRead(B, &ba));

  PetscCall(MatCreateMPIAIJWithSeqAIJ(PETSC_COMM_WORLD, M, N, A2, B2, NULL, &mat2));
  PetscCall(MatEqual(mat, mat2, &equal));
  if (equal) PetscCall(MatMultEqual(mat, mat2, 2, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateMPIAIJWithSeqAIJ()");

  PetscCall(MatDestroy(&mat));
  PetscCall(MatDestroy(&mat2));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: 2
    output_file: output/empty.out

    test:
      suffix: 1
      args: -mat_type aij

    test:
      suffix: 2
      args: -mat_type aijkokkos
      requires: kokkos_kernels

    test:
      suffix: 3
      args: -mat_type aijcusparse
      requires: cuda

    test:
      suffix: 4
      args: -mat_type aijhipsparse
      requires: hip

TEST*/
