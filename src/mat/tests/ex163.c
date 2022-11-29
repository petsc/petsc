
static char help[] = "Tests MatTransposeMatMult() on MatLoad() matrix \n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A, C, Bdense, Cdense;
  PetscViewer  fd;                       /* viewer */
  char         file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool    flg, viewmats = PETSC_FALSE;
  PetscMPIInt  rank, size;
  PetscReal    fill = 1.0;
  PetscInt     m, n, i, j, BN = 10, rstart, rend, *rows, *cols;
  PetscScalar *Barray, *Carray, rval, *array;
  Vec          x, y;
  PetscRandom  rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Determine file from which we read the matrix A */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");

  /* Load matrix A */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Print (for testing only) */
  PetscCall(PetscOptionsHasName(NULL, NULL, "-view_mats", &viewmats));
  if (viewmats) {
    if (rank == 0) printf("A_aij:\n");
    PetscCall(MatView(A, 0));
  }

  /* Test MatTransposeMatMult_aij_aij() */
  PetscCall(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, fill, &C));
  if (viewmats) {
    if (rank == 0) printf("\nC = A_aij^T * A_aij:\n");
    PetscCall(MatView(C, 0));
  }
  PetscCall(MatDestroy(&C));
  PetscCall(MatGetLocalSize(A, &m, &n));

  /* create a dense matrix Bdense */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Bdense));
  PetscCall(MatSetSizes(Bdense, m, PETSC_DECIDE, PETSC_DECIDE, BN));
  PetscCall(MatSetType(Bdense, MATDENSE));
  PetscCall(MatSetFromOptions(Bdense));
  PetscCall(MatSetUp(Bdense));
  PetscCall(MatGetOwnershipRange(Bdense, &rstart, &rend));

  PetscCall(PetscMalloc3(m, &rows, BN, &cols, m * BN, &array));
  for (i = 0; i < m; i++) rows[i] = rstart + i;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  for (j = 0; j < BN; j++) {
    cols[j] = j;
    for (i = 0; i < m; i++) {
      PetscCall(PetscRandomGetValue(rand, &rval));
      array[m * j + i] = rval;
    }
  }
  PetscCall(MatSetValues(Bdense, m, rows, BN, cols, array, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Bdense, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Bdense, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFree3(rows, cols, array));
  if (viewmats) {
    if (rank == 0) printf("\nBdense:\n");
    PetscCall(MatView(Bdense, 0));
  }

  /* Test MatTransposeMatMult_aij_dense() */
  PetscCall(MatTransposeMatMult(A, Bdense, MAT_INITIAL_MATRIX, fill, &C));
  PetscCall(MatTransposeMatMult(A, Bdense, MAT_REUSE_MATRIX, fill, &C));
  if (viewmats) {
    if (rank == 0) printf("\nC=A^T*Bdense:\n");
    PetscCall(MatView(C, 0));
  }

  /* Check accuracy */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Cdense));
  PetscCall(MatSetSizes(Cdense, n, PETSC_DECIDE, PETSC_DECIDE, BN));
  PetscCall(MatSetType(Cdense, MATDENSE));
  PetscCall(MatSetFromOptions(Cdense));
  PetscCall(MatSetUp(Cdense));
  PetscCall(MatAssemblyBegin(Cdense, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Cdense, MAT_FINAL_ASSEMBLY));

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (size == 1) {
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m, NULL, &x));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, NULL, &y));
  } else {
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, m, PETSC_DECIDE, NULL, &x));
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, NULL, &y));
  }

  /* Cdense[:,j] = A^T * Bdense[:,j] */
  PetscCall(MatDenseGetArray(Bdense, &Barray));
  PetscCall(MatDenseGetArray(Cdense, &Carray));
  for (j = 0; j < BN; j++) {
    PetscCall(VecPlaceArray(x, Barray));
    PetscCall(VecPlaceArray(y, Carray));

    PetscCall(MatMultTranspose(A, x, y));

    PetscCall(VecResetArray(x));
    PetscCall(VecResetArray(y));
    Barray += m;
    Carray += n;
  }
  PetscCall(MatDenseRestoreArray(Bdense, &Barray));
  PetscCall(MatDenseRestoreArray(Cdense, &Carray));
  if (viewmats) {
    if (rank == 0) printf("\nCdense:\n");
    PetscCall(MatView(Cdense, 0));
  }

  PetscCall(MatEqual(C, Cdense, &flg));
  if (!flg) {
    if (rank == 0) printf(" C != Cdense\n");
  }

  /* Free data structures */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&Bdense));
  PetscCall(MatDestroy(&Cdense));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex163.out

   test:
      suffix: 2
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex163.out

TEST*/
