
static char help[] = "Tests MatIncreaseOverlap() and MatCreateSubmatrices() for the parallel case.\n\
This example is similar to ex40.c; here the index sets used are random.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  PetscInt    nd = 2, ov = 1, i, j, lsize, m, n, *idx, bs;
  PetscMPIInt rank, size;
  PetscBool   flg;
  Mat         A, B, *submatA, *submatB;
  char        file[PETSC_MAX_PATH_LEN];
  PetscViewer fd;
  IS         *is1, *is2;
  PetscRandom r;
  PetscBool   test_unsorted = PETSC_FALSE;
  PetscScalar rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nd", &nd, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ov", &ov, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_unsorted", &test_unsorted, NULL));

  /* Read matrix A and RHS */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Read the same matrix as a seq matrix B */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_SELF, &B));
  PetscCall(MatSetType(B, MATSEQAIJ));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLoad(B, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatGetBlockSize(A, &bs));

  /* Create the Random no generator */
  PetscCall(MatGetSize(A, &m, &n));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetFromOptions(r));

  /* Create the IS corresponding to subdomains */
  PetscCall(PetscMalloc1(nd, &is1));
  PetscCall(PetscMalloc1(nd, &is2));
  PetscCall(PetscMalloc1(m, &idx));
  for (i = 0; i < m; i++) idx[i] = i;

  /* Create the random Index Sets */
  for (i = 0; i < nd; i++) {
    /* Skip a few,so that the IS on different procs are different*/
    for (j = 0; j < rank; j++) PetscCall(PetscRandomGetValue(r, &rand));
    PetscCall(PetscRandomGetValue(r, &rand));
    lsize = (PetscInt)(rand * (m / bs));
    /* shuffle */
    for (j = 0; j < lsize; j++) {
      PetscInt k, swap, l;

      PetscCall(PetscRandomGetValue(r, &rand));
      k = j + (PetscInt)(rand * ((m / bs) - j));
      for (l = 0; l < bs; l++) {
        swap            = idx[bs * j + l];
        idx[bs * j + l] = idx[bs * k + l];
        idx[bs * k + l] = swap;
      }
    }
    if (!test_unsorted) PetscCall(PetscSortInt(lsize * bs, idx));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, lsize * bs, idx, PETSC_COPY_VALUES, is1 + i));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, lsize * bs, idx, PETSC_COPY_VALUES, is2 + i));
    PetscCall(ISSetBlockSize(is1[i], bs));
    PetscCall(ISSetBlockSize(is2[i], bs));
  }

  if (!test_unsorted) {
    PetscCall(MatIncreaseOverlap(A, nd, is1, ov));
    PetscCall(MatIncreaseOverlap(B, nd, is2, ov));

    for (i = 0; i < nd; ++i) {
      PetscCall(ISSort(is1[i]));
      PetscCall(ISSort(is2[i]));
    }
  }

  PetscCall(MatCreateSubMatrices(A, nd, is1, is1, MAT_INITIAL_MATRIX, &submatA));
  PetscCall(MatCreateSubMatrices(B, nd, is2, is2, MAT_INITIAL_MATRIX, &submatB));

  /* Now see if the serial and parallel case have the same answers */
  for (i = 0; i < nd; ++i) {
    PetscCall(MatEqual(submatA[i], submatB[i], &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%" PetscInt_FMT "-th parallel submatA != seq submatB", i);
  }

  /* Free Allocated Memory */
  for (i = 0; i < nd; ++i) {
    PetscCall(ISDestroy(&is1[i]));
    PetscCall(ISDestroy(&is2[i]));
  }
  PetscCall(MatDestroySubMatrices(nd, &submatA));
  PetscCall(MatDestroySubMatrices(nd, &submatB));

  PetscCall(PetscRandomDestroy(&r));
  PetscCall(PetscFree(is1));
  PetscCall(PetscFree(is2));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFree(idx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 5 -ov 2

   test:
      suffix: 2
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 8 -ov 2
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex

   test:
      suffix: unsorted_baij_mpi
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/cfd.1.10 -nd 8 -mat_type baij -test_unsorted

   test:
      suffix: unsorted_baij_seq
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/cfd.1.10 -nd 8 -mat_type baij -test_unsorted

   test:
      suffix: unsorted_mpi
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 8 -test_unsorted

   test:
      suffix: unsorted_seq
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 8 -test_unsorted

TEST*/
