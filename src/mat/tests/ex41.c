
static char help[] = "Tests MatIncreaseOverlap() - the parallel case. This example\n\
is similar to ex40.c; here the index sets used are random. Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  PetscInt    nd = 2, ov = 1, i, j, m, n, *idx, lsize;
  PetscMPIInt rank;
  PetscBool   flg;
  Mat         A, B;
  char        file[PETSC_MAX_PATH_LEN];
  PetscViewer fd;
  IS         *is1, *is2;
  PetscRandom r;
  PetscScalar rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nd", &nd, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ov", &ov, NULL));

  /* Read matrix and RHS */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Read the matrix again as a seq matrix */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_SELF, &B));
  PetscCall(MatSetType(B, MATSEQAIJ));
  PetscCall(MatLoad(B, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Create the Random no generator */
  PetscCall(MatGetSize(A, &m, &n));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetFromOptions(r));

  /* Create the IS corresponding to subdomains */
  PetscCall(PetscMalloc1(nd, &is1));
  PetscCall(PetscMalloc1(nd, &is2));
  PetscCall(PetscMalloc1(m, &idx));

  /* Create the random Index Sets */
  for (i = 0; i < nd; i++) {
    for (j = 0; j < rank; j++) PetscCall(PetscRandomGetValue(r, &rand));
    PetscCall(PetscRandomGetValue(r, &rand));
    lsize = (PetscInt)(rand * m);
    for (j = 0; j < lsize; j++) {
      PetscCall(PetscRandomGetValue(r, &rand));
      idx[j] = (PetscInt)(rand * m);
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, lsize, idx, PETSC_COPY_VALUES, is1 + i));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, lsize, idx, PETSC_COPY_VALUES, is2 + i));
  }

  PetscCall(MatIncreaseOverlap(A, nd, is1, ov));
  PetscCall(MatIncreaseOverlap(B, nd, is2, ov));

  /* Now see if the serial and parallel case have the same answers */
  for (i = 0; i < nd; ++i) {
    PetscInt sz1, sz2;
    PetscCall(ISEqual(is1[i], is2[i], &flg));
    PetscCall(ISGetSize(is1[i], &sz1));
    PetscCall(ISGetSize(is2[i], &sz2));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "proc:[%d], i=%" PetscInt_FMT ", flg =%d  sz1 = %" PetscInt_FMT " sz2 = %" PetscInt_FMT, rank, i, (int)flg, sz1, sz2);
  }

  /* Free Allocated Memory */
  for (i = 0; i < nd; ++i) {
    PetscCall(ISDestroy(&is1[i]));
    PetscCall(ISDestroy(&is2[i]));
  }
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
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 3 -ov 1

TEST*/
