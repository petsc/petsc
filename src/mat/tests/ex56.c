
static char help[] = "Test the use of MatSetValuesBlocked(), MatZeroRows() for rectangular MatBAIJ matrix, test MatSetValuesBlocked() for MatSBAIJ matrix (-test_mat_sbaij).";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A;
  PetscInt    bs = 3, m = 4, n = 6, i, j, val = 10, row[2], col[3], eval, rstart;
  PetscMPIInt size, rank;
  PetscScalar x[6][9], y[3][3], one = 1.0;
  PetscBool   flg, testsbaij = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-test_mat_sbaij", &testsbaij));

  if (testsbaij) {
    PetscCall(MatCreateSBAIJ(PETSC_COMM_WORLD, bs, m * bs, n * bs, PETSC_DECIDE, PETSC_DECIDE, 1, NULL, 1, NULL, &A));
  } else {
    PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD, bs, m * bs, n * bs, PETSC_DECIDE, PETSC_DECIDE, 1, NULL, 1, NULL, &A));
  }
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  eval = 9;

  PetscCall(PetscOptionsHasName(NULL, NULL, "-ass_extern", &flg));
  if (flg && (size != 1)) rstart = m * ((rank + 1) % size);
  else rstart = m * (rank);

  row[0] = rstart + 0;
  row[1] = rstart + 2;
  col[0] = rstart + 0;
  col[1] = rstart + 1;
  col[2] = rstart + 3;
  for (i = 0; i < 6; i++) {
    for (j = 0; j < 9; j++) x[i][j] = (PetscScalar)val++;
  }

  PetscCall(MatSetValuesBlocked(A, 2, row, 3, col, &x[0][0], INSERT_VALUES));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*
  This option does not work for rectangular matrices
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  */

  PetscCall(MatSetValuesBlocked(A, 2, row, 3, col, &x[0][0], INSERT_VALUES));

  /* Do another MatSetValues to test the case when only one local block is specified */
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) y[i][j] = (PetscScalar)(10 + i * eval + j);
  }
  PetscCall(MatSetValuesBlocked(A, 1, row, 1, col, &y[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-zero_rows", &flg));
  if (flg) {
    col[0] = rstart * bs + 0;
    col[1] = rstart * bs + 1;
    col[2] = rstart * bs + 2;
    PetscCall(MatZeroRows(A, 3, col, one, 0, 0));
  }

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      filter: grep -v " MPI process"

   test:
      suffix: 4
      nsize: 3
      args: -ass_extern
      filter: grep -v " MPI process"

   test:
      suffix: 5
      nsize: 3
      args: -ass_extern -zero_rows
      filter: grep -v " MPI process"

TEST*/
