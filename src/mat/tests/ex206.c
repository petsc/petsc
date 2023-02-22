static char help[] = "Reads binary matrix - twice\n";

#include <petscmat.h>
int main(int argc, char **args)
{
  Mat         A;
  PetscViewer fd;                       /* viewer */
  char        file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "First MatLoad! \n"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-f2", file, sizeof(file), &flg));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Second MatLoad! \n"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: aij_1
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type aij -mat_block_size 1

   test:
      suffix: aij_2
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type aij -mat_block_size 1

   test:
      suffix: aij_2_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type aij -mat_block_size 1

   test:
      suffix: baij_1_2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -mat_block_size 2

   test:
      suffix: baij_2_1_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type baij -mat_block_size 1

   test:
      suffix: baij_2_2
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -mat_block_size 2

   test:
      suffix: baij_2_2_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type baij -mat_block_size 2

   test:
      suffix: sbaij_1_1
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type sbaij -mat_block_size 1

   test:
      suffix: sbaij_1_2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type sbaij -mat_block_size 2

   test:
      suffix: sbaij_2_1_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type sbaij -mat_block_size 1

   test:
      suffix: sbaij_2_2
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type sbaij -mat_block_size 2

TEST*/
