
static char help[] = "Tests MatLoad() MatView() for MPIBAIJ.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A, B;
  char        file[PETSC_MAX_PATH_LEN];
  PetscBool   flg;
  PetscViewer fd;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");

  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));

  /*
     Load the matrix; then destroy the viewer.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /*
     Open another binary file.  Note that we use FILE_MODE_WRITE to indicate writing to the file
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "fileoutput", FILE_MODE_WRITE, &fd));
  PetscCall(PetscViewerBinarySetFlowControl(fd, 3));
  /*
     Save the matrix and vector; then destroy the viewer.
  */
  PetscCall(MatView(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* load the new matrix */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "fileoutput", FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLoad(B, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatEqual(A, B, &flg));
  if (flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Matrices are equal\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Matrices are not equal\n"));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 2
      nsize: 5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 3
      nsize: 7
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 4
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

   test:
      suffix: 5
      nsize: 5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

   test:
      suffix: 6
      nsize: 7
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

TEST*/
