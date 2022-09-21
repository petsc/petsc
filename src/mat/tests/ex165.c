
static char help[] = "Tests C=A^T*B via MatTranspose() and MatMatMult(). \n\
                     Contributed by Alexander Grayver, Jan. 2012 \n\n";
/* Example:
  mpiexec -n <np> ./ex165 -fA A.dat -fB B.dat -view_C
 */

#include <petscmat.h>
int main(int argc, char **args)
{
  Mat         A, AT, B, C;
  PetscViewer viewer;
  PetscBool   flg;
  char        file[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fA", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Input fileA not specified");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-fB", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Input fileB not specified");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetType(B, MATDENSE));
  PetscCall(MatLoad(B, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &AT));
  PetscCall(MatMatMult(AT, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-view_C", &flg));
  if (flg) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "C.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
    PetscCall(MatView(C, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&AT));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}
