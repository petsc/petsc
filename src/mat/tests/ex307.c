static char help[] = "Tests MatConvert() from MATAIJ to MATSELL across multiple processes.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A, B;
  PetscViewer viewer;
  PetscBool   flg;
  char        file[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatConvert(A, MATSELL, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatMultEqual(A, B, 10, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMult() differs between MATAIJ and converted MATSELL");
  PetscCall(MatMultAddEqual(A, B, 10, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMultAdd() differs between MATAIJ and converted MATSELL");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !defined(PETSC_USE_64BIT_INDICES) double

   test:
      nsize: 2
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64
      output_file: output/empty.out

TEST*/
