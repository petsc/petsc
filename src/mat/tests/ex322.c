static char help[] = "Tests MatCopy() from MATDENSE into a matrix of another type.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat       A, B;
  char      type[256] = MATAIJ;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-type", type, sizeof(type), NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 6, 6, NULL, &A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatConvert(A, type, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatZeroEntries(B));
  PetscCall(MatCopy(A, B, SAME_NONZERO_PATTERN));
  PetscCall(MatMultEqual(A, B, 3, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatCopy() produced incorrect values");
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      args: -type aij
      output_file: output/empty.out

TEST*/
