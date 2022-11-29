static char help[] = "Tests MatMultHermitianTranspose() for real numbers.\n\n";
#include <petsc.h>

int main(int argc, char **args)
{
  Mat         A, AHT;
  Vec         x, y;
  PetscRandom rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 10, 10));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatSetValue(A, 0, 0, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateHermitianTranspose(A, &AHT));
  PetscCall(MatCreateVecs(AHT, &x, &y));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(VecSetRandom(y, rand));
  PetscCall(PetscRandomDestroy(&rand));

  PetscCall(MatMultHermitianTranspose(AHT, y, x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&AHT));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
