static char help[] = "Tests MatMultHermitianTranspose() for real numbers.\n\n";
#include <petsc.h>

int main(int argc, char **args)
{
  Mat            A, AHT;
  Vec            x, y;
  PetscRandom    rand;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args, (char*)0, help); if (ierr) return ierr;

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetType(A, MATAIJ));
  CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 10, 10));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatSetValue(A, 0, 0, 1.0, INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateHermitianTranspose(A, &AHT));
  CHKERRQ(MatCreateVecs(AHT, &x, &y));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(VecSetRandom(y, rand));
  CHKERRQ(PetscRandomDestroy(&rand));

  CHKERRQ(MatMultHermitianTranspose(AHT, y, x));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&AHT));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
