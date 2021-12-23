static char help[] = "Tests MatMultHermitianTranspose() for real numbers.\n\n";
#include <petsc.h>

int main(int argc, char **args)
{
  Mat            A, AHT;
  Vec            x, y;
  PetscRandom    rand;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args, (char*)0, help); if (ierr) return ierr;

  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 10, 10);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatSetValue(A, 0, 0, 1.0, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateHermitianTranspose(A, &AHT);CHKERRQ(ierr);
  ierr = MatCreateVecs(AHT, &x, &y);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = VecSetRandom(y, rand);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);

  ierr = MatMultHermitianTranspose(AHT, y, x);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&AHT);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
