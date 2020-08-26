
#include <petscmat.h>

int main(int argc,char **argv)
{
   PetscErrorCode ierr;
   Mat            A, B;
   const char     *pfx;

   ierr = PetscInitialize(&argc, &argv, NULL, NULL);if (ierr) return ierr;
   ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
   ierr = MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
   ierr = MatSetUp(A);CHKERRQ(ierr);
   ierr = MatSetOptionsPrefix(A, "foo_");CHKERRQ(ierr);
   ierr = MatGetDiagonalBlock(A, &B);CHKERRQ(ierr);
   /* Test set options prefix with the string obtained from get options prefix */
   ierr = PetscObjectGetOptionsPrefix((PetscObject)A,&pfx);CHKERRQ(ierr);
   ierr = MatSetOptionsPrefix(B, pfx);CHKERRQ(ierr);
   ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
