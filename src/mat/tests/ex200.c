
#include <petscmat.h>

int main(int argc,char **argv)
{
   PetscErrorCode ierr;
   Mat            A, B;
   const char     *pfx;

   ierr = PetscInitialize(&argc, &argv, NULL, NULL);if (ierr) return ierr;
   CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
   CHKERRQ(MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE));
   CHKERRQ(MatSetUp(A));
   CHKERRQ(MatSetOptionsPrefix(A, "foo_"));
   CHKERRQ(MatGetDiagonalBlock(A, &B));
   /* Test set options prefix with the string obtained from get options prefix */
   CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)A,&pfx));
   CHKERRQ(MatSetOptionsPrefix(B, pfx));
   CHKERRQ(MatDestroy(&A));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
