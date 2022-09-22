
#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         A, B;
  const char *pfx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetOptionsPrefix(A, "foo_"));
  PetscCall(MatGetDiagonalBlock(A, &B));
  /* Test set options prefix with the string obtained from get options prefix */
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)A, &pfx));
  PetscCall(MatSetOptionsPrefix(B, pfx));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
