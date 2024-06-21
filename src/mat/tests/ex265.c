static char help[] = "Tests inserting new block into SBAIJ and BAIJ matrix \n ";

#include <petscdmda.h>

int main(int argc, char **argv)
{
  DM          dm;
  Mat         A;
  PetscInt    idm = 0, idn = 8;
  PetscScalar v[] = {1, 2, 3, 4};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetValuesBlocked(A, 1, &idm, 1, &idn, v, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -dm_mat_type {{aij baij sbaij}separate output} -mat_view

   test:
     suffix: 2
     nsize: 2
     args: -dm_mat_type {{aij baij sbaij}separate output} -mat_view

   test:
     suffix: 3
     nsize: 3
     args: -dm_mat_type {{aij baij sbaij}separate output} -mat_view

TEST*/
