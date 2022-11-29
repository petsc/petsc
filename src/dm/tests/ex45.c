/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

  Detected bug in DMCreateMatrix() for skinny domains with periodic boundary conditions in overestimating nonzero preallocation

  Creation Date : 08-12-2016

  Last Modified : Thu 08 Dec 2016 10:46:02 AM CET

  Created By : Davide Monsorno

_._._._._._._._._._._._._._._._._._._._._.*/

#include <petscdmda.h>

int main(int argc, char *argv[])
{
  PetscInt nx = 2;
  PetscInt ny = 2;
  PetscInt nz = 128;
  DM       da;
  Mat      A;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMCreateMatrix(da, &A));

  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 5

TEST*/
