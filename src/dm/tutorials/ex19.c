/*
     Demonstrates creating domain decomposition DAs and how to shuffle around data between the two
 */

#include <petscdm.h>
#include <petscdmda.h>

static char help[] = "Test for DMDA with overlap.\n\n";

int main(int argc, char **argv)
{
  DM da;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  /* Build of the DMDA -- 1D -- boundary_none */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "1D -- DM_BOUNDARY_NONE\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 1));
  PetscCall(DMDASetSizes(da, 8, 1, 1));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  PetscCall(DMDASetDof(da, 1));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "n1d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 1D -- boundary_ghosted */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "1D -- DM_BOUNDARY_GHOSTED\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 1));
  PetscCall(DMDASetSizes(da, 8, 1, 1));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "g1d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 1D -- boundary_periodic */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "1D -- DM_BOUNDARY_PERIODIC\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 1));
  PetscCall(DMDASetSizes(da, 8, 1, 1));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "p1d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 2D -- boundary_none */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D -- DM_BOUNDARY_NONE\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 2));
  PetscCall(DMDASetSizes(da, 8, 8, 1));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "n2d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 2D -- boundary_ghosted */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D -- DM_BOUNDARY_GHOSTED\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 2));
  PetscCall(DMDASetSizes(da, 8, 8, 1));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "g2d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 2D -- boundary_periodic */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D -- DM_BOUNDARY_PERIODIC\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 2));
  PetscCall(DMDASetSizes(da, 8, 8, 1));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "p2d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 3D -- boundary_none */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "3D -- DM_BOUNDARY_NONE\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 3));
  PetscCall(DMDASetSizes(da, 8, 8, 8));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "n3d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 3D -- boundary_ghosted */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "3D -- DM_BOUNDARY_GHOSTED\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 3));
  PetscCall(DMDASetSizes(da, 8, 8, 8));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "g3d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* Build of the DMDA -- 3D -- boundary_periodic */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "3D -- DM_BOUNDARY_PERIODIC\n"));
  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 3));
  PetscCall(DMDASetSizes(da, 8, 8, 8));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC));
  PetscCall(DMDASetDof(da, 2));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOverlap(da, 1, 1, 1));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetOptionsPrefix(da, "p3d_"));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  /* test moving data in and out */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
