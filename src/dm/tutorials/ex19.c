/*
     Demonstrates creating domain decomposition DAs and how to shuffle around data between the two
 */

#include <petscdm.h>
#include <petscdmda.h>

static char help[] = "Test for DMDA with overlap.\n\n";

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* Build of the DMDA -- 1D -- boundary_none */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"1D -- DM_BOUNDARY_NONE\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 1));
  CHKERRQ(DMDASetSizes(da, 8, 1, 1));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  CHKERRQ(DMDASetDof(da, 1));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"n1d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 1D -- boundary_ghosted */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"1D -- DM_BOUNDARY_GHOSTED\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 1));
  CHKERRQ(DMDASetSizes(da, 8, 1, 1));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"g1d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 1D -- boundary_periodic */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"1D -- DM_BOUNDARY_PERIODIC\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 1));
  CHKERRQ(DMDASetSizes(da, 8, 1, 1));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"p1d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 2D -- boundary_none */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"2D -- DM_BOUNDARY_NONE\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 2));
  CHKERRQ(DMDASetSizes(da, 8, 8, 1));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"n2d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 2D -- boundary_ghosted */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"2D -- DM_BOUNDARY_GHOSTED\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 2));
  CHKERRQ(DMDASetSizes(da, 8, 8, 1));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"g2d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 2D -- boundary_periodic */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"2D -- DM_BOUNDARY_PERIODIC\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 2));
  CHKERRQ(DMDASetSizes(da, 8, 8, 1));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"p2d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 3D -- boundary_none */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"3D -- DM_BOUNDARY_NONE\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 3));
  CHKERRQ(DMDASetSizes(da, 8, 8, 8));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"n3d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 3D -- boundary_ghosted */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"3D -- DM_BOUNDARY_GHOSTED\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 3));
  CHKERRQ(DMDASetSizes(da, 8, 8, 8));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"g3d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* Build of the DMDA -- 3D -- boundary_periodic */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"3D -- DM_BOUNDARY_PERIODIC\n"));
  CHKERRQ(DMDACreate(PETSC_COMM_WORLD, &da));
  CHKERRQ(DMSetDimension(da, 3));
  CHKERRQ(DMDASetSizes(da, 8, 8, 8));
  CHKERRQ(DMDASetBoundaryType(da, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC));
  CHKERRQ(DMDASetDof(da, 2));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMDASetOverlap(da,1,1,1));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetOptionsPrefix(da,"p3d_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  /* test moving data in and out */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
