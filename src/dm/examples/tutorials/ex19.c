/*
     Demonstrates creating domain decomposition DAs and how to shuffle around data between the two
 */

#include <math.h>
#include <petscdmda.h>

static char help[] = "Test for DMDA with overlap.\n\n";

int main(int argc,char **argv) 
{
  PetscErrorCode ierr;
  DM             da;
  /* Initialize the Petsc context */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /* Build of the DMDA -- 1D -- boundary_none */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"1D -- DMDA_BOUNDARY_NONE\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 1);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, 1, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 1);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"n1d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* Build of the DMDA -- 1D -- boundary_ghosted */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"1D -- DMDA_BOUNDARY_GHOSTED\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 1);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, 1, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"g1d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* Build of the DMDA -- 1D -- boundary_periodic */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"1D -- DMDA_BOUNDARY_PERIODIC\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 1);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, 1, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"p1d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* Build of the DMDA -- 2D -- boundary_none */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"2D -- DMDA_BOUNDARY_NONE\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 2);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, -8, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"n2d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* Build of the DMDA -- 2D -- boundary_ghosted */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"2D -- DMDA_BOUNDARY_GHOSTED\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 2);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, -8, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"g2d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);


  /* Build of the DMDA -- 2D -- boundary_periodic */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"2D -- DMDA_BOUNDARY_PERIODIC\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 2);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, -8, 1);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"p2d_");CHKERRQ(ierr);
ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* Build of the DMDA -- 3D -- boundary_none */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"3D -- DMDA_BOUNDARY_NONE\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 3);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, -8, -8);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"n3d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* Build of the DMDA -- 3D -- boundary_ghosted */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"3D -- DMDA_BOUNDARY_GHOSTED\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 3);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, -8, -8);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"g3d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);


  /* Build of the DMDA -- 3D -- boundary_periodic */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"3D -- DMDA_BOUNDARY_PERIODIC\n");CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_WORLD, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, 3);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -8, -8, -8);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 2);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOverlap(da,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da,"p3d_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* test moving data in and out */
  ierr = PetscFinalize();
  return 0;
}
