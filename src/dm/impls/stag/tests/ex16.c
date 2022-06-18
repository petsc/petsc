static char help[] = "Test DMStag refinement and coarsening\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM              dm,dmCoarsened,dmRefined;
  PetscInt        dim;
  PetscBool       flg;

  /* Create a DMStag object */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option\n"));
    return 1;
  }
  if (dim == 1) {
    PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,2,3,DMSTAG_STENCIL_BOX,1,NULL,&dm));
  } else if (dim == 2) {
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,6,PETSC_DECIDE,PETSC_DECIDE,2,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
  } else if (dim == 3) {
    PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,6,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option with value 1, 2, or 3\n"));
    return 1;
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMView(dm,PETSC_VIEWER_STDOUT_WORLD));

  /* Create a refined DMStag object */
  PetscCall(DMRefine(dm,PetscObjectComm((PetscObject)dm),&dmRefined));
  PetscCall(DMView(dmRefined,PETSC_VIEWER_STDOUT_WORLD));

  /* Create a coarsened DMStag object */
  PetscCall(DMCoarsen(dm,PetscObjectComm((PetscObject)dm),&dmCoarsened));
  PetscCall(DMView(dmCoarsened,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMDestroy(&dmCoarsened));
  PetscCall(DMDestroy(&dmRefined));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -dim 1 -stag_grid_x 2

   test:
      suffix: 2
      nsize: 1
      args: -dim 2 -stag_grid_x 6 -stag_grid_y 4

   test:
      suffix: 3
      nsize: 6
      args: -dim 3 -stag_grid_x 6 -stag_grid_y 4 -stag_grid_z 4

   test:
      suffix: 4
      nsize: 2
      args: -dim 1 -stag_grid_x 8

   test:
      suffix: 5
      nsize: 4
      args: -dim 2 -stag_grid_x 4 -stag_grid_y 8

   test:
      suffix: 6
      nsize: 12
      args: -dim 3 -stag_grid_x 4 -stag_grid_y 4 -stag_grid_z 12

TEST*/
