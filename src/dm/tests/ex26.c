
static char help[] = "Tests error message in DMCreateColoring() with periodic boundary conditions. \n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            J;
  PetscErrorCode ierr;
  DM             da;
  MatFDColoring  matfdcoloring = 0;
  ISColoring     iscoloring;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,2,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMSetMatType(da,MATAIJ));
  CHKERRQ(DMCreateMatrix(da,&J));
  CHKERRQ(DMCreateColoring(da,IS_COLORING_LOCAL,&iscoloring));
  CHKERRQ(MatFDColoringCreate(J,iscoloring,&matfdcoloring));
  CHKERRQ(MatFDColoringSetUp(J,iscoloring,matfdcoloring));
  CHKERRQ(ISColoringDestroy(&iscoloring));

  /* free spaces */
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(MatFDColoringDestroy(&matfdcoloring));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}
