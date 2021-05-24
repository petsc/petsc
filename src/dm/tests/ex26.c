
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
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,2,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = DMCreateColoring(da,IS_COLORING_LOCAL,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(J,iscoloring,matfdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

  /* free spaces */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&matfdcoloring);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
