
static char help[] = "Tests error message in DMCreateColoring() with periodic boundary conditions. \n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            J;
  DM             da;
  MatFDColoring  matfdcoloring = 0;
  ISColoring     iscoloring;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,2,0,0,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetMatType(da,MATAIJ));
  PetscCall(DMCreateMatrix(da,&J));
  PetscCall(DMCreateColoring(da,IS_COLORING_LOCAL,&iscoloring));
  PetscCall(MatFDColoringCreate(J,iscoloring,&matfdcoloring));
  PetscCall(MatFDColoringSetUp(J,iscoloring,matfdcoloring));
  PetscCall(ISColoringDestroy(&iscoloring));

  /* free spaces */
  PetscCall(MatDestroy(&J));
  PetscCall(MatFDColoringDestroy(&matfdcoloring));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
