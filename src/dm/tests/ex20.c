static char help[] = "Tests DMDACreate3d() memory usage\n\n";

#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM             dm;
  Vec            X,Y;
  PetscErrorCode ierr;
  PetscInt       dof = 10;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,-128,-128,-128,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,NULL,&dm));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(PetscMemoryTrace("DMDACreate3d        "));

  CHKERRQ(DMCreateGlobalVector(dm,&X));
  CHKERRQ(PetscMemoryTrace("DMCreateGlobalVector"));
  CHKERRQ(DMCreateGlobalVector(dm,&Y));
  CHKERRQ(PetscMemoryTrace("DMCreateGlobalVector"));

  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}
