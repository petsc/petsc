static char help[] = "Tests DMDACreate3d() memory usage\n\n";

#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM             dm;
  Vec            X,Y;
  PetscErrorCode ierr;
  PetscInt       dof = 10;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,-128,-128,-128,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = PetscMemoryTrace("DMDACreate3d        ");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = PetscMemoryTrace("DMCreateGlobalVector");
  ierr = DMCreateGlobalVector(dm,&Y);CHKERRQ(ierr);
  ierr = PetscMemoryTrace("DMCreateGlobalVector");CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

