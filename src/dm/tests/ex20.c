static char help[] = "Tests DMDACreate3d() memory usage\n\n";

#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM             dm;
  Vec            X,Y;
  PetscInt       dof = 10;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,-128,-128,-128,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,NULL,&dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(PetscMemoryTrace("DMDACreate3d        "));

  PetscCall(DMCreateGlobalVector(dm,&X));
  PetscCall(PetscMemoryTrace("DMCreateGlobalVector"));
  PetscCall(DMCreateGlobalVector(dm,&Y));
  PetscCall(PetscMemoryTrace("DMCreateGlobalVector"));

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
