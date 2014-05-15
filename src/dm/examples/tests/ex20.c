static char help[] = "Tests DMDACreate3d() memory usage\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DM dm;
  Vec X,Y;
  PetscLogDouble mem,oldmem;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,-128,-128,-128,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,10,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  oldmem = 0;
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"DMDACreate3d        : %8.3f MB + %8.3f MB\n",mem*1e-6,(mem-oldmem)*1e-6);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  oldmem = mem;
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"DMCreateGlobalVector: %8.3f MB + %8.3f MB\n",mem*1e-6,(mem-oldmem)*1e-6);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&Y);CHKERRQ(ierr);
  oldmem = mem;
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"DMCreateGlobalVector: %8.3f MB + %8.3f MB\n",mem*1e-6,(mem-oldmem)*1e-6);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

