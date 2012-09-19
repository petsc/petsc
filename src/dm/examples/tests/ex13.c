
static char help[] = "Tests loading DM vector from file.\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       M = PETSC_DECIDE,N = PETSC_DECIDE;
  DM             da;
  Vec            global;
  PetscViewer    bviewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",FILE_MODE_READ,&bviewer);CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_WORLD,&da);CHKERRQ(ierr);

  ierr = DMLoad(da,bviewer);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = VecLoad(global,bviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&bviewer);CHKERRQ(ierr);


  ierr = VecView(global,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);


  /* Free memory */
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

