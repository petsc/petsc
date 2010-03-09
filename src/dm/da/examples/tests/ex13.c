
static char help[] = "Tests loading DA vector from file.\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       M = PETSC_DECIDE,N = PETSC_DECIDE;
  DA             da;
  Vec            global;
  PetscViewer    bviewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",FILE_MODE_READ,&bviewer);CHKERRQ(ierr);
  ierr = DALoad(bviewer,M,N,PETSC_DECIDE,&da);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr); 
  ierr = VecLoadIntoVector(bviewer,global);CHKERRQ(ierr); 
  ierr = PetscViewerDestroy(bviewer);CHKERRQ(ierr);


  ierr = VecView(global,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);


  /* Free memory */
  ierr = VecDestroy(global);CHKERRQ(ierr); 
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
