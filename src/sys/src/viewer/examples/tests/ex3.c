/*$Id: ex3.c,v 1.7 2001/01/22 23:01:30 bsmith Exp balay $*/

static char help[] = "Tests dynamic loading of viewer.\n\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int     ierr;
  PetscViewer  viewer;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
    


