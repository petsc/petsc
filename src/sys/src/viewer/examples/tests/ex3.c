/*$Id: ex3.c,v 1.5 2001/01/15 21:43:18 bsmith Exp bsmith $*/

static char help[] = "Tests dynamic loading of viewer.\n\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int     ierr;
  PetscViewer  viewer;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
    


