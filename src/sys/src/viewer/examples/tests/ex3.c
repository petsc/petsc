/*$Id: ex3.c,v 1.4 1999/10/24 14:01:08 bsmith Exp bsmith $*/

static char help[] = "Tests dynamic loading of viewer.\n\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int     ierr;
  PetscViewer  viewer;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRA(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRA(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRA(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&viewer);CHKERRA(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


