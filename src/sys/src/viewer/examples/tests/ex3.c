#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.1 1999/01/13 04:34:57 bsmith Exp bsmith $";
#endif

static char help[] = "Tests dynamic loading of viewer.\n\n";

#include "petsc.h"

int main(int argc,char **args)
{
  int     ierr;
  Viewer  viewer;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = ViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRA(ierr);
  ierr = ViewerSetFromOptions(viewer);CHKERRA(ierr);
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);

  ierr = ViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&viewer);CHKERRA(ierr);
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


