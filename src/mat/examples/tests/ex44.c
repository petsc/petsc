/*$Id: ex44.c,v 1.10 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = 
"Loads matrix dumped by ex43.\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     C;
  PetscViewer  viewer;
  int     ierr;

  PetscInitialize(&argc,&args,0,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",PETSC_BINARY_RDONLY,&viewer); 
        CHKERRA(ierr);
  MatLoad(viewer,MATMPIDENSE,&C);CHKERRA(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRA(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


