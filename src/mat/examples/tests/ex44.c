#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex44.c,v 1.2 1997/07/09 20:55:45 balay Exp balay $";
#endif

static char help[] = 
"Loads matrix dumped by ex43.\n\n";

#include <stdio.h>
#include "mat.h"


int main(int argc,char **args)
{
  Mat     C;
  Viewer  viewer;
  int     ierr;

  PetscInitialize(&argc,&args,0,0,help);
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"matrix.dat",BINARY_RDONLY,&viewer); 
         CHKERRA(ierr);
  MatLoad(viewer,MATMPIDENSE,&C); CHKERRA(ierr);
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


