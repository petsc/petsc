#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex44.c,v 1.3 1997/09/22 15:23:47 balay Exp bsmith $";
#endif

static char help[] = 
"Loads matrix dumped by ex43.\n\n";

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


