#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.1 1998/06/03 16:24:37 bsmith Exp bsmith $";
#endif

static char help[] = "Tests ISStrideToGeneral()\n\n";

#include "is.h"
#include <math.h>

int main(int argc,char **argv)
{
  int        i, n, ierr,*ii,start,stride,step = 2;
  IS         is;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = OptionsGetInt(PETSC_NULL,"-step",&step,PETSC_NULL);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,10,0,step,&is); CHKERRA(ierr);

  ierr = ISStrideToGeneral(is);CHKERRA(ierr);

  ierr = ISDestroy(is); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 






