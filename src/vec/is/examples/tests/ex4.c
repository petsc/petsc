#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.3 1998/12/03 03:56:25 bsmith Exp bsmith $";
#endif

static char help[] = "Tests ISStrideToGeneral()\n\n";

#include "is.h"

#undef __FUNC__
#define __FUNC__ "main"
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
 






