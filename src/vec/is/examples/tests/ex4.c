/*$Id: ex4.c,v 1.7 1999/10/24 14:01:46 bsmith Exp balay $*/

static char help[] = "Tests ISStrideToGeneral()\n\n";

#include "petscis.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        ierr,step = 2;
  IS         is;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = OptionsGetInt(PETSC_NULL,"-step",&step,PETSC_NULL);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,10,0,step,&is);CHKERRA(ierr);

  ierr = ISStrideToGeneral(is);CHKERRA(ierr);

  ierr = ISDestroy(is);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 






