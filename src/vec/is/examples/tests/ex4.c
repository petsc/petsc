/*$Id: ex4.c,v 1.9 2001/01/15 21:44:31 bsmith Exp bsmith $*/

static char help[] = "Tests ISStrideToGeneral()\n\n";

#include "petscis.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        ierr,step = 2;
  IS         is;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-step",&step,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,10,0,step,&is);CHKERRQ(ierr);

  ierr = ISStrideToGeneral(is);CHKERRQ(ierr);

  ierr = ISDestroy(is);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
 






