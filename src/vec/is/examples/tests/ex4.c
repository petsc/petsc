/*$Id: ex4.c,v 1.11 2001/01/22 23:02:58 bsmith Exp balay $*/

static char help[] = "Tests ISStrideToGeneral()\n\n";

#include "petscis.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        ierr,step = 2;
  IS         is;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscOptionsGetInt(PETSC_NULL,"-step",&step,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,10,0,step,&is);CHKERRQ(ierr);

  ierr = ISStrideToGeneral(is);CHKERRQ(ierr);

  ierr = ISDestroy(is);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 






