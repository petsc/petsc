/*$Id: PetscGetTime.c,v 1.14 2001/01/15 21:49:39 bsmith Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  int        i,ierr;
  
  PetscInitialize(&argc,&argv,0,0);
 /* To take care of paging effects */
  ierr = PetscGetTime(&y);CHKERRQ(ierr);

  for (i=0; i<2; i++) {
    ierr = PetscGetTime(&x);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);
    ierr = PetscGetTime(&y);CHKERRQ(ierr);

    fprintf(stdout,"%-15s : %e sec\n","PetscGetTime",(y-x)/10.0);
  }

  PetscFinalize();
  PetscFunctionReturn(0);
}
