/*$Id: PetscGetTime.c,v 1.11 1999/10/24 14:04:16 bsmith Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PLogDouble x,y;
  int        i,ierr;
  
  PetscInitialize(&argc,&argv,0,0);
 /* To take care of paging effects */
  ierr = PetscGetTime(&y);CHKERRA(ierr);

  for (i=0; i<2; i++) {
    ierr = PetscGetTime(&x);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);
    ierr = PetscGetTime(&y);CHKERRA(ierr);

    fprintf(stderr,"%-15s : %e sec\n","PetscGetTime",(y-x)/10.0);
  }

  PetscFinalize();
  PetscFunctionReturn(0);
}
