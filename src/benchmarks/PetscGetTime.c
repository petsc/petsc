#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscGetTime.c,v 1.8 1998/03/31 23:34:07 balay Exp bsmith $";
#endif

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv)
{
  PLogDouble x, y;
  int        i,ierr;
  
  PetscInitialize(&argc, &argv,0,0);
 /* To take care of paging effects */
  ierr = PetscGetTime(&y); CHKERRA(ierr);

  for ( i=0; i<2; i++ ) {
    ierr = PetscGetTime(&x); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);
    ierr = PetscGetTime(&y); CHKERRA(ierr);

    fprintf(stderr,"%-15s : %e sec\n","PetscGetTime", (y-x)/10.0);
  }

  PetscFinalize();
  PetscFunctionReturn(0);
}
