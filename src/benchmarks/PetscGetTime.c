#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscGetTime.c,v 1.7 1997/10/19 03:30:47 bsmith Exp balay $";
#endif

#include "petsc.h"

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
