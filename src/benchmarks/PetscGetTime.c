#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscGetTime.c,v 1.6 1997/07/09 21:01:29 balay Exp bsmith $";
#endif

#include "petsc.h"

int main( int argc, char **argv)
{
  PLogDouble x, y;
  int        i;
  
  PetscInitialize(&argc, &argv,0,0);
 /* To take care of paging effects */
  y = PetscGetTime();

  for ( i=0; i<2; i++ ) {
    x = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();
    y = PetscGetTime();

    fprintf(stderr,"%-15s : %e sec\n","PetscGetTime", (y-x)/10.0);
  }

  PetscFinalize();
  PetscFunctionReturn(0);
}
