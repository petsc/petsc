#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscGetTime.c,v 1.5 1997/03/09 18:00:35 bsmith Exp balay $";
#endif

#include "stdio.h"
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
  return 0;
}
