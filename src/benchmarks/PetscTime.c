#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscTime.c,v 1.8 1997/03/09 18:00:35 bsmith Exp balay $";
#endif

#include "stdio.h"
#include "petsc.h"
#include "pinclude/ptime.h"

int main( int argc, char **argv)
{
  PLogDouble x, y;
  int        i;

  PetscInitialize(&argc, &argv,0,0);
  /* To take care of paging effects */
  PetscTime(y);

  for ( i=0; i<2; i++ ) { 
    PetscTime(x);
    PetscTime(y); 
    PetscTime(y);
    PetscTime(y);
    PetscTime(y);
    PetscTime(y);
    PetscTime(y); 
    PetscTime(y);
    PetscTime(y);
    PetscTime(y);
    PetscTime(y);

    fprintf(stderr,"%-15s : %e sec\n","PetscTime",(y-x)/10.0);
  }
  PetscTime(x);
  PetscSleep(10);
  PetscTime(y); 
  fprintf(stderr,"%-15s : %e sec - Slept for 10 sec \n","PetscTime",(y-x));

  PetscFinalize();
  return 0;
}
