#ifndef lint
static char vcid[] = "$Id: MPI_Wtime.c,v 1.3 1996/03/06 17:40:32 balay Exp $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y;
  
  PetscInitialize(&argc, &argv,0,0,0);
 /* To take care of paging effects */
  y = PetscGetTime();

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
  PetscFinalize();
  return 0;
}
