#ifndef lint
static char vcid[] = "$Id: MPI_Wtime.c,v 1.4 1996/03/19 21:30:13 bsmith Exp balay $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y;
  
  PetscInitialize(&argc, &argv,0,0);
 /* To take care of paging effects */
  y = MPI_Wtime();

  x = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();

  fprintf(stderr,"%-15s : %e sec\n","MPI_Wtime",(y-x)/10.0);
  y = MPI_Wtick();
  fprintf(stderr,"%-15s : %e sec\n","MPI_Wtick",y);


  PetscFinalize();
  return 0;
}
