#ifndef lint
static char vcid[] = "$Id: aij.c,v 1.153 1996/03/04 05:15:52 bsmith Exp bsmith $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y;
  
  PetscInitialize(&argc, &argv,0,0,0);
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
  PetscFinalize();
  return 0;
}
