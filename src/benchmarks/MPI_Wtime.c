#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y;
  
  PetscInitialize(&argc, &argv,0,0,0);

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
