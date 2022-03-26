
#include <petscsys.h>

int main(int argc,char **argv)
{
  double x,y;
  int    ierr;

  PetscCall(PetscInitialize(&argc,&argv,0,0));
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

  fprintf(stdout,"%-15s : %e sec\n","MPI_Wtime",(y-x)/10.0);
  y = MPI_Wtick();
  fprintf(stdout,"%-15s : %e sec\n","MPI_Wtick",y);

  x    = MPI_Wtime();
  PetscCall(PetscSleep(10));
  y    = MPI_Wtime();
  fprintf(stdout,"%-15s : %e sec - Slept for 10 sec \n","MPI_Wtime",(y-x));

  PetscCall(PetscFinalize());
  return 0;
}
