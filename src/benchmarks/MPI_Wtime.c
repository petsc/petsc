#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: MPI_Wtime.c,v 1.8 1999/02/22 21:35:57 balay Exp bsmith $";
#endif

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
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

  x = MPI_Wtime();
  PetscSleep(10);
  y = MPI_Wtime();
  fprintf(stderr,"%-15s : %e sec - Slept for 10 sec \n","MPI_Wtime",(y-x));

  PetscFinalize();
  PetscFunctionReturn(0);
}
