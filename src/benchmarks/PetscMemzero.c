#ifndef lint
static char vcid[] = "$Id: MPI_Wtime.c,v 1.3 1996/03/06 17:40:32 balay Exp $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y,z;
  int A[10000];

  PetscInitialize(&argc, &argv,0,0,0);
  /* To take care of paging effects */
  PetscMemzero(A,sizeof(int)*10000);
  x = PetscGetTime();

  x = PetscGetTime();
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  PetscMemzero(A,sizeof(int)*10000);
  y = PetscGetTime();
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  PetscMemzero(A,sizeof(int)*0);
  z = PetscGetTime();

  fprintf(stderr,"%s : \n","PetscMemzero");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per byte",(2*y-x-z)/100000.0);

  PetscFinalize();
  return 0;
}
