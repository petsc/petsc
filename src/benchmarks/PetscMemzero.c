#ifndef lint
static char vcid[] = "$Id: PetscMemzero.c,v 1.4 1996/03/08 16:05:00 balay Exp balay $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y,z;
  Scalar A[10000];

  PetscInitialize(&argc, &argv,0,0,0);
  /* To take care of paging effects */
  PetscMemzero(A,sizeof(Scalar)*0);
  x = PetscGetTime();

  x = PetscGetTime();
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  y = PetscGetTime();
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  z = PetscGetTime();

  fprintf(stderr,"%s : \n","PetscMemzero");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per byte",(2*y-x-z)/100000.0);

  PetscFinalize();
  return 0;
}
