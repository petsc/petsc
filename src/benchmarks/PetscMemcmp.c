#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y,z;
  int A[10000], B[10000],i;

  PetscInitialize(&argc, &argv,0,0,0);

  for (i=0; i<10000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscMemcmp(A,B,sizeof(int)*10000);
  x = PetscGetTime();

  x = PetscGetTime();
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  PetscMemcmp(A,B,sizeof(int)*10000);
  y = PetscGetTime();
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  PetscMemcmp(A,B,sizeof(int)*0);
  z = PetscGetTime();

  fprintf(stderr,"%s : \n","PetscMemcmp");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per byte",(2*y-x-z)/100000.0);

  PetscFinalize();
  return 0;
}
