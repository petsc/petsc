#ifndef lint
static char vcid[] = "$Id: PetscMemcpy.c,v 1.3 1996/03/07 23:17:49 balay Exp balay $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y,z;
  int i,A[10000], B[10000];

  PetscInitialize(&argc, &argv,0,0,0);
  for (i=0; i<10000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscMemcpy(A,B,sizeof(int)*10000);
  x = PetscGetTime();

  x = PetscGetTime();
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  PetscMemcpy(A,B,sizeof(int)*10000);
  y = PetscGetTime();
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  PetscMemcpy(A,B,sizeof(int)*0);
  z = PetscGetTime();

  fprintf(stderr,"%s : \n","PetscMemcpy");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per byte",(2*y-x-z)/(sizeof(int)*100000.0));

  PetscFinalize();
  return 0;
}
