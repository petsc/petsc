
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  PetscScalar    A[10000];

  PetscCall(PetscInitialize(&argc,&argv,0,0));
  /* To take care of paging effects */
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscTime(&x));

  PetscCall(PetscTime(&x));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*10000));
  PetscCall(PetscTime(&y));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscMemzero(A,sizeof(PetscScalar)*0));
  PetscCall(PetscTime(&z));

  fprintf(stdout,"%s : \n","PetscMemzero");
  fprintf(stdout,"    %-15s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-15s : %e sec\n","Per PetscScalar",(2*y-x-z)/100000.0);

  PetscCall(PetscFinalize());
  return 0;
}
