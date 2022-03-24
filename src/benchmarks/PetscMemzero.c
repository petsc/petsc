
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  PetscScalar    A[10000];

  CHKERRQ(PetscInitialize(&argc,&argv,0,0));
  /* To take care of paging effects */
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscTime(&x));

  CHKERRQ(PetscTime(&x));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*10000));
  CHKERRQ(PetscTime(&y));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscMemzero(A,sizeof(PetscScalar)*0));
  CHKERRQ(PetscTime(&z));

  fprintf(stdout,"%s : \n","PetscMemzero");
  fprintf(stdout,"    %-15s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-15s : %e sec\n","Per PetscScalar",(2*y-x-z)/100000.0);

  CHKERRQ(PetscFinalize());
  return 0;
}
