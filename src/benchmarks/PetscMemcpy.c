
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscScalar    *A,*B;

  CHKERRQ(PetscInitialize(&argc,&argv,0,0));
  CHKERRQ(PetscCalloc1(8000000,&A));
  CHKERRQ(PetscMalloc1(8000000,&B));

  for (i=0; i<8000000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  CHKERRQ(PetscArraycpy(A,B,8000000));
  CHKERRQ(PetscTime(&x));
  CHKERRQ(PetscArraycpy(A,B,8000000));
  CHKERRQ(PetscTime(&x));

  fprintf(stdout,"%s : \n","PetscMemcpy");
  fprintf(stdout,"    %-15s : %e MB/s\n","Bandwidth",10.0*8*8/(y-x));

  CHKERRQ(PetscFree(A));
  CHKERRQ(PetscFree(B));
  CHKERRQ(PetscFinalize());
  return 0;
}
