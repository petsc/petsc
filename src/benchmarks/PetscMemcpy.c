
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscScalar    *A,*B;

  PetscCall(PetscInitialize(&argc,&argv,0,0));
  PetscCall(PetscCalloc1(8000000,&A));
  PetscCall(PetscMalloc1(8000000,&B));

  for (i=0; i<8000000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscCall(PetscArraycpy(A,B,8000000));
  PetscCall(PetscTime(&x));
  PetscCall(PetscArraycpy(A,B,8000000));
  PetscCall(PetscTime(&x));

  fprintf(stdout,"%s : \n","PetscMemcpy");
  fprintf(stdout,"    %-15s : %e MB/s\n","Bandwidth",10.0*8*8/(y-x));

  PetscCall(PetscFree(A));
  PetscCall(PetscFree(B));
  PetscCall(PetscFinalize());
  return 0;
}
