
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscInt       i;

  PetscCall(PetscInitialize(&argc,&argv,0,0));
  /* To take care of paging effects */
  PetscCall(PetscTime(&y));

  for (i=0; i<2; i++) {
    PetscCall(PetscTime(&x));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    PetscCall(PetscTime(&y));
    fprintf(stdout,"%-15s : %e sec\n","PetscTime",(y-x)/10.0);
  }

  PetscCall(PetscFinalize());
  return 0;
}
