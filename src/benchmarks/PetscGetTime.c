
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscInt       i;

  CHKERRQ(PetscInitialize(&argc,&argv,0,0));
  /* To take care of paging effects */
  CHKERRQ(PetscTime(&y));

  for (i=0; i<2; i++) {
    CHKERRQ(PetscTime(&x));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    CHKERRQ(PetscTime(&y));
    fprintf(stdout,"%-15s : %e sec\n","PetscTime",(y-x)/10.0);
  }

  CHKERRQ(PetscFinalize());
  return 0;
}
