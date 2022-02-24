
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscInt       i;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
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

  ierr = PetscFinalize();
  return ierr;
}
