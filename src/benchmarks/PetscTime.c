
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  int            i;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
  /* To take care of paging effects */
  PetscTime(&y);

  for (i=0; i<2; i++) {
    PetscTime(&x);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    PetscTime(&y);
    fprintf(stdout,"%-15s : %e sec\n","PetscTime",(y-x)/10.0);
  }
  PetscTime(&x);
  ierr = PetscSleep(10);CHKERRQ(ierr);
  PetscTime(&y);
  fprintf(stdout,"%-15s : %e sec - Slept for 10 sec \n","PetscTime",(y-x));
  ierr = PetscFinalize();
  return ierr;
}
