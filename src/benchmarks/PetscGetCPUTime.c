
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  long int       i,j,A[100000],ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
  /* To take care of paging effects */
  CHKERRQ(PetscGetCPUTime(&y));

  for (i=0; i<2; i++) {
    CHKERRQ(PetscGetCPUTime(&x));

    /*
       Do some work for at least 1 ms. Most CPU timers
       cannot measure anything less than that
     */

    for (j=0; j<20000*(i+1); j++) A[j]=i+j;
    CHKERRQ(PetscGetCPUTime(&y));
    fprintf(stdout,"%-15s : %e sec\n","PetscGetCPUTime",(y-x)/10.0);
  }

  ierr = PetscFinalize();
  return ierr;
}
