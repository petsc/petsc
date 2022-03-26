
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  long int       i,j,A[100000],ierr;

  PetscCall(PetscInitialize(&argc,&argv,0,0));
  /* To take care of paging effects */
  PetscCall(PetscGetCPUTime(&y));

  for (i=0; i<2; i++) {
    PetscCall(PetscGetCPUTime(&x));

    /*
       Do some work for at least 1 ms. Most CPU timers
       cannot measure anything less than that
     */

    for (j=0; j<20000*(i+1); j++) A[j]=i+j;
    PetscCall(PetscGetCPUTime(&y));
    fprintf(stdout,"%-15s : %e sec\n","PetscGetCPUTime",(y-x)/10.0);
  }

  PetscCall(PetscFinalize());
  return 0;
}
