#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscGetCPUTime.c,v 1.1 1997/07/23 16:22:44 balay Exp bsmith $";
#endif

#include "petsc.h"

int main( int argc, char **argv)
{
  PLogDouble x, y;
  long int   i,j,A[100000];
  
  PetscInitialize(&argc, &argv,0,0);
 /* To take care of paging effects */
  y = PetscGetCPUTime();

  for ( i=0; i<2; i++ ) {
    x = PetscGetCPUTime();

    /* 
       Do some work for at least 1 ms. Most CPU timers
       cannot measure anything less than that
     */
       
    for (j=0; j<20000*(i+1); j++) {
      A[j]=i+j;
    }
    y = PetscGetCPUTime();
    fprintf(stderr,"%-15s : %e sec\n","PetscGetCPUTime", (y-x)/10.0);
  }

  PetscFinalize();
  PetscFunctionReturn(0);
}
