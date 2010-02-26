
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  long int   i,j,A[100000],ierr;
  
  PetscInitialize(&argc,&argv,0,0);
 /* To take care of paging effects */
  ierr = PetscGetCPUTime(&y);CHKERRQ(ierr);

  for (i=0; i<2; i++) {
    ierr = PetscGetCPUTime(&x);CHKERRQ(ierr);

    /* 
       Do some work for at least 1 ms. Most CPU timers
       cannot measure anything less than that
     */
       
    for (j=0; j<20000*(i+1); j++) {
      A[j]=i+j;
    }
    ierr = PetscGetCPUTime(&y);CHKERRQ(ierr);
    fprintf(stdout,"%-15s : %e sec\n","PetscGetCPUTime",(y-x)/10.0);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
