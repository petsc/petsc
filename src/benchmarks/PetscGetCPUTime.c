/*$Id: PetscGetCPUTime.c,v 1.7 2000/01/11 21:03:44 bsmith Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  long int   i,j,A[100000],ierr;
  
  PetscInitialize(&argc,&argv,0,0);
 /* To take care of paging effects */
  ierr = PetscGetCPUTime(&y);CHKERRA(ierr);

  for (i=0; i<2; i++) {
    ierr = PetscGetCPUTime(&x);CHKERRA(ierr);

    /* 
       Do some work for at least 1 ms. Most CPU timers
       cannot measure anything less than that
     */
       
    for (j=0; j<20000*(i+1); j++) {
      A[j]=i+j;
    }
    ierr = PetscGetCPUTime(&y);CHKERRA(ierr);
    fprintf(stderr,"%-15s : %e sec\n","PetscGetCPUTime",(y-x)/10.0);
  }

  PetscFinalize();
  PetscFunctionReturn(0);
}
