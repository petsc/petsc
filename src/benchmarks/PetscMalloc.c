#include "stdio.h"
#include "petsc.h"
#include "sys.h"

int main( int argc, char **argv)
{
  double   x, y, value;
  void     *arr[1000];
  int      ierr, i, flg, rand1[1000], rand2[1000];
  SYRandom r;
  
  PetscInitialize(&argc, &argv,0,0,0);
  
  ierr = SYRandomCreate( MPI_COMM_SELF,RANDOM_DEFAULT,&r); CHKERRQ(ierr);
  for (i=0; i<1000; i++) {
    ierr    = SYRandomGetValue(r, &value); CHKERRQ(ierr);
    rand1[i] = (int ) (value* 144327);
    ierr    = SYRandomGetValue(r, &value); CHKERRQ(ierr);
    rand2[i] = (int ) (value* 144327);
  }
  
  /* Do all mallocs */
  for (i=0 ; i< 1000; i++) {
    arr[i] = PetscMalloc(rand1[i]); CHKPTRA( arr[i]);
  }
  
  x = PetscGetTime(); 

  /* Do some frees */
  for (i=0; i< 1000; i+=2) {
    PetscFree(arr[i]);
  }

  /* Do some mallocs */
  for (i=0; i< 1000; i+=2) {
    arr[i] = PetscMalloc(rand2[i]); CHKPTRA( arr[i]);
 }
  y = PetscGetTime();
  
  for (i=0; i< 1000; i++) {
    PetscFree(arr[i]);
  }
  
  fprintf(stderr,"%-15s : %e sec , with options : ","PLogEvent",(y-x)/10.0);
  if(OptionsHasName(PETSC_NULL,"-trmalloc",&flg),flg) fprintf(stderr,"-trmalloc ");
  fprintf(stderr,"\n"); 
  
  SYRandomDestroy(r);
  PetscFinalize();
  return 0;
}
