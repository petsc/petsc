/*$Id: PetscMalloc.c,v 1.17 1999/11/05 14:48:10 bsmith Exp bsmith $*/

#include "petsc.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PLogDouble  x,y;
  double      value;
  void        *arr[1000],*dummy;
  int         ierr,i,rand1[1000],rand2[1000];
  PetscRandom r;
  PetscTruth  flg;
  
  PetscInitialize(&argc,&argv,0,0);
  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRQ(ierr);
  for (i=0; i<1000; i++) {
    ierr    = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    rand1[i] = (int)(value* 144327);
    ierr    = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    rand2[i] = (int)(value* 144327);
  }
  
  /* Take care of paging effects */
  dummy = PetscMalloc(100);CHKPTRA(dummy);
  ierr = PetscFree(dummy);CHKERRA(ierr);
  ierr = PetscGetTime(&x);CHKERRA(ierr);

  /* Do all mallocs */
  for (i=0 ; i< 1000; i++) {
    arr[i] = PetscMalloc(rand1[i]);CHKPTRA(arr[i]);
  }
  
  ierr = PetscGetTime(&x);CHKERRA(ierr);

  /* Do some frees */
  for (i=0; i< 1000; i+=2) {
    ierr = PetscFree(arr[i]);CHKERRA(ierr);
  }

  /* Do some mallocs */
  for (i=0; i< 1000; i+=2) {
    arr[i] = PetscMalloc(rand2[i]);CHKPTRA(arr[i]);
 }
  ierr = PetscGetTime(&y);CHKERRA(ierr);
  
  for (i=0; i< 1000; i++) {
    ierr = PetscFree(arr[i]);CHKERRA(ierr);
  }
  
  fprintf(stderr,"%-15s : %e sec, with options : ","PetscMalloc",(y-x)/500.0);
  if(OptionsHasName(PETSC_NULL,"-trmalloc",&flg),flg) fprintf(stderr,"-trmalloc ");
  fprintf(stderr,"\n"); 
  
  PetscRandomDestroy(r);
  PetscFinalize();
  PetscFunctionReturn(0);
}
