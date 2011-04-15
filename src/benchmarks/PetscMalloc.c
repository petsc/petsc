
#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble  x,y;
  double      value;
  void        *arr[1000],*dummy;
  int         ierr,i,rand1[1000],rand2[1000];
  PetscRandom r;
  PetscBool   flg;
  
  PetscInitialize(&argc,&argv,0,0);
  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  for (i=0; i<1000; i++) {
    ierr    = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    rand1[i] = (int)(value* 144327);
    ierr    = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    rand2[i] = (int)(value* 144327);
  }
  
  /* Take care of paging effects */
  ierr = PetscMalloc(100,&dummy);CHKERRQ(ierr);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  ierr = PetscGetTime(&x);CHKERRQ(ierr);

  /* Do all mallocs */
  for (i=0 ; i< 1000; i++) {
    ierr = PetscMalloc(rand1[i],& arr[i]);CHKERRQ(ierr);
  }
  
  ierr = PetscGetTime(&x);CHKERRQ(ierr);

  /* Do some frees */
  for (i=0; i< 1000; i+=2) {
    ierr = PetscFree(arr[i]);CHKERRQ(ierr);
  }

  /* Do some mallocs */
  for (i=0; i< 1000; i+=2) {
    ierr = PetscMalloc(rand2[i],&arr[i]);CHKERRQ(ierr);
 }
  ierr = PetscGetTime(&y);CHKERRQ(ierr);
  
  for (i=0; i< 1000; i++) {
    ierr = PetscFree(arr[i]);CHKERRQ(ierr);
  }
  
  fprintf(stdout,"%-15s : %e sec, with options : ","PetscMalloc",(y-x)/500.0);
  if(PetscOptionsHasName(PETSC_NULL,"-malloc",&flg),flg) fprintf(stdout,"-malloc ");
  fprintf(stdout,"\n"); 
  
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
