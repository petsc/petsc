
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  double         value;
  void           *arr[1000],*dummy;
  int            i,rand1[1000],rand2[1000];
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  for (i=0; i<1000; i++) {
    ierr     = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    rand1[i] = (int)(value* 144327);
    ierr     = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    rand2[i] = (int)(value* 144327);
  }

  /* Take care of paging effects */
  ierr = PetscMalloc1(100,&dummy);CHKERRQ(ierr);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  ierr = PetscTime(&x);CHKERRQ(ierr);

  /* Do all mallocs */
  for (i=0; i< 1000; i++) {
    ierr = PetscMalloc1(rand1[i],&arr[i]);CHKERRQ(ierr);
  }

  ierr = PetscTime(&x);CHKERRQ(ierr);

  /* Do some frees */
  for (i=0; i< 1000; i+=2) {
    ierr = PetscFree(arr[i]);CHKERRQ(ierr);
  }

  /* Do some mallocs */
  for (i=0; i< 1000; i+=2) {
    ierr = PetscMalloc1(rand2[i],&arr[i]);CHKERRQ(ierr);
  }
  ierr = PetscTime(&y);CHKERRQ(ierr);

  for (i=0; i< 1000; i++) {
    ierr = PetscFree(arr[i]);CHKERRQ(ierr);
  }

  fprintf(stdout,"%-15s : %e sec, with options : ","PetscMalloc",(y-x)/500.0);
  ierr = PetscOptionsHasName(NULL,"-malloc",&flg);CHKERRQ(ierr);
  if (flg) fprintf(stdout,"-malloc ");
  fprintf(stdout,"\n");

  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
