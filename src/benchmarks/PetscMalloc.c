
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  double         value;
  void           *arr[1000],*dummy;
  int            i,rand1[1000],rand2[1000];
  PetscRandom    r;
  PetscBool      flg;

  CHKERRQ(PetscInitialize(&argc,&argv,0,0));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));
  for (i=0; i<1000; i++) {
    CHKERRQ(PetscRandomGetValue(r,&value));
    rand1[i] = (int)(value* 144327);
    CHKERRQ(PetscRandomGetValue(r,&value));
    rand2[i] = (int)(value* 144327);
  }

  /* Take care of paging effects */
  CHKERRQ(PetscMalloc1(100,&dummy));
  CHKERRQ(PetscFree(dummy));
  CHKERRQ(PetscTime(&x));

  /* Do all mallocs */
  for (i=0; i< 1000; i++) {
    CHKERRQ(PetscMalloc1(rand1[i],&arr[i]));
  }

  CHKERRQ(PetscTime(&x));

  /* Do some frees */
  for (i=0; i< 1000; i+=2) {
    CHKERRQ(PetscFree(arr[i]));
  }

  /* Do some mallocs */
  for (i=0; i< 1000; i+=2) {
    CHKERRQ(PetscMalloc1(rand2[i],&arr[i]));
  }
  CHKERRQ(PetscTime(&y));

  for (i=0; i< 1000; i++) {
    CHKERRQ(PetscFree(arr[i]));
  }

  fprintf(stdout,"%-15s : %e sec, with options : ","PetscMalloc",(y-x)/500.0);
  CHKERRQ(PetscOptionsHasName(NULL,"-malloc",&flg));
  if (flg) fprintf(stdout,"-malloc ");
  fprintf(stdout,"\n");

  CHKERRQ(PetscRandomDestroy(&r));
  CHKERRQ(PetscFinalize());
  return 0;
}
