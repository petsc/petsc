
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

  PetscCall(PetscInitialize(&argc,&argv,0,0));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
  PetscCall(PetscRandomSetFromOptions(r));
  for (i=0; i<1000; i++) {
    PetscCall(PetscRandomGetValue(r,&value));
    rand1[i] = (int)(value* 144327);
    PetscCall(PetscRandomGetValue(r,&value));
    rand2[i] = (int)(value* 144327);
  }

  /* Take care of paging effects */
  PetscCall(PetscMalloc1(100,&dummy));
  PetscCall(PetscFree(dummy));
  PetscCall(PetscTime(&x));

  /* Do all mallocs */
  for (i=0; i< 1000; i++) {
    PetscCall(PetscMalloc1(rand1[i],&arr[i]));
  }

  PetscCall(PetscTime(&x));

  /* Do some frees */
  for (i=0; i< 1000; i+=2) {
    PetscCall(PetscFree(arr[i]));
  }

  /* Do some mallocs */
  for (i=0; i< 1000; i+=2) {
    PetscCall(PetscMalloc1(rand2[i],&arr[i]));
  }
  PetscCall(PetscTime(&y));

  for (i=0; i< 1000; i++) {
    PetscCall(PetscFree(arr[i]));
  }

  fprintf(stdout,"%-15s : %e sec, with options : ","PetscMalloc",(y-x)/500.0);
  PetscCall(PetscOptionsHasName(NULL,"-malloc",&flg));
  if (flg) fprintf(stdout,"-malloc ");
  fprintf(stdout,"\n");

  PetscCall(PetscRandomDestroy(&r));
  PetscCall(PetscFinalize());
  return 0;
}
