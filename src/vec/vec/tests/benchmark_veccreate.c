static char help[] = "Benchmark VecCreate() for GPU vectors.\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>
#include <petsctime.h>
#include <petscdevice.h>

int main(int argc,char **argv)
{
  PetscInt        i,n = 5, iter = 10;
  Vec             x;
  PetscLogDouble  v0,v1;
  PetscMemType    memtype;
  PetscScalar    *array;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-iter",&iter,NULL));

  for (i=0; i<iter; i++) {
    CHKERRQ(PetscTime(&v0));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
    CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
    CHKERRQ(VecSetFromOptions(x));
    /* make sure the vector's array exists */
    CHKERRQ(VecGetArrayAndMemType(x,&array,&memtype));
    CHKERRQ(VecRestoreArrayAndMemType(x,&array));
    CHKERRQ(WaitForCUDA());
    CHKERRQ(PetscTime(&v1));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Iteration %" PetscInt_FMT ": Time= %g\n",i,(double)(v1-v0)));
  }
  CHKERRQ(PetscFinalize());
  return 0;
}
/*TEST
  build:
      requires: cuda
  test:
      args: -vec_type cuda
TEST*/
