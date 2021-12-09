static char help[] = "Benchmark VecCreate() for GPU vectors.\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>
#include <petsctime.h>
#include <petscdevice.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 5, iter = 10;
  Vec            x;
  PetscLogDouble v0,v1;
  PetscMemType   memtype;
  PetscScalar    *array;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-iter",&iter,NULL);CHKERRQ(ierr);

  for (i=0; i<iter; i++) {
    ierr = PetscTime(&v0);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);
    /* make sure the vector's array exists */
    ierr = VecGetArrayAndMemType(x,&array,&memtype);CHKERRQ(ierr);
    ierr = VecRestoreArrayAndMemType(x,&array);CHKERRQ(ierr);
    ierr = WaitForCUDA();CHKERRQ(ierr);
    ierr = PetscTime(&v1);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Iteration %" PetscInt_FMT ": Time= %g\n",i,(double)(v1-v0));CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}
/*TEST
  build:
      requires: cuda
  test:
      args: -vec_type cuda
TEST*/
