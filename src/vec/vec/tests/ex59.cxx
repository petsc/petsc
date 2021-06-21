
static char help[] = "Test VecCreate{Seq|MPI}ViennaCLWithArrays.\n\n";

#include "petsc.h"
#include "petscviennacl.h"

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            x,y;
  PetscMPIInt    size;
  PetscInt       n = 5;
  PetscScalar    xHost[5] = {0.,1.,2.,3.,4.};

  ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  if (size == 1) {
    ierr = VecCreateSeqViennaCLWithArrays(PETSC_COMM_WORLD,1,n,xHost,NULL,&x);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIViennaCLWithArrays(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,xHost,NULL,&x);CHKERRQ(ierr);
  }
  /* print x should be equivalent too xHost */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecSet(x,42.0);CHKERRQ(ierr);
  /* print x should be all 42 */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (size == 1) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_WORLD,1,n,xHost,&y);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,xHost,&y);CHKERRQ(ierr);
  }

  /* print y should be all 42 */
  ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: viennacl define(PETSC_HAVE_VIENNACL_NO_CUDA)

   test:
      nsize: 1
      suffix: 1
      args: -viennacl_backend opencl

   test:
      nsize: 2
      suffix: 2
      args: -viennacl_backend opencl

TEST*/
