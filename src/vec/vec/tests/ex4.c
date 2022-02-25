
static char help[] = "Scatters from a parallel vector into sequential vectors.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       n   = 5,idx1[2] = {0,3},idx2[2] = {1,4};
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* create two vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  /* create two index sets */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);

  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  if (rank == 0) {ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      filter: grep -v type
      diff_args: -j

   test:
      diff_args: -j
      suffix: cuda
      args: -vec_type cuda
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: cuda

   test:
      diff_args: -j
      suffix: cuda2
      nsize: 2
      args: -vec_type cuda
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: cuda

   test:
      diff_args: -j
      suffix: kokkos
      args: -vec_type kokkos
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: kokkos_kernels

   test:
      diff_args: -j
      suffix: kokkos2
      nsize: 2
      args: -vec_type kokkos
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: !sycl kokkos_kernels

   testset:
      diff_args: -j
      requires: hip
      filter: grep -v type
      args: -vec_type hip
      output_file: output/ex4_1.out
      test:
        suffix: hip
      test:
        suffix: hip2
        nsize: 2
TEST*/
