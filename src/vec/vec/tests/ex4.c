
static char help[] = "Scatters from a parallel vector into sequential vectors.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       n   = 5,idx1[2] = {0,3},idx2[2] = {1,4};
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&y));
  CHKERRQ(VecSetSizes(y,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(y));

  /* create two index sets */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,2,idx1,PETSC_COPY_VALUES,&is1));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,2,idx2,PETSC_COPY_VALUES,&is2));

  CHKERRQ(VecSet(x,one));
  CHKERRQ(VecSet(y,two));
  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  if (rank == 0) CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscFinalize());
  return 0;
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
