static char help[] = "Tests VecPlaceArray() and VecReciprocal().\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n=5,bs;
  PetscBool      iscuda,iskokkos,iship;
  Vec            x,x1,x2,x3;
  PetscScalar    *px;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  /* create vector of length 2*n */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&x));
  CHKERRQ(VecSetSizes(x,3*n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));

  /* create two vectors of length n without array */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)x,VECSEQCUDA,&iscuda));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)x,VECSEQKOKKOS,&iskokkos));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)x,VECSEQHIP,&iship));
  CHKERRQ(VecGetBlockSize(x,&bs));
  if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
    CHKERRQ(VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    CHKERRQ(VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    CHKERRQ(VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
#endif
  } else if (iskokkos) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    CHKERRQ(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    CHKERRQ(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    CHKERRQ(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
#endif
  } else if (iship) {
#if defined(PETSC_HAVE_HIP)
    CHKERRQ(VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    CHKERRQ(VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    CHKERRQ(VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
#endif
  } else {
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
  }

  CHKERRQ(VecGetArrayWrite(x,&px));
  CHKERRQ(VecPlaceArray(x1,px));
  CHKERRQ(VecPlaceArray(x2,px+n));
  CHKERRQ(VecPlaceArray(x3,px+2*n));
  CHKERRQ(VecSet(x1,1.0));
  CHKERRQ(VecSet(x2,0.0));
  CHKERRQ(VecSet(x3,2.0));
  CHKERRQ(VecResetArray(x1));
  CHKERRQ(VecResetArray(x2));
  CHKERRQ(VecResetArray(x3));
  CHKERRQ(VecRestoreArrayWrite(x,&px));
  CHKERRQ(VecDestroy(&x1));
  CHKERRQ(VecDestroy(&x2));
  CHKERRQ(VecDestroy(&x3));

  CHKERRQ(VecView(x,NULL));
  CHKERRQ(VecReciprocal(x));
  CHKERRQ(VecView(x,NULL));

  CHKERRQ(VecDestroy(&x));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

     testset:
       output_file: output/ex60_1.out
       diff_args: -j
       test:
         suffix: 1
       test:
         suffix: 1_cuda
         args: -vec_type cuda
         filter: sed -e 's/seqcuda/seq/'
         requires: cuda
       test:
         suffix: 1_kokkos
         args: -vec_type kokkos
         filter: sed -e 's/seqkokkos/seq/'
         requires: kokkos_kernels
       test:
         suffix: 1_hip
         args: -vec_type hip
         filter: sed -e 's/seqhip/seq/'
         requires: hip

TEST*/
