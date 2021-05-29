static char help[] = "Tests VecPlaceArray() and VecReciprocal().\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n=5,bs;
  PetscBool      iscuda,iskokkos,iship;
  Vec            x,x1,x2,x3;
  PetscScalar    *px;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create vector of length 2*n */
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,3*n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* create two vectors of length n without array */
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQCUDA,&iscuda);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQKOKKOS,&iskokkos);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQHIP,&iship);CHKERRQ(ierr);
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
  if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
    ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1);CHKERRQ(ierr);
    ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2);CHKERRQ(ierr);
    ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3);CHKERRQ(ierr);
#endif
  } else if (iskokkos) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    ierr = VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1);CHKERRQ(ierr);
    ierr = VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2);CHKERRQ(ierr);
    ierr = VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3);CHKERRQ(ierr);
#endif
  } else if (iship) {
#if defined(PETSC_HAVE_HIP)
    ierr = VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1);CHKERRQ(ierr);
    ierr = VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2);CHKERRQ(ierr);
    ierr = VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3);CHKERRQ(ierr);
#endif
  } else {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3);CHKERRQ(ierr);
  }

  ierr = VecGetArrayWrite(x,&px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+n);CHKERRQ(ierr);
  ierr = VecPlaceArray(x3,px+2*n);CHKERRQ(ierr);
  ierr = VecSet(x1,1.0);CHKERRQ(ierr);
  ierr = VecSet(x2,0.0);CHKERRQ(ierr);
  ierr = VecSet(x3,2.0);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(x3);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(x,&px);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&x3);CHKERRQ(ierr);

  ierr = VecView(x,NULL);CHKERRQ(ierr);
  ierr = VecReciprocal(x);CHKERRQ(ierr);
  ierr = VecView(x,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
         TODO: broken
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
