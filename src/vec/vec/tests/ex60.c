static char help[] = "Tests VecPlaceArray() and VecReciprocal().\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n=5,bs;
  PetscBool      iscuda,iskokkos,iship;
  Vec            x,x1,x2,x3;
  PetscScalar    *px;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* create vector of length 2*n */
  PetscCall(VecCreate(PETSC_COMM_SELF,&x));
  PetscCall(VecSetSizes(x,3*n,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));

  /* create two vectors of length n without array */
  PetscCall(PetscObjectTypeCompare((PetscObject)x,VECSEQCUDA,&iscuda));
  PetscCall(PetscObjectTypeCompare((PetscObject)x,VECSEQKOKKOS,&iskokkos));
  PetscCall(PetscObjectTypeCompare((PetscObject)x,VECSEQHIP,&iship));
  PetscCall(VecGetBlockSize(x,&bs));
  if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    PetscCall(VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    PetscCall(VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
#endif
  } else if (iskokkos) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    PetscCall(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    PetscCall(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    PetscCall(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
#endif
  } else if (iship) {
#if defined(PETSC_HAVE_HIP)
    PetscCall(VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    PetscCall(VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    PetscCall(VecCreateSeqHIPWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
#endif
  } else {
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x3));
  }

  PetscCall(VecGetArrayWrite(x,&px));
  PetscCall(VecPlaceArray(x1,px));
  PetscCall(VecPlaceArray(x2,px+n));
  PetscCall(VecPlaceArray(x3,px+2*n));
  PetscCall(VecSet(x1,1.0));
  PetscCall(VecSet(x2,0.0));
  PetscCall(VecSet(x3,2.0));
  PetscCall(VecResetArray(x1));
  PetscCall(VecResetArray(x2));
  PetscCall(VecResetArray(x3));
  PetscCall(VecRestoreArrayWrite(x,&px));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&x3));

  PetscCall(VecView(x,NULL));
  PetscCall(VecReciprocal(x));
  PetscCall(VecView(x,NULL));

  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
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
