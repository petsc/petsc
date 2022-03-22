
static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
  Using a blocked send and a strided receive.\n\n";

/*
        0 1 2 3 | 4 5 6 7 ||  8 9 10 11

     Scatter first and third block to first processor and
     second and third block to second processor
*/

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       i,blocks[2],nlocal;
  PetscMPIInt    size,rank;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;
  PetscViewer    subviewer;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheck(size == 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 2 processors");

  /* create two vectors */
  if (rank == 0) nlocal = 8;
  else nlocal = 4;
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,nlocal,12));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecCreate(PETSC_COMM_SELF,&y));
  PetscCall(VecSetSizes(y,8,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(y));

  /* create two index sets */
  if (rank == 0) {
    blocks[0] = 0; blocks[1] = 2;
  } else {
    blocks[0] = 1; blocks[1] = 2;
  }
  PetscCall(ISCreateBlock(PETSC_COMM_SELF,4,2,blocks,PETSC_COPY_VALUES,&is1));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,8,0,1,&is2));

  for (i=0; i<12; i++) {
    value = i;
    PetscCall(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(VecScatterCreate(x,is1,y,is2,&ctx));
  PetscCall(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&subviewer));
  PetscCall(VecView(y,subviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&subviewer));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 2
      output_file: output/ex23_1.out
      filter: grep -v "  type:"
      diff_args: -j
      test:
        suffix: standard
        args: -vec_type standard
      test:
        requires: cuda
        suffix: cuda
        args: -vec_type cuda
      test:
        requires: viennacl
        suffix:  viennacl
        args: -vec_type viennacl
      test:
        requires: !sycl kokkos_kernels
        suffix: kokkos
        args: -vec_type kokkos
      test:
        requires: hip
        suffix: hip
        args: -vec_type hip

TEST*/
