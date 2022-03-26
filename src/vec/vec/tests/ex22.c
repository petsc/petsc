
static char help[] = "Scatters from a parallel vector to a parallel vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5,N,i;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  N    = size*n;
  PetscCall(VecCreate(PETSC_COMM_WORLD,&y));
  PetscCall(VecSetSizes(y,n,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(y));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,n,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));

  /* create two index sets */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&is1));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,(n*(rank+1))%N,1,&is2));

  /* fill local part of parallel vector x */
  value = (PetscScalar)(rank+1);
  for (i=n*rank; i<n*(rank+1); i++) {
    PetscCall(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(VecSet(y,zero));

  PetscCall(VecScatterCreate(x,is1,y,is2,&ctx));
  PetscCall(VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 4
      output_file: output/ex22_1.out
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
