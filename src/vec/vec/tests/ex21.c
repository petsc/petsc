
static char help[] = "Tests VecMax() with index.\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5,idx;
  PetscReal      value,value2;
  Vec            x;
  PetscScalar    one = 1.0;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* create vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));

  PetscCall(VecSet(x,one));
  PetscCall(VecSetValue(x,0,0.0,INSERT_VALUES));
  PetscCall(VecSetValue(x,n-1,2.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecMax(x,&idx,&value));
  PetscCall(VecMax(x,NULL,&value2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Maximum value %g index %" PetscInt_FMT " (no index %g)\n",(double)value,idx,(double)value2));
  PetscCall(VecMin(x,&idx,&value));
  PetscCall(VecMin(x,NULL,&value2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Minimum value %g index %" PetscInt_FMT " (no index %g)\n",(double)value,idx,(double)value2));

  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      diff_args: -j
      filter: grep -v type | grep -v " MPI process" | grep -v Process
      output_file: output/ex21_1.out

      test:
         suffix: 1
         args: -vec_type {{seq mpi}}

      test:
         requires: cuda
         suffix: 1_cuda
         args: -vec_type {{cuda mpicuda}}

      test:
         requires: kokkos_kernels
         suffix: 1_kokkos
         args: -vec_type {{kokkos mpikokkos}}

      test:
         requires: hip
         suffix: 1_hip
         args: -vec_type {{hip mpihip}}

   testset:
      diff_args: -j
      filter: grep -v type
      output_file: output/ex21_2.out
      nsize: 2

      test:
         suffix: 2

      test:
         requires: cuda
         suffix: 2_cuda
         args: -vec_type cuda

      test:
         requires: kokkos_kernels
         suffix: 2_kokkos
         args: -vec_type kokkos

      test:
         requires: hip
         suffix: 2_hip
         args: -vec_type hip

TEST*/
