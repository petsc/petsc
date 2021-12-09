
static char help[] = "Tests VecMax() with index.\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,idx;
  PetscReal      value,value2;
  Vec            x;
  PetscScalar    one = 1.0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* create vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSetValue(x,0,0.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(x,n-1,2.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecMax(x,&idx,&value);CHKERRQ(ierr);
  ierr = VecMax(x,NULL,&value2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Maximum value %g index %" PetscInt_FMT " (no index %g)\n",(double)value,idx,(double)value2);CHKERRQ(ierr);
  ierr = VecMin(x,&idx,&value);CHKERRQ(ierr);
  ierr = VecMin(x,NULL,&value2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Minimum value %g index %" PetscInt_FMT " (no index %g)\n",(double)value,idx,(double)value2);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      diff_args: -j
      filter: grep -v type | grep -v "MPI processes" | grep -v Process
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
