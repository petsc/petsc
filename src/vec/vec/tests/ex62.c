static char help[] = "Test Vector conversions.\n\n";

#include <petscvec.h>

#define LEN 32

int main(int argc, char **argv)
{
  PetscMPIInt size;
  PetscInt    n = LEN;
  PetscInt    i;
  PetscScalar array[LEN];
  Vec         x, y, z;
  PetscReal   nrm, ans;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* create x with an existing array */
  for (i = 0; i < n; i++) array[i] = 1.0;

  if (size == 1) PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, array, &x));
  else PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, array, &x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecScale(x, 5.0)); // x = {5,..}

  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecSetSizes(y, n, PETSC_DECIDE));
  PetscCall(VecSet(y, 2.0)); // y = {2,..}

  PetscCall(VecAXPY(x, -2.0, y)); // x += -2.0*y
  PetscCall(VecNorm(x, NORM_2, &nrm));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &z));
  PetscCall(VecSetType(z, VECSTANDARD));
  PetscCall(VecSetSizes(z, n, PETSC_DECIDE));
  PetscCall(VecSet(z, 1.0));
  PetscCall(VecNorm(z, NORM_2, &ans));
  PetscCheck(PetscAbs(nrm - ans) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Diff is too big, %g\n", (double)nrm);
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 1
      output_file: output/empty.out
      test:
        args: -vec_type {{seq mpi standard}}
        suffix: standard

      test:
        requires: cuda
        args: -vec_type {{seqcuda mpicuda cuda}}
        suffix: cuda
      test:
        requires: hip
        args: -vec_type {{seqhip mpihip hip}}
        suffix: hip
      test:
        requires: viennacl
        args: -vec_type {{seqviennacl mpiviennacl viennacl}}
        suffix: viennacl
      test:
        requires: kokkos_kernels
        args: -vec_type {{seqkokkos mpikokkos kokkos}}
        suffix: kokkos

   testset:
      nsize: 2
      output_file: output/empty.out
      test:
        args: -vec_type {{mpi standard}}
        suffix: standard_2
      test:
        requires: cuda
        args: -vec_type {{mpicuda cuda}}
        suffix: cuda_2
      test:
        requires: hip
        args: -vec_type {{mpihip hip}}
        suffix: hip_2
      test:
        requires: viennacl
        args: -vec_type {{mpiviennacl viennacl}}
        suffix: viennacl_2
      test:
        requires: kokkos_kernels
        args: -vec_type {{mpikokkos kokkos}}
        suffix: kokkos_2

TEST*/
