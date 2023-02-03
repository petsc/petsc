static char help[] = "Test of Kokkos matrix assemble with 1D Laplacian. Kokkos version of ex5cu \n\n";

#include <petscconf.h>
#include <petscmat.h>

/*
    Include Kokkos files
*/
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

#include <petscaijdevice.h>

int main(int argc, char **argv)
{
  Mat                        A;
  PetscInt                   N = 11, nz = 3, Istart, Iend, num_threads = 128;
  PetscSplitCSRDataStructure d_mat;
  PetscLogEvent              event;
  Vec                        x, y;
  PetscMPIInt                rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz_row", &nz, NULL)); // for debugging, will be wrong if nz<3
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_threads", &num_threads, NULL));
  if (nz > N + 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "warning decreasing nz\n"));
    nz = N + 1;
  }
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscLogEventRegister("GPU operator", MAT_CLASSID, &event));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetType(A, MATAIJKOKKOS));
  PetscCall(MatSeqAIJSetPreallocation(A, nz, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, nz, NULL, nz - 1, NULL));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  // assemble end on CPU. We are not assembling redundant here, and ignoring off proc entries, but we could
  for (int i = Istart; i < Iend + 1; i++) {
    PetscScalar values[] = {1, 1, 1, 1};
    PetscInt    js[] = {i - 1, i}, nn = (i == N) ? 1 : 2; // negative indices are ignored but >= N are not, so clip end
    PetscCall(MatSetValues(A, nn, js, nn, js, values, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // test Kokkos
  PetscCall(VecSet(x, 1.0));
  PetscCall(MatMult(A, x, y));
  PetscCall(VecViewFromOptions(y, NULL, "-ex5_vec_view"));

  // assemble on GPU
  if (Iend < N) Iend++; // elements, ignore off processor entries so do redundant
  PetscCall(PetscLogEventBegin(event, 0, 0, 0, 0));
  PetscCall(MatKokkosGetDeviceMatWrite(A, &d_mat));
  Kokkos::fence();
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(Istart, Iend + 1), KOKKOS_LAMBDA(int i) {
      PetscScalar values[] = {1, 1, 1, 1};
      PetscInt    js[] = {i - 1, i}, nn = (i == N) ? 1 : 2;
      static_cast<void>(MatSetValuesDevice(d_mat, nn, js, nn, js, values, ADD_VALUES));
    });
  Kokkos::fence();
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(VecSet(x, 1.0));
  PetscCall(MatMult(A, x, y));
  PetscCall(VecViewFromOptions(y, NULL, "-ex5_vec_view"));
  PetscCall(PetscLogEventEnd(event, 0, 0, 0, 0));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
#else
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_COR, "Kokkos kernels required");
#endif
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: kokkos_kernels !defined(PETSC_HAVE_CUDA_CLANG)

   test:
     suffix: 0
     requires: kokkos_kernels double !complex !single
     args: -n 11 -ex5_vec_view
     nsize:  2

TEST*/
