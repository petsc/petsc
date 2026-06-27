static char help[] = "Solves a tridiagonal linear system with CUDA managed memory.\n\n";

#include <petscdevice_cuda.h>
#include <petscksp.h>

// adapted from ksp/tutorials/ex23.c

int main(int argc, char **args)
{
  Vec          x, b, u;
  Mat          A;
  KSP          ksp;
  PetscReal    norm, tol = 1000. * PETSC_MACHINE_EPSILON;
  PetscInt     i, n, N = 32, col[3], its, rstart, rend;
  PetscScalar  value[3];
  PetscScalar *xarray, *uarray;
  PetscScalar *array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetType(A, MATAIJCUSPARSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetLocalSize(A, &n, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend)); // Same row/column range for this test

  if (!rstart) {
    rstart   = 1;
    i        = 0;
    col[0]   = 0;
    col[1]   = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  }
  if (rend == N) {
    rend     = N - 1;
    i        = N - 1;
    col[0]   = N - 2;
    col[1]   = N - 1;
    value[0] = -1.0;
    value[1] = 2.0;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  }

  value[0] = -1.0;
  value[1] = 2.0;
  value[2] = -1.0;
  for (i = rstart; i < rend; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCallCUDA(cudaMallocManaged((void **)&xarray, n * sizeof(PetscScalar), cudaMemAttachGlobal));
  PetscCallCUDA(cudaMallocManaged((void **)&uarray, n * sizeof(PetscScalar), cudaMemAttachGlobal));

  // Use a managed array as a host array parameter, it will be a host array in petsc's view;
  // similarily, use it as a device array parameter, it will be a device array.
  // One can use the same array on both host and device parameters.
  PetscCall(VecCreateMPICUDAWithArrays(PETSC_COMM_WORLD, 1, n, N, xarray, xarray, &x));
  PetscCall(VecCreateMPICUDAWithArrays(PETSC_COMM_WORLD, 1, n, N, uarray, uarray, &u));

  PetscCall(MatCreateVecs(A, NULL, &b));
  PetscCall(VecSet(u, 1.0)); // Set u on device

  PetscCall(VecGetArray(u, &array)); // Do a cudaMemcpyDeviceToHost from uarray to uarray!
  PetscCheck(array == uarray && array[0] == 1.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "u is out of sync");
  array[0] = 2.0;
  PetscCall(VecRestoreArray(u, &array));
  PetscCall(MatMult(A, u, b));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));

  PetscCallCUDA(cudaFree(xarray));
  PetscCallCUDA(cudaFree(uarray));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: cuda !complex !single

  test:
    nsize: {{1 2}}
    output_file: output/empty.out

TEST*/
