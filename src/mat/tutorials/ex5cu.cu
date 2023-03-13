static char help[] = "Test of CUDA matrix assemble with simple matrix.\n\n";

// This a minimal example of the use of the CUDA MatAIJ metadata for assembly.
//
// The matrix must be a type 'aijcusparse' and must first be assembled on the CPU to provide the nonzero pattern.
// Next, get a pointer to a simple CSR mirror (PetscSplitCSRDataStructure) of the matrix data on
//    the GPU with MatCUSPARSEGetDeviceMatWrite().
// Then use this object to populate the matrix on the GPU with MatSetValuesDevice().
// Finally call MatAssemblyBegin/End() and the matrix is ready to use on the GPU without matrix data movement between the
//    host and GPU.

#include <petscconf.h>
#include <petscmat.h>
#include <petscdevice_cuda.h>
#include <assert.h>

#include <petscaijdevice.h>

__global__ void assemble_on_gpu(PetscSplitCSRDataStructure d_mat, PetscInt start, PetscInt end, PetscInt N, PetscMPIInt rank)
{
  const PetscInt inc = blockDim.x, my0 = threadIdx.x;
  PetscInt       i;
  int            err;

  for (i = start + my0; i < end + 1; i += inc) {
    PetscInt    js[] = {i - 1, i}, nn = (i == N) ? 1 : 2; // negative indices are ignored but >= N are not, so clip end
    PetscScalar values[] = {1, 1, 1, 1};
    err                  = MatSetValuesDevice(d_mat, nn, js, nn, js, values, ADD_VALUES);
    if (err) assert(0);
  }
}

PetscErrorCode assemble_on_cpu(Mat A, PetscInt start, PetscInt end, PetscInt N, PetscMPIInt)
{
  PetscFunctionBeginUser;
  for (PetscInt i = start; i < end + 1; i++) {
    PetscInt    js[] = {i - 1, i}, nn = (i == N) ? 1 : 2;
    PetscScalar values[] = {1, 1, 1, 1};
    PetscCall(MatSetValues(A, nn, js, nn, js, values, ADD_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                        A;
  PetscInt                   N = 11, nz = 3, Istart, Iend, num_threads = 128;
  PetscSplitCSRDataStructure d_mat;
  PetscLogEvent              event;
  PetscMPIInt                rank, size;
  PetscBool                  testmpiseq = PETSC_FALSE;
  Vec                        x, y;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_threads", &num_threads, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz_row", &nz, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-testmpiseq", &testmpiseq, NULL));
  if (nz < 3) nz = 3;
  if (nz > N + 1) nz = N + 1;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscLogEventRegister("GPU operator", MAT_CLASSID, &event));
  PetscCall(MatCreateAIJCUSPARSE(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, nz, NULL, nz - 1, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  /* current GPU assembly code does not support offprocessor values insertion */
  PetscCall(assemble_on_cpu(A, Istart, Iend, N, rank));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // test
  PetscCall(VecSet(x, 1.0));
  PetscCall(MatMult(A, x, y));
  PetscCall(VecViewFromOptions(y, NULL, "-ex5_vec_view"));

  if (testmpiseq && size == 1) {
    PetscCall(MatConvert(A, MATSEQAIJ, MAT_INPLACE_MATRIX, &A));
    PetscCall(MatConvert(A, MATMPIAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));
  }
  PetscCall(PetscLogEventBegin(event, 0, 0, 0, 0));
  PetscCall(MatCUSPARSEGetDeviceMatWrite(A, &d_mat));
  assemble_on_gpu<<<1, num_threads>>>(d_mat, Istart, Iend, N, rank);
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(event, 0, 0, 0, 0));

  // test
  PetscCall(VecSet(x, 1.0));
  PetscCall(MatMult(A, x, y));
  PetscCall(VecViewFromOptions(y, NULL, "-ex5_vec_view"));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: cuda !defined(PETSC_HAVE_CUDA_CLANG)

   test:
      suffix: 0
      diff_args: -j
      args: -n 11 -ex5_vec_view
      nsize: 1

   test:
      suffix: 1
      diff_args: -j
      args: -n 11 -ex5_vec_view
      nsize: 2

   test:
      suffix: 2
      diff_args: -j
      args: -n 11 -testmpiseq -ex5_vec_view
      nsize: 1

TEST*/
