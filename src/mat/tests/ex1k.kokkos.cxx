static char help[] = "Benchmarking MatMult() with AIJ and its subclass matrix types\n";

/*
Usage:
  mpirun -n <np> ./ex1k
    -f <file>        : input petsc matrix binary file; one can convert a file from MatrixMarket using mat/tests/ex72.c
    -mat_type <type> : aij or its subclass. Default is aij.
    -n <num>         : run MatMult() this many times and report average time. Default is 500.

Notes:
  It uses CPU-timer to measure the time.

Examples:
  On OLCF Summit (with GPU-aware MPI)
    # 6 MPI ranks:
    # 6 resource sets (-n 6), 1 MPI rank per RS (-a 1), 7 CPU cores per RS (-c 7), and 1 GPU per RS (-g 1), 6 RSs per node (-r 6)
    jsrun --smpiargs "-gpu" -n 6 -a 1 -c 7 -g 1 -r 6 ./ex1k -f 1138_bus.aij -mat_type aijcusparse

    # 1 MPI rank
    jsrun --smpiargs "-gpu" -n 1 -a 1 -c 7 -g 1 -r 1 ./ex1k -f 1138_bus.aij -mat_type aijcusparse

  On OLCF Crusher:
    # 1 MPI rank
    # run with 1 node (-N1), 1 mpi rank (-n1), 2 hardware threads per rank (-c2)
    srun -N1 -n1 -c2 --gpus-per-node=8 --gpu-bind=closest ./ex1k -f HV15R.aij -mat_type aijkokkos

    # 8 MPI ranks
    srun -N1 -n8 -c2 --gpus-per-node=8 --gpu-bind=closest ./ex1k -f HV15R.aij -mat_type aijkokkos
*/
#include <petscmat.h>
#include <petscdevice.h>

#if defined(PETSC_HAVE_CUDA)
  #include <petscdevice_cuda.h>
  #define SyncDevice() PetscCallCUDA(cudaDeviceSynchronize())
#elif defined(PETSC_HAVE_HIP)
  #include <petscdevice_hip.h>
  #define SyncDevice() PetscCallHIP(hipDeviceSynchronize())
#elif defined(PETSC_HAVE_KOKKOS)
  #include <Kokkos_Core.hpp>
  #define SyncDevice() Kokkos::fence()
#else
  #define SyncDevice()
#endif

int main(int argc, char **args)
{
  Mat            A, A2;
  Vec            x, y, x2, y2;
  PetscViewer    fd;
  char           matfile[PETSC_MAX_PATH_LEN];
  char           mattype[64];
  PetscBool      flg;
  PetscLogStage  stage;
  PetscInt       i, n = 500, nskip = 5, M, N;
  MatInfo        info;
  PetscLogDouble tstart = 0, tend = 0, avgTime;
  PetscRandom    rctx;
  PetscReal      norm;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Read options -n */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* Load the matrix from a binary file */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", matfile, PETSC_MAX_PATH_LEN, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a petsc matrix binary file with the -f option");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_type", mattype, sizeof(mattype), &flg));
  if (!flg) PetscCall(PetscStrncpy(mattype, MATAIJ, sizeof(mattype)));

  /* Read the matrix file to A2 */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, matfile, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A2));
  PetscCall(MatSetType(A2, MATAIJ));
  PetscCall(MatLoad(A2, fd));
  PetscCall(MatCreateVecs(A2, &x2, &y2));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatGetSize(A2, &M, &N));
  PetscCall(MatGetInfo(A2, MAT_GLOBAL_SUM, &info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Input matrix %s: %" PetscInt_FMT " x %" PetscInt_FMT "; %lld nonzeros; %.1f per row\n", matfile, M, N, (long long)info.nz_used, (double)info.nz_used / (double)M));

  /* Copy A2 to A and convert A to the specified type */
  PetscCall(MatDuplicate(A2, MAT_COPY_VALUES, &A));
  PetscCall(MatConvert(A, mattype, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatCreateVecs(A, &x, &y));

  /* Init x, x2 with the same value */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(VecSetRandom(x2, rctx));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecCopy(x2, x));

  /* Compute the reference y2 = A2 x2 */
  PetscCall(MatMult(A2, x2, y2));

  /* Measure y = Ax */
  PetscCall(PetscLogStageRegister("MatMult", &stage));
  for (i = 0; i < n + nskip; i++) {
    if (i == nskip) {
      SyncDevice();
      PetscCall(PetscLogStagePush(stage));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&tstart));
    }
    PetscCall(MatMult(A, x, y));
  }
  SyncDevice();
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&tend));
  avgTime = (tend - tstart) * 1e6 / n; /* microseconds */
  PetscCall(PetscLogStagePop());

  /* Validate y against y2 */
  PetscCall(VecAYPX(y2, -1, y));
  PetscCall(VecNorm(y2, NORM_2, &norm));
  PetscCheck(norm < 1e-6, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMult() error with norm %g", (double)norm);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMult() average time (us) with %d MPI ranks = %8.2f\n", size, avgTime));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&A2));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&y2));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -n 2 -f ${DATAFILESPATH}/matrices/small
    nsize: 1
    filter: grep "DOES_NOT_EXIST"
    output_file: output/empty.out
    requires: !complex double !single kokkos_kernels

    test:
      suffix: 1
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: 2
      args: -mat_type aijkokkos

    test:
      suffix: 3
      requires: hip
      args: -mat_type aijhipsparse

TEST*/
