static char help[] = "Benchmarking VecMDot() or VecMAXPY()\n";
/*
  Usage:
   mpirun -n <np> ./ex2k -vec_type <vector type>
     -n  <n>  # number of data points of vector sizes from 128, 256, 512 and up. Maxima and default is 23.
     -m  <m>  # run each VecMDot() m times to get the average time, default is 100.
     -test_name <VecMDot or VecMAXPY>  # test to run, by default it is VecMDot

  Example:

  Running on Crusher at OLCF:
  # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex2k -vec_type kokkos
*/

#include <petscvec.h>
#include <petscdevice.h>

int main(int argc, char **argv)
{
  PetscInt           i, j, k, N, n, m = 100, nsamples, ny, maxys;
  PetscLogDouble     tstart, tend, times[8];
  Vec                x, *ys;
  PetscScalar       *vals;
  PetscMPIInt        size;
  PetscDeviceContext dctx;
  char               testName[64] = "VecMDot"; // By default, test VecMDot
  PetscBool          testMDot, testMAXPY;
  PetscRandom        rnd;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage1;
#endif
  // clang-format off
  // Try vectors of these (local) sizes. The max is very close to 2^31
  PetscInt  Ns[]  = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                     65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                     8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912};
  PetscInt  Ys[] = {1, 3, 8, 30}; // try this number of y vectors in VecMDot
  // clang-format on

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));

  n = nsamples = sizeof(Ns) / sizeof(Ns[0]); // length of Ns[]
  ny           = sizeof(Ys) / sizeof(Ys[0]); // length of Ys[]
  maxys        = Ys[ny - 1];                 // at most this many y vectors

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL)); // Up to vectors of local size 2^{n+6}
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL)); // Run each VecMDot() m times
  PetscCall(PetscOptionsGetString(NULL, NULL, "-test_name", testName, sizeof(testName), NULL));
  PetscCall(PetscStrncmp(testName, "VecMDot", sizeof(testName), &testMDot));
  PetscCall(PetscStrncmp(testName, "VecMAXPY", sizeof(testName), &testMAXPY));
  PetscCheck(testMDot || testMAXPY, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unsupported test name: %s", testName);
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscMalloc1(maxys, &vals));
  for (j = 0; j < maxys; j++) PetscCall(PetscRandomGetValue(rnd, &vals[j]));

  PetscCall(PetscLogStageRegister("Profiling", &stage1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Vector(N)   "));
  for (j = 0; j < ny; j++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s(nv=%" PetscInt_FMT ") ", testName, Ys[j]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " (us)\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------------------------------\n"));

  nsamples = PetscMin(nsamples, n);
  for (k = 0; k < nsamples; k++) { // for vector (local) size N
    N = Ns[k];
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetSizes(x, N, PETSC_DECIDE));
    PetscCall(VecSetUp(x));
    PetscCall(VecDuplicateVecs(x, maxys, &ys));
    PetscCall(VecSetRandom(x, rnd));
    for (i = 0; i < maxys; i++) PetscCall(VecSetRandom(ys[i], rnd));

    for (j = 0; j < ny; j++) { // try Ys[j] y vectors
      // Warm-up
      for (i = 0; i < 2; i++) {
        if (testMDot) PetscCall(VecMDot(x, Ys[j], ys, vals));
        else if (testMAXPY) PetscCall(VecMAXPY(x, Ys[j], vals, ys));
      }
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

      PetscCall(PetscLogStagePush(stage1)); // use LogStage so that -log_view result will be clearer
      PetscCall(PetscTime(&tstart));
      for (i = 0; i < m; i++) {
        if (testMDot) PetscCall(VecMDot(x, Ys[j], ys, vals));
        else if (testMAXPY) PetscCall(VecMAXPY(x, Ys[j], vals, ys));
      }
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&tend));
      times[j] = (tend - tstart) * 1e6 / m;
      PetscCall(PetscLogStagePop());
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12" PetscInt_FMT, N));
    for (j = 0; j < ny; j++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.1f ", times[j]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroyVecs(maxys, &ys));
  }

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscFree(vals));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    args: -n 2 -m 2 -test_name {{VecMDot  VecMAXPY}}
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

    test:
      suffix: standard

    test:
      requires: kokkos_kernels
      suffix: kok
      args: -vec_type kokkos

TEST*/
