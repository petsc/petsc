static char help[] = "Benchmarking VecMDot() or VecMAXPY()\n";
/*
  Usage:
   mpirun -n <np> ./ex2k -vec_type <vector type>
     -n  <n>  # number of data points of vector sizes from 128, 256, 512 and up. Maxima and default is 23.
     -m  <m>  # run each VecMDot() m times to get the average time, default is 100.
     -test_name <VecMDot or VecMAXPY>  # test to run, by default it is VecMDot
     -output_bw <bool> # output bandwidth instead of time

  Example:

  Running on Frontier at OLCF:
  # run with 1 mpi rank (-n1), 32 CPUs (-c32)
  $ srun -n1 -c32 --gpus-per-node=8 --gpu-bind=closest ./ex2k -vec_type kokkos
*/

#include <petscvec.h>
#include <petscdevice.h>

int main(int argc, char **argv)
{
  PetscInt           i, j, k, M, N, mcount, its = 100, nsamples, ncount, maxN;
  PetscLogDouble     tstart, tend, times[8], fom; // figure of merit
  Vec                x, *ys;
  PetscScalar       *vals;
  PetscMPIInt        size;
  PetscDeviceContext dctx;
  char               testName[64] = "VecMDot"; // By default, test VecMDot
  PetscBool          testMDot, testMAXPY;
  PetscBool          outputBW = PETSC_FALSE; // output bandwidth instead of time
  PetscRandom        rnd;
  PetscLogStage      stage1;
  // clang-format off
  // Try vectors of these (local) sizes. The max is very close to 2^31
  PetscInt  Ms[]  = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                     65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                     8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912};
  PetscInt  Ns[] = {1, 3, 8, 30}; // try this number of y vectors in VecMDot
  // clang-format on

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));

  mcount = sizeof(Ms) / sizeof(Ms[0]); // length of Ms[]
  ncount = sizeof(Ns) / sizeof(Ns[0]); // length of Ns[]
  maxN   = Ns[ncount - 1];             // at most this many y vectors

  nsamples = mcount;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &mcount, NULL)); // Up to vectors of local size 2^{mcount+6}
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &its, NULL));    // Run each VecMDot() its times
  PetscCall(PetscOptionsGetString(NULL, NULL, "-test_name", testName, sizeof(testName), NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-output_bw", &outputBW, NULL));
  PetscCall(PetscStrncmp(testName, "VecMDot", sizeof(testName), &testMDot));
  PetscCall(PetscStrncmp(testName, "VecMAXPY", sizeof(testName), &testMAXPY));
  PetscCheck(testMDot || testMAXPY, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unsupported test name: %s", testName);
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscMalloc1(maxN, &vals));
  for (j = 0; j < maxN; j++) vals[j] = 3.14 + j; // same across all processes

  PetscCall(PetscLogStageRegister("Profiling", &stage1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Vector(N)   "));
  for (j = 0; j < ncount; j++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   %s-%" PetscInt_FMT " ", testName, Ns[j]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, outputBW ? " (GB/s)\n" : " (us)\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------------------------------\n"));

  nsamples = PetscMin(nsamples, mcount);
  for (k = 0; k < nsamples; k++) { // for vector (local) size M
    M = Ms[k];
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetSizes(x, M, PETSC_DECIDE));
    PetscCall(VecSetUp(x));
    PetscCall(VecDuplicateVecs(x, maxN, &ys));
    PetscCall(VecSetRandom(x, rnd));
    for (i = 0; i < maxN; i++) PetscCall(VecSetRandom(ys[i], rnd));

    for (j = 0; j < ncount; j++) { // try N y vectors
      // Warm-up
      N = Ns[j];
      for (i = 0; i < 2; i++) {
        if (testMDot) PetscCall(VecMDot(x, N, ys, vals));
        else if (testMAXPY) PetscCall(VecMAXPY(x, N, vals, ys));
      }
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

      PetscCall(PetscLogStagePush(stage1)); // use LogStage so that -log_view result will be clearer
      PetscCall(PetscTime(&tstart));
      for (i = 0; i < its; i++) {
        if (testMDot) PetscCall(VecMDot(x, N, ys, vals));
        else if (testMAXPY) PetscCall(VecMAXPY(x, N, vals, ys));
      }
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&tend));
      times[j] = (tend - tstart) * 1e6 / its;
      PetscCall(PetscLogStagePop());
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12" PetscInt_FMT, M));
    for (j = 0; j < ncount; j++) {
      N = Ns[j];
      if (outputBW) {
        // Read N y vectors and x vector of size M, and then write vals[] of size N
        PetscLogDouble bytes = (M * (N + 1.0) + N) * sizeof(PetscScalar);
        fom                  = (bytes / times[j]) * 1e-3;
      } else {
        fom = times[j];
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.1f ", fom));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroyVecs(maxN, &ys));
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
