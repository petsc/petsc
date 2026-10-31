static char help[] = "Benchmarking device kernel launch time\n";
/*
  Running example on Summit at OLCF:
  # run with total 1 resource set (RS) (-n1), 1 RS per node (-r1), 1 MPI rank (-a1), 7 cores (-c7) and 1 GPU (-g1) per RS
  $ jsrun -n1 -a1 -c7 -g1 -r1  ./ex1k
  Average asynchronous device kernel launch time = 4.86 microseconds
  Average synchronous device kernel launch time  = 12.83 microseconds

  Frontier@OLCF
  $ srun -n1 -c32 --cpu-bind=threads --gpus-per-node=8 --gpu-bind=closest ./ex1k
  Average asynchronous device kernel launch time = 1.88 microseconds
  Average synchronous device kernel launch time  = 7.78 microseconds

  Aurora@ALCF
  $ mpiexec -n 1 ./ex1k
  Average asynchronous device kernel launch time = 3.34 microseconds
  Average synchronous device kernel launch time  = 6.24 microseconds

  Perlmutter@NERSC
  $ srun -n 1 --gpus-per-task=1 ./ex1k
  Average asynchronous device kernel launch time = 2.31 microseconds
  Average synchronous device kernel launch time  = 7.13 microseconds
*/

#include <petsc_kokkos.hpp>
#include <petscdevice.h>

int main(int argc, char **argv)
{
  PetscInt           i, n = 100000, N = 256;
  PetscLogDouble     tstart, tend, time;
  PetscDeviceContext saved = nullptr, work = nullptr;
  PetscBool          reset = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_stream_reset", &reset, NULL));
  if (reset) {
    PetscCall(PetscDeviceContextGetCurrentContext(&saved));
    PetscCall(PetscDeviceContextDuplicate(saved, &work));
    PetscCall(PetscDeviceContextSetStreamType(work, PETSC_STREAM_NONBLOCKING));
    PetscCall(PetscDeviceContextSetUp(work));
    PetscCall(PetscDeviceContextSetCurrentContext(work));
    // Reconfigure the selected context before Kokkos captures its stream.
    PetscCall(PetscDeviceContextSetStreamType(work, PETSC_STREAM_DEFAULT));
    PetscCall(PetscDeviceContextSetUp(work));
  }
  PetscCall(PetscKokkosInitializeCheck());
  {
    Kokkos::DefaultExecutionSpace                      exec = PetscGetKokkosExecutionSpace();
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(exec, 0, N);

    if (reset) {
#if defined(KOKKOS_ENABLE_CUDA)
      cudaStream_t *stream;

      PetscCall(PetscDeviceContextGetStreamHandle(work, (void **)&stream));
      PetscCheck(exec.cuda_stream() == *stream, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos must use the reconfigured context's stream");
#elif defined(KOKKOS_ENABLE_HIP)
      hipStream_t *stream;

      PetscCall(PetscDeviceContextGetStreamHandle(work, (void **)&stream));
      PetscCheck(exec.hip_stream() == *stream, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos must use the reconfigured context's stream");
#endif
    }
    PetscCallCXX(exec.fence()); // Initialize device runtime to get more accurate timing below
    // Launch a sequence of kernels asynchronously. Previous launched kernels do not need to be completed before launching a new one
    PetscCall(PetscTime(&tstart));
    for (i = 0; i < n; i++) PetscCallCXX(Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const PetscInt &i){}));
    PetscCall(PetscTime(&tend));
    PetscCallCXX(exec.fence());
    time = (tend - tstart) * 1e6 / n;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Average asynchronous device kernel launch time = %.2f microseconds\n", time));

    // Launch a sequence of kernels synchronously. Only launch a new kernel after the one before it has been completed
    PetscCall(PetscTime(&tstart));
    for (i = 0; i < n; i++) {
      PetscCallCXX(Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const PetscInt &i){}));
      PetscCallCXX(exec.fence());
    }
    PetscCall(PetscTime(&tend));
    time = (tend - tstart) * 1e6 / n;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Average synchronous device kernel launch time  = %.2f microseconds\n", time));
  }

  if (reset) {
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&work));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    requires: kokkos
    args: -n 2
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

  test:
    suffix: cuda_stream_reset
    requires: kokkos cuda
    args: -n 2 -device_enable lazy -test_stream_reset
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

  test:
    suffix: hip_stream_reset
    requires: kokkos hip
    args: -n 2 -device_enable lazy -test_stream_reset
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

TEST*/
