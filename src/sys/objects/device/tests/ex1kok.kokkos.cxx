static char help[] = "Benchmarking device kernel launch time\n";
/*
  Running example on Summit at OLCF:
  # run with total 1 resource set (RS) (-n1), 1 RS per node (-r1), 1 MPI rank (-a1), 7 cores (-c7) and 1 GPU (-g1) per RS
  $ jsrun -n1 -a1 -c7 -g1 -r1  ./ex1kok
  Average asynchronous device kernel launch time = 4.86 microseconds
  Average synchronous device kernel launch time  = 12.83 microseconds

  Frontier@OLCF
  $ srun -n1 -c32 --cpu-bind=threads --gpus-per-node=8 --gpu-bind=closest ./ex1kok
  Average asynchronous device kernel launch time = 1.88 microseconds
  Average synchronous device kernel launch time  = 7.78 microseconds

  Aurora@ALCF
  $ mpirun -n 1 ./ex1kok
  Average asynchronous device kernel launch time = 3.34 microseconds
  Average synchronous device kernel launch time  = 6.24 microseconds

  Perlmutter@NERSC
  $ srun -n 1 --gpus-per-task=1 ./ex1kok
  Average asynchronous device kernel launch time = 2.31 microseconds
  Average synchronous device kernel launch time  = 7.13 microseconds
*/

#include <petscsys.h>
#include <petsc_kokkos.hpp>

int main(int argc, char **argv)
{
  PetscInt       i, n = 100000, N = 256;
  PetscLogDouble tstart, tend, time;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscKokkosInitializeCheck());
  {
    Kokkos::DefaultExecutionSpace                      exec = PetscGetKokkosExecutionSpace();
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(exec, 0, N);

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

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    requires: kokkos
    args: -n 2
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

TEST*/
