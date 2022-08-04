static char help[] = "Benchmarking memory bandwidth with VecAXPY() on parallel vectors\n";
/*
  Usage:
   mpirun -n <np> ./ex1k -vec_type <device vector type>
     -n  <n>  # number of data points of vector sizes from 128, 256, 512 and up. Maxima and default is 23.
     -m  <m>  # run each VecAXPY() m times to get the average time, default is 1000.

  Example:

  Running on Crusher at OLCF:
  # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex1k -vec_type kokkos
*/

#include <petscvec.h>
#include <petscdevice.h>

#if defined(PETSC_HAVE_CUDA)
  #define SyncDevice() PetscCallCUDA(cudaDeviceSynchronize())
#elif defined(PETSC_HAVE_HIP)
  #define SyncDevice() PetscCallHIP(hipDeviceSynchronize())
#elif defined(PETSC_HAVE_KOKKOS)
  #include <Kokkos_Core.hpp>
  #define SyncDevice() Kokkos::fence()
#else
  #define SyncDevice() 0
#endif

int main(int argc,char **argv)
{
  PetscInt        i,k,N,n,m = 1000,nsamples;
  PetscLogDouble  tstart,tend,time;
  Vec             x,y;
  PetscScalar     alpha=3.14;
  PetscLogDouble  bandwidth;
  PetscMPIInt     size;
  PetscInt        Ns[] = { /* Use explicit sizes so that one can add sizes very close to 2^31 */
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
    134217728,
    268435456,
    536870912
  };
  n = nsamples = sizeof(Ns)/sizeof(Ns[0]);

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));  /* Up to vectors of local size 2^{n+6} */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));  /* Run each VecAXPY() m times */

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vector size (N)   Time (us)   Bandwidth (GB/s)\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"----------------------------------------------\n"));

  nsamples = PetscMin(nsamples,n);
  for (k=0; k<nsamples; k++) {
    N = Ns[k];
    PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetSizes(x,N,PETSC_DECIDE));
    PetscCall(VecSetUp(x));
    PetscCall(VecDuplicate(x,&y));
    PetscCall(VecSet(x,2.5));
    PetscCall(VecSet(y,4.0));

    /* Warm-up */
    for (i=0; i<4; i++) PetscCall(VecAXPY(x,alpha,y));
    SyncDevice();
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

    PetscCall(PetscTime(&tstart));
    for (i=0; i<m; i++) PetscCall(VecAXPY(x,alpha,y));
    SyncDevice();
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    PetscCall(PetscTime(&tend));
    time      = (tend-tstart)*1e6/m;
    bandwidth = 3*N*size*sizeof(PetscScalar)/time*1e-3; /* read x, y and write y */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%12" PetscInt_FMT ", %12.4f, %8.2f\n",N,time,bandwidth));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: kokkos_kernels

  test:
    args: -n 2 -m 2 -vec_type kokkos
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

TEST*/
