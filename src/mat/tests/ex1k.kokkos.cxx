static char help[] = "Benchmarking MatMult() with AIJ and its subclass matrix types\n";

/*
Usage:
  mpirun -n <np> ./ex256
    -f <file>        : input petsc matrix binary file; one can convert a file from MatrixMarket using mat/tests/ex72.c
    -mat_type <type> : aij or its subclass. Default is aij.
    -n <num>         : run MatMult() this many times and report average time. Default is 500.

Examples:
  On OLCF Summit (with GPU-aware MPI)
    # 6 MPI ranks:
    # 6 resource sets (-n 6), 1 MPI rank per RS (-a 1), 7 CPU cores per RS (-c 7), and 1 GPU per RS (-g 1), 6 RSs per node (-r 6)
    jsrun --smpiargs "-gpu" -n 6 -a 1 -c 7 -g 1 -r 6 ./ex256 -f 1138_bus.petsc -mat_type aijcusparse

    # 1 MPI rank
    jsrun --smpiargs "-gpu" -n 1 -a 1 -c 7 -g 1 -r 1 ./ex256 -f 1138_bus.petsc -mat_type aijcusparse

  On OLCF Crusher:
    # 1 MPI rank
    # run with 1 node (-N1), 1 mpi rank (-n1), 2 hardware threads per rank (-c2)
    srun -N1 -n1 -c2 --gpus-per-node=8 --gpu-bind=closest ./ex256 -f HV15R.aij -mat_type aijkokkos

    # 8 MPI ranks
    srun -N1 -n8 -c2 --gpus-per-node=8 --gpu-bind=closest ./ex256 -f HV15R.aij -mat_type aijkokkos
*/
#include <petscmat.h>
#include <petscdevice.h>

#if defined(PETSC_HAVE_CUDA)
  #define SyncDevice() PetscCallCUDA(cudaDeviceSynchronize())
#elif defined(PETSC_HAVE_HIP)
  #define SyncDevice() PetscCallHIP(hipDeviceSynchronize())
#elif defined(PETSC_HAVE_KOKKOS)
  #include <Kokkos_Core.hpp>
  #define SyncDevice() Kokkos::fence()
#else
  #define SyncDevice()
#endif

int main(int argc,char **args)
{
  Mat             A1,A2;
  Vec             x1,y1,x2,y2;
  PetscViewer     fd;
  char            matfile[PETSC_MAX_PATH_LEN];
  char            mattype[64];
  PetscBool       flg;
  PetscLogStage   stage;
  PetscInt        i,n=500,nskip=5,M,N;
  MatInfo         info;
  PetscLogDouble  tstart=0,tend=0,avgTime;
  PetscRandom     rctx;
  PetscReal       norm;
  PetscMPIInt     size;

  PetscInitialize(&argc,&args,(char *)0,help);
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Read options -n */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Load the matrix from a binary file */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",matfile,PETSC_MAX_PATH_LEN,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a petsc matrix binary file with the -f option");
  PetscCall(PetscOptionsGetString(NULL,NULL,"-mat_type",mattype,sizeof(mattype),&flg));
  if (!flg) PetscCall(PetscStrncpy(mattype,MATAIJ,sizeof(mattype)));

  /* Read the matrix file to A1 */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,matfile,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A1));
  PetscCall(MatSetType(A1,MATAIJ));
  PetscCall(MatLoad(A1,fd));
  PetscCall(MatCreateVecs(A1,&x1,&y1));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatGetSize(A1,&M,&N));
  PetscCall(MatGetInfo(A1,MAT_GLOBAL_SUM,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Input matrix %s: %" PetscInt_FMT " x %" PetscInt_FMT "; %lld nonzeros; %.1f per row\n",matfile,M,N,(long long)info.nz_used,(double)info.nz_used/(double)M));

  /* Copy A1 to A2 and convert A2 to the specified type */
  PetscCall(MatDuplicate(A1,MAT_COPY_VALUES,&A2));
  PetscCall(MatConvert(A2,mattype,MAT_INPLACE_MATRIX,&A2));
  PetscCall(MatCreateVecs(A2,&x2,&y2));

  /* Init x1, x2 with the same value */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  PetscCall(VecSetRandom(x1,rctx));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecCopy(x1,x2));

  /* Compute the reference y1 = A1 x1 */
  PetscCall(MatMult(A1,x1,y1));

  /* Measure y2 = A2 x2 */
  PetscCall(PetscLogStageRegister("MatMult", &stage));
  for (i=0; i<n + nskip; i++) {
    if (i == nskip) {
      SyncDevice();
      PetscCall(PetscLogStagePush(stage));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&tstart));
    }
    PetscCall(MatMult(A2,x2,y2));
  }
  SyncDevice();
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&tend));
  avgTime = (tend- tstart)*1e6/n; /* microseconds */
  PetscCall(PetscLogStagePop());

  /* Validate y2 against y1 */
  PetscCall(VecAYPX(y1,-1,y2));
  PetscCall(VecNorm(y1,NORM_2,&norm));
  PetscCheck(norm<1e-6,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"MatMult() error with norm %g",(double)norm);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatMult() average time (us) with %d MPI ranks = %8.2f\n",size,avgTime));

  PetscCall(MatDestroy(&A1));
  PetscCall(MatDestroy(&A2));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&y1));
  PetscCall(VecDestroy(&y2));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -f ${DATAFILESPATH}/matrices/small
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

TEST*/
