static char help[] = "Benchmarking MatProduct with AIJ and its subclass matrix types\n";

/*
Usage:
  mpirun -n <np> ./ex2k
    -A <filename>     : input petsc binary file for matrix A; one can convert a file from MatrixMarket using mat/tests/ex72.c
    -P <filename>     : input petsc binary file for matrix P; optional, if not given, P = A
    -mat_type  <str>  : aij or its subclass. Default is aij.
    -prod_type <str>  : AP, AtP, APt, PtAP or PAPt. Default is AP.
    -n <num>          : run MatProductNumeric() this many times and report average time. Default is 100.

Notes:
  It uses CPU-timer to measure the time.

Examples:
  On OLCF Summit (with GPU-aware MPI)
    # 6 MPI ranks:
    # 6 resource sets (-n 6), 1 MPI rank per RS (-a 1), 7 CPU cores per RS (-c 7), and 1 GPU per RS (-g 1), 6 RSs per node (-r 6)
    jsrun --smpiargs "-gpu" -n 6 -a 1 -c 7 -g 1 -r 6 ./ex2k -A cage12.aij -mat_type aijcusparse

    # 1 MPI rank
    jsrun --smpiargs "-gpu" -n 1 -a 1 -c 7 -g 1 -r 1 ./ex2k -A cage12.aij -mat_type aijcusparse

  On OLCF Crusher:
    # 1 MPI rank
    # run with 1 node (-N1), 1 mpi rank (-n1), 2 hardware threads per rank (-c2)
    srun -N1 -n1 -c2 --gpus-per-node=8 --gpu-bind=closest ./ex2k -A HV15R.aij -mat_type aijkokkos

    # 8 MPI ranks
    srun -N1 -n8 -c2 --gpus-per-node=8 --gpu-bind=closest ./ex2k -A HV15R.aij -mat_type aijkokkos
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
  Mat            A, P, C;
  Mat            A2, P2, C2; /* Shadow matrices (of MATAIJ) of A,P,C for initialization and validation */
  char           matTypeStr[64], prodTypeStr[32];
  char           fileA[PETSC_MAX_PATH_LEN], fileP[PETSC_MAX_PATH_LEN];
  PetscViewer    fdA, fdP;
  PetscBool      flg, flgA, flgP, equal = PETSC_FALSE;
  PetscLogStage  stage;
  PetscInt       i, n = 100, nskip = 2, M, N;
  MatInfo        info;
  PetscLogDouble tstart = 0, tend = 0, avgTime;
  PetscMPIInt    size;
  MatProductType prodType;
  PetscBool      isAP, isAtP, isAPt, isPtAP, isPAPt;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Read options -n */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* Load the matrix from a binary file */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-A", fileA, PETSC_MAX_PATH_LEN, &flgA));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-P", fileP, PETSC_MAX_PATH_LEN, &flgP));
  PetscCheck(flgA, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must give a petsc matrix binary file with the -A option");

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_type", matTypeStr, sizeof(matTypeStr), &flg));
  if (!flg) PetscCall(PetscStrncpy(matTypeStr, MATAIJ, sizeof(matTypeStr))); /* Inject the default if not provided */

  PetscCall(PetscOptionsGetString(NULL, NULL, "-prod_type", prodTypeStr, sizeof(prodTypeStr), &flg));
  if (!flg) PetscCall(PetscStrncpy(prodTypeStr, "AP", sizeof(prodTypeStr))); /* Inject the default if not provided */

  PetscCall(PetscStrcmp(prodTypeStr, "AP", &isAP));
  PetscCall(PetscStrcmp(prodTypeStr, "AtP", &isAtP));
  PetscCall(PetscStrcmp(prodTypeStr, "APt", &isAPt));
  PetscCall(PetscStrcmp(prodTypeStr, "PtAP", &isPtAP));
  PetscCall(PetscStrcmp(prodTypeStr, "PAPt", &isPAPt));

  if (isAP) prodType = MATPRODUCT_AB;
  else if (isAtP) prodType = MATPRODUCT_AtB;
  else if (isAPt) prodType = MATPRODUCT_ABt;
  else if (isPtAP) prodType = MATPRODUCT_PtAP;
  else if (isPAPt) prodType = MATPRODUCT_RARt;
  else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported product type %s", prodTypeStr);

  /* Read the matrix file to A2 */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileA, FILE_MODE_READ, &fdA));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A2));
  PetscCall(MatSetType(A2, MATAIJ));
  PetscCall(MatLoad(A2, fdA));
  PetscCall(PetscViewerDestroy(&fdA));

  PetscCall(MatGetSize(A2, &M, &N));
  PetscCall(MatGetInfo(A2, MAT_GLOBAL_SUM, &info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Input matrix A: %s, %" PetscInt_FMT " x %" PetscInt_FMT ", %lld nonzeros, %.1f per row\n", fileA, M, N, (long long)info.nz_used, (double)info.nz_used / (double)M));

  /* Copy A2 to A and convert A to the specified type */
  PetscCall(MatDuplicate(A2, MAT_COPY_VALUES, &A));
  PetscCall(MatConvert(A, matTypeStr, MAT_INPLACE_MATRIX, &A));

  /* Init P, P2 similarly */
  if (flgP) { /* If user provided P */
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileP, FILE_MODE_READ, &fdP));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &P2));
    PetscCall(MatSetType(P2, MATAIJ));
    PetscCall(MatLoad(P2, fdP));
    PetscCall(PetscViewerDestroy(&fdP));

    PetscCall(MatDuplicate(P2, MAT_COPY_VALUES, &P));
    PetscCall(MatConvert(P, matTypeStr, MAT_INPLACE_MATRIX, &P));

    PetscCall(MatGetSize(P2, &M, &N));
    PetscCall(MatGetInfo(P2, MAT_GLOBAL_SUM, &info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Input matrix P: %s, %" PetscInt_FMT " x %" PetscInt_FMT ", %lld nonzeros, %.1f per row\n", fileP, M, N, (long long)info.nz_used, (double)info.nz_used / (double)M));
  } else { /* otherwise just let P = A */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Input matrix P = A\n"));
    P2 = A2;
    P  = A;
  }

  /* Compute the reference C2 */
  PetscCall(MatProductCreate(A2, P2, NULL, &C2));
  PetscCall(MatProductSetType(C2, prodType));
  PetscCall(MatProductSetFill(C2, PETSC_DEFAULT));
  PetscCall(MatProductSetFromOptions(C2));
  PetscCall(MatProductSymbolic(C2));
  PetscCall(MatProductNumeric(C2));
  PetscCall(MatGetSize(C2, &M, &N));
  PetscCall(MatGetInfo(C2, MAT_GLOBAL_SUM, &info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Mat product  C = %s: %" PetscInt_FMT " x %" PetscInt_FMT ", %lld nonzeros, %.1f per row\n", prodTypeStr, M, N, (long long)info.nz_used, (double)info.nz_used / (double)M));

  /* Compute C */
  PetscCall(MatProductCreate(A, P, NULL, &C));
  PetscCall(MatProductSetType(C, prodType));
  PetscCall(MatProductSetAlgorithm(C, MATPRODUCTALGORITHMBACKEND));
  PetscCall(MatProductSetFill(C, PETSC_DEFAULT));
  PetscCall(MatProductSetFromOptions(C));

  /* Measure  MatProductSymbolic */
  PetscCall(PetscLogStageRegister("MatProductSymbolic", &stage));
  PetscCall(PetscLogStagePush(stage));
  SyncDevice();
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&tstart));
  PetscCall(MatProductSymbolic(C));
  SyncDevice();
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&tend));
  avgTime = (tend - tstart) * 1e6; /* microseconds */
  PetscCall(PetscLogStagePop());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nMatProductSymbolic()         time (us) with %d MPI ranks = %8.2f\n", size, avgTime));

  /* Measure  MatProductNumeric */
  PetscCall(PetscLogStageRegister("MatProductNumeric", &stage));
  for (i = 0; i < n + nskip; i++) {
    if (i == nskip) {
      SyncDevice();
      PetscCall(PetscLogStagePush(stage));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&tstart));
    }
    PetscCall(MatProductReplaceMats(A, P, NULL, C));
    PetscCall(MatProductNumeric(C));
  }
  SyncDevice();
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&tend));
  avgTime = (tend - tstart) * 1e6 / n; /* microseconds */
  PetscCall(PetscLogStagePop());

  PetscCall(MatMultEqual(C, C2, 8, &equal)); /* Not MatEqual() since C and C2 are not necessarily bitwise equal */

  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Matrix production error");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatProductNumeric()  average time (us) with %d MPI ranks = %8.2f\n", size, avgTime));

  PetscCall(MatDestroy(&A));
  if (flgP) PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&C));

  PetscCall(MatDestroy(&A2));
  if (flgP) PetscCall(MatDestroy(&P2));
  PetscCall(MatDestroy(&C2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -n 2 -A ${DATAFILESPATH}/matrices/small
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
