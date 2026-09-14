static char help[] = "Tests MatMatMult() of an MPIAIJ-family matrix with an MPIDENSE-family matrix, including the batched path, against a host reference\n\n";

// Contributed by: Steven Dargaville

#include <petscmat.h>

/* The product of the matrices whose types are set with -A_mat_type and -B_mat_type is compared against
   the product of host MATAIJ and MATDENSE copies holding the same values */
static PetscErrorCode CheckProduct(Mat C, Mat Ch, const char *label)
{
  Mat       Ct;
  PetscReal nrm, err;

  PetscFunctionBeginUser;
  /* a host matrix is duplicated, a device one is copied back to the host */
  PetscCall(MatConvert(C, MATDENSE, MAT_INITIAL_MATRIX, &Ct));
  PetscCall(MatNorm(Ch, NORM_INFINITY, &nrm));
  PetscCall(MatAXPY(Ct, -1.0, Ch, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(Ct, NORM_INFINITY, &err));
  PetscCheck(err <= 100.0 * PETSC_SMALL * PetscMax(nrm, 1.0), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: error %g relative to %g", label, (double)err, (double)nrm);
  PetscCall(MatDestroy(&Ct));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat           A, B, C, Ah, Bh, Ch;
  char          atype[256], btype[256];
  PetscScalar   vals[4];
  PetscInt      m = 20, n = 7, rstart, rend, ncols, cols[4];
  PetscBool     block_diagonal = PETSC_FALSE, check_copies = PETSC_FALSE, tridiagonal = PETSC_FALSE;
  PetscLogEvent event;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscStrncpy(atype, MATAIJ, sizeof(atype)));
  PetscCall(PetscStrncpy(btype, MATDENSE, sizeof(btype)));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "MatMatMult() test options", "Mat");
  PetscCall(PetscOptionsInt("-m", "Number of rows and columns of A", NULL, m, &m, NULL));
  PetscCall(PetscOptionsInt("-n", "Number of columns of B", NULL, n, &n, NULL));
  PetscCall(PetscOptionsFList("-A_mat_type", "Type of A", "MatSetType", MatList, atype, atype, sizeof(atype), NULL));
  PetscCall(PetscOptionsFList("-B_mat_type", "Type of B", "MatSetType", MatList, btype, btype, sizeof(btype), NULL));
  PetscCall(PetscOptionsBool("-block_diagonal", "Only keep the columns of A owned by this process, so the off-diagonal blocks have no columns", NULL, block_diagonal, &block_diagonal, NULL));
  PetscCall(PetscOptionsBool("-tridiagonal", "Make A tridiagonal, so that only the first and last row of each off-diagonal block are nonempty and the block is stored in compressed-row format", NULL, tridiagonal, &tridiagonal, NULL));
  PetscCall(PetscOptionsBool("-check_copies", "Check that a product of device matrices does not copy between the host and the device", NULL, check_copies, &check_copies, NULL));
  PetscOptionsEnd();
  if (check_copies) PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogEventRegister("ProductCheck", MAT_CLASSID, &event));

  /* host reference matrices, deliberately not calling MatSetFromOptions() so they keep their host types */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Ah));
  PetscCall(MatSetSizes(Ah, PETSC_DECIDE, PETSC_DECIDE, m, m));
  PetscCall(MatSetType(Ah, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(Ah, 4, NULL));
  PetscCall(MatMPIAIJSetPreallocation(Ah, 4, NULL, 4, NULL));
  PetscCall(MatGetOwnershipRange(Ah, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    if (tridiagonal) {
      ncols = 0;
      if (i > 0) {
        cols[ncols] = i - 1;
        vals[ncols] = -1.0;
        ncols++;
      }
      cols[ncols] = i;
      vals[ncols] = 2.0;
      ncols++;
      if (i < m - 1) {
        cols[ncols] = i + 1;
        vals[ncols] = -1.0;
        ncols++;
      }
    } else {
      ncols   = 4;
      cols[0] = i;
      cols[1] = (i + 1) % m;
      cols[2] = (i + m / 2) % m;
      cols[3] = (i * 7 + 3) % m;
      for (PetscInt k = 0; k < ncols; k++) vals[k] = 1.0 + 0.1 * (PetscReal)(i + cols[k]);
    }
    /* duplicate columns accumulate with ADD_VALUES */
    for (PetscInt k = 0; k < ncols; k++) {
      if (block_diagonal && (cols[k] < rstart || cols[k] >= rend)) continue;
      PetscCall(MatSetValue(Ah, i, cols[k], vals[k], ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Ah, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Ah, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &Bh));
  PetscCall(MatGetOwnershipRange(Bh, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    for (PetscInt j = 0; j < n; j++) PetscCall(MatSetValue(Bh, i, j, 0.5 + PetscSinReal((PetscReal)(i + 3 * j)), INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(Bh, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Bh, MAT_FINAL_ASSEMBLY));

  /* the matrices under test hold the same values with the requested types */
  PetscCall(MatDuplicate(Ah, MAT_COPY_VALUES, &A));
  PetscCall(MatConvert(A, atype, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatDuplicate(Bh, MAT_COPY_VALUES, &B));
  PetscCall(MatConvert(B, btype, MAT_INPLACE_MATRIX, &B));

  PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatMatMult(Ah, Bh, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Ch));
  PetscCall(CheckProduct(C, Ch, "MAT_INITIAL_MATRIX"));

  /* change the values of both pairs identically and reuse the products */
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatShift(A, 1.0));
  PetscCall(MatScale(B, -0.5));
  PetscCall(MatScale(Ah, 2.0));
  PetscCall(MatShift(Ah, 1.0));
  PetscCall(MatScale(Bh, -0.5));

  PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatMatMult(Ah, Bh, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Ch));
  PetscCall(CheckProduct(C, Ch, "MAT_REUSE_MATRIX"));

  /* the reuse above brings the modified values to the device (MatShift() may run on the host), so any copy logged
     in a further reuse is made by the product itself */
  PetscCall(PetscLogEventBegin(event, 0, 0, 0, 0));
  PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(PetscLogEventEnd(event, 0, 0, 0, 0));
  PetscCall(CheckProduct(C, Ch, "MAT_REUSE_MATRIX again"));

#if PetscDefined(HAVE_DEVICE)
  /* a product of device matrices must run entirely on the device, in either direction and with any number of processes */
  if (check_copies) {
    PetscEventPerfInfo info;

    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info));
    PetscCheck(info.GpuToCpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected GPU to CPU copies (%g bytes) in MatMatMult()", info.GpuToCpuCount, info.GpuToCpuSize);
    PetscCheck(info.CpuToGpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected CPU to GPU copies (%g bytes) in MatMatMult()", info.CpuToGpuCount, info.CpuToGpuSize);
  }
#endif

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&Ah));
  PetscCall(MatDestroy(&Bh));
  PetscCall(MatDestroy(&Ch));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out

    test:
      suffix: host
      nsize: {{1 3}}
      args: -block_diagonal {{0 1}}

    test:
      suffix: host_batch
      nsize: 3
      args: -matproduct_batch_size {{2 3 4}} -tridiagonal {{0 1}}

    # the local block of B is a host MATSEQDENSE here, so the scatter of the off-process rows
    # does not go through device memory and -use_gpu_aware_mpi would only duplicate the runs
    test:
      suffix: kokkos
      requires: kokkos_kernels
      nsize: {{1 3}}
      args: -A_mat_type aijkokkos

    test:
      suffix: kokkos_batch
      requires: kokkos_kernels
      nsize: 3
      args: -A_mat_type aijkokkos -matproduct_batch_size {{0 3}} -tridiagonal {{0 1}}

  # -check_copies in parallel requires GPU-aware MPI: without it PetscSF stages the device buffers of the
  # scatter through the host, and those copies are logged inside MatMatMult(). The plain device tests run
  # with that staging, which is valid on every build, and the _copies_par tests cover the GPU-aware path

  testset:
    requires: cuda kokkos_kernels
    args: -A_mat_type aijkokkos -B_mat_type densecuda
    output_file: output/empty.out

    test:
      suffix: kokkos_cuda
      nsize: 3
      args: -matproduct_batch_size {{0 3}} -tridiagonal {{0 1}} -use_gpu_aware_mpi 0

    test:
      suffix: kokkos_cuda_copies
      nsize: 1
      requires: defined(PETSC_USE_LOG)
      args: -check_copies

    test:
      suffix: kokkos_cuda_copies_par
      nsize: 3
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -check_copies -matproduct_batch_size {{0 3}}

  testset:
    requires: cuda
    args: -A_mat_type aijcusparse
    output_file: output/empty.out

    test:
      suffix: cuda
      nsize: 3
      args: -B_mat_type {{dense densecuda}} -matproduct_batch_size {{0 3}} -tridiagonal {{0 1}} -use_gpu_aware_mpi 0

    test:
      suffix: cuda_copies
      nsize: 1
      requires: defined(PETSC_USE_LOG)
      args: -B_mat_type densecuda -check_copies

    test:
      suffix: cuda_copies_par
      nsize: 3
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -B_mat_type densecuda -check_copies -matproduct_batch_size {{0 3}}

  testset:
    requires: hip kokkos_kernels
    args: -A_mat_type aijkokkos -B_mat_type densehip
    output_file: output/empty.out

    test:
      suffix: kokkos_hip
      nsize: 3
      args: -matproduct_batch_size {{0 3}} -tridiagonal {{0 1}} -use_gpu_aware_mpi 0

    test:
      suffix: kokkos_hip_copies
      nsize: 1
      requires: defined(PETSC_USE_LOG)
      args: -check_copies

    test:
      suffix: kokkos_hip_copies_par
      nsize: 3
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -check_copies -matproduct_batch_size {{0 3}}

  testset:
    requires: hip
    args: -A_mat_type aijhipsparse
    output_file: output/empty.out

    test:
      suffix: hip
      nsize: 3
      args: -B_mat_type {{dense densehip}} -matproduct_batch_size {{0 3}} -tridiagonal {{0 1}} -use_gpu_aware_mpi 0

    test:
      suffix: hip_copies
      nsize: 1
      requires: defined(PETSC_USE_LOG)
      args: -B_mat_type densehip -check_copies

    test:
      suffix: hip_copies_par
      nsize: 3
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -B_mat_type densehip -check_copies -matproduct_batch_size {{0 3}}

TEST*/
