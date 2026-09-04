static const char help[] = "Test MatDiagonalScale() on dense matrices with scaling Vecs of any type\n\n";

// Contributed by: Steven Dargaville

#include <petscmat.h>

/* The scaling Vecs take their type from -vec_type, independently of the Mat type set
   with -mat_type. Scaling is verified against a duplicate matrix scaled with VECSTANDARD
   Vecs holding the same values: for a device matrix both scalings execute the same kernel
   on identical data, so MatEqual() is exact, and no cross-type Vec or Mat copies are
   needed */
int main(int argc, char **args)
{
  Mat           A, B;
  Vec           l, r, lstd, rstd;
  PetscInt      m = 5, n = 4, mloc, nloc, rstart, rend;
  PetscBool     equal = PETSC_FALSE, check_copies = PETSC_FALSE;
  PetscLogEvent event;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_copies", &check_copies, NULL));
  if (check_copies) PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogEventRegister("ScaleCheck", MAT_CLASSID, &event));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatGetLocalSize(A, &mloc, &nloc));

  // l and lstd match A's row layout, r and rstd its column layout
  PetscCall(VecCreate(PETSC_COMM_WORLD, &l));
  PetscCall(VecSetSizes(l, mloc, m));
  PetscCall(VecSetFromOptions(l));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &lstd));
  PetscCall(VecSetSizes(lstd, mloc, m));
  PetscCall(VecSetType(lstd, VECSTANDARD));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &r));
  PetscCall(VecSetSizes(r, nloc, n));
  PetscCall(VecSetFromOptions(r));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &rstd));
  PetscCall(VecSetSizes(rstd, nloc, n));
  PetscCall(VecSetType(rstd, VECSTANDARD));

  PetscCall(VecGetOwnershipRange(l, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscCall(VecSetValue(l, i, (PetscScalar)(i + 2), INSERT_VALUES));
    PetscCall(VecSetValue(lstd, i, (PetscScalar)(i + 2), INSERT_VALUES));
  }
  PetscCall(VecGetOwnershipRange(r, &rstart, &rend));
  for (PetscInt j = rstart; j < rend; j++) {
    PetscCall(VecSetValue(r, j, (PetscScalar)(j + 3), INSERT_VALUES));
    PetscCall(VecSetValue(rstd, j, (PetscScalar)(j + 3), INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(l));
  PetscCall(VecAssemblyEnd(l));
  PetscCall(VecAssemblyBegin(lstd));
  PetscCall(VecAssemblyEnd(lstd));
  PetscCall(VecAssemblyBegin(r));
  PetscCall(VecAssemblyEnd(r));
  PetscCall(VecAssemblyBegin(rstd));
  PetscCall(VecAssemblyEnd(rstd));

  // left scaling only
  PetscCall(MatDiagonalScale(A, l, NULL));
  PetscCall(MatDiagonalScale(B, lstd, NULL));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Left scaling gives the wrong result");

  // right scaling on top of the left scaling
  PetscCall(MatDiagonalScale(A, NULL, r));
  PetscCall(MatDiagonalScale(B, NULL, rstd));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Right scaling gives the wrong result");

  // both sides in one call; l and r were already consumed by the calls above, so any copy logged here is the matrix
  PetscCall(PetscLogEventBegin(event, 0, 0, 0, 0));
  PetscCall(MatDiagonalScale(A, l, r));
  PetscCall(PetscLogEventEnd(event, 0, 0, 0, 0));
  PetscCall(MatDiagonalScale(B, lstd, rstd));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Two-sided scaling gives the wrong result");

#if PetscDefined(HAVE_DEVICE)
  /* a device matrix must never come back to the host to be scaled, and a device-resident scaling Vec must be
     consumed in place; a host scaling Vec is copied to the device by design, so only the first check applies then */
  if (check_copies) {
    PetscEventPerfInfo info;
    const PetscScalar *array;
    PetscMemType       mtype;

    PetscCall(VecGetArrayReadAndMemType(l, &array, &mtype));
    PetscCall(VecRestoreArrayReadAndMemType(l, &array));
    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info));
    PetscCheck(info.GpuToCpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected GPU to CPU copies (%g bytes) in MatDiagonalScale()", info.GpuToCpuCount, info.GpuToCpuSize);
    PetscCheck(!PetscMemTypeDevice(mtype) || info.CpuToGpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected CPU to GPU copies (%g bytes) in MatDiagonalScale() with device scaling Vecs", info.CpuToGpuCount, info.CpuToGpuSize);
  }
#endif

  PetscCall(VecDestroy(&l));
  PetscCall(VecDestroy(&lstd));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&rstd));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: cpu
    nsize: {{1 2}}
    output_file: output/empty.out

  test:
    suffix: kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -vec_type kokkos
    output_file: output/empty.out

  # -check_copies requires GPU-aware MPI: without it PetscSF stages the device buffers of the
  # right scaling through the host, and those copies are logged inside MatDiagonalScale()

  testset:
    nsize: {{1 2}}
    requires: cuda
    args: -mat_type densecuda -vec_type {{cuda standard}}
    output_file: output/empty.out

    test:
      suffix: cuda

    test:
      suffix: cuda_copies
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -check_copies

  testset:
    nsize: {{1 2}}
    requires: cuda kokkos_kernels
    args: -mat_type densecuda -vec_type kokkos
    output_file: output/empty.out

    test:
      suffix: densecuda_vec_kokkos

    test:
      suffix: densecuda_vec_kokkos_copies
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -check_copies

  testset:
    nsize: {{1 2}}
    requires: hip
    args: -mat_type densehip -vec_type {{hip standard}}
    output_file: output/empty.out

    test:
      suffix: hip

    test:
      suffix: hip_copies
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -check_copies

  testset:
    nsize: {{1 2}}
    requires: hip kokkos_kernels
    args: -mat_type densehip -vec_type kokkos
    output_file: output/empty.out

    test:
      suffix: densehip_vec_kokkos

    test:
      suffix: densehip_vec_kokkos_copies
      requires: defined(PETSC_HAVE_MPI_GPU_AWARE) defined(PETSC_USE_LOG)
      args: -check_copies

TEST*/
