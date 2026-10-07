static char help[] = "Test MatSetValuesCOO() for MPIAIJ and its subclasses \n\n";

#include <petscmat.h>
#include <petscdevice.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)

static PetscErrorCode PreallocateDeviceCOO(Mat A, PetscInt n, const PetscInt i[], const PetscInt j[])
{
  PetscDeviceContext saved, current;
  PetscDevice        device;
  PetscDeviceType    type;
  PetscInt          *hi, *hj, *di, *dj;
  void              *stream;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextDuplicate(saved, &current));
  PetscCall(PetscDeviceContextSetStreamType(current, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(current));
  PetscCall(PetscDeviceContextSetCurrentContext(current));
  PetscCall(PetscDeviceContextGetDevice(current, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  PetscCall(PetscDeviceContextGetStreamHandle(current, &stream));
  // Keep valid device pointers on ranks with no COO entries.
  PetscCall(PetscDeviceMalloc(current, PETSC_MEMTYPE_HOST, n + 1, PETSC_DECIDE, &hi));
  PetscCall(PetscDeviceMalloc(current, PETSC_MEMTYPE_HOST, n + 1, PETSC_DECIDE, &hj));
  PetscCall(PetscDeviceCalloc(current, PETSC_MEMTYPE_DEVICE, n + 1, PETSC_DECIDE, &di));
  PetscCall(PetscDeviceCalloc(current, PETSC_MEMTYPE_DEVICE, n + 1, PETSC_DECIDE, &dj));
  PetscCall(PetscDeviceContextSynchronize(current));
  for (PetscInt k = 0; k < n; ++k) {
    hi[k] = i[k];
    hj[k] = j[k];
  }
  // Preallocation must read the updated indices, not the initial zeros.
  PetscCall(PetscDeviceContextDelay(current, 0.05));
  #if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCallCUDA(cudaMemcpyAsync(di, hi, n * sizeof(*di), cudaMemcpyHostToDevice, *(cudaStream_t *)stream));
    PetscCallCUDA(cudaMemcpyAsync(dj, hj, n * sizeof(*dj), cudaMemcpyHostToDevice, *(cudaStream_t *)stream));
  }
  #endif
  #if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCallHIP(hipMemcpyAsync(di, hi, n * sizeof(*di), hipMemcpyHostToDevice, *(hipStream_t *)stream));
    PetscCallHIP(hipMemcpyAsync(dj, hj, n * sizeof(*dj), hipMemcpyHostToDevice, *(hipStream_t *)stream));
  }
  #endif
  PetscCall(MatSetPreallocationCOO(A, n, di, dj));
  PetscCall(PetscDeviceFree(current, di));
  PetscCall(PetscDeviceFree(current, dj));
  PetscCall(PetscDeviceFree(current, hi));
  PetscCall(PetscDeviceFree(current, hj));
  PetscCall(PetscDeviceContextSynchronize(current));
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  PetscCall(PetscDeviceContextDestroy(&current));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCOOStreams(Mat ref, Mat A, PetscInt n, const PetscScalar values[])
{
  Mat                   twice;
  PetscDeviceContext    saved, current;
  PetscDevice           device;
  PetscDeviceType       type;
  PetscScalar          *host, *pinned, *gpu;
  const PetscStreamType streams[] = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  void                 *stream;

  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(ref, MAT_COPY_VALUES, &twice));
  PetscCall(MatScale(twice, 2));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  PetscCall(PetscMalloc1(n + 1, &host));
  PetscCall(PetscDeviceMalloc(saved, PETSC_MEMTYPE_HOST, n + 1, PETSC_DECIDE, &pinned));
  PetscCall(PetscDeviceMalloc(saved, PETSC_MEMTYPE_DEVICE, n + 1, PETSC_DECIDE, &gpu));
  PetscCall(PetscDeviceContextSynchronize(saved));
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &current));
    PetscCall(PetscDeviceContextSetStreamType(current, streams[k]));
    PetscCall(PetscDeviceContextSetUp(current));
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    PetscCall(PetscDeviceContextGetStreamHandle(current, &stream));
    for (PetscInt memory = 0; memory < 3; ++memory) {
      for (PetscInt add = 0; add < 2; ++add) {
        PetscBool equal, idle;

        // Warm allocations and communication before delaying the next update.
        PetscCall(MatSetValuesCOO(A, values, INSERT_VALUES));
        for (PetscInt i = 0; i < n; ++i) host[i] = pinned[i] = values[i];
        // Clear the device input so a stale read differs from the delayed upload.
        if (memory == 2) PetscCall(PetscDeviceArrayZero(current, gpu, n));
        PetscCall(PetscDeviceContextSynchronize(current));
        PetscCall(PetscDeviceContextDelay(current, 0.05));
  #if PetscDefined(HAVE_CUDA)
        if (type == PETSC_DEVICE_CUDA && memory == 2) PetscCallCUDA(cudaMemcpyAsync(gpu, pinned, n * sizeof(*gpu), cudaMemcpyHostToDevice, *(cudaStream_t *)stream));
  #endif
  #if PetscDefined(HAVE_HIP)
        if (type == PETSC_DEVICE_HIP && memory == 2) PetscCallHIP(hipMemcpyAsync(gpu, pinned, n * sizeof(*gpu), hipMemcpyHostToDevice, *(hipStream_t *)stream));
  #endif
        PetscCall(MatSetValuesCOO(A, memory == 2 ? gpu : (memory == 1 ? pinned : host), add ? ADD_VALUES : INSERT_VALUES));
        if (streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
          PetscCall(PetscDeviceContextQueryIdle(current, &idle));
          PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSetValuesCOO() returned with work pending on a barrier context");
        }
        // Host inputs must be reusable on return, even if device work is pending.
        if (memory != 2)
          for (PetscInt i = 0; i < n; ++i) host[i] = pinned[i] = 0;
        PetscCall(PetscDeviceContextSynchronize(current));
        PetscCall(PetscDeviceContextSetCurrentContext(saved));
        PetscCall(MatMultEqual(add ? twice : ref, A, 10, &equal));
        PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect COO values (stream %s, memory %" PetscInt_FMT ", add %" PetscInt_FMT ")", PetscStreamTypes[streams[k]], memory, add);
        PetscCall(PetscDeviceContextSetCurrentContext(current));
      }
    }
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&current));
  }
  PetscCall(MatZeroEntries(A));
  PetscCall(PetscFree(host));
  PetscCall(PetscDeviceFree(saved, gpu));
  PetscCall(PetscDeviceFree(saved, pinned));
  PetscCall(PetscDeviceContextSynchronize(saved));
  PetscCall(MatDestroy(&twice));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  Checks that MatSetValuesCOO() can still be called after MatShift() on a matrix with m local rows and n local columns, whose COO
  pattern already contains every diagonal entry so the shift creates no new nonzeros
*/
static PetscErrorCode TestShiftCOO(PetscInt m, PetscInt n)
{
  Mat          D;
  PetscInt     rstart, rend, N, ncoo;
  PetscInt    *coo_i, *coo_j;
  PetscScalar  readback, expected;
  PetscScalar *coo_v;

  PetscFunctionBeginUser;
  PetscCallMPI(MPIU_Allreduce(&n, &N, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &D));
  PetscCall(MatSetSizes(D, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(D));
  PetscCall(MatGetOwnershipRange(D, &rstart, &rend));
  ncoo = 2 * (rend - rstart);
  PetscCall(PetscMalloc3(ncoo, &coo_i, ncoo, &coo_j, ncoo, &coo_v));
  for (PetscInt r = rstart, k = 0; r < rend; r++) {
    coo_i[k]   = r;
    coo_j[k++] = r % N;
    coo_i[k]   = r;
    coo_j[k++] = (r + 1) % N;
  }
  for (PetscInt k = 0; k < ncoo; k++) coo_v[k] = 1.0;
  PetscCall(MatSetPreallocationCOO(D, ncoo, coo_i, coo_j));
  PetscCall(MatSetValuesCOO(D, coo_v, INSERT_VALUES));
  PetscCall(MatShift(D, 2.0));
  PetscCall(MatShift(D, 2.0));
  PetscCall(MatSetValuesCOO(D, coo_v, ADD_VALUES));
  for (PetscInt r = rstart; r < rend; r++) {
    PetscCall(MatGetValue(D, r, r % N, &readback));
    expected = r < N ? 6.0 : 2.0;
    if (readback != expected) PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatSetValuesCOO() after MatShift() gave %g in row %" PetscInt_FMT ", expected %g\n", (double)PetscRealPart(readback), r, (double)PetscRealPart(expected)));
  }
  PetscCall(PetscFree3(coo_i, coo_j, coo_v));
  PetscCall(MatDestroy(&D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat            A, B, C;
  const PetscInt M = 18, N = 18;
  PetscBool      equal, isHypre;
  PetscScalar   *vals;
  PetscBool      flg = PETSC_FALSE, freecoo = PETSC_FALSE, missing_diagonal = PETSC_FALSE, device_indices = PETSC_FALSE, coo_streams = PETSC_FALSE;
  PetscInt       ncoos = 1;
  PetscMPIInt    size;

  // clang-format off
  /* Construct 18 x 18 matrices, which are big enough to have complex communication patterns but still small enough for debugging */
  // i0/j0[] has a dense diagonal
  PetscInt i0[] = {7, 7, 8, 8, 9, 16, 17,  9, 10, 1, 1, -2, 2, 3, 3, 14, 4, 5, 10, 13,  9,  9, 10, 1, 0, 0, 5,  5,  6, 6, 13, 13, 14, -14, 4, 4, 5, 11, 11, 12, 15, 15, 16};
  PetscInt j0[] = {7, 6, 8, 4, 9, 16, 17, 16, 10, 2, 1,  3, 2, 4, 3, 14, 4, 5, 15, 13, 10, 16, 11, 2, 0, 1, 5, -11, 0, 6, 15, 17, 11,  13, 4, 8, 2, 11, 17, 12,  3, 15,  9};

  // i0/j0[] miss some diagonals
  PetscInt i1[] = {8, 5, 15, 16, 6, 13, 4, 17, 8,  9, 9,  10, -6, 12, 7, 3, -4, 1, 1, 2, 5,  5, 6, 14, 17, 8,  9,  9, 10, 4,  5, 10, 11, 1, 2};
  PetscInt j1[] = {2, 3, 16,  9, 5, 17, 1, 13, 4, 10, 16, 11, -5, 12, 1, 7, -1, 2, 7, 3, 6, 11, 0, 11, 13, 4, 10, 16, 11, 8, -2, 15, 12, 7, 3};

  PetscInt i2[] = {3, 4, 1, 10, 0, 1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 5,  5, 6, 4, 17, 0, 1, 1, 8, 5,  5, 6, 4, 7, 8, 5};
  PetscInt j2[] = {7, 1, 2, 11, 5, 2, 7, 3, 2, 7, 3, 8, 4, 9, 3, 5, 7, 3, 6, 11, 0, 1, 13, 5, 2, 7, 4, 6, 11, 0, 1, 3, 4, 2};
  // clang-format on

  typedef struct {
    PetscInt *i, *j, n;
  } coo_data;

  coo_data coos[3] = {
    {i0, j0, PETSC_STATIC_ARRAY_LENGTH(i0)},
    {i1, j1, PETSC_STATIC_ARRAY_LENGTH(i1)},
    {i2, j2, PETSC_STATIC_ARRAY_LENGTH(i2)}
  };
  coo_data mycoo;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ignore_remote", &flg, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ncoos", &ncoos, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-missing_diagonal", &missing_diagonal, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-device_indices", &device_indices, NULL));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-coo_streams", &coo_streams, NULL));

  mycoo.n = 0;
  if (ncoos > 1) {
    PetscLayout map;

    freecoo = PETSC_TRUE;
    PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &map));
    PetscCall(PetscLayoutSetSize(map, ncoos));
    PetscCall(PetscLayoutSetUp(map));
    PetscCall(PetscLayoutGetLocalSize(map, &ncoos));
    for (PetscInt i = 0; i < ncoos; i++) mycoo.n += coos[i % 3].n;
    PetscCall(PetscMalloc2(mycoo.n, &mycoo.i, mycoo.n, &mycoo.j));
    mycoo.n = 0;
    for (PetscInt i = 0; i < ncoos; i++) {
      PetscCall(PetscArraycpy(mycoo.i + mycoo.n, coos[i % 3].i, coos[i % 3].n));
      PetscCall(PetscArraycpy(mycoo.j + mycoo.n, coos[i % 3].j, coos[i % 3].n));
      mycoo.n += coos[i % 3].n;
    }
    PetscCall(PetscLayoutDestroy(&map));
  } else if (ncoos == 1 && PetscGlobalRank < 3) mycoo = coos[PetscGlobalRank];

  if (missing_diagonal && PetscGlobalRank == 0) mycoo = coos[1];

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(A, MATAIJ));
  // Do not preallocate A to also test MatHash with MAT_IGNORE_OFF_PROC_ENTRIES
  // PetscCall(MatSeqAIJSetPreallocation(A, 2, NULL));
  // PetscCall(MatMPIAIJSetPreallocation(A, 2, NULL, 2, NULL));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, flg));

  PetscCall(PetscMalloc1(mycoo.n, &vals));
  for (PetscInt k = 0; k < mycoo.n; k++) {
    vals[k] = mycoo.j[k];
    PetscCall(MatSetValue(A, mycoo.i[k], mycoo.j[k], vals[k], ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A, NULL, "-a_view"));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetOption(B, MAT_IGNORE_OFF_PROC_ENTRIES, flg));
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  if (device_indices) PetscCall(PreallocateDeviceCOO(B, mycoo.n, mycoo.n ? mycoo.i : NULL, mycoo.n ? mycoo.j : NULL));
  else
#endif
    PetscCall(MatSetPreallocationCOO(B, mycoo.n, mycoo.i, mycoo.j));

#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  if (coo_streams) PetscCall(TestCOOStreams(A, B, mycoo.n, vals));
#endif

  /* Test with ADD_VALUES on a zeroed matrix */
  PetscCall(MatSetValuesCOO(B, vals, ADD_VALUES));
  PetscCall(MatMultEqual(A, B, 10, &equal));
  if (!equal) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSetValuesCOO() failed\n"));
  PetscCall(MatViewFromOptions(B, NULL, "-b_view"));

  /* Test with MatDuplicate on a zeroed matrix */
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &C));
  PetscCall(MatSetValuesCOO(C, vals, ADD_VALUES));
  PetscCall(MatMultEqual(A, C, 10, &equal));
  if (!equal) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSetValuesCOO() on duplicated matrix failed\n"));
  PetscCall(MatViewFromOptions(C, NULL, "-c_view"));

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATHYPRE, &isHypre));

  /* Test MatSetValues() on a matrix preallocated with MatSetPreallocationCOO() with no intervening
     MatSetValuesCOO() that would perform matrix assembly.

     MATHYPRE is left out because this sequence trips a separate defect in hypre's device IJ
     assemble, which reads the off-diagonal column map without mirroring it to the device
     (hypre-space/hypre#1625). */
  if (!isHypre) {
    Mat          D;
    PetscInt     rstart, rend, remote_col, ncoo;
    PetscInt    *coo_i, *coo_j;
    PetscScalar  readback;
    PetscScalar *coo_v;
    PetscBool    isHostAIJ;

    PetscCall(MatCreate(PETSC_COMM_WORLD, &D));
    PetscCall(MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, M, N));
    PetscCall(MatSetFromOptions(D));
    PetscCall(MatGetOwnershipRange(D, &rstart, &rend));

    /* A diagonal entry and one in the first column past this rank's rows, which another rank owns
       whenever there is more than one */
    remote_col = rend % N;
    ncoo       = 2 * (rend - rstart);
    PetscCall(PetscMalloc2(ncoo, &coo_i, ncoo, &coo_j));
    for (PetscInt r = rstart, k = 0; r < rend; r++) {
      coo_i[k] = r;
      coo_j[k] = r;
      k++;
      coo_i[k] = r;
      coo_j[k] = remote_col;
      k++;
    }
    PetscCall(MatSetPreallocationCOO(D, ncoo, coo_i, coo_j));
    PetscCall(PetscFree2(coo_i, coo_j));

    for (PetscInt r = rstart; r < rend; r++) {
      PetscCall(MatSetValue(D, r, r, 2.0, INSERT_VALUES));
      PetscCall(MatSetValue(D, r, remote_col, 5.0, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));

    PetscCall(MatGetValue(D, rstart, remote_col, &readback));
    if (readback != 5.0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatSetValues() after MatSetPreallocationCOO() gave %g, expected 5\n", (double)PetscRealPart(readback)));

    /* MatSetValues() above only wrote into the preallocated nonzero pattern, so MatSetValuesCOO() remains usable */
    PetscCall(PetscMalloc1(ncoo, &coo_v));
    for (PetscInt k = 0; k < ncoo; k++) coo_v[k] = (k % 2) ? 7.0 : 0.0;
    PetscCall(MatSetValuesCOO(D, coo_v, INSERT_VALUES));
    PetscCall(MatGetValue(D, rstart, remote_col, &readback));
    if (readback != 7.0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatSetValuesCOO() after MatSetValues() gave %g, expected 7\n", (double)PetscRealPart(readback)));

    /* Zeroing no rows leaves the nonzero pattern, and so MatSetValuesCOO(), intact */
    PetscCall(MatZeroRows(D, 0, NULL, 1.0, NULL, NULL));
    PetscCall(MatSetValuesCOO(D, coo_v, ADD_VALUES));
    PetscCall(MatGetValue(D, rstart, remote_col, &readback));
    if (readback != 14.0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatSetValuesCOO() after an empty MatZeroRows() gave %g, expected 14\n", (double)PetscRealPart(readback)));

    /* MatAXPY() with DIFFERENT_NONZERO_PATTERN of a matrix whose nonzeros are a subset of those of D leaves the nonzero pattern,
       and so MatSetValuesCOO(), intact. Device implementations of MatAXPY(), such as that of MATAIJKOKKOS, replace the matrix,
       so only the host types MATSEQAIJ and MATMPIAIJ are tested. */
    PetscCall(PetscObjectTypeCompareAny((PetscObject)D, &isHostAIJ, MATSEQAIJ, MATMPIAIJ, ""));
    if (isHostAIJ) {
      Mat Diag;

      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, rend - rstart, rend - rstart, M, N, 1, NULL, 0, NULL, &Diag));
      for (PetscInt r = rstart; r < rend; r++) PetscCall(MatSetValue(Diag, r, r, 1.0, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(Diag, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Diag, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAXPY(D, 1.0, Diag, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&Diag));
      PetscCall(MatSetValuesCOO(D, coo_v, INSERT_VALUES));
      PetscCall(MatGetValue(D, rstart, remote_col, &readback));
      if (readback != 7.0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatSetValuesCOO() after MatAXPY() gave %g, expected 7\n", (double)PetscRealPart(readback)));
    }
    PetscCall(PetscFree(coo_v));
    PetscCall(MatDestroy(&D));
  }

  /* Test that aij->diag is set up correctly on COO matrices */
  if (!isHypre) { // TODO: MATHYPRE currently does not support MatSetValues
    PetscCall(MatShift(A, 2.0));
    PetscCall(MatShift(B, 2.0));
    PetscCall(MatMultEqual(A, B, 10, &equal));
    if (!equal) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatShift() on a duplicated COO matrix failed\n"));

    /* A square matrix whose last process owns no rows, which on one process is a 0 x 0 matrix */
    PetscCall(TestShiftCOO(PetscGlobalRank == size - 1 ? 0 : 3, PetscGlobalRank == size - 1 ? 0 : 3));
    /* A matrix with more rows than columns */
    PetscCall(TestShiftCOO(3, 2));
  }

  PetscCall(PetscFree(vals));
  if (freecoo) PetscCall(PetscFree2(mycoo.i, mycoo.j));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2 3}}
    args: -ignore_remote {{0 1}} -missing_diagonal {{0 1}}
    filter: grep -v type | grep -v "Mat Object"

    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -mat_type aijkokkos

    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: hip
      requires: hip
      args: -mat_type aijhipsparse

    test:
      suffix: aij
      args: -mat_type aij

    test:
      suffix: hypre
      requires: hypre
      args: -mat_type hypre

  testset:
    output_file: output/empty.out
    nsize: 1
    args: -ncoos 3
    filter: grep -v type | grep -v "Mat Object"

    test:
      suffix: 2_kokkos
      requires: kokkos_kernels
      args: -mat_type aijkokkos

    test:
      suffix: 2_cuda
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: 2_hip
      requires: hip
      args: -mat_type aijhipsparse

    test:
      suffix: 2_aij
      args: -mat_type aij

    test:
      suffix: 2_hypre
      requires: hypre
      args: -mat_type hypre

  testset:
    output_file: output/empty.out
    nsize: {{1 2 4}}
    args: -device_indices
    test:
      suffix: cuda_device_indices
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: hip_device_indices
      requires: hip
      args: -mat_type aijhipsparse

  testset:
    output_file: output/empty.out
    nsize: {{1 2 4}}
    args: -coo_streams
    test:
      suffix: cuda_streams
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: hip_streams
      requires: hip
      args: -mat_type aijhipsparse

TEST*/
