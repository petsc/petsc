const char help[] = "Test VecGetLocalVector(), asynchronous array access, CPU binding, copies to host vectors, and host array handoffs";

#include <petscvec.h>
#include <petscdevice.h>
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  #if PetscDefined(HAVE_CUDA)
    #include <petscdevice_cuda.h>
  #endif
  #if PetscDefined(HAVE_HIP)
    #include <petscdevice_hip.h>
  #endif

static PetscErrorCode TestBindToCPU(Vec x)
{
  const PetscStreamType streams[] = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  PetscDeviceContext    saved, dctx;
  PetscScalar          *a;
  PetscInt              n;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecSetPinnedMemoryMin(x, 0));
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    PetscCall(VecGetArrayWrite(x, &a));
    for (PetscInt i = 0; i < n; ++i) a[i] = 1;
    PetscCall(VecRestoreArrayWrite(x, &a));
    PetscCall(VecScale(x, 2));
    PetscCall(PetscDeviceContextSynchronize(saved));
    PetscCall(PetscDeviceContextDuplicate(saved, &dctx));
    PetscCall(PetscDeviceContextSetStreamType(dctx, streams[k]));
    PetscCall(PetscDeviceContextSetUp(dctx));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
    // Delay the device-to-host copy after the GPU values are ready.
    PetscCall(PetscDeviceContextDelay(dctx, 0.1));
    PetscCall(VecBindToCPU(x, PETSC_TRUE));
    PetscCall(VecGetArray(x, &a));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(a[i] == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect host value after VecBindToCPU(): entry %" PetscInt_FMT " is %g, expected 2", i, (double)PetscRealPart(a[i]));
    PetscCall(VecRestoreArray(x, &a));
    PetscCall(VecBindToCPU(x, PETSC_FALSE));
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCopyToHost(Vec x)
{
  const PetscStreamType streams[] = {PETSC_STREAM_DEFAULT, PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  PetscDeviceContext    saved, dctx;
  PetscDevice           device;
  PetscDeviceType       type;
  Vec                   y;
  PetscScalar          *a, *host = NULL;
  const PetscScalar    *ar;
  PetscInt              n, N;
  PetscBool             bound = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-copy_bound", &bound, NULL));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetSize(x, &N));
  PetscCall(VecSetPinnedMemoryMin(x, 0));
  if (bound) {
    PetscCall(VecDuplicate(x, &y));
    PetscCall(VecSetPinnedMemoryMin(y, 0));
    PetscCall(VecBindToCPU(y, PETSC_TRUE));
  } else {
  #if PetscDefined(HAVE_CUDA)
    if (type == PETSC_DEVICE_CUDA) PetscCallCUDA(cudaMallocHost((void **)&host, PetscMax(n, 1) * sizeof(*host)));
  #endif
  #if PetscDefined(HAVE_HIP)
    if (type == PETSC_DEVICE_HIP) PetscCallHIP(hipHostMalloc((void **)&host, PetscMax(n, 1) * sizeof(*host), hipHostMallocDefault));
  #endif
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, N, host, &y));
  }
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &dctx));
    PetscCall(PetscDeviceContextSetStreamType(dctx, streams[k]));
    PetscCall(PetscDeviceContextSetUp(dctx));
    for (PetscInt from_host = 0; from_host < 2; ++from_host) {
      PetscCall(VecGetArrayWrite(x, &a));
      for (PetscInt i = 0; i < n; ++i) a[i] = from_host ? 4 : 2;
      PetscCall(VecRestoreArrayWrite(x, &a));
      if (!from_host) PetscCall(VecScale(x, 2));
      PetscCall(VecGetArrayWrite(y, &a));
      for (PetscInt i = 0; i < n; ++i) a[i] = 1;
      PetscCall(VecRestoreArrayWrite(y, &a));
      PetscCall(PetscDeviceContextSynchronize(saved));
      PetscCall(PetscDeviceContextSetCurrentContext(dctx));
      PetscCall(PetscDeviceContextDelay(dctx, 0.1));
      PetscCall(VecCopy(x, y));
      PetscCall(VecGetArrayRead(y, &ar));
      for (PetscInt i = 0; i < n; ++i)
        PetscCheck(ar[i] == 4, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect host value after VecCopy(): entry %" PetscInt_FMT " is %g, expected 4 (bound %d, host source %" PetscInt_FMT ", stream type %d)", i, (double)PetscRealPart(ar[i]), (int)bound, from_host, (int)streams[k]);
      PetscCall(VecRestoreArrayRead(y, &ar));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(PetscDeviceContextSetCurrentContext(saved));
    }
    PetscCall(PetscDeviceContextDestroy(&dctx));
  }
  PetscCall(VecDestroy(&y));
  #if PetscDefined(HAVE_CUDA)
  if (host && type == PETSC_DEVICE_CUDA) PetscCallCUDA(cudaFreeHost(host));
  #endif
  #if PetscDefined(HAVE_HIP)
  if (host && type == PETSC_DEVICE_HIP) PetscCallHIP(hipHostFree(host));
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestHostArrayHandoff(Vec v)
{
  const PetscStreamType streams[] = {PETSC_STREAM_DEFAULT, PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  PetscDeviceContext    saved, dctx;
  PetscDevice           device;
  PetscDeviceType       type;
  PetscInt              n, test = 0;
  PetscBool             replace = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-handoff_case", &test, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-handoff_replace", &replace, NULL));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  PetscCall(VecGetLocalSize(v, &n));
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    Vec                x, snapshot = NULL;
    PetscScalar       *host = NULL, *replacement = NULL, *a, *b = NULL;
    const PetscScalar *ar;
    void              *stream;

    PetscCall(VecDuplicate(v, &x));
    PetscCall(VecSetPinnedMemoryMin(x, PETSC_INT_MAX));
    PetscCall(VecGetArrayWrite(x, &a));
    for (PetscInt i = 0; i < n; ++i) a[i] = 1;
    PetscCall(VecRestoreArrayWrite(x, &a));
    // Allocate device storage before delaying transfers.
    PetscCall(VecGetArrayReadAndMemType(x, &ar, NULL));
    PetscCall(VecRestoreArrayReadAndMemType(x, &ar));
    if (test != 3) {
  #if PetscDefined(HAVE_CUDA)
      if (type == PETSC_DEVICE_CUDA) PetscCallCUDA(cudaMallocHost((void **)&host, PetscMax(n, 1) * sizeof(*host)));
  #endif
  #if PetscDefined(HAVE_HIP)
      if (type == PETSC_DEVICE_HIP) PetscCallHIP(hipHostMalloc((void **)&host, PetscMax(n, 1) * sizeof(*host), hipHostMallocDefault));
  #endif
      for (PetscInt i = 0; i < n; ++i) host[i] = 2;
      PetscCall(VecPlaceArray(x, host));
    }
    if (replace) {
      PetscCall(PetscMalloc1(n, &replacement));
      for (PetscInt i = 0; i < n; ++i) replacement[i] = 3;
    }
    if (test == 0 || test == 3) PetscCall(VecScale(x, 2));
    if (test == 2) {
      PetscCall(VecDuplicate(x, &snapshot));
      PetscCall(VecGetArrayWriteAndMemType(snapshot, &b, NULL));
    }
    PetscCall(PetscDeviceContextSynchronize(saved));
    PetscCall(PetscDeviceContextDuplicate(saved, &dctx));
    PetscCall(PetscDeviceContextSetStreamType(dctx, streams[k]));
    PetscCall(PetscDeviceContextSetUp(dctx));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
    PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
    // Case 1 has a ready host array and an idle context.
    if (test != 1) PetscCall(PetscDeviceContextDelay(dctx, 0.1));
    if (test == 2) {
      // The offload mask is BOTH while this upload still reads the host array.
      PetscCall(VecGetArrayReadAndMemTypeAsync(x, &ar, NULL));
      if (n) {
  #if PetscDefined(HAVE_CUDA)
        if (type == PETSC_DEVICE_CUDA) PetscCallCUDA(cudaMemcpyAsync(b, ar, n * sizeof(*ar), cudaMemcpyDeviceToDevice, *(cudaStream_t *)stream));
  #endif
  #if PetscDefined(HAVE_HIP)
        if (type == PETSC_DEVICE_HIP) PetscCallHIP(hipMemcpyAsync(b, ar, n * sizeof(*ar), hipMemcpyDeviceToDevice, *(hipStream_t *)stream));
  #endif
      }
      PetscCall(VecRestoreArrayReadAndMemType(x, &ar));
      PetscCall(VecRestoreArrayWriteAndMemType(snapshot, &b));
    }
    if (replace) PetscCall(VecReplaceArray(x, replacement));
    else PetscCall(VecResetArray(x));
    if (host)
      for (PetscInt i = 0; i < n; ++i) {
        PetscCheck(host[i] == (test == 0 ? 4 : 2), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect outgoing host array entry %" PetscInt_FMT " (case %" PetscInt_FMT ", replace %d, stream type %d)", i, test, (int)replace, (int)streams[k]);
        host[i] = 9;
      }
    PetscCall(VecGetArrayRead(x, &ar));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(ar[i] == (replace ? 3 : 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect active host array entry %" PetscInt_FMT, i);
    PetscCall(VecRestoreArrayRead(x, &ar));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    if (snapshot) {
      PetscCall(VecGetArrayRead(snapshot, &ar));
      for (PetscInt i = 0; i < n; ++i)
        PetscCheck(ar[i] == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Host array reuse corrupted the upload: entry %" PetscInt_FMT " is %g, expected 2 (replace %d, stream type %d)", i, (double)PetscRealPart(ar[i]), (int)replace, (int)streams[k]);
      PetscCall(VecRestoreArrayRead(snapshot, &ar));
    }
    PetscCall(VecDestroy(&snapshot));
    PetscCall(VecDestroy(&x));
  #if PetscDefined(HAVE_CUDA)
    if (host && type == PETSC_DEVICE_CUDA) PetscCallCUDA(cudaFreeHost(host));
  #endif
  #if PetscDefined(HAVE_HIP)
    if (host && type == PETSC_DEVICE_HIP) PetscCallHIP(hipHostFree(host));
  #endif
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode TestArrayAsync(Vec x)
{
  const PetscMemoryAccessMode modes[] = {PETSC_MEMORY_ACCESS_READ, PETSC_MEMORY_ACCESS_WRITE, PETSC_MEMORY_ACCESS_READ_WRITE};
  PetscDeviceContext          dctx    = NULL;
  PetscInt                    n, N;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetSize(x, &N));
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(modes); ++k) {
    Vec                view;
    PetscScalar       *a;
    PetscScalar        sum, expected;
    const PetscScalar *ar;
    PetscMemType       mtype;

    PetscCall(VecGetArray(x, &a));
    for (PetscInt i = 0; i < n; ++i) a[i] = 2;
    PetscCall(VecRestoreArray(x, &a));
    if (modes[k] == PETSC_MEMORY_ACCESS_READ) PetscCall(VecGetArrayReadAndMemTypeAsync(x, &ar, &mtype));
    else if (modes[k] == PETSC_MEMORY_ACCESS_WRITE) PetscCall(VecGetArrayWriteAndMemTypeAsync(x, &a, &mtype));
    else PetscCall(VecGetArrayAndMemTypeAsync(x, &a, &mtype));
    if (PetscMemTypeDevice(mtype)) PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    if (dctx) PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(VecCreateMPIWithArrayAndMemType(PetscObjectComm((PetscObject)x), mtype, 1, n, N, modes[k] == PETSC_MEMORY_ACCESS_READ ? (PetscScalar *)ar : a, &view));
    if (modes[k] == PETSC_MEMORY_ACCESS_WRITE) PetscCall(VecSet(view, 3));
    else if (modes[k] == PETSC_MEMORY_ACCESS_READ_WRITE) PetscCall(VecShift(view, 1));
    expected = (modes[k] == PETSC_MEMORY_ACCESS_READ ? 2 : 3) * N;
    PetscCall(VecSum(view, &sum));
    PetscCheck(sum == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected borrowed array values");
    if (dctx) PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(VecDestroy(&view));
    if (modes[k] == PETSC_MEMORY_ACCESS_READ) PetscCall(VecRestoreArrayReadAndMemType(x, &ar));
    else if (modes[k] == PETSC_MEMORY_ACCESS_WRITE) PetscCall(VecRestoreArrayWriteAndMemType(x, &a));
    else PetscCall(VecRestoreArrayAndMemType(x, &a));
    PetscCall(VecSum(x, &sum));
    PetscCheck(sum == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected vector values after restoring the array");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec                global, global_copy, local;
  PetscMPIInt        rank;
  PetscMemType       memtype;
  PetscScalar       *array;
  PetscInt           N = 10;
  const PetscScalar *copy_array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &global));
  PetscCall(PetscObjectSetName((PetscObject)global, "global"));
  PetscCall(VecSetSizes(global, rank == 0 ? N : 0, N));
  PetscCall(VecSetFromOptions(global));
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  {
    PetscBool test_bind = PETSC_FALSE, test_copy = PETSC_FALSE, test_handoff = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_bind_to_cpu", &test_bind, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_copy_to_host", &test_copy, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_host_array_handoff", &test_handoff, NULL));
    if (test_bind) PetscCall(TestBindToCPU(global));
    if (test_copy) PetscCall(TestCopyToHost(global));
    if (test_handoff) PetscCall(TestHostArrayHandoff(global));
  }
#endif
  PetscCall(TestArrayAsync(global));
  PetscCall(VecBindToCPU(global, PETSC_TRUE));
  PetscCall(TestArrayAsync(global));
  PetscCall(VecBindToCPU(global, PETSC_FALSE));
  PetscCall(TestArrayAsync(global));
  PetscCall(VecSetRandom(global, NULL));
  PetscCall(VecDuplicate(global, &global_copy));
  PetscCall(VecCopy(global, global_copy));

  PetscCall(VecGetArrayRead(global_copy, &copy_array));
  PetscCall(VecGetArrayAndMemType(global, &array, &memtype));
  PetscCall(VecRestoreArrayAndMemType(global, &array));
  if (rank == 0) {
    PetscOffloadMask mask;

    PetscCall(VecGetOffloadMask(global, &mask));
    PetscCheck(PetscMemTypeHost(memtype) || mask == PETSC_OFFLOAD_GPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected offload state");
  }

  PetscCall(VecCreateLocalVector(global, &local));
  PetscCall(PetscObjectSetName((PetscObject)local, "local"));
  PetscCall(VecGetLocalVector(global, local));
  if (rank == 0) {
    const PetscScalar *local_array;
    PetscOffloadMask   mask;

    PetscCall(VecGetOffloadMask(local, &mask));
    PetscCheck(PetscMemTypeHost(memtype) || mask == PETSC_OFFLOAD_GPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected local vector offload state");
    PetscCall(VecGetOffloadMask(global, &mask));
    PetscCheck(PetscMemTypeHost(memtype) || mask == PETSC_OFFLOAD_GPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected global vector offload state");

    PetscCall(VecGetArrayRead(local, &local_array));
    for (PetscInt i = 0; i < N; i++) {
      PetscCheck(copy_array[i] == local_array[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecGetLocalVector() value mismatch: local[%" PetscInt_FMT "] = %g, global[%" PetscInt_FMT "] = %g", i, (double)PetscRealPart(local_array[i]), i, (double)PetscRealPart(copy_array[i]));
    }
    PetscCall(VecRestoreArrayRead(local, &local_array));
  }
  PetscCall(VecRestoreLocalVector(global, local));
  PetscCall(VecDestroy(&local));
  PetscCall(VecRestoreArrayRead(global_copy, &copy_array));
  PetscCall(VecDestroy(&global_copy));
  PetscCall(VecDestroy(&global));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: cuda
    nsize: {{1 2}}
    suffix: 0
    args: -vec_type mpicuda
    output_file: output/empty.out

  test:
    nsize: {{1 2}}
    suffix: cpu
    args: -vec_type standard
    output_file: output/empty.out

  test:
    requires: hip
    nsize: {{1 2}}
    suffix: hip
    args: -vec_type hip
    output_file: output/empty.out

  test:
    requires: cuda
    nsize: {{1 2}}
    suffix: bind_cuda
    args: -vec_type cuda -test_bind_to_cpu
    output_file: output/empty.out

  test:
    requires: hip
    nsize: {{1 2}}
    suffix: bind_hip
    args: -vec_type hip -test_bind_to_cpu
    output_file: output/empty.out

  test:
    requires: cuda
    nsize: {{1 2}}
    suffix: copy_cuda
    args: -vec_type cuda -test_copy_to_host -copy_bound {{0 1}}
    output_file: output/empty.out

  test:
    requires: hip
    nsize: {{1 2}}
    suffix: copy_hip
    args: -vec_type hip -test_copy_to_host -copy_bound {{0 1}}
    output_file: output/empty.out

  test:
    requires: cuda
    nsize: {{1 2}}
    suffix: handoff_cuda
    args: -vec_type cuda -test_host_array_handoff -handoff_case {{0 1 2}} -handoff_replace {{0 1}}
    output_file: output/empty.out

  test:
    requires: hip
    nsize: {{1 2}}
    suffix: handoff_hip
    args: -vec_type hip -test_host_array_handoff -handoff_case {{0 1 2}} -handoff_replace {{0 1}}
    output_file: output/empty.out

  test:
    requires: cuda
    nsize: {{1 2}}
    suffix: replace_owned_cuda
    args: -vec_type cuda -test_host_array_handoff -handoff_case 3 -handoff_replace
    output_file: output/empty.out

  test:
    requires: hip
    nsize: {{1 2}}
    suffix: replace_owned_hip
    args: -vec_type hip -test_host_array_handoff -handoff_case 3 -handoff_replace
    output_file: output/empty.out

TEST*/
