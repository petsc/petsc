static const char help[] = "Test CUDA/HIP SF link reuse, cleanup, and completion on barrier contexts, and Kokkos VecScatter ordering with the current context.\n";

#include <petscsf.h>
#include <petscdevice.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif

static PetscErrorCode TestKokkosScatter(PetscDeviceContext saved, PetscDeviceContext dctx[])
{
  Vec                x, y;
  VecScatter         scatter;
  IS                 from, to;
  PetscDeviceContext current;
  const PetscScalar *values;
  PetscInt           source[2], target[] = {0, 2};
  PetscMPIInt        rank, size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  // Initialize Kokkos on a context that stays alive until PetscFinalize().
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, 4, PETSC_DECIDE));
  PetscCall(VecSetType(x, VECMPIKOKKOS));
  PetscCall(VecCreate(PETSC_COMM_SELF, &y));
  PetscCall(VecSetSizes(y, 4, 4));
  PetscCall(VecSetType(y, VECSEQKOKKOS));
  source[0] = 4 * rank + 3;
  source[1] = 4 * ((rank + 1) % size) + 1;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 2, source, PETSC_COPY_VALUES, &from));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 2, target, PETSC_COPY_VALUES, &to));
  PetscCall(VecScatterCreate(x, from, y, to, &scatter));
  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));

  // Warm the cached links on Kokkos' stream, then reuse them on two different streams.
  for (PetscInt pass = 0; pass < 3; ++pass) {
    current = pass ? dctx[pass - 1] : saved;
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    for (PetscInt local = 0; local < 2; ++local) {
      for (PetscInt reverse = 0; reverse < 2; ++reverse) {
        Vec         input = reverse ? y : x, output = reverse ? x : y;
        ScatterMode mode = reverse ? (local ? SCATTER_REVERSE_LOCAL : SCATTER_REVERSE) : (local ? SCATTER_FORWARD_LOCAL : SCATTER_FORWARD);

        PetscCall(VecSet(input, 1));
        PetscCall(VecSet(output, 0));
        PetscCall(PetscDeviceContextSynchronize(saved));
        PetscCall(PetscDeviceContextDelay(saved, 0.05));
        PetscCall(VecScale(input, 2));
        PetscCall(VecScatterBegin(scatter, input, output, INSERT_VALUES, mode));
        PetscCall(PetscDeviceContextDelay(current, 0.05));
        PetscCall(VecScatterEnd(scatter, input, output, INSERT_VALUES, mode));
        PetscCall(VecScale(output, 3));
        PetscCall(PetscDeviceContextSynchronize(saved));
        PetscCall(PetscDeviceContextSynchronize(current));
        PetscCall(VecGetArrayRead(output, &values));
        for (PetscInt i = 0; i < 4; ++i) {
          PetscScalar expected = 0;

          if (i == (reverse ? 3 : 0) || (i == (reverse ? 1 : 2) && (!local || size == 1))) expected = 6;
          PetscCheck(values[i] == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pass %" PetscInt_FMT ", mode %d: incorrect Kokkos scatter result at local entry %" PetscInt_FMT, pass, (int)mode, i);
        }
        PetscCall(VecRestoreArrayRead(output, &values));
      }
    }
  }
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscSF            sf;
  PetscSFNode        remote[2];
  PetscDeviceContext saved, dctx[2], current;
  PetscDevice        device;
  PetscDeviceType    type;
  PetscMemType       mtype;
  PetscStreamType    streamtype = PETSC_STREAM_NONBLOCKING;
  PetscInt           local[]    = {0, 2};
  PetscInt          *root, *leaf, *update, *input, *output;
  PetscMPIInt        rank, size;
  PetscBool          kokkos = PETSC_FALSE, cleanup = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-stream_type", PetscStreamTypes, (PetscEnum *)&streamtype, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_kokkos", &kokkos, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_cleanup", &cleanup, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  mtype = type == PETSC_DEVICE_CUDA ? PETSC_MEMTYPE_CUDA : PETSC_MEMTYPE_HIP;
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(PetscDeviceContextDuplicate(saved, &dctx[i]));
    PetscCall(PetscDeviceContextSetStreamType(dctx[i], streamtype));
    PetscCall(PetscDeviceContextSetUp(dctx[i]));
  }
  PetscCall(PetscDeviceContextSetCurrentContext(dctx[0]));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_DEVICE, 4, PETSC_DECIDE, &root));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_DEVICE, 4, PETSC_DECIDE, &leaf));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_DEVICE, 4, PETSC_DECIDE, &update));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_HOST, 4, PETSC_DECIDE, &input));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_HOST, 4, PETSC_DECIDE, &output));
  PetscCall(PetscDeviceContextSynchronize(dctx[0]));
  remote[0].rank = remote[1].rank = (rank + 1) % size;
  remote[0].index                 = 3;
  remote[1].index                 = 1;
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetGraph(sf, 4, 2, local, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES));
  PetscCall(PetscSFSetUp(sf));

  // Initialize the link on A, then reuse it on B and A. Complete each pass before switching contexts.
  for (PetscInt pass = 0; pass < 3; ++pass) {
    current = dctx[pass % 2];
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    for (PetscInt i = 0; i < 4; ++i) input[i] = 100 * pass + 4 * rank + i;
    for (PetscInt op = 0; op < 3; ++op) {
      if (pass) PetscCall(PetscDeviceContextDelay(current, 0.05));
      PetscCall(PetscDeviceMemcpy(current, root, input, 4 * sizeof(*root)));
      PetscCall(PetscDeviceMemcpy(current, leaf, input, 4 * sizeof(*leaf)));
      if (op == 0) {
        PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT, mtype, root, mtype, leaf, MPI_REPLACE));
        if (pass) PetscCall(PetscDeviceContextDelay(current, 0.05));
        PetscCall(PetscSFBcastEnd(sf, MPIU_INT, root, leaf, MPI_REPLACE));
      } else if (op == 1) {
        PetscCall(PetscSFReduceWithMemTypeBegin(sf, MPIU_INT, mtype, leaf, mtype, root, MPI_SUM));
        if (pass) PetscCall(PetscDeviceContextDelay(current, 0.05));
        PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leaf, root, MPI_SUM));
      } else {
        PetscCall(PetscSFFetchAndOpWithMemTypeBegin(sf, MPIU_INT, mtype, root, mtype, leaf, mtype, update, MPI_SUM));
        if (pass) PetscCall(PetscDeviceContextDelay(current, 0.05));
        PetscCall(PetscSFFetchAndOpEnd(sf, MPIU_INT, root, leaf, update, MPI_SUM));
      }
      if (streamtype == PETSC_STREAM_DEFAULT_WITH_BARRIER || streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
        PetscBool idle;

        PetscCall(PetscDeviceContextQueryIdle(current, &idle));
        PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SF operation %" PetscInt_FMT " returned with work pending on a barrier context", op);
      }
      if (op != 1) {
        PetscCall(PetscDeviceMemcpy(current, output, op == 0 ? leaf : update, 4 * sizeof(*output)));
        PetscCall(PetscDeviceContextSynchronize(current));
        for (PetscInt i = 0; i < 2; ++i)
          PetscCheck(output[local[i]] == 100 * pass + 4 * remote[i].rank + remote[i].index, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pass %" PetscInt_FMT ", operation %" PetscInt_FMT ", incorrect leaf %" PetscInt_FMT, pass, op, local[i]);
      }
      if (op != 0) {
        PetscMPIInt prev = (rank + size - 1) % size;

        PetscCall(PetscDeviceMemcpy(current, output, root, 4 * sizeof(*output)));
        PetscCall(PetscDeviceContextSynchronize(current));
        for (PetscInt i = 0; i < 4; ++i) {
          PetscInt expected = input[i];

          if (i == 3) expected += 100 * pass + 4 * prev;
          if (i == 1) expected += 100 * pass + 4 * prev + 2;
          PetscCheck(output[i] == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pass %" PetscInt_FMT ", operation %" PetscInt_FMT ", incorrect root %" PetscInt_FMT, pass, op, i);
        }
      }
    }
  }

  if (cleanup) {
    for (PetscInt contiguous = 0; contiguous < 2; ++contiguous) {
      PetscMPIInt prev = (rank + size - 1) % size;

      // Contiguous indices isolate link buffer cleanup from the synchronous metadata cleanup.
      if (contiguous) {
        local[1]        = 1;
        remote[0].index = 1;
        remote[1].index = 2;
        PetscCall(PetscSFSetGraph(sf, 4, 2, local, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES));
      }
      PetscCall(PetscDeviceContextSetCurrentContext(dctx[0]));
      for (PetscInt i = 0; i < 4; ++i) input[i] = 4 * rank + i + 1;
      for (PetscInt pass = 0; pass < 2; ++pass) {
        PetscCall(PetscDeviceMemcpy(dctx[0], root, input, 4 * sizeof(*root)));
        PetscCall(PetscDeviceMemcpy(dctx[0], leaf, input, 4 * sizeof(*leaf)));
        // Warm the link before delaying local scatter or remote unpacking.
        if (pass && size == 1) PetscCall(PetscDeviceContextDelay(dctx[0], 0.05));
        PetscCall(PetscSFReduceWithMemTypeBegin(sf, MPIU_INT, mtype, leaf, mtype, root, MPI_SUM));
        if (pass && size > 1) PetscCall(PetscDeviceContextDelay(dctx[0], 0.05));
        PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leaf, root, MPI_SUM));
        if (!pass) PetscCall(PetscDeviceContextSynchronize(dctx[0]));
      }
      // Cleanup must respect the link's stream even when a different context is current.
      PetscCall(PetscDeviceContextSetCurrentContext(dctx[1]));
      if (contiguous) PetscCall(PetscSFDestroy(&sf));
      else PetscCall(PetscSFReset(sf));
      PetscCall(PetscDeviceMemcpy(dctx[0], output, root, 4 * sizeof(*output)));
      PetscCall(PetscDeviceContextSynchronize(dctx[0]));
      for (PetscInt i = 0; i < 4; ++i) {
        PetscInt expected = input[i];

        for (PetscInt j = 0; j < 2; ++j)
          if (i == remote[j].index) expected += 4 * prev + local[j] + 1;
        PetscCheck(output[i] == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Contiguous indices %" PetscInt_FMT ": incorrect root %" PetscInt_FMT " after SF cleanup", contiguous, i);
      }
    }
  }

  if (kokkos) PetscCall(TestKokkosScatter(saved, dctx));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscDeviceFree(dctx[0], root));
  PetscCall(PetscDeviceFree(dctx[0], leaf));
  PetscCall(PetscDeviceFree(dctx[0], update));
  PetscCall(PetscDeviceFree(dctx[0], input));
  PetscCall(PetscDeviceFree(dctx[0], output));
  PetscCall(PetscDeviceContextSynchronize(dctx[0]));
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  for (PetscInt i = 0; i < 2; ++i) PetscCall(PetscDeviceContextDestroy(&dctx[i]));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out
    args: -sf_type {{basic neighbor}} -use_gpu_aware_mpi 0 -stream_type {{nonblocking default_with_barrier nonblocking_with_barrier}} -test_cleanup

    test:
      suffix: cuda
      requires: cuda
      args: -sf_backend cuda

    test:
      suffix: hip
      requires: hip
      args: -sf_backend hip

  test:
    suffix: kokkos_cuda
    requires: kokkos_kernels cuda
    nsize: {{1 2}}
    args: -test_kokkos -sf_type {{basic neighbor}} -sf_backend cuda -use_gpu_aware_mpi 0
    output_file: output/empty.out

  test:
    suffix: kokkos_hip
    requires: kokkos_kernels hip
    nsize: {{1 2}}
    args: -test_kokkos -sf_type {{basic neighbor}} -sf_backend hip -use_gpu_aware_mpi 0
    output_file: output/empty.out

  test:
    suffix: cuda_nvshmem
    requires: cuda nvshmem
    nsize: 2
    args: -sf_backend cuda -use_nvshmem -use_nvshmem_get {{0 1}}
    output_file: output/empty.out

TEST*/
