const char help[] = "Test VecGetLocalVector() and asynchronous array access";

#include <petscvec.h>
#include <petscdevice.h>

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

TEST*/
