static char help[] = "Tests VecCUDAGetArray() and friends along with OpenMP target offload\n\n";

#include <petscvec.h>
#include <petscdevice.h>

int main(int argc, char **argv)
{
  const PetscInt     n = 16;
  PetscScalar       *a;
  const PetscScalar *ar;
  Vec                x;
  PetscDeviceContext dctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetType(x, VECCUDA));

  // A PETSc write queued on the current-context stream; the first OpenMP kernel overwrites the
  // same buffer, so the two race (write-after-write) unless PETSc's stream is synchronized first.
  PetscCall(VecSet(x, 100.0));

  /*
     This test interleaves PETSc Vec operations with OpenMP target offload to exercise their
     stream interaction. PETSc queues work on the current device context's stream, and the
     VecCUDAGetArray*() calls return without synchronizing; the omp target regions run on a
     separate, runtime-managed stream with no implicit ordering relative to PETSc's. So whenever
     an OpenMP kernel touches a buffer a preceding PETSc operation is still writing on PETSc's
     stream (e.g. VecSet() or VecScale() here), the current device context must be synchronized
     first, or the two streams race. The omp target regions have no nowait, so they block the
     host on return, which orders the subsequent PETSc calls after the kernel.
  */

  // Write access: overwrite with x[i] = i + 1.
  PetscCall(VecCUDAGetArrayWrite(x, &a));
  PetscCall(PetscDeviceContextSynchronize(dctx));
#pragma omp target teams distribute parallel for is_device_ptr(a)
  for (PetscInt i = 0; i < n; i++) a[i] = i + 1.0;
  PetscCall(VecCUDARestoreArrayWrite(x, &a));

  // A PETSc operation the next OpenMP kernel depends on: double every entry, x[i] = 2(i + 1).
  PetscCall(VecScale(x, 2.0));

  // Read-write access: add to what VecScale() produced, so sync PETSc's stream first, x[i] = 2(i + 1) + 3.
  PetscCall(VecCUDAGetArray(x, &a));
  PetscCall(PetscDeviceContextSynchronize(dctx));
#pragma omp target teams distribute parallel for is_device_ptr(a)
  for (PetscInt i = 0; i < n; i++) a[i] = a[i] + 3.0;
  PetscCall(VecCUDARestoreArray(x, &a));

  // A PETSc operation consuming the OpenMP result; the synchronous target above ordered it, x[i] = 2(i + 1) + 4.
  PetscCall(VecShift(x, 1.0));

  // Read-only access round-trip.
  PetscCall(VecCUDAGetArrayRead(x, &ar));
  PetscCall(VecCUDARestoreArrayRead(x, &ar));

  PetscCall(PetscObjectSetName((PetscObject)x, "x"));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: cuda defined(PETSC_HAVE_OPENMP_TARGET_OFFLOAD_CC)
    output_file: output/ex67_1.out

TEST*/
