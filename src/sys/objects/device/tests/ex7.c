static const char help[] = "Tests PetscDeviceAllocate().\n\n";

#include "petscdevicetestcommon.h"

#define DebugPrintf(comm, ...) PetscPrintf((comm), "[DEBUG OUTPUT] " __VA_ARGS__)

static PetscErrorCode IncrementSize(PetscRandom rand, PetscInt *value)
{
  PetscReal rval;

  PetscFunctionBegin;
  // set the interval such that *value += rval never goes below 0 or above 500
  PetscCall(PetscRandomSetInterval(rand, -(*value), 500 - (*value)));
  PetscCall(PetscRandomGetValueReal(rand, &rval));
  *value += (PetscInt)rval;
  PetscCall(DebugPrintf(PetscObjectComm((PetscObject)rand), "n: %" PetscInt_FMT "\n", *value));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestAllocate(PetscDeviceContext dctx, PetscRandom rand, PetscMemType mtype)
{
  PetscScalar *ptr, *tmp_ptr;
  PetscInt     n = 10;

  PetscFunctionBegin;
  if (PetscMemTypeDevice(mtype)) {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    // host device context cannot handle this
    if (dtype == PETSC_DEVICE_HOST) PetscFunctionReturn(0);
  }
  // test basic allocation, deallocation
  PetscCall(IncrementSize(rand, &n));
  PetscCall(PetscDeviceMalloc(dctx, mtype, n, &ptr));
  PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "PetscDeviceMalloc() return NULL pointer for %s allocation size %" PetscInt_FMT, PetscMemTypeToString(mtype), n);
  // this ensures the host pointer is at least valid
  if (PetscMemTypeHost(mtype)) {
    for (PetscInt i = 0; i < n; ++i) ptr[i] = (PetscScalar)i;
  }
  PetscCall(PetscDeviceFree(dctx, ptr));

  // test alignment of various types
  {
    char     *char_ptr;
    short    *short_ptr;
    int      *int_ptr;
    double   *double_ptr;
    long int *long_int_ptr;

    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, &char_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, &short_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, &int_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, &double_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, &long_int_ptr));

    // if an error occurs here, it means the alignment system is broken!
    PetscCall(PetscDeviceFree(dctx, char_ptr));
    PetscCall(PetscDeviceFree(dctx, short_ptr));
    PetscCall(PetscDeviceFree(dctx, int_ptr));
    PetscCall(PetscDeviceFree(dctx, double_ptr));
    PetscCall(PetscDeviceFree(dctx, long_int_ptr));
  }

  // test that calloc() produces cleared memory
  PetscCall(IncrementSize(rand, &n));
  PetscCall(PetscDeviceCalloc(dctx, mtype, n, &ptr));
  PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "PetscDeviceCalloc() returned NULL pointer for %s allocation size %" PetscInt_FMT, PetscMemTypeToString(mtype), n);
  if (PetscMemTypeHost(mtype)) {
    tmp_ptr = ptr;
  } else {
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n, &tmp_ptr));
    PetscCall(PetscDeviceArrayCopy(dctx, tmp_ptr, ptr, n));
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt i = 0; i < n; ++i) PetscCheck(tmp_ptr[i] == (PetscScalar)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceCalloc() returned memory that was not cleared, ptr[%" PetscInt_FMT "] %g != 0", i, (double)PetscAbsScalar(tmp_ptr[i]));
  if (tmp_ptr == ptr) {
    tmp_ptr = NULL;
  } else {
    PetscCall(PetscDeviceFree(dctx, tmp_ptr));
  }
  PetscCall(PetscDeviceFree(dctx, ptr));

  // test that devicearrayzero produces cleared memory
  PetscCall(IncrementSize(rand, &n));
  PetscCall(PetscDeviceMalloc(dctx, mtype, n, &ptr));
  PetscCall(PetscDeviceArrayZero(dctx, ptr, n));
  PetscCall(PetscMalloc1(n, &tmp_ptr));
  PetscCall(PetscDeviceRegisterMemory(tmp_ptr, PETSC_MEMTYPE_HOST, n * sizeof(*tmp_ptr)));
  for (PetscInt i = 0; i < n; ++i) tmp_ptr[i] = (PetscScalar)i;
  PetscCall(PetscDeviceArrayCopy(dctx, tmp_ptr, ptr, n));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt i = 0; i < n; ++i) PetscCheck(tmp_ptr[i] == (PetscScalar)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceArrayZero() did not not clear memory, ptr[%" PetscInt_FMT "] %g != 0", i, (double)PetscAbsScalar(tmp_ptr[i]));
  PetscCall(PetscDeviceFree(dctx, tmp_ptr));
  PetscCall(PetscDeviceFree(dctx, ptr));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestAsyncCoherence(PetscDeviceContext dctx, PetscRandom rand)
{
  const PetscInt      nsub = 2;
  const PetscInt      n    = 1024;
  PetscScalar        *ptr, *tmp_ptr;
  PetscDeviceType     dtype;
  PetscDeviceContext *sub;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  // ensure the streams are nonblocking
  PetscCall(PetscDeviceContextForkWithStreamType(dctx, PETSC_STREAM_GLOBAL_NONBLOCKING, nsub, &sub));
  // do a warmup to ensure each context acquires any necessary data structures
  for (PetscInt i = 0; i < nsub; ++i) {
    PetscCall(PetscDeviceMalloc(sub[i], PETSC_MEMTYPE_HOST, n, &ptr));
    PetscCall(PetscDeviceFree(sub[i], ptr));
    if (dtype != PETSC_DEVICE_HOST) {
      PetscCall(PetscDeviceMalloc(sub[i], PETSC_MEMTYPE_DEVICE, n, &ptr));
      PetscCall(PetscDeviceFree(sub[i], ptr));
    }
  }

  // allocate on one
  PetscCall(PetscDeviceMalloc(sub[0], PETSC_MEMTYPE_HOST, n, &ptr));
  // free on the other
  PetscCall(PetscDeviceFree(sub[1], ptr));

  // allocate on one
  PetscCall(PetscDeviceMalloc(sub[0], PETSC_MEMTYPE_HOST, n, &ptr));
  // zero on the other
  PetscCall(PetscDeviceArrayZero(sub[1], ptr, n));
  PetscCall(PetscDeviceContextSynchronize(sub[1]));
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt i = 0; i < n; ++i) PetscCheck(ptr[i] == (PetscScalar)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceArrayZero() was not properly serialized, ptr[%" PetscInt_FMT "] %g != 0", i, (double)PetscAbsScalar(ptr[i]));
  }
  PetscCall(PetscDeviceFree(sub[1], ptr));

  // test the transfers are serialized
  if (dtype != PETSC_DEVICE_HOST) {
    PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_DEVICE, n, &ptr));
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n, &tmp_ptr));
    PetscCall(PetscDeviceArrayCopy(sub[0], tmp_ptr, ptr, n));
    PetscCall(PetscDeviceContextSynchronize(sub[0]));
    for (PetscInt i = 0; i < n; ++i) {
      for (PetscInt i = 0; i < n; ++i) PetscCheck(tmp_ptr[i] == (PetscScalar)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceArrayCopt() was not properly serialized, ptr[%" PetscInt_FMT "] %g != 0", i, (double)PetscAbsScalar(tmp_ptr[i]));
    }
    PetscCall(PetscDeviceFree(sub[1], ptr));
  }

  PetscCall(PetscDeviceContextJoin(dctx, nsub, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &sub));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;
  PetscRandom        rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // A vile hack. The -info output is used to test correctness in this test which prints --
  // among other things -- the PetscObjectId of the PetscDevicContext and the allocated memory.
  //
  // Due to device and host creating slightly different number of objects on startup there will
  // be a mismatch in the ID's. So for the tests involving the host we sit here creating
  // PetscContainers (and incrementing the global PetscObjectId counter) until it reaches some
  // arbitrarily high number to ensure that our first PetscDeviceContext has the same ID across
  // systems.
  if (PETSC_DEVICE_DEFAULT() == PETSC_DEVICE_HOST) {
    PetscObjectId id, prev_id = 0;

    do {
      PetscContainer c;

      PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &c));
      PetscCall(PetscObjectGetId((PetscObject)c, &id));
      // sanity check, in case PetscContainer ever stops being a PetscObject
      PetscCheck(id > prev_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscObjectIds are not increasing for successively created PetscContainers! current: %" PetscInt64_FMT ", previous: %" PetscInt64_FMT, id, prev_id);
      prev_id = id;
      PetscCall(PetscContainerDestroy(&c));
    } while (id < 10);
  }
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  // this seed just so happens to keep the allocation size increasing
  PetscCall(PetscRandomSetSeed(rand, 123));
  PetscCall(PetscRandomSeed(rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  PetscCall(TestAllocate(dctx, rand, PETSC_MEMTYPE_HOST));
  PetscCall(TestAllocate(dctx, rand, PETSC_MEMTYPE_DEVICE));
  PetscCall(TestAsyncCoherence(dctx, rand));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
   requires: defined(PETSC_HAVE_CXX)

  testset:
   requires: defined(PETSC_USE_INFO), defined(PETSC_USE_DEBUG)
   args: -info :device
   suffix: with_info
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -default_device_type host
     filter: sed -e 's/host/IMPL/g' -e 's/cuda/IMPL/g' -e 's/hip/IMPL/g' -e 's/sycl/IMPL/g'
     suffix: host_with_device
   test:
     requires: cuda
     args: -default_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -default_device_type hip
     suffix: hip
   test:
     requires: sycl
     args: -default_device_type sycl
     suffix: sycl

  testset:
   output_file: ./output/ExitSuccess.out
   requires: !defined(PETSC_USE_DEBUG)
   filter: grep -v "\[DEBUG OUTPUT\]"
   suffix: no_info
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -default_device_type host
     suffix: host_with_device
   test:
     requires: cuda
     args: -default_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -default_device_type hip
     suffix: hip
   test:
     requires: sycl
     args: -default_device_type sycl
     suffix: sycl
TEST*/
