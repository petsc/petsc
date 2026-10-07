static const char help[] = "Tests device allocation, alignment, and asynchronous memory operations.\n\n";

#include "petscdevicetestcommon.h"
#include <petscdevice_cuda.h>
#include <petscdevice_hip.h>

#define DebugPrintf(comm, ...) PetscPrintf((comm), "[DEBUG OUTPUT] " __VA_ARGS__)

static PetscErrorCode IncrementSize(PetscRandom rand, PetscInt *value)
{
  PetscReal rval;

  PetscFunctionBeginUser;
  // set the interval such that *value += rval never goes below 0 or above 500
  PetscCall(PetscRandomSetInterval(rand, -(*value), 500 - (*value)));
  PetscCall(PetscRandomGetValueReal(rand, &rval));
  *value += (PetscInt)rval;
  PetscCall(DebugPrintf(PetscObjectComm((PetscObject)rand), "n: %" PetscInt_FMT "\n", *value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestAllocate(PetscDeviceContext dctx, PetscRandom rand, PetscMemType mtype)
{
  PetscScalar *ptr, *tmp_ptr;
  PetscInt     n = 10;

  PetscFunctionBeginUser;
  if (PetscMemTypeDevice(mtype)) {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    // host device context cannot handle this
    if (dtype == PETSC_DEVICE_HOST) PetscFunctionReturn(PETSC_SUCCESS);
  }
  // test basic allocation, deallocation
  PetscCall(IncrementSize(rand, &n));
  PetscCall(PetscDeviceMalloc(dctx, mtype, n, PETSC_DECIDE, &ptr));
  PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "PetscDeviceMalloc() return NULL pointer for %s allocation size %" PetscInt_FMT, PetscMemTypeToString(mtype), n);
  // this ensures the host pointer is at least valid
  if (PetscMemTypeHost(mtype)) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
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

    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, PETSC_DECIDE, &char_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, PETSC_DECIDE, &short_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, PETSC_DECIDE, &int_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, PETSC_DECIDE, &double_ptr));
    PetscCall(PetscDeviceMalloc(dctx, mtype, 1, PETSC_DECIDE, &long_int_ptr));

    // if an error occurs here, it means the alignment system is broken!
    PetscCall(PetscDeviceFree(dctx, char_ptr));
    PetscCall(PetscDeviceFree(dctx, short_ptr));
    PetscCall(PetscDeviceFree(dctx, int_ptr));
    PetscCall(PetscDeviceFree(dctx, double_ptr));
    PetscCall(PetscDeviceFree(dctx, long_int_ptr));
  }

  // test that calloc() produces cleared memory
  PetscCall(IncrementSize(rand, &n));
  PetscCall(PetscDeviceCalloc(dctx, mtype, n, PETSC_DECIDE, &ptr));
  PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "PetscDeviceCalloc() returned NULL pointer for %s allocation size %" PetscInt_FMT, PetscMemTypeToString(mtype), n);
  if (PetscMemTypeHost(mtype)) {
    tmp_ptr = ptr;
  } else {
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &tmp_ptr));
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
  PetscCall(PetscDeviceMalloc(dctx, mtype, n, PETSC_DECIDE, &ptr));
  PetscCall(PetscDeviceArrayZero(dctx, ptr, n));
  PetscCall(PetscMalloc1(n, &tmp_ptr));
  PetscCall(PetscDeviceRegisterMemory(tmp_ptr, PETSC_MEMTYPE_HOST, n * sizeof(*tmp_ptr)));
  for (PetscInt i = 0; i < n; ++i) tmp_ptr[i] = (PetscScalar)i;
  PetscCall(PetscDeviceArrayCopy(dctx, tmp_ptr, ptr, n));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt i = 0; i < n; ++i) PetscCheck(tmp_ptr[i] == (PetscScalar)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceArrayZero() did not clear memory, ptr[%" PetscInt_FMT "] %g != 0", i, (double)PetscAbsScalar(tmp_ptr[i]));
  PetscCall(PetscDeviceFree(dctx, tmp_ptr));
  PetscCall(PetscDeviceFree(dctx, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestAsyncCoherence(PetscDeviceContext dctx, PetscRandom rand)
{
  const PetscInt      nsub = 2;
  const PetscInt      n    = 1024;
  PetscScalar        *ptr, *tmp_ptr;
  PetscDeviceType     dtype;
  PetscDeviceContext *sub;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  // ensure the streams are nonblocking
  PetscCall(PetscDeviceContextForkWithStreamType(dctx, PETSC_STREAM_NONBLOCKING, nsub, &sub));
  // do a warmup to ensure each context acquires any necessary data structures
  for (PetscInt i = 0; i < nsub; ++i) {
    PetscCall(PetscDeviceMalloc(sub[i], PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &ptr));
    PetscCall(PetscDeviceFree(sub[i], ptr));
    if (dtype != PETSC_DEVICE_HOST) {
      PetscCall(PetscDeviceMalloc(sub[i], PETSC_MEMTYPE_DEVICE, n, PETSC_DECIDE, &ptr));
      PetscCall(PetscDeviceFree(sub[i], ptr));
    }
  }

  // allocate on one
  PetscCall(PetscDeviceMalloc(sub[0], PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &ptr));
  // free on the other
  PetscCall(PetscDeviceFree(sub[1], ptr));

  // allocate on one
  PetscCall(PetscDeviceMalloc(sub[0], PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &ptr));
  // zero on the other
  PetscCall(PetscDeviceArrayZero(sub[1], ptr, n));
  PetscCall(PetscDeviceContextSynchronize(sub[1]));
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt i = 0; i < n; ++i) PetscCheck(ptr[i] == (PetscScalar)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceArrayZero() was not properly serialized, ptr[%" PetscInt_FMT "] %g != 0", i, (double)PetscAbsScalar(ptr[i]));
  }
  PetscCall(PetscDeviceFree(sub[1], ptr));

  // test the transfers are serialized
  if (dtype != PETSC_DEVICE_HOST) {
    PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_DEVICE, n, PETSC_DECIDE, &ptr));
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &tmp_ptr));
    PetscCall(PetscDeviceArrayCopy(sub[0], tmp_ptr, ptr, n));
    PetscCall(PetscDeviceContextSynchronize(sub[0]));
    for (PetscInt i = 0; i < n; ++i) {
      for (PetscInt i = 0; i < n; ++i) PetscCheck(tmp_ptr[i] == (PetscScalar)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceArrayCopt() was not properly serialized, ptr[%" PetscInt_FMT "] %g != 0", i, (double)PetscAbsScalar(tmp_ptr[i]));
    }
    PetscCall(PetscDeviceFree(sub[1], ptr));
  }

  PetscCall(PetscDeviceContextJoin(dctx, nsub, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckAlignment(const void *ptr, size_t alignment)
{
  PetscFunctionBeginUser;
  PetscCheck(!((PETSC_UINTPTR_T)ptr % alignment), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointer %p does not satisfy %zu-byte alignment", ptr, alignment);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckBarrierCompletion(PetscDeviceContext dctx, const char operation[])
{
  PetscStreamType stype;
  PetscBool       idle;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
  PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
  PetscCheck(idle || (stype != PETSC_STREAM_DEFAULT_WITH_BARRIER && stype != PETSC_STREAM_NONBLOCKING_WITH_BARRIER), PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s returned with pending work on stream type %s", operation, PetscStreamTypes[stype]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMemoryStreamTypes(PetscDeviceContext dctx, PetscMemType mtype)
{
  const PetscStreamType types[] = {PETSC_STREAM_DEFAULT, PETSC_STREAM_NONBLOCKING, PETSC_STREAM_DEFAULT_WITH_BARRIER, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  const PetscInt        n       = 4097;
  PetscDeviceType       dtype;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  if (dtype != PETSC_DEVICE_CUDA && dtype != PETSC_DEVICE_HIP) PetscFunctionReturn(PETSC_SUCCESS);
  for (size_t t = 0; t < PETSC_STATIC_ARRAY_LENGTH(types); ++t) {
    PetscDeviceContext ctx;
    unsigned char     *ptr, *host;

    PetscCall(PetscDeviceContextDuplicate(dctx, &ctx));
    PetscCall(PetscDeviceContextSetStreamType(ctx, types[t]));
    PetscCall(PetscDeviceContextSetUp(ctx));
    PetscCall(PetscDeviceMalloc(ctx, PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &host));
    // Warm the pool so allocation cannot hide a missing barrier through runtime initialization.
    PetscCall(PetscDeviceMalloc(ctx, mtype, n, 256, &ptr));
    PetscCall(PetscDeviceFree(ctx, ptr));
    PetscCall(PetscDeviceContextSynchronize(ctx));
    for (PetscInt clear = 0; clear < 2; ++clear) {
      PetscCall(PetscDeviceContextDelay(ctx, 0.02));
      if (clear) PetscCall(PetscDeviceCalloc(ctx, mtype, n, 256, &ptr));
      else PetscCall(PetscDeviceMalloc(ctx, mtype, n, 256, &ptr));
      PetscCall(CheckBarrierCompletion(ctx, clear ? "PetscDeviceCalloc()" : "PetscDeviceMalloc()"));
      PetscCall(CheckAlignment(ptr, 256));
      PetscCall(PetscDeviceContextSynchronize(ctx));
      if (clear) {
        PetscCall(PetscDeviceArrayCopy(ctx, host, ptr, n));
        PetscCall(PetscDeviceContextSynchronize(ctx));
        for (PetscInt i = 0; i < n; ++i) PetscCheck(!host[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Calloc did not clear byte %" PetscInt_FMT, i);
      }
      PetscCall(PetscDeviceContextDelay(ctx, 0.02));
      PetscCall(PetscDeviceMemset(ctx, ptr, 19 + clear, n));
      PetscCall(CheckBarrierCompletion(ctx, "PetscDeviceMemset()"));
      PetscCall(PetscDeviceContextSynchronize(ctx));
      PetscCall(PetscDeviceContextDelay(ctx, 0.02));
      PetscCall(PetscDeviceArrayCopy(ctx, host, ptr, n));
      PetscCall(CheckBarrierCompletion(ctx, "PetscDeviceMemcpy()"));
      PetscCall(PetscDeviceContextSynchronize(ctx));
      for (PetscInt i = 0; i < n; ++i) PetscCheck(host[i] == 19 + clear, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect value at byte %" PetscInt_FMT, i);
      PetscCall(PetscDeviceContextDelay(ctx, 0.02));
      PetscCall(PetscDeviceFree(ctx, ptr));
      PetscCall(CheckBarrierCompletion(ctx, "PetscDeviceFree()"));
      PetscCall(PetscDeviceContextSynchronize(ctx));
    }
    PetscCall(PetscDeviceFree(ctx, host));
    PetscCall(PetscDeviceContextSynchronize(ctx));
    PetscCall(PetscDeviceContextDestroy(&ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestPinnedAllocationReuse(PetscDeviceContext dctx)
{
  const PetscInt     n = 32769;
  PetscDeviceContext ctx;
  unsigned char     *input, *device, *result;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextDuplicate(dctx, &ctx));
  PetscCall(PetscDeviceContextSetStreamType(ctx, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(ctx));
  PetscCall(PetscDeviceMalloc(ctx, PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &result));
  PetscCall(PetscDeviceMalloc(ctx, PETSC_MEMTYPE_DEVICE, n, PETSC_DECIDE, &device));
  PetscCall(PetscDeviceMalloc(ctx, PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &input));
  for (PetscInt pass = 0; pass < 4; ++pass) {
    PetscCall(PetscDeviceMemset(ctx, input, 37 + pass, n));
    PetscCall(PetscDeviceContextSynchronize(ctx));
    PetscCall(PetscDeviceContextDelay(ctx, 0.05));
    PetscCall(PetscDeviceArrayCopy(ctx, device, input, n));
    PetscCall(PetscDeviceFree(ctx, input));
    // Reallocation, including debug initialization, must follow the pending upload.
    PetscCall(PetscDeviceMalloc(ctx, PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &input));
    PetscCall(PetscDeviceArrayCopy(ctx, result, device, n));
    PetscCall(PetscDeviceContextSynchronize(ctx));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(result[i] == 37 + pass, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pinned allocation reuse corrupted byte %" PetscInt_FMT " (got %u, expected %" PetscInt_FMT ")", i, (unsigned)result[i], 37 + pass);
  }
  PetscCall(PetscDeviceFree(ctx, input));
  PetscCall(PetscDeviceFree(ctx, device));
  PetscCall(PetscDeviceFree(ctx, result));
  PetscCall(PetscDeviceContextSynchronize(ctx));
  PetscCall(PetscDeviceContextDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMemoryAccessOrdering(PetscDeviceContext dctx)
{
  const PetscInt     n = 32769;
  PetscDeviceContext producer, consumer;
  unsigned char     *src[2], *dest, *host;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextDuplicate(dctx, &producer));
  PetscCall(PetscDeviceContextSetStreamType(producer, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(producer));
  PetscCall(PetscDeviceContextDuplicate(producer, &consumer));
  PetscCall(PetscDeviceMalloc(producer, PETSC_MEMTYPE_DEVICE, n, PETSC_DECIDE, &src[0]));
  PetscCall(PetscDeviceMalloc(producer, PETSC_MEMTYPE_DEVICE, n, PETSC_DECIDE, &src[1]));
  PetscCall(PetscDeviceMalloc(producer, PETSC_MEMTYPE_DEVICE, n, PETSC_DECIDE, &dest));
  PetscCall(PetscDeviceMalloc(consumer, PETSC_MEMTYPE_HOST, n, PETSC_DECIDE, &host));
  PetscCall(PetscDeviceContextSynchronize(producer));
  PetscCall(PetscDeviceContextSynchronize(consumer));
  for (PetscInt pass = 0; pass < 4; ++pass) {
    const PetscInt value = 37 + pass;

    // The consumer must wait for the most recent memset, including repeated writes.
    PetscCall(PetscDeviceMemset(producer, dest, 17, n));
    PetscCall(PetscDeviceContextDelay(producer, 0.05));
    PetscCall(PetscDeviceMemset(producer, dest, value, n));
    PetscCall(PetscDeviceArrayCopy(consumer, host, dest, n));
    PetscCall(PetscDeviceContextSynchronize(consumer));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(host[i] == value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Consumer read stale memset byte %" PetscInt_FMT " (got %u, expected %" PetscInt_FMT ")", i, (unsigned)host[i], value);
    PetscCall(PetscDeviceContextSynchronize(producer));

    PetscCall(PetscDeviceMemset(producer, src[0], value, n));
    PetscCall(PetscDeviceMemset(producer, src[1], value + 32, n));
    PetscCall(PetscDeviceContextSynchronize(producer));
    // The destination dependency must cover the latest copy.
    PetscCall(PetscDeviceArrayCopy(producer, dest, src[0], n));
    PetscCall(PetscDeviceContextDelay(producer, 0.05));
    PetscCall(PetscDeviceArrayCopy(producer, dest, src[1], n));
    PetscCall(PetscDeviceArrayCopy(consumer, host, dest, n));
    PetscCall(PetscDeviceContextSynchronize(consumer));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(host[i] == value + 32, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Consumer read stale copy byte %" PetscInt_FMT " (got %u, expected %" PetscInt_FMT ")", i, (unsigned)host[i], value + 32);
    PetscCall(PetscDeviceContextSynchronize(producer));

    // The source dependency must protect a pending read from a later overwrite.
    PetscCall(PetscDeviceArrayCopy(producer, dest, src[0], n));
    PetscCall(PetscDeviceContextDelay(producer, 0.05));
    PetscCall(PetscDeviceArrayCopy(producer, dest, src[0], n));
    PetscCall(PetscDeviceMemset(consumer, src[0], value + 64, n));
    PetscCall(PetscDeviceArrayCopy(producer, host, dest, n));
    PetscCall(PetscDeviceContextSynchronize(producer));
    PetscCall(PetscDeviceContextSynchronize(consumer));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(host[i] == value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Source overwritten before its copy at byte %" PetscInt_FMT " (got %u, expected %" PetscInt_FMT ")", i, (unsigned)host[i], value);
  }
  PetscCall(PetscDeviceFree(producer, src[0]));
  PetscCall(PetscDeviceFree(producer, src[1]));
  PetscCall(PetscDeviceFree(producer, dest));
  PetscCall(PetscDeviceFree(consumer, host));
  PetscCall(PetscDeviceContextSynchronize(producer));
  PetscCall(PetscDeviceContextSynchronize(consumer));
  PetscCall(PetscDeviceContextDestroy(&producer));
  PetscCall(PetscDeviceContextDestroy(&consumer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestTypeAlignment(PetscDeviceContext dctx)
{
  char        *prefix;
  short       *s;
  double      *d;
  PetscScalar *v;
  struct Triple {
    double x, y, z;
  } *triple;

  PetscFunctionBeginUser;
  for (PetscInt pass = 0; pass < 8; ++pass) {
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, pass + 1, PETSC_DECIDE, &prefix));
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 3, PETSC_DECIDE, &s));
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 3, 1, &d));
    PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_HOST, 3, PETSC_DECIDE, &v));
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 3, PETSC_DECIDE, &triple));
    PetscCall(CheckAlignment(s, PETSC_DEVICE_ALIGNOF(*s)));
    PetscCall(CheckAlignment(d, PETSC_DEVICE_ALIGNOF(*d)));
    PetscCall(CheckAlignment(v, PETSC_DEVICE_ALIGNOF(*v)));
    PetscCall(CheckAlignment(triple, PETSC_DEVICE_ALIGNOF(*triple)));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    for (PetscInt i = 0; i < 3; ++i) {
      PetscCheck(v[i] == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scalar calloc did not clear entry %" PetscInt_FMT, i);
      s[i]        = (short)i;
      d[i]        = i + 0.5;
      triple[i].x = d[i];
      triple[i].y = s[i];
      triple[i].z = triple[i].x + triple[i].y;
      PetscCheck(triple[i].z == 2 * i + 0.5, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect structure entry");
    }
    PetscCall(PetscDeviceFree(dctx, d));
    PetscCall(PetscDeviceFree(dctx, prefix));
    PetscCall(PetscDeviceFree(dctx, triple));
    PetscCall(PetscDeviceFree(dctx, v));
    PetscCall(PetscDeviceFree(dctx, s));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestAlignmentStress(PetscDeviceContext dctx, PetscMemType mtype)
{
  const size_t       alignments[] = {(size_t)PETSC_DECIDE, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
  const PetscInt     sizes[]      = {0, 1, 3, 7, 17, 255, 256, 257, 2047, 2048, 2049, 4097};
  PetscInt           lengths[12] = {0}, values[12] = {0}, sequence = 0;
  unsigned char     *buffers[12] = {NULL}, *host;
  PetscDeviceContext ctx[2];
  PetscDeviceType    dtype;
  PetscBool          cupm;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  cupm = (PetscBool)(dtype == PETSC_DEVICE_CUDA || dtype == PETSC_DEVICE_HIP);
  if (!cupm && PetscMemTypeDevice(mtype)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 4097, PETSC_DECIDE, &host));
  for (PetscInt c = 0; c < 2; ++c) {
    PetscCall(PetscDeviceContextDuplicate(dctx, &ctx[c]));
    PetscCall(PetscDeviceContextSetStreamType(ctx[c], c ? PETSC_STREAM_NONBLOCKING_WITH_BARRIER : PETSC_STREAM_NONBLOCKING));
    PetscCall(PetscDeviceContextSetUp(ctx[c]));
  }
  for (PetscInt pass = 0; pass < 2; ++pass) {
    for (size_t a = 0; a < PETSC_STATIC_ARRAY_LENGTH(alignments); ++a) {
      size_t alignment = alignments[a];
      size_t expected  = alignment == (size_t)PETSC_DECIDE ? PETSC_DEVICE_ALIGNOF(*host) : alignment;

      if (!cupm && alignment != (size_t)PETSC_DECIDE && PETSC_MEMALIGN % alignment) continue;
      for (PetscInt clear = 0; clear < 2; ++clear) {
        for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(sizes); ++k, ++sequence) {
          PetscInt           slot = sequence % 12, n = sizes[(k + pass) % PETSC_STATIC_ARRAY_LENGTH(sizes)];
          PetscDeviceContext alloc = ctx[sequence % 2], use = ctx[(sequence + 1) % 2];

          PetscCall(PetscDeviceFree(use, buffers[slot]));
          if (clear) PetscCall(PetscDeviceCalloc(alloc, mtype, n, alignment, &buffers[slot]));
          else PetscCall(PetscDeviceMalloc(alloc, mtype, n, alignment, &buffers[slot]));
          lengths[slot] = n;
          values[slot]  = 1 + sequence % 251;
          if (!n) {
            PetscCheck(!buffers[slot], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Zero-sized allocation did not return NULL");
            continue;
          }
          PetscCall(CheckAlignment(buffers[slot], expected));
          if (clear) {
            PetscCall(PetscDeviceArrayCopy(use, host, buffers[slot], n));
            PetscCall(PetscDeviceContextSynchronize(use));
            for (PetscInt j = 0; j < n; ++j) PetscCheck(!host[j], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Calloc did not clear byte %" PetscInt_FMT, j);
          }
          PetscCall(PetscDeviceMemset(use, buffers[slot], values[slot], n));
          // Check all live allocations to detect overlap or corruption during pool reuse.
          for (PetscInt b = 0; b < 12; ++b) {
            if (!lengths[b]) continue;
            PetscCall(PetscDeviceArrayCopy(alloc, host, buffers[b], lengths[b]));
            PetscCall(PetscDeviceContextSynchronize(alloc));
            for (PetscInt j = 0; j < lengths[b]; ++j) PetscCheck(host[j] == values[b], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Allocation %" PetscInt_FMT " byte %" PetscInt_FMT " was corrupted", b, j);
          }
        }
      }
    }
  }
  for (PetscInt b = 11; b >= 0; --b) PetscCall(PetscDeviceFree(ctx[b % 2], buffers[b]));
  for (PetscInt c = 0; c < 2; ++c) {
    PetscCall(PetscDeviceContextSynchronize(ctx[c]));
    PetscCall(PetscDeviceContextDestroy(&ctx[c]));
  }
  PetscCall(PetscDeviceFree(dctx, host));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar ScalarValue(PetscInt i)
{
#if PetscDefined(USE_COMPLEX)
  return PetscCMPLX(i + 1, 2 * i + 1);
#else
  return i + 1;
#endif
}

static PetscErrorCode TestScalarAlignment(PetscDeviceContext dctx)
{
  PetscDeviceType dtype;
  PetscScalar    *values[8], *host;
  char           *prefix;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  if (dtype != PETSC_DEVICE_CUDA && dtype != PETSC_DEVICE_HIP) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 7, sizeof(PetscScalar), &host));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt i = 0; i < 7; ++i) host[i] = ScalarValue(i);
  for (PetscInt pass = 0; pass < 4; ++pass) {
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, 3, 1, &prefix));
    for (PetscInt b = 0; b < 8; ++b) {
      PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, 7, sizeof(PetscScalar), &values[b]));
      PetscCall(CheckAlignment(values[b], sizeof(PetscScalar)));
      PetscCall(PetscDeviceArrayCopy(dctx, values[b], host, 7));
#if PetscDefined(HAVE_CUDA)
      if (dtype == PETSC_DEVICE_CUDA) {
        cublasHandle_t      handle;
        cublasPointerMode_t mode;
        const PetscReal     alpha = 2;

        PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, &handle));
        PetscCallCUBLAS(cublasGetPointerMode(handle, &mode));
        PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  #if PetscDefined(USE_COMPLEX)
    #if PetscDefined(USE_REAL_SINGLE)
        PetscCallCUBLAS(cublasCsscal(handle, 7, &alpha, (cuComplex *)values[b], 1));
    #else
        PetscCallCUBLAS(cublasZdscal(handle, 7, &alpha, (cuDoubleComplex *)values[b], 1));
    #endif
  #elif PetscDefined(USE_REAL_SINGLE)
        PetscCallCUBLAS(cublasSscal(handle, 7, &alpha, values[b], 1));
  #else
        PetscCallCUBLAS(cublasDscal(handle, 7, &alpha, values[b], 1));
  #endif
        PetscCallCUBLAS(cublasSetPointerMode(handle, mode));
      }
#endif
#if PetscDefined(HAVE_HIP)
      if (dtype == PETSC_DEVICE_HIP) {
        hipblasHandle_t      handle;
        hipblasPointerMode_t mode;
        const PetscReal      alpha = 2;

        PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, &handle));
        PetscCallHIPBLAS(hipblasGetPointerMode(handle, &mode));
        PetscCallHIPBLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
  #if PetscDefined(USE_COMPLEX)
    #if PetscDefined(USE_REAL_SINGLE)
        PetscCallHIPBLAS(hipblasCsscal(handle, 7, &alpha, (hipFloatComplex *)values[b], 1));
    #else
        PetscCallHIPBLAS(hipblasZdscal(handle, 7, &alpha, (hipDoubleComplex *)values[b], 1));
    #endif
  #elif PetscDefined(USE_REAL_SINGLE)
        PetscCallHIPBLAS(hipblasSscal(handle, 7, &alpha, values[b], 1));
  #else
        PetscCallHIPBLAS(hipblasDscal(handle, 7, &alpha, values[b], 1));
  #endif
        PetscCallHIPBLAS(hipblasSetPointerMode(handle, mode));
      }
#endif
    }
    PetscCall(PetscDeviceContextSynchronize(dctx));
    for (PetscInt b = 7; b >= 0; --b) {
      PetscCall(PetscDeviceArrayCopy(dctx, host, values[b], 7));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      for (PetscInt i = 0; i < 7; ++i) PetscCheck(host[i] == 2 * ScalarValue(i), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect scaled scalar at entry %" PetscInt_FMT, i);
      PetscCall(PetscDeviceFree(dctx, values[b]));
    }
    PetscCall(PetscDeviceFree(dctx, prefix));
    for (PetscInt i = 0; i < 7; ++i) host[i] = ScalarValue(i);
  }
  PetscCall(PetscDeviceFree(dctx, host));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestAlignmentOrdering(PetscDeviceContext dctx, PetscMemType mtype)
{
  const PetscInt     n = 4097;
  PetscDeviceContext ctx[2], current;
  PetscDeviceType    dtype;
  unsigned char     *output[8], *buffer;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  if (dtype != PETSC_DEVICE_CUDA && dtype != PETSC_DEVICE_HIP) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt c = 0; c < 2; ++c) {
    PetscCall(PetscDeviceContextDuplicate(dctx, &ctx[c]));
    PetscCall(PetscDeviceContextSetStreamType(ctx[c], PETSC_STREAM_NONBLOCKING));
    PetscCall(PetscDeviceContextSetUp(ctx[c]));
  }
  for (PetscInt k = 0; k < 8; ++k) PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n, 128, &output[k]));
  // Warm the pool before delaying allocation, initialization, and reuse.
  PetscCall(PetscDeviceCalloc(ctx[0], mtype, n, 128, &buffer));
  PetscCall(PetscDeviceFree(ctx[0], buffer));
  PetscCall(PetscDeviceContextSynchronize(ctx[0]));
  for (PetscInt k = 0; k < 8; ++k) {
    PetscDeviceContext producer = ctx[k % 2], consumer = ctx[(k + 1) % 2];

    PetscCall(PetscDeviceContextDelay(producer, 0.02));
    PetscCall(PetscDeviceCalloc(producer, mtype, n, 128, &buffer));
    PetscCall(CheckAlignment(buffer, 128));
    if (k % 2) PetscCall(PetscDeviceMemset(producer, buffer, k + 1, n));
    PetscCall(PetscDeviceArrayCopy(consumer, output[k], buffer, n));
    // Free must wait for the other stream's read before this chunk can be reused.
    PetscCall(PetscDeviceFree(producer, buffer));
  }
  for (PetscInt c = 0; c < 2; ++c) PetscCall(PetscDeviceContextSynchronize(ctx[c]));
  for (PetscInt k = 0; k < 8; ++k) {
    for (PetscInt j = 0; j < n; ++j)
      PetscCheck(output[k][j] == (k % 2 ? k + 1 : 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream handoff %" PetscInt_FMT " corrupted byte %" PetscInt_FMT " (got %u, memory type %s)", k, j, (unsigned)output[k][j], PetscMemTypeToString(mtype));
    PetscCall(PetscDeviceFree(dctx, output[k]));
  }
  if (PetscMemTypeHost(mtype)) {
    const PetscInt nreuse = 32769;
    unsigned char *input, *device, *result;

    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, nreuse, 128, &result));
    PetscCall(PetscDeviceMalloc(ctx[0], PETSC_MEMTYPE_DEVICE, nreuse, 128, &device));
    PetscCall(PetscDeviceMalloc(ctx[0], PETSC_MEMTYPE_HOST, nreuse, 128, &input));
    PetscCall(PetscDeviceMemset(ctx[0], input, 37, nreuse));
    PetscCall(PetscDeviceContextSynchronize(ctx[0]));
    PetscCall(PetscDeviceContextDelay(ctx[0], 0.05));
    PetscCall(PetscDeviceArrayCopy(ctx[0], device, input, nreuse));
    PetscCall(PetscDeviceFree(ctx[0], input));
    // Reallocation, including debug initialization, must follow the pending upload.
    PetscCall(PetscDeviceMalloc(ctx[0], PETSC_MEMTYPE_HOST, nreuse, 128, &input));
    PetscCall(PetscDeviceArrayCopy(ctx[1], result, device, nreuse));
    PetscCall(PetscDeviceContextSynchronize(ctx[1]));
    for (PetscInt j = 0; j < nreuse; ++j) PetscCheck(result[j] == 37, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pinned allocation reuse corrupted byte %" PetscInt_FMT " (got %u)", j, (unsigned)result[j]);
    PetscCall(PetscDeviceFree(ctx[0], input));
    PetscCall(PetscDeviceFree(ctx[1], device));
    PetscCall(PetscDeviceFree(dctx, result));
    PetscCall(PetscDeviceContextSynchronize(ctx[0]));
    PetscCall(PetscDeviceContextSynchronize(ctx[1]));
  }
  // Typed accesses must also use the explicit context while another remains current.
  if (PetscMemTypeDevice(mtype)) PetscCall(TestScalarAlignment(ctx[0]));
  PetscCall(PetscDeviceContextGetCurrentContext(&current));
  PetscCheck(current == dctx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Explicit-context allocation changed the current context");
  for (PetscInt c = 0; c < 2; ++c) PetscCall(PetscDeviceContextDestroy(&ctx[c]));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;
  PetscRandom        rand;
  PetscBool          test_alignment = PETSC_FALSE, test_stream_types = PETSC_FALSE, test_pinned_reuse = PETSC_FALSE, test_memory_access = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_alignment", &test_alignment, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_stream_types", &test_stream_types, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_pinned_reuse", &test_pinned_reuse, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_memory_access", &test_memory_access, NULL));
  if (test_stream_types) {
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(TestMemoryStreamTypes(dctx, PETSC_MEMTYPE_HOST));
    PetscCall(TestMemoryStreamTypes(dctx, PETSC_MEMTYPE_DEVICE));
  } else if (test_pinned_reuse) {
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(TestPinnedAllocationReuse(dctx));
  } else if (test_memory_access) {
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(TestMemoryAccessOrdering(dctx));
  } else if (test_alignment) {
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(TestTypeAlignment(dctx));
    PetscCall(TestAlignmentStress(dctx, PETSC_MEMTYPE_HOST));
    PetscCall(TestAlignmentStress(dctx, PETSC_MEMTYPE_DEVICE));
    PetscCall(TestAlignmentOrdering(dctx, PETSC_MEMTYPE_HOST));
    PetscCall(TestAlignmentOrdering(dctx, PETSC_MEMTYPE_DEVICE));
    // NULL selects the null context for the current device, including its host pool.
    PetscCall(TestTypeAlignment(NULL));
    PetscCall(PetscDeviceContextSynchronize(NULL));
  } else {
    // A vile hack. The -info output is used to test correctness in this test which prints --
    // among other things -- the PetscObjectId of the PetscDevicContext and the allocated memory.
    //
    // Due to device and host creating slightly different number of objects on startup there will
    // be a mismatch in the ID's. So for the tests involving the host we sit here creating
    // PetscContainers (and incrementing the global PetscObjectId counter) until it reaches some
    // arbitrarily high number to ensure that our first PetscDeviceContext has the same ID across
    // systems.
    {
      PetscObjectId prev_id = 0;

      do {
        PetscContainer c;
        PetscObjectId  id;

        PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &c));
        PetscCall(PetscObjectGetId((PetscObject)c, &id));
        // sanity check, in case PetscContainer ever stops being a PetscObject
        PetscCheck(id > prev_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscObjectIds are not increasing for successively created PetscContainers! current: %" PetscInt64_FMT ", previous: %" PetscInt64_FMT, id, prev_id);
        prev_id = id;
        PetscCall(PetscContainerDestroy(&c));
      } while (prev_id < 50);
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
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -test_memory_access
    output_file: output/ExitSuccess.out
    test:
      suffix: memory_access_cuda
      requires: cuda
      args: -default_device_type cuda
    test:
      suffix: memory_access_hip
      requires: hip
      args: -default_device_type hip

  testset:
    requires: defined(PETSC_USE_DEBUG)
    args: -test_pinned_reuse
    output_file: output/ExitSuccess.out
    test:
      suffix: pinned_reuse_cuda
      requires: cuda
      args: -default_device_type cuda
    test:
      suffix: pinned_reuse_hip
      requires: hip
      args: -default_device_type hip

  testset:
    args: -test_stream_types
    output_file: output/ExitSuccess.out
    test:
      suffix: stream_types_cuda
      requires: cuda
      args: -default_device_type cuda
    test:
      suffix: stream_types_hip
      requires: hip
      args: -default_device_type hip

  testset:
    args: -test_alignment
    output_file: output/ExitSuccess.out
    test:
      suffix: alignment_host
      args: -default_device_type host -options_left 0
    test:
      suffix: alignment_cuda
      requires: cuda
      args: -default_device_type cuda
    test:
      suffix: alignment_hip
      requires: hip
      args: -default_device_type hip

  testset:
    requires: defined(PETSC_USE_INFO) defined(PETSC_USE_DEBUG) defined(PETSC_DEVICELANGUAGE_CXX)
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
      TODO: unclear if it is needed
      args: -default_device_type sycl
      suffix: sycl

  testset:
    output_file: output/ExitSuccess.out
    requires: !defined(PETSC_USE_DEBUG) defined(PETSC_DEVICELANGUAGE_CXX)
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
      TODO: unclear if it is needed
      args: -default_device_type sycl
      suffix: sycl

  test:
    requires: !defined(PETSC_DEVICELANGUAGE_CXX)
    output_file: output/ExitSuccess.out
    filter: grep -v "\[DEBUG OUTPUT\]"
    suffix: no_cxx

TEST*/
