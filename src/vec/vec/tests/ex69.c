static char help[] = "Test CUDA/HIP vector operations on the current device context.\n";

#include <petsc/private/vecimpl.h>
#include <petscdevice.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif

static PetscErrorCode CopyDevice(PetscDeviceType type, PetscDeviceContext dctx, PetscInt n, const PetscScalar *src, PetscScalar *dst)
{
  void *stream;

  PetscFunctionBeginUser;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) PetscCallCUDA(cudaMemcpyAsync(dst, src, (size_t)n * sizeof(*dst), cudaMemcpyDeviceToDevice, *(cudaStream_t *)stream));
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) PetscCallHIP(hipMemcpyAsync(dst, src, (size_t)n * sizeof(*dst), hipMemcpyDeviceToDevice, *(hipStream_t *)stream));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillHost(Vec x, PetscScalar value)
{
  PetscScalar *a;
  PetscInt     n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayWrite(x, &a));
  for (PetscInt i = 0; i < n; ++i) a[i] = value;
  PetscCall(VecRestoreArrayWrite(x, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *const operations[] = {"Scale",           "Set",          "Copy",       "Swap", "AXPY",    "AYPX", "AXPBY", "AXPBYPCZ",  "MAXPY", "WAXPY",       "PointwiseDivide", "PointwiseMult", "PointwiseMax",
                                         "PointwiseMaxAbs", "PointwiseMin", "Reciprocal", "Abs",  "SqrtAbs", "Exp",  "Log",   "Conjugate", "Shift", "SetStdBasis", "PointwiseSign"};

static PetscErrorCode Apply(size_t op, Vec x, Vec y, Vec z)
{
  static const PetscScalar alpha[] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
  Vec                      xy[]    = {x, y, x, y, x, y, x, y, x};

  PetscFunctionBeginUser;
  switch (op) {
  case 0:
    PetscCall(VecScale(x, 3));
    break;
  case 1:
    PetscCall(VecSet(x, 4));
    break;
  case 2:
    PetscCall(VecCopy(x, y));
    break;
  case 3:
    PetscCall(VecSwap(x, y));
    break;
  case 4:
    PetscCall(VecAXPY(y, 2, x));
    break;
  case 5:
    PetscCall(VecAYPX(y, 2, x));
    break;
  case 6:
    PetscCall(VecAXPBY(y, 2, 3, x));
    break;
  case 7:
    PetscCall(VecAXPBYPCZ(z, 2, 3, 4, x, y));
    break;
  case 8:
    for (size_t nv = 1; nv <= PETSC_STATIC_ARRAY_LENGTH(alpha); ++nv) {
      PetscInt n;

      PetscCall(PetscIntCast(nv, &n));
      PetscCall(VecMAXPY(z, n, alpha, xy));
    }
    break;
  case 9:
    PetscCall(VecWAXPY(z, 2, x, y));
    break;
  case 10:
    PetscCall(VecPointwiseDivide(z, x, y));
    break;
  case 11:
    PetscCall(VecPointwiseMult(z, x, y));
    break;
  case 12:
    PetscCall(VecPointwiseMax(z, x, y));
    break;
  case 13:
    PetscCall(VecPointwiseMaxAbs(z, x, y));
    break;
  case 14:
    PetscCall(VecPointwiseMin(z, x, y));
    break;
  case 15:
    PetscCall(VecReciprocal(x));
    break;
  case 16:
    PetscCall(VecAbs(x));
    break;
  case 17:
    PetscCall(VecSqrtAbs(x));
    break;
  case 18:
    PetscCall(VecExp(x));
    break;
  case 19:
    PetscCall(VecLog(x));
    break;
  case 20:
    PetscCall(VecConjugate(x));
    break;
  case 21:
    PetscCall(VecShift(x, 3));
    break;
  case 22:
    PetscCall(VecSetStdBasis(z, 0));
    break;
  case 23:
    PetscCall(VecPointwiseSign(y, x, VEC_SIGN_ZERO_TO_ZERO));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Check(Vec x, Vec ref, const char name[])
{
  const PetscScalar *a, *b;
  PetscInt           n;
  PetscBool          skip_reductions = PETSC_FALSE;

  PetscFunctionBeginUser;
#if PetscDefined(HAVE_HIPROCM)
  #if PETSC_PKG_HIP_VERSION_LT(6, 1, 0)
  /* rocBLAS host-result reductions on nonblocking streams are unreliable before ROCm 6.1.0.
     Retain the entrywise checks on these releases.
     https://github.com/ROCm/rocBLAS/blob/rocm-6.1.0/CHANGELOG.md#rocblas-410-for-rocm-61 */
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x, &skip_reductions, VECSEQHIP, VECMPIHIP, ""));
  #endif
#endif
  if (!skip_reductions) {
    PetscScalar dot, expected_dot;
    PetscReal   norm, expected_norm;

    PetscCall(VecNorm(x, NORM_2, &norm));
    PetscCall(VecNorm(ref, NORM_2, &expected_norm));
    PetscCheck(PetscAbsReal(norm - expected_norm) <= 100 * PETSC_MACHINE_EPSILON * PetscMax(expected_norm, 1), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: norm %g != %g", name, (double)norm, (double)expected_norm);
    PetscCall(VecNorm(x, NORM_INFINITY, &norm));
    PetscCall(VecNorm(ref, NORM_INFINITY, &expected_norm));
    PetscCheck(PetscAbsReal(norm - expected_norm) <= 100 * PETSC_MACHINE_EPSILON * PetscMax(expected_norm, 1), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: infinity norm %g != %g", name, (double)norm, (double)expected_norm);
    PetscCall(VecDot(x, x, &dot));
    PetscCall(VecDot(ref, ref, &expected_dot));
    PetscCheck(PetscAbsScalar(dot - expected_dot) <= 100 * PETSC_MACHINE_EPSILON * PetscMax(PetscAbsScalar(expected_dot), 1), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: incorrect dot product", name);
  }
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayRead(x, &a));
  PetscCall(VecGetArrayRead(ref, &b));
  for (PetscInt i = 0; i < n; ++i) PetscCheck(PetscAbsScalar(a[i] - b[i]) <= 100 * PETSC_MACHINE_EPSILON * PetscMax(PetscAbsScalar(b[i]), 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s: incorrect entry %" PetscInt_FMT, name, i);
  PetscCall(VecRestoreArrayRead(ref, &b));
  PetscCall(VecRestoreArrayRead(x, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestOperations(Vec v, PetscDeviceType type, PetscDeviceContext dctx)
{
  Vec                x[3], ref[3], src[3];
  const PetscScalar *s[3];
  PetscScalar       *a[3];
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(v, &n));
  for (PetscInt j = 0; j < 3; ++j) {
    PetscCall(VecDuplicate(v, &x[j]));
    PetscCall(VecDuplicate(v, &src[j]));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &ref[j]));
    PetscCall(VecSetSizes(ref[j], n, PETSC_DECIDE));
    PetscCall(VecSetType(ref[j], VECSTANDARD));
  }
  for (size_t op = 0; op < PETSC_STATIC_ARRAY_LENGTH(operations); ++op) {
    // Warm allocations and kernels before delaying the input writes.
    for (PetscInt pass = 0; pass < 2; ++pass) {
      for (PetscInt j = 0; j < 3; ++j) {
        PetscScalar value = j + 2;

        if (!j && (op == 16 || op == 17)) value = -2;
#if PetscDefined(USE_COMPLEX)
        if (!j && op == 20) value = PetscCMPLX(2, 1);
#endif
        PetscCall(FillHost(src[j], value));
        PetscCall(FillHost(ref[j], value));
        PetscCall(FillHost(x[j], 1));
        PetscCall(VecGetArrayReadAndMemType(src[j], &s[j], NULL));
        PetscCall(VecGetArrayAndMemType(x[j], &a[j], NULL));
      }
      PetscCall(Apply(op, ref[0], ref[1], ref[2]));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      if (pass) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
      for (PetscInt j = 0; j < 3; ++j) {
        PetscCall(CopyDevice(type, dctx, n, s[j], a[j]));
        PetscCall(VecRestoreArrayAndMemType(x[j], &a[j]));
        PetscCall(VecRestoreArrayReadAndMemType(src[j], &s[j]));
      }
      PetscCall(Apply(op, x[0], x[1], x[2]));
      for (PetscInt j = 0; j < 3; ++j) PetscCall(Check(x[j], ref[j], operations[op]));
    }
  }
  for (PetscInt j = 0; j < 3; ++j) {
    PetscCall(VecDestroy(&src[j]));
    PetscCall(VecDestroy(&ref[j]));
    PetscCall(VecDestroy(&x[j]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMAXPYCoefficients(Vec x, PetscDeviceContext dctx)
{
  const PetscInt counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Vec            y, ys[9], ref;
  PetscScalar   *alpha;
  struct {
    PetscReal   padding;
    PetscScalar values[9];
  }       *host;
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecSet(y, 2));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ref));
  PetscCall(VecSetSizes(ref, n, PETSC_DECIDE));
  PetscCall(VecSetType(ref, VECSTANDARD));
  // A scalar array inside a C struct need not satisfy CUDA/HIP complex alignment.
  PetscCall(PetscMalloc1(1, &host));
  alpha = host->values;
  for (PetscInt j = 0; j < 9; ++j) ys[j] = y;
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(counts); ++k) {
    PetscInt nv = counts[k];

    for (PetscInt pass = 0; pass < 2; ++pass) {
      PetscScalar expected = 0;
      char        name[80];

      PetscCall(VecSet(x, 0));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      if (pass) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
      for (PetscInt i = 0; i < 4; ++i) {
        for (PetscInt j = 0; j < nv; ++j) {
          alpha[j] = i + j + 1;
#if PetscDefined(USE_COMPLEX)
          alpha[j] += PETSC_i * (j + 1);
#endif
          expected += 2 * alpha[j];
        }
        PetscCall(VecMAXPY(x, nv, alpha, ys));
      }
      // Callers may overwrite their coefficients as soon as VecMAXPY() returns.
      for (PetscInt j = 0; j < nv; ++j) alpha[j] = -100;
      PetscCall(FillHost(ref, expected));
      PetscCall(PetscSNPrintf(name, sizeof(name), "MAXPY coefficient lifetime (nv=%" PetscInt_FMT ", delayed=%" PetscInt_FMT ")", nv, pass));
      PetscCall(Check(x, ref, name));
    }
  }
  PetscCall(PetscFree(host));
  PetscCall(VecDestroy(&ref));
  PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestExplicitBackendContext(Vec x, PetscDeviceType type, PetscDeviceContext current)
{
  PetscDeviceContext other;
  Vec                src, ref;
  PetscInt           n;
  PetscScalar       *a;
  const PetscScalar *s;
  PetscErrorCode (*shift)(Vec, PetscScalar, PetscDeviceContext) = NULL;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecDuplicate(x, &src));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ref));
  PetscCall(VecSetSizes(ref, n, PETSC_DECIDE));
  PetscCall(VecSetType(ref, VECSTANDARD));
  PetscCall(FillHost(ref, 5));
  PetscCall(FillHost(x, 1));
  PetscCall(FillHost(src, 2));
  PetscCall(VecGetArrayAndMemType(x, &a, NULL));
  PetscCall(VecGetArrayReadAndMemType(src, &s, NULL));
  PetscCall(PetscDeviceContextSynchronize(current));
  PetscCall(PetscDeviceContextDuplicate(current, &other));
  PetscCall(PetscDeviceContextSetStreamType(other, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(other));
  PetscCall(PetscDeviceContextDelay(other, 0.05));
  PetscCall(CopyDevice(type, other, n, s, a));
  PetscCall(VecRestoreArrayAndMemType(x, &a));
  PetscCall(VecRestoreArrayReadAndMemType(src, &s));
  PetscCall(PetscObjectQueryFunction((PetscObject)x, VecAsyncFnName(Shift), &shift));
  PetscCheck(shift, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing asynchronous shift implementation");
  PetscCall((*shift)(x, 3, other));
  PetscCall(PetscDeviceContextWaitForContext(current, other));
  PetscCall(Check(x, ref, "explicit context"));
  PetscCall(PetscDeviceContextDestroy(&other));
  PetscCall(VecDestroy(&src));
  PetscCall(VecDestroy(&ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeviceArray(Vec x, PetscDeviceType type, PetscMemoryAccessMode mode, PetscBool restore, PetscScalar **a)
{
  const PetscScalar *r = *a;

  PetscFunctionBeginUser;
  switch (type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    if (mode == PETSC_MEMORY_ACCESS_READ) {
      if (restore) PetscCall(VecCUDARestoreArrayRead(x, &r));
      else PetscCall(VecCUDAGetArrayRead(x, &r));
      *a = (PetscScalar *)r;
    } else if (mode == PETSC_MEMORY_ACCESS_WRITE) {
      if (restore) PetscCall(VecCUDARestoreArrayWrite(x, a));
      else PetscCall(VecCUDAGetArrayWrite(x, a));
    } else {
      if (restore) PetscCall(VecCUDARestoreArray(x, a));
      else PetscCall(VecCUDAGetArray(x, a));
    }
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    if (mode == PETSC_MEMORY_ACCESS_READ) {
      if (restore) PetscCall(VecHIPRestoreArrayRead(x, &r));
      else PetscCall(VecHIPGetArrayRead(x, &r));
      *a = (PetscScalar *)r;
    } else if (mode == PETSC_MEMORY_ACCESS_WRITE) {
      if (restore) PetscCall(VecHIPRestoreArrayWrite(x, a));
      else PetscCall(VecHIPGetArrayWrite(x, a));
    } else {
      if (restore) PetscCall(VecHIPRestoreArray(x, a));
      else PetscCall(VecHIPGetArray(x, a));
    }
    break;
#endif
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDeviceArrays(Vec x, PetscDeviceType type, PetscDeviceContext current)
{
  const PetscMemoryAccessMode modes[] = {PETSC_MEMORY_ACCESS_READ, PETSC_MEMORY_ACCESS_WRITE, PETSC_MEMORY_ACCESS_READ_WRITE};
  Vec                         src, ref;
  const PetscScalar          *s;
  PetscScalar                *a, *device_array;
  PetscInt                    n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecDuplicate(x, &src));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ref));
  PetscCall(VecSetSizes(ref, n, PETSC_DECIDE));
  PetscCall(VecSetType(ref, VECSTANDARD));
  PetscCall(FillHost(src, 3));
  PetscCall(VecGetArrayReadAndMemType(src, &s, NULL));
  for (size_t i = 0; i < PETSC_STATIC_ARRAY_LENGTH(modes); ++i) {
    PetscMemoryAccessMode mode = modes[i];

    PetscCall(VecGetArrayAndMemType(x, &a, NULL));
    device_array = a;
    PetscCall(VecRestoreArrayAndMemType(x, &a));
    PetscCall(FillHost(x, 2));
    PetscCall(FillHost(ref, mode == PETSC_MEMORY_ACCESS_WRITE ? 3 : 2));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(PetscDeviceContextDelay(current, 0.05));
    PetscCall(CopyDevice(type, current, n, s, device_array));
    // A read accessor must upload the CPU values after the queued device write.
    PetscCall(DeviceArray(x, type, mode, PETSC_FALSE, &a));
    PetscCall(DeviceArray(x, type, mode, PETSC_TRUE, &a));
    PetscCall(Check(x, ref, "device array"));
  }
  PetscCall(VecRestoreArrayReadAndMemType(src, &s));
  PetscCall(VecDestroy(&src));
  PetscCall(VecDestroy(&ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestRandom(Vec x, PetscDeviceType type, PetscDeviceContext current, PetscDeviceContext saved)
{
  PetscRandom        random;
  const PetscScalar *a;
  PetscInt           n;

  PetscFunctionBeginUser;
  if (type != PETSC_DEVICE_CUDA) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetLocalSize(x, &n));
  for (PetscInt pass = 0; pass < 2; ++pass) {
    // Also use a generator created and warmed on another context.
    if (pass) PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &random));
    PetscCall(PetscRandomSetType(random, PETSCCURAND));
    PetscCall(PetscRandomSetInterval(random, 2, 3));
    PetscCall(VecSetRandom(x, random));
    PetscCall(PetscDeviceContextSynchronize(pass ? saved : current));
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    PetscCall(PetscDeviceContextDelay(current, 0.05));
    PetscCall(VecSet(x, -1));
    PetscCall(VecSetRandom(x, random));
    PetscCall(VecGetArrayRead(x, &a));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(PetscRealPart(a[i]) >= 2 && PetscRealPart(a[i]) <= 3, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Random value outside the requested interval");
    PetscCall(VecRestoreArrayRead(x, &a));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(PetscRandomDestroy(&random));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec                   x;
  PetscDeviceContext    saved, current;
  PetscDevice           device;
  PetscDeviceType       type;
  const PetscStreamType streams[] = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  PetscMPIInt           rank;
  PetscBool             empty = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank", &empty, NULL));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &current));
    PetscCall(PetscDeviceContextSetStreamType(current, streams[k]));
    PetscCall(PetscDeviceContextSetUp(current));
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, empty && rank ? 0 : 4, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(x));
    PetscCall(TestMAXPYCoefficients(x, current));
    PetscCall(TestOperations(x, type, current));
    PetscCall(TestExplicitBackendContext(x, type, current));
    PetscCall(TestDeviceArrays(x, type, current));
    PetscCall(TestRandom(x, type, current, saved));
    PetscCall(VecDestroy(&x));
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&current));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: cuda
    suffix: cuda
    args: -vec_type seqcuda
    output_file: output/empty.out

  test:
    requires: cuda
    suffix: cuda_mpi
    nsize: 2
    args: -vec_type mpicuda -empty_rank {{0 1}}
    output_file: output/empty.out

  test:
    requires: hip
    suffix: hip
    args: -vec_type seqhip
    output_file: output/empty.out

  test:
    requires: hip
    suffix: hip_mpi
    nsize: 2
    args: -vec_type mpihip -empty_rank {{0 1}}
    output_file: output/empty.out

TEST*/
