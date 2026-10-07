static const char help[] = "Tests PetscDeviceContextQueryIdle() and PetscDeviceContextDelay().\n\n";

#include "petscdevicetestcommon.h"

static PetscErrorCode CheckIdle(PetscDeviceContext dctx, const char operation[])
{
  PetscBool idle = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
  if (!idle) {
    PetscCall(PetscDeviceContextView(dctx, NULL));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext was not idle after %s!", operation);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestQueryIdle(PetscDeviceContext dctx)
{
  PetscDeviceContext other = NULL;

  PetscFunctionBegin;
  // Should of course be idle after synchronization
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(CheckIdle(dctx, "synchronization"));

  // Creating an unrelated device context should leave it idle
  PetscCall(PetscDeviceContextCreate(&other));
  PetscCall(CheckIdle(dctx, "creating unrelated dctx"));

  // Destroying an unrelated device context shouldn't change things either
  PetscCall(PetscDeviceContextDestroy(&other));
  PetscCall(CheckIdle(dctx, "destroying unrelated dctx"));

  // Duplicating shouldn't change it either
  PetscCall(PetscDeviceContextDuplicate(dctx, &other));
  PetscCall(CheckIdle(dctx, "duplication"));

  // Another ctx waiting on it (which may make the other ctx non-idle) should not make the
  // current one non-idle.
  PetscCall(PetscDeviceContextWaitForContext(other, dctx));
  PetscCall(PetscDeviceContextSynchronize(other));
  PetscCall(CheckIdle(dctx, "other context waited on it"));
  PetscCall(CheckIdle(other, "synchronizing the waiting context"));

  PetscCall(PetscDeviceContextDestroy(&other));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDelay(PetscDeviceContext dctx)
{
  PetscDeviceContext other;
  PetscDeviceType    type;
  PetscLogDouble     start, end;
  PetscBool          idle, synchronous;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextDuplicate(dctx, &other));
  PetscCall(PetscDeviceContextSetStreamType(other, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(other));
  PetscCall(PetscDeviceContextGetDeviceType(other, &type));
  synchronous = (PetscBool)(type == PETSC_DEVICE_HOST);
  PetscCall(PetscDeviceContextDelay(other, 0));
  PetscCall(PetscDeviceContextDelay(other, 0.001));
  PetscCall(PetscDeviceContextSynchronize(other));
  PetscCall(PetscTime(&start));
  PetscCall(PetscDeviceContextDelay(other, 0.1));
  PetscCall(PetscTime(&end));
  PetscCall(PetscDeviceContextQueryIdle(other, &idle));
  if (synchronous) {
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Context is not idle after a synchronous delay");
    PetscCheck(end - start >= 0.09, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Synchronous delay returned too early: %g seconds", (double)(end - start));
  } else PetscCheck(!idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Delayed device context is unexpectedly idle");
  PetscCall(PetscDeviceContextSynchronize(other));
  PetscCall(CheckIdle(other, "synchronizing a delayed context"));
  PetscCall(PetscDeviceContextDestroy(&other));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDefaultContext(void)
{
  PetscDeviceContext saved, standard, work, resolved, host;
  PetscDevice        device;
  PetscDeviceType    type;
  PetscStreamType    streamtype;
  PetscBool          initialized[PETSC_DEVICE_MAX];
  PetscInt           value  = 0;
  PetscInt          *empty  = NULL;
  const PetscInt    *source = &value;
  void              *stream;

  PetscFunctionBeginUser;
  PetscCheck(PetscDeviceContextDefault, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Default placeholder is null");
  // No-work calls must not initialize a device just to resolve the placeholder.
  for (PetscInt i = 0; i < PETSC_DEVICE_MAX; ++i) initialized[i] = PetscDeviceInitialized((PetscDeviceType)i);
  PetscCall(PetscDeviceMalloc(PetscDeviceContextDefault, PETSC_MEMTYPE_HOST, 0, PETSC_DECIDE, &empty));
  PetscCheck(!empty, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Zero-length allocation returned a pointer");
  PetscCall(PetscDeviceCalloc(PetscDeviceContextDefault, PETSC_MEMTYPE_HOST, 0, PETSC_DECIDE, &empty));
  PetscCheck(!empty, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Zero-length allocation returned a pointer");
  PetscCall(PetscDeviceMemcpy(PetscDeviceContextDefault, NULL, NULL, 0));
  PetscCall(PetscDeviceMemcpy(PetscDeviceContextDefault, &value, source, sizeof(value)));
  PetscCall(PetscDeviceMemset(PetscDeviceContextDefault, NULL, 0, 0));
  PetscCall(PetscDeviceFree(PetscDeviceContextDefault, empty));
  PetscCall(PetscDeviceContextDelay(PetscDeviceContextDefault, 0));
  PetscCall(PetscDeviceContextWaitForContext(PetscDeviceContextDefault, PetscDeviceContextDefault));
  for (PetscInt i = 0; i < PETSC_DEVICE_MAX; ++i) PetscCheck(PetscDeviceInitialized((PetscDeviceType)i) == initialized[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "No-work call initialized device type %s", PetscDeviceTypes[i]);
  // The placeholder must work before any explicit context creation, including lazy initialization.
  PetscCall(PetscDeviceContextSetUp(PetscDeviceContextDefault));
  PetscCall(PetscDeviceContextGetStreamType(PetscDeviceContextDefault, &streamtype));
  PetscCheck(streamtype == PETSC_STREAM_DEFAULT, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect default stream type");
  PetscCall(PetscDeviceContextGetDefaultContext(&standard));
  PetscCheck(standard != PetscDeviceContextDefault, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Getter returned the placeholder");
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextDuplicate(PetscDeviceContextDefault, &work));
  PetscCall(PetscDeviceContextSetStreamType(work, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(work));
  PetscCall(PetscDeviceContextSetCurrentContext(work));
  PetscCall(PetscDeviceContextGetDefaultContext(&resolved));
  PetscCheck(resolved == standard, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Changing streams changed the default context");
  PetscCall(PetscDeviceContextGetStreamType(PetscDeviceContextDefault, &streamtype));
  PetscCheck(streamtype == PETSC_STREAM_DEFAULT, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Placeholder selected the current stream");
  PetscCall(PetscDeviceContextGetDeviceType(PetscDeviceContextDefault, &type));
  if (type != PETSC_DEVICE_HOST) PetscCall(PetscDeviceContextGetStreamHandle(PetscDeviceContextDefault, &stream));
  PetscCall(PetscDeviceContextDelay(PetscDeviceContextDefault, 0));
  PetscCall(PetscDeviceContextWaitForContext(work, PetscDeviceContextDefault));
  PetscCall(PetscDeviceContextWaitForContext(PetscDeviceContextDefault, work));
  PetscCall(PetscDeviceContextSynchronize(PetscDeviceContextDefault));
  PetscCall(CheckIdle(PetscDeviceContextDefault, "default context synchronization"));
  PetscCall(PetscDeviceContextViewFromOptions(PetscDeviceContextDefault, NULL, "-default_context_view"));

  PetscCall(PetscDeviceCreate(PETSC_DEVICE_HOST, PETSC_DECIDE, &device));
  PetscCall(PetscDeviceConfigure(device));
  PetscCall(PetscDeviceContextCreate(&host));
  PetscCall(PetscDeviceContextSetDevice(host, device));
  PetscCall(PetscDeviceDestroy(&device));
  PetscCall(PetscDeviceContextSetUp(host));
  PetscCall(PetscDeviceContextSetCurrentContext(host));
  PetscCall(PetscDeviceContextGetDeviceType(PetscDeviceContextDefault, &type));
  PetscCheck(type == PETSC_DEVICE_HOST, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Placeholder did not follow the current device");
  PetscCall(PetscDeviceContextGetDefaultContext(&resolved));
  PetscCall(PetscDeviceContextGetDeviceType(standard, &type));
  PetscCheck((resolved == standard) == (type == PETSC_DEVICE_HOST), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Default contexts were not cached per device");
  PetscCall(PetscDeviceContextSetCurrentContext(work));
  PetscCall(PetscDeviceContextSetCurrentContext(PetscDeviceContextDefault));
  PetscCall(PetscDeviceContextGetCurrentContext(&resolved));
  PetscCheck(resolved == standard, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Setting current did not resolve the placeholder");
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  PetscCall(PetscDeviceContextDestroy(&host));
  PetscCall(PetscDeviceContextDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx = NULL;
  PetscDeviceType    type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(TestDefaultContext());

  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(PetscDeviceContextSetStreamType(dctx, PETSC_STREAM_DEFAULT));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscCall(TestQueryIdle(dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &type));
  if (type == PETSC_DEVICE_HOST || type == PETSC_DEVICE_CUDA || type == PETSC_DEVICE_HIP) PetscCall(TestDelay(dctx));
  PetscCall(PetscDeviceContextDestroy(&dctx));

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(TestQueryIdle(dctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscDeviceContextSetCurrentContext(PetscDeviceContextDefault));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: defined(PETSC_DEVICELANGUAGE_CXX)
    output_file: output/ExitSuccess.out
    args: -device_enable {{lazy eager}}
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
    suffix: no_cxx

TEST*/
