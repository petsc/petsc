static const char help[] = "Tests PetscDeviceContextQueryIdle.\n\n";

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
  // current one non-idle...
  PetscCall(PetscDeviceContextWaitForContext(other, dctx));
  // ...unless it is the null ctx, in which case it being "idle" is equivalent to asking
  // whether the whole device (which includes other streams) is idle. Since the other ctx might
  // be busy, we should explicitly synchronize on the null ctx
  PetscCall(PetscDeviceContextSynchronize(NULL /* equivalently dctx if dctx = NULL */));
  PetscCall(CheckIdle(dctx, "other context waited on it, and synchronizing the NULL context"));
  // both contexts should be idle
  PetscCall(CheckIdle(other, "waiting on other context, and synchronizing the NULL context"));

  PetscCall(PetscDeviceContextDestroy(&other));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(PetscDeviceContextSetStreamType(dctx, PETSC_STREAM_GLOBAL_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscCall(TestQueryIdle(dctx));
  PetscCall(PetscDeviceContextDestroy(&dctx));

  PetscCall(TestQueryIdle(NULL));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
    output_file: ./output/ExitSuccess.out
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
      args: -default_device_type sycl
      suffix: sycl

  test:
    requires: !cxx
    output_file: ./output/ExitSuccess.out
    suffix: no_cxx

TEST*/
