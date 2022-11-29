static const char help[] = "Tests creation and destruction of PetscDeviceContext.\n\n";

#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx = NULL, ddup = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* basic creation and destruction */
  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(PetscDeviceContextDestroy(&dctx));
  PetscCall(AssertDeviceContextDoesNotExist(dctx));
  /* double free is no-op */
  PetscCall(PetscDeviceContextDestroy(&dctx));
  PetscCall(AssertDeviceContextDoesNotExist(dctx));

  /* test global context returns a valid context */
  dctx = NULL;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(AssertDeviceContextExists(dctx));
  /* test locally setting to null doesn't clobber the global */
  dctx = NULL;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(AssertDeviceContextExists(dctx));

  /* test duplicate */
  PetscCall(PetscDeviceContextDuplicate(dctx, &ddup));
  /* both device contexts should exist */
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(AssertDeviceContextExists(ddup));

  /* destroying the dup should leave the original untouched */
  PetscCall(PetscDeviceContextDestroy(&ddup));
  PetscCall(AssertDeviceContextDoesNotExist(ddup));
  PetscCall(AssertDeviceContextExists(dctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
    output_file: ./output/ExitSuccess.out
    nsize: {{1 2 4}}
    args: -device_enable {{lazy eager}}
    test:
      requires: !device
      suffix: host_no_device
    test:
      requires: device
      args: -root_device_context_device_type host
      suffix: host_with_device
    test:
      requires: cuda
      args: -root_device_context_device_type cuda
      suffix: cuda
    test:
      requires: hip
      args: -root_device_context_device_type hip
      suffix: hip
    test:
      requires: sycl
      args: -root_device_context_device_type sycl
      suffix: sycl

  testset:
    requires: !cxx
    output_file: ./output/ExitSuccess.out
    suffix: no_cxx

TEST*/
