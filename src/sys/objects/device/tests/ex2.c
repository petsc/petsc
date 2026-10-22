static const char help[] = "Tests creation and destruction of PetscDeviceContext.\n\n";

#include "petscdevicetestcommon.h"

static PetscDeviceContext rootContext = NULL, userContext = NULL;

static PetscErrorCode CheckRootContextOwnership(void)
{
  PetscDeviceContext extra = userContext;
  PetscInt           refs;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetReference((PetscObject)userContext, &refs));
  PetscCheck(refs == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Root cleanup changed the application context reference count to %" PetscInt_FMT, refs);
  PetscCall(PetscObjectGetReference((PetscObject)rootContext, &refs));
  PetscCheck(refs == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Root cleanup left %" PetscInt_FMT " references to the root context", refs);
  PetscCall(PetscDeviceContextDestroy(&extra));
  PetscCall(PetscDeviceContextDestroy(&userContext));
  PetscCall(PetscDeviceContextDestroy(&rootContext));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx = NULL, ddup = NULL;
  PetscBool          test_root_ownership = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_root_ownership", &test_root_ownership, NULL));
  // With lazy initialization, registering before root creation makes this run after root cleanup.
  if (test_root_ownership) PetscCall(PetscRegisterFinalize(CheckRootContextOwnership));

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

  if (test_root_ownership) {
    rootContext = dctx;
    PetscCall(PetscObjectReference((PetscObject)rootContext));
    PetscCall(PetscDeviceContextDuplicate(rootContext, &userContext));
    // Keep both contexts alive so an incorrect cleanup can be checked without accessing freed storage.
    PetscCall(PetscObjectReference((PetscObject)userContext));
    PetscCall(PetscDeviceContextSetCurrentContext(userContext));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: root_ownership
    requires: defined(PETSC_DEVICELANGUAGE_CXX)
    args: -test_root_ownership -device_enable lazy -default_device_type host
    output_file: output/ExitSuccess.out

  testset:
    requires: defined(PETSC_DEVICELANGUAGE_CXX)
    output_file: output/ExitSuccess.out
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
    requires: !defined(PETSC_DEVICELANGUAGE_CXX)
    output_file: output/ExitSuccess.out
    suffix: no_cxx

TEST*/
