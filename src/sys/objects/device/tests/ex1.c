static const char help[] = "Tests creation and destruction of PetscDevice.\n\n";

#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  const PetscInt n      = 10;
  PetscDevice    device = NULL;
  PetscDevice    devices[10];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // would have just done
  //
  // const PetscInt n = 10;
  // PetscDevice devices[n];
  //
  // but alas the reliably insane MSVC balks at this to the tune of
  // 'ex1.c(9): error C2057: expected constant expression'. So instead we have a runtime check
  PetscCheck(PETSC_STATIC_ARRAY_LENGTH(devices) == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to update n");

  /* normal create and destroy */
  PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT(), PETSC_DECIDE, &device));
  PetscCall(AssertDeviceExists(device));
  PetscCall(PetscDeviceDestroy(&device));
  PetscCall(AssertDeviceDoesNotExist(device));
  /* should not destroy twice */
  PetscCall(PetscDeviceDestroy(&device));
  PetscCall(AssertDeviceDoesNotExist(device));

  /* test reference counting */
  device = NULL;
  PetscCall(PetscArrayzero(devices, n));
  PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT(), PETSC_DECIDE, &device));
  PetscCall(AssertDeviceExists(device));
  for (int i = 0; i < n; ++i) {
    PetscCall(PetscDeviceReference_Internal(device));
    devices[i] = device;
  }
  PetscCall(AssertDeviceExists(device));
  for (int i = 0; i < n; ++i) {
    PetscCall(PetscDeviceDestroy(&devices[i]));
    PetscCall(AssertDeviceExists(device));
    PetscCall(AssertDeviceDoesNotExist(devices[i]));
  }
  PetscCall(PetscDeviceDestroy(&device));
  PetscCall(AssertDeviceDoesNotExist(device));

  /* test the default devices exist */
  device = NULL;
  PetscCall(PetscArrayzero(devices, n));
  {
    PetscDeviceContext dctx;
    /* global context will have the default device */
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextGetDevice(dctx, &device));
  }
  PetscCall(AssertDeviceExists(device));
  /* test reference counting for default device */
  for (int i = 0; i < n; ++i) {
    PetscCall(PetscDeviceReference_Internal(device));
    devices[i] = device;
  }
  PetscCall(AssertDeviceExists(device));
  for (int i = 0; i < n; ++i) {
    PetscCall(PetscDeviceDestroy(&devices[i]));
    PetscCall(AssertDeviceExists(device));
    PetscCall(AssertDeviceDoesNotExist(devices[i]));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
    output_file: ./output/ExitSuccess.out
    nsize: {{1 2 5}}
    args: -device_enable {{none lazy eager}}
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

  testset:
    requires: !cxx
    output_file: ./output/ExitSuccess.out
    suffix: no_cxx

TEST*/
