static const char help[] = "Tests PetscDeviceContextSetDevice.\n\n";

#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx   = NULL;
  PetscDevice        device = NULL, other_device = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(AssertDeviceContextExists(dctx));

  PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT(), PETSC_DECIDE, &device));
  PetscCall(PetscDeviceConfigure(device));
  PetscCall(PetscDeviceView(device, NULL));

  PetscCall(PetscDeviceContextSetDevice(dctx, device));
  PetscCall(PetscDeviceContextGetDevice(dctx, &other_device));
  PetscCall(AssertPetscDevicesValidAndEqual(device, other_device, "PetscDevice after setdevice() does not match original PetscDevice"));
  // output here should be a duplicate of output above
  PetscCall(PetscDeviceView(other_device, NULL));

  // setup, test that this doesn't clobber the device
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscCall(PetscDeviceContextGetDevice(dctx, &other_device));
  PetscCall(AssertPetscDevicesValidAndEqual(device, other_device, "PetscDevice after setdevice() does not match original PetscDevice"));
  // once again output of this view should not change anything
  PetscCall(PetscDeviceView(other_device, NULL));

  PetscCall(PetscDeviceContextView(dctx, NULL));
  PetscCall(PetscDeviceContextDestroy(&dctx));

  // while we have destroyed the device context (which should decrement the PetscDevice's
  // refcount), we still hold a reference ourselves. Check that it remains valid
  PetscCall(PetscDeviceView(device, NULL));
  PetscCall(PetscDeviceContextCreate(&dctx));
  // PetscDeviceContext secretly keeps the device reference alive until the device context
  // itself is recycled. So create a new context here such that PetscDeviceDestroy() is called
  PetscCall(PetscDeviceView(device, NULL));

  // setup will attach the default device
  PetscCall(PetscDeviceContextSetUp(dctx));
  // check that it has, the attached device should not be equal to ours
  PetscCall(PetscDeviceContextGetDevice(dctx, &other_device));
  // None C++ builds have dummy devices (NULL)
  if (PetscDefined(HAVE_CXX)) PetscCheck(device != other_device, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext still has old PetscDevice attached after being recycled!");

  PetscCall(PetscDeviceContextDestroy(&dctx));
  PetscCall(PetscDeviceDestroy(&device));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
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

  testset:
    requires: !cxx
    output_file: ./output/ExitSuccess.out
    suffix: no_cxx

TEST*/
