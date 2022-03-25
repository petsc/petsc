static const char help[] = "Tests creation and destruction of PetscDevice.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  const PetscInt n = 10;
  PetscDevice    device = NULL;
  PetscDevice    devices[n];

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  /* normal create and destroy */
  PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device));
  PetscCall(AssertDeviceExists(device));
  PetscCall(PetscDeviceDestroy(&device));
  PetscCall(AssertDeviceDoesNotExist(device));
  /* should not destroy twice */
  PetscCall(PetscDeviceDestroy(&device));
  PetscCall(AssertDeviceDoesNotExist(device));

  /* test reference counting */
  device = NULL;
  PetscCall(PetscArrayzero(devices,n));
  PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device));
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
  PetscCall(PetscArrayzero(devices,n));
  {
    PetscDeviceContext dctx;
    /* global context will have the default device */
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextGetDevice(dctx,&device));
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

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 testset:
   TODO: broken in ci
   requires: !device
   suffix: no_device
   filter: Error: grep -E -o -e ".*No support for this operation for this object type" -e ".*PETSc is not configured with device support.*" -e "^\[0\]PETSC ERROR:.*[0-9]{1} [A-z]+\(\)"
   test:
     requires: debug
     suffix:   debug
   test:
     requires: !debug
     suffix:   opt

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 2 5}}
   test:
     requires: cuda
     suffix: cuda
   test:
     requires: hip
     suffix: hip
   test:
     requires: sycl
     suffix: sycl

TEST*/
